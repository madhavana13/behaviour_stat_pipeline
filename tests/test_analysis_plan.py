"""Per-experiment analysis plan: config merge, primary/secondary endpoints, frozen-plan check."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from behaviour_stat_pipeline.stats.common import (
    ConfigError,
    check_groups_in_data,
    endpoint_role,
    load_stats_config,
    ordered_instances,
    read_groups,
    resolve_config,
    validate_stats_config,
)
from behaviour_stat_pipeline.stats.endpoints.linear_models import assemble_model_data, validate_model_spec
from behaviour_stat_pipeline.stats.provenance import frozen_plan_status

REPO = Path(__file__).resolve().parents[1]
STATS = REPO / "configs" / "stats.yaml"


GROUPS = {"order": ["control", "ifs"]}


def exp_cfg(**analysis) -> dict:
    return {"assay_instances": {"OFT2": {"assay_type": "OFT"}}, "groups": GROUPS, "analysis": analysis}


def test_merge_replaces_lists_and_merges_mappings():
    base = {"secondary_two_way_anova": {"type": 2, "endpoints": {"nor": ["a", "b"]}}}
    merged = resolve_config(base, {"analysis": {"secondary_two_way_anova": {"endpoints": {"nor": ["c"]}}}})
    assert merged["secondary_two_way_anova"] == {"type": 2, "endpoints": {"nor": ["c"]}}


def test_unknown_analysis_key_is_an_error():
    with pytest.raises(ConfigError, match="unknown keys \\['endpoint'\\]"):
        resolve_config({}, {"analysis": {"endpoint": {}}})


def test_primary_endpoint_typo_is_an_error():
    e = exp_cfg(secondary_two_way_anova={"endpoints": {"nor": ["di_explore"]}},
                primary_endpoints={"nor": ["di_explor"]})
    with pytest.raises(ConfigError, match="di_explor"):
        load_stats_config(STATS, e)


def test_endpoint_role_by_instance_type_and_model():
    e = exp_cfg(
        secondary_two_way_anova={"endpoints": {"oft": ["total_distance_moved", "time_in_center"]}},
        linear_models={"models": [{"name": "sessions", "assays": ["oft2"], "endpoints": ["time_in_center"],
                                   "fixed": "group"}]},
        primary_endpoints={"oft": ["total_distance_moved"], "sessions": ["time_in_center"]},
    )
    cfg = load_stats_config(STATS, e)
    assert endpoint_role(cfg, "oft2", "total_distance_moved") == "primary"  # declared for the assay type
    assert endpoint_role(cfg, "oft2", "time_in_center") == "exploratory"
    assert endpoint_role(cfg, "sessions", "time_in_center") == "primary"


def test_secondary_role_and_overlap_with_primary():
    e = exp_cfg(
        secondary_two_way_anova={"endpoints": {"oft": ["total_distance_moved", "time_in_center", "rears"]}},
        primary_endpoints={"oft": ["time_in_center"]},
        secondary_endpoints={"oft2": ["total_distance_moved"]},
    )
    cfg = load_stats_config(STATS, e)
    assert endpoint_role(cfg, "oft2", "time_in_center") == "primary"
    assert endpoint_role(cfg, "oft2", "total_distance_moved") == "secondary"
    assert endpoint_role(cfg, "oft2", "rears") == "exploratory"

    e["analysis"]["secondary_endpoints"] = {"oft2": ["time_in_center"]}  # also primary via the assay type
    with pytest.raises(ConfigError, match="both primary_endpoints and secondary_endpoints"):
        load_stats_config(STATS, e)


def test_secondary_endpoint_typo_is_an_error():
    e = exp_cfg(secondary_two_way_anova={"endpoints": {"oft": ["rears"]}}, secondary_endpoints={"oft": ["rear"]})
    with pytest.raises(ConfigError, match="secondary_endpoints.oft"):
        load_stats_config(STATS, e)


def test_correction_per_role():
    cfg = load_stats_config(STATS, exp_cfg())
    assert cfg.secondary_two_way_anova_correction == {"primary": "holm", "secondary": "fdr_bh", "exploratory": "fdr_bh"}
    # one method still applies to every role
    cfg = load_stats_config(STATS, exp_cfg(time_to_event={"correction_across_endpoints": "bonferroni"}))
    assert set(cfg.time_to_event_correction.values()) == {"bonferroni"}
    # an experiment may override one role; the others keep the stats.yaml defaults
    cfg = load_stats_config(STATS, exp_cfg(secondary_two_way_anova={"correction_across_endpoints": {"secondary": "holm"}}))
    assert cfg.secondary_two_way_anova_correction == {"primary": "holm", "secondary": "holm", "exploratory": "fdr_bh"}
    # after merging, a mapping must name every role
    with pytest.raises(ConfigError, match="must name exactly"):
        validate_stats_config({"secondary_two_way_anova": {"correction_across_endpoints": {"primary": "holm"}}})


def timeline_cfg(**instances):
    return load_stats_config(STATS, {"assay_instances": instances, "groups": GROUPS})


def test_sessions_follow_declared_order_not_names():
    cfg = timeline_cfg(OFT1={"order": 1}, OFT2={"order": 4}, OFT3={"order": 2}, OFT4={"order": 3})
    assert ordered_instances(cfg, ["oft2", "oft3", "oft4"], baseline="oft1", what="m") == ["oft3", "oft4", "oft2"]


def test_missing_order_names_the_fields_to_fill():
    cfg = timeline_cfg(OFT1={"order": 1}, OFT2={"order": None}, OFT3={})
    with pytest.raises(ConfigError, match=r"\['OFT2', 'OFT3'\]"):
        ordered_instances(cfg, ["oft2", "oft3"], baseline="oft1", what="m")


def test_baseline_must_come_before_sessions():
    cfg = timeline_cfg(OFT1={"order": 3}, OFT2={"order": 1}, OFT3={"order": 2})
    with pytest.raises(ConfigError, match="does not come before"):
        ordered_instances(cfg, ["oft2", "oft3"], baseline="oft1", what="m")


def test_invalid_or_duplicate_order_is_an_error():
    with pytest.raises(ConfigError, match="same order 2"):
        timeline_cfg(OFT1={"order": 2}, OFT2={"order": 2})
    with pytest.raises(ConfigError, match="must be an integer"):
        timeline_cfg(OFT1={"order": "first"})


def test_day_column_added_and_required_when_used():
    frames = {a: pd.DataFrame({"mouse_id": ["m1", "m2"], "group": ["c", "i"], "y": [1.0, 2.0]})
              for a in ["oft2", "oft3"]}
    spec = {"name": "m", "assays": ["oft2", "oft3"], "stack_as": "session", "endpoints": ["y"],
            "fixed": "group * day", "random": "~ 1 | mouse_id"}
    long = assemble_model_data(frames, spec, id_col="mouse_id", factors=["group"],
                               instance_days={"oft2": 1.0, "oft3": 8.0})
    assert long.groupby("session")["day"].first().to_dict() == {"OFT2": 1.0, "OFT3": 8.0}
    with pytest.raises(ValueError, match=r"day is not set .*\['OFT3'\]"):
        assemble_model_data(frames, spec, id_col="mouse_id", factors=["group"], instance_days={"oft2": 1.0})
    # day is only available for stacked sessions
    errs = validate_model_spec({"name": "m", "assays": ["oft2"], "endpoints": ["y"], "fixed": "group * day"},
                               "m", ["group"])
    assert any("fixed uses ['day']" in e for e in errs)


def test_groups_block_fills_labels_and_distinct_colours():
    order, labels, colors, errors = read_groups(
        {"groups": {"order": ["control", "ifs", "stressed"], "labels": {"ifs": "IFS"}, "colors": {"control": "#E69F00"}}}
    )
    assert not errors
    assert labels == {"control": "Control", "ifs": "IFS", "stressed": "Stressed"}
    assert colors["control"] == "#E69F00"
    assert len(set(colors.values())) == 3  # palette skips the colour already used


def test_groups_block_errors():
    _, _, _, errors = read_groups({"groups": {"order": ["a", "b"], "colors": {"a": "red", "b": "red"}}})
    assert any("share a colour" in e for e in errors)
    _, _, _, errors = read_groups({"group_order": ["a", "b"]})
    assert any("group_order has moved" in e for e in errors) and any("groups.order is not set" in e for e in errors)
    _, _, _, errors = read_groups({"groups": {"order": ["a"], "labels": {"x": "X"}}})
    assert any("not in groups.order" in e for e in errors)
    with pytest.raises(ConfigError, match=r"\['stressed'\] are in the data"):
        check_groups_in_data(pd.Series(["a", "stressed"]), ["a", "b"], where="T")


def test_group_order_only_in_experiment_config():
    e = {"groups": {"order": ["ifs", "control"]},
         "analysis": {"secondary_two_way_anova": {"factor_orders": {"group": ["control", "ifs"]}}}}
    with pytest.raises(ConfigError, match="belongs in the experiment config"):
        load_stats_config(STATS, e)
    cfg = load_stats_config(STATS, {"groups": {"order": ["ifs", "control"]}})
    assert cfg.secondary_two_way_anova_factor_orders["group"] == ["ifs", "control"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_frozen_plan_detects_changes(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({"analysis": {"primary_endpoints": {"nor": ["di_explore"]}}}), encoding="utf-8")
    git("add", "exp.yaml")
    git("commit", "-q", "-m", "plan")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()

    assert frozen_plan_status(tmp_path, path, sha)["unchanged"] is True
    # Adding the freeze marker outside `analysis` does not count as a change
    path.write_text(path.read_text(encoding="utf-8") + f"analysis_frozen_at: {sha}\n", encoding="utf-8")
    assert frozen_plan_status(tmp_path, path, sha)["unchanged"] is True

    path.write_text(yaml.safe_dump({"analysis": {"primary_endpoints": {"nor": ["di_iz"]}}}), encoding="utf-8")
    status = frozen_plan_status(tmp_path, path, sha)
    assert status["unchanged"] is False and "primary_endpoints" in status["detail"]
    assert frozen_plan_status(tmp_path, path, None)["unchanged"] is None
