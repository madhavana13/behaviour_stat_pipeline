"""Per-experiment analysis plan: config merge, primary endpoints, frozen-plan check."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from behaviour_stat_pipeline.stats.common import (
    ConfigError,
    endpoint_role,
    load_stats_config,
    resolve_config,
)
from behaviour_stat_pipeline.stats.provenance import frozen_plan_status

REPO = Path(__file__).resolve().parents[1]
STATS = REPO / "configs" / "stats.yaml"


def exp_cfg(**analysis) -> dict:
    return {"assay_instances": {"OFT2": {"assay_type": "OFT"}}, "analysis": analysis}


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
