"""Ingestion settings that used to be hardcoded: NOR objects/patterns, VIT filename tokens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from behaviour_stat_pipeline.assays.nor import NORConfig, derive_nor_metrics
from behaviour_stat_pipeline.assays.registry import NORRunner, RunContext, VITRunner

REPO = Path(__file__).resolve().parents[2]
ALIASES = yaml.safe_load(open(REPO / "configs" / "schema_aliases.yaml", encoding="utf-8"))


def nor_sessions(a: str, b: str):
    """Two mice: m1 familiar=a, m2 familiar=b. TEST has an empty second copy of each object."""
    ids = {"mouse_id": ["m1", "m2"], "group": ["c", "i"], "sex": ["female", "male"], "timepoint": "Single",
           "raw_id": ["r1", "r2"], "source_file": "f.csv"}
    fam = pd.DataFrame({**ids, "session": "FAM"})
    for obj, vals in [(a, [10.0, 0.0]), (b, [0.0, 12.0])]:
        for copy in (0, 1):
            fam[f"time_exploring_{obj}_{copy}"] = vals
            fam[f"time_in_interaction_zone_of_{obj}_{copy}"] = [v * 2 for v in vals]
    test = pd.DataFrame({**ids, "session": "TEST"})
    for obj, vals in [(a, [4.0, 9.0]), (b, [6.0, 3.0])]:
        test[f"time_exploring_{obj}_0"] = vals
        test[f"time_in_interaction_zone_of_{obj}_0"] = [v * 2 for v in vals]
        test[f"time_exploring_{obj}_1"] = np.nan
        test[f"time_in_interaction_zone_of_{obj}_1"] = np.nan
    return fam, test


def test_nor_results_do_not_depend_on_object_names():
    a = derive_nor_metrics(*nor_sessions("flask", "lego_tower"), schema_aliases=ALIASES,
                           cfg=NORConfig(objects=("flask", "lego_tower")))
    b = derive_nor_metrics(*nor_sessions("cube", "ball"), schema_aliases=ALIASES,
                           cfg=NORConfig(objects=("cube", "ball")))
    cols = ["fam_familiar_explore_time", "test_familiar_explore_time", "test_novel_explore_time",
            "di_explore", "di_iz", "pi_explore"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert b["familiar_type"].tolist() == ["cube", "ball"]
    # m1: familiar cube 4s, novel ball 6s -> DI = (6-4)/10
    assert b.loc[0, "di_explore"] == pytest.approx(0.2)
    assert b.loc[0, "fam_familiar_explore_time"] == 20.0  # both FAM copies summed


def test_nor_wrong_objects_name_the_problem():
    with pytest.raises(ValueError, match=r"no columns for .*time_exploring_cup_<n>"):
        derive_nor_metrics(*nor_sessions("flask", "lego_tower"), schema_aliases=ALIASES,
                           cfg=NORConfig(objects=("flask", "cup")))
    with pytest.raises(ValueError, match="exactly two different object types"):
        NORConfig(objects=("flask", "flask"))


def test_nor_test_data_in_two_copies_is_an_error():
    fam, test = nor_sessions("flask", "lego_tower")
    test["time_exploring_flask_1"] = 1.0
    with pytest.raises(ValueError, match="several copies of the same object"):
        derive_nor_metrics(fam, test, schema_aliases=ALIASES, cfg=NORConfig(objects=("flask", "lego_tower")))


def test_nor_objects_must_be_configured():
    ctx = RunContext(root=Path("."), pattern="", subjects_path="", raw_id_col="", out="", cfg={},
                     schema_aliases={}, instance_cfg={})
    with pytest.raises(ValueError, match="assay_instances.NOR.objects"):
        NORRunner().nor_config(ctx)


def test_vit_custom_filename_tokens():
    tokens = {"timepoint": {"pre": "Before", "post": "After"}, "phase": {"hnd": "hand", "std": "standing"}}
    assert VITRunner.infer_vit_timepoint_phase_from_filename(Path("x_post_std_output_summary.csv"), tokens) == (
        "After", "standing")


def test_vit_ambiguous_filename_is_an_error():
    tokens = {"timepoint": {"pre": "Before", "post": "After"}, "phase": {"hand": "hand", "standing": "standing"}}
    with pytest.raises(ValueError, match="several of the tokens"):
        VITRunner.infer_vit_timepoint_phase_from_filename(Path("pre_post_hand.csv"), tokens)
