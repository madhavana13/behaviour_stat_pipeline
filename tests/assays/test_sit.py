"""SIT ingestion: ID parsing, DI (target, TEST first 2 min vs empty cage, FAM, as fractions of
tracked time), target id, QC, strict aliases."""

from pathlib import Path

import pandas as pd
import pytest

from behaviour_stat_pipeline.assays.registry import RUNNER_REGISTRY
from behaviour_stat_pipeline.assays.sit import SITConfig, derive_sit_metrics
from behaviour_stat_pipeline.common.schema import load_schema_aliases
from behaviour_stat_pipeline.common.yaml_io import ConfigError
from behaviour_stat_pipeline.io import extract_mouse_id

REPO = Path(__file__).resolve().parents[2]
ALIASES = load_schema_aliases(REPO / "configs" / "schema_aliases.yaml")


def session(ids, **raw_cols):
    df = pd.DataFrame({"mouse_id": ids, "group": "cup", "sex": "male", "timepoint": "Single",
                       "raw_id": ids, "source_file": "f.csv", "social_animal": ["SM1", "SM1", "SM2"]})
    for col, values in raw_cols.items():
        df[col.replace("_", " ")] = values
    return df


def fam_test():
    ids = ["27A_ID1", "27A_ID2", "27A_ID3"]
    fam = session(ids, time_in_arena=[110, 109, 110], time_in_interaction_zone=[40, 0, 50],
                  time_exploring_cage=[20, 0, 25])
    test = session(ids, time_in_arena=[290, 288, 180], time_in_interaction_zone=[200, 0, 150],
                   time_exploring_cage=[100, 0, 90])
    test["time in interaction zone (first 2 min)"] = [60, 0, 50]
    test["time exploring cage (first 2 min)"] = [30, 0, 25]
    return fam, test


def test_sit_registered_and_hab_ids_parsed():
    assert "SIT" in RUNNER_REGISTRY
    assert extract_mouse_id("06_HE2_SIT_27A_HAB_ID4") == "27A_ID4"
    assert extract_mouse_id("06_HE2_SIT_27A_TEST_ID8") == "27A_ID8"


def di(a, b):
    return (a - b) / (a + b)


def test_di_compares_tracked_fractions_of_first_2_min_of_test_and_fam():
    fam, test = fam_test()
    cfg = SITConfig(target_col="social_animal")
    out = derive_sit_metrics(fam, test, schema_aliases=ALIASES, cfg=cfg).set_index("mouse_id")
    # FAM over its tracked time (110 s), TEST first 2 min over the 120-s window; not the 5-min 200 s
    assert out.loc["27A_ID1", "fam_time_in_interaction_zone_fraction"] == pytest.approx(40 / 110)
    assert out.loc["27A_ID1", "test_time_in_interaction_zone_first2min_fraction"] == pytest.approx(60 / 120)
    assert out.loc["27A_ID1", "di_iz"] == pytest.approx(di(60 / 120, 40 / 110))
    assert out.loc["27A_ID1", "di_explore"] == pytest.approx(di(30 / 120, 20 / 110))
    assert out.loc["27A_ID1", "pi_iz"] == pytest.approx((60 / 120) / (60 / 120 + 40 / 110))
    assert pd.isna(out.loc["27A_ID2", "di_iz"])          # zero time on both sides: undefined, not 0
    assert out["target_id"].tolist() == ["SM1", "SM1", "SM2"]
    assert out["iz_low_explore"].tolist() == [False, True, False]
    assert out["test_short_recording"].tolist() == [False, False, True]


def test_test_window_capped_by_tracked_time():
    fam, test = fam_test()
    test["time in arena"] = [290, 288, 100]          # stopped before the end of the 2-min window
    out = derive_sit_metrics(fam, test, schema_aliases=ALIASES, cfg=SITConfig()).set_index("mouse_id")
    assert out.loc["27A_ID3", "test_time_in_interaction_zone_first2min_fraction"] == pytest.approx(50 / 100)


def test_unknown_target_column_is_an_error():
    fam, test = fam_test()
    with pytest.raises(ValueError, match="target_col"):
        derive_sit_metrics(fam, test, schema_aliases=ALIASES, cfg=SITConfig(target_col="target"))


def test_missing_first_2_min_columns_is_an_error():
    fam, test = fam_test()
    test = test.drop(columns=["time in interaction zone (first 2 min)"])
    with pytest.raises(ValueError, match="time_in_interaction_zone_first2min"):
        derive_sit_metrics(fam, test, schema_aliases=ALIASES, cfg=SITConfig())


def test_duplicate_key_in_schema_aliases_is_an_error(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("sit:\n  raw: {a: b}\nsit:\n  raw: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate key 'sit'"):
        load_schema_aliases(path)
