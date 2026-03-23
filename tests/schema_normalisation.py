import pandas as pd

from behaviour_stat_pipeline.common.schema import load_schema_aliases, normalize_headers


def test_nor_schema_aliases():
    df = pd.DataFrame({
        "Unnamed: 0": ["27A_FAM_ID1"],
        "time exploring flask_0": [1.2],
        "time in interaction zone of flask_0": [2.3],
        "average speed": [4.5],
    })

    schema = load_schema_aliases("configs/schema_aliases.yaml")
    out, rename_log, suspicious = normalize_headers(df, assay="NOR", schema=schema)

    assert "time_exploring_flask_0" in out.columns
    assert "time_in_interaction_zone_of_flask_0" in out.columns
    assert "mean_speed" in out.columns
    assert any(old == "time exploring flask_0" for old, _ in rename_log)