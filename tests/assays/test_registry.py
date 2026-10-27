from pathlib import Path
import subprocess
import sys

import pytest

from behaviour_stat_pipeline.assays.registry import (
    VITRunner,
    RUNNER_REGISTRY,
)

REPO = Path(__file__).resolve().parents[2]


def test_registry_contains_expected_assays():
    assert {"VIT", "OFT", "EPM", "NOR", "LM", "SPL"}.issubset(set(RUNNER_REGISTRY))


def test_spl_uses_oft_aliases_and_names_itself_in_errors():
    import pandas as pd
    import yaml
    from behaviour_stat_pipeline.assays.oft import qc_oft

    aliases = yaml.safe_load(open(REPO / "configs" / "schema_aliases.yaml", encoding="utf-8"))
    assert aliases["spl"] == aliases["oft"]
    with pytest.raises(ValueError, match="^SPL QC failed"):
        qc_oft(pd.DataFrame({"mouse_id": ["m1"]}), label="SPL")


@pytest.mark.skipif(not (REPO / "data/experiments/EXP-2026-002/raw/oft1").exists(), reason="raw data not available")
def test_spl_ingestion_equals_oft_on_the_same_export(tmp_path):
    """SPL is ingested with the same parameters as OFT: same raw export -> same processed table."""
    import pandas as pd

    exp = "configs/experiments/EXP-2026-002.yaml"
    raw = "data/experiments/EXP-2026-002/raw/oft1"
    out = {}
    for assay in ["oft", "spl"]:
        out[assay] = tmp_path / f"{assay}.csv"
        subprocess.run(
            [sys.executable, "scripts/run_assay.py", "--assay", assay, "--root", raw,
             "--pattern", "*_output_summary.csv", "--config", "configs/stats.yaml",
             "--experiment-config", exp, "--out", str(out[assay])],
            cwd=REPO, check=True, capture_output=True, text=True,
        )
    a, b = pd.read_csv(out["oft"]), pd.read_csv(out["spl"])
    assert set(b["assay"]) == {"SPL"}
    # Columns that only carry the assay name (assay, session, classifier__assay, ...) differ by design
    label_cols = [c for c in a.columns if set(a[c].astype(str)) == {"OFT"} and set(b[c].astype(str)) == {"SPL"}]
    assert "assay" in label_cols
    pd.testing.assert_frame_equal(a.drop(columns=label_cols), b.drop(columns=label_cols))
    assert b.filter(like="classifier__").shape[1] > 1  # classifier report merged as for OFT


def test_vit_filename_inference_first_min_hand():
    runner = VITRunner()
    tp, ph = runner.infer_vit_timepoint_phase_from_filename(
        Path("test_vit_first_min_hand_results_output_summary.csv")
    )
    assert tp == "Before"
    assert ph == "hand"


def test_vit_filename_inference_second_min_standing():
    runner = VITRunner()
    tp, ph = runner.infer_vit_timepoint_phase_from_filename(
        Path("test_vit_second_min_standing_results_output_summary.csv")
    )
    assert tp == "After"
    assert ph == "standing"


def test_vit_filename_inference_invalid():
    runner = VITRunner()
    with pytest.raises(ValueError):
        runner.infer_vit_timepoint_phase_from_filename(
            Path("weird_file_name.csv")
        )