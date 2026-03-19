from pathlib import Path
import pytest

from behaviour_stat_pipeline.assays.registry import (
    VITRunner,
    RUNNER_REGISTRY,
)


def test_registry_contains_expected_assays():
    assert {"VIT", "OFT", "EPM", "NOR", "LM"}.issubset(set(RUNNER_REGISTRY))


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