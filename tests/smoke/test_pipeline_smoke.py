"""
Smoke tests: the real EXP-2026-002 configuration runs end to end (stats, then figures).

Outputs go to a temporary directory: tests never write into outputs/, which holds the
results you report.
"""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "stats.yaml"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiments" / "EXP-2026-002.yaml"

ASSAYS = ["OFT1", "OFT2", "OFT3", "OFT4", "NOR", "VIT", "LM", "MISC"]


def run(cmd):
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, (
        "Command failed:\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def experiment_config(tmp_path_factory) -> Path:
    """The real experiment config with paths.out_dir pointed at a temporary directory."""
    tmp = tmp_path_factory.mktemp("smoke")
    cfg = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["out_dir"] = str(tmp / "outputs")
    path = tmp / "experiment.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_stats_smoke(experiment_config):
    for assay in ASSAYS:
        run([sys.executable, "scripts/run_stats.py", "--config", str(CONFIG),
             "--experiment-config", str(experiment_config), "--assays", assay])


def test_sensitivity_analysis_uses_its_own_exclusions(tmp_path):
    """A mouse excluded from the primary analysis only is back in the _sensitivity tables."""
    import pandas as pd

    cfg = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    ex = pd.read_csv(REPO_ROOT / cfg["paths"]["exclusions"])
    ex = pd.concat([ex, pd.DataFrame([{"mouse_id": "4F_ID2", "assay": "NOR", "endpoint": "*",
                                       "exclude_from": "primary", "reason": "test"}])], ignore_index=True)
    ex.to_csv(tmp_path / "exclusions.csv", index=False)
    cfg["paths"].update(exclusions=str(tmp_path / "exclusions.csv"), out_dir=str(tmp_path / "out"))
    exp = tmp_path / "experiment.yaml"
    exp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run([sys.executable, "scripts/run_stats.py", "--config", str(CONFIG), "--experiment-config", str(exp),
         "--assays", "NOR"])

    tables = tmp_path / "out" / "NOR" / "tables"
    primary = pd.read_csv(tables / "NOR_secondary_two_way_anova.csv")
    sensitivity = pd.read_csv(tables / "NOR_secondary_two_way_anova_sensitivity.csv")
    assert "4F_ID2" in primary["excluded_mouse_ids"].iloc[0]
    assert "4F_ID2" not in sensitivity["excluded_mouse_ids"].iloc[0]
    assert sensitivity["n"].max() == primary["n"].max() + 1
    assert set(sensitivity["analysis_scope"]) == {"sensitivity"}


def test_figures_smoke(experiment_config):
    for assay in ASSAYS:
        run([sys.executable, "scripts/run_figures.py", "--config", str(CONFIG),
             "--experiment-config", str(experiment_config), "--assays", assay])
