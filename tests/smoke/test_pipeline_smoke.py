from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "stats.yaml"

ASSAYS = ["OFT", "NOR", "EPM", "VIT", "LM"]

def run(cmd):
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, f"Command failed:\n{result.stderr}"

def test_stats_smoke():
    for assay in ASSAYS:
        run(
            [
                sys.executable,
                "scripts/run_stats.py",
                "--config",
                str(CONFIG),
                "--assays",
                assay,
            ]
        )

def test_figures_smoke():
    for assay in ASSAYS:
        run(
            [
                sys.executable,
                "scripts/run_figures.py",
                "--config",
                str(CONFIG),
                "--assays",
                assay,
            ]
        )
