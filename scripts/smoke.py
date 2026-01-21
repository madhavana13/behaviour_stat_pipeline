from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml


def _load_raw_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Config YAML must parse to dict, got: {type(obj)}")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for behaviour_stat_pipeline.")
    parser.add_argument("--config", required=True, type=str, help="Path to stats.yaml (used by stats + figures).")
    parser.add_argument("--assay", default="nor", type=str, help="Assay to smoke test (default: nor).")
    parser.add_argument(
        "--expect-corr",
        action="store_true",
        help="If set, assert that correlations output CSV is created.",
    )
    parser.add_argument(
        "--expect-scatter",
        action="store_true",
        help="If set, assert that at least one scatter figure is created.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    # --- import here so smoke fails fast if packaging/imports are broken ---
    from behaviour_stat_pipeline.stats.common import load_stats_config, read_processed
    from behaviour_stat_pipeline.stats.endpoints.correlations import run_correlations
    from behaviour_stat_pipeline.figures.runner import run_figures_from_config

    # Load both typed config + raw YAML dict (raw is still consumed by correlations/figures expansion logic)
    cfg = load_stats_config(cfg_path)
    cfg_raw = _load_raw_yaml(cfg_path)

    assay = args.assay.lower()

    # 1) Can we read processed data?
    df = read_processed(cfg.processed_dir, assay)

    # 2) Run correlations if enabled (and optionally assert output)
    corr_enabled_raw = bool(cfg_raw.get("correlations", {}).get("enabled", False))
    corr_out = cfg.out_tables / f"{assay.upper()}_correlations.csv"

    if corr_enabled_raw:
        run_correlations(
            assay=assay,
            df=df,
            cfg=cfg_raw,
            out_csv_path=str(corr_out),
        )
        print(f"[smoke] correlations wrote: {corr_out}")
    else:
        print("[smoke] correlations disabled in YAML; skipping")

    if args.expect_corr and not corr_out.exists():
        raise AssertionError(f"Expected correlations CSV not found: {corr_out}")

    # 3) Run figures (this should include your manual boxpoints + auto-from-correlations scatter if enabled)
    written = run_figures_from_config(
        config=cfg_raw,
        processed_dir=cfg.processed_dir,
        out_figures=cfg.out_figures,
        assay=assay,
    )
    print(f"[smoke] figures wrote {len(written)} file(s) into: {cfg.out_figures}")

    # Always assert at least one figure exists (basic pipeline integrity)
    if len(written) == 0:
        raise AssertionError("Expected at least one figure to be written, but got 0.")

    # Optional: assert at least one scatter exists (useful when you rely on auto_from_correlations)
    if args.expect_scatter:
        scatter_hits = [p for p in written if "scatter" in Path(p).name.lower()]
        if len(scatter_hits) == 0:
            raise AssertionError("Expected at least one scatter figure, but none were written.")
        print(f"[smoke] scatter figures: {len(scatter_hits)}")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[smoke] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
