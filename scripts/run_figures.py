from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from behaviour_stat_pipeline.figures.runner import run_figures_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (stats.yaml)")
    ap.add_argument("--assays", nargs="+", required=True, help="Assays to run figures for (e.g. nor oft)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    out_figures = Path(paths.get("out_figures", "outputs/figures"))

    id_cols = cfg.get("id_cols", {})
    mouse_id_col = id_cols.get("mouse", "mouse_id")

    # factor column names
    factors_cfg = cfg.get("factors", {})
    group_col = factors_cfg.get("group", {}).get("column", "group")
    sex_col = factors_cfg.get("sex", {}).get("column", "sex")

    written_all = []
    for assay in args.assays:
        written = run_figures_from_config(
            config=cfg,
            processed_dir=processed_dir,
            out_figures=out_figures,
            assay=assay,
            mouse_id_col=mouse_id_col,
            group_col=group_col,
            sex_col=sex_col,
        )
        written_all.extend(written)

    if written_all:
        print("Wrote figures:")
        for p in written_all:
            print(f" - {p}")
    else:
        print("No figures written (figures disabled or no matching specs).")


if __name__ == "__main__":
    main()
