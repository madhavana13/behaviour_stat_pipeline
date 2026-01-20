from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import pandas as pd

from behaviour_stat_pipeline.stats.common import (
    load_stats_config,
    read_processed,
    read_subjects,
    merge_metadata,
    require_columns,
    coerce_categoricals,
    drop_single_level_factors,
    cell_counts,
)


def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/stats.yaml")
    ap.add_argument("--assays", nargs="*", default=None, help="Optional subset, e.g. nor vit")
    args = ap.parse_args()

    cfg = load_stats_config(args.config)

    assays = cfg.assays if args.assays is None else args.assays

    # Create output dirs (no manual steps)
    cfg.out_tables.mkdir(parents=True, exist_ok=True)
    cfg.out_figures.mkdir(parents=True, exist_ok=True)
    cfg.out_logs.mkdir(parents=True, exist_ok=True)

    subjects = read_subjects(cfg.subjects_path)

    for assay in assays:
        df = read_processed(cfg.processed_dir, assay)

        # Join metadata (the key to being robust to level changes)
        merged = merge_metadata(df, subjects, mouse_id_col=cfg.mouse_id_col)

        # Determine factor roles by name
        within = cfg.within_factors.get(assay, [])
        between = cfg.between_factors
        blocking = cfg.blocking_factors

        # Validate columns exist (after join)
        require_columns(merged, [cfg.mouse_id_col], f"{assay}: merged")
        require_columns(merged, between, f"{assay}: between_factors")
        require_columns(merged, within, f"{assay}: within_factors")
        require_columns(merged, blocking, f"{assay}: blocking_factors") if blocking else None

        # Coerce to categoricals (important for formulas later)
        merged = coerce_categoricals(merged, between + within + blocking)

        # Drop single-level factors (subset robustness)
        kept_between, dropped_between = drop_single_level_factors(merged, between)
        kept_within, dropped_within = drop_single_level_factors(merged, within)
        kept_blocking, dropped_blocking = drop_single_level_factors(merged, blocking)

        # Cell counts for between-subject design
        counts = cell_counts(merged, kept_between)
        counts_path = cfg.out_tables / f"{assay.upper()}_cell_counts.csv"
        counts.to_csv(counts_path, index=False)

        # Basic summary log
        n_rows = len(merged)
        n_mice = merged[cfg.mouse_id_col].nunique(dropna=True)

        def fmt_drop(dropped):
            if not dropped:
                return "None"
            return "; ".join([f"{c} (levels observed: {lvls})" for c, lvls in dropped])

        summary = textwrap.dedent(
            f"""
            ASSAY: {assay}
            processed_file_rows: {n_rows}
            unique_mice: {n_mice}

            factor_roles (from config):
              between_factors: {between}
              within_factors:  {within}
              blocking_factors:{blocking}

            factors_kept (>=2 observed levels):
              between: {kept_between}
              within:  {kept_within}
              blocking:{kept_blocking}

            factors_dropped (single-level in this subset):
              between: {fmt_drop(dropped_between)}
              within:  {fmt_drop(dropped_within)}
              blocking:{fmt_drop(dropped_blocking)}

            wrote:
              {counts_path.as_posix()}
            """.strip()
            + "\n"
        )

        log_path = cfg.out_logs / f"{assay.upper()}_stats_summary.txt"
        write_text(log_path, summary)

        print(summary)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
