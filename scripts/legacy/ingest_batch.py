import argparse
from pathlib import Path

import pandas as pd

from behaviour_stat_pipeline.io import load_subjects, load_tracking_csv, merge_subject_metadata


def infer_vit_timepoint_phase_from_filename(p: Path) -> tuple[str, str]:
    """
    Parse VIT filenames like:
      *_first_min_hand_output_summary.csv
      *_first_min_standing_output_summary.csv
      *_second_min_hand_output_summary.csv
      *_second_min_standing_output_summary.csv

    Returns:
      timepoint: "Before" or "After"
      phase: "hand" or "standing"
    """
    name = p.name.lower()

    # timepoint
    if "first_min" in name:
        timepoint = "Before"
    elif "second_min" in name:
        timepoint = "After"
    else:
        raise ValueError(
            f"Cannot infer VIT timepoint from filename (expected 'first_min' or 'second_min'): {p.name}"
        )

    # phase
    # Be strict: require explicit token
    if "hand" in name:
        phase = "hand"
    elif "standing" in name:
        phase = "standing"
    else:
        raise ValueError(
            f"Cannot infer VIT phase from filename (expected 'hand' or 'standing'): {p.name}"
        )

    return timepoint, phase


def main():
    """Batch-ingest VIT tracking CSVs into one long-format dataset with
    explicit mouse_id, timepoint (Before/After), and phase (hand/standing)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder to search recursively for CSVs")
    ap.add_argument(
        "--pattern",
        default="*_output_summary.csv",
        help="Glob pattern for files (recursive). Example: '*_output_summary.csv'",
    )
    ap.add_argument("--subjects", default="data/metadata/subjects.csv", help="subjects.csv path")
    ap.add_argument("--raw_id_col", default="Unnamed: 0", help="Raw ID column in tracking CSV")
    ap.add_argument(
        "--assay",
        default="VIT",
        help="Assay label to attach to rows (e.g., VIT, OFT, EPM)",
    )
    ap.add_argument(
        "--infer_vit",
        action="store_true",
        help="Infer VIT timepoint+phase from filename tokens (first_min/second_min + hand/standing).",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output CSV path. If omitted, defaults to data/interim/{assay}_long.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching {args.pattern}")

    subjects = load_subjects(args.subjects)

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = load_tracking_csv(f, raw_id_col=args.raw_id_col)
        df = merge_subject_metadata(df, subjects)

        df["assay"] = args.assay
        df["source_file"] = str(f)

        if args.infer_vit:
            timepoint, phase = infer_vit_timepoint_phase_from_filename(f)
            df["timepoint"] = timepoint
            df["phase"] = phase
        else:
            # Generic mode: only timepoint inference exists in this script version.
            # If you need generic non-VIT timepoints later, we can add that back cleanly.
            raise ValueError(
                "This script is currently configured for VIT batching with --infer_vit. "
                "Run with --infer_vit for VIT outputs."
            )

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # QC: enforce uniqueness for VIT long table
    key_cols = ["mouse_id", "timepoint", "phase"]
    dup = out.duplicated(subset=key_cols)
    if dup.any():
        examples = out.loc[dup, key_cols + ["source_file"]].head(15)
        raise ValueError(
            "Duplicate rows detected for key mouse_id × timepoint × phase.\n"
            "This usually indicates duplicate files being ingested or incorrect filename parsing.\n\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # Default output path if not provided
    out_path = Path(args.out) if args.out.strip() else Path(f"data/interim/{args.assay}_long.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nSaved combined dataset: {out_path}")
    print(f"Total rows: {len(out)} | Unique mice: {out['mouse_id'].nunique()} | Files: {len(files)}")
    print("\nCounts by timepoint × phase × group × sex:")
    print(out[["timepoint", "phase", "group", "sex"]].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()