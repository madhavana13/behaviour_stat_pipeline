import argparse
from pathlib import Path

import pandas as pd

# Assay modules
from behaviour_stat_pipeline.assays.vit import qc_vit, derive_vit_metrics, VITConfig

# Reuse ingestion logic from your ingest_batch script
from behaviour_stat_pipeline.io import load_subjects, load_tracking_csv, merge_subject_metadata


def infer_vit_timepoint_phase_from_filename(p: Path) -> tuple[str, str]:
    name = p.name.lower()
    if "first_min" in name:
        timepoint = "Before"
    elif "second_min" in name:
        timepoint = "After"
    else:
        raise ValueError(f"Cannot infer VIT timepoint from filename: {p.name}")

    if "hand" in name:
        phase = "hand"
    elif "standing" in name:
        phase = "standing"
    else:
        raise ValueError(f"Cannot infer VIT phase from filename: {p.name}")

    return timepoint, phase


def run_vit(args: argparse.Namespace) -> None:
    root = Path(args.root)
    files = sorted(root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching {args.pattern}")

    subjects = load_subjects(args.subjects)

    dfs = []
    for f in files:
        df = load_tracking_csv(f, raw_id_col=args.raw_id_col)
        df = merge_subject_metadata(df, subjects)

        df["assay"] = "VIT"
        df["source_file"] = str(f)

        tp, ph = infer_vit_timepoint_phase_from_filename(f)
        df["timepoint"] = tp
        df["phase"] = ph

        dfs.append(df)

    vit_long = pd.concat(dfs, ignore_index=True)

    # Strict uniqueness for VIT
    key = ["mouse_id", "timepoint", "phase"]
    if vit_long.duplicated(subset=key).any():
        examples = vit_long.loc[vit_long.duplicated(subset=key, keep=False), key + ["source_file"]].head(20)
        raise ValueError(
            "Duplicate rows detected for mouse_id × timepoint × phase.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # QC + derive
    qc_vit(vit_long)
    vit_processed = derive_vit_metrics(vit_long, cfg=VITConfig())

    out_path = Path(args.out) if args.out else Path("data/processed/VIT_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vit_processed.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(vit_processed)} | Unique mice: {vit_processed['mouse_id'].nunique()} | Files: {len(files)}")
    print(vit_processed[["timepoint", "phase", "group", "sex"]].value_counts().sort_index().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True, choices=["VIT"], help="Which assay pipeline to run")
    ap.add_argument("--root", required=True, help="Root folder containing raw CSV outputs")
    ap.add_argument("--pattern", default="*_output_summary.csv", help="Glob pattern for raw CSVs")
    ap.add_argument("--subjects", default="data/metadata/subjects.csv", help="Path to subjects.csv")
    ap.add_argument("--raw_id_col", default="Unnamed: 0", help="Raw ID column in tracking CSV")
    ap.add_argument("--out", default="", help="Output processed CSV path")
    args = ap.parse_args()

    if args.assay == "VIT":
        run_vit(args)
    else:
        raise ValueError(f"Unsupported assay: {args.assay}")


if __name__ == "__main__":
    main()