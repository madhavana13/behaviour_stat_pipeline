# src/behaviour_stat_pipeline/run_assay.py

import argparse
from pathlib import Path

import pandas as pd

# Assay modules
from behaviour_stat_pipeline.assays.vit import qc_vit, derive_vit_metrics, VITConfig
from behaviour_stat_pipeline.assays.oft import qc_oft, derive_oft_metrics, OFTConfig


# Reuse ingestion logic
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

    dfs: list[pd.DataFrame] = []
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

    # QC + derive
    qc_vit(vit_long)
    vit_processed = derive_vit_metrics(vit_long, cfg=VITConfig())

    out_path = Path(args.out) if args.out else Path("data/processed/VIT_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vit_processed.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(
        f"Rows: {len(vit_processed)} | Unique mice: {vit_processed['mouse_id'].nunique()} | Files: {len(files)}"
    )
    print(vit_processed[["timepoint", "phase", "group", "sex"]].value_counts().sort_index().to_string())

def run_oft(args: argparse.Namespace) -> None:
    root = Path(args.root)
    files = sorted(root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching {args.pattern}")

    subjects = load_subjects(args.subjects)

    dfs = []
    for f in files:
        df = load_tracking_csv(f, raw_id_col=args.raw_id_col)
        df = merge_subject_metadata(df, subjects)

        df["assay"] = "OFT"
        df["source_file"] = str(f)

        # OFT/EPM do not encode timepoint in filename; treat this run as one session.
        df["timepoint"] = "Single"

        dfs.append(df)

    oft_long = pd.concat(dfs, ignore_index=True)

    # QC + derive
    qc_oft(oft_long)
    oft_processed = derive_oft_metrics(oft_long, cfg=OFTConfig())

    out_path = Path(args.out) if args.out else Path("data/processed/OFT_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oft_processed.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(oft_processed)} | Unique mice: {oft_processed['mouse_id'].nunique()} | Files: {len(files)}")
    print(oft_processed[["timepoint", "group", "sex"]].value_counts().sort_index().to_string())


ASSAY_RUNNERS = {
    "VIT": run_vit, 
    "OFT": run_oft,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True, choices=sorted(ASSAY_RUNNERS.keys()), help="Which assay pipeline to run")
    ap.add_argument("--root", required=True, help="Root folder containing raw CSV outputs")
    ap.add_argument("--pattern", default="*_output_summary.csv", help="Glob pattern for raw CSVs")
    ap.add_argument("--subjects", default="data/metadata/subjects.csv", help="Path to subjects.csv")
    ap.add_argument("--raw_id_col", default="Unnamed: 0", help="Raw ID column in tracking CSV")
    ap.add_argument("--out", default="", help="Output processed CSV path")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    ASSAY_RUNNERS[args.assay](args)


if __name__ == "__main__":
    main()
