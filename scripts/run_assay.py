# src/behaviour_stat_pipeline/run_assay.py

import argparse
from pathlib import Path

import pandas as pd

# Assay modules
from behaviour_stat_pipeline.assays.vit import qc_vit, derive_vit_metrics, VITConfig
from behaviour_stat_pipeline.assays.oft import qc_oft, derive_oft_metrics, OFTConfig
from behaviour_stat_pipeline.assays.epm import qc_epm, derive_epm_metrics, EPMConfig
from behaviour_stat_pipeline.common.qc import warn_missing_subjects_coverage
from behaviour_stat_pipeline.assays.nor import derive_nor_metrics, NORConfig, summarize_nor_sanity
from behaviour_stat_pipeline.common.derive import save_processed



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
        df = merge_subject_metadata(df, subjects, on_missing_subjects="warn_drop")

        df["assay"] = "VIT"
        df["source_file"] = str(f)

        tp, ph = infer_vit_timepoint_phase_from_filename(f)
        df["timepoint"] = tp
        df["phase"] = ph

        dfs.append(df)

    vit_long = pd.concat(dfs, ignore_index=True)
    from behaviour_stat_pipeline.common.qc import warn_missing_subjects_coverage
    warn_missing_subjects_coverage(vit_long["mouse_id"], subjects, where="VIT")

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
        df = merge_subject_metadata(df, subjects, on_missing_subjects="warn_drop")

        df["assay"] = "OFT"
        df["source_file"] = str(f)

        # OFT/EPM do not encode timepoint in filename; treat this run as one session.
        df["timepoint"] = "Single"

        dfs.append(df)

    oft_long = pd.concat(dfs, ignore_index=True)
    from behaviour_stat_pipeline.common.qc import warn_missing_subjects_coverage
    warn_missing_subjects_coverage(oft_long["mouse_id"], subjects, where="OFT")

    # QC + derive
    qc_oft(oft_long)
    oft_processed = derive_oft_metrics(oft_long, cfg=OFTConfig())

    out_path = Path(args.out) if args.out else Path("data/processed/OFT_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oft_processed.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(oft_processed)} | Unique mice: {oft_processed['mouse_id'].nunique()} | Files: {len(files)}")
    print(oft_processed[["timepoint", "group", "sex"]].value_counts().sort_index().to_string())

def run_epm(args: argparse.Namespace) -> None:
    root = Path(args.root)
    files = sorted(root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching {args.pattern}")

    subjects = load_subjects(args.subjects)

    dfs = []
    for f in files:
        df = load_tracking_csv(f, raw_id_col=args.raw_id_col)
        df = merge_subject_metadata(df, subjects, on_missing_subjects="warn_drop")

        df["assay"] = "EPM"
        df["source_file"] = str(f)
        df["timepoint"] = "Single"  # session label for now

        dfs.append(df)

    epm_long = pd.concat(dfs, ignore_index=True)
    from behaviour_stat_pipeline.common.qc import warn_missing_subjects_coverage
    warn_missing_subjects_coverage(epm_long["mouse_id"], subjects, where="EPM")


    qc_epm(epm_long)
    epm_processed = derive_epm_metrics(epm_long, cfg=EPMConfig())

    out_path = Path(args.out) if args.out else Path("data/processed/EPM_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epm_processed.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(epm_processed)} | Unique mice: {epm_processed['mouse_id'].nunique()} | Files: {len(files)}")
    print(epm_processed[["timepoint", "group", "sex"]].value_counts().sort_index().to_string())

def run_nor(args: argparse.Namespace) -> None:
    root = Path(args.root)

    fam_files = sorted(root.rglob("*FAM*_output_summary.csv"))
    test_files = sorted(root.rglob("*TEST*_output_summary.csv"))

    if not fam_files:
        raise FileNotFoundError(f"No FAM files found under {root} matching *FAM*_output_summary.csv")
    if not test_files:
        raise FileNotFoundError(f"No TEST files found under {root} matching *TEST*_output_summary.csv")

    subjects = load_subjects(args.subjects)

    def _load(files: list[Path], session: str) -> pd.DataFrame:
        dfs = []
        for f in files:
            df = load_tracking_csv(f, raw_id_col=args.raw_id_col)
            df = merge_subject_metadata(df, subjects, on_missing_subjects="warn_drop")

            df["assay"] = "NOR"
            df["source_file"] = str(f)
            df["timepoint"] = "Single"   # per your current plan: analyze sessions separately across runs
            df["session"] = session

            dfs.append(df)
        out = pd.concat(dfs, ignore_index=True)
        warn_missing_subjects_coverage(out["mouse_id"], subjects, where=f"NOR {session}")
        return out

    fam = _load(fam_files, "FAM")
    test = _load(test_files, "TEST")

    nor_processed = derive_nor_metrics(fam, test, cfg=NORConfig())
    print(summarize_nor_sanity(nor_processed, min_total_s=20.0))

    out_path = Path(args.out) if args.out else Path("data/processed/NOR_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sanity = summarize_nor_sanity(nor_processed, min_total_s=20.0)

    res = save_processed(
        nor_processed,
        out_csv=out_path,
        assay="NOR",
        files_n=len(fam_files) + len(test_files),
        extra_lines=[sanity],
    )

    print(f"Saved: {res.out_csv}")
    print(f"Summary: {res.out_summary}")

ASSAY_RUNNERS = {
    "VIT": run_vit, 
    "OFT": run_oft,
    "EPM": run_epm,
    "NOR": run_nor,
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
