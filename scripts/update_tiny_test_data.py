from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


PROVENANCE_CANDIDATES = [
    "raw_id",
    "raw_id_fam",
    "raw_id_test",
    "source_file",
    "source_file_fam",
    "source_file_test",
]


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}\n{e}") from e


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def choose_mouse_ids(subjects_csv: Path, n: int, seed: int | None) -> list[str]:
    subjects = read_csv_safe(subjects_csv)
    if "mouse_id" not in subjects.columns:
        raise ValueError(f"'mouse_id' column not found in {subjects_csv}")

    ids = subjects["mouse_id"].dropna().astype(str).unique().tolist()
    if len(ids) < n:
        raise ValueError(f"Requested {n} IDs but only found {len(ids)} in {subjects_csv}")

    rng = random.Random(seed)
    return sorted(rng.sample(ids, n))


def filter_processed_file(processed_csv: Path, mouse_ids: set[str]) -> pd.DataFrame:
    df = read_csv_safe(processed_csv)
    if "mouse_id" not in df.columns:
        raise ValueError(f"'mouse_id' column not found in processed file: {processed_csv}")
    return df[df["mouse_id"].astype(str).isin(mouse_ids)].copy()


def collect_raw_refs(df: pd.DataFrame) -> set[str]:
    refs: set[str] = set()

    for col in df.columns:
        col_l = col.lower()
        if "raw_id" in col_l or "source_file" in col_l:
            vals = df[col].dropna().astype(str).str.strip()

            for v in vals:
                if not v:
                    continue

                # Split combined provenance fields like "file1 | file2"
                parts = [part.strip() for part in v.split("|") if part.strip()]
                refs.update(parts)

    return refs


def copy_raw_matches(raw_dir: Path, raw_refs: set[str], out_dir: Path) -> list[Path]:
    ensure_dir(out_dir)
    copied: list[Path] = []

    if not raw_dir.exists():
        return copied

    all_files = [p for p in raw_dir.rglob("*") if p.is_file()]

    for f in all_files:
        f_str = str(f)
        stem = f.stem
        name = f.name

        matched = False
        for ref in raw_refs:
            # Match exact path, filename, or stem-contained provenance token
            if ref == f_str or ref == name or ref == stem or ref in f_str or ref in stem:
                matched = True
                break

        if matched:
            dst = out_dir / f.name
            shutil.copy2(f, dst)
            copied.append(dst)

    return copied


def assay_name_from_processed(path: Path) -> str:
    return path.stem.replace("_processed", "").upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment ID, e.g. EXP-2026-002")
    parser.add_argument("--n", type=int, default=4, help="Number of mouse IDs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root path",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing tiny_test_data/<experiment> before rebuilding",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    exp = args.experiment

    exp_root = repo_root / "data" / "experiments" / exp
    subjects_csv = exp_root / "metadata" / "subjects.csv"
    processed_dir = exp_root / "processed"
    raw_root = exp_root / "raw"

    out_root = repo_root / "behaviour_stat_pipeline_review" / "tiny_test_data" / exp
    out_raw_root = out_root / "raw"
    out_processed_root = out_root / "processed"

    if args.fresh:
        clear_dir(out_root)
    else:
        ensure_dir(out_root)

    ensure_dir(out_raw_root)
    ensure_dir(out_processed_root)

    shutil.copy2(subjects_csv, out_root / "subjects.csv")

    chosen_ids = choose_mouse_ids(subjects_csv, args.n, args.seed)
    chosen_set = set(chosen_ids)

    print(f"Chosen mouse IDs ({len(chosen_ids)}): {chosen_ids}")

    processed_files = sorted(processed_dir.glob("*_processed.csv"))
    if not processed_files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")

    summary_rows: list[dict] = []

    for processed_csv in processed_files:
        assay = assay_name_from_processed(processed_csv)
        filtered = filter_processed_file(processed_csv, chosen_set)

        if filtered.empty:
            print(f"[SKIP] {assay}: no sampled IDs found")
            continue

        out_processed_csv = out_processed_root / processed_csv.name
        filtered.to_csv(out_processed_csv, index=False)

        raw_refs = collect_raw_refs(filtered)
        print(f"[DEBUG] {assay}: collected {len(raw_refs)} provenance refs")
        assay_raw_dir = raw_root / assay.lower()
        copied = copy_raw_matches(assay_raw_dir, raw_refs, out_raw_root / assay.lower())

        summary_rows.append(
            {
                "assay": assay,
                "processed_rows": len(filtered),
                "raw_refs_found": len(raw_refs),
                "raw_files_copied": len(copied),
                "processed_out": str(out_processed_csv.relative_to(repo_root)),
            }
        )

        print(f"[OK] {assay}: wrote {len(filtered)} processed rows, copied {len(copied)} raw files")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_root / "tiny_data_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    selected_path = out_root / "selected_mouse_ids.txt"
    selected_path.write_text("\n".join(chosen_ids) + "\n", encoding="utf-8")

    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote sampled IDs: {selected_path}")


if __name__ == "__main__":
    main()