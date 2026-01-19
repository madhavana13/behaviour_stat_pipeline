from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SaveProcessedResult:
    out_csv: Path
    out_summary: Path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_processed(
    df: pd.DataFrame,
    *,
    out_csv: str | Path,
    assay: str,
    files_n: int | None = None,
    extra_lines: Iterable[str] = (),
    summary_dir: str | Path = "outputs/logs",
) -> SaveProcessedResult:
    """
    Save processed CSV + a human-readable run summary file.

    This is intended for .bat-driven runs: nothing important should only appear in stdout.

    Writes:
    - out_csv (CSV)
    - outputs/logs/{ASSAY}_processed_summary.txt (or same name if you prefer)

    Returns paths for downstream use.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_summary = summary_dir / f"{assay}_processed_summary.txt"

    # Write CSV
    df.to_csv(out_csv, index=False)

    # Build summary text
    lines: list[str] = []
    lines.append(f"[{_now_stamp()}] ASSAY={assay}")
    lines.append(f"Saved CSV: {out_csv.as_posix()}")
    lines.append(f"Rows: {len(df)}")

    if "mouse_id" in df.columns:
        lines.append(f"Unique mice: {df['mouse_id'].nunique()}")

    if files_n is not None:
        lines.append(f"Files: {files_n}")

    # Optional standard breakdowns if columns exist
    breakdown_cols = [c for c in ["timepoint", "phase", "session", "group", "sex"] if c in df.columns]
    if breakdown_cols:
        lines.append("")
        lines.append("Counts:")
        counts = df[breakdown_cols].value_counts().sort_index()
        lines.append(counts.to_string())

    # Any extra caller-provided lines (warnings, sanity summaries, etc.)
    for s in extra_lines:
        s = str(s).strip("\n")
        if s:
            lines.append("")
            lines.append(s)

    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return SaveProcessedResult(out_csv=out_csv, out_summary=out_summary)
