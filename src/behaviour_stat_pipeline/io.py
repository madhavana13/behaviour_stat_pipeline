from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Repository root (src/behaviour_stat_pipeline/io.py -> repo); configs are found relative to it
REPO_ROOT = Path(__file__).resolve().parents[2]

from behaviour_stat_pipeline.common.schema import load_schema_aliases, normalize_headers


_MOUSE_ID_PATTERNS = [
    re.compile(r"(?P<cage>\d+[A-Z]?)_(?:FAM|TEST)_ID(?P<num>\d+)\b", re.IGNORECASE),
    re.compile(r"(?P<cage>\d+[A-Z]?)_ID(?P<num>\d+)\b", re.IGNORECASE),
]


def extract_mouse_id(raw: str) -> str:
    s = str(raw).strip()

    if re.fullmatch(r"(?i)ID\d+", s):
        return s.upper()

    for pat in _MOUSE_ID_PATTERNS:
        m = pat.search(s)
        if m:
            return f"{m.group('cage')}_ID{m.group('num')}"
    raise ValueError(f"Could not extract mouse_id from raw identifier: {raw!r}")


def load_subjects(
    subjects_csv: Union[str, Path],
    *,
    allowed_groups: Optional[set[str]] = None,
    strict_groups: bool = False,
    allowed_sex: Optional[set[str]] = None,
    strict_sex: bool = True,
) -> pd.DataFrame:
    subjects = pd.read_csv(subjects_csv)

    required = {"mouse_id", "group", "sex"}
    missing = required - set(subjects.columns)
    if missing:
        raise ValueError(f"subjects.csv missing required columns: {sorted(missing)}")

    if subjects["mouse_id"].isna().any():
        raise ValueError("subjects.csv contains NA mouse_id")

    if subjects["mouse_id"].duplicated().any():
        dups = subjects.loc[subjects["mouse_id"].duplicated(), "mouse_id"].unique()
        raise ValueError(f"Duplicate mouse_id in subjects.csv: {dups}")

    subjects["mouse_id"] = subjects["mouse_id"].astype(str).str.strip()
    subjects["group"] = subjects["group"].astype(str).str.strip().str.lower()
    subjects["sex"] = subjects["sex"].astype(str).str.strip().str.lower()

    if allowed_groups is not None:
        bad_groups = set(subjects["group"].unique()) - allowed_groups
        if bad_groups:
            msg = f"Unexpected group labels in subjects.csv: {sorted(bad_groups)}"
            if strict_groups:
                raise ValueError(msg)
            else:
                print(f"WARNING: {msg}")

    if allowed_sex is None:
        allowed_sex = {"male", "female"}
    bad_sex = set(subjects["sex"].unique()) - allowed_sex
    if bad_sex:
        msg = f"Unexpected sex labels in subjects.csv: {sorted(bad_sex)}"
        if strict_sex:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")

    return subjects


def load_tracking_csv(
    path: Union[str, Path],
    raw_id_col: str = "Unnamed: 0",
    *,
    assay: str | None = None,
    schema_path: Union[str, Path] = REPO_ROOT / "configs" / "schema_aliases.yaml",
) -> pd.DataFrame:
    """
    Load a tracking summary CSV, normalize vendor headers, and construct canonical identifiers.

    Adds:
      - raw_id   : original identifier string from file
      - mouse_id : canonical globally unique id (via extract_mouse_id)
    """
    df = pd.read_csv(path)

    schema = load_schema_aliases(schema_path)
    df, rename_log, suspicious = normalize_headers(df, assay=assay, schema=schema)

    if rename_log:
        print(f"[Schema] {Path(path).name}: renamed {len(rename_log)} columns for assay={assay}")
        for old, new in rename_log[:20]:
            print(f"  {old} -> {new}")
        if len(rename_log) > 20:
            print(f"  ... and {len(rename_log) - 20} more")

    if suspicious:
        print(f"[Schema WARNING] {Path(path).name}: suspicious columns retained unchanged: {suspicious}")

    if raw_id_col not in df.columns:
        if "Unnamed" in df.columns and "0" in df.columns and df["0"].isna().all():
            raw_id_col = "Unnamed"
            df = df.drop(columns=["0"])
        elif len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
            raw_id_col = df.columns[0]
        else:
            raise ValueError(
                f"Raw ID column '{raw_id_col}' not found in {path}. "
                f"Columns present: {list(df.columns)[:10]}..."
            )

    df = df.copy()
    df["raw_id"] = df[raw_id_col].astype(str).str.strip()
    df["mouse_id"] = df["raw_id"].apply(extract_mouse_id)
    return df


def merge_subject_metadata(
    df: pd.DataFrame,
    subjects: pd.DataFrame,
    *,
    on_missing_subjects: str = "warn_drop",
) -> pd.DataFrame:
    out = df.merge(subjects, on="mouse_id", how="left", validate="many_to_one")

    if out["group"].isna().any() or out["sex"].isna().any():
        missing_ids = out.loc[out["group"].isna() | out["sex"].isna(), "mouse_id"].unique()
        missing_ids_sorted = sorted(map(str, missing_ids))

        if on_missing_subjects == "error":
            raise ValueError(
                "Some mouse_ids in the tracking CSV are missing from subjects.csv: "
                f"{missing_ids_sorted}"
            )

        if on_missing_subjects == "warn_drop":
            print(
                "WARNING: Some mouse_ids in the tracking CSV are missing from subjects.csv. "
                "Dropping these rows.\n"
                f"Missing mouse_ids ({len(missing_ids_sorted)}): {missing_ids_sorted}"
            )
            out = out.loc[~(out["group"].isna() | out["sex"].isna())].copy()
        else:
            raise ValueError(
                f"Invalid on_missing_subjects={on_missing_subjects!r}. "
                "Use 'error' or 'warn_drop'."
            )

    return out