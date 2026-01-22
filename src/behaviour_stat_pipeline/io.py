from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd


_MOUSE_ID_PATTERNS = [
    # NOR-style: 27A_FAM_ID9 or 27B_TEST_ID15
    re.compile(r"(?P<cage>\d+[A-Z]?)_(?:FAM|TEST)_ID(?P<num>\d+)\b", re.IGNORECASE),
    # Generic: 27A_ID9
    re.compile(r"(?P<cage>\d+[A-Z]?)_ID(?P<num>\d+)\b", re.IGNORECASE),
]

def extract_mouse_id(raw: str) -> str:
    s = str(raw).strip()

    # NEW: accept already-normalized internal IDs like "ID10"
    if re.fullmatch(r"(?i)ID\d+", s):
        return s.upper()

    for pat in _MOUSE_ID_PATTERNS:
        m = pat.search(s)
        if m:
            return f"{m.group('cage')}_ID{m.group('num')}"
    raise ValueError(f"Could not extract mouse_id from raw identifier: {raw!r}")


def load_subjects(subjects_csv: Union[str, Path]) -> pd.DataFrame:
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

    # normalize coding (important for stable modelling later)
    subjects["mouse_id"] = subjects["mouse_id"].astype(str).str.strip()
    subjects["group"] = subjects["group"].astype(str).str.strip().str.lower()
    subjects["sex"] = subjects["sex"].astype(str).str.strip().str.lower()

    allowed_groups = {"cup", "tail", "control", "stress"}
    bad_groups = set(subjects["group"].unique()) - allowed_groups
    if bad_groups:
        raise ValueError(f"Unexpected group labels in subjects.csv: {sorted(bad_groups)}")

    allowed_sex = {"male", "female"}
    bad_sex = set(subjects["sex"].unique()) - allowed_sex
    if bad_sex:
        raise ValueError(f"Unexpected sex labels in subjects.csv: {sorted(bad_sex)}")

    return subjects


def load_tracking_csv(path: Union[str, Path], raw_id_col: str = "Unnamed: 0") -> pd.DataFrame:
    """
    Load a tracking summary CSV and construct canonical identifiers.

    Adds:
      - raw_id   : original identifier string from file
      - mouse_id : canonical globally unique id (via extract_mouse_id)
    """
    df = pd.read_csv(path)

    # Robust raw ID column detection (Excel can mangle "Unnamed: 0")
    if raw_id_col not in df.columns:
        if "Unnamed" in df.columns and "0" in df.columns and df["0"].isna().all():
            raw_id_col = "Unnamed"
            df = df.drop(columns=["0"])
        elif df.columns[0].lower().startswith("unnamed"):
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
    on_missing_subjects: str = "warn_drop",  # {"error", "warn_drop"}
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
