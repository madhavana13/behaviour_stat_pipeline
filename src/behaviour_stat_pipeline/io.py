from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd


_ID_RE = re.compile(r"(ID\d+)$")


def extract_mouse_id(raw_id: Union[str, int]) -> str:
    """
    Extract canonical mouse_id from raw tracking identifiers.
    Accepts either 'ID17' or strings ending in '_ID17' (or similar prefixes).
    """
    s = str(raw_id).strip()
    m = _ID_RE.search(s)
    if m is None:
        raise ValueError(f"Could not extract mouse_id from raw id: {raw_id!r}")
    return m.group(1)


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

    allowed_groups = {"cup", "tail"}
    bad_groups = set(subjects["group"].unique()) - allowed_groups
    if bad_groups:
        raise ValueError(f"Unexpected group labels in subjects.csv: {sorted(bad_groups)}")

    allowed_sex = {"male", "female"}
    bad_sex = set(subjects["sex"].unique()) - allowed_sex
    if bad_sex:
        raise ValueError(f"Unexpected sex labels in subjects.csv: {sorted(bad_sex)}")

    return subjects


def load_tracking_csv(path: Union[str, Path], raw_id_col: str = "Unnamed: 0") -> pd.DataFrame:
    df = pd.read_csv(path)
    if raw_id_col not in df.columns:
        raise ValueError(
            f"Raw ID column '{raw_id_col}' not found in {path}. "
            f"Columns present: {list(df.columns)[:10]}..."
        )
    df = df.copy()
    df["mouse_id"] = df[raw_id_col].apply(extract_mouse_id)
    return df


def merge_subject_metadata(df: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(subjects, on="mouse_id", how="left", validate="many_to_one")

    if out["group"].isna().any() or out["sex"].isna().any():
        missing_ids = out.loc[out["group"].isna() | out["sex"].isna(), "mouse_id"].unique()
        raise ValueError(
            "Some mouse_ids in the tracking CSV are missing from subjects.csv: "
            f"{sorted(missing_ids)}"
        )

    return out
