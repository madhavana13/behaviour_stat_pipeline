from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd


_ID_RE = re.compile(r"(ID\d+)$")


def extract_mouse_id(raw_id: str) -> str:
    raw_id = str(raw_id)
    m = re.search(r"(27[AB]_ID\d+)$", raw_id)
    if not m:
        raise ValueError(f"Could not extract unique mouse_id from: {raw_id}")
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


def load_tracking_csv(path, raw_id_col: str = "Unnamed: 0"):
    df = pd.read_csv(path)

    # Robust raw ID column detection
    if raw_id_col not in df.columns:
        # Excel sometimes splits "Unnamed: 0" into two columns: "Unnamed" and "0"
        if "Unnamed" in df.columns and "0" in df.columns and df["0"].isna().all():
            raw_id_col = "Unnamed"
            df = df.drop(columns=["0"])
        # Fallback: first column looks like IDs
        elif df.columns[0].lower().startswith("unnamed"):
            raw_id_col = df.columns[0]
        else:
            raise ValueError(
                f"Raw ID column not found. Tried '{raw_id_col}'. "
                f"Columns present: {list(df.columns)[:10]}..."
            )

    df = df.copy()
    df["raw_id"] = df[raw_id_col].astype(str).str.strip()  # keep original identifier
    df["mouse_id"] = df["raw_id"].apply(extract_mouse_id)
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
