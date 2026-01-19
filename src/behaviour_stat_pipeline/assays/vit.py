"""
VIT (Voluntary Interaction Test) assay helpers.

Assumptions / design:
- Long-format table with columns:
    mouse_id, group, sex, timepoint (Before/After), phase (hand/standing)
- One observation per mouse_id × timepoint × phase (enforced by QC).

This module is intentionally narrow:
- derive_vit_metrics(df): add derived columns (does not drop rows)
- qc_vit(df): enforce design constraints + basic sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"mouse_id", "group", "sex", "timepoint", "phase"}
ALLOWED_TIMEPOINTS = {"Before", "After"}
ALLOWED_PHASES = {"hand", "standing"}


@dataclass(frozen=True)
class VITConfig:
    """Configuration for derived metrics and QC."""
    # If you later decide to compute within-mouse change scores, do it explicitly.
    compute_change_scores: bool = False


def qc_vit(df: pd.DataFrame) -> None:
    """
    Strict QC for VIT ingestion output.

    Raises ValueError with informative messages if violations are detected.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"VIT QC failed: missing required columns: {sorted(missing)}")

    # factor levels
    bad_tp = set(df["timepoint"].dropna().unique()) - ALLOWED_TIMEPOINTS
    if bad_tp:
        raise ValueError(f"VIT QC failed: unexpected timepoint values: {sorted(bad_tp)}")

    bad_ph = set(df["phase"].dropna().unique()) - ALLOWED_PHASES
    if bad_ph:
        raise ValueError(f"VIT QC failed: unexpected phase values: {sorted(bad_ph)}")

    # uniqueness
    key = ["mouse_id", "timepoint", "phase"]
    if df.duplicated(subset=key).any():
        examples = df.loc[df.duplicated(subset=key, keep=False), key].head(20)
        raise ValueError(
            "VIT QC failed: duplicate rows for mouse_id × timepoint × phase.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # basic missingness in metadata
    for col in ["mouse_id", "group", "sex"]:
        if df[col].isna().any():
            raise ValueError(f"VIT QC failed: NA values found in column '{col}'")

    # quick balance print suggestion (not enforced)
    # We intentionally do not enforce perfect balance here, since later datasets may be unbalanced.


def derive_vit_metrics(df: pd.DataFrame, cfg: VITConfig | None = None) -> pd.DataFrame:
    """
    Add derived metrics for VIT.

    This function:
    - does not drop any rows
    - returns a copy with additional columns
    - keeps raw columns intact

    Derived columns added:
    - is_after: 0/1 indicator
    - is_hand: 0/1 indicator

    (Future) If you decide to compute within-mouse change scores, implement them here
    and store them as separate columns, without overwriting raw outcomes.
    """
    cfg = cfg or VITConfig()

    out = df.copy()

    # deterministic coding
    out["is_after"] = (out["timepoint"] == "After").astype(int)
    out["is_hand"] = (out["phase"] == "hand").astype(int)

    # NOTE: We do NOT compute effect-size-like change scores by default because
    # they depend on which raw outcome column you treat as primary.
    # When you pick primary outcomes (e.g., time_in_hand_quadrant or contact_time),
    # we can add:
    #   delta_hand = hand_after - hand_before
    # without altering the raw table.

    return out


# Optional: declare suggested primary outcomes once you decide column names
PRIMARY_OUTCOMES: list[str] = []