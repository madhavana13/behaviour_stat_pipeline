"""
OFT (Open Field Test) assay helpers.

Design decision:
- Processed output = one row per mouse_id × timepoint.

Current implementation:
- OFT exports are typically already one row per mouse per session (summary metrics).
- We therefore treat the ingestion output as already at the session level and:
    - QC: checks required columns + uniqueness on mouse_id × timepoint
    - Derive: adds cross-assay flags (is_after) only; does not aggregate

If you later ingest frame-wise OFT data, we can extend derive_oft_metrics
to aggregate to mouse_id × timepoint explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key

REQUIRED_COLUMNS = {"mouse_id", "group", "sex", "timepoint"}


@dataclass(frozen=True)
class OFTConfig:
    """Configuration placeholder for future OFT-specific derived metrics."""
    pass


def qc_oft(df: pd.DataFrame) -> None:
    """
    QC for OFT ingestion output.

    Assumes ingestion has already produced session-level rows (summary per mouse).
    Enforces uniqueness at mouse_id × timepoint.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"OFT QC failed: missing required columns: {sorted(missing)}")

    for col in ["mouse_id", "group", "sex", "timepoint"]:
        if df[col].isna().any():
            raise ValueError(f"OFT QC failed: NA values found in column '{col}'")

    assert_unique_key(df, ["mouse_id", "timepoint"], where="OFT long/session-level")


def derive_oft_metrics(df: pd.DataFrame, cfg: OFTConfig | None = None) -> pd.DataFrame:
    """
    Add derived metrics for OFT.

    For now:
    - does not drop rows
    - returns a copy
    - adds only cross-assay flags

    Later we can add canonical OFT outcomes and/or change scores explicitly.
    """
    cfg = cfg or OFTConfig()

    qc_oft(df)

    out = df.copy()
    out["is_after"] = (out["timepoint"].astype(str).str.lower() == "after").astype(int)
    return out


PRIMARY_OUTCOMES: list[str] = []
