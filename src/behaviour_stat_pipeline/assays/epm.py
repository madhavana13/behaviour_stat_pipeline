"""
EPM (Elevated Plus Maze) assay helpers.

Design decision:
- Processed output = one row per mouse_id × timepoint.

Current implementation:
- EPM exports are typically already one row per mouse per session (summary metrics).
- We therefore treat the ingestion output as session-level rows and:
    - QC: checks required columns + uniqueness on mouse_id × timepoint
    - Derive: adds cross-assay flags only; does not aggregate

If you later ingest event-wise or frame-wise EPM data, we can extend derive_epm_metrics
to aggregate to mouse_id × timepoint explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key

REQUIRED_COLUMNS = {"mouse_id", "group", "sex", "timepoint"}


@dataclass(frozen=True)
class EPMConfig:
    """Configuration placeholder for future EPM-specific derived metrics."""
    pass


def qc_epm(df: pd.DataFrame) -> None:
    """
    QC for EPM ingestion output.

    Assumes ingestion has already produced session-level rows (summary per mouse).
    Enforces uniqueness at mouse_id × timepoint.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"EPM QC failed: missing required columns: {sorted(missing)}")

    for col in ["mouse_id", "group", "sex", "timepoint"]:
        if df[col].isna().any():
            raise ValueError(f"EPM QC failed: NA values found in column '{col}'")

    assert_unique_key(df, ["mouse_id", "timepoint"], where="EPM long/session-level")


def derive_epm_metrics(df: pd.DataFrame, cfg: EPMConfig | None = None) -> pd.DataFrame:
    """
    Add derived metrics for EPM.

    For now:
    - does not drop rows
    - returns a copy
    - adds only cross-assay flags

    Later we can add canonical EPM outcomes (e.g., open-arm ratio) explicitly.
    """
    cfg = cfg or EPMConfig()

    qc_epm(df)

    out = df.copy()
    return out


PRIMARY_OUTCOMES: list[str] = []
