# src/behaviour_stat_pipeline/assays/lm.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np

from behaviour_stat_pipeline.common.qc import assert_unique_key

REQUIRED_ID_COLS = {"mouse_id", "group", "sex", "timepoint"}
REQUIRED_RAW_COLS = {
    "time_in_arena",
    "total_distance_moved",
    "time_in_protected_alley",
    "distance_moved_in_protected_alley",
    "transitions_into_protected_alley",
    "time_in_central_alley",
    "distance_moved_in_central_alley",
    "transitions_into_central_alley",
    "time_in_narrow_alley",
    "distance_moved_in_narrow_alley",
    "transitions_into_narrow_alley",
}

@dataclass(frozen=True)
class LMConfig:
    pass

def qc_lm(df: pd.DataFrame) -> None:
    missing = (REQUIRED_ID_COLS | REQUIRED_RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"LM QC failed: missing columns: {sorted(missing)}")

    for col in ["mouse_id", "group", "sex", "timepoint"]:
        if df[col].isna().any():
            raise ValueError(f"LM QC failed: NA values in '{col}'")

    assert_unique_key(df, ["mouse_id", "timepoint"], where="LM session-level")

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return num / denom

def derive_lm_metrics(df: pd.DataFrame, cfg: LMConfig | None = None) -> pd.DataFrame:
    cfg = cfg or LMConfig()
    qc_lm(df)

    out = df.copy()

    # Canonical endpoints (copies; raw columns preserved)
    out["lm_time_total_s"] = out["time_in_arena"]
    out["lm_time_protected_s"] = out["time_in_protected_alley"]
    out["lm_time_center_s"] = out["time_in_central_alley"]
    out["lm_time_narrow_s"] = out["time_in_narrow_alley"]

    out["lm_dist_total"] = out["total_distance_moved"]
    out["lm_dist_protected"] = out["distance_moved_in_protected_alley"]
    out["lm_dist_center"] = out["distance_moved_in_central_alley"]
    out["lm_dist_narrow"] = out["distance_moved_in_narrow_alley"]

    out["lm_trans_protected_n"] = out["transitions_into_protected_alley"]
    out["lm_trans_center_n"] = out["transitions_into_central_alley"]
    out["lm_trans_narrow_n"] = out["transitions_into_narrow_alley"]

    # Derived (append-only)
    out["lm_pct_time_narrow"] = _safe_div(out["lm_time_narrow_s"], out["lm_time_total_s"])
    out["lm_pct_time_center"] = _safe_div(out["lm_time_center_s"], out["lm_time_total_s"])
    out["lm_open_like_index"] = _safe_div(
        out["lm_time_narrow_s"],
        out["lm_time_narrow_s"] + out["lm_time_protected_s"],
    )
    # --- Unprotected zone (center + narrow) ---
    out["lm_time_unprotected_s"] = out["lm_time_center_s"] + out["lm_time_narrow_s"]

    out["lm_pct_time_unprotected"] = _safe_div(
        out["lm_time_unprotected_s"], out["lm_time_total_s"]
    )

    # Narrow preference *within* unprotected area
    out["lm_pct_narrow_within_unprotected"] = _safe_div(
        out["lm_time_narrow_s"], out["lm_time_unprotected_s"]
    )

    # Direct contrast: narrow vs center
    out["lm_delta_time_narrow_minus_center_s"] = (
        out["lm_time_narrow_s"] - out["lm_time_center_s"]
    )


    out["is_after"] = (out["timepoint"].astype(str).str.lower() == "after").astype(int)

    # Contract invariant
    assert_unique_key(out, ["mouse_id", "timepoint"], where="LM processed")

    return out

PRIMARY_OUTCOMES: list[str] = [
    "lm_open_like_index",
    "lm_pct_time_narrow",
    "lm_time_narrow_s",
]
