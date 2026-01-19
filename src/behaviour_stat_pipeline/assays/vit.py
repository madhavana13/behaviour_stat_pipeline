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

import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key

def vit_missing_cells_report(df: pd.DataFrame) -> str:
    """
    Report missing mouse_id × timepoint × phase cells.

    Intended for logging only (no dropping).
    """
    required = {"mouse_id", "timepoint", "phase"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        return f"VIT completeness: cannot evaluate (missing columns: {sorted(missing_cols)})"

    expected_cells = [("Before", "hand"), ("Before", "standing"), ("After", "hand"), ("After", "standing")]

    mice = pd.Index(sorted(df["mouse_id"].dropna().astype(str).unique()), name="mouse_id")

    # observed cells per mouse
    obs = (
        df[["mouse_id", "timepoint", "phase"]]
        .dropna()
        .drop_duplicates()
        .set_index(["mouse_id", "timepoint", "phase"])
        .sort_index()
    )

    # full expected index
    full = pd.MultiIndex.from_tuples(
        [(mid, tp, ph) for mid in mice for tp, ph in expected_cells],
        names=["mouse_id", "timepoint", "phase"],
    )

    missing_idx = full.difference(obs.index)
    if len(missing_idx) == 0:
        return "VIT completeness: OK (all mouse_id × timepoint × phase cells present)."

    miss_df = missing_idx.to_frame(index=False)

    per_mouse = (
        miss_df.groupby("mouse_id")[["timepoint", "phase"]]
        .apply(lambda x: [f"{tp}:{ph}" for tp, ph in x.to_numpy()])
        .to_dict()
    )

    examples = list(per_mouse.items())[:10]
    example_str = "; ".join([f"{mid} -> {cells}" for mid, cells in examples])

    return (
        f"VIT completeness WARNING: {len(per_mouse)}/{len(mice)} mice missing ≥1 cell.\n"
        f"Examples (up to 10): {example_str}"
    )

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

    # uniqueness (reusable QC primitive)
    assert_unique_key(df, ["mouse_id", "timepoint", "phase"], where="VIT long")

    # basic missingness in metadata
    for col in ["mouse_id", "group", "sex"]:
        if df[col].isna().any():
            raise ValueError(f"VIT QC failed: NA values found in column '{col}'")

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

    return out


# Optional: declare suggested primary outcomes once you decide column names
PRIMARY_OUTCOMES: list[str] = []
