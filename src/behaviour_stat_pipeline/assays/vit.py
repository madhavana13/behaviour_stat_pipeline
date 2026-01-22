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

import numpy as np

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


def _derive_vit_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert VIT long → wide and append standard contrasts.

    Wide columns:
      {phase}__{timepoint}__{metric}

    Derived:
      delta_after_minus_before__{phase}__{metric}
      did__{metric}
    """
    id_cols = ["mouse_id", "group", "sex"]
    within = ["timepoint", "phase"]

    metric_cols = [
        c for c in df_long.columns
        if c not in id_cols + within and pd.api.types.is_numeric_dtype(df_long[c])
    ]

    if not metric_cols:
        raise ValueError("[VIT] no numeric metric columns found for wide derivation")

    wide = (
        df_long
        .set_index(id_cols + within)[metric_cols]
        .unstack(["timepoint", "phase"])
    )

    # normalize column order → phase__timepoint__metric
    wide = wide.reorder_levels([2, 1, 0], axis=1).sort_index(axis=1)
    wide.columns = [
        f"{phase}__{timepoint}__{metric}"
        for phase, timepoint, metric in wide.columns
    ]
    wide = wide.reset_index()

    derived = {}

    def col(ph, tp, m):
        return f"{ph}__{tp}__{m}"

    phases = ["hand", "standing"]
    metrics = metric_cols

    for ph in phases:
        for m in metrics:
            a = col(ph, "After", m)
            b = col(ph, "Before", m)
            out = f"delta_after_minus_before__{ph}__{m}"
            if a in wide.columns and b in wide.columns:
                derived[out] = wide[a] - wide[b]
            else:
                derived[out] = np.nan

    for m in metrics:
        da = f"delta_after_minus_before__hand__{m}"
        ds = f"delta_after_minus_before__standing__{m}"
        out = f"did__{m}"
        if da in derived and ds in derived:
            derived[out] = derived[da] - derived[ds]
        else:
            derived[out] = np.nan

    return pd.concat([wide, pd.DataFrame(derived)], axis=1)


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

    # deterministic coding (kept for completeness / diagnostics)
    out["is_after"] = (out["timepoint"] == "After").astype(int)
    out["is_hand"] = (out["phase"] == "hand").astype(int)

    # WIDE + CONTRASTS (authoritative processed output)
    wide = _derive_vit_wide(out)
    return wide


# Optional: declare suggested primary outcomes once you decide column names
PRIMARY_OUTCOMES: list[str] = []
