"""
VIT (Voluntary Interaction Test) assay helpers.

Assumptions / design:
- Long-format table with columns:
    mouse_id, group, sex, timepoint (Before/After), phase (hand/standing)
- One observation per mouse_id × timepoint × phase (enforced by QC).

Processed output contract:
- One row per mouse
- Base columns:
    {phase}__{timepoint}__{metric}
- Derived columns:
    delta_after_minus_before__{phase}__{metric}
    did__{metric}

Only schema-approved VIT metrics are allowed into the processed endpoint table.
Structural/diagnostic numeric columns must not be promoted to endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key

PROVENANCE_COLS = ["raw_id", "source_file"]

def vit_missing_cells_report(df: pd.DataFrame) -> str:
    """
    Report missing mouse_id × timepoint × phase cells.

    Intended for logging only (no dropping).
    """
    required = {"mouse_id", "timepoint", "phase"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        return f"VIT completeness: cannot evaluate (missing columns: {sorted(missing_cols)})"

    expected_cells = [
        ("Before", "hand"),
        ("Before", "standing"),
        ("After", "hand"),
        ("After", "standing"),
    ]

    mice = pd.Index(sorted(df["mouse_id"].dropna().astype(str).unique()), name="mouse_id")

    obs = (
        df[["mouse_id", "timepoint", "phase"]]
        .dropna()
        .drop_duplicates()
        .set_index(["mouse_id", "timepoint", "phase"])
        .sort_index()
    )

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

# Canonical VIT endpoint schema for processed outputs.
# These are the only measures that should be widened and contrasted.
VIT_ALLOWED_METRICS = (
    "time_in_homecage",
    "total_distance_moved",
    "mean_speed",
    "time_in_left_upper_quadrant",
    "distance_moved_in_left_upper_quadrant",
    "transitions_into_left_upper_quadrant",
    "time_in_right_upper_quadrant",
    "distance_moved_in_right_upper_quadrant",
    "transitions_into_right_upper_quadrant",
    "time_in_left_lower_quadrant",
    "distance_moved_in_left_lower_quadrant",
    "transitions_into_left_lower_quadrant",
    "time_in_right_lower_quadrant",
    "distance_moved_in_right_lower_quadrant",
    "transitions_into_right_lower_quadrant",
)

# Known legacy/raw spellings that may still survive normalization.
# Keep this narrow and explicit.
VIT_LEGACY_RENAMES = {
    "time in left upper quadrant": "time_in_left_upper_quadrant",
    "time in right upper quadrant": "time_in_right_upper_quadrant",
    "time in left lower quadrant": "time_in_left_lower_quadrant",
    "time in right lower quadrant": "time_in_right_lower_quadrant",
}

# Numeric columns that are structural/diagnostic and must never become endpoints.
VIT_EXCLUDED_NUMERIC_COLUMNS = {
    "frame",
    "sl_no",
    "cage",
    "tse_box",
    "is_after",
    "is_hand",
}


@dataclass(frozen=True)
class VITConfig:
    """
    Configuration for derived metrics and QC.

    strict_metric_schema:
        If True, error when numeric columns outside the approved schema are present
        after legacy renaming and exclusions.
        If False, ignore extra numeric columns silently.
    """
    strict_metric_schema: bool = False


def qc_vit(df: pd.DataFrame) -> None:
    """
    Strict QC for VIT ingestion output.

    Raises ValueError with informative messages if violations are detected.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"VIT QC failed: missing required columns: {sorted(missing)}")

    bad_tp = set(df["timepoint"].dropna().unique()) - ALLOWED_TIMEPOINTS
    if bad_tp:
        raise ValueError(f"VIT QC failed: unexpected timepoint values: {sorted(bad_tp)}")

    bad_ph = set(df["phase"].dropna().unique()) - ALLOWED_PHASES
    if bad_ph:
        raise ValueError(f"VIT QC failed: unexpected phase values: {sorted(bad_ph)}")

    assert_unique_key(df, ["mouse_id", "timepoint", "phase"], where="VIT long")

    for col in ["mouse_id", "group", "sex"]:
        if df[col].isna().any():
            raise ValueError(f"VIT QC failed: NA values found in column '{col}'")


def _canonicalize_vit_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize known legacy VIT metric names to canonical snake_case names.

    This is intentionally narrow: we only rename specific known legacy headers.
    """
    rename_map = {c: VIT_LEGACY_RENAMES[c] for c in df.columns if c in VIT_LEGACY_RENAMES}
    if not rename_map:
        return df.copy()

    out = df.copy()
    out = out.rename(columns=rename_map)

    # Defensive check: no duplicate columns after renaming.
    dupes = out.columns[out.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"VIT metric canonicalization created duplicate columns: {sorted(set(dupes))}"
        )
    return out


def _select_vit_metric_columns(df: pd.DataFrame, cfg: VITConfig) -> list[str]:
    """
    Return schema-approved metric columns for VIT wide derivation.
    """
    id_cols = {"mouse_id", "group", "sex", "timepoint", "phase"}

    numeric_candidates = [
        c for c in df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    unexpected_numeric = sorted(
        c for c in numeric_candidates
        if c not in VIT_ALLOWED_METRICS and c not in VIT_EXCLUDED_NUMERIC_COLUMNS
    )

    if cfg.strict_metric_schema and unexpected_numeric:
        raise ValueError(
            "VIT strict metric schema failed. Unexpected numeric columns present: "
            f"{unexpected_numeric}"
        )

    metric_cols = [c for c in VIT_ALLOWED_METRICS if c in df.columns]

    if not metric_cols:
        raise ValueError(
            "VIT derivation failed: no schema-approved metrics found. "
            f"Expected at least one of: {list(VIT_ALLOWED_METRICS)}"
        )

    return metric_cols


def _derive_vit_wide(df_long: pd.DataFrame, cfg: VITConfig) -> pd.DataFrame:
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

    df_long = _canonicalize_vit_metric_columns(df_long)
    metric_cols = _select_vit_metric_columns(df_long, cfg=cfg)
    prov_cols = [c for c in PROVENANCE_COLS if c in df_long.columns]

    wide = (
        df_long
        .set_index(id_cols + within)[metric_cols]
        .unstack(["timepoint", "phase"])
    )

    if not isinstance(wide.columns, pd.MultiIndex) or wide.columns.nlevels != 3:
        raise ValueError(
            f"Unexpected VIT wide column structure: expected 3-level MultiIndex, got {wide.columns}"
        )

    # (metric, timepoint, phase) -> (phase, timepoint, metric)
    wide = wide.reorder_levels([2, 1, 0], axis=1).sort_index(axis=1)
    wide.columns = [
        f"{phase}__{timepoint}__{metric}"
        for phase, timepoint, metric in wide.columns
    ]
    wide = wide.reset_index()

    if prov_cols:
        prov_wide = (
            df_long
            .set_index(id_cols + within)[prov_cols]
            .unstack(["timepoint", "phase"])
        )

        prov_wide = prov_wide.reorder_levels([2, 1, 0], axis=1).sort_index(axis=1)

        prov_wide.columns = [
            f"{phase}__{timepoint}__{col}"
            for phase, timepoint, col in prov_wide.columns
        ]

    prov_wide = prov_wide.reset_index()
    derived: dict[str, pd.Series | float] = {}

    def col(ph: str, tp: str, m: str) -> str:
        return f"{ph}__{tp}__{m}"

    phases = ["hand", "standing"]

    # After - Before within phase
    for ph in phases:
        for m in metric_cols:
            a = col(ph, "After", m)
            b = col(ph, "Before", m)
            out = f"delta_after_minus_before__{ph}__{m}"
            derived[out] = wide[a] - wide[b] if a in wide.columns and b in wide.columns else np.nan

    # Difference-in-differences
    for m in metric_cols:
        da = f"delta_after_minus_before__hand__{m}"
        ds = f"delta_after_minus_before__standing__{m}"
        out = f"did__{m}"
        derived[out] = derived[da] - derived[ds] if da in derived and ds in derived else np.nan

    result = pd.concat([wide, pd.DataFrame(derived)], axis=1)

    if prov_cols:
        result = result.merge(prov_wide, on=id_cols, how="left")

    return result


def derive_vit_metrics(df: pd.DataFrame, cfg: VITConfig | None = None) -> pd.DataFrame:
    """
    Derive schema-safe processed VIT metrics.

    Notes
    -----
    - Returns one row per mouse in wide format.
    - Does not allow structural numeric columns to become endpoints.
    - Keeps the processed output contract stable.
    """
    cfg = cfg or VITConfig()
    out = df.copy()

    # These may still be useful for temporary diagnostics during development,
    # but they must never enter the processed endpoint schema because
    # _select_vit_metric_columns() excludes them explicitly.
    out["is_after"] = (out["timepoint"] == "After").astype(int)
    out["is_hand"] = (out["phase"] == "hand").astype(int)

    return _derive_vit_wide(out, cfg=cfg)


PRIMARY_OUTCOMES: list[str] = []