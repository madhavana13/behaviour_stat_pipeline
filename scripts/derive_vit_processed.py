from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd


INPUT_LONG = Path("data/interim/VIT_long.csv")
OUTPUT_WIDE = Path("data/processed/VIT_processed.csv")

ID_COLS = ["mouse_id", "group", "sex"]
WITHIN_COLS = ["timepoint", "phase"]


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def derive_vit_wide(
    df_long: pd.DataFrame,
    *,
    phase_order: tuple[str, str] = ("hand", "standing"),
    time_order: tuple[str, str] = ("Before", "After"),
) -> pd.DataFrame:
    # --- validate required columns
    req = ID_COLS + WITHIN_COLS
    missing = [c for c in req if c not in df_long.columns]
    if missing:
        raise ValueError(f"Missing required columns in VIT long: {missing}")

    # --- identify metric columns (append-only: we carry ALL numeric measures)
    metric_cols = [
        c for c in df_long.columns
        if c not in req and _is_numeric_series(df_long[c])
    ]
    if not metric_cols:
        raise ValueError("No numeric metric columns found to pivot/derive.")

    # --- pivot to wide: index = mouse, columns = (phase,timepoint,metric)
    # enforce uniqueness by (mouse_id,timepoint,phase)
    dup = df_long.duplicated(["mouse_id", "timepoint", "phase"]).sum()
    if dup != 0:
        raise ValueError(f"Found {dup} duplicate (mouse_id,timepoint,phase) rows.")

    wide = (
        df_long
        .set_index(ID_COLS + WITHIN_COLS)[metric_cols]
        .unstack(WITHIN_COLS)
    )

    # wide columns are a MultiIndex: (metric, timepoint, phase) due to unstack order.
    # Let's reorder to (phase, timepoint, metric) for naming.
    if not isinstance(wide.columns, pd.MultiIndex):
        raise ValueError("Unexpected: wide columns are not a MultiIndex.")

    # Determine current levels
    # Typically: level0=metric, level1=timepoint, level2=phase
    if wide.columns.nlevels != 3:
        raise ValueError(f"Unexpected MultiIndex levels: {wide.columns.nlevels}")

    wide = wide.reorder_levels([2, 1, 0], axis=1).sort_index(axis=1)

    # Flatten to stable names
    wide.columns = [
        f"{phase}__{timepoint}__{metric}"
        for (phase, timepoint, metric) in wide.columns.to_list()
    ]

    wide = wide.reset_index()

    # --- derived contrasts (append-only), built in one go to avoid fragmentation
    derived = {}

    def col(phase: str, timepoint: str, metric: str) -> str:
        return f"{phase}__{timepoint}__{metric}"

    # After - Before within phase
    for ph in phase_order:
        for m in metric_cols:
            c_after = col(ph, time_order[1], m)
            c_before = col(ph, time_order[0], m)
            out = f"delta_after_minus_before__{ph}__{m}"
            if c_after in wide.columns and c_before in wide.columns:
                derived[out] = wide[c_after] - wide[c_before]
            else:
                derived[out] = np.nan

    # hand - standing within timepoint
    ph_a, ph_b = phase_order
    for tp in time_order:
        for m in metric_cols:
            c_a = col(ph_a, tp, m)
            c_b = col(ph_b, tp, m)
            out = f"delta_{ph_a}_minus_{ph_b}__{tp}__{m}"
            if c_a in wide.columns and c_b in wide.columns:
                derived[out] = wide[c_a] - wide[c_b]
            else:
                derived[out] = np.nan

    # Difference-in-differences
    for m in metric_cols:
        d_a = f"delta_after_minus_before__{ph_a}__{m}"
        d_b = f"delta_after_minus_before__{ph_b}__{m}"
        out = f"did__{m}"
        # These are in derived dict, not wide yet
        if d_a in derived and d_b in derived and not np.isscalar(derived[d_a]) and not np.isscalar(derived[d_b]):
            derived[out] = derived[d_a] - derived[d_b]
        else:
            derived[out] = np.nan

    wide = pd.concat([wide, pd.DataFrame(derived)], axis=1)


    # hand - standing within timepoint (phase difference)
    ph_a, ph_b = phase_order
    for tp in time_order:
        for m in metric_cols:
            c_a = col(ph_a, tp, m)
            c_b = col(ph_b, tp, m)
            out = f"delta_{ph_a}_minus_{ph_b}__{tp}__{m}"
            if c_a in wide.columns and c_b in wide.columns:
                wide[out] = wide[c_a] - wide[c_b]
            else:
                wide[out] = np.nan

    # Difference-in-differences: (After-Before)_hand - (After-Before)_standing
    for m in metric_cols:
        d_a = f"delta_after_minus_before__{ph_a}__{m}"
        d_b = f"delta_after_minus_before__{ph_b}__{m}"
        out = f"did__{m}"
        if d_a in wide.columns and d_b in wide.columns:
            wide[out] = wide[d_a] - wide[d_b]
        else:
            wide[out] = np.nan

    return wide


def main() -> int:
    if not INPUT_LONG.exists():
        print(f"ERROR: missing input {INPUT_LONG}", file=sys.stderr)
        return 1

    df_long = pd.read_csv(INPUT_LONG)

    wide = derive_vit_wide(df_long)

    OUTPUT_WIDE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUTPUT_WIDE, index=False)

    # minimal report
    n_mice = wide["mouse_id"].nunique()
    print(f"Wrote {OUTPUT_WIDE} | rows={len(wide)} mice={n_mice} cols={wide.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
