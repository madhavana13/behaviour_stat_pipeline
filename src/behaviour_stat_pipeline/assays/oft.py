"""
OFT (Open Field Test) assay helpers.

Also used for the sucrose splash test (SPL), which is ingested with the same
parameters as OFT (EthoVision summary + classifier report, one session per mouse);
`label` only changes the assay name in messages.

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

def _coalesce_duplicate_rows_oft(df: pd.DataFrame, label: str = "OFT") -> pd.DataFrame:
    """
    Coalesce duplicate OFT rows at mouse_id × timepoint when they are
    complementary partial records.

    Rules, column-wise within each duplicated key:
    - one non-missing + one missing -> keep the non-missing value
    - both missing -> keep missing
    - both non-missing and equal -> keep the value
    - both non-missing and different -> raise

    Provenance columns are combined conservatively.
    """
    key = ["mouse_id", "timepoint"]
    provenance_cols = {"raw_id", "source_file", "assay"}

    dup_mask = df.duplicated(subset=key, keep=False)
    if not dup_mask.any():
        return df

    resolved_rows: list[pd.Series] = []
    conflicting_keys: list[tuple] = []

    for key_vals, block in df.groupby(key, dropna=False, sort=False):
        if len(block) == 1:
            resolved_rows.append(block.iloc[0].copy())
            continue

        resolved = block.iloc[0].copy()

        compare_cols = [c for c in block.columns if c not in set(key) | provenance_cols]

        block_conflict = False
        for col in compare_cols:
            vals = block[col]
            non_na = vals.dropna()

            if non_na.empty:
                resolved[col] = pd.NA
                continue

            unique_non_na = pd.unique(non_na)

            if len(unique_non_na) == 1:
                resolved[col] = unique_non_na[0]
            else:
                block_conflict = True
                break

        if block_conflict:
            conflicting_keys.append(key_vals)
            continue

        if isinstance(key_vals, tuple):
            for k, v in zip(key, key_vals):
                resolved[k] = v
        else:
            resolved[key[0]] = key_vals

        for col in provenance_cols.intersection(block.columns):
            non_na = block[col].dropna().astype(str).unique().tolist()
            if not non_na:
                resolved[col] = pd.NA
            elif col == "source_file":
                resolved[col] = " | ".join(non_na)
            else:
                resolved[col] = non_na[0]

        resolved_rows.append(resolved)

    if conflicting_keys:
        raise ValueError(
            f"{label} QC failed: duplicate rows with genuinely conflicting non-missing values "
            f"detected for keys {conflicting_keys[:10]}"
        )

    print(
        f"WARNING [{label}]: complementary duplicate rows detected. "
        "Rows were coalesced at mouse_id × timepoint."
    )
    out = pd.DataFrame(resolved_rows).reset_index(drop=True)
    return out


def qc_oft(df: pd.DataFrame, label: str = "OFT") -> pd.DataFrame:
    """
    QC for OFT ingestion output.

    Assumes ingestion has already produced session-level rows (summary per mouse).
    Enforces uniqueness at mouse_id × timepoint.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{label} QC failed: missing required columns: {sorted(missing)}")

    for col in ["mouse_id", "group", "sex", "timepoint"]:
        if df[col].isna().any():
            raise ValueError(f"{label} QC failed: NA values found in column '{col}'")

    # NEW: resolve safe duplicates first
    df = _coalesce_duplicate_rows_oft(df, label=label)

    # enforce uniqueness AFTER resolution
    assert_unique_key(df, ["mouse_id", "timepoint"], where=f"{label} long/session-level")

    return df


def derive_oft_metrics(df: pd.DataFrame, cfg: OFTConfig | None = None, label: str = "OFT") -> pd.DataFrame:
    """
    Add derived metrics for OFT.

    For now:
    - does not drop rows
    - returns a copy
    - adds only cross-assay flags

    Later we can add canonical OFT outcomes and/or change scores explicitly.
    """
    cfg = cfg or OFTConfig()

    df = qc_oft(df, label=label)
    out = df.copy()
    out["is_after"] = (out["timepoint"].astype(str).str.lower() == "after").astype(int)
    return out


PRIMARY_OUTCOMES: list[str] = []
