# src/behaviour_stat_pipeline/common/qc.py

from __future__ import annotations

from typing import Sequence
import pandas as pd


def assert_unique_key(df: pd.DataFrame, key: Sequence[str], *, where: str = "") -> None:
    """
    Assert that `key` uniquely identifies rows in `df`.

    Raises:
        ValueError with examples of duplicated keys.
    """
    missing = [c for c in key if c not in df.columns]
    if missing:
        raise ValueError(f"Missing key columns {missing} for uniqueness check{_fmt_where(where)}")

    dup_mask = df.duplicated(subset=list(key), keep=False)
    if dup_mask.any():
        examples = df.loc[dup_mask, list(key)].head(20)
        raise ValueError(
            f"Duplicate rows detected for key {list(key)}{_fmt_where(where)}.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )


def _fmt_where(where: str) -> str:
    return f" [{where}]" if where else ""

def warn_missing_subjects_coverage(
    observed_mouse_ids: pd.Series,
    subjects: pd.DataFrame,
    *,
    where: str = "",
) -> None:
    subj_ids = set(subjects["mouse_id"].astype(str))
    obs_ids = set(observed_mouse_ids.dropna().astype(str))
    missing = sorted(subj_ids - obs_ids)
    if missing:
        tag = f" [{where}]" if where else ""
        print(
            f"WARNING{tag}: {len(missing)} subject mouse_ids not present in this run's data: {missing}"
        )
