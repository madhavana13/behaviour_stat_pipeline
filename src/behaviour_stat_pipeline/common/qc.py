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
