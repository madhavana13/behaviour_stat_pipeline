from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _canonical_assay_key(assay: str | None, *, casefold: bool = True) -> str | None:
    if assay is None:
        return None
    assay = str(assay).strip()
    return assay.lower() if casefold else assay


def load_schema_aliases(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema aliases file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Schema aliases file must parse to a dict: {path}")
    return data


def normalize_headers(
    df: pd.DataFrame,
    *,
    assay: str | None,
    schema: dict[str, Any],
) -> tuple[pd.DataFrame, list[tuple[str, str]], list[str]]:
    """
    Normalize raw vendor/export headers to canonical internal names.

    Returns
    -------
    df_norm : DataFrame
        Copy with normalized headers.
    rename_log : list[tuple[str, str]]
        Pairs of (old_name, new_name) for applied renames.
    suspicious : list[str]
        Suspicious columns retained unchanged, e.g. blank/Unnamed/H1.
    """
    out = df.copy()

    global_cfg = schema.get("global", {}) or {}
    normalize_whitespace = bool(global_cfg.get("normalize_whitespace", True))
    casefold_assay_keys = bool(global_cfg.get("casefold_assay_keys", True))

    assay_key = _canonical_assay_key(assay, casefold=casefold_assay_keys)
    assay_map = {}
    if assay_key is not None:
        assay_cfg = schema.get(assay_key, {}) or {}
        assay_map = assay_cfg.get("raw", {}) or {}

    rename_log: list[tuple[str, str]] = []
    suspicious: list[str] = []
    new_cols: list[str] = []

    for col in out.columns:
        old = str(col)

        cleaned = old.strip()
        if normalize_whitespace:
            cleaned = re.sub(r"\s+", " ", cleaned)

        if cleaned == "" or cleaned.startswith("Unnamed") or re.fullmatch(r"H\d+", cleaned):
            suspicious.append(cleaned if cleaned else "<blank>")
            new_cols.append(cleaned if cleaned else old)
            continue

        new = assay_map.get(cleaned, cleaned)

        if new != old:
            rename_log.append((old, new))

        new_cols.append(new)

    duplicates = sorted({c for c in new_cols if new_cols.count(c) > 1})
    if duplicates:
        raise ValueError(
            f"[Schema] Duplicate columns after normalization for assay={assay!r}: {duplicates}"
        )

    out.columns = new_cols
    return out, rename_log, suspicious