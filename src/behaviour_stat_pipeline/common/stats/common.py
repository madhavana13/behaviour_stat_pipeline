from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class StatsConfig:
    mouse_id_col: str
    processed_dir: Path
    subjects_path: Path
    out_tables: Path
    out_figures: Path
    out_logs: Path
    between_factors: List[str]
    within_factors: Dict[str, List[str]]
    blocking_factors: List[str]
    cluster_col: str
    assays: List[str]


def load_stats_config(path: str | Path) -> StatsConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mouse_id_col = cfg["id_cols"]["mouse"]

    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"])
    subjects_path = Path(paths["metadata_subjects"])
    out_tables = Path(paths["out_tables"])
    out_figures = Path(paths["out_figures"])
    out_logs = Path(paths["out_logs"])

    design = cfg["design"]
    between_factors = list(design.get("between_factors", []))
    within_factors = dict(design.get("within_factors", {}))

    optional = cfg.get("optional", {})
    blocking_factors = list(optional.get("blocking_factors", []))
    cluster_col = optional.get("cluster_col", mouse_id_col)

    assays = list(cfg.get("assays", []))

    return StatsConfig(
        mouse_id_col=mouse_id_col,
        processed_dir=processed_dir,
        subjects_path=subjects_path,
        out_tables=out_tables,
        out_figures=out_figures,
        out_logs=out_logs,
        between_factors=between_factors,
        within_factors=within_factors,
        blocking_factors=blocking_factors,
        cluster_col=cluster_col,
        assays=assays,
    )


def read_processed(processed_dir: Path, assay: str) -> pd.DataFrame:
    # Your files are named e.g., NOR_processed.csv, VIT_processed.csv
    fname = f"{assay.upper()}_processed.csv"
    path = processed_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)


def read_subjects(subjects_path: Path) -> pd.DataFrame:
    if not subjects_path.exists():
        raise FileNotFoundError(f"Subjects metadata not found: {subjects_path}")
    return pd.read_csv(subjects_path)


def merge_metadata(
    df: pd.DataFrame,
    subjects: pd.DataFrame,
    mouse_id_col: str,
) -> pd.DataFrame:
    if mouse_id_col not in df.columns:
        raise KeyError(f"Processed data missing '{mouse_id_col}' column")

    if mouse_id_col not in subjects.columns:
        raise KeyError(f"Subjects metadata missing '{mouse_id_col}' column")

    merged = df.merge(subjects, on=mouse_id_col, how="left", validate="m:1")

    missing = merged[mouse_id_col][merged.isna().any(axis=1) & merged[mouse_id_col].notna()]
    # Above is conservative; better: check any missing in key metadata cols later.

    return merged


def require_columns(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns: {missing}")


def coerce_categoricals(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        # Keep NaNs as NaNs; categories handle this fine
        df[c] = df[c].astype("category")
    return df


def drop_single_level_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """
    Returns:
      kept_factors, dropped_info where dropped_info is [(col, levels_observed)]
    """
    kept: List[str] = []
    dropped: List[Tuple[str, List[str]]] = []
    for c in factor_cols:
        if c not in df.columns:
            continue
        observed = [x for x in df[c].dropna().unique().tolist()]
        if len(observed) <= 1:
            dropped.append((c, [str(x) for x in observed]))
        else:
            kept.append(c)
    return kept, dropped


def cell_counts(df: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame({"n": [len(df)]})
    # counts per unique combination of factor levels
    out = (
        df.groupby(factors, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(by=factors)
        .reset_index(drop=True)
    )
    return out
