from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
    primary_one_sample_enabled: bool
    primary_one_sample_group_col: str
    primary_one_sample_collapse_over: list[str]
    primary_one_sample_correction: str
    primary_one_sample_run_ttest: bool
    primary_one_sample_run_wilcoxon: bool
    primary_one_sample_nulls: dict[str, dict[str, float]]
    diagnostics_enabled: bool
    diagnostics_groupings: list[list[str]]
    diagnostics_endpoints: dict[str, list[str]]
    diagnostics_min_n_test: int
    diagnostics_var_ratio_thresh: float
    diagnostics_mad_z_thresh: float
    secondary_two_way_anova_enabled: bool
    secondary_two_way_anova_factors: list[str]
    secondary_two_way_anova_include_interaction: bool
    secondary_two_way_anova_typ: int
    secondary_two_way_anova_endpoints: dict[str, list[str]]
    secondary_two_way_anova_correction: str
    secondary_two_way_anova_posthoc: dict
    secondary_two_way_anova_factor_orders: dict[str, list[str]]
    correlations_enabled: bool
    correlations_correction: str
    correlations_min_n: int
    correlations_specs: dict


    def to_dict(self) -> Dict[str, Any]:
        """
        Canonical conversion for legacy code paths that still expect a dict config.

        IMPORTANT:
        - New code should prefer typed attribute access.
        - This method exists only to support transitional callers (e.g. scripts).
        """
        return asdict(self)

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
    diag = cfg.get("diagnostics", {})
    diagnostics_enabled = bool(diag.get("enabled", False))
    diagnostics_groupings = list(diag.get("groupings", []))
    diagnostics_endpoints = dict(diag.get("endpoints", {}))
    diagnostics_min_n_test = int(diag.get("min_n_test", 3))
    diagnostics_var_ratio_thresh = float(diag.get("var_ratio_thresh", 4.0))
    diagnostics_mad_z_thresh = float(diag.get("mad_z_thresh", 3.5))
    pos = cfg.get("primary_one_sample", {})
    primary_one_sample_enabled = bool(pos.get("enabled", False))
    primary_one_sample_group_col = str(pos.get("group_col", "group"))
    primary_one_sample_collapse_over = list(pos.get("collapse_over", []))
    primary_one_sample_correction = str(pos.get("correction", "holm"))
    primary_one_sample_run_ttest = bool(pos.get("run_ttest", True))
    primary_one_sample_run_wilcoxon = bool(pos.get("run_wilcoxon", True))
    primary_one_sample_nulls = dict(pos.get("nulls", {}))
    sec = cfg.get("secondary_two_way_anova", {})
    secondary_two_way_anova_enabled = bool(sec.get("enabled", False))
    secondary_two_way_anova_factors = list(sec.get("factors", ["group", "sex"]))
    secondary_two_way_anova_include_interaction = bool(sec.get("include_interaction", True))
    secondary_two_way_anova_typ = int(sec.get("type", 3))
    secondary_two_way_anova_endpoints = dict(sec.get("endpoints", {}))
    secondary_two_way_anova_correction = str(sec.get("correction_across_endpoints", "none"))
    secondary_two_way_anova_posthoc = dict(sec.get("posthoc", {}))
    secondary_two_way_anova_factor_orders = dict(sec.get("factor_orders", {}))    

    design = cfg["design"]
    between_factors = list(design.get("between_factors", []))
    within_factors = dict(design.get("within_factors", {}))

    optional = cfg.get("optional", {})
    blocking_factors = list(optional.get("blocking_factors", []))
    cluster_col = optional.get("cluster_col", mouse_id_col)

    assays = list(cfg.get("assays", []))
    corr = cfg.get("correlations", {})
    correlations_enabled = bool(corr.get("enabled", False))
    correlations_correction = str(corr.get("correction", "fdr_bh"))
    correlations_min_n = int(corr.get("min_n", 5))
    correlations_specs = dict(corr.get("specs", {}))

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
        diagnostics_enabled=diagnostics_enabled,
        diagnostics_groupings=diagnostics_groupings,
        diagnostics_endpoints=diagnostics_endpoints,
        diagnostics_min_n_test=diagnostics_min_n_test,
        diagnostics_var_ratio_thresh=diagnostics_var_ratio_thresh,
        diagnostics_mad_z_thresh=diagnostics_mad_z_thresh,
        primary_one_sample_enabled=primary_one_sample_enabled,
        primary_one_sample_group_col=primary_one_sample_group_col,
        primary_one_sample_collapse_over=primary_one_sample_collapse_over,
        primary_one_sample_correction=primary_one_sample_correction,
        primary_one_sample_run_ttest=primary_one_sample_run_ttest,
        primary_one_sample_run_wilcoxon=primary_one_sample_run_wilcoxon,
        primary_one_sample_nulls=primary_one_sample_nulls,
        secondary_two_way_anova_enabled=secondary_two_way_anova_enabled,
        secondary_two_way_anova_factors=secondary_two_way_anova_factors,
        secondary_two_way_anova_include_interaction=secondary_two_way_anova_include_interaction,
        secondary_two_way_anova_typ=secondary_two_way_anova_typ,
        secondary_two_way_anova_endpoints=secondary_two_way_anova_endpoints,
        secondary_two_way_anova_correction=secondary_two_way_anova_correction,
        secondary_two_way_anova_posthoc=secondary_two_way_anova_posthoc,
        secondary_two_way_anova_factor_orders=secondary_two_way_anova_factor_orders,
        correlations_enabled=correlations_enabled,
        correlations_correction=correlations_correction,
        correlations_min_n=correlations_min_n,
        correlations_specs=correlations_specs
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
        df.groupby(factors, dropna=False, observed=False)
        .size()
        .reset_index(name="n")
        .sort_values(by=factors)
        .reset_index(drop=True)
    )
    return out
