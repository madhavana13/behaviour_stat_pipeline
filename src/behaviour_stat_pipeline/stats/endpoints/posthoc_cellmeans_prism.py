"""
Prism-style post hoc comparisons on CELL MEANS for a 2×2 design (two factors, each with 2 levels).

What this does (per endpoint):
- Fits OLS with pooled variance (MSE) from the ANOVA model:
    y ~ C(f1) * C(f2)   (or additive if include_interaction=False)
- Computes selected pairwise comparisons of *cell means* using the pooled MSE:
    t = (mean_A - mean_B) / sqrt(MSE * (1/nA + 1/nB))
    df = df_resid from the OLS model
- Applies Sidak adjustment across the selected pairs (k = number of selected pairs)
- Adds Prism-like cosmetic columns:
    contrast_label, diff_prism_direction, significance_label

Important:
- This module is PURE: it does not read cfg, does not do I/O.
- Caller (run_stats.py) passes df, endpoints, factors, selected_pairs, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


def _q(name: str) -> str:
    """Patsy-safe quoting for column names with spaces/symbols."""
    return f'Q("{name}")'


def _significance_label(p: float) -> str:
    """Prism-like threshold labels based on adjusted p-value."""
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "<0.0001"
    elif p < 1e-3:
        return "<0.001"
    elif p < 1e-2:
        return "<0.01"
    elif p < 5e-2:
        return "<0.05"
    else:
        return "ns"


def _infer_two_levels(
    s: pd.Series, *, order: Optional[List[str]] = None
) -> List[str]:
    """
    Infer exactly two levels from the data, deterministically.
    If order is provided, respects it (filtered to observed).
    """
    observed = [str(x) for x in s.dropna().astype(str).unique().tolist()]
    obs_set = set(observed)

    if order:
        levels = [str(x) for x in order if str(x) in obs_set]
    else:
        levels = sorted(obs_set)

    if len(levels) != 2:
        raise ValueError(f"Expected exactly 2 observed levels, got {levels}")
    return levels


def _decode_cell(
    cell_spec: Dict[str, int],
    factors: List[str],
    levels_by_factor: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Convert a cell spec like {group:0, sex:1} into actual levels using levels_by_factor.
    """
    out: Dict[str, str] = {}
    for f in factors:
        if f not in cell_spec:
            raise KeyError(f"Cell spec missing '{f}': {cell_spec}")
        idx = int(cell_spec[f])
        levels = levels_by_factor[f]
        if idx < 0 or idx >= len(levels):
            raise IndexError(f"Index {idx} out of range for factor '{f}' levels {levels}")
        out[f] = str(levels[idx])
    return out


def _cell_mask(d: pd.DataFrame, cell: Dict[str, str], factors: List[str]) -> np.ndarray:
    mask = np.ones(len(d), dtype=bool)
    for f in factors:
        mask &= (d[f].astype(str) == str(cell[f]))
    return mask


def run_posthoc_cellmeans_prism(
    df: pd.DataFrame,
    *,
    assay: str,
    endpoints: List[str],
    factors: List[str],
    include_interaction: bool,
    factor_orders: Optional[Dict[str, List[str]]] = None,
    selected_pairs: Optional[List[Dict[str, Any]]] = None,
    correction: str = "sidak",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame
        Must contain the endpoint columns and factor columns.
    assay : str
        For reporting.
    endpoints : list[str]
        Endpoint column names to analyze.
    factors : list[str]
        Exactly two factor column names, e.g. ["group", "sex"].
    include_interaction : bool
        If True: fit y ~ C(f1)*C(f2). If False: fit additive model.
    factor_orders : dict[str, list[str]] | None
        Optional explicit ordering of levels per factor.
        Used to map {0,1} indices in selected_pairs to actual levels.
        If absent/empty, levels are inferred deterministically (sorted).
    selected_pairs : list[dict] | None
        Each item:
          {"name": "...", "a": {"group":0, "sex":0}, "b": {"group":1, "sex":0}}
    correction : str
        Currently supports only "sidak" (pure Sidak across K selected pairs).

    Returns
    -------
    DataFrame with one row per endpoint × selected pair.
    """
    if selected_pairs is None:
        selected_pairs = []
    if not selected_pairs:
        raise ValueError(f"{assay}: selected_pairs is empty; nothing to compute.")

    if len(factors) != 2:
        raise ValueError(f"{assay}: expected exactly 2 factors, got {factors}")

    correction = str(correction).lower().strip()
    if correction != "sidak":
        raise ValueError(f"{assay}: only 'sidak' correction implemented (got {correction})")

    factor_orders = factor_orders or {}

    f1, f2 = factors

    # Infer/lock factor levels (must be exactly 2 each)
    levels_by_factor: Dict[str, List[str]] = {
        f1: _infer_two_levels(df[f1], order=factor_orders.get(f1)),
        f2: _infer_two_levels(df[f2], order=factor_orders.get(f2)),
    }

    # k for Sidak = number of requested comparisons (Prism-like in your current use)
    k = len(selected_pairs)

    rows: List[Dict[str, Any]] = []

    for endpoint in endpoints:
        needed = [endpoint, f1, f2]
        d = df.dropna(subset=needed).copy()

        # If a factor collapses to <2 levels after dropna, skip the entire endpoint
        if d[f1].astype(str).nunique(dropna=False) < 2 or d[f2].astype(str).nunique(dropna=False) < 2:
            rows.append(
                {
                    "assay": assay,
                    "endpoint": endpoint,
                    "pair_name": "SKIPPED",
                    "contrast_label": "",
                    "diff_prism_direction": np.nan,
                    "cell_a": "",
                    "cell_b": "",
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "n_a": 0,
                    "n_b": 0,
                    "diff_a_minus_b": np.nan,
                    "mse": np.nan,
                    "se_pooled": np.nan,
                    "t": np.nan,
                    "df_error": np.nan,
                    "p_value": np.nan,
                    "p_value_sidak": np.nan,
                    "significance_label": "",
                    "reason": f"factor has <2 levels after dropna on {needed}",
                }
            )
            continue

        # Fit pooled-variance ANOVA model via OLS
        if include_interaction:
            formula = f"{_q(endpoint)} ~ C({_q(f1)}) * C({_q(f2)})"
        else:
            formula = f"{_q(endpoint)} ~ C({_q(f1)}) + C({_q(f2)})"

        model = smf.ols(formula, data=d, eval_env=0).fit()
        mse = float(model.mse_resid)
        df_error = float(model.df_resid)

        for pair in selected_pairs:
            pair_name = str(pair.get("name", ""))

            cell_a = _decode_cell(pair["a"], factors, levels_by_factor)
            cell_b = _decode_cell(pair["b"], factors, levels_by_factor)

            mask_a = _cell_mask(d, cell_a, factors)
            mask_b = _cell_mask(d, cell_b, factors)

            ya = d.loc[mask_a, endpoint].astype(float)
            yb = d.loc[mask_b, endpoint].astype(float)

            n_a = int(ya.shape[0])
            n_b = int(yb.shape[0])

            mean_a = float(ya.mean()) if n_a > 0 else np.nan
            mean_b = float(yb.mean()) if n_b > 0 else np.nan
            diff = (mean_a - mean_b) if (n_a > 0 and n_b > 0) else np.nan

            se = float(np.sqrt(mse * (1.0 / n_a + 1.0 / n_b))) if (n_a > 0 and n_b > 0) else np.nan
            t_stat = float(diff / se) if (np.isfinite(se) and se > 0) else np.nan

            # Two-tailed p-value
            p = float(2.0 * st.t.sf(np.abs(t_stat), df_error)) if np.isfinite(t_stat) else np.nan

            # Pure Sidak across k selected comparisons
            p_sidak = float(1.0 - (1.0 - p) ** k) if np.isfinite(p) else np.nan

            # --- Prism-style presentation (cosmetic only) ---
            def _level_idx(factor: str, level: str) -> int:
                return levels_by_factor[factor].index(str(level))

            score_a = sum(_level_idx(f, cell_a[f]) for f in factors)
            score_b = sum(_level_idx(f, cell_b[f]) for f in factors)

            # Put "later levels" first: higher score vs lower score
            if score_a >= score_b:
                cell_1, cell_2 = cell_a, cell_b
                mean_1, mean_2 = mean_a, mean_b
            else:
                cell_1, cell_2 = cell_b, cell_a
                mean_1, mean_2 = mean_b, mean_a

            diff_prism = (mean_1 - mean_2) if (np.isfinite(mean_1) and np.isfinite(mean_2)) else np.nan

            def _pretty(cell: Dict[str, str]) -> str:
                return ":".join(str(cell[f]).capitalize() for f in factors)

            contrast_label = f"{_pretty(cell_1)} vs {_pretty(cell_2)}"
            sig_label = _significance_label(p_sidak)

            rows.append(
                {
                    "assay": assay,
                    "endpoint": endpoint,
                    "pair_name": pair_name,
                    "contrast_label": contrast_label,
                    "diff_prism_direction": diff_prism,
                    "cell_a": ";".join([f"{f}={cell_a[f]}" for f in factors]),
                    "cell_b": ";".join([f"{f}={cell_b[f]}" for f in factors]),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "diff_a_minus_b": diff,
                    "mse": mse,
                    "se_pooled": se,
                    "t": t_stat,
                    "df_error": df_error,
                    "p_value": p,
                    "p_value_sidak": p_sidak,
                    "significance_label": sig_label,
                }
            )

    return pd.DataFrame(rows)
