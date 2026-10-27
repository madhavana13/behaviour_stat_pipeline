"""
Prism-style post hoc comparisons on CELL MEANS for one- or two-factor designs
(any number of levels per factor).

What this does (per endpoint):
- Fits OLS with pooled variance (MSE) from the ANOVA model:
    y ~ C(f1)                       (one factor)
    y ~ C(f1) * C(f2)               (or additive if include_interaction=False)
- Computes pairwise comparisons of *cell means* using the pooled MSE:
    t = (mean_A - mean_B) / sqrt(MSE * (1/nA + 1/nB))
    df = df_resid from the OLS model
- Applies Sidak adjustment across the comparisons (k = number of comparisons)
- Adds Prism-like cosmetic columns:
    contrast_label, diff_prism_direction, significance_label

Which comparisons:
- selected_pairs: explicit cells, each factor given by level name ({group: ifs, sex: male})
  or by position in the level order ({group: 1, sex: 1}).
- comparisons: generated, e.g. {factor: group, by: [sex]} = every pair of group levels
  within each sex. `by` must list all other factors (comparisons are of cells).

Important:
- This module is PURE: it does not read cfg, does not do I/O.
- Caller (run_stats.py) passes df, endpoints, factors, selected_pairs, etc.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Any, Dict, List, Optional

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


def _infer_levels(s: pd.Series, *, order: Optional[List[str]] = None) -> List[str]:
    """
    Observed levels (at least two), deterministically ordered.
    If order is provided, its levels come first (filtered to observed), then the rest sorted.
    """
    observed = [str(x) for x in s.dropna().astype(str).unique().tolist()]
    preferred = [str(x) for x in (order or []) if str(x) in set(observed)]
    levels = preferred + sorted(x for x in observed if x not in preferred)
    if len(levels) < 2:
        raise ValueError(f"Expected at least 2 observed levels, got {levels}")
    return levels


def _decode_cell(
    cell_spec: Dict[str, Any],
    factors: List[str],
    levels_by_factor: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Convert a cell spec into actual levels. Each factor is given either by level name
    ({group: "ifs"}) or by position in the level order ({group: 1}).
    """
    out: Dict[str, str] = {}
    for f in factors:
        if f not in cell_spec:
            raise KeyError(f"Cell spec missing '{f}': {cell_spec}")
        value = cell_spec[f]
        levels = levels_by_factor[f]
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            idx = int(value)
            if idx < 0 or idx >= len(levels):
                raise IndexError(f"Index {idx} out of range for factor '{f}' levels {levels}")
            out[f] = str(levels[idx])
        else:
            if str(value) not in levels:
                raise ValueError(f"Level {value!r} of factor '{f}' not observed; levels are {levels}")
            out[f] = str(value)
    return out


def expand_comparisons(
    comparisons: List[Dict[str, Any]],
    factors: List[str],
    levels_by_factor: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    {factor: group, by: [sex]} -> one pair per combination of group levels within each sex.
    Returns selected_pairs-style dicts with level names.
    """
    pairs: List[Dict[str, Any]] = []
    for comp in comparisons:
        factor = comp["factor"]
        by = list(comp.get("by", []) or [])
        others = [f for f in factors if f != factor]
        if factor not in factors or sorted(by) != sorted(others):
            raise ValueError(
                f"comparison {comp}: factor must be one of {factors} and `by` must list all other "
                f"factors ({others}); comparisons are between cells"
            )
        for by_levels in product(*[levels_by_factor[b] for b in by]) if by else [()]:
            fixed = dict(zip(by, by_levels))
            for a, b in combinations(levels_by_factor[factor], 2):
                suffix = "" if not by else "|" + ",".join(f"{k}={v}" for k, v in fixed.items())
                pairs.append({
                    "name": f"{factor}:{a}_vs_{b}{suffix}",
                    "a": {**fixed, factor: a},
                    "b": {**fixed, factor: b},
                })
    return pairs


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
    comparisons: Optional[List[Dict[str, Any]]] = None,
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
        One or two factor column names, e.g. ["group", "sex"] or ["group"].
    include_interaction : bool
        If True: fit y ~ C(f1)*C(f2). If False: fit additive model.
    factor_orders : dict[str, list[str]] | None
        Optional explicit ordering of levels per factor (positions in selected_pairs refer to it).
        If absent/empty, levels are inferred deterministically (sorted).
    selected_pairs : list[dict] | None
        Each item: {"name": "...", "a": {"group": "control", "sex": 0}, "b": {...}}
    comparisons : list[dict] | None
        Generated pairs, e.g. [{"factor": "group", "by": ["sex"]}].
    correction : str
        Currently supports only "sidak" (pure Sidak across all K comparisons).

    Returns
    -------
    DataFrame with one row per endpoint × comparison.
    """
    if len(factors) not in (1, 2):
        raise ValueError(f"{assay}: expected one or two factors, got {factors}")

    correction = str(correction).lower().strip()
    if correction != "sidak":
        raise ValueError(f"{assay}: only 'sidak' correction implemented (got {correction})")

    factor_orders = factor_orders or {}

    # Infer/lock factor levels (at least 2 each)
    levels_by_factor: Dict[str, List[str]] = {
        f: _infer_levels(df[f], order=factor_orders.get(f)) for f in factors
    }

    pairs = list(selected_pairs or []) + expand_comparisons(list(comparisons or []), factors, levels_by_factor)
    if not pairs:
        raise ValueError(f"{assay}: no selected_pairs or comparisons; nothing to compute.")

    # k for Sidak = number of requested comparisons (Prism-like in your current use)
    k = len(pairs)

    rows: List[Dict[str, Any]] = []

    for endpoint in endpoints:
        needed = [endpoint, *factors]
        d = df.dropna(subset=needed).copy()

        # If a factor collapses to <2 levels after dropna, skip the entire endpoint
        if min(d[f].astype(str).nunique(dropna=False) for f in factors) < 2:
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
        terms = [f"C({_q(f)})" for f in factors]
        formula = f"{_q(endpoint)} ~ " + (" * " if include_interaction else " + ").join(terms)

        model = smf.ols(formula, data=d, eval_env=0).fit()
        mse = float(model.mse_resid)
        df_error = float(model.df_resid)

        for pair in pairs:
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

            # Pure Sidak across k comparisons
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
