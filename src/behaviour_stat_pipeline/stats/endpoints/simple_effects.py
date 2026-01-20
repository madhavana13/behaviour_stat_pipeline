from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class SimpleEffectsSpec:
    factor: str   # e.g., "group"
    within: str   # e.g., "sex"


def _quoted(name: str) -> str:
    # Patsy-safe quoting for arbitrary column names
    return f'Q("{name}")'


def _fit_two_way_model(
    d: pd.DataFrame,
    endpoint: str,
    f1: str,
    f2: str,
    include_interaction: bool,
) :
    y = _quoted(endpoint)
    f1q = _quoted(f1)
    f2q = _quoted(f2)

    if include_interaction:
        formula = f"{y} ~ C({f1q}) * C({f2q})"
    else:
        formula = f"{y} ~ C({f1q}) + C({f2q})"

    return smf.ols(formula, data=d, eval_env=0).fit()


def _design_row(model, levels: dict[str, Any]) -> np.ndarray:
    """
    Build the design row for a single combination of factor levels
    using the model's design_info (so columns match params).
    """
    design_info = model.model.data.design_info
    df = pd.DataFrame([levels])
    X = patsy.build_design_matrices([design_info], df, return_type="dataframe")[0]
    return X.iloc[0].to_numpy(dtype=float)


def run_simple_effects(
    df: pd.DataFrame,
    assay: str,
    endpoint: str,
    factors: list[str],
    include_interaction: bool,
    contrast_specs: list[SimpleEffectsSpec],
) -> pd.DataFrame:
    """
    Model-based simple effects using linear contrasts on the fitted OLS model.

    For each spec:
      - compare `factor` levels within each level of `within`
    Example (2x2):
      factor=group, within=sex -> cup vs tail at sex=male and sex=female
    """
    if len(factors) != 2:
        raise ValueError(f"simple effects require exactly 2 factors, got {factors}")

    f1, f2 = factors
    needed = [endpoint, f1, f2]
    d = df.dropna(subset=needed).copy()

    # Guard
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise KeyError(f"{assay}:{endpoint}: missing columns for simple effects: {missing}")

    # Need at least 2 levels in each factor
    if d[f1].nunique(dropna=False) < 2 or d[f2].nunique(dropna=False) < 2:
        return pd.DataFrame(
            [{
                "assay": assay,
                "endpoint": endpoint,
                "contrast": "SKIPPED",
                "reason": f"factor has <2 levels ({f1}={d[f1].nunique(dropna=False)}, {f2}={d[f2].nunique(dropna=False)})",
            }]
        )

    model = _fit_two_way_model(d, endpoint, f1, f2, include_interaction=include_interaction)

    params = model.params.to_numpy(dtype=float)
    cov = model.cov_params().to_numpy(dtype=float)
    df_resid = float(model.df_resid)

    rows: list[dict[str, Any]] = []

    for spec in contrast_specs:
        factor = spec.factor
        within = spec.within

        if set([factor, within]) != set([f1, f2]):
            raise ValueError(
                f"{assay}:{endpoint}: spec uses {factor}/{within} but factors are {factors}"
            )

        factor_levels = list(pd.Series(d[factor].unique()).dropna())
        within_levels = list(pd.Series(d[within].unique()).dropna())

        # we only support 2-level comparisons for now (as in your 2x2 designs)
        if len(factor_levels) != 2:
            raise ValueError(
                f"{assay}:{endpoint}: simple effects currently supports 2 levels in '{factor}', got {factor_levels}"
            )

        a, b = factor_levels[0], factor_levels[1]

        for w in within_levels:
            # build design rows for (factor=a, within=w) and (factor=b, within=w)
            levels_a = {factor: a, within: w}
            levels_b = {factor: b, within: w}

            xa = _design_row(model, levels_a)
            xb = _design_row(model, levels_b)
            c = xa - xb

            est = float(c @ params)
            se = float(np.sqrt(c @ cov @ c))
            t_stat = est / se if se > 0 else np.nan
            p = float(2 * st.t.sf(np.abs(t_stat), df_resid)) if np.isfinite(t_stat) else np.nan

            rows.append({
                "assay": assay,
                "endpoint": endpoint,
                "contrast": f"{factor}_within_{within}",
                "within_level": str(w),
                "level_a": str(a),
                "level_b": str(b),
                "estimate_a_minus_b": est,
                "se": se,
                "t": t_stat,
                "df_resid": df_resid,
                "p_value": p,
                "n": int(model.nobs),
                "model": "OLS",
                "anova_type": "type2",
                "include_interaction": bool(include_interaction),
            })

    return pd.DataFrame(rows)
