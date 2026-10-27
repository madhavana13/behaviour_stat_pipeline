from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


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

def run_two_way_anova(
    df: pd.DataFrame,
    assay: str,
    endpoint: str,
    factors: list[str],
    include_interaction: bool = True,
    typ: int = 3,
) -> pd.DataFrame:
    """
    Tidy one- or two-way ANOVA table (Type III by default) for:
        endpoint ~ C(f1)              (one factor, e.g. a single-sex study)
        endpoint ~ C(f1) * C(f2)      (two factors)
    Each factor may have any number of levels (>= 2).
    Type III uses sum-to-zero (effect) coding, matching Prism/SPSS/car::Anova.

    Robust to arbitrary column names (spaces, punctuation) via patsy Q("...").
    """
    if len(factors) not in (1, 2):
        raise ValueError(f"ANOVA supports one or two factors, got {factors}")

    # Drop rows missing outcome or factor values
    needed = [endpoint, *factors]

    # Guard: ensure required columns exist
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"{assay}:{endpoint}: missing columns for ANOVA: {missing}")
    d = df.dropna(subset=needed).copy()

    # Guard: factors must have >= 2 levels to estimate effects
    n_levels = {f: d[f].nunique(dropna=False) for f in factors}
    if min(n_levels.values()) < 2:
        raise ValueError(
            f"{assay}:{endpoint}: cannot run ANOVA because a factor has <2 levels ({n_levels}). "
            f"For a design with one level of a factor (e.g. one sex), set "
            f"secondary_two_way_anova.factors to the remaining factor in the experiment's analysis plan."
        )

    # Patsy-safe quoting for arbitrary column names
    y = f'Q("{endpoint}")'

    # Type III SS are only meaningful with sum-to-zero contrasts; the default
    # Treatment coding tests "simple" effects at the reference level instead.
    coding = ", Sum" if typ == 3 else ""
    terms = [f'C(Q("{f}"){coding})' for f in factors]
    formula = f"{y} ~ " + (" * " if include_interaction else " + ").join(terms)

    # Fit OLS and compute ANOVA
    model = smf.ols(formula, data=d, eval_env=0).fit()
    aov = sm.stats.anova_lm(model, typ=typ).reset_index().rename(columns={"index": "term"})
    # Keep term labels independent of coding (figures parse them)
    aov["term"] = aov["term"].str.replace(", Sum)", ")", regex=False)

    # Attach metadata
    aov["assay"] = assay
    aov["endpoint"] = endpoint
    aov["n"] = int(model.nobs)
    aov["typ"] = int(typ)
    aov["include_interaction"] = bool(include_interaction and len(factors) == 2)

    # Standardize columns
    aov = aov.rename(columns={"PR(>F)": "p_value"})
    aov["significance_label"] = aov["p_value"].apply(_significance_label)
    keep = [c for c in ["assay", "endpoint", "term", "sum_sq", "df", "F", "p_value", "n", "typ", "include_interaction"] if c in aov.columns]
    return aov[keep].reset_index(drop=True)
