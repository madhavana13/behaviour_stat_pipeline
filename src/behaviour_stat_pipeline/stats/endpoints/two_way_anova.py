from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_two_way_anova(
    df: pd.DataFrame,
    assay: str,
    endpoint: str,
    factors: list[str],
    include_interaction: bool = True,
    typ: int = 3,
) -> pd.DataFrame:
    """
    Tidy two-way ANOVA table (Type III by default) for:
        endpoint ~ C(f1) * C(f2)

    Robust to arbitrary column names (spaces, punctuation) via patsy Q("...").
    """
    if len(factors) != 2:
        raise ValueError(f"two-way ANOVA requires exactly 2 factors, got {factors}")

    f1, f2 = factors

    # Drop rows missing outcome or factor values
    needed = [endpoint, f1, f2]
    d = df.dropna(subset=needed).copy()

    # Guard: ensure required columns exist
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise KeyError(f"{assay}:{endpoint}: missing columns for two-way ANOVA: {missing}")

    # Guard: factors must have >= 2 levels to estimate effects
    if d[f1].nunique(dropna=False) < 2 or d[f2].nunique(dropna=False) < 2:
        raise ValueError(
            f"{assay}:{endpoint}: cannot run two-way ANOVA because one factor has <2 levels "
            f"(levels: {f1}={d[f1].nunique(dropna=False)}, {f2}={d[f2].nunique(dropna=False)})."
        )

    # Patsy-safe quoting for arbitrary column names
    y = f'Q("{endpoint}")'
    f1q = f'Q("{f1}")'
    f2q = f'Q("{f2}")'

    if include_interaction:
        formula = f"{y} ~ C({f1q}) * C({f2q})"
    else:
        formula = f"{y} ~ C({f1q}) + C({f2q})"

    # Fit OLS and compute ANOVA
    model = smf.ols(formula, data=d, eval_env=0).fit()
    aov = sm.stats.anova_lm(model, typ=typ).reset_index().rename(columns={"index": "term"})

    # Attach metadata
    aov["assay"] = assay
    aov["endpoint"] = endpoint
    aov["n"] = int(model.nobs)
    aov["typ"] = int(typ)
    aov["include_interaction"] = bool(include_interaction)

    # Standardize columns
    aov = aov.rename(columns={"PR(>F)": "p_value"})
    keep = [c for c in ["assay", "endpoint", "term", "sum_sq", "df", "F", "p_value", "n", "typ", "include_interaction"] if c in aov.columns]
    return aov[keep].reset_index(drop=True)
