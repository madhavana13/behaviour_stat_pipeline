from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def holm_adjust(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    mask = p.notna()
    adj = pd.Series(np.nan, index=p.index, dtype=float)
    if mask.sum() > 0:
        adj_vals = multipletests(p[mask].to_numpy(), method="holm")[1]
        adj.loc[mask] = adj_vals
    return adj

def adjust_pvalues(pvals: pd.Series, method: str) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    mask = p.notna()
    out = pd.Series(np.nan, index=p.index, dtype=float)
    if mask.sum() == 0 or method.lower() == "none":
        return out if method.lower() != "none" else p

    adj = multipletests(p[mask].to_numpy(), method=method.lower())[1]
    out.loc[mask] = adj
    return out


def adjust_by_role(df: pd.DataFrame, family_cols: list[str], methods: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """
    Adjust df["p_value"] within each family (family_cols + endpoint_role), using the method
    configured for that role. Returns (p_adj, correction_method).
    """
    p_adj = pd.Series(np.nan, index=df.index, dtype=float)
    method_col = df["endpoint_role"].map(methods)
    if method_col.isna().any():
        missing = sorted(df.loc[method_col.isna(), "endpoint_role"].astype(str).unique())
        raise ValueError(f"no correction method configured for endpoint role(s) {missing}")
    for _, idx in df.groupby(family_cols + ["endpoint_role"]).groups.items():
        p_adj.loc[idx] = adjust_pvalues(df.loc[idx, "p_value"], method=method_col.loc[idx].iloc[0])
    return p_adj, method_col


def _cell_levels(cell: str) -> dict[str, str]:
    """'group=cup;sex=female' -> {'group': 'cup', 'sex': 'female'}"""
    return dict(part.split("=", 1) for part in str(cell).split(";") if "=" in part)


def posthoc_gate(posthoc: pd.DataFrame, aov: pd.DataFrame | None, alpha: float = 0.05) -> pd.Series:
    """
    'passed' if some ANOVA term involving the compared factor (main effect or interaction)
    has p_adj < alpha for that endpoint, 'failed' if none does, '' if it cannot be determined.
    """
    out = pd.Series("", index=posthoc.index, dtype=object)
    if aov is None or aov.empty:
        return out
    terms = aov[aov["p_adj"].notna()]
    for i, row in posthoc.iterrows():
        a, b = _cell_levels(row.get("cell_a", "")), _cell_levels(row.get("cell_b", ""))
        compared = [f for f in a if a.get(f) != b.get(f)]
        if len(compared) != 1:
            continue
        ep_terms = terms[terms["endpoint"] == row["endpoint"]]
        relevant = ep_terms[ep_terms["term"].astype(str).str.contains(f'Q("{compared[0]}")', regex=False)]
        if relevant.empty:
            continue
        out.loc[i] = "passed" if (relevant["p_adj"] < alpha).any() else "failed"
    return out


def significance_label(p: float) -> str:
    """Prism-like threshold labels based on an (adjusted) p-value."""
    if p is None or not np.isfinite(p):
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
