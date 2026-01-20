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