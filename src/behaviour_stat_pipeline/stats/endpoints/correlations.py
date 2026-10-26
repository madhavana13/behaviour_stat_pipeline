from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import re


@dataclass(frozen=True)
class CorrelationResult:
    assay: str
    spec_name: str
    analysis_type: str  # "corr" or "ols"
    method: str
    grouping: str
    group_key: str
    x: str
    y: str
    n: int
    statistic: float
    p_value: float
    # regression-only fields (NaN for corr methods)
    beta_x: float
    se_x: float
    t_x: float
    df_resid: float

def _slug(s: str) -> str:
    """
    Create a filesystem- and CSV-safe identifier from a column name.
    """
    s = str(s).lower().strip()
    s = re.sub(r"\s+", "_", s)          # spaces → underscores
    s = re.sub(r"[^a-z0-9_]+", "", s)   # drop everything else
    return s


def _safe_query(df: pd.DataFrame, expr: Optional[str]) -> pd.DataFrame:
    if expr is None or str(expr).strip() == "" or str(expr).lower() == "null":
        return df
    return df.query(expr, engine="python")


def _group_iterator(df: pd.DataFrame, group_cols: List[str]):
    """Yield (group_key_string, sub_df). If group_cols empty => one pooled group."""
    if not group_cols:
        yield ("ALL", df)
        return

    g = df.groupby(group_cols, dropna=False, observed=False)
    for key, sub in g:
        if not isinstance(key, tuple):
            key = (key,)
        key_str = ";".join([f"{c}={v}" for c, v in zip(group_cols, key)])
        yield (key_str, sub)

def _resolve_corr_specs(cfg: dict, assay: str) -> list[dict]:
    corr = cfg.get("correlations", {}) or {}
    defaults = corr.get("defaults", {}) or {}
    default_groupings = corr.get("default_groupings", [[]]) or [[]]

    resolved = []
    for i, s in enumerate(corr.get("specs", []) or []):
        if not isinstance(s, dict):
            raise TypeError(f"correlations.specs[{i}] must be a dict")

        out = dict(defaults)  # start with global defaults
        out.update(s)         # spec overrides defaults

        out.setdefault("assay", assay.lower())
        out.setdefault("method", "spearman")

        if out.get("groupings", None) is None:
            out["groupings"] = default_groupings

        # auto-name
        if not out.get("name"):
            out["name"] = f"{assay.lower()}_{_slug(out['x'])}_vs_{_slug(out['y'])}"

        # required keys after resolution
        for k in ["name", "assay", "method", "x", "y", "groupings"]:
            if k not in out:
                raise ValueError(f"Resolved spec missing key '{k}'")

        resolved.append(out)

    return resolved

def _corr_one(method: str, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    if method == "spearman":
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)
    raise ValueError(f"Unknown correlation method: {method}")


def _ols_one(
    df: pd.DataFrame,
    x: str,
    y: str,
    covariates: List[str],
    se: str,
    cluster_col: Optional[str],
) -> Tuple[float, float, float, float, float]:
    """
    Fit OLS: y ~ x + covariates with chosen SE.
    Returns: beta_x, se_x, t_x, p_x, df_resid
    """
    cols = [y, x] + list(covariates)
    d = df[cols].dropna()

    yv = d[y].astype(float).to_numpy()
    X = d[[x] + list(covariates)].astype(float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(yv, X)
    # use_t: robust covariances default to normal-based p-values in statsmodels,
    # which are anti-conservative at behavioural sample sizes
    if se == "robust_hc3":
        res = model.fit(cov_type="HC3", use_t=True)
    elif se == "robust_hc1":
        res = model.fit(cov_type="HC1", use_t=True)
    elif se == "cluster":
        if not cluster_col:
            raise ValueError("se='cluster' requires cluster_col")
        # cluster groups aligned to d after dropna
        groups = df.loc[d.index, cluster_col]
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    elif se == "classical":
        res = model.fit()
    else:
        raise ValueError(f"Unknown se option: {se!r}")

    beta_x = float(res.params[x])
    se_x = float(res.bse[x])
    t_x = float(res.tvalues[x])
    p_x = float(res.pvalues[x])
    df_resid = float(res.df_resid)
    return beta_x, se_x, t_x, p_x, df_resid


def _adjust_pvalues(pvals: List[float], method: str) -> List[float]:
    method = (method or "none").lower()
    if method in ("none", "no", "null", ""):
        return [float(p) for p in pvals]

    # statsmodels names
    mapping = {
        "bonferroni": "bonferroni",
        "holm": "holm",
        "sidak": "sidak",
        "fdr_bh": "fdr_bh",
        "fdr_by": "fdr_by",
    }
    if method not in mapping:
        raise ValueError(f"Unknown p-value correction method: {method}")

    _, p_adj, _, _ = multipletests(pvals, method=mapping[method])
    return [float(x) for x in p_adj]


def run_correlations(
    *,
    assay: str,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    out_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run correlations/regressions for a given assay based on cfg['correlations'].
    Writes one CSV per assay if out_csv_path is given (caller controls filename).
    Returns the results dataframe.
    """
    c = cfg.get("correlations", {})
    if not c.get("enabled", False):
        return pd.DataFrame()

    specs = _resolve_corr_specs(cfg, assay)
    if not isinstance(specs, list):
        raise ValueError("correlations.specs must be a list")

    default_groupings = c.get("default_groupings", [[]])
    correction = c.get("correction", "fdr_bh")
    min_n = int(c.get("min_n", 6))

    results: List[CorrelationResult] = []

    for spec in specs:
        if str(spec.get("assay", "")).lower() != assay.lower():
            continue

        name = str(spec.get("name", "unnamed"))
        method = str(spec.get("method", "spearman")).lower()
        x = str(spec.get("x"))
        y = str(spec.get("y"))

        if x not in df.columns or y not in df.columns:
            raise ValueError(f"[{assay}:{name}] missing columns: x='{x}' or y='{y}' not in processed CSV")

        groupings = spec.get("groupings", None)
        if groupings is None:
            groupings = default_groupings
        if not isinstance(groupings, list):
            raise ValueError(f"[{assay}:{name}] groupings must be a list (of lists of column names)")

        filt = spec.get("filter", None)
        covariates = list(spec.get("covariates", []) or [])
        se = str(spec.get("se", "robust_hc3")).lower()
        cluster_col = spec.get("cluster_col", None)

        d0 = _safe_query(df, filt)

        for grouping_cols in groupings:
            grouping_cols = list(grouping_cols or [])
            grouping_label = "+".join(grouping_cols) if grouping_cols else "pooled"

            # validate grouping columns
            for gc in grouping_cols:
                if gc not in d0.columns:
                    raise ValueError(f"[{assay}:{name}] grouping column '{gc}' not in df")

            for group_key, sub in _group_iterator(d0, grouping_cols):
                # drop missing x/y for corr methods
                sub_xy = sub[[x, y]].dropna()
                n = int(sub_xy.shape[0])
                if n < min_n:
                    continue

                if method in ("pearson", "spearman"):
                    stat, p = _corr_one(method, sub_xy[x].astype(float).to_numpy(), sub_xy[y].astype(float).to_numpy())
                    row = CorrelationResult(
                        assay=assay.upper(),
                        spec_name=name,
                        analysis_type="corr",
                        method=method,
                        grouping=grouping_label,
                        group_key=group_key,
                        x=x,
                        y=y,
                        n=n,
                        statistic=stat,
                        p_value=float(p),
                        beta_x=float("nan"),
                        se_x=float("nan"),
                        t_x=float("nan"),
                        df_resid=float("nan"),
                    )
                    results.append(row)

                elif method == "ols":
                    # OLS uses covariates; effective N is after dropping missing in y, x, and covariates
                    cols_ols = [y, x] + list(covariates)
                    for col in cols_ols:
                        if col not in sub.columns:
                            raise ValueError(f"[{assay}:{name}] OLS missing column '{col}'")

                    sub_ols = sub[cols_ols].dropna()
                    n_ols = int(sub_ols.shape[0])
                    if n_ols < min_n:
                        continue

                    beta_x, se_x, t_x, p_x, df_resid = _ols_one(
                        sub, x=x, y=y, covariates=covariates, se=se, cluster_col=cluster_col
                    )
                    row = CorrelationResult(
                        assay=assay.upper(),
                        spec_name=name,
                        analysis_type="ols",
                        method=f"ols[{se}]",
                        grouping=grouping_label,
                        group_key=group_key,
                        x=x,
                        y=y,
                        n=n_ols,  # <-- use effective N for OLS
                        statistic=float("nan"),
                        p_value=float(p_x),
                        beta_x=beta_x,
                        se_x=se_x,
                        t_x=t_x,
                        df_resid=df_resid,
                    )
                    results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame([r.__dict__ for r in results])

    # Apply correction within each (assay, spec_name) slice
    # (This matches "within spec" scope, which is usually what you want scientifically.)
    corr_method = (cfg.get("correlations", {}).get("correction", "fdr_bh") or "none").lower()

    # --- multiplicity metadata (defensibility / reproducibility) ---
    out["correction_method"] = corr_method
    out["correction_scope"] = "within_spec"
    out["m_tests"] = (
        out.groupby(["assay", "spec_name"])["p_value"]
           .transform("size")
           .astype(int)
    )

    out["p_adj"] = np.nan
    for (a, s), sub_idx in out.groupby(["assay", "spec_name"]).groups.items():
        pvals = out.loc[sub_idx, "p_value"].astype(float).tolist()
        out.loc[sub_idx, "p_adj"] = _adjust_pvalues(pvals, corr_method)

    out = out.sort_values(["assay", "spec_name", "grouping", "group_key", "x", "y"]).reset_index(drop=True)
    if out_csv_path:
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv_path, index=False)
    return out

