from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from statsmodels.stats.multitest import fdrcorrection


def _iqr(x: pd.Series) -> float:
    q75, q25 = np.nanpercentile(x, [75, 25])
    return float(q75 - q25)


def _mad(x: pd.Series) -> float:
    x = x.dropna().to_numpy()
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _cv(mean: float, sd: float) -> float:
    if mean is None or sd is None or np.isnan(mean) or np.isnan(sd) or mean == 0:
        return np.nan
    return float(sd / abs(mean))


# statsmodels is used for robust regression; keep import inside function to avoid hard dependency at import time


def rout_endpoint_flags(
    x: pd.Series,
    ids: Optional[pd.Series],
    *,
    assay: str,
    endpoint: str,
    q: float = 0.01,
    min_n: int = 6,
) -> pd.DataFrame:
    """
    ROUT-like outlier flagging (flag only; no exclusions).

    Approach:
      - Fit robust intercept-only model y = mu + e using statsmodels.RLM
      - Compute (approx) studentized residuals
      - Two-sided p-values from a t(n-1) approximation
      - FDR correction at level q (Prism's ROUT 'Q')

    Notes:
      - This is conceptually similar to Prism ROUT, but Prism's exact internal implementation is closed-source.
      - Output is flags only; downstream stats should NOT filter based on this.

    Returns:
      DataFrame with columns:
        assay, endpoint, mouse_id (if provided), value, rout_resid, p_value, q, flagged
    """
    # coerce numeric
    x = pd.to_numeric(x, errors="coerce")
    mask = x.notna()

    if mask.sum() < min_n:
        return pd.DataFrame(columns=[
            "assay", "endpoint", "mouse_id", "value",
            "rout_resid", "p_value", "q", "flagged"
        ])

    y = x.loc[mask].to_numpy()
    mouse_id = None
    if ids is not None:
        mouse_id = ids.loc[mask].astype(str).to_numpy()

    # Robust intercept-only regression
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError(
            "ROUT flagging requires statsmodels. Install with: pip install statsmodels"
        ) from e

    X = np.ones((y.shape[0], 1))  # intercept only
    rlm = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight())
    res = rlm.fit()

    # Residual scale: use robust scale estimate if available; else fallback to sample sd
    # statsmodels RLM exposes res.scale (robust estimate)
    scale = float(getattr(res, "scale", np.std(res.resid, ddof=1)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(res.resid, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        # can't compute meaningful residual z
        return pd.DataFrame(columns=[
            "assay", "endpoint", "mouse_id", "value",
            "rout_resid", "p_value", "q", "flagged"
        ])

    rout_z = res.resid / scale  # approx standardized residuals
    prism_like_flag = np.abs(rout_z) > 3.5


    # Two-sided p-values under Normal approx
    dfree = max(int(y.shape[0] - 1), 1)   # intercept-only => p=1
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(rout_z), df=dfree))


    # FDR correction at level q
    rejected, qvals = fdrcorrection(pvals, alpha=q)


    out = pd.DataFrame({
        "assay": assay,
        "endpoint": endpoint,
        "value": y,
        "robust_z": rout_z,
        "p_value": pvals,
        "q_value": qvals,
        "q_fdr": float(q),
        "flagged_fdr": rejected,
        "flagged_prism": np.abs(rout_z) > 3.5,
    })

    if mouse_id is not None:
        out.insert(2, "mouse_id", mouse_id)

    return out


def statsmodels_fdr(pvals: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    FDR correction helper.
    Returns:
      rejected: boolean array
      qvals: adjusted p-values
    """
    try:
        from statsmodels.stats.multitest import fdrcorrection
    except ImportError as e:
        raise ImportError(
            "FDR correction requires statsmodels. Install with: pip install statsmodels"
        ) from e

    rejected, qvals = fdrcorrection(pvals, alpha=alpha)
    return rejected.astype(bool), qvals.astype(float)


def rout_diagnostics(
    df: pd.DataFrame,
    assay: str,
    endpoints: list[str],
    *,
    group_cols: list[str],
    id_col: str = "mouse_id",
    q: float = 0.01,
    min_n: int = 6,
    keep_all: bool = False,  # set True temporarily for debugging vs Prism
) -> pd.DataFrame:
    """
    Run ROUT-like flagging for multiple endpoints.

    Behavior:
      - If group_cols is non-empty: run ROUT *within each group cell* (Prism-like when you run per column/group)
      - Else: run ROUT pooled across all rows

    Returns:
      One long table with group columns included.
      By default returns only flagged rows, unless keep_all=True.
    """
    rows: list[pd.DataFrame] = []

    # Decide splits
    if group_cols:
        # groupby returns (key, subdf); key can be scalar or tuple
        grouped = df.groupby(group_cols, dropna=False, observed=False)
        groups_iter = list(grouped)
    else:
        groups_iter = [(None, df)]

    for ep in endpoints:
        if ep not in df.columns:
            continue

        for key, sub in groups_iter:
            ids = sub[id_col] if id_col in sub.columns else None

            flags = rout_endpoint_flags(
                sub[ep],
                ids,
                assay=assay,
                endpoint=ep,
                q=q,
                min_n=min_n,
            )
            if flags.empty:
                continue

            # Attach group labels for traceability
            if group_cols:
                if not isinstance(key, tuple):
                    key = (key,)
                for col, val in zip(group_cols, key):
                    flags[col] = val

            flags = flags.loc[
                flags["flagged_fdr"] | flags["flagged_prism"]
            ].copy()

            if flags.empty:
                continue

            if not flags.empty:
                rows.append(flags)

    if rows:
        out = pd.concat(rows, ignore_index=True)

        # Nice column order
        front = ["assay", "endpoint"]
        if group_cols:
            front += group_cols
        if "mouse_id" in out.columns:
            front += ["mouse_id"]
        rest = [c for c in out.columns if c not in front]
        return out[front + rest]

    base_cols = ["assay", "endpoint"] + (group_cols or []) + ["mouse_id", "value", "rout_resid", "p_value", "q", "flagged"]
    return pd.DataFrame(columns=base_cols)


def endpoint_diagnostics(
    df: pd.DataFrame,
    assay: str,
    endpoints: list[str],
    group_cols: list[str],
    id_col: str = "mouse_id",
    min_n_test: int = 3,
    var_ratio_thresh: float = 4.0,
    mad_z_thresh: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      diag_df: one row per endpoint with summary + variance flags
      flag_df: one row per (endpoint, id) flagged by robust MAD z-score (no exclusions)
    """
    diag_rows = []
    flag_rows = []

    for ep in endpoints:
        if ep not in df.columns:
            diag_rows.append({
                "assay": assay, "endpoint": ep,
                "status": "missing_column",
            })
            continue

        x_all = pd.to_numeric(df[ep], errors="coerce")
        n_total = int(x_all.shape[0])
        n_nonmissing = int(x_all.notna().sum())
        missing_frac = float(1.0 - (n_nonmissing / n_total)) if n_total > 0 else np.nan

        # global summaries (collapsed)
        mean = float(x_all.mean(skipna=True)) if n_nonmissing else np.nan
        sd = float(x_all.std(skipna=True, ddof=1)) if n_nonmissing > 1 else np.nan
        median = float(x_all.median(skipna=True)) if n_nonmissing else np.nan
        iqr = _iqr(x_all) if n_nonmissing else np.nan
        mad = _mad(x_all) if n_nonmissing else np.nan
        vmin = float(np.nanmin(x_all)) if n_nonmissing else np.nan
        vmax = float(np.nanmax(x_all)) if n_nonmissing else np.nan
        cv = _cv(mean, sd)

        # Robust outlier flagging by MAD z-score (no exclusions), within each group cell
        # so that genuine group differences are not flagged as outliers
        if id_col in df.columns:
            if group_cols:
                cells = x_all.groupby([df[c] for c in group_cols], dropna=False, observed=True)
            else:
                cells = [(None, x_all)]
            for _, x_cell in cells:
                x_cell = x_cell.dropna()
                cell_mad = _mad(x_cell)
                if x_cell.size < 5 or np.isnan(cell_mad) or cell_mad <= 0:
                    continue
                robust_z = 0.6745 * (x_cell - x_cell.median()) / cell_mad
                mask = robust_z.abs() > mad_z_thresh
                flagged = df.loc[mask[mask].index, [id_col] + list(group_cols)].copy()
                flagged["assay"] = assay
                flagged["endpoint"] = ep
                flagged["value"] = x_cell[mask].values
                flagged["robust_z_mad"] = robust_z[mask].values
                for _, r in flagged.iterrows():
                    flag_rows.append(r.to_dict())

        # Variance per group cell
        var_table = None
        if group_cols:
            g = df[group_cols].copy()
            d = pd.concat([g, x_all.rename("y")], axis=1)
            d = d.dropna(subset=["y"])
            if not d.empty:
                var_table = (
                    d.groupby(group_cols, dropna=False, observed=False)["y"]
                    .agg(n="size", var=lambda s: float(np.var(s, ddof=1)) if s.size > 1 else np.nan)
                    .reset_index()
                )

        # Variance ratio + Brown–Forsythe (median-centered Levene)
        var_ratio = np.nan
        bf_p = np.nan
        bf_flag = False

        if var_table is not None and "var" in var_table.columns:
            vars_ = var_table["var"].dropna().to_numpy()
            if vars_.size >= 2 and np.nanmin(vars_) > 0:
                var_ratio = float(np.nanmax(vars_) / np.nanmin(vars_))

        # Brown–Forsythe across GROUP ONLY (recommended default) if group is in group_cols
        # If you pass group_cols=["group"] it does what you want; if group×sex, it tests across cells.
        if group_cols:
            d = pd.concat([df[group_cols], x_all.rename("y")], axis=1).dropna(subset=["y"])
            # require at least 2 groups/cells with enough n
            if not d.empty:
                groups = []
                for _, sub in d.groupby(group_cols, dropna=False, observed=False):
                    if sub.shape[0] >= min_n_test:
                        groups.append(sub["y"].to_numpy())
                if len(groups) >= 2:
                    # Brown–Forsythe == Levene(center='median')
                    bf_stat, bf_p = stats.levene(*groups, center="median")
                    bf_p = float(bf_p)
                    bf_flag = (bf_p < 0.05) or (not np.isnan(var_ratio) and var_ratio >= var_ratio_thresh)

        diag_rows.append({
            "assay": assay,
            "endpoint": ep,
            "status": "ok",
            "n_total": n_total,
            "n_nonmissing": n_nonmissing,
            "missing_frac": missing_frac,
            "mean": mean,
            "sd": sd,
            "cv": cv,
            "median": median,
            "iqr": iqr,
            "mad": mad,
            "min": vmin,
            "max": vmax,
            "var_ratio": var_ratio,
            "brown_forsythe_p": bf_p,
            "hetero_flag": bool(bf_flag),
            "mad_z_thresh": mad_z_thresh,
            "var_ratio_thresh": var_ratio_thresh,
            "min_n_test": min_n_test,
            "grouping": "×".join(group_cols) if group_cols else "",
        })

    diag_df = pd.DataFrame(diag_rows)
    flag_df = pd.DataFrame(flag_rows)
    return diag_df, flag_df
