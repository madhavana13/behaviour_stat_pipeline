from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


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

        # Robust outlier flagging by MAD z-score (no exclusions)
        if id_col in df.columns and n_nonmissing >= 5 and not np.isnan(mad) and mad > 0:
            med = np.nanmedian(x_all)
            robust_z = 0.6745 * (x_all - med) / mad
            mask = robust_z.abs() > mad_z_thresh
            flagged = df.loc[mask.fillna(False), [id_col]].copy()
            flagged["assay"] = assay
            flagged["endpoint"] = ep
            flagged["value"] = x_all[mask.fillna(False)].values
            flagged["robust_z_mad"] = robust_z[mask.fillna(False)].values
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
