from __future__ import annotations

import pandas as pd

from behaviour_stat_pipeline.stats.models import one_sample_tests
from behaviour_stat_pipeline.stats.corrections import holm_adjust


def run_one_sample_by_group(
    df: pd.DataFrame,
    assay: str,
    endpoint: str,
    group_col: str,
    mu0: float,
    id_col: str | None = None,
    run_ttest: bool = True,
    run_wilcoxon: bool = True,
    correction: str = "holm",
) -> pd.DataFrame:
    d = df.dropna(subset=[endpoint, group_col]).copy()

    rows = []
    for g, sub in d.groupby(group_col, dropna=False, observed=False):
        x = sub[endpoint].dropna()

        if id_col is not None:
            if id_col not in sub.columns:
                raise ValueError(f"id_col {id_col!r} not found in dataframe")
            dupes = sub.loc[sub[id_col].duplicated(), id_col].tolist()
            if dupes:
                raise ValueError(
                    f"Duplicate {id_col} values detected in group {g!r}: {dupes[:10]}"
                )

        if len(x) < 2:
            row = {
                "assay": assay,
                "endpoint": endpoint,
                "group_col": group_col,
                "group": str(g),
                "mu0": mu0,
                "n": len(x),
                "mean": x.mean() if len(x) else None,
                "sd": x.std(ddof=1) if len(x) > 1 else None,
                "median": x.median() if len(x) else None,
                "t_stat": None,
                "t_p": None,
                "mean_diff": None,
                "ci_low": None,
                "ci_high": None,
                "w_stat": None,
                "w_p": None,
            }
            rows.append(row)
            continue

        res = one_sample_tests(x, mu0=mu0)

        row = {
            "assay": assay,
            "endpoint": endpoint,
            "group_col": group_col,
            "group": str(g),
            "mu0": mu0,
            "n": res["n"],
            "mean": res["mean"],
            "sd": res["sd"],
            "median": res["median"],
            "t_stat": res["t_stat"] if run_ttest else None,
            "t_p": res["t_p"] if run_ttest else None,
            "mean_diff": res["mean_diff"] if run_ttest else None,
            "ci_low": res["ci_low"] if run_ttest else None,
            "ci_high": res["ci_high"] if run_ttest else None,
            "w_stat": res["w_stat"] if run_wilcoxon else None,
            "w_p": res["w_p"] if run_wilcoxon else None,
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    if run_ttest and correction.lower() == "holm" and "t_p" in out.columns:
        valid = out["t_p"].notna()
        out["t_p_adj_holm"] = None
        out.loc[valid, "t_p_adj_holm"] = holm_adjust(out.loc[valid, "t_p"])
    else:
        out["t_p_adj_holm"] = None

    if run_wilcoxon and correction.lower() == "holm" and "w_p" in out.columns:
        valid = out["w_p"].notna()
        out["w_p_adj_holm"] = None
        out.loc[valid, "w_p_adj_holm"] = holm_adjust(out.loc[valid, "w_p"])
    else:
        out["w_p_adj_holm"] = None

    return out.sort_values(["assay", "endpoint", "group"]).reset_index(drop=True)

def run_one_sample_pooled(
    df: pd.DataFrame,
    assay: str,
    endpoint: str,
    mu0: float,
    id_col: str | None = None,
    run_ttest: bool = True,
    run_wilcoxon: bool = True,
) -> pd.DataFrame:

    d = df.dropna(subset=[endpoint]).copy()
    x = d[endpoint]

    if id_col is not None:
        if id_col not in d.columns:
            raise ValueError(f"id_col {id_col!r} not found in dataframe")
        dupes = d.loc[d[id_col].duplicated(), id_col].tolist()
        if dupes:
            raise ValueError(f"Duplicate {id_col} values detected: {dupes[:10]}")

    if len(x) < 2:
        raise ValueError("Not enough observations for pooled one-sample test")

    res = one_sample_tests(x, mu0=mu0)

    row = {
        "assay": assay,
        "endpoint": endpoint,
        "group_col": "pooled",
        "group": "all",
        "mu0": mu0,
        "n": res["n"],
        "mean": res["mean"],
        "sd": res["sd"],
        "median": res["median"],
        "t_stat": res["t_stat"] if run_ttest else None,
        "t_p": res["t_p"] if run_ttest else None,
        "mean_diff": res["mean_diff"] if run_ttest else None,
        "ci_low": res["ci_low"] if run_ttest else None,
        "ci_high": res["ci_high"] if run_ttest else None,
        "w_stat": res["w_stat"] if run_wilcoxon else None,
        "w_p": res["w_p"] if run_wilcoxon else None,
    }

    return pd.DataFrame([row])