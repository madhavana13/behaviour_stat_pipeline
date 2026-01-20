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
    id_col: str,
    run_ttest: bool = True,
    run_wilcoxon: bool = True,
    correction: str = "holm",
) -> pd.DataFrame:
    d = df.dropna(subset=[endpoint, group_col]).copy()

    rows = []
    for g, sub in d.groupby(group_col, dropna=False, observed=False):
        res = one_sample_tests(sub[endpoint], mu0=mu0)

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
        }

        if run_ttest:
            row.update({
                "t_stat": res["t_stat"],
                "t_p": res["t_p"],
                "mean_diff": res["mean_diff"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
            })
        else:
            row.update({"t_stat": None, "t_p": None, "mean_diff": None, "ci_low": None, "ci_high": None})

        if run_wilcoxon:
            row.update({"w_stat": res["w_stat"], "w_p": res["w_p"]})
        else:
            row.update({"w_stat": None, "w_p": None})

        rows.append(row)

    out = pd.DataFrame(rows)

    # adjust across groups within this endpoint (primary family)
    if run_ttest and correction.lower() == "holm" and "t_p" in out.columns:
        out["t_p_adj_holm"] = holm_adjust(out["t_p"])
    else:
        out["t_p_adj_holm"] = None

    return out.sort_values(["assay", "endpoint", "group"]).reset_index(drop=True)
