import pandas as pd

from behaviour_stat_pipeline.stats.models import one_sample_tests
from behaviour_stat_pipeline.stats.corrections import holm_adjust


PRIMARY_DI_ENDPOINTS = ["di_explore"]  # you can add "di_iz" later


def run_nor_di_one_sample(
    df: pd.DataFrame,
    endpoint: str = "di_explore",
    group_col: str = "group",
    mu0: float = 0.0,
) -> pd.DataFrame:
    """
    Primary: one-sample t-test vs mu0 within each group (sex collapsed).
    Sensitivity: Wilcoxon signed-rank vs mu0.

    Returns one row per group.
    """
    d = df.copy()
    d = d.dropna(subset=[endpoint, group_col])

    rows = []
    for g, sub in d.groupby(group_col, dropna=False):
        res = one_sample_tests(sub[endpoint], mu0=mu0)
        rows.append({
            "assay": "NOR",
            "endpoint": endpoint,
            "group": str(g),
            "mu0": mu0,
            **res,
        })

    out = pd.DataFrame(rows)

    # Holm adjustment across groups for the primary t-tests
    out["t_p_adj_holm"] = holm_adjust(out["t_p"])

    return out.sort_values(["endpoint", "group"]).reset_index(drop=True)

def run_nor_all_primary_di(
    df: pd.DataFrame,
    endpoints: list[str] = ["di_explore", "di_iz"],
    mu0: float = 0.0,
) -> pd.DataFrame:
    frames = []
    for ep in endpoints:
        frames.append(run_nor_di_one_sample(df, endpoint=ep, mu0=mu0))
    return pd.concat(frames, ignore_index=True)
