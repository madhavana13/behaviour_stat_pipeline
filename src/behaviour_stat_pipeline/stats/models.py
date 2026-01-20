from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def one_sample_tests(
    x: pd.Series,
    mu0: float = 0.0,
) -> dict:
    """
    Primary: one-sample t-test (mean vs mu0)
    Sensitivity: Wilcoxon signed-rank (median vs mu0)

    Returns a dict of summary + test results.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = int(x.shape[0])

    out = {
        "n": n,
        "mean": float(x.mean()) if n else np.nan,
        "sd": float(x.std(ddof=1)) if n > 1 else np.nan,
        "median": float(x.median()) if n else np.nan,
        "mu0": mu0,
    }

    if n >= 2:
        t_res = stats.ttest_1samp(x.to_numpy(), popmean=mu0, nan_policy="omit")
        out["t_stat"] = float(t_res.statistic)
        out["t_p"] = float(t_res.pvalue)

        # 95% CI for mean difference (x - mu0)
        se = x.std(ddof=1) / np.sqrt(n)
        df = n - 1
        tcrit = stats.t.ppf(0.975, df)
        diff_mean = x.mean() - mu0
        out["mean_diff"] = float(diff_mean)
        out["ci_low"] = float(diff_mean - tcrit * se)
        out["ci_high"] = float(diff_mean + tcrit * se)
    else:
        out.update({"t_stat": np.nan, "t_p": np.nan, "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan})

    # Wilcoxon signed-rank on differences (x - mu0)
    # Needs at least 1 non-zero difference; SciPy also behaves better with n>=5.
    diffs = (x - mu0).to_numpy()
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs != 0]

    if diffs.size >= 1:
        try:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
            out["w_stat"] = float(w_res.statistic)
            out["w_p"] = float(w_res.pvalue)
        except ValueError:
            out["w_stat"] = np.nan
            out["w_p"] = np.nan
    else:
        out["w_stat"] = np.nan
        out["w_p"] = np.nan

    return out
