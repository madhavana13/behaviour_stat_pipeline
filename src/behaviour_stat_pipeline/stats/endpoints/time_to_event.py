"""
Time-to-event analysis for latency endpoints.

Why: a latency is missing when the event never happened during the trial
(e.g. the mouse never entered the narrow alley). Dropping those animals removes
the most extreme responders and biases group comparisons towards "no effect".
Here such animals are right-censored at the end of their recording instead.

Censoring rule (per animal):
  - latency observed                 -> time = latency, event = 1
  - latency missing, animal recorded -> time = recording length, event = 0
  - animal not recorded              -> excluded (missing data, e.g. lost video)
Recording length comes from a per-animal duration column if configured and
present, otherwise from the nominal trial duration in the experiment config.

Model (mirrors the Type II two-way ANOVA used elsewhere):
  - Cox proportional hazards, Efron ties
  - group and sex: hazard ratios from the additive model  group + sex
  - interaction: likelihood-ratio test of  group * sex  vs  group + sex
  - all three terms also get likelihood-ratio p-values against the model without that term
  - log-rank test of group, stratified by sex (Prism-style comparison)
A hazard ratio > 1 means the event happens sooner (shorter latency).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff


def _q(name: str) -> str:
    return f'Q("{name}")'


def build_survival_data(
    df: pd.DataFrame,
    endpoint: str,
    *,
    trial_duration_s: Optional[float],
    duration_col: Optional[str] = None,
    tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """
    Return df with added columns `_time`, `_event` (1 observed, 0 censored).
    Animals without a recording are dropped. Fails if an observed latency exceeds
    the recording length (wrong duration config) or no duration is available.
    """
    d = df.copy()
    latency = pd.to_numeric(d[endpoint], errors="coerce")

    if duration_col and duration_col in d.columns:
        duration = pd.to_numeric(d[duration_col], errors="coerce")
        recorded = duration.notna()
        if trial_duration_s is not None:
            duration = duration.fillna(float(trial_duration_s))
    elif trial_duration_s is not None:
        duration = pd.Series(float(trial_duration_s), index=d.index)
        recorded = pd.Series(True, index=d.index)
    else:
        raise ValueError(
            f"{endpoint}: no trial duration available. Set trial_duration_s for this assay "
            f"in the experiment config, or time_to_event.duration_col in stats.yaml."
        )

    too_long = latency.notna() & (latency > duration + tolerance_s)
    if too_long.any():
        bad = d.loc[too_long, endpoint].head(5).tolist()
        raise ValueError(
            f"{endpoint}: observed latencies exceed the recording length ({bad} > "
            f"{duration[too_long].head(5).tolist()}); check trial_duration_s / duration_col."
        )

    d["_event"] = latency.notna().astype(int)
    d["_time"] = latency.where(latency.notna(), duration)
    return d.loc[recorded & d["_time"].notna()].copy()


def _fit_cox(d: pd.DataFrame, rhs: str):
    model = PHReg.from_formula(f"_time ~ {rhs}", data=d, status=d["_event"].to_numpy(), ties="efron")
    return model.fit(disp=False)


def _km_median(time: np.ndarray, event: np.ndarray) -> float:
    """Kaplan-Meier median (NaN if survival never drops to 0.5, i.e. not reached)."""
    if len(time) == 0:
        return np.nan
    sf = SurvfuncRight(time, event)
    surv = sf.surv_prob
    times = sf.surv_times
    below = np.flatnonzero(surv <= 0.5 + 1e-10)  # e.g. 3/6 can land just above 0.5
    if below.size == 0:
        return np.nan
    i = below[0]
    # R/survival convention: if S(t) is exactly 0.5 on an interval, take the midpoint
    if np.isclose(surv[i], 0.5) and i + 1 < len(times):
        return float((times[i] + times[i + 1]) / 2.0)
    return float(times[i])


def summarise_groups(d: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, sub in d.groupby(group_cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **{c: str(v) for c, v in zip(group_cols, key)},
            "n": int(len(sub)),
            "n_events": int(sub["_event"].sum()),
            "n_censored": int((sub["_event"] == 0).sum()),
            "km_median": _km_median(sub["_time"].to_numpy(), sub["_event"].to_numpy()),
        })
    return rows


def run_time_to_event(
    df: pd.DataFrame,
    *,
    assay: str,
    endpoint: str,
    factors: list[str],
    trial_duration_s: Optional[float],
    duration_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    tests : one row per term (group, sex, interaction) + one log-rank row
    summary : one row per group and per group x sex cell (n, events, censored, KM median)
    """
    if len(factors) != 2:
        raise ValueError(f"time-to-event requires exactly 2 factors, got {factors}")
    f1, f2 = factors

    d = build_survival_data(df, endpoint, trial_duration_s=trial_duration_s, duration_col=duration_col)
    d = d.dropna(subset=[f1, f2])
    for f in (f1, f2):
        d[f] = d[f].astype(str)

    base = {"assay": assay, "endpoint": endpoint, "n": int(len(d)),
            "n_events": int(d["_event"].sum()), "n_censored": int((d["_event"] == 0).sum())}

    summary = pd.DataFrame(
        [{"assay": assay, "endpoint": endpoint, "grouping": f1, **r} for r in summarise_groups(d, [f1])]
        + [{"assay": assay, "endpoint": endpoint, "grouping": f"{f1}+{f2}", **r} for r in summarise_groups(d, [f1, f2])]
    )

    if d[f1].nunique() < 2 or d[f2].nunique() < 2 or d["_event"].sum() == 0:
        return pd.DataFrame([{**base, "term": "SKIPPED", "reason": "factor with <2 levels or no events"}]), summary

    t1, t2 = f"C({_q(f1)})", f"C({_q(f2)})"
    m_add = _fit_cox(d, f"{t1} + {t2}")
    m_int = _fit_cox(d, f"{t1} * {t2}")
    m_f1 = _fit_cox(d, t1)
    m_f2 = _fit_cox(d, t2)

    rows = []
    for term, factor, reduced_llf in [
        (f1, f1, m_f2.llf),   # group adjusted for sex
        (f2, f2, m_f1.llf),   # sex adjusted for group
    ]:
        names = list(m_add.model.exog_names)
        idx = next(i for i, n in enumerate(names) if n.startswith(f"C({_q(factor)})"))
        beta, se = float(m_add.params[idx]), float(m_add.bse[idx])
        chi2 = float(2.0 * (m_add.llf - reduced_llf))
        level = names[idx].split("[T.")[-1].rstrip("]")
        ref_level = sorted(d[factor].unique())[0]
        rows.append({
            **base, "term": term, "model": f"{f1} + {f2}",
            "contrast": f"{level} vs {ref_level}",
            "hazard_ratio": float(np.exp(beta)),
            "hr_ci_low": float(np.exp(beta - 1.959963984540054 * se)),
            "hr_ci_high": float(np.exp(beta + 1.959963984540054 * se)),
            "wald_p": float(m_add.pvalues[idx]),
            "lr_chi2": chi2, "lr_df": 1, "p_value": float(stats.chi2.sf(chi2, 1)),
        })

    chi2 = float(2.0 * (m_int.llf - m_add.llf))
    rows.append({
        **base, "term": f"{f1}:{f2}", "model": f"{f1} * {f2} vs {f1} + {f2}",
        "contrast": "", "hazard_ratio": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan, "wald_p": np.nan,
        "lr_chi2": chi2, "lr_df": 1, "p_value": float(stats.chi2.sf(chi2, 1)),
    })

    lr_stat, lr_p = survdiff(d["_time"].to_numpy(), d["_event"].to_numpy(), d[f1].to_numpy(), strata=d[f2].to_numpy())
    rows.append({
        **base, "term": f"{f1} (log-rank, stratified by {f2})", "model": "log-rank",
        "contrast": "", "hazard_ratio": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan, "wald_p": np.nan,
        "lr_chi2": float(lr_stat), "lr_df": int(d[f1].nunique() - 1), "p_value": float(lr_p),
    })

    return pd.DataFrame(rows), summary
