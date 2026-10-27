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


def _null_llf(d: pd.DataFrame) -> float:
    """Cox partial log-likelihood with no covariates (all hazard ratios 1), Efron ties."""
    model = PHReg(d["_time"].to_numpy(), np.zeros((len(d), 1)), status=d["_event"].to_numpy(), ties="efron")
    return float(model.loglike(np.zeros(1)))


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
    One or two factors, any number of levels each.

    Returns
    -------
    tests : per factor, a likelihood-ratio test against the model without it (df = levels - 1).
            Two-level factor: that row also carries the hazard ratio of level vs reference.
            More levels: an omnibus row (term = factor) plus one hazard-ratio row per level
            (term = "factor[level vs reference]", p_value = Wald p).
            Two factors: interaction LR test; log-rank test of the first factor
            (stratified by the second, if any).
    summary : one row per level of the first factor (and per cell with two factors):
              n, events, censored, KM median
    """
    if len(factors) not in (1, 2):
        raise ValueError(f"time-to-event supports one or two factors, got {factors}")
    f1 = factors[0]
    f2 = factors[1] if len(factors) == 2 else None

    d = build_survival_data(df, endpoint, trial_duration_s=trial_duration_s, duration_col=duration_col)
    d = d.dropna(subset=list(factors))
    for f in factors:
        d[f] = d[f].astype(str)

    base = {"assay": assay, "endpoint": endpoint, "n": int(len(d)),
            "n_events": int(d["_event"].sum()), "n_censored": int((d["_event"] == 0).sum())}

    summary_rows = [{"assay": assay, "endpoint": endpoint, "grouping": f1, **r} for r in summarise_groups(d, [f1])]
    if f2:
        summary_rows += [{"assay": assay, "endpoint": endpoint, "grouping": f"{f1}+{f2}", **r}
                         for r in summarise_groups(d, [f1, f2])]
    summary = pd.DataFrame(summary_rows)

    if min(d[f].nunique() for f in factors) < 2 or d["_event"].sum() == 0:
        return pd.DataFrame([{**base, "term": "SKIPPED", "reason": "factor with <2 levels or no events"}]), summary

    term_of = {f: f"C({_q(f)})" for f in factors}
    model_label = " + ".join(factors)
    m_add = _fit_cox(d, " + ".join(term_of[f] for f in factors))
    # Log-likelihood of the model without each factor (the null model for a single factor)
    reduced_llf = (
        {f1: _fit_cox(d, term_of[f2]).llf, f2: _fit_cox(d, term_of[f1]).llf} if f2 else {f1: _null_llf(d)}
    )

    rows = []
    names = list(m_add.model.exog_names)
    for factor in factors:
        levels = sorted(d[factor].unique())
        ref_level = levels[0]  # patsy's treatment reference: first level in sorted order
        df_term = len(levels) - 1
        chi2 = float(2.0 * (m_add.llf - reduced_llf[factor]))
        p_lr = float(stats.chi2.sf(chi2, df_term))
        idxs = [i for i, n in enumerate(names) if n.startswith(term_of[factor] + "[")]

        def hr_fields(i):
            beta, se = float(m_add.params[i]), float(m_add.bse[i])
            return {
                "hazard_ratio": float(np.exp(beta)),
                "hr_ci_low": float(np.exp(beta - 1.959963984540054 * se)),
                "hr_ci_high": float(np.exp(beta + 1.959963984540054 * se)),
                "wald_p": float(m_add.pvalues[i]),
            }

        if df_term == 1:
            level = names[idxs[0]].split("[T.")[-1].rstrip("]")
            rows.append({**base, "term": factor, "model": model_label, "contrast": f"{level} vs {ref_level}",
                         **hr_fields(idxs[0]), "lr_chi2": chi2, "lr_df": 1, "p_value": p_lr})
        else:
            rows.append({**base, "term": factor, "model": model_label, "contrast": "omnibus",
                         "hazard_ratio": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan, "wald_p": np.nan,
                         "lr_chi2": chi2, "lr_df": df_term, "p_value": p_lr})
            for i in idxs:
                level = names[i].split("[T.")[-1].rstrip("]")
                hr = hr_fields(i)
                rows.append({**base, "term": f"{factor}[{level} vs {ref_level}]", "model": model_label,
                             "contrast": f"{level} vs {ref_level}", **hr,
                             "lr_chi2": np.nan, "lr_df": np.nan, "p_value": hr["wald_p"]})

    if f2:
        m_int = _fit_cox(d, f"{term_of[f1]} * {term_of[f2]}")
        df_int = (d[f1].nunique() - 1) * (d[f2].nunique() - 1)
        chi2 = float(2.0 * (m_int.llf - m_add.llf))
        rows.append({
            **base, "term": f"{f1}:{f2}", "model": f"{f1} * {f2} vs {f1} + {f2}",
            "contrast": "", "hazard_ratio": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan, "wald_p": np.nan,
            "lr_chi2": chi2, "lr_df": df_int, "p_value": float(stats.chi2.sf(chi2, df_int)),
        })

    lr_stat, lr_p = survdiff(
        d["_time"].to_numpy(), d["_event"].to_numpy(), d[f1].to_numpy(),
        strata=d[f2].to_numpy() if f2 else None,
    )
    rows.append({
        **base, "term": f"{f1} (log-rank, stratified by {f2})" if f2 else f"{f1} (log-rank)", "model": "log-rank",
        "contrast": "", "hazard_ratio": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan, "wald_p": np.nan,
        "lr_chi2": float(lr_stat), "lr_df": int(d[f1].nunique() - 1), "p_value": float(lr_p),
    })

    return pd.DataFrame(rows), summary
