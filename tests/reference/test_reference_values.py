"""
Validate pipeline statistics against independent reference values from base R.

Reference values: tests/reference/reference_values.json, produced by
generate_reference.R from fixture.csv. R is not needed to run these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behaviour_stat_pipeline.stats.corrections import adjust_pvalues
from behaviour_stat_pipeline.stats.endpoints.correlations import _ols_one, run_correlations
from behaviour_stat_pipeline.stats.endpoints.one_sample import run_one_sample_by_group
from behaviour_stat_pipeline.stats.endpoints.posthoc_cellmeans_prism import run_posthoc_cellmeans_prism
from behaviour_stat_pipeline.stats.endpoints.two_way_anova import run_two_way_anova

HERE = Path(__file__).resolve().parent
REF = json.loads((HERE / "reference_values.json").read_text(encoding="utf-8"))
RTOL = 1e-8  # closed-form statistics: agreement to numerical precision


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(HERE / "fixture.csv")


def close(actual, expected, rtol=RTOL):
    np.testing.assert_allclose(float(actual), float(expected), rtol=rtol, atol=0)


# ---------------------------------------------------------------------------
# One-sample
# ---------------------------------------------------------------------------

def test_one_sample_by_group(df):
    out = run_one_sample_by_group(df, assay="REF", endpoint="di", group_col="group", mu0=0.0, id_col="mouse_id")
    for _, row in out.iterrows():
        ref = REF["one_sample"][row["group"]]
        assert row["n"] == ref["n"]
        close(row["t_stat"], ref["t"])
        close(row["t_p"], ref["t_p"])
        close(row["ci_low"], ref["ci_low"])
        close(row["ci_high"], ref["ci_high"])
        close(row["w_stat"], ref["w_min"])
        close(row["w_p"], ref["w_p"])
        close(row["t_p_adj_holm"], ref["t_p_holm"])
        close(row["w_p_adj_holm"], ref["w_p_holm"])


# ---------------------------------------------------------------------------
# Two-way ANOVA
# ---------------------------------------------------------------------------

TERM_KEYS = {
    'C(Q("group"))': "group",
    'C(Q("sex"))': "sex",
    'C(Q("group")):C(Q("sex"))': "interaction",
}


@pytest.mark.parametrize("typ", [2, 3])
def test_two_way_anova(df, typ):
    out = run_two_way_anova(df, assay="REF", endpoint="y", factors=["group", "sex"], include_interaction=True, typ=typ)
    ref = REF["anova"][f"type{typ}"]
    rows = out[out["term"].isin(TERM_KEYS)]
    assert len(rows) == 3, out["term"].tolist()
    for _, row in rows.iterrows():
        r = ref[TERM_KEYS[row["term"]]]
        close(row["sum_sq"], r["sum_sq"])
        close(row["df"], r["df"])
        close(row["F"], r["F"])
        close(row["p_value"], r["p"])
    resid = out[out["term"] == "Residual"].iloc[0]
    close(resid["df"], ref["df_resid"])


# ---------------------------------------------------------------------------
# Posthoc on cell means
# ---------------------------------------------------------------------------

def test_posthoc_cellmeans(df):
    pairs = [
        {"name": "group_within_sex_0", "a": {"group": 0, "sex": 0}, "b": {"group": 1, "sex": 0}},
        {"name": "sex_within_group_0", "a": {"group": 0, "sex": 0}, "b": {"group": 0, "sex": 1}},
        {"name": "sex_within_group_1", "a": {"group": 1, "sex": 0}, "b": {"group": 1, "sex": 1}},
        {"name": "group_within_sex_1", "a": {"group": 0, "sex": 1}, "b": {"group": 1, "sex": 1}},
    ]
    out = run_posthoc_cellmeans_prism(
        df, assay="REF", endpoints=["y"], factors=["group", "sex"], include_interaction=True,
        factor_orders={"group": ["control", "ifs"], "sex": ["female", "male"]},
        selected_pairs=pairs, correction="sidak",
    )
    assert len(out) == len(pairs)
    for _, row in out.iterrows():
        r = REF["posthoc"][row["pair_name"]]
        close(row["diff_a_minus_b"], r["diff"])
        close(row["se_pooled"], r["se"])
        close(row["t"], r["t"])
        close(row["df_error"], r["df"])
        close(row["p_value"], r["p"])
        close(row["p_value_sidak"], r["p_sidak"])


# ---------------------------------------------------------------------------
# Correlations and OLS
# ---------------------------------------------------------------------------

def _corr(df, method):
    cfg = {"correlations": {
        "enabled": True, "correction": "none", "min_n": 5, "default_groupings": [[]],
        "specs": [{"assay": "ref", "x": "x2", "y": "y", "method": method}],
    }}
    out = run_correlations(assay="ref", df=df, cfg=cfg)
    assert len(out) == 1
    return out.iloc[0]


def test_pearson(df):
    row = _corr(df, "pearson")
    ref = REF["correlations"]["pearson"]
    assert row["n"] == ref["n"]
    close(row["statistic"], ref["r"])
    close(row["p_value"], ref["p"], rtol=1e-6)


def test_spearman(df):
    # SciPy's spearmanr p-value is the asymptotic t approximation, which matches
    # R's cor.test(exact = FALSE). R's default for small n without ties is exact.
    row = _corr(df, "spearman")
    ref = REF["correlations"]["spearman"]
    close(row["statistic"], ref["rho"])
    close(row["p_value"], ref["p_asymptotic_t"], rtol=1e-6)


def test_ols_hc3(df):
    beta, se, t, p, df_resid = _ols_one(df, x="x2", y="y", covariates=[], se="robust_hc3", cluster_col=None)
    ref = REF["ols_hc3"]
    close(beta, ref["beta"])
    close(se, ref["se"])
    close(t, ref["t"])
    close(df_resid, ref["df_resid"])
    # Small-sample inference uses the t distribution on residual df, not the normal
    close(p, ref["p_t"], rtol=1e-6)


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["holm", "bonferroni", "fdr_bh", "fdr_by", "sidak"])
def test_adjust_pvalues(method):
    ref = REF["p_adjust"]
    out = adjust_pvalues(pd.Series(ref["input"]), method)
    np.testing.assert_allclose(out.to_numpy(), np.minimum(ref[method], 1.0), rtol=1e-10)
