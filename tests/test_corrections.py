"""Per-role correction families and post hoc gating on the corrected ANOVA."""

from __future__ import annotations

import pandas as pd
import pytest

from behaviour_stat_pipeline.stats.corrections import adjust_by_role, adjust_pvalues, posthoc_gate

GROUP, SEX, INTER = 'C(Q("group"))', 'C(Q("sex"))', 'C(Q("group")):C(Q("sex"))'


def test_each_role_is_its_own_family_with_its_own_method():
    df = pd.DataFrame({
        "term": [GROUP] * 5,
        "endpoint_role": ["primary", "secondary", "secondary", "exploratory", "exploratory"],
        "p_value": [0.04, 0.01, 0.03, 0.01, 0.03],
    })
    methods = {"primary": "holm", "secondary": "fdr_bh", "exploratory": "holm"}
    p_adj, method = adjust_by_role(df, ["term"], methods)
    assert p_adj[0] == pytest.approx(0.04)  # family of one
    assert list(p_adj[1:3]) == pytest.approx(list(adjust_pvalues(df.p_value[1:3], "fdr_bh")))
    assert list(p_adj[3:5]) == pytest.approx(list(adjust_pvalues(df.p_value[3:5], "holm")))
    assert list(method) == ["holm", "fdr_bh", "fdr_bh", "holm", "holm"]


def test_terms_are_separate_families():
    df = pd.DataFrame({"term": [GROUP, GROUP, SEX], "endpoint_role": ["secondary"] * 3, "p_value": [0.01, 0.02, 0.01]})
    p_adj, _ = adjust_by_role(df, ["term"], {"secondary": "bonferroni"})
    assert list(p_adj) == pytest.approx([0.02, 0.04, 0.01])


def test_missing_role_method_is_an_error():
    df = pd.DataFrame({"term": [GROUP], "endpoint_role": ["secondary"], "p_value": [0.01]})
    with pytest.raises(ValueError, match="secondary"):
        adjust_by_role(df, ["term"], {"primary": "holm"})


def test_posthoc_gate_uses_terms_with_the_compared_factor():
    aov = pd.DataFrame({
        "endpoint": ["a", "a", "a", "b", "b", "b"],
        "term": [GROUP, SEX, INTER] * 2,
        "p_adj": [0.20, 0.01, 0.30, 0.20, 0.50, 0.04],
    })
    posthoc = pd.DataFrame({
        "endpoint": ["a", "a", "b", "c"],
        "cell_a": ["group=cup;sex=male", "group=cup;sex=female", "group=cup;sex=male", "group=cup;sex=male"],
        "cell_b": ["group=tail;sex=male", "group=cup;sex=male", "group=tail;sex=male", "group=tail;sex=male"],
    })
    gate = posthoc_gate(posthoc, aov)
    assert list(gate) == [
        "failed",  # group comparison, neither group nor interaction significant
        "passed",  # sex comparison, sex main effect significant
        "passed",  # group comparison, interaction significant
        "",        # no ANOVA for this endpoint
    ]
