"""
Validate the R/nlme linear-model backend against independent implementations:

- gls without random effects or variance weights == OLS Type III ANOVA and cell-mean
  contrasts (base R reference values in reference_values.json)
- lme random-intercept model == statsmodels MixedLM (REML)
- gls with varIdent == statsmodels WLS using the estimated variance ratios

Tests that need R are skipped when Rscript is not available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from behaviour_stat_pipeline.stats.endpoints.linear_models import (
    RBackendError,
    apply_transforms,
    assemble_model_data,
    find_rscript,
    fit_models_r,
    validate_model_spec,
)

HERE = Path(__file__).resolve().parent
REF = json.loads((HERE / "reference_values.json").read_text(encoding="utf-8"))

try:
    find_rscript()
    HAVE_R = True
except RBackendError:
    HAVE_R = False
needs_r = pytest.mark.skipif(not HAVE_R, reason="Rscript not available")


def close(actual, expected, rtol=1e-6):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=rtol, atol=0)


@needs_r
def test_gls_equals_ols_type3_and_cell_contrasts():
    df = pd.read_csv(HERE / "fixture.csv")
    spec = {"name": "t", "assays": ["x"], "endpoints": ["y"], "fixed": "group * sex", "random": None,
            "contrasts": [{"factor": "group", "by": ["sex"]}]}
    out = fit_models_r(df, spec, id_col="mouse_id", factors=["group", "sex"],
                       factor_orders={"group": ["control", "ifs"], "sex": ["female", "male"]})
    aov = out["anova"].set_index("term")
    for term, key in [("group", "group"), ("sex", "sex"), ("group:sex", "interaction")]:
        close(aov.loc[term, "F"], REF["anova"]["type3"][key]["F"])
        close(aov.loc[term, "p_value"], REF["anova"]["type3"][key]["p"])
        assert aov.loc[term, "den_df"] == REF["anova"]["type3"]["df_resid"]

    con = out["contrasts"].set_index("by_level")
    for sex, pair in [("female", "group_within_sex_0"), ("male", "group_within_sex_1")]:
        row, ref = con.loc[f"sex={sex}"], REF["posthoc"][pair]
        close(row["estimate"], ref["diff"])
        close(row["se"], ref["se"])
        close(row["t"], ref["t"])
        close(row["p_value"], ref["p"])
        assert row["df"] == ref["df"]


@needs_r
def test_lme_random_intercept_equals_statsmodels_mixedlm():
    df = pd.read_csv(HERE / "fixture_repeated.csv")
    spec = {"name": "t", "assays": ["x"], "endpoints": ["y"], "fixed": "group * session",
            "random": "~ 1 | mouse_id"}
    out = fit_models_r(df, spec, id_col="mouse_id", factors=["group", "session"])
    coefs = out["coefficients"]

    d = df.dropna(subset=["y"])
    sm_fit = smf.mixedlm("y ~ C(group, Sum) * C(session, Sum)", d, groups=d["mouse_id"]).fit(reml=True)
    k = len(coefs)
    # Same sum-to-zero coding and term order in R and patsy, so coefficients align by position
    close(coefs["estimate"], sm_fit.fe_params.to_numpy()[:k], rtol=1e-4)
    close(coefs["se"], sm_fit.bse_fe.to_numpy()[:k], rtol=1e-3)

    re = out["random_effects"].set_index("component")["std_dev"]
    close(re["(Intercept)"], np.sqrt(float(sm_fit.cov_re.iloc[0, 0])), rtol=1e-3)
    close(re["Residual"], np.sqrt(sm_fit.scale), rtol=1e-3)


@needs_r
def test_gls_varident_equals_wls_with_estimated_weights():
    df = pd.read_csv(HERE / "fixture_repeated.csv")
    df = df[df["session"] == "S1"].dropna(subset=["y"])
    spec = {"name": "t", "assays": ["x"], "endpoints": ["y"], "fixed": "group * sex", "random": None,
            "variance_by": ["group"]}
    out = fit_models_r(df, spec, id_col="mouse_id", factors=["group", "sex"])
    info = out["model_info"].iloc[0]
    ratios = dict(kv.split("=") for kv in info["variance_ratios"].split(";"))
    sd = df["group"].map(lambda g: float(ratios[g]))
    wls = smf.wls("y ~ C(group, Sum) * C(sex, Sum)", df, weights=1.0 / sd**2).fit()
    close(out["coefficients"]["estimate"], wls.params.to_numpy(), rtol=1e-6)
    close(out["coefficients"]["se"], wls.bse.to_numpy(), rtol=1e-6)
    assert 0.0 <= info["hetero_p"] <= 1.0


@needs_r
def test_baseline_covariate_assembled_per_mouse_and_endpoint():
    rep = pd.read_csv(HERE / "fixture_repeated.csv")
    frames = {
        "base": rep[rep["session"] == "S1"][["mouse_id", "group", "sex", "y"]],
        "s2": rep[rep["session"] == "S2"][["mouse_id", "group", "sex", "y"]],
        "s3": rep[rep["session"] == "S3"][["mouse_id", "group", "sex", "y"]],
    }
    spec = {"name": "t", "assays": ["s2", "s3"], "stack_as": "session", "baseline": "base",
            "endpoints": ["y"], "fixed": "group * session + baseline", "random": "~ 1 | mouse_id"}
    long = assemble_model_data(frames, spec, id_col="mouse_id", factors=["group", "sex"])
    assert len(long) == 2 * 24
    m = long.merge(frames["base"][["mouse_id", "y"]].rename(columns={"y": "b"}), on="mouse_id")
    assert np.allclose(m["baseline__y"], m["b"], equal_nan=True)
    out = fit_models_r(long, spec, id_col="mouse_id", factors=["group", "sex"])
    assert out["model_info"]["status"].eq("ok").all()
    assert "baseline" in set(out["anova"]["term"])


@needs_r
def test_log_transform_equals_fitting_logged_data():
    df = pd.read_csv(HERE / "fixture.csv")
    spec = {"name": "t", "assays": ["x"], "endpoints": ["y"], "fixed": "group * sex", "random": None,
            "transforms": {"y": "log"}}
    a = fit_models_r(apply_transforms(df, spec), spec, id_col="mouse_id", factors=["group", "sex"])
    b = fit_models_r(df.assign(y=np.log(df["y"])), {**spec, "transforms": {}}, id_col="mouse_id",
                     factors=["group", "sex"])
    close(a["anova"]["F"], b["anova"]["F"], rtol=1e-10)


def test_transform_outside_domain_fails():
    df = pd.DataFrame({"y": [1.0, 0.0, 2.0], "baseline__y": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="needs values > 0"):
        apply_transforms(df, {"name": "t", "transforms": {"y": "log"}})
    errs = validate_model_spec({"name": "m", "assays": ["a"], "endpoints": ["y"], "fixed": "group",
                                "transforms": {"y": "box-cox", "z": "log"}}, "m", ["group"])
    assert any("'box-cox' not supported" in e for e in errs)
    assert any("'z' is not one of the model's endpoints" in e for e in errs)


def test_validation_rejects_redundant_and_unsafe_specs():
    base = {"name": "m", "endpoints": ["y"], "fixed": "group * session", "stack_as": "session"}
    errs = validate_model_spec({**base, "assays": ["a", "b"], "random": None}, "m", ["group", "sex"])
    assert any("need a random effect" in e for e in errs)
    errs = validate_model_spec({**base, "assays": ["a", "b"], "random": "~1|mouse_id", "baseline": "a",
                                "fixed": "group * session + baseline"}, "m", ["group", "sex"])
    assert any("covariate or as session, not both" in e for e in errs)
    errs = validate_model_spec({**base, "assays": ["a"], "fixed": "group + cage"}, "m", ["group", "sex"])
    assert any("fixed uses ['cage']" in e for e in errs)
