"""
Linear and linear mixed models (fixed + random effects, ANCOVA covariates,
unequal residual variances) via an optional R backend (nlme).

Why R: statsmodels cannot model unequal residual variances in mixed models and
gives only z-tests (no small-sample denominator df) for them. nlme ships with
every R installation and does both. R is only needed for this stage; the rest of
the pipeline stays pure Python.

One model spec (stats.yaml -> linear_models.models[i]) covers one design and
many endpoints:
  assays:      one or more assay instances; more than one are stacked into long
               format with a factor named `stack_as` (e.g. session = OFT2/OFT3/OFT4)
  baseline:    optional assay instance whose value of the same endpoint is added
               per mouse as covariate `baseline` (ANCOVA). It must not also be a
               stacked level: a session is either a covariate or a repeated level.
  fixed:       right-hand side, e.g. "group * sex * session + baseline"
  random:      nlme syntax, e.g. "~ 1 | mouse_id"; null -> no random effects (gls)
  variance_by: factors with their own residual variance (nlme varIdent); [] -> equal
  contrasts:   [{factor: group, by: [session]}] -> pairwise levels of `factor`
               within each level combination of `by`, cell means equally weighted
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parents[1] / "r" / "fit_nlme.R"

_ALLOWED_MODEL_KEYS = {
    "name", "assays", "stack_as", "baseline", "endpoints", "fixed", "random", "covariates",
    "variance_by", "contrasts", "method", "transforms",
}

# name -> (function, domain check, domain description)
TRANSFORMS = {
    "log": (np.log, lambda x: x > 0, "> 0"),
    "log1p": (np.log1p, lambda x: x >= 0, ">= 0 (e.g. counts)"),
    "sqrt": (np.sqrt, lambda x: x >= 0, ">= 0"),
    "logit": (lambda x: np.log(x / (1 - x)), lambda x: (x > 0) & (x < 1), "strictly between 0 and 1 (proportions)"),
}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_FORMULA_FUNCS = {"I", "log", "poly"}


class RBackendError(RuntimeError):
    """R is missing, or the R model fit failed as a whole."""


def find_rscript(configured: Optional[str] = None) -> str:
    """Locate Rscript: config value, BSP_RSCRIPT env var, PATH, then standard Windows installs."""
    candidates = [configured, os.environ.get("BSP_RSCRIPT"), shutil.which("Rscript")]
    candidates += sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True)
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise RBackendError(
        "linear_models needs R (with the recommended package nlme) but Rscript was not found. "
        "Install R, or set linear_models.rscript in stats.yaml or the BSP_RSCRIPT environment variable."
    )


def formula_variables(rhs: str) -> set[str]:
    """Variable names in a formula right-hand side (operators and known functions removed)."""
    return {t for t in _IDENT.findall(rhs or "") if t not in _FORMULA_FUNCS}


def validate_model_spec(spec: dict, where: str, factors: list[str]) -> list[str]:
    """Problems with one model spec, without looking at data."""
    errors: list[str] = []
    unknown = sorted(set(spec) - _ALLOWED_MODEL_KEYS)
    if unknown:
        errors.append(f"{where}: unknown keys {unknown}; allowed: {sorted(_ALLOWED_MODEL_KEYS)}")
    for key in ("name", "assays", "endpoints", "fixed"):
        if not spec.get(key):
            errors.append(f"{where}: missing required key '{key}'")
    assays = [str(a).lower() for a in spec.get("assays", []) or []]
    stacked = len(assays) > 1
    stack_as = spec.get("stack_as")
    if stacked and not stack_as:
        errors.append(f"{where}: several assays are stacked; set stack_as (e.g. session)")
    if stacked and not spec.get("random"):
        errors.append(
            f"{where}: stacked repeated measures need a random effect (e.g. random: '~ 1 | mouse_id'); "
            f"without one, repeated observations of a mouse are treated as independent mice"
        )
    baseline = spec.get("baseline")
    if baseline and str(baseline).lower() in assays:
        errors.append(
            f"{where}: baseline '{baseline}' is also a stacked level; use it either as covariate or as session, not both"
        )
    random = spec.get("random")
    if random and not str(random).strip().startswith("~"):
        errors.append(f"{where}: random must be an nlme formula starting with '~', e.g. '~ 1 | mouse_id'")
    if spec.get("method", "REML") not in ("REML", "ML"):
        errors.append(f"{where}: method must be REML or ML")
    covariates = list(spec.get("covariates", []) or []) + (["baseline"] if baseline else [])
    allowed_vars = set(factors) | set(covariates) | ({stack_as} if stack_as else set())
    used = formula_variables(str(spec.get("fixed", "")))
    bad = sorted(used - allowed_vars)
    if bad:
        errors.append(f"{where}: fixed uses {bad}; allowed: factors {factors}, covariates {covariates}, stack_as")
    if baseline and "baseline" not in used:
        errors.append(f"{where}: baseline is set but 'baseline' is not in fixed")
    for v in spec.get("variance_by", []) or []:
        if v not in set(factors) | ({stack_as} if stack_as else set()):
            errors.append(f"{where}: variance_by '{v}' must be a factor")
    for i, c in enumerate(spec.get("contrasts", []) or []):
        if not isinstance(c, dict) or not c.get("factor"):
            errors.append(f"{where}.contrasts[{i}]: needs 'factor' (and optional 'by' list)")
            continue
        for v in [c["factor"], *(c.get("by", []) or [])]:
            if v not in used:
                errors.append(f"{where}.contrasts[{i}]: '{v}' is not a factor in fixed")
    eps = list(spec.get("endpoints", []) or [])
    dupes = sorted({e for e in eps if eps.count(e) > 1})
    if dupes:
        errors.append(f"{where}: duplicated endpoints {dupes}")
    transforms = spec.get("transforms", {}) or {}
    if not isinstance(transforms, dict):
        errors.append(f"{where}.transforms: expected a mapping endpoint -> transform")
    else:
        for ep, tr in transforms.items():
            if ep not in eps:
                errors.append(f"{where}.transforms: '{ep}' is not one of the model's endpoints")
            if tr not in TRANSFORMS:
                errors.append(f"{where}.transforms.{ep}: '{tr}' not supported; choose one of {sorted(TRANSFORMS)}")
    return errors


def apply_transforms(long: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Transform endpoints (and their baseline__ columns, so the covariate is on the
    same scale) as configured in spec['transforms']. Fails if any value is outside
    the transform's domain instead of silently producing NaN/inf.
    """
    long = long.copy()
    for ep, tr in (spec.get("transforms", {}) or {}).items():
        func, ok, domain = TRANSFORMS[tr]
        for col in [ep, f"baseline__{ep}"]:
            if col not in long.columns:
                continue
            x = pd.to_numeric(long[col], errors="coerce")
            bad = x.notna() & ~ok(x)
            if bad.any():
                raise ValueError(
                    f"linear_models '{spec['name']}': {tr} transform of {col} needs values {domain}; "
                    f"found {x[bad].head(5).tolist()}"
                )
            long[col] = func(x)
    return long


def assemble_model_data(
    frames: dict[str, pd.DataFrame],
    spec: dict,
    *,
    id_col: str,
    factors: list[str],
) -> pd.DataFrame:
    """
    Build the analysis table: one row per mouse (single assay) or mouse x stacked
    assay. `frames` maps assay instance (lower case) -> processed data with
    metadata merged and exclusions already applied (excluded values set to NaN).
    Adds baseline__<endpoint> columns when spec['baseline'] is set.
    """
    endpoints = list(spec["endpoints"])
    covariates = list(spec.get("covariates", []) or [])
    stack_as = spec.get("stack_as")
    random_vars = sorted(formula_variables(str(spec.get("random") or "")) - {stack_as})
    keep = list(dict.fromkeys([id_col, *factors, *covariates, *random_vars]))

    pieces = []
    for assay in [str(a).lower() for a in spec["assays"]]:
        df = frames[assay]
        missing = [c for c in keep + endpoints if c not in df.columns]
        if missing:
            raise KeyError(f"linear_models '{spec['name']}': {assay} is missing columns {missing}")
        piece = df[keep + endpoints].copy()
        if stack_as:
            piece[stack_as] = assay.upper()
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)

    key = [id_col] + ([stack_as] if stack_as else [])
    dup = long.duplicated(key)
    if dup.any():
        raise ValueError(f"linear_models '{spec['name']}': duplicate rows for {key}: {long.loc[dup, key].head().values.tolist()}")

    baseline = spec.get("baseline")
    if baseline:
        base = frames[str(baseline).lower()]
        missing = [c for c in [id_col, *endpoints] if c not in base.columns]
        if missing:
            raise KeyError(f"linear_models '{spec['name']}': baseline {baseline} is missing columns {missing}")
        base = base[[id_col, *endpoints]].rename(columns={e: f"baseline__{e}" for e in endpoints})
        if base[id_col].duplicated().any():
            raise ValueError(f"linear_models '{spec['name']}': duplicate {id_col} in baseline {baseline}")
        long = long.merge(base, on=id_col, how="left", validate="m:1")
    return long


def _factor_levels(long: pd.DataFrame, names: list[str], orders: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {}
    for f in names:
        observed = [str(x) for x in long[f].dropna().astype(str).unique()]
        preferred = [str(x) for x in (orders.get(f) or []) if str(x) in observed]
        out[f] = preferred + sorted(x for x in observed if x not in preferred)
    return out


def fit_models_r(
    long: pd.DataFrame,
    spec: dict,
    *,
    id_col: str,
    factors: list[str],
    factor_orders: Optional[dict[str, list[str]]] = None,
    rscript: Optional[str] = None,
    timeout_s: int = 1800,
) -> dict[str, pd.DataFrame]:
    """
    Fit spec['fixed'] / spec['random'] for every endpoint in R.
    Returns tables: anova, coefficients, contrasts, random_effects, model_info.
    """
    rscript = find_rscript(rscript)
    stack_as = spec.get("stack_as")
    factor_names = [f for f in [*factors, stack_as] if f and f in long.columns]
    levels = _factor_levels(long, factor_names, factor_orders or {})
    if stack_as:
        # Keep the configured assay order for the stacked factor
        levels[stack_as] = [a.upper() for a in spec["assays"] if a.upper() in levels[stack_as]]

    r_spec: dict[str, Any] = {
        "endpoints": list(spec["endpoints"]),
        "fixed": str(spec["fixed"]),
        "random": spec.get("random") or None,
        "variance_by": list(spec.get("variance_by", []) or []),
        "method": spec.get("method", "REML"),
        "id_col": id_col,
        "covariates": list(spec.get("covariates", []) or []) + (["baseline"] if spec.get("baseline") else []),
        "per_endpoint_columns": {"baseline": "baseline__%s"} if spec.get("baseline") else {},
        "factor_levels": levels,
        "contrasts": [{"factor": c["factor"], "by": list(c.get("by", []) or [])} for c in spec.get("contrasts", []) or []],
    }

    with tempfile.TemporaryDirectory(prefix="bsp_nlme_") as tmp:
        tmp = Path(tmp)
        (tmp / "spec.json").write_text(json.dumps(r_spec), encoding="utf-8")
        long.to_csv(tmp / "data.csv", index=False)
        proc = subprocess.run(
            [rscript, "--vanilla", str(R_SCRIPT), str(tmp / "spec.json"), str(tmp / "data.csv"), str(tmp)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RBackendError(f"R model fit failed for '{spec['name']}':\n{proc.stderr.strip()}")
        out = {}
        for name in ["anova", "coefficients", "contrasts", "random_effects", "model_info"]:
            df = pd.read_csv(tmp / f"{name}.csv")
            # R writes an empty data.frame as a single "" header
            out[name] = pd.DataFrame() if list(df.columns) in ([""], ["Unnamed: 0"]) else df
    for df in out.values():
        if not df.empty:
            df.insert(0, "model", spec["name"])
    return out
