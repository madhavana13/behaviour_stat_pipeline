from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import yaml

from behaviour_stat_pipeline.stats.endpoints.linear_models import validate_model_spec


class ConfigError(ValueError):
    """Raised when a config file is malformed or uses unsupported options."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently keeping the last one."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(
                f"Duplicate key {key!r} in {node.start_mark.name}, line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml_strict(path: str | Path) -> dict:
    """Load YAML, failing on duplicate keys."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_UniqueKeyLoader) or {}


# Keys accepted in each stats section. Anything else is a typo or an unsupported
# option and must fail, not fall back to a default.
_ALLOWED_KEYS = {
    "primary_one_sample": {
        "enabled", "group_col", "collapse_over", "correction", "run_ttest", "run_wilcoxon", "nulls",
    },
    "diagnostics": {
        "enabled", "groupings", "endpoints", "min_n_test", "var_ratio_thresh", "mad_z_thresh",
    },
    "secondary_two_way_anova": {
        "enabled", "type", "include_interaction", "factors", "correction_across_endpoints",
        "factor_orders", "endpoints", "posthoc",
    },
    "secondary_two_way_anova.posthoc": {"enabled", "method", "correction", "selected_pairs"},
    "time_to_event": {"enabled", "factors", "duration_col", "correction_across_endpoints", "endpoints"},
    "linear_models": {"enabled", "rscript", "correction_across_endpoints", "contrast_correction", "models"},
    "correlations": {"enabled", "defaults", "default_groupings", "correction", "min_n", "specs"},
    "correlations.spec": {
        "assay", "x", "y", "name", "method", "groupings", "filter", "covariates", "se", "cluster_col",
    },
}

_MULTITEST_METHODS = {"none", "bonferroni", "holm", "sidak", "fdr_bh", "fdr_by"}


# Sections an experiment's `analysis:` block may set. stats.yaml holds method
# defaults; the experiment decides what is analysed and what is primary.
_ANALYSIS_KEYS = {
    "assays", "design", "primary_endpoints", "primary_one_sample", "diagnostics",
    "secondary_two_way_anova", "time_to_event", "linear_models", "correlations",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Mappings merge recursively; lists and scalars in `override` replace those in `base`."""
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_config(stats_cfg: dict, exp_cfg: Optional[dict]) -> dict:
    """
    Merge an experiment's `analysis:` block over the shared stats config.
    Endpoint lists, model lists and specs in the experiment replace the shared ones.
    """
    analysis = (exp_cfg or {}).get("analysis", {}) or {}
    unknown = sorted(set(analysis) - _ANALYSIS_KEYS)
    if unknown:
        raise ConfigError(f"experiment analysis: unknown keys {unknown}; allowed: {sorted(_ANALYSIS_KEYS)}")
    return _deep_merge(stats_cfg, analysis)


def analysis_endpoints(cfg: dict, key: str, instance_types: dict[str, str]) -> set[str]:
    """All endpoints analysed for an assay instance (incl. its assay type's lists) or a model name."""
    keys = {key.lower(), instance_types.get(key.lower(), key.lower())}
    out: set[str] = set()
    for section, field_name in [("primary_one_sample", "nulls"), ("secondary_two_way_anova", "endpoints"),
                                ("time_to_event", "endpoints")]:
        by_assay = (cfg.get(section, {}) or {}).get(field_name, {}) or {}
        for k in keys:
            out |= set(by_assay.get(k, {}) or {})
    for spec in (cfg.get("linear_models", {}) or {}).get("models", []) or []:
        if isinstance(spec, dict) and str(spec.get("name", "")).lower() == key.lower():
            out |= set(spec.get("endpoints", []) or [])
    return out


def validate_primary_endpoints(cfg: dict, instance_types: dict[str, str]) -> list[str]:
    """Every declared primary endpoint must be analysed somewhere (catches typos)."""
    errors = []
    primary = cfg.get("primary_endpoints", {}) or {}
    if not isinstance(primary, dict):
        return ["primary_endpoints: expected a mapping of assay instance or model name -> endpoint list"]
    for key, eps in primary.items():
        analysed = analysis_endpoints(cfg, str(key), instance_types)
        missing = sorted(set(eps or []) - analysed)
        if missing:
            errors.append(
                f"primary_endpoints.{key}: {missing} are not analysed for '{key}' "
                f"(not in primary_one_sample.nulls, secondary_two_way_anova/time_to_event endpoints or a model named '{key}')"
            )
    return errors


def validate_stats_config(cfg: dict) -> None:
    """
    Check the stats sections for unknown keys, unsupported methods and duplicated endpoints.
    Collects every problem and raises one ConfigError listing them all.
    """
    errors: list[str] = []

    def check_keys(section: dict, allowed_key: str, where: str) -> None:
        if not isinstance(section, dict):
            errors.append(f"{where}: expected a mapping, got {type(section).__name__}")
            return
        unknown = sorted(set(section) - _ALLOWED_KEYS[allowed_key])
        if unknown:
            errors.append(f"{where}: unknown keys {unknown}; allowed: {sorted(_ALLOWED_KEYS[allowed_key])}")

    def check_choice(value: Any, choices: set, where: str) -> None:
        if str(value).lower() not in choices:
            errors.append(f"{where}: {value!r} not supported; choose one of {sorted(choices)}")

    def check_endpoint_lists(by_assay: Any, where: str) -> None:
        for assay, eps in (by_assay or {}).items():
            eps = list(eps or [])
            dupes = sorted({e for e in eps if eps.count(e) > 1})
            if dupes:
                errors.append(f"{where}.{assay}: duplicated endpoints {dupes}")

    pos = cfg.get("primary_one_sample", {}) or {}
    check_keys(pos, "primary_one_sample", "primary_one_sample")
    # Only Holm is implemented for the one-sample family (see endpoints/one_sample.py)
    check_choice(pos.get("correction", "holm"), {"holm", "none"}, "primary_one_sample.correction")

    diag = cfg.get("diagnostics", {}) or {}
    check_keys(diag, "diagnostics", "diagnostics")
    check_endpoint_lists(diag.get("endpoints"), "diagnostics.endpoints")

    sec = cfg.get("secondary_two_way_anova", {}) or {}
    check_keys(sec, "secondary_two_way_anova", "secondary_two_way_anova")
    check_choice(sec.get("type", 3), {"2", "3"}, "secondary_two_way_anova.type")
    check_choice(
        sec.get("correction_across_endpoints", "none"), _MULTITEST_METHODS,
        "secondary_two_way_anova.correction_across_endpoints",
    )
    check_endpoint_lists(sec.get("endpoints"), "secondary_two_way_anova.endpoints")
    posthoc = sec.get("posthoc", {}) or {}
    check_keys(posthoc, "secondary_two_way_anova.posthoc", "secondary_two_way_anova.posthoc")
    if posthoc:
        check_choice(posthoc.get("method", "cell_means_prism"), {"cell_means_prism"}, "secondary_two_way_anova.posthoc.method")
        check_choice(posthoc.get("correction", "sidak"), {"sidak"}, "secondary_two_way_anova.posthoc.correction")

    tte = cfg.get("time_to_event", {}) or {}
    check_keys(tte, "time_to_event", "time_to_event")
    check_choice(
        tte.get("correction_across_endpoints", "holm"), _MULTITEST_METHODS,
        "time_to_event.correction_across_endpoints",
    )
    check_endpoint_lists(tte.get("endpoints"), "time_to_event.endpoints")
    # A censored latency must not also be analysed as if missing at random
    for assay, eps in (tte.get("endpoints") or {}).items():
        both = sorted(set(eps or []) & set((sec.get("endpoints") or {}).get(assay, []) or []))
        if both:
            errors.append(
                f"time_to_event.endpoints.{assay}: {both} also listed in secondary_two_way_anova.endpoints; "
                f"ANOVA would drop censored animals"
            )

    lm_cfg = cfg.get("linear_models", {}) or {}
    check_keys(lm_cfg, "linear_models", "linear_models")
    for key in ("correction_across_endpoints", "contrast_correction"):
        check_choice(lm_cfg.get(key, "holm"), _MULTITEST_METHODS, f"linear_models.{key}")
    model_factors = list((cfg.get("design", {}) or {}).get("between_factors", []) or []) + list(
        (cfg.get("optional", {}) or {}).get("blocking_factors", []) or []
    )
    names = []
    for i, spec in enumerate(lm_cfg.get("models", []) or []):
        if not isinstance(spec, dict):
            errors.append(f"linear_models.models[{i}]: expected a mapping")
            continue
        errors.extend(validate_model_spec(spec, f"linear_models.models[{i}]", model_factors))
        names.append(spec.get("name"))
    dup_names = sorted({n for n in names if n and names.count(n) > 1})
    if dup_names:
        errors.append(f"linear_models.models: duplicated model names {dup_names}")

    corr = cfg.get("correlations", {}) or {}
    check_keys(corr, "correlations", "correlations")
    check_choice(corr.get("correction", "fdr_bh"), _MULTITEST_METHODS, "correlations.correction")
    defaults = corr.get("defaults", {}) or {}
    check_keys(defaults, "correlations.spec", "correlations.defaults")
    for i, spec in enumerate(corr.get("specs", []) or []):
        where = f"correlations.specs[{i}]"
        check_keys(spec, "correlations.spec", where)
        if not isinstance(spec, dict):
            continue
        merged = {**defaults, **spec}
        for k in ("assay", "x", "y"):
            if not merged.get(k):
                errors.append(f"{where}: missing required key '{k}'")
        check_choice(merged.get("method", "spearman"), {"pearson", "spearman", "ols"}, f"{where}.method")
        check_choice(merged.get("se", "robust_hc3"), {"robust_hc3", "robust_hc1", "cluster", "classical"}, f"{where}.se")

    if errors:
        raise ConfigError("Invalid stats config:\n  - " + "\n  - ".join(errors))


@dataclass(frozen=True)
class StatsConfig:
    mouse_id_col: str
    processed_dir: Path
    subjects_path: Path
    out_tables: Path
    out_figures: Path
    out_logs: Path
    between_factors: List[str]
    within_factors: Dict[str, List[str]]
    blocking_factors: List[str]
    cluster_col: str
    assays: List[str]
    primary_one_sample_enabled: bool
    primary_one_sample_group_col: str
    primary_one_sample_collapse_over: list[str]
    primary_one_sample_correction: str
    primary_one_sample_run_ttest: bool
    primary_one_sample_run_wilcoxon: bool
    primary_one_sample_nulls: dict[str, dict[str, float]]
    diagnostics_enabled: bool
    diagnostics_groupings: list[list[str]]
    diagnostics_endpoints: dict[str, list[str]]
    diagnostics_min_n_test: int
    diagnostics_var_ratio_thresh: float
    diagnostics_mad_z_thresh: float
    secondary_two_way_anova_enabled: bool
    secondary_two_way_anova_factors: list[str]
    secondary_two_way_anova_include_interaction: bool
    secondary_two_way_anova_typ: int
    secondary_two_way_anova_endpoints: dict[str, list[str]]
    secondary_two_way_anova_correction: str
    secondary_two_way_anova_posthoc: dict
    secondary_two_way_anova_factor_orders: dict[str, list[str]]
    correlations_enabled: bool
    correlations_correction: str
    correlations_min_n: int
    correlations_specs: list[dict]
    correlations_defaults: dict
    correlations_default_groupings: list[list[str]]
    time_to_event_enabled: bool = False
    time_to_event_factors: list[str] = field(default_factory=lambda: ["group", "sex"])
    time_to_event_duration_col: Optional[str] = None
    time_to_event_correction: str = "holm"
    time_to_event_endpoints: dict[str, list[str]] = field(default_factory=dict)
    linear_models_enabled: bool = False
    linear_models_rscript: Optional[str] = None
    linear_models_correction: str = "holm"
    linear_models_contrast_correction: str = "holm"
    linear_models_models: list[dict] = field(default_factory=list)
    # From the experiment config: assay instance (lower case) -> nominal trial length in s
    trial_durations_s: dict[str, float] = field(default_factory=dict)
    # assay instance or model name (lower case) -> primary endpoints; everything else is exploratory
    primary_endpoints: dict[str, list[str]] = field(default_factory=dict)
    # assay instance (lower case) -> assay type (lower case), e.g. oft2 -> oft
    instance_types: dict[str, str] = field(default_factory=dict)
    # Set per run from provenance; stamped onto every output table
    run_id: str = ""


    def to_dict(self) -> Dict[str, Any]:
        """
        Canonical conversion for legacy code paths that still expect a dict config.

        IMPORTANT:
        - New code should prefer typed attribute access.
        - This method exists only to support transitional callers (e.g. scripts).
        """
        return asdict(self)

def load_stats_config(path: str | Path, experiment_config: Optional[dict] = None) -> StatsConfig:
    """
    Load stats.yaml, merge the experiment's `analysis:` block over it (if given), and validate
    the result. The merged config is what every stage sees.
    """
    path = Path(path)
    instance_types = {
        str(name).lower(): str((inst or {}).get("assay_type", name)).lower()
        for name, inst in ((experiment_config or {}).get("assay_instances", {}) or {}).items()
    }
    cfg = resolve_config(load_yaml_strict(path), experiment_config)
    validate_stats_config(cfg)
    plan_errors = validate_primary_endpoints(cfg, instance_types)
    if plan_errors:
        raise ConfigError("Invalid analysis plan:\n  - " + "\n  - ".join(plan_errors))

    mouse_id_col = cfg["id_cols"]["mouse"]
    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"])
    subjects_path = Path(paths["metadata_subjects"])
    out_tables = Path(paths["out_tables"])
    out_figures = Path(paths["out_figures"])
    out_logs = Path(paths["out_logs"])
    diag = cfg.get("diagnostics", {})
    diagnostics_enabled = bool(diag.get("enabled", False))
    diagnostics_groupings = list(diag.get("groupings", []))
    diagnostics_endpoints = dict(diag.get("endpoints", {}))
    diagnostics_min_n_test = int(diag.get("min_n_test", 3))
    diagnostics_var_ratio_thresh = float(diag.get("var_ratio_thresh", 4.0))
    diagnostics_mad_z_thresh = float(diag.get("mad_z_thresh", 3.5))
    pos = cfg.get("primary_one_sample", {})
    primary_one_sample_enabled = bool(pos.get("enabled", False))
    primary_one_sample_group_col = str(pos.get("group_col", "group"))
    primary_one_sample_collapse_over = list(pos.get("collapse_over", []))
    primary_one_sample_correction = str(pos.get("correction", "holm"))
    primary_one_sample_run_ttest = bool(pos.get("run_ttest", True))
    primary_one_sample_run_wilcoxon = bool(pos.get("run_wilcoxon", True))
    primary_one_sample_nulls = dict(pos.get("nulls", {}))
    sec = cfg.get("secondary_two_way_anova", {})
    secondary_two_way_anova_enabled = bool(sec.get("enabled", False))
    secondary_two_way_anova_factors = list(sec.get("factors", ["group", "sex"]))
    secondary_two_way_anova_include_interaction = bool(sec.get("include_interaction", True))
    secondary_two_way_anova_typ = int(sec.get("type", 3))
    secondary_two_way_anova_endpoints = dict(sec.get("endpoints", {}))
    secondary_two_way_anova_correction = str(sec.get("correction_across_endpoints", "none"))
    secondary_two_way_anova_posthoc = dict(sec.get("posthoc", {}))
    secondary_two_way_anova_factor_orders = dict(sec.get("factor_orders", {}))    

    design = cfg["design"]
    between_factors = list(design.get("between_factors", []))
    within_factors = dict(design.get("within_factors", {}))

    optional = cfg.get("optional", {})
    blocking_factors = list(optional.get("blocking_factors", []))
    cluster_col = optional.get("cluster_col", mouse_id_col)

    assays = list(cfg.get("assays", []))
    corr = cfg.get("correlations", {})
    correlations_enabled = bool(corr.get("enabled", False))
    correlations_correction = str(corr.get("correction", "fdr_bh"))
    correlations_min_n = int(corr.get("min_n", 5))
    correlations_specs_raw = corr.get("specs", []) or []
    if not isinstance(correlations_specs_raw, list):
        raise TypeError("correlations.specs must be a list of spec dicts")

    # shallow validation: each entry must be a dict
    for i, s in enumerate(correlations_specs_raw):
        if not isinstance(s, dict):
            raise TypeError(f"correlations.specs[{i}] must be a dict")

    correlations_specs = correlations_specs_raw
    correlations_defaults = dict(corr.get("defaults", {}) or {})
    correlations_default_groupings = list(corr.get("default_groupings", [[]]) or [[]])

    tte = cfg.get("time_to_event", {}) or {}
    lm_cfg = cfg.get("linear_models", {}) or {}


    return StatsConfig(
        mouse_id_col=mouse_id_col,
        processed_dir=processed_dir,
        subjects_path=subjects_path,
        out_tables=out_tables,
        out_figures=out_figures,
        out_logs=out_logs,
        between_factors=between_factors,
        within_factors=within_factors,
        blocking_factors=blocking_factors,
        cluster_col=cluster_col,
        assays=assays, 
        diagnostics_enabled=diagnostics_enabled,
        diagnostics_groupings=diagnostics_groupings,
        diagnostics_endpoints=diagnostics_endpoints,
        diagnostics_min_n_test=diagnostics_min_n_test,
        diagnostics_var_ratio_thresh=diagnostics_var_ratio_thresh,
        diagnostics_mad_z_thresh=diagnostics_mad_z_thresh,
        primary_one_sample_enabled=primary_one_sample_enabled,
        primary_one_sample_group_col=primary_one_sample_group_col,
        primary_one_sample_collapse_over=primary_one_sample_collapse_over,
        primary_one_sample_correction=primary_one_sample_correction,
        primary_one_sample_run_ttest=primary_one_sample_run_ttest,
        primary_one_sample_run_wilcoxon=primary_one_sample_run_wilcoxon,
        primary_one_sample_nulls=primary_one_sample_nulls,
        secondary_two_way_anova_enabled=secondary_two_way_anova_enabled,
        secondary_two_way_anova_factors=secondary_two_way_anova_factors,
        secondary_two_way_anova_include_interaction=secondary_two_way_anova_include_interaction,
        secondary_two_way_anova_typ=secondary_two_way_anova_typ,
        secondary_two_way_anova_endpoints=secondary_two_way_anova_endpoints,
        secondary_two_way_anova_correction=secondary_two_way_anova_correction,
        secondary_two_way_anova_posthoc=secondary_two_way_anova_posthoc,
        secondary_two_way_anova_factor_orders=secondary_two_way_anova_factor_orders,
        correlations_enabled=correlations_enabled,
        correlations_correction=correlations_correction,
        correlations_min_n=correlations_min_n,
        correlations_specs=correlations_specs,
        correlations_defaults=correlations_defaults,
        correlations_default_groupings=correlations_default_groupings,
        time_to_event_enabled=bool(tte.get("enabled", False)),
        time_to_event_factors=list(tte.get("factors", ["group", "sex"])),
        time_to_event_duration_col=tte.get("duration_col") or None,
        time_to_event_correction=str(tte.get("correction_across_endpoints", "holm")),
        time_to_event_endpoints=dict(tte.get("endpoints", {}) or {}),
        linear_models_enabled=bool(lm_cfg.get("enabled", False)),
        linear_models_rscript=lm_cfg.get("rscript") or None,
        linear_models_correction=str(lm_cfg.get("correction_across_endpoints", "holm")),
        linear_models_contrast_correction=str(lm_cfg.get("contrast_correction", "holm")),
        linear_models_models=list(lm_cfg.get("models", []) or []),
        primary_endpoints={
            str(k).lower(): list(v or []) for k, v in (cfg.get("primary_endpoints", {}) or {}).items()
        },
        instance_types=instance_types,
    )


def endpoint_role(cfg: StatsConfig, key: str, endpoint: str) -> str:
    """'primary' if declared in the experiment's primary_endpoints (by instance, assay type or model), else 'exploratory'."""
    k = key.lower()
    keys = {k, cfg.instance_types.get(k, k)}
    for kk in keys:
        if endpoint in cfg.primary_endpoints.get(kk, []):
            return "primary"
    return "exploratory"


def read_processed(processed_dir: Path, assay: str) -> pd.DataFrame:
    # Your files are named e.g., NOR_processed.csv, VIT_processed.csv
    fname = f"{assay.upper()}_processed.csv"
    path = processed_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)


def read_subjects(subjects_path: Path) -> pd.DataFrame:
    if not subjects_path.exists():
        raise FileNotFoundError(f"Subjects metadata not found: {subjects_path}")
    return pd.read_csv(subjects_path)


def merge_metadata(
    df: pd.DataFrame,
    subjects: pd.DataFrame,
    mouse_id_col: str,
) -> pd.DataFrame:
    if mouse_id_col not in df.columns:
        raise KeyError(f"Processed data missing '{mouse_id_col}' column")

    if mouse_id_col not in subjects.columns:
        raise KeyError(f"Subjects metadata missing '{mouse_id_col}' column")

    merged = df.merge(subjects, on=mouse_id_col, how="left", validate="m:1")

    missing = merged[mouse_id_col][merged.isna().any(axis=1) & merged[mouse_id_col].notna()]
    # Above is conservative; better: check any missing in key metadata cols later.

    return merged


def require_columns(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns: {missing}")


def coerce_categoricals(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        # Keep NaNs as NaNs; categories handle this fine
        df[c] = df[c].astype("category")
    return df


def drop_single_level_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """
    Returns:
      kept_factors, dropped_info where dropped_info is [(col, levels_observed)]
    """
    kept: List[str] = []
    dropped: List[Tuple[str, List[str]]] = []
    for c in factor_cols:
        if c not in df.columns:
            continue
        observed = [x for x in df[c].dropna().unique().tolist()]
        if len(observed) <= 1:
            dropped.append((c, [str(x) for x in observed]))
        else:
            kept.append(c)
    return kept, dropped


def cell_counts(df: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame({"n": [len(df)]})
    # counts per unique combination of factor levels
    out = (
        df.groupby(factors, dropna=False, observed=False)
        .size()
        .reset_index(name="n")
        .sort_values(by=factors)
        .reset_index(drop=True)
    )
    return out
