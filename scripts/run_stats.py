from __future__ import annotations
from dataclasses import replace

import argparse
from pathlib import Path
import sys
import textwrap

import pandas as pd

from behaviour_stat_pipeline.common.exclusions import (
    load_exclusions,
    apply_exclusions_for_endpoint,
    exclusions_for_endpoint,
)

from behaviour_stat_pipeline.stats.common import (
    load_stats_config,
    load_yaml_strict,
    read_processed,
    read_subjects,
    merge_metadata,
    require_columns,
    coerce_categoricals,
    cell_counts,
)

from behaviour_stat_pipeline.stats.endpoints.one_sample import (
    run_one_sample_by_group,
    run_one_sample_pooled,
)

from behaviour_stat_pipeline.stats.endpoints.two_way_anova import (
    run_two_way_anova,
    _significance_label,
)

from behaviour_stat_pipeline.stats.endpoints.posthoc_cellmeans_prism import (
    run_posthoc_cellmeans_prism,
)

from behaviour_stat_pipeline.stats.endpoints.correlations import run_correlations
from behaviour_stat_pipeline.stats.endpoints.time_to_event import run_time_to_event
from behaviour_stat_pipeline.stats.diagnostics import endpoint_diagnostics, rout_diagnostics
from behaviour_stat_pipeline.stats.corrections import adjust_pvalues
from behaviour_stat_pipeline.stats.provenance import build_provenance, write_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


# -------------------------
# Utility
# -------------------------

def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")

def write_table(df: pd.DataFrame, path: Path, cfg) -> None:
    """Write an output table stamped with the run_id from {ASSAY}_provenance.json."""
    df = df.copy()
    df["run_id"] = cfg.run_id
    df.to_csv(path, index=False)

def load_yaml(path: str) -> dict:
    if not path:
        return {}
    return load_yaml_strict(path)

def apply_assay_instance_aliases(cfg, exp_cfg: dict):
    """
    Allow assay instances such as OFT1/OFT2 to reuse endpoint/config
    definitions from their canonical assay type, e.g. OFT.

    This is runtime aliasing only: it does not modify the YAML file.

    Handles:
    1. Dict-style config blocks:
       e.g. secondary_two_way_anova_endpoints["oft"] -> ["oft2"]

    2. List-style correlation specs:
       e.g. {"assay": "oft", ...} -> {"assay": "oft2", ...}
    """
    instances = exp_cfg.get("assay_instances", {}) or {}

    # -------------------------
    # 1. Dict-style assay config
    # -------------------------
    dict_attrs = [
        "within_factors",
        "primary_one_sample_nulls",
        "secondary_two_way_anova_endpoints",
        "diagnostics_endpoints",
        "time_to_event_endpoints",
    ]

    # cfg is a frozen dataclass, but the dict objects inside it are mutable.
    # We only add runtime aliases if the instance key is absent.
    for instance_name, inst in instances.items():
        inst = inst or {}
        assay_type = str(inst.get("assay_type", instance_name))

        instance_key = instance_name.lower()
        type_key = assay_type.lower()

        for attr in dict_attrs:
            d = getattr(cfg, attr, None)

            if not isinstance(d, dict):
                continue

            if type_key in d and instance_key not in d:
                d[instance_key] = d[type_key]

    # -------------------------
    # 2. List-style correlation specs
    # -------------------------
    corr_specs = getattr(cfg, "correlations_specs", None)

    if isinstance(corr_specs, list):
        extra_specs = []

        for instance_name, inst in instances.items():
            inst = inst or {}
            assay_type = str(inst.get("assay_type", instance_name))

            instance_key = instance_name.lower()
            type_key = assay_type.lower()

            # If user explicitly configured the instance, do not duplicate it.
            instance_already_defined = any(
                isinstance(spec, dict)
                and str(spec.get("assay", "")).lower() == instance_key
                for spec in corr_specs
            )

            if instance_already_defined:
                continue

            # Copy all specs for the canonical assay type to the assay instance.
            matching_type_specs = [
                spec for spec in corr_specs
                if isinstance(spec, dict)
                and str(spec.get("assay", "")).lower() == type_key
            ]

            for spec in matching_type_specs:
                new_spec = dict(spec)
                new_spec["assay"] = instance_key
                extra_specs.append(new_spec)

        if extra_specs:
            # cfg is frozen, so we must return a new dataclass instance.
            cfg = replace(
                cfg,
                correlations_specs=corr_specs + extra_specs,
            )

    return cfg
    
def log_written(path: Path) -> None:
    print(f"Wrote: {path}")

def exclusion_summary(applied: pd.DataFrame) -> dict:
    """
    Summarise exclusions applied to one assay × endpoint analysis.

    Returns fields that can be attached to every output row.
    """
    if applied is None or applied.empty:
        return {
            "n_excluded": 0,
            "excluded_mouse_ids": "",
            "exclusion_reasons": "",
        }

    return {
        "n_excluded": applied["mouse_id"].nunique(),
        "excluded_mouse_ids": ";".join(
            sorted(applied["mouse_id"].astype(str).unique())
        ),
        "exclusion_reasons": "; ".join(
            sorted(applied["reason"].astype(str).unique())
        ),
    }


def analysis_df_for_endpoint(
    df: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    assay: str,
    endpoint: str,
    cfg,
    analysis_scope: str = "primary",
) -> tuple[pd.DataFrame, dict]:
    """
    Construct the endpoint-specific analysis dataframe.

    Exclusions are applied at the assay × endpoint level.
    Processed files remain unchanged.
    """
    df_ep, applied = apply_exclusions_for_endpoint(
        df,
        exclusions,
        assay=assay.upper(),
        endpoint=endpoint,
        analysis_scope=analysis_scope,
        mouse_id_col=cfg.mouse_id_col,
    )

    return df_ep, exclusion_summary(applied)


def mask_excluded_values(
    df: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    assay: str,
    columns: list[str],
    cfg,
    analysis_scope: str = "primary",
) -> pd.DataFrame:
    """
    Set excluded mice's values to NaN, column by column.

    For analyses that use several endpoints at once (correlations, diagnostics),
    so that a mouse excluded from one endpoint still contributes to the others.
    """
    out = df.copy()
    for col in dict.fromkeys(columns):
        if col not in out.columns:
            continue
        applied = exclusions_for_endpoint(
            exclusions,
            assay=assay.upper(),
            endpoint=col,
            analysis_scope=analysis_scope,
        )
        if applied.empty:
            continue
        excluded_ids = set(applied["mouse_id"].astype(str))
        out.loc[out[cfg.mouse_id_col].astype(str).isin(excluded_ids), col] = float("nan")
    return out

# -------------------------
# Core pipeline stages
# -------------------------

def prepare_analysis_df(df: pd.DataFrame, subjects: pd.DataFrame, cfg, assay: str) -> pd.DataFrame:
    """Ensure all required columns exist and are properly typed."""

    within = cfg.within_factors.get(assay, [])
    between = cfg.between_factors
    blocking = cfg.blocking_factors

    needed = [cfg.mouse_id_col] + between + within + blocking
    missing = [c for c in needed if c not in df.columns]

    if missing:
        df = merge_metadata(df, subjects, mouse_id_col=cfg.mouse_id_col)

    require_columns(df, [cfg.mouse_id_col], f"{assay}: merged")
    require_columns(df, between, f"{assay}: between_factors")
    require_columns(df, within, f"{assay}: within_factors")
    if blocking:
        require_columns(df, blocking, f"{assay}: blocking_factors")

    df = coerce_categoricals(df, between + within + blocking)

    return df


def run_primary_tests(df: pd.DataFrame, cfg, assay: str, exclusions: pd.DataFrame) -> None:
    """Confirmatory one-sample tests."""
    if not getattr(cfg, "primary_one_sample_enabled", False):
        return

    assay_nulls = cfg.primary_one_sample_nulls.get(assay, {})
    if not assay_nulls:
        return

    frames = []

    for endpoint, mu0 in assay_nulls.items():
        if endpoint not in df.columns:
            print(
                f"WARNING [{assay.upper()}]: skipping missing one-sample endpoint: "
                f"{endpoint}"
            )
            continue

        df_ep, excl_info = analysis_df_for_endpoint(
            df,
            exclusions,
            assay=assay,
            endpoint=endpoint,
            cfg=cfg,
            analysis_scope="primary",
        )

        by_group = run_one_sample_by_group(
            df_ep,
            assay=assay.upper(),
            endpoint=endpoint,
            group_col=cfg.primary_one_sample_group_col,
            mu0=float(mu0),
            id_col=cfg.mouse_id_col,
            run_ttest=cfg.primary_one_sample_run_ttest,
            run_wilcoxon=cfg.primary_one_sample_run_wilcoxon,
            correction=cfg.primary_one_sample_correction,
        )

        pooled = run_one_sample_pooled(
            df_ep,
            assay=assay.upper(),
            endpoint=endpoint,
            mu0=float(mu0),
            id_col=cfg.mouse_id_col,
            run_ttest=cfg.primary_one_sample_run_ttest,
            run_wilcoxon=cfg.primary_one_sample_run_wilcoxon,
        )

        res = pd.concat([by_group, pooled], ignore_index=True)

        for k, v in excl_info.items():
            res[k] = v

        res["analysis_scope"] = "primary"
        frames.append(res)

    if not frames:
        print(f"WARNING [{assay.upper()}]: no one-sample tests were run.")
        return

    out = pd.concat(frames, ignore_index=True)
    out_path = cfg.out_tables / f"{assay.upper()}_primary_one_sample.csv"
    write_table(out, out_path, cfg)
    log_written(out_path)

def run_anova(df: pd.DataFrame, cfg, assay: str, exclusions: pd.DataFrame) -> pd.DataFrame | None:
    """Two-way ANOVA + correction."""
    if not getattr(cfg, "secondary_two_way_anova_enabled", False):
        return None

    endpoints = cfg.secondary_two_way_anova_endpoints.get(assay, [])
    if not endpoints:
        return None

    available_endpoints = [ep for ep in endpoints if ep in df.columns]
    missing_endpoints = [ep for ep in endpoints if ep not in df.columns]

    if missing_endpoints:
        print(
            f"WARNING [{assay.upper()}]: skipping missing ANOVA endpoints: "
            f"{missing_endpoints}"
        )

    if not available_endpoints:
        print(f"WARNING [{assay.upper()}]: no available ANOVA endpoints; skipping ANOVA.")
        return None

    frames = []
    for ep in available_endpoints:
        df_ep, excl_info = analysis_df_for_endpoint(
            df,
            exclusions,
            assay=assay,
            endpoint=ep,
            cfg=cfg,
            analysis_scope="primary",
        )

        res = run_two_way_anova(
            df_ep,
            assay=assay.upper(),
            endpoint=ep,
            factors=cfg.secondary_two_way_anova_factors,
            include_interaction=cfg.secondary_two_way_anova_include_interaction,
            typ=cfg.secondary_two_way_anova_typ,
        )

        for k, v in excl_info.items():
            res[k] = v

        res["analysis_scope"] = "primary"
        frames.append(res)

    aov = pd.concat(frames, ignore_index=True)

    method = cfg.secondary_two_way_anova_correction
    if method.lower() != "none":
        aov["p_adj"] = aov.groupby("term")["p_value"].transform(
            lambda s: adjust_pvalues(s, method=method)
        )
    else:
        aov["p_adj"] = aov["p_value"]

    aov["significance_label"] = aov["p_adj"].apply(_significance_label)

    out_path = cfg.out_tables / f"{assay.upper()}_secondary_two_way_anova.csv"
    write_table(aov, out_path, cfg)
    log_written(out_path)

    return aov


def run_posthoc(df: pd.DataFrame, cfg, assay: str, exclusions: pd.DataFrame,) -> None:
    posthoc_cfg = getattr(cfg, "secondary_two_way_anova_posthoc", {}) or {}
    if not posthoc_cfg or not posthoc_cfg.get("enabled", True):
        return

    endpoints = cfg.secondary_two_way_anova_endpoints.get(assay, [])
    available_endpoints = [ep for ep in endpoints if ep in df.columns]
    missing_endpoints = [ep for ep in endpoints if ep not in df.columns]

    if missing_endpoints:
        print(
            f"WARNING [{assay.upper()}]: skipping missing posthoc endpoints: "
            f"{missing_endpoints}"
        )

    if not available_endpoints:
        print(f"WARNING [{assay.upper()}]: no available posthoc endpoints; skipping posthoc.")
        return

    posthoc_frames = []

    for ep in available_endpoints:
        df_ep, excl_info = analysis_df_for_endpoint(
            df,
            exclusions,
            assay=assay,
            endpoint=ep,
            cfg=cfg,
            analysis_scope="primary",
        )

        res = run_posthoc_cellmeans_prism(
            df=df_ep,
            assay=assay.upper(),
            endpoints=[ep],
            factors=cfg.secondary_two_way_anova_factors,
            include_interaction=cfg.secondary_two_way_anova_include_interaction,
            factor_orders=getattr(cfg, "secondary_two_way_anova_factor_orders", {}) or {},
            selected_pairs=list(posthoc_cfg.get("selected_pairs", [])),
            correction=str(posthoc_cfg.get("correction", "sidak")),
        )

        for k, v in excl_info.items():
            res[k] = v

        res["analysis_scope"] = "primary"
        posthoc_frames.append(res)

    if not posthoc_frames:
        return

    df_posthoc = pd.concat(posthoc_frames, ignore_index=True)

    out_path = cfg.out_tables / f"{assay.upper()}_secondary_posthoc_cellmeans.csv"
    write_table(df_posthoc, out_path, cfg)
    log_written(out_path)


def trial_durations_from_experiment(exp_cfg: dict) -> dict[str, float]:
    """assay_instances.<NAME>.trial_duration_s -> {name.lower(): seconds}; must be a positive number."""
    out = {}
    for name, inst in (exp_cfg.get("assay_instances", {}) or {}).items():
        value = (inst or {}).get("trial_duration_s")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"assay_instances.{name}.trial_duration_s must be a positive number, got {value!r}")
        out[name.lower()] = float(value)
    return out


def run_time_to_event_stage(df: pd.DataFrame, cfg, assay: str, exclusions: pd.DataFrame) -> None:
    """Censored latency endpoints: Cox model + log-rank, corrected across endpoints per term."""
    if not cfg.time_to_event_enabled:
        return

    endpoints = cfg.time_to_event_endpoints.get(assay, [])
    if not endpoints:
        return

    missing_endpoints = [ep for ep in endpoints if ep not in df.columns]
    if missing_endpoints:
        print(f"WARNING [{assay.upper()}]: skipping missing time-to-event endpoints: {missing_endpoints}")
    available_endpoints = [ep for ep in endpoints if ep in df.columns]
    if not available_endpoints:
        return

    test_frames, summary_frames = [], []
    for ep in available_endpoints:
        df_ep, excl_info = analysis_df_for_endpoint(
            df, exclusions, assay=assay, endpoint=ep, cfg=cfg, analysis_scope="primary",
        )
        tests, summary = run_time_to_event(
            df_ep,
            assay=assay.upper(),
            endpoint=ep,
            factors=cfg.time_to_event_factors,
            trial_duration_s=cfg.trial_durations_s.get(assay),
            duration_col=cfg.time_to_event_duration_col,
        )
        for k, v in excl_info.items():
            tests[k] = v
        tests["analysis_scope"] = "primary"
        test_frames.append(tests)
        summary_frames.append(summary)

    out = pd.concat(test_frames, ignore_index=True)
    method = cfg.time_to_event_correction
    out["p_adj"] = out.groupby("term")["p_value"].transform(lambda s: adjust_pvalues(s, method=method))
    out["correction_method"] = method
    out["significance_label"] = out["p_adj"].apply(_significance_label)

    out_path = cfg.out_tables / f"{assay.upper()}_time_to_event.csv"
    write_table(out, out_path, cfg)
    log_written(out_path)

    summary_path = cfg.out_tables / f"{assay.upper()}_time_to_event_summary.csv"
    write_table(pd.concat(summary_frames, ignore_index=True), summary_path, cfg)
    log_written(summary_path)


def run_diagnostics(df: pd.DataFrame, cfg, assay: str, exclusions: pd.DataFrame) -> None:
    if not getattr(cfg, "diagnostics_enabled", False):
        return

    endpoints = cfg.diagnostics_endpoints.get(assay, [])
    if not endpoints:
        return

    available_endpoints = [ep for ep in endpoints if ep in df.columns]
    missing_endpoints = [ep for ep in endpoints if ep not in df.columns]

    if missing_endpoints:
        print(
            f"WARNING [{assay.upper()}]: skipping missing diagnostic endpoints: "
            f"{missing_endpoints}"
        )

    if not available_endpoints:
        print(f"WARNING [{assay.upper()}]: no available diagnostic endpoints; skipping diagnostics.")
        return

    df = mask_excluded_values(df, exclusions, assay=assay, columns=available_endpoints, cfg=cfg)

    for grouping in cfg.diagnostics_groupings:
        diag_df, flag_df = endpoint_diagnostics(
            df,
            assay=assay.upper(),
            endpoints=available_endpoints,
            group_cols=grouping,
            id_col=cfg.mouse_id_col,
            min_n_test=cfg.diagnostics_min_n_test,
            var_ratio_thresh=cfg.diagnostics_var_ratio_thresh,
            mad_z_thresh=cfg.diagnostics_mad_z_thresh,
        )

        suffix = "_".join(grouping)
        write_table(diag_df, cfg.out_tables / f"{assay.upper()}_endpoint_diagnostics_by_{suffix}.csv", cfg)
        write_table(flag_df, cfg.out_tables / f"{assay.upper()}_flagged_points_madz_by_{suffix}.csv", cfg)

        rout_df = rout_diagnostics(
            df,
            assay=assay.upper(),
            endpoints=available_endpoints,
            group_cols=grouping,
            id_col=cfg.mouse_id_col,
            q=0.01,
            min_n=6,
        )

        write_table(rout_df, cfg.out_tables / f"{assay.upper()}_flagged_points_rout_by_{suffix}.csv", cfg)
        log_written(cfg.out_tables / f"{assay.upper()}_endpoint_diagnostics_by_{suffix}.csv")
        log_written(cfg.out_tables / f"{assay.upper()}_flagged_points_madz_by_{suffix}.csv")
        log_written(cfg.out_tables / f"{assay.upper()}_flagged_points_rout_by_{suffix}.csv")

def run_exploratory(df: pd.DataFrame, cfg, assay: str, exclusions: pd.DataFrame) -> None:
    if not getattr(cfg, "correlations_enabled", False):
        print(f"WARNING [{assay.upper()}]: correlations disabled; skipping correlations.")
        return

    corr_specs = getattr(cfg, "correlations_specs", [])
    matching_specs = [
        spec for spec in corr_specs
        if isinstance(spec, dict)
        and str(spec.get("assay", "")).lower() == assay.lower()
    ]

    if not matching_specs:
        configured_assays = sorted({
            str(spec.get("assay", "")).lower()
            for spec in corr_specs
            if isinstance(spec, dict)
        })
        print(
            f"WARNING [{assay.upper()}]: no correlation specs matched assay='{assay}'. "
            f"Configured correlation assay keys: {configured_assays}"
        )
        return

    missing_columns = []
    for spec in matching_specs:
        for col in [spec.get("x"), spec.get("y")]:
            if col and col not in df.columns:
                missing_columns.append(col)

    if missing_columns:
        print(
            f"WARNING [{assay.upper()}]: correlation specs matched, but columns are missing: "
            f"{sorted(set(missing_columns))}"
        )
        return

    default_covariates = list(cfg.correlations_defaults.get("covariates", []) or [])
    corr_columns = []
    for spec in matching_specs:
        corr_columns += [spec.get("x"), spec.get("y")]
        corr_columns += list(spec.get("covariates", default_covariates) or [])
    df = mask_excluded_values(df, exclusions, assay=assay, columns=corr_columns, cfg=cfg)

    out_path = cfg.out_tables / f"{assay.upper()}_correlations.csv"

    # run_correlations expects the original nested YAML-like shape:
    # cfg["correlations"]["enabled"], cfg["correlations"]["specs"], etc.
    corr_cfg = {
        "correlations": {
            "enabled": getattr(cfg, "correlations_enabled", False),
            "defaults": cfg.correlations_defaults,
            "default_groupings": cfg.correlations_default_groupings,
            "correction": cfg.correlations_correction,
            "min_n": cfg.correlations_min_n,
            "specs": corr_specs,
        }
    }

    corr_df = run_correlations(
        assay=assay,
        df=df,
        cfg=corr_cfg,
        out_csv_path=None,
    )

    if corr_df.empty:
        print(
            f"WARNING [{assay.upper()}]: correlation specs matched but produced no rows. "
            f"This usually means all groups had n < min_n after dropping missing x/y. "
            f"min_n={corr_cfg['correlations']['min_n']}; matching specs={matching_specs}"
        )
        return

    write_table(corr_df, out_path, cfg)
    log_written(out_path)

def write_summary(
    df: pd.DataFrame,
    cfg,
    assay: str,
    exclusions: pd.DataFrame,
) -> None:
    between = cfg.between_factors

    counts = cell_counts(df, between)
    counts_path = cfg.out_tables / f"{assay.upper()}_cell_counts.csv"
    write_table(counts, counts_path, cfg)

    assay_upper = assay.upper()

    if exclusions is None or exclusions.empty:
        exclusion_lines = [
            "exclusion_file: none or empty",
            "n_exclusion_rows_applying_to_assay: 0",
        ]
    else:
        ex = exclusions.copy()
        assay_mask = ex["assay"].astype(str).str.upper().isin([assay_upper, "*"])
        ex_assay = ex.loc[assay_mask].copy()

        if ex_assay.empty:
            exclusion_lines = [
                "exclusion_file: loaded",
                "n_exclusion_rows_applying_to_assay: 0",
            ]
        else:
            exclusion_lines = [
                "exclusion_file: loaded",
                f"n_exclusion_rows_applying_to_assay: {len(ex_assay)}",
                "exclusions_applying_to_assay:",
            ]

            for _, row in ex_assay.iterrows():
                exclusion_lines.append(
                    "  - "
                    f"mouse_id={row.get('mouse_id', '')}, "
                    f"assay={row.get('assay', '')}, "
                    f"endpoint={row.get('endpoint', '')}, "
                    f"exclude_from={row.get('exclude_from', '')}, "
                    f"reason={row.get('reason', '')}"
                )

    exclusion_text = "\n".join(exclusion_lines)

    summary = textwrap.dedent(
        f"""
        ASSAY: {assay}
        n_rows_processed_pre_exclusion: {len(df)}
        n_mice_processed_pre_exclusion: {df[cfg.mouse_id_col].nunique()}

        {exclusion_text}

        wrote:
          {counts_path}
        """.strip()
    )

    log_path = cfg.out_logs / f"{assay.upper()}_stats_summary.txt"
    write_text(log_path, summary)

    print(summary)

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment-config", default="", help="Path to experiment YAML config")
    ap.add_argument("--assays", nargs="*", default=None)
    ap.add_argument(
        "--require-clean",
        action="store_true",
        help="Refuse to run if tracked files have uncommitted changes (use for results you will report)",
    )
    args = ap.parse_args()

    cfg = load_stats_config(args.config)
    exp_cfg = load_yaml(args.experiment_config)
    exclusions_path = (
        exp_cfg.get("paths", {}).get("metadata_exclusions", "")
        or exp_cfg.get("paths", {}).get("exclusions", "")
    )
    exclusions = load_exclusions(exclusions_path)

    exp_paths = exp_cfg.get("paths", {})

    cfg = replace(
        cfg,
        processed_dir=Path(exp_paths.get("processed_dir", cfg.processed_dir)),
        subjects_path=Path(exp_paths.get("metadata_subjects", cfg.subjects_path)),
        out_tables=Path(exp_paths.get("out_tables", cfg.out_tables)),
        out_logs=Path(exp_paths.get("out_logs", cfg.out_logs)),
        out_figures=Path(exp_paths.get("out_figures", cfg.out_figures)),
    )
    cfg = apply_assay_instance_aliases(cfg, exp_cfg)
    cfg = replace(cfg, trial_durations_s=trial_durations_from_experiment(exp_cfg))
    assays = cfg.assays if args.assays is None else [a.lower() for a in args.assays]

    subjects = read_subjects(cfg.subjects_path)

    base_out_tables = cfg.out_tables
    base_out_logs = cfg.out_logs
    base_out_figures = cfg.out_figures

    for assay in assays:
        assay_key = assay.upper()

        assay_cfg = replace(
            cfg,
            out_tables=base_out_tables.parent / assay_key / "tables",
            out_logs=base_out_logs.parent / assay_key / "logs",
            out_figures=base_out_figures.parent / assay_key / "figures",
        )

        assay_cfg.out_tables.mkdir(parents=True, exist_ok=True)
        assay_cfg.out_logs.mkdir(parents=True, exist_ok=True)
        assay_cfg.out_figures.mkdir(parents=True, exist_ok=True)

        prov = build_provenance(
            repo_root=REPO_ROOT,
            files={
                "stats_config": args.config,
                "experiment_config": args.experiment_config or None,
                "subjects": cfg.subjects_path,
                "exclusions": exclusions_path or None,
                "processed": assay_cfg.processed_dir / f"{assay_key}_processed.csv",
            },
            argv=sys.argv,
        )
        if prov["git"]["dirty"]:
            msg = f"uncommitted changes in tracked files: {prov['git']['dirty_files']}"
            if args.require_clean:
                raise SystemExit(f"ERROR: --require-clean set and {msg}")
            print(f"WARNING [{assay_key}]: {msg}; outputs cannot be traced to a commit.")
        assay_cfg = replace(assay_cfg, run_id=prov["run_id"])
        write_provenance(prov, assay_cfg.out_logs / f"{assay_key}_provenance.json")

        df = read_processed(assay_cfg.processed_dir, assay)
        df = prepare_analysis_df(df, subjects, assay_cfg, assay)

        run_primary_tests(df, assay_cfg, assay,exclusions)
        run_anova(df, assay_cfg, assay, exclusions)
        run_posthoc(df, assay_cfg, assay, exclusions)
        run_time_to_event_stage(df, assay_cfg, assay, exclusions)
        run_diagnostics(df, assay_cfg, assay, exclusions)
        run_exploratory(df, assay_cfg, assay, exclusions)

        write_summary(df, assay_cfg, assay, exclusions)

        print("\nDone.\n")


if __name__ == "__main__":
    main()