from __future__ import annotations
from dataclasses import replace

import argparse
from pathlib import Path
import textwrap
import yaml

import pandas as pd

from behaviour_stat_pipeline.stats.common import (
    load_stats_config,
    read_processed,
    read_subjects,
    merge_metadata,
    require_columns,
    coerce_categoricals,
    drop_single_level_factors,
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
from behaviour_stat_pipeline.stats.diagnostics import endpoint_diagnostics, rout_diagnostics
from behaviour_stat_pipeline.stats.corrections import adjust_pvalues


# -------------------------
# Utility
# -------------------------

def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")

def load_yaml(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    
def apply_assay_instance_aliases(cfg, exp_cfg: dict):
    """
    Allow assay instances such as OFT1/OFT2 to reuse endpoint/config
    definitions from their canonical assay type, e.g. OFT.

    This is runtime aliasing only: it does not modify the YAML file.
    """
    instances = exp_cfg.get("assay_instances", {}) or {}

    dict_attrs = [
        "within_factors",
        "primary_one_sample_nulls",
        "secondary_two_way_anova_endpoints",
        "diagnostics_endpoints",
    ]

    for instance_name, inst in instances.items():
        inst = inst or {}
        assay_type = str(inst.get("assay_type", instance_name))

        instance_key = instance_name.lower()
        type_key = assay_type.lower()

        for attr in dict_attrs:
            d = getattr(cfg, attr, None)
            if isinstance(d, dict) and type_key in d and instance_key not in d:
                d[instance_key] = d[type_key]

    return cfg
    
def log_written(path: Path) -> None:
    print(f"Wrote: {path}")
    
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


def run_primary_tests(df: pd.DataFrame, cfg, assay: str) -> None:
    """Confirmatory one-sample tests."""
    if not getattr(cfg, "primary_one_sample_enabled", False):
        return

    assay_nulls = cfg.primary_one_sample_nulls.get(assay, {})
    if not assay_nulls:
        return

    frames = []
    for endpoint, mu0 in assay_nulls.items():
        by_group = run_one_sample_by_group(
            df,
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
            df,
            assay=assay.upper(),
            endpoint=endpoint,
            mu0=float(mu0),
            id_col=cfg.mouse_id_col,
            run_ttest=cfg.primary_one_sample_run_ttest,
            run_wilcoxon=cfg.primary_one_sample_run_wilcoxon,
        )

        frames.append(pd.concat([by_group, pooled], ignore_index=True))

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cfg.out_tables / f"{assay.upper()}_primary_one_sample.csv", index=False)


def run_anova(df: pd.DataFrame, cfg, assay: str) -> pd.DataFrame | None:
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
        frames.append(
            run_two_way_anova(
                df,
                assay=assay.upper(),
                endpoint=ep,
                factors=cfg.secondary_two_way_anova_factors,
                include_interaction=cfg.secondary_two_way_anova_include_interaction,
                typ=cfg.secondary_two_way_anova_typ,
            )
        )

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
    aov.to_csv(out_path, index=False)
    log_written(out_path)

    return aov


def run_posthoc(df: pd.DataFrame, cfg, assay: str) -> None:
    posthoc_cfg = getattr(cfg, "secondary_two_way_anova_posthoc", {}) or {}
    if not posthoc_cfg:
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

    df_posthoc = run_posthoc_cellmeans_prism(
        df=df,
        assay=assay.upper(),
        endpoints=available_endpoints,
        factors=cfg.secondary_two_way_anova_factors,
        include_interaction=cfg.secondary_two_way_anova_include_interaction,
        factor_orders=getattr(cfg, "secondary_two_way_anova_factor_orders", {}) or {},
        selected_pairs=list(posthoc_cfg.get("selected_pairs", [])),
        correction=str(posthoc_cfg.get("correction", "sidak")),
    )

    df_posthoc.to_csv(cfg.out_tables / f"{assay.upper()}_secondary_posthoc_cellmeans.csv", index=False)
    out_path = cfg.out_tables / f"{assay.upper()}_secondary_posthoc_cellmeans.csv"
    df_posthoc.to_csv(out_path, index=False)
    log_written(out_path)


def run_diagnostics(df: pd.DataFrame, cfg, assay: str) -> None:
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
        diag_df.to_csv(cfg.out_tables / f"{assay.upper()}_endpoint_diagnostics_by_{suffix}.csv", index=False)
        flag_df.to_csv(cfg.out_tables / f"{assay.upper()}_flagged_points_madz_by_{suffix}.csv", index=False)

        rout_df = rout_diagnostics(
            df,
            assay=assay.upper(),
            endpoints=available_endpoints,
            group_cols=["group"],
            id_col=cfg.mouse_id_col,
            q=0.01,
            min_n=6,
        )

        rout_df.to_csv(cfg.out_tables / f"{assay.upper()}_flagged_points_rout_by_{suffix}.csv", index=False)
        log_written(cfg.out_tables / f"{assay.upper()}_endpoint_diagnostics_by_{suffix}.csv")
        log_written(cfg.out_tables / f"{assay.upper()}_flagged_points_madz_by_{suffix}.csv")
        log_written(cfg.out_tables / f"{assay.upper()}_flagged_points_rout_by_{suffix}.csv")

def run_exploratory(df: pd.DataFrame, cfg, assay: str) -> None:
    if not cfg.correlations_enabled:
        return

    out_path = cfg.out_tables / f"{assay.upper()}_correlations.csv"
    run_correlations(
        assay=assay,
        df=df,
        cfg=cfg.__dict__,
        out_csv_path=str(out_path),
    )
    log_written(out_path)

def write_summary(df: pd.DataFrame, cfg, assay: str) -> None:
    between = cfg.between_factors

    counts = cell_counts(df, between)
    counts_path = cfg.out_tables / f"{assay.upper()}_cell_counts.csv"
    counts.to_csv(counts_path, index=False)

    summary = textwrap.dedent(
        f"""
        ASSAY: {assay}
        n_rows: {len(df)}
        n_mice: {df[cfg.mouse_id_col].nunique()}

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
    args = ap.parse_args()

    cfg = load_stats_config(args.config)
    exp_cfg = load_yaml(args.experiment_config)

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
    assays = cfg.assays if args.assays is None else [a.lower() for a in args.assays]

    cfg.out_tables.mkdir(parents=True, exist_ok=True)
    cfg.out_logs.mkdir(parents=True, exist_ok=True)
    cfg.out_figures.mkdir(parents=True, exist_ok=True)

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

        df = read_processed(assay_cfg.processed_dir, assay)
        df = prepare_analysis_df(df, subjects, assay_cfg, assay)

        run_primary_tests(df, assay_cfg, assay)
        run_anova(df, assay_cfg, assay)
        run_posthoc(df, assay_cfg, assay)
        run_diagnostics(df, assay_cfg, assay)
        run_exploratory(df, assay_cfg, assay)

        write_summary(df, assay_cfg, assay)

        print("\nDone.\n")


if __name__ == "__main__":
    main()