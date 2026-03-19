from __future__ import annotations

import argparse
from pathlib import Path
import textwrap
import yaml
import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.formula.api as smf


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
from behaviour_stat_pipeline.stats.diagnostics import endpoint_diagnostics, rout_diagnostics
from behaviour_stat_pipeline.stats.endpoints.one_sample import (
    run_one_sample_by_group,
    run_one_sample_pooled,
)
from behaviour_stat_pipeline.stats.endpoints.two_way_anova import run_two_way_anova, _significance_label
from behaviour_stat_pipeline.stats.endpoints.simple_effects import run_simple_effects, SimpleEffectsSpec
from behaviour_stat_pipeline.stats.common import load_stats_config
from behaviour_stat_pipeline.stats.corrections import adjust_pvalues
from behaviour_stat_pipeline.stats.endpoints.correlations import run_correlations


def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")

def run_sensitivity_low_fam_explore(
    merged: pd.DataFrame,
    assay: str,
    cfg,
) -> None:
    """
    Sensitivity analysis: exclude mice with low FAM exploration (<20s).
    Does NOT modify main pipeline outputs.
    Writes separate *_sens_lowfamexplore_excluded.csv files.
    """

    if assay.upper() != "NOR":
        return

    if "fam_explore_low_explore" not in merged.columns:
        print(f"[{assay.upper()}] Skipping sensitivity: flag not found")
        return

    df = merged.loc[~merged["fam_explore_low_explore"]].copy()

    print(
        f"[{assay.upper()}] Sensitivity (exclude low FAM explore): "
        f"{len(merged)} → {len(df)} mice"
    )

    suffix = "_sens_lowfamexplore_excluded"

    # --- One-sample ---
    if getattr(cfg, "primary_one_sample_enabled", False):
        assay_nulls = cfg.primary_one_sample_nulls.get(assay, {})
        if assay_nulls:
            frames = []
            for endpoint, mu0 in assay_nulls.items():
                frames.append(
                    run_one_sample_by_group(
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
                )
            out = pd.concat(frames, ignore_index=True)
            out.to_csv(
                cfg.out_tables / f"{assay.upper()}_primary_one_sample{suffix}.csv",
                index=False,
            )

    # --- ANOVA ---
    if getattr(cfg, "secondary_two_way_anova_enabled", False):
        ep_list = cfg.secondary_two_way_anova_endpoints.get(assay, [])
        if ep_list:
            frames = []
            for ep in ep_list:
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
            aov_all = pd.concat(frames, ignore_index=True)

            method = cfg.secondary_two_way_anova_correction
            if method.lower() != "none":
                aov_all["p_adj"] = aov_all.groupby("term", dropna=False, observed=False)["p_value"].transform(
                    lambda s: adjust_pvalues(s, method=method)
                )
            else:
                aov_all["p_adj"] = aov_all["p_value"]

            aov_all["significance_label"] = aov_all["p_adj"].apply(_significance_label)

            aov_all.to_csv(
                cfg.out_tables / f"{assay.upper()}_secondary_two_way_anova{suffix}.csv",
                index=False,
            )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/stats.yaml")
    ap.add_argument("--assays", nargs="*", default=None, help="Optional subset, e.g. nor vit")
    args = ap.parse_args()

    cfg = load_stats_config(args.config)

    assays = cfg.assays if args.assays is None else args.assays

    # Create output dirs (no manual steps)
    cfg.out_tables.mkdir(parents=True, exist_ok=True)
    cfg.out_figures.mkdir(parents=True, exist_ok=True)
    cfg.out_logs.mkdir(parents=True, exist_ok=True)

    subjects = read_subjects(cfg.subjects_path)

    for assay in assays:
        df = read_processed(cfg.processed_dir, assay)

        # Use processed as the analysis-ready contract.
        # Only merge metadata if required factor columns are missing.
        within = cfg.within_factors.get(assay, [])
        between = cfg.between_factors
        blocking = cfg.blocking_factors

        needed = [cfg.mouse_id_col] + between + within + blocking
        missing_in_processed = [c for c in needed if c not in df.columns]

        if missing_in_processed:
            merged = merge_metadata(df, subjects, mouse_id_col=cfg.mouse_id_col)
        else:
            merged = df.copy()

        # Determine factor roles by name
        within = cfg.within_factors.get(assay, [])
        between = cfg.between_factors
        blocking = cfg.blocking_factors

        # Validate columns exist (after join)
        require_columns(merged, [cfg.mouse_id_col], f"{assay}: merged")
        require_columns(merged, between, f"{assay}: between_factors")
        require_columns(merged, within, f"{assay}: within_factors")
        require_columns(merged, blocking, f"{assay}: blocking_factors") if blocking else None

        # Coerce to categoricals (important for formulas later)
        merged = coerce_categoricals(merged, between + within + blocking)
        run_sensitivity_low_fam_explore(merged, assay, cfg)
        # --- Primary one-sample endpoints (config-driven) ---
        if getattr(cfg, "primary_one_sample_enabled", False):
            assay_nulls = cfg.primary_one_sample_nulls.get(assay, {})
            if assay_nulls:
                frames = []
                for endpoint, mu0 in assay_nulls.items():
                    by_group = run_one_sample_by_group(
                        merged,
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
                        merged,
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
        
        # --- Secondary/exploratory: correlations / regression (config-driven) ---
        # Use the validated StatsConfig for gating (no dict-style .get on dataclass).
        cfg = load_stats_config(args.config)

        with Path(args.config).open("r", encoding="utf-8") as f:
            cfg_raw = yaml.safe_load(f)

        # --- Diagnostics (config-driven; no copy-paste per assay) ---
        if getattr(cfg, "diagnostics_enabled", False):
            ep_list = cfg.diagnostics_endpoints.get(assay, [])
            if ep_list:
                for grouping in cfg.diagnostics_groupings:
                    diag_df, flag_df = endpoint_diagnostics(
                        merged,
                        assay=assay.upper(),
                        endpoints=ep_list,
                        group_cols=grouping,
                        id_col=cfg.mouse_id_col,
                        min_n_test=cfg.diagnostics_min_n_test,
                        var_ratio_thresh=cfg.diagnostics_var_ratio_thresh,
                        mad_z_thresh=cfg.diagnostics_mad_z_thresh,
                    )

                    suffix = "_".join(grouping)
                    diag_df.to_csv(cfg.out_tables / f"{assay.upper()}_endpoint_diagnostics_by_{suffix}.csv", index=False)
                    flag_df.to_csv(cfg.out_tables / f"{assay.upper()}_flagged_points_madz_by_{suffix}.csv", index=False)
                    rout_flag_df = rout_diagnostics(
                        merged,
                        assay=assay.upper(),
                        endpoints=ep_list,
                        group_cols=["group"],          # not used by ROUT, but API symmetry
                        id_col=cfg.mouse_id_col,
                        q=0.01,                       # Prism-like default; later we’ll config-drive this
                        min_n=6,
                        keep_all=False,
                    )

                    rout_flag_df.to_csv(
                        cfg.out_tables / f"{assay.upper()}_flagged_points_rout_by_{suffix}.csv",
                        index=False,
                    )

        # --- Secondary/exploratory: two-way ANOVA (config-driven) ---
        if getattr(cfg, "secondary_two_way_anova_enabled", False):
            ep_list = cfg.secondary_two_way_anova_endpoints.get(assay, [])
            if ep_list:
                frames = []
                for ep in ep_list:
                    frames.append(
                        run_two_way_anova(
                            merged,
                            assay=assay.upper(),
                            endpoint=ep,
                            factors=cfg.secondary_two_way_anova_factors,
                            include_interaction=cfg.secondary_two_way_anova_include_interaction,
                            typ=cfg.secondary_two_way_anova_typ,
                        )
                    )
                aov_all = pd.concat(frames, ignore_index=True)

                # Correct across endpoints for each term separately (recommended)
                # e.g., adjust all "C(group)" p-values across endpoints, separately from interaction p-values
                method = cfg.secondary_two_way_anova_correction
                if method.lower() != "none":
                    aov_all["p_adj"] = aov_all.groupby("term", dropna=False, observed=False)["p_value"].transform(
                        lambda s: adjust_pvalues(s, method=method)
                    )
                else:
                    aov_all["p_adj"] = aov_all["p_value"]
                # Correct across endpoints for each term separately (recommended)
                method = cfg.secondary_two_way_anova_correction
                if method.lower() != "none":
                    aov_all["p_adj"] = aov_all.groupby("term", dropna=False, observed=False)["p_value"].transform(
                        lambda s: adjust_pvalues(s, method=method)
                    )
                else:
                    aov_all["p_adj"] = aov_all["p_value"]

                # ✅ ADD THIS HERE (use p_adj, not p_value)
                aov_all["significance_label"] = aov_all["p_adj"].apply(_significance_label)

                aov_all.to_csv(cfg.out_tables / f"{assay.upper()}_secondary_two_way_anova.csv", index=False)

                aov_all.to_csv(cfg.out_tables / f"{assay.upper()}_secondary_two_way_anova.csv", index=False)

        # --- Secondary/exploratory: posthoc cell means (Prism-style) ---
        from behaviour_stat_pipeline.stats.endpoints.posthoc_cellmeans_prism import run_posthoc_cellmeans_prism
        posthoc_config = getattr(cfg, "secondary_two_way_anova_posthoc", {}) or {}
        if posthoc_config:
            posthoc_df = run_posthoc_cellmeans_prism(
            df=merged,
            assay=assay.upper(),
            endpoints=cfg.secondary_two_way_anova_endpoints.get(assay, []),
            factors=cfg.secondary_two_way_anova_factors,
            include_interaction=cfg.secondary_two_way_anova_include_interaction,
            factor_orders=getattr(cfg, "secondary_two_way_anova_factor_orders", {}) or {},
            selected_pairs=list(posthoc_config.get("selected_pairs", [])),
            correction=str(posthoc_config.get("correction", "sidak")),
            )
            posthoc_df.to_csv(cfg.out_tables / f"{assay.upper()}_secondary_posthoc_cellmeans.csv", index=False)
      
        # --- correlations / regression (config-driven) ---
        if cfg.correlations_enabled:
            out_csv = str(cfg.out_tables / f"{assay.upper()}_correlations.csv")

            # ensure processed df exists in this scope (use the one you already use for ANOVA etc.)
            # df_processed should already exist; if not, read it here deterministically:
            df_processed = read_processed(cfg.processed_dir, assay)

            run_correlations(
                assay=assay,
                df=df_processed,
                cfg=cfg_raw,          # IMPORTANT: raw YAML dict, not cfg.to_dict()
                out_csv_path=out_csv,
            )

            print(f"[{assay.upper()}] Wrote correlations: {out_csv}")

        # Drop single-level factors (subset robustness)
        kept_between, dropped_between = drop_single_level_factors(merged, between)
        kept_within, dropped_within = drop_single_level_factors(merged, within)
        kept_blocking, dropped_blocking = drop_single_level_factors(merged, blocking)

        # Cell counts for between-subject design
        counts = cell_counts(merged, kept_between)
        counts_path = cfg.out_tables / f"{assay.upper()}_cell_counts.csv"
        counts.to_csv(counts_path, index=False)

        # Basic summary log
        n_rows = len(merged)
        n_mice = merged[cfg.mouse_id_col].nunique(dropna=True)
       

        def fmt_drop(dropped):
            if not dropped:
                return "None"
            return "; ".join([f"{c} (levels observed: {lvls})" for c, lvls in dropped])

        summary = textwrap.dedent(
            f"""
            ASSAY: {assay}
            processed_file_rows: {n_rows}
            unique_mice: {n_mice}

            factor_roles (from config):
              between_factors: {between}
              within_factors:  {within}
              blocking_factors:{blocking}

            factors_kept (>=2 observed levels):
              between: {kept_between}
              within:  {kept_within}
              blocking:{kept_blocking}

            factors_dropped (single-level in this subset):
              between: {fmt_drop(dropped_between)}
              within:  {fmt_drop(dropped_within)}
              blocking:{fmt_drop(dropped_blocking)}

            wrote:
              {counts_path.as_posix()}
            """.strip()
            + "\n"
        )

        log_path = cfg.out_logs / f"{assay.upper()}_stats_summary.txt"
        write_text(log_path, summary)

        print(summary)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
