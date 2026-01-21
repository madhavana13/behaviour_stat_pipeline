from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

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
from behaviour_stat_pipeline.stats.diagnostics import endpoint_diagnostics
from behaviour_stat_pipeline.stats.endpoints.one_sample import run_one_sample_by_group
from behaviour_stat_pipeline.stats.endpoints.two_way_anova import run_two_way_anova
from behaviour_stat_pipeline.stats.endpoints.simple_effects import run_simple_effects, SimpleEffectsSpec
from behaviour_stat_pipeline.stats.corrections import adjust_pvalues



def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


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
        # --- Primary one-sample endpoints (config-driven) ---
        if getattr(cfg, "primary_one_sample_enabled", False):
            assay_nulls = cfg.primary_one_sample_nulls.get(assay, {})
            if assay_nulls:
                frames = []
                for endpoint, mu0 in assay_nulls.items():
                    frames.append(
                        run_one_sample_by_group(
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
                    )
                out = pd.concat(frames, ignore_index=True)
                out.to_csv(cfg.out_tables / f"{assay.upper()}_primary_one_sample.csv", index=False)

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

                aov_all.to_csv(cfg.out_tables / f"{assay.upper()}_secondary_two_way_anova.csv", index=False)

        # --- Post hoc: Prism-exact selected cell-mean comparisons (pooled MSE + Sidak) ---

        def significance_label(p):
            if not np.isfinite(p):
                return ""
            if p < 1e-4:
                return "≤0.0001"
            elif p < 1e-3:
                return "≤0.001"
            elif p < 1e-2:
                return "≤0.01"
            elif p < 5e-2:
                return "≤0.05"
            else:
                return "ns"

        posthoc = getattr(cfg, "secondary_two_way_anova_posthoc", None)
        if posthoc and posthoc.get("enabled", False):

            method = str(posthoc.get("method", ""))
            if method != "cell_means_prism":
                # If you still want to support other methods later, handle them here.
                # For now, only Prism-exact cell-means is implemented.
                raise ValueError(f"{assay}: unsupported posthoc method '{method}'. Expected 'cell_means_prism'.")

            # Factors and ordering (no hardcoded group labels)
            factors = cfg.secondary_two_way_anova_factors  # e.g. ["group", "sex"]
            if len(factors) != 2:
                raise ValueError(f"{assay}: cell_means_prism requires exactly 2 factors, got {factors}")
            f1, f2 = factors

            factor_orders = getattr(cfg, "secondary_two_way_anova_factor_orders", {}) or {}

            levels_by_factor: dict[str, list[str]] = {}
            for factor in factors:
                observed = merged[factor].dropna().astype(str).unique().tolist()
                order = factor_orders.get(factor, [])

                if order:
                    # keep only provided levels that are actually observed
                    levels = [str(x) for x in order if str(x) in set(observed)]
                else:
                    # deterministic inference
                    levels = sorted(set(map(str, observed)))

                if len(levels) != 2:
                    raise ValueError(
                        f"{assay}: posthoc requires exactly 2 levels for factor '{factor}', "
                        f"found {len(levels)} levels: {levels}"
                    )

                levels_by_factor[factor] = levels

            # Decode YAML cell specs like {group: 0, sex: 1} to {group: "control", sex: "male"}
            def _decode_cell(cell_spec: dict) -> dict[str, str]:
                out: dict[str, str] = {}
                for factor in factors:
                    if factor not in cell_spec:
                        raise KeyError(f"{assay}: selected_pairs cell spec missing key '{factor}': {cell_spec}")
                    idx = int(cell_spec[factor])
                    out[factor] = levels_by_factor[factor][idx]
                return out

            selected_pairs = list(posthoc.get("selected_pairs", []))
            if not selected_pairs:
                raise ValueError(f"{assay}: posthoc.method=cell_means_prism but no selected_pairs provided in config.")

            corr = str(posthoc.get("correction", "sidak")).lower()
            if corr != "sidak":
                raise ValueError(f"{assay}: cell_means_prism currently supports correction='sidak' only, got '{corr}'.")

            include_interaction = bool(cfg.secondary_two_way_anova_include_interaction)

            # Patsy-safe quoting (column names may have spaces)
            def _Q(name: str) -> str:
                return f'Q("{name}")'

            # Precompute K for Sidak within each endpoint (Prism: family = selected comparisons for that endpoint)
            K = len(selected_pairs)

            posthoc_rows = []

            for endpoint in cfg.secondary_two_way_anova_endpoints.get(assay, []):
                # Fit the SAME OLS model as the ANOVA step (Type II affects ANOVA table, but posthoc uses pooled MSE/df_resid)
                needed = [endpoint, f1, f2]
                d = merged.dropna(subset=needed).copy()

                # ensure factors have 2 levels in this endpoint subset
                if d[f1].nunique(dropna=False) < 2 or d[f2].nunique(dropna=False) < 2:
                    # write a skipped row to keep pipeline transparent
                    posthoc_rows.append({
                        "assay": assay.upper(),
                        "endpoint": endpoint,
                        "pair_name": "SKIPPED",
                        "cell_a": "",
                        "cell_b": "",
                        "mean_a": np.nan,
                        "mean_b": np.nan,
                        "n_a": 0,
                        "n_b": 0,
                        "diff_a_minus_b": np.nan,
                        "mse": np.nan,
                        "se_pooled": np.nan,
                        "t": np.nan,
                        "df_error": np.nan,
                        "p_value": np.nan,
                        "p_value_sidak": np.nan,
                        "reason": f"factor has <2 levels in subset (after dropna on {needed})",
                    })
                    continue

                y = _Q(endpoint)
                f1q = _Q(f1)
                f2q = _Q(f2)

                if include_interaction:
                    formula = f"{y} ~ C({_Q(f1)}) * C({_Q(f2)})"
                else:
                    formula = f"{y} ~ C({_Q(f1)}) + C({_Q(f2)})"

                model = smf.ols(formula, data=d, eval_env=0).fit()
                mse = float(model.mse_resid)
                df_error = float(model.df_resid)

                # Compute each selected pair using Prism pooled-MSE t-test on CELL MEANS
                for pair in selected_pairs:
                    pair_name = str(pair.get("name", ""))

                    cell_a = _decode_cell(pair["a"])
                    cell_b = _decode_cell(pair["b"])


                    # subset rows for each cell
                    mask_a = np.ones(len(d), dtype=bool)
                    mask_b = np.ones(len(d), dtype=bool)
                    for factor in factors:
                        mask_a &= (d[factor].astype(str) == cell_a[factor])
                        mask_b &= (d[factor].astype(str) == cell_b[factor])

                    ya = d.loc[mask_a, endpoint].astype(float)
                    yb = d.loc[mask_b, endpoint].astype(float)

                    n_a = int(ya.shape[0])
                    n_b = int(yb.shape[0])

                    mean_a = float(ya.mean()) if n_a > 0 else np.nan
                    mean_b = float(yb.mean()) if n_b > 0 else np.nan
                    diff = mean_a - mean_b if (n_a > 0 and n_b > 0) else np.nan

                    # --- Prism-style presentation (cosmetic only; does NOT affect p-values) ---
                    def _level_idx(factor: str, level: str) -> int:
                        return levels_by_factor[factor].index(str(level))

                    score_a = sum(_level_idx(f, cell_a[f]) for f in factors)
                    score_b = sum(_level_idx(f, cell_b[f]) for f in factors)

                    if score_a >= score_b:
                        cell_1, cell_2 = cell_a, cell_b
                        mean_1, mean_2 = mean_a, mean_b
                    else:
                        cell_1, cell_2 = cell_b, cell_a
                        mean_1, mean_2 = mean_b, mean_a

                    diff_prism = (mean_1 - mean_2) if (np.isfinite(mean_1) and np.isfinite(mean_2)) else np.nan

                    def _pretty(cell: dict[str, str]) -> str:
                        return ":".join(str(cell[f]).capitalize() for f in factors)

                    contrast_label = f"{_pretty(cell_1)} vs {_pretty(cell_2)}"

                    # Prism pooled SE uses ANOVA MSE (not cell variances)
                    se = float(np.sqrt(mse * (1.0 / n_a + 1.0 / n_b))) if (n_a > 0 and n_b > 0) else np.nan
                    t_stat = float(diff / se) if (se is not None and np.isfinite(se) and se > 0) else np.nan
                    p = float(2 * st.t.sf(np.abs(t_stat), df_error)) if np.isfinite(t_stat) else np.nan

                    # Sidak adjustment across the K selected comparisons (per endpoint)
                    p_sidak = float(1 - (1 - p) ** K) if (p is not None and np.isfinite(p)) else np.nan

                    sig_label = significance_label(p_sidak)

                    posthoc_rows.append({
                        "assay": assay.upper(),
                        "endpoint": endpoint,
                        "pair_name": pair_name,
                        "contrast_label": contrast_label,
                        "diff_prism_direction": diff_prism,
                        "cell_a": ";".join([f"{k}={cell_a[k]}" for k in factors]),
                        "cell_b": ";".join([f"{k}={cell_b[k]}" for k in factors]),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "n_a": n_a,
                        "n_b": n_b,
                        "diff_a_minus_b": diff,
                        "mse": mse,
                        "se_pooled": se,
                        "t": t_stat,
                        "df_error": df_error,
                        "p_value": p,
                        "p_value_sidak": p_sidak,
                        "significance_label": sig_label

                    })

            posthoc_df = pd.DataFrame(posthoc_rows)
            posthoc_df.to_csv(cfg.out_tables / f"{assay.upper()}_secondary_posthoc_cellmeans.csv", index=False)

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
