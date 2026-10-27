from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, ttest_rel, wilcoxon
import statsmodels.formula.api as smf


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def mean_ci95(x: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    mean = x.mean()
    if n == 1:
        return mean, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    ci = 1.96 * se
    return mean, ci


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def build_group_color_map(groups: list[str]) -> dict[str, str]:
    """Deterministic, data-driven color map with no hardcoded group names."""
    return {g: f"C{i}" for i, g in enumerate(sorted(groups))}


def safe_slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment-config",
        required=True,
        help="Path to experiment YAML config",
    )
    ap.add_argument(
        "--assay",
        default="vit",
        help="Assay name; default='vit'",
    )
    ap.add_argument(
        "--before-col",
        default="hand__Before__time_in_right_lower_quadrant",
        help="Wide-format column for the 'before' hand phase endpoint",
    )
    ap.add_argument(
        "--after-col",
        default="hand__After__time_in_right_lower_quadrant",
        help="Wide-format column for the 'after' hand phase endpoint",
    )
    ap.add_argument(
        "--ylabel",
        default="Time in right lower quadrant (s)",
        help="Y-axis label for plots",
    )
    args = ap.parse_args()

    exp_cfg = load_yaml(args.experiment_config)
    exp_paths = exp_cfg.get("paths", {})

    processed_dir = Path(exp_paths["processed_dir"])
    # exploratory output is kept apart from pipeline results
    out_figures = Path(exp_paths["out_dir"]) / "exploratory" / "figures"
    out_tables = Path(exp_paths["out_dir"]) / "exploratory" / "tables"

    assay = args.assay.lower()
    assay_prefix = assay.upper()
    data_path = processed_dir / f"{assay_prefix}_processed.csv"

    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    required = ["mouse_id", args.before_col, args.after_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    endpoint_slug = safe_slug(args.after_col.replace("hand__After__", ""))

    # -------------------------------------------------
    # Basic paired data prep
    # -------------------------------------------------
    df = df.copy()
    df["before"] = pd.to_numeric(df[args.before_col], errors="coerce")
    df["after"] = pd.to_numeric(df[args.after_col], errors="coerce")
    df["delta_after_minus_before"] = df["after"] - df["before"]

    paired = df.dropna(subset=["before", "after"]).copy()

    if paired.empty:
        raise ValueError("No complete paired observations after dropping missing before/after values.")

    # -------------------------------------------------
    # Flag bottom 10% performers based on baseline ("before")
    # -------------------------------------------------
    cutoff = paired["before"].quantile(0.10)
    paired["low_10pct_before"] = paired["before"] <= cutoff

    print("\n=== Low-performer flag ===")
    print(f"10th percentile cutoff (before): {cutoff:.6f}")
    print(f"Flagged animals: {int(paired['low_10pct_before'].sum())} / {len(paired)}")

    # -------------------------------------------------
    # Summaries
    # -------------------------------------------------
    before_mean, before_ci = mean_ci95(paired["before"])
    after_mean, after_ci = mean_ci95(paired["after"])
    delta_mean, delta_ci = mean_ci95(paired["delta_after_minus_before"])

    print("\n=== Paired summary ===")
    print(f"N paired: {len(paired)}")
    print(f"Before mean: {before_mean:.6f}")
    print(f"After mean: {after_mean:.6f}")
    print(f"Delta mean (After - Before): {delta_mean:.6f}")

    # paired t-test
    t_res = ttest_rel(paired["after"], paired["before"], nan_policy="omit")
    print("\n=== Paired t-test (After vs Before) ===")
    print(t_res)

    # Wilcoxon signed-rank
    try:
        w_res = wilcoxon(paired["after"], paired["before"])
        print("\n=== Wilcoxon signed-rank test (After vs Before) ===")
        print(w_res)
        w_stat = float(w_res.statistic)
        w_p = float(w_res.pvalue)
    except ValueError:
        print("\n=== Wilcoxon signed-rank test ===")
        print("Could not compute Wilcoxon (likely all paired differences are zero).")
        w_stat = np.nan
        w_p = np.nan

    # association between baseline and change
    rho_delta, p_delta = spearmanr(
        paired["before"],
        paired["delta_after_minus_before"],
        nan_policy="omit",
    )
    print("\n=== Spearman: baseline vs delta ===")
    print(f"rho = {rho_delta:.6f}, p = {p_delta:.6f}")

    # -------------------------------------------------
    # Write tables
    # -------------------------------------------------
    summary_rows = [
        {"metric": "n_paired", "value": len(paired)},
        {"metric": "before_mean", "value": before_mean},
        {"metric": "before_ci95", "value": before_ci},
        {"metric": "after_mean", "value": after_mean},
        {"metric": "after_ci95", "value": after_ci},
        {"metric": "delta_mean_after_minus_before", "value": delta_mean},
        {"metric": "delta_ci95_after_minus_before", "value": delta_ci},
        {"metric": "paired_t_stat", "value": float(t_res.statistic)},
        {"metric": "paired_t_p", "value": float(t_res.pvalue)},
        {"metric": "wilcoxon_stat", "value": w_stat},
        {"metric": "wilcoxon_p", "value": w_p},
        {"metric": "spearman_rho_before_vs_delta", "value": rho_delta},
        {"metric": "spearman_p_before_vs_delta", "value": p_delta},
    ]
    pd.DataFrame(summary_rows).to_csv(
        out_tables / f"{assay_prefix}_{endpoint_slug}_before_after_summary.csv",
        index=False,
    )

    paired[
        ["mouse_id", "before", "after", "delta_after_minus_before", "low_10pct_before"]
    ].to_csv(
        out_tables / f"{assay_prefix}_{endpoint_slug}_paired_values.csv",
        index=False,
    )

    # -------------------------------------------------
    # Scatter: before vs after
    # both axes should start at 0
    # -------------------------------------------------
    max_val = np.nanmax([paired["before"].max(), paired["after"].max()])
    lim_top = max_val * 1.1 if pd.notna(max_val) else 1.0

    plt.figure(figsize=(6, 5))

    high = paired.loc[~paired["low_10pct_before"]]
    low = paired.loc[paired["low_10pct_before"]]

    plt.scatter(
        high["before"],
        high["after"],
        alpha=0.5,
        label="Other animals",
    )
    plt.scatter(
        low["before"],
        low["after"],
        alpha=0.95,
        marker="o",
        s=70,
        label="Bottom 10% at baseline",
    )

    plt.plot([0, lim_top], [0, lim_top], linestyle="--", linewidth=1)
    plt.axvline(cutoff, linestyle=":", linewidth=1)

    plt.xlabel(f"Before: {args.ylabel}")
    plt.ylabel(f"After: {args.ylabel}")
    plt.title("Before vs after (bottom 10% baseline highlighted)")
    plt.xlim(left=0, right=lim_top)
    plt.ylim(bottom=0, top=lim_top)
    plt.legend(frameon=False)
    savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_before_vs_after_low10.png")

    # -------------------------------------------------
    # Histogram: delta
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.hist(paired["delta_after_minus_before"].dropna(), bins=12)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(f"Delta (After - Before): {args.ylabel}")
    plt.ylabel("Count")
    plt.title("Distribution of paired change")
    savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_delta_histogram.png")

    # -------------------------------------------------
    # Long data for trajectories
    # -------------------------------------------------
    id_vars = ["mouse_id", "low_10pct_before"]
    if "group" in paired.columns:
        id_vars.append("group")
    if "sex" in paired.columns:
        id_vars.append("sex")

    df_long = paired[id_vars + ["before", "after"]].melt(
        id_vars=id_vars,
        value_vars=["before", "after"],
        var_name="timepoint",
        value_name="value",
    )

    df_long["timepoint"] = df_long["timepoint"].map(
        {
            "before": "Before",
            "after": "After",
        }
    )

    x_map = {"Before": 0, "After": 1}
    df_long["x"] = df_long["timepoint"].map(x_map)

    # -------------------------------------------------
    # Trajectory: all animals
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    for _, sub in df_long.groupby("mouse_id"):
        sub = sub.sort_values("x")
        low = bool(sub["low_10pct_before"].iloc[0])

        plt.plot(
            sub["x"],
            sub["value"],
            alpha=0.75 if low else 0.20,
            linewidth=2.0 if low else 1.0,
        )

    plt.xticks([0, 1], ["Before", "After"])
    plt.xlabel("Hand phase")
    plt.ylabel(args.ylabel)
    plt.title("Animal trajectories (bottom 10% baseline highlighted)")
    plt.ylim(bottom=0)
    savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_trajectory_low10.png")

    # -------------------------------------------------
    # Spaghetti + mean ± 95% CI
    # -------------------------------------------------
    traj_rows = []
    for tp, sub in df_long.groupby("timepoint", sort=False):
        mean, ci95 = mean_ci95(sub["value"])
        traj_rows.append({"timepoint": tp, "mean": mean, "ci95": ci95})

    traj_summary = pd.DataFrame(traj_rows)
    traj_summary["x"] = traj_summary["timepoint"].map(x_map)
    traj_summary = traj_summary.sort_values("x")

    plt.figure(figsize=(6, 5))
    for _, sub in df_long.groupby("mouse_id"):
        sub = sub.sort_values("x")
        plt.plot(sub["x"], sub["value"], alpha=0.15)

    plt.plot(traj_summary["x"], traj_summary["mean"], linewidth=2)
    plt.errorbar(
        traj_summary["x"],
        traj_summary["mean"],
        yerr=traj_summary["ci95"],
        fmt="none",
        capsize=4,
        linewidth=1.5,
    )
    plt.xticks([0, 1], ["Before", "After"])
    plt.xlabel("Hand phase")
    plt.ylabel(args.ylabel)
    plt.title("Trajectory with mean ± 95% CI")
    plt.ylim(bottom=0)
    savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_trajectory_mean_ci.png")

    traj_summary.to_csv(
        out_tables / f"{assay_prefix}_{endpoint_slug}_trajectory_mean_ci.csv",
        index=False,
    )

    # -------------------------------------------------
    # Group-aware plots and models
    # -------------------------------------------------
    if "group" not in df_long.columns:
        print("WARNING: 'group' column not found. Skipping group-based plots and models.")
    else:
        groups = sorted(df_long["group"].dropna().unique())
        color_map = build_group_color_map(groups)

        # individual trajectories colored by group
        plt.figure(figsize=(7, 5))
        for _, sub in df_long.groupby("mouse_id"):
            sub = sub.sort_values("x")
            g = sub["group"].iloc[0]
            plt.plot(
                sub["x"],
                sub["value"],
                color=color_map.get(g, "black"),
                alpha=0.5,
            )

        plt.xticks([0, 1], ["Before", "After"])
        plt.xlabel("Hand phase")
        plt.ylabel(args.ylabel)
        plt.title("Animal trajectories (colored by group)")
        plt.ylim(bottom=0)

        handles = [plt.Line2D([0], [0], color=color_map[g], label=g) for g in groups]
        plt.legend(handles=handles, title="Group", frameon=False)
        savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_trajectory_by_group.png")

        # mean trajectory by group
        plt.figure(figsize=(6, 5))
        for g in groups:
            sub = df_long[df_long["group"] == g]
            means = (
                sub.groupby("timepoint")["value"]
                .mean()
                .reindex(["Before", "After"])
            )
            plt.plot(
                [0, 1],
                means.values,
                label=g,
                linewidth=2,
                color=color_map[g],
            )

        plt.xticks([0, 1], ["Before", "After"])
        plt.xlabel("Hand phase")
        plt.ylabel(args.ylabel)
        plt.title("Mean trajectory by group")
        plt.legend(frameon=False)
        plt.ylim(bottom=0, top=df_long["value"].max() * 1.1)
        savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_mean_trajectory_by_group.png")

        # separate panel per group
        for g in groups:
            sub_g = df_long[df_long["group"] == g].copy()

            plt.figure(figsize=(7, 5))
            for _, sub in sub_g.groupby("mouse_id"):
                sub = sub.sort_values("x")
                plt.plot(
                    sub["x"],
                    sub["value"],
                    color=color_map[g],
                    alpha=0.4,
                )

            plt.xticks([0, 1], ["Before", "After"])
            plt.xlabel("Hand phase")
            plt.ylabel(args.ylabel)
            plt.title(f"Trajectory: {g}")
            plt.ylim(bottom=0)
            savefig(out_figures / f"{assay_prefix}_{endpoint_slug}_trajectory_group_{safe_slug(g)}.png")

        # repeated-measures style model
        # note: OLS ignores within-mouse dependence; MixedLM is the inferential model here
        df_long["timepoint"] = pd.Categorical(df_long["timepoint"], categories=["Before", "After"], ordered=True)

        ols_res = smf.ols(
            "value ~ group * timepoint",
            data=df_long,
        ).fit()
        print("\n=== OLS: value ~ group * timepoint ===")
        print(ols_res.summary())

        mixed_res = smf.mixedlm(
            "value ~ group * timepoint",
            data=df_long,
            groups=df_long["mouse_id"],
        ).fit()
        print("\n=== MixedLM: value ~ group * timepoint + (1|mouse_id) ===")
        print(mixed_res.summary())

        # save model coefficient tables
        ols_res.summary2().tables[1].to_csv(
            out_tables / f"{assay_prefix}_{endpoint_slug}_ols_coefficients.csv"
        )
        mixed_res.summary().tables[1].to_csv(
            out_tables / f"{assay_prefix}_{endpoint_slug}_mixedlm_coefficients.csv"
        )
    # -------------------------------------------------
    # Extract low-performer IDs
    # -------------------------------------------------
    low_ids = paired.loc[paired["low_10pct_before"], "mouse_id"].tolist()

    print("\n=== Low-performer mouse IDs (bottom 10% baseline) ===")
    print(low_ids)
    low_animals_df = paired.loc[paired["low_10pct_before"]].copy()

    cols_to_save = ["mouse_id", "before", "after", "delta_after_minus_before"]
    for col in ["group", "sex"]:
        if col in low_animals_df.columns:
            cols_to_save.append(col)

    low_animals_df[cols_to_save].to_csv(
        out_tables / f"{assay_prefix}_{endpoint_slug}_low10_full.csv",
        index=False,
    ) 
    high_animals_df = paired.loc[~paired["low_10pct_before"]].copy()

    high_animals_df[cols_to_save].to_csv(
        out_tables / f"{assay_prefix}_{endpoint_slug}_high90_full.csv",
        index=False,
    )   
    print("\nDone.")
    print(f"Read:    {data_path}")
    print(f"Figures: {out_figures}")
    print(f"Tables:  {out_tables}")


if __name__ == "__main__":
    main()