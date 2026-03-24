from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment-config",
        required=True,
        help="Path to experiment YAML config",
    )
    ap.add_argument(
        "--assay",
        default="nor",
        help="Assay name; default='nor'",
    )
    ap.add_argument(
        "--low-threshold",
        type=float,
        default=20.0,
        help="Reference threshold for low exploration (default: 20 s)",
    )
    args = ap.parse_args()

    exp_cfg = load_yaml(args.experiment_config)
    exp_paths = exp_cfg.get("paths", {})

    processed_dir = Path(exp_paths["processed_dir"])
    out_figures = Path(exp_paths["out_figures"])
    out_tables = Path(exp_paths["out_tables"])

    assay = args.assay.lower()
    assay_prefix = assay.upper()
    data_path = processed_dir / f"{assay_prefix}_processed.csv"

    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    required = [
        "mouse_id",
        "fam_total_explore_time",
        "test_total_explore_time",
        "di_explore",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    df["fam_low_explore"] = df["fam_total_explore_time"] < args.low_threshold
    if df["fam_low_explore"].dtype != bool:
        df["fam_low_explore"] = df["fam_low_explore"].astype(bool)

    df["delta_explore"] = df["test_total_explore_time"] - df["fam_total_explore_time"]

    # -------------------------------------------------
    # Summaries
    # -------------------------------------------------
    group_mean = df.groupby("fam_low_explore")["di_explore"].mean()
    group_desc = df.groupby("fam_low_explore")["di_explore"].describe()
    group_var = df.groupby("fam_low_explore")["di_explore"].var(ddof=1)

    print("\n=== Mean DI by exploration group ===")
    print(group_mean)

    print("\n=== DI describe() by exploration group ===")
    print(group_desc)

    print("\n=== DI variance by exploration group ===")
    print(group_var)

    rho_test, p_test = spearmanr(
        df["test_total_explore_time"],
        df["di_explore"],
        nan_policy="omit",
    )
    print("\n=== Spearman: TEST total exploration vs DI ===")
    print(f"rho = {rho_test:.6f}, p = {p_test:.6f}")

    full_mean_di = df["di_explore"].mean()
    df_high = df.loc[~df["fam_low_explore"]].copy()
    high_mean_di = df_high["di_explore"].mean()

    print("\n=== Sensitivity analysis ===")
    print(f"Full-sample mean DI: {full_mean_di:.6f}")
    print(f"Excluding low explorers mean DI: {high_mean_di:.6f}")
    print(f"Difference: {high_mean_di - full_mean_di:.6f}")

    delta_desc = df["delta_explore"].describe()
    print("\n=== Delta exploration (TEST - FAM) ===")
    print(delta_desc)

    rho_delta, p_delta = spearmanr(
        df["delta_explore"],
        df["di_explore"],
        nan_policy="omit",
    )
    print("\n=== Spearman: delta exploration vs DI ===")
    print(f"rho = {rho_delta:.6f}, p = {p_delta:.6f}")

    # -------------------------------------------------
    # Write tables
    # -------------------------------------------------
    group_desc.to_csv(out_tables / f"{assay_prefix}_di_by_exploration_group_describe.csv")
    group_var.rename("variance").to_csv(
        out_tables / f"{assay_prefix}_di_by_exploration_group_variance.csv"
    )
    delta_desc.to_csv(
        out_tables / f"{assay_prefix}_delta_explore_describe.csv",
        header=["value"],
    )

    summary_rows = [
        {"metric": "mean_di_full_sample", "value": full_mean_di},
        {"metric": "mean_di_excluding_low_explorers", "value": high_mean_di},
        {"metric": "difference_excluding_minus_full", "value": high_mean_di - full_mean_di},
        {"metric": "spearman_rho_test_explore_vs_di", "value": rho_test},
        {"metric": "spearman_p_test_explore_vs_di", "value": p_test},
        {"metric": "spearman_rho_delta_explore_vs_di", "value": rho_delta},
        {"metric": "spearman_p_delta_explore_vs_di", "value": p_delta},
        {"metric": "n_total", "value": len(df)},
        {"metric": "n_low_explorers", "value": int(df["fam_low_explore"].sum())},
        {"metric": "n_non_low_explorers", "value": int((~df["fam_low_explore"]).sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(
        out_tables / f"{assay_prefix}_exploration_bias_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # Scatter: TEST exploration vs DI
    # y-axis is DI, so do NOT force to start at 0
    # x-axis is exploration time, so set x lower bound to 0
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(df["test_total_explore_time"], df["di_explore"], alpha=0.8)
    plt.axhline(0, linewidth=1)
    plt.axvline(args.low_threshold, linestyle="--", linewidth=1)
    plt.xlabel("TEST total exploration time (s)")
    plt.ylabel("Discrimination Index (DI)")
    plt.title("TEST exploration vs novelty preference")
    plt.xlim(left=0)
    savefig(out_figures / f"{assay_prefix}_exploration_vs_DI.png")

    # -------------------------------------------------
    # Scatter: delta exploration vs DI
    # Neither axis should be forced to start at 0
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(df["delta_explore"], df["di_explore"], alpha=0.8)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Delta exploration (TEST - FAM, s)")
    plt.ylabel("Discrimination Index (DI)")
    plt.title("Change in exploration vs novelty preference")
    savefig(out_figures / f"{assay_prefix}_delta_exploration_vs_DI.png")

    # -------------------------------------------------
    # Histogram: delta exploration
    # x can be negative; y=count should start at 0 naturally
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.hist(df["delta_explore"].dropna(), bins=12)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Delta exploration (TEST - FAM, s)")
    plt.ylabel("Count")
    plt.title("Distribution of change in exploration")
    savefig(out_figures / f"{assay_prefix}_delta_exploration_histogram.png")

    # -------------------------------------------------
    # Long data for trajectories
    # -------------------------------------------------
    df_long = df.melt(
        id_vars=["mouse_id", "fam_low_explore"],
        value_vars=["fam_total_explore_time", "test_total_explore_time"],
        var_name="session",
        value_name="exploration_time",
    )

    df_long["session"] = df_long["session"].map(
        {
            "fam_total_explore_time": "FAM",
            "test_total_explore_time": "TEST",
        }
    )

    x_map = {"FAM": 0, "TEST": 1}
    df_long["x"] = df_long["session"].map(x_map)

    # -------------------------------------------------
    # Trajectory: all animals
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    for _, sub in df_long.groupby("mouse_id"):
        sub = sub.sort_values("x")
        plt.plot(sub["x"], sub["exploration_time"], alpha=0.35)
    plt.xticks([0, 1], ["FAM", "TEST"])
    plt.xlabel("Session")
    plt.ylabel("Exploration time (s)")
    plt.title("Exploration trajectory (FAM → TEST)")
    plt.ylim(bottom=0)
    savefig(out_figures / f"{assay_prefix}_exploration_trajectory.png")

    # -------------------------------------------------
    # Trajectory: low explorers highlighted
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    for _, sub in df_long.groupby("mouse_id"):
        sub = sub.sort_values("x")
        low = bool(sub["fam_low_explore"].iloc[0])
        plt.plot(
            sub["x"],
            sub["exploration_time"],
            color="red" if low else "black",
            alpha=0.6 if low else 0.2,
        )
    plt.xticks([0, 1], ["FAM", "TEST"])
    plt.xlabel("Session")
    plt.ylabel("Exploration time (s)")
    plt.title("Trajectory (low explorers highlighted)")
    plt.ylim(bottom=0)
    savefig(out_figures / f"{assay_prefix}_exploration_trajectory_low_flagged.png")

    # -------------------------------------------------
    # Group-aware long data + color map
    # -------------------------------------------------
    df_long_group = None
    groups: list[str] = []
    color_map: dict[str, str] = {}

    if "group" not in df.columns:
        print("WARNING: 'group' column not found. Skipping group-based plots and models.")
    else:
        df_long_group = df_long.merge(
            df[["mouse_id", "group"]],
            on="mouse_id",
            how="left",
        )
        groups = sorted(df_long_group["group"].dropna().unique())
        color_map = build_group_color_map(groups)

        # -------------------------------------------------
        # Trajectory: colored by group
        # -------------------------------------------------
        plt.figure(figsize=(7, 5))
        for _, sub in df_long_group.groupby("mouse_id"):
            sub = sub.sort_values("x")
            g = sub["group"].iloc[0]
            plt.plot(
                sub["x"],
                sub["exploration_time"],
                color=color_map.get(g, "black"),
                alpha=0.5,
            )

        plt.xticks([0, 1], ["FAM", "TEST"])
        plt.xlabel("Session")
        plt.ylabel("Exploration time (s)")
        plt.title("Exploration trajectory (colored by group)")
        plt.ylim(bottom=0)

        handles = [plt.Line2D([0], [0], color=color_map[g], label=g) for g in groups]
        plt.legend(handles=handles, title="Group", frameon=False)
        savefig(out_figures / f"{assay_prefix}_exploration_trajectory_by_group.png")

        # -------------------------------------------------
        # Mean trajectory per group
        # keep same group colors; do not hardcode names
        # -------------------------------------------------
        plt.figure(figsize=(6, 5))
        for g in groups:
            sub = df_long_group[df_long_group["group"] == g]
            means = (
                sub.groupby("session")["exploration_time"]
                .mean()
                .reindex(["FAM", "TEST"])
            )
            plt.plot(
                [0, 1],
                means.values,
                label=g,
                linewidth=2,
                color=color_map[g],
            )

        plt.xticks([0, 1], ["FAM", "TEST"])
        plt.xlabel("Session")
        plt.ylabel("Exploration time (s)")
        plt.title("Mean exploration trajectory by group")
        plt.legend(frameon=False)
        plt.ylim(bottom=0, top=df_long_group["exploration_time"].max() * 1.1)
        savefig(out_figures / f"{assay_prefix}_exploration_mean_trajectory_by_group.png")

        # -------------------------------------------------
        # Repeated-measures models
        # -------------------------------------------------
        df_long_group["session"] = df_long_group["session"].astype("category")

        ols_res = smf.ols(
            "exploration_time ~ group * session",
            data=df_long_group,
        ).fit()
        print(ols_res.summary())

        mixed_res = smf.mixedlm(
            "exploration_time ~ group * session",
            data=df_long_group,
            groups=df_long_group["mouse_id"],
        ).fit()
        print(mixed_res.summary())

        # -------------------------------------------------
        # Trajectory: separate panel per group
        # -------------------------------------------------
        for g in groups:
            sub_g = df_long_group[df_long_group["group"] == g].copy()

            plt.figure(figsize=(7, 5))
            for _, sub in sub_g.groupby("mouse_id"):
                sub = sub.sort_values("x")
                plt.plot(
                    sub["x"],
                    sub["exploration_time"],
                    color=color_map[g],
                    alpha=0.4,
                )

            plt.xticks([0, 1], ["FAM", "TEST"])
            plt.xlabel("Session")
            plt.ylabel("Exploration time (s)")
            plt.title(f"Exploration trajectory: {g}")
            plt.ylim(bottom=0)
            savefig(out_figures / f"{assay_prefix}_exploration_trajectory_group_{g}.png")

    # -------------------------------------------------
    # Mean ± 95% CI trajectory
    # -------------------------------------------------
    traj_rows = []
    for session, sub in df_long.groupby("session", sort=False):
        mean, ci95 = mean_ci95(sub["exploration_time"])
        traj_rows.append(
            {
                "session": session,
                "mean": mean,
                "ci95": ci95,
            }
        )

    traj_summary = pd.DataFrame(traj_rows)
    traj_summary["x"] = traj_summary["session"].map(x_map)
    traj_summary = traj_summary.sort_values("x")

    plt.figure(figsize=(6, 5))
    for _, sub in df_long.groupby("mouse_id"):
        sub = sub.sort_values("x")
        plt.plot(sub["x"], sub["exploration_time"], alpha=0.15)

    plt.plot(traj_summary["x"], traj_summary["mean"], linewidth=2)
    plt.errorbar(
        traj_summary["x"],
        traj_summary["mean"],
        yerr=traj_summary["ci95"],
        fmt="none",
        capsize=4,
        linewidth=1.5,
    )
    plt.xticks([0, 1], ["FAM", "TEST"])
    plt.xlabel("Session")
    plt.ylabel("Exploration time (s)")
    plt.title("Exploration trajectory with mean ± 95% CI")
    plt.ylim(bottom=0)
    savefig(out_figures / f"{assay_prefix}_exploration_trajectory_mean_CI.png")

    traj_summary.to_csv(
        out_tables / f"{assay_prefix}_exploration_trajectory_mean_ci.csv",
        index=False,
    )

    print("\nDone.")
    print(f"Read:    {data_path}")
    print(f"Figures: {out_figures}")
    print(f"Tables:  {out_tables}")


if __name__ == "__main__":
    main()