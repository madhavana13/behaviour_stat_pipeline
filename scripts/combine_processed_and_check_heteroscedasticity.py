from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import levene
import matplotlib.pyplot as plt
import argparse
import yaml
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "experiments"

ID_COLUMNS = {
    "raw_id",
    "mouse_id",
    "group",
    "sex",
    "fam_frame",
    "fam_cage",
    "fam_social_animal",
    "assay",
    "source_file",
    "timepoint",
    "is_after",
    "session",
    "familiar_type",
    "experiment_id",
}

UNNAMED_PATTERN = re.compile(r"^Unnamed: \d+$", flags=re.IGNORECASE)

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def infer_assay_from_filename(path: Path) -> str:
    return path.stem.replace("_processed", "")
    
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not UNNAMED_PATTERN.match(str(c))]
    return df.loc[:, keep_cols].copy()


def ensure_required_columns(df: pd.DataFrame, assay: str, experiment_id: str) -> pd.DataFrame:
    out = df.copy()
    if "assay" not in out.columns:
        out["assay"] = assay
    if "experiment_id" not in out.columns:
        out["experiment_id"] = experiment_id
    return out


def get_numeric_endpoint_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if UNNAMED_PATTERN.match(str(col)):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def get_processed_files_for_experiment(experiment_id: str) -> list[Path]:
    processed_dir = DATA_DIR / experiment_id / "processed"
    if not processed_dir.exists():
        return []
    return sorted(processed_dir.glob("*_processed.csv"))


def build_master_wide(experiment_id: str, files: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in files:
        assay = infer_assay_from_filename(path)
        df = pd.read_csv(path)
        df = clean_columns(df)
        df = ensure_required_columns(df, assay=assay, experiment_id=experiment_id)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No processed files found for {experiment_id}")

    return pd.concat(dfs, ignore_index=True, sort=False)


def build_master_long(experiment_id: str, files: list[Path]) -> pd.DataFrame:
    long_parts = []

    for path in files:
        assay = infer_assay_from_filename(path)
        df = pd.read_csv(path)
        df = clean_columns(df)
        df = ensure_required_columns(df, assay=assay, experiment_id=experiment_id)

        endpoint_cols = get_numeric_endpoint_columns(df)

        id_vars = [
            c for c in [
                "experiment_id",
                "assay",
                "mouse_id",
                "group",
                "sex",
                "timepoint",
                "is_after",
                "familiar_type",
                "session",
                "source_file",
            ]
            if c in df.columns
        ]

        long_df = df.melt(
            id_vars=id_vars,
            value_vars=endpoint_cols,
            var_name="endpoint",
            value_name="value",
        ).dropna(subset=["value"])

        long_parts.append(long_df)

    if not long_parts:
        raise FileNotFoundError(f"Could not build long file for {experiment_id}")

    return pd.concat(long_parts, ignore_index=True, sort=False)


def run_levene_by_group(master_long: pd.DataFrame, min_n_per_group: int = 3) -> pd.DataFrame:
    results = []

    for (assay, endpoint), subdf in master_long.groupby(["assay", "endpoint"], dropna=False):
        if "group" not in subdf.columns:
            continue

        tmp = subdf.dropna(subset=["group", "value"]).copy()
        if tmp.empty:
            continue

        counts = tmp.groupby("group")["value"].count()
        eligible_groups = counts[counts >= min_n_per_group].index.tolist()

        if len(eligible_groups) < 2:
            results.append({
                "assay": assay,
                "endpoint": endpoint,
                "test": "levene_by_group",
                "n_groups_tested": len(eligible_groups),
                "group_sizes": "; ".join(f"{g}:{int(counts[g])}" for g in counts.index),
                "statistic": np.nan,
                "p_value": np.nan,
                "heteroscedastic_p_lt_0_05": np.nan,
                "note": "fewer than 2 groups with sufficient n",
            })
            continue

        samples = [
            tmp.loc[tmp["group"] == g, "value"].to_numpy()
            for g in eligible_groups
        ]
        sample_sds = [np.std(s, ddof=1) if len(s) > 1 else np.nan for s in samples]

        # Skip if all groups are constant or if Levene is not informative
        if all(np.isfinite(sd) and sd == 0 for sd in sample_sds):
            results.append({
                "assay": assay,
                "endpoint": endpoint,
                "test": "levene_by_group",
                "n_groups_tested": len(eligible_groups),
                "group_sizes": "; ".join(f"{g}:{int(counts[g])}" for g in counts.index),
                "statistic": np.nan,
                "p_value": np.nan,
                "heteroscedastic_p_lt_0_05": np.nan,
                "note": "all groups had zero variance",
            })
            continue

        stat, pval = levene(*samples, center="median")

        results.append({
            "assay": assay,
            "endpoint": endpoint,
            "test": "levene_by_group",
            "n_groups_tested": len(eligible_groups),
            "group_sizes": "; ".join(f"{g}:{int(counts[g])}" for g in counts.index),
            "statistic": float(stat),
            "p_value": float(pval),
            "heteroscedastic_p_lt_0_05": bool(pval < 0.05),
            "note": "",
        })

    return pd.DataFrame(results)

def compute_variability_summary(
    master_long: pd.DataFrame,
    group_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare variability between groups per assay x endpoint.
    Group order is driven by experiment YAML when provided.
    """

    results = []

    for (assay, endpoint), subdf in master_long.groupby(["assay", "endpoint"], dropna=False):

        if "group" not in subdf.columns:
            continue

        tmp = subdf.dropna(subset=["group", "value"]).copy()

        if tmp["group"].nunique() < 2:
            continue

        summary = {}

        for group_name, gdf in tmp.groupby("group"):
            values = gdf["value"].to_numpy()

            if len(values) < 3:
                continue

            mean = np.mean(values)
            sd = np.std(values, ddof=1)

            summary[group_name] = {
                "mean": mean,
                "sd": sd,
                "cv": sd / mean if mean != 0 else np.nan,
                "n": len(values),
            }

        if len(summary) < 2:
            continue

        groups = list(summary.keys())

        if group_order is not None:
            ordered_groups = [g for g in group_order if g in groups]
            remaining_groups = [g for g in groups if g not in ordered_groups]
            groups = ordered_groups + remaining_groups

        if len(groups) != 2:
            continue

        g1, g2 = groups

        sd1 = summary[g1]["sd"]
        sd2 = summary[g2]["sd"]
        cv1 = summary[g1]["cv"]
        cv2 = summary[g2]["cv"]

        sd_ratio = sd1 / sd2 if pd.notna(sd2) and sd2 != 0 else np.nan
        cv_ratio = cv1 / cv2 if pd.notna(cv2) and cv2 != 0 else np.nan

        results.append({
            "assay": assay,
            "endpoint": endpoint,
            "group_1": g1,
            "group_2": g2,
            "group_1_n": summary[g1]["n"],
            "group_2_n": summary[g2]["n"],
            "group_1_mean": summary[g1]["mean"],
            "group_2_mean": summary[g2]["mean"],
            "group_1_sd": summary[g1]["sd"],
            "group_2_sd": summary[g2]["sd"],
            "group_1_cv": summary[g1]["cv"],
            "group_2_cv": summary[g2]["cv"],
            "sd_ratio_group1_over_group2": sd_ratio,
            "cv_ratio_group1_over_group2": cv_ratio,
        })

    return pd.DataFrame(results)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_sign_test_variability(
    df: pd.DataFrame,
    exclude_assays: list[str] | None = None,
) -> pd.DataFrame:
    """
    Perform binomial sign test:
    H0: P(SD ratio < 1) = 0.5
    """

    if exclude_assays is None:
        exclude_assays = ["VIT"]

    df = df.loc[~df["assay"].isin(exclude_assays)].copy()

    results = []

    # overall
    n_total = len(df)
    n_success = int((df["sd_ratio_group1_over_group2"] < 1).sum())

    test = binomtest(n_success, n_total, p=0.5, alternative="greater")

    results.append({
        "assay": "ALL_excl_VIT",
        "n_endpoints": n_total,
        "n_sd_ratio_lt_1": n_success,
        "proportion": n_success / n_total,
        "p_value": test.pvalue,
    })

    # per assay
    for assay, g in df.groupby("assay"):
        n = len(g)
        k = int((g["sd_ratio_group1_over_group2"] < 1).sum())

        test = binomtest(k, n, p=0.5, alternative="greater")

        results.append({
            "assay": assay,
            "n_endpoints": n,
            "n_sd_ratio_lt_1": k,
            "proportion": k / n,
            "p_value": test.pvalue,
        })

    return pd.DataFrame(results)

def plot_variability_summary(
    variability_df: pd.DataFrame,
    levene_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot log(SD ratio) per endpoint, colored by assay.
    """

    df = variability_df.copy()

    # --- identify group columns dynamically ---
    # assumes exactly two groups like cup / tail
    group_cols = [c for c in df.columns if c.endswith("_sd")]
    group_cols = sorted(group_cols)

    if len(group_cols) != 2:
        raise ValueError("Expected exactly two groups for SD comparison.")

    g1 = group_cols[0].replace("_sd", "")
    g2 = group_cols[1].replace("_sd", "")

    # Compute ratio and log ratio
    df["sd_ratio"] = df[f"{g1}_sd"] / df[f"{g2}_sd"]
    df["sd_ratio"] = df["sd_ratio"].replace([np.inf, -np.inf], np.nan)
    df.loc[df["sd_ratio"] <= 0, "sd_ratio"] = np.nan
    df["log_sd_ratio"] = np.log(df["sd_ratio"])
    df = df.dropna(subset=["log_sd_ratio"]).copy()

    # Merge significance info (if available)
    if not levene_df.empty:
        sig = levene_df[["assay", "endpoint", "p_value"]].copy()
        df = df.merge(sig, on=["assay", "endpoint"], how="left")
        df["significant"] = df["p_value"] < 0.05
    else:
        df["significant"] = False

    # Create label for x-axis
    df["label"] = df["assay"] + " | " + df["endpoint"]

    # Sort for readability
    df = df.sort_values(by=["assay", "endpoint"]).reset_index(drop=True)

    # --- Plot ---
    plt.figure(figsize=(max(14, len(df) * 0.35), 7))

    x = np.arange(len(df))
    y = df["log_sd_ratio"]

    # scatter
    for i in range(len(df)):
        if df.loc[i, "significant"]:
            plt.scatter(x[i], y[i], marker="*", s=100)  # highlight
        else:
            plt.scatter(x[i], y[i])

    # horizontal reference line
    plt.axhline(0, linestyle="--")

    # axis formatting
    plt.xticks(x, df["label"], rotation=90)
    plt.ylabel("log(SD ratio) [group1 / group2]")
    plt.title("Variability comparison across endpoints")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_variability_summary_excluding_vit(
    csv_path: str | Path,
    output_path: str | Path,
    exclude_assays: list[str] | None = None,
) -> None:
    """
    Plot log(SD ratio) for variability summary, excluding problematic assays
    such as VIT by default.

    Assumes:
        - group_1 and group_2 are already ordered correctly upstream
        - sd_ratio_group1_over_group2 is group_1 / group_2
    """
    if exclude_assays is None:
        exclude_assays = ["VIT"]

    df = pd.read_csv(csv_path).copy()

    # Exclude assays like VIT
    df = df.loc[~df["assay"].isin(exclude_assays)].copy()

    # Keep only finite, positive ratios
    df["sd_ratio_group1_over_group2"] = pd.to_numeric(
        df["sd_ratio_group1_over_group2"], errors="coerce"
    )
    df = df.loc[np.isfinite(df["sd_ratio_group1_over_group2"])].copy()
    df = df.loc[df["sd_ratio_group1_over_group2"] > 0].copy()

    if df.empty:
        raise ValueError("No valid rows available for plotting after filtering.")

    # Robust plotting scale
    df["log_sd_ratio"] = np.log(df["sd_ratio_group1_over_group2"])

    # Create readable labels
    df["label"] = df["assay"] + " | " + df["endpoint"]

    # Sort within assay by ratio for easier visual inspection
    df = df.sort_values(["assay", "log_sd_ratio", "endpoint"]).reset_index(drop=True)

    # Dynamic width so labels fit
    fig_width = max(14, len(df) * 0.28)
    plt.figure(figsize=(fig_width, 7))

    x = np.arange(len(df))
    y = df["log_sd_ratio"].to_numpy()

    plt.scatter(x, y)
    plt.axhline(0, linestyle="--")

    plt.xticks(x, df["label"], rotation=90)
    plt.ylabel("log(SD ratio)")
    plt.title("Variability summary by endpoint (excluding VIT)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_variability_assay_summary(
    csv_path: str | Path,
    exclude_assays: list[str] | None = None,
) -> pd.DataFrame:
    if exclude_assays is None:
        exclude_assays = ["VIT"]

    df = pd.read_csv(csv_path).copy()
    df = df.loc[~df["assay"].isin(exclude_assays)].copy()

    rows = []
    for assay, g in df.groupby("assay"):
        rows.append({
            "assay": assay,
            "n_endpoints": len(g),
            "prop_sd_ratio_lt_1": (g["sd_ratio_group1_over_group2"] < 1).mean(),
            "median_sd_ratio": g["sd_ratio_group1_over_group2"].median(),
            "median_cv_ratio": g["cv_ratio_group1_over_group2"].median(),
        })

    summary = pd.DataFrame(rows).sort_values("assay").reset_index(drop=True)

    print("\n=== Assay-level variability summary (excluding VIT) ===")
    print(summary.to_string(index=False))
    return summary

def write_summary_log(
    experiment_id: str,
    files: list[Path],
    master_wide: pd.DataFrame,
    master_long: pd.DataFrame,
    levene_df: pd.DataFrame,
    log_path: Path,
) -> None:
    tested = int(levene_df["p_value"].notna().sum()) if not levene_df.empty else 0
    flagged = (
        int(levene_df["heteroscedastic_p_lt_0_05"].eq(True).sum())
        if not levene_df.empty else 0
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_id}\n")
        f.write(f"Processed files combined: {len(files)}\n")
        for path in files:
            f.write(f"  - {path.name}\n")
        f.write("\n")
        f.write(f"MASTER wide rows: {len(master_wide)}\n")
        f.write(f"MASTER wide columns: {master_wide.shape[1]}\n")
        f.write(f"MASTER long rows: {len(master_long)}\n")
        f.write(f"Unique assays: {master_long['assay'].nunique() if 'assay' in master_long.columns else 0}\n")
        f.write(f"Unique endpoints: {master_long['endpoint'].nunique() if 'endpoint' in master_long.columns else 0}\n")
        f.write("\n")
        f.write(f"Levene tests run: {tested}\n")
        f.write(f"Endpoints with p < 0.05: {flagged}\n")


def process_experiment(
    experiment_id: str,
    group_order: list[str] | None = None,
) -> None:
    files = get_processed_files_for_experiment(experiment_id)
    if not files:
        print(f"[SKIP] No processed files found for {experiment_id}")
        return

    output_tables = PROJECT_ROOT / "outputs" / experiment_id / "tables"
    output_logs = PROJECT_ROOT / "outputs" / experiment_id / "logs"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_logs.mkdir(parents=True, exist_ok=True)

    master_wide = build_master_wide(experiment_id, files)
    master_long = build_master_long(experiment_id, files)
    levene_df = run_levene_by_group(master_long, min_n_per_group=3)
    var_df = compute_variability_summary(master_long, group_order=group_order)
    var_df.to_csv(output_tables / "MASTER_variability_summary.csv", index=False)
    sign_df = run_sign_test_variability(var_df)

    wide_path = output_tables / "MASTER_processed_wide.csv"
    long_path = output_tables / "MASTER_processed_long.csv"
    levene_path = output_tables / "MASTER_heteroscedasticity_levene_by_group.csv"
    log_path = output_logs / "MASTER_combined_summary.txt"
    plot_path = output_tables / "MASTER_variability_summary.png"
    variability_path = output_tables / "MASTER_variability_summary.csv"
    plot_ex_vit_path = output_tables / "MASTER_variability_summary_excluding_VIT.png"
    sign_path = output_tables / "MASTER_variability_sign_test.csv"

    plot_variability_summary(
        variability_df=var_df,
        levene_df=levene_df,
        output_path=plot_path,
    )

    master_wide.to_csv(wide_path, index=False)
    master_long.to_csv(long_path, index=False)
    levene_df.to_csv(levene_path, index=False)
    sign_df.to_csv(sign_path, index=False)

    write_summary_log(
        experiment_id=experiment_id,
        files=files,
        master_wide=master_wide,
        master_long=master_long,
        levene_df=levene_df,
        log_path=log_path,
    )

    print(f"[DONE] {experiment_id}")
    print(f"  - {wide_path}")
    print(f"  - {long_path}")
    print(f"  - {levene_path}")
    print(f"  - {log_path}")
    print(f"  - {variability_path}")
    print(f"  - {plot_path}")
    print(f"  - {plot_ex_vit_path}")
    print("\n=== Sign test for variability (SD ratio < 1) ===")
    print(sign_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to experiment YAML, e.g. configs/experiments/EXP-2025-001.yaml",
    )
    args = parser.parse_args()

    experiment_config_path = PROJECT_ROOT / args.experiment_config
    if not experiment_config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_config_path}")

    exp_cfg = load_yaml(experiment_config_path)
    experiment_id = exp_cfg.get("experiment_id", experiment_config_path.stem)
    group_order = exp_cfg.get("group_order", None)

    process_experiment(experiment_id, group_order=group_order)


if __name__ == "__main__":
    main()