import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
from pathlib import Path


# =============================================================================
# Cross-assay locomotor consistency: OFT2 vs LM
# EXP-2026-002
#
# Question:
# Do animals that move more in OFT2 also move more in the Line Maze?
# Does this relationship differ between control and IFS animals?
#
# Notes:
# - Correlations are descriptive.
# - Group-specific correlations are exploratory because n per group is small.
# - Formal group difference in slope is tested using an interaction model.
# =============================================================================


# -------------------------
# Configuration
# -------------------------
experiment_id = "EXP-2026-002"

oft2_path = Path(f"data/experiments/{experiment_id}/processed/OFT2_processed.csv")
lm_path = Path(f"data/experiments/{experiment_id}/processed/LM_processed.csv")

out_base = Path(f"outputs/{experiment_id}/cross_assay")
fig_dir = out_base / "figures"
table_dir = out_base / "tables"
model_dir = out_base / "models"

fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# A priori exclusions
exclude_ids = ["4F_ID32", "4F_ID12"]


# -------------------------
# Helper functions
# -------------------------
def format_p(p):
    """Format p-values for figure text."""
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def safe_pearson(x, y):
    """Compute Pearson correlation safely."""
    if len(x) < 3:
        return np.nan, np.nan
    if np.isclose(np.std(x, ddof=1), 0) or np.isclose(np.std(y, ddof=1), 0):
        return np.nan, np.nan
    return pearsonr(x, y)


def safe_spearman(x, y):
    """Compute Spearman correlation safely."""
    if len(x) < 3:
        return np.nan, np.nan
    if np.isclose(np.std(x, ddof=1), 0) or np.isclose(np.std(y, ddof=1), 0):
        return np.nan, np.nan
    return spearmanr(x, y)


# -------------------------
# Load data
# -------------------------
oft2 = pd.read_csv(oft2_path)
lm = pd.read_csv(lm_path)

# Keep group/sex only from OFT2 to avoid duplicated columns after merge.
oft2_sub = oft2[["mouse_id", "group", "sex", "total_distance_moved"]].rename(
    columns={"total_distance_moved": "oft2_distance"}
)

lm_sub = lm[["mouse_id", "total_distance_moved"]].rename(
    columns={"total_distance_moved": "lm_total_distance"}
)

# Merge within animal
df = oft2_sub.merge(lm_sub, on="mouse_id", how="inner")

# Drop missing values
df = df.dropna(
    subset=["oft2_distance", "lm_total_distance", "group", "sex"]
).copy()

# Apply a priori exclusions
df = df[~df["mouse_id"].isin(exclude_ids)].copy()

# Standardize for model-based comparison
df["z_oft2_distance"] = (
    df["oft2_distance"] - df["oft2_distance"].mean()
) / df["oft2_distance"].std(ddof=1)

df["z_lm_total_distance"] = (
    df["lm_total_distance"] - df["lm_total_distance"].mean()
) / df["lm_total_distance"].std(ddof=1)


# -------------------------
# Print merged data
# -------------------------
print("\nMerged data after exclusion:")
print(df[["mouse_id", "group", "sex", "oft2_distance", "lm_total_distance"]])
print("\nN animals:", len(df))
print("Excluded IDs:", exclude_ids)


# =============================================================================
# 1. Overall correlation
# =============================================================================
pearson_r, pearson_p = safe_pearson(
    df["oft2_distance"],
    df["lm_total_distance"]
)

spearman_rho, spearman_p = safe_spearman(
    df["oft2_distance"],
    df["lm_total_distance"]
)

overall_corr_summary = pd.DataFrame([
    {
        "analysis": "overall",
        "n": len(df),
        "excluded_ids": ";".join(exclude_ids),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }
])

out_overall_corr = table_dir / "OFT2_LM_locomotor_consistency_overall.csv"
overall_corr_summary.to_csv(out_overall_corr, index=False)

print("\nOverall cross-assay locomotor consistency: OFT2 vs LM")
print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")
print(f"Spearman rho = {spearman_rho:.3f}, p = {spearman_p:.4f}")
print(f"Saved: {out_overall_corr}")


# =============================================================================
# 2. Correlations split by group
# =============================================================================
group_corr_results = []

print("\nCross-assay locomotor consistency by group:")

for group_name, sub in df.groupby("group", observed=True):
    sub = sub.dropna(subset=["oft2_distance", "lm_total_distance"]).copy()
    n_group = len(sub)

    pearson_r_g, pearson_p_g = safe_pearson(
        sub["oft2_distance"],
        sub["lm_total_distance"]
    )

    spearman_rho_g, spearman_p_g = safe_spearman(
        sub["oft2_distance"],
        sub["lm_total_distance"]
    )

    group_corr_results.append({
        "group": group_name,
        "n": n_group,
        "excluded_ids": ";".join(exclude_ids),
        "pearson_r": pearson_r_g,
        "pearson_p": pearson_p_g,
        "spearman_rho": spearman_rho_g,
        "spearman_p": spearman_p_g,
    })

    if n_group < 4:
        print(f"{group_name}: n = {n_group}; too few animals for stable correlation")
    else:
        print(
            f"{group_name}: "
            f"n = {n_group}, "
            f"Pearson r = {pearson_r_g:.3f}, p = {pearson_p_g:.4f}; "
            f"Spearman rho = {spearman_rho_g:.3f}, p = {spearman_p_g:.4f}"
        )

group_corr_summary = pd.DataFrame(group_corr_results)

out_group_corr = table_dir / "OFT2_LM_locomotor_consistency_by_group.csv"
group_corr_summary.to_csv(out_group_corr, index=False)

print(f"\nSaved group-wise correlations: {out_group_corr}")


# =============================================================================
# 3. Adjusted OLS model: overall association controlling for group and sex
# =============================================================================
adjusted_model = smf.ols(
    "z_lm_total_distance ~ z_oft2_distance + C(group) * C(sex)",
    data=df
).fit(cov_type="HC3")

out_adjusted_model = model_dir / "OFT2_LM_adjusted_locomotor_model_HC3.txt"

with open(out_adjusted_model, "w", encoding="utf-8") as f:
    f.write("Adjusted OLS model with HC3 robust SE\n")
    f.write("Formula: z_lm_total_distance ~ z_oft2_distance + C(group) * C(sex)\n\n")
    f.write(str(adjusted_model.summary()))

print("\nAdjusted OLS model with HC3 robust SE")
print(adjusted_model.summary())
print(f"Saved: {out_adjusted_model}")


# =============================================================================
# 4. Formal test: does OFT2-LM slope differ by group?
# =============================================================================
slope_model = smf.ols(
    "z_lm_total_distance ~ z_oft2_distance * C(group) + C(sex)",
    data=df
).fit(cov_type="HC3")

out_slope_model = model_dir / "OFT2_LM_group_slope_interaction_model_HC3.txt"

with open(out_slope_model, "w", encoding="utf-8") as f:
    f.write("Group difference in OFT2-LM locomotor slope\n")
    f.write("Formula: z_lm_total_distance ~ z_oft2_distance * C(group) + C(sex)\n\n")
    f.write("Key term: z_oft2_distance:C(group)[T.ifs]\n")
    f.write("This tests whether the OFT2-LM locomotor relationship differs between IFS and control animals.\n\n")
    f.write(str(slope_model.summary()))

print("\nGroup difference in OFT2-LM locomotor slope")
print(slope_model.summary())
print(f"Saved: {out_slope_model}")


# =============================================================================
# 5. Two-panel scatterplot with side information panel
# =============================================================================
from matplotlib.lines import Line2D

markers = {"female": "o", "male": "^"}
colors = {"control": "tab:blue", "ifs": "tab:orange"}

# Create figure: 2 plots + 1 side panel for text/legend
fig, axes = plt.subplots(
    1, 3,
    figsize=(10.5, 4.6),
    dpi=300,
    gridspec_kw={"width_ratios": [1, 1, 0.72]}
)

ax_control, ax_ifs, ax_info = axes

# Turn off the info axis frame
ax_info.axis("off")

# Shared axis limits across both plots
x_min = df["oft2_distance"].min()
x_max = df["oft2_distance"].max()
y_min = df["lm_total_distance"].min()
y_max = df["lm_total_distance"].max()

x_pad = 0.05 * (x_max - x_min)
y_pad = 0.08 * (y_max - y_min)

xlim = (x_min - x_pad, x_max + x_pad)
ylim = (y_min - y_pad, y_max + y_pad)

# Map group name to axis
group_axes = {
    "control": ax_control,
    "ifs": ax_ifs,
}

# Plot each group in its own panel
for group_name, ax in group_axes.items():
    sub_group = df[df["group"] == group_name].copy()

    # Scatter by sex
    for sex, sub_sex in sub_group.groupby("sex", observed=True):
        ax.scatter(
            sub_sex["oft2_distance"],
            sub_sex["lm_total_distance"],
            marker=markers.get(sex, "o"),
            color=colors.get(group_name, "gray"),
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    # Group-specific regression line
    if len(sub_group) >= 3:
        x_g = sub_group["oft2_distance"].to_numpy()
        y_g = sub_group["lm_total_distance"].to_numpy()

        slope_g, intercept_g = np.polyfit(x_g, y_g, 1)
        x_line_g = np.linspace(x_g.min(), x_g.max(), 100)
        y_line_g = intercept_g + slope_g * x_line_g

        ax.plot(
            x_line_g,
            y_line_g,
            linestyle="--",
            linewidth=1.5,
            color=colors.get(group_name, "gray"),
            alpha=0.9,
            zorder=2,
        )

    # Optional: label the highest LM-distance animal within each group
    if len(sub_group) > 0:
        extreme_row = sub_group.loc[sub_group["lm_total_distance"].idxmax()]
        ax.annotate(
            extreme_row["mouse_id"],
            xy=(extreme_row["oft2_distance"], extreme_row["lm_total_distance"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    # Styling
    ax.set_title(group_name.capitalize(), fontsize=12, pad=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)

# Axis labels
ax_control.set_ylabel("LM total distance moved", fontsize=11)
ax_control.set_xlabel("OFT2 total distance moved", fontsize=11, labelpad=8)
ax_ifs.set_xlabel("OFT2 total distance moved", fontsize=11, labelpad=8)

# Main title
fig.suptitle("OFT2–LM locomotor consistency by group", fontsize=16, y=0.98)

# Build stats text for side panel
stats_lines = [
    f"Excluded: {', '.join(exclude_ids)}",
    "",
    f"Overall",
    f"  r = {pearson_r:.2f}, p {format_p(pearson_p)}, n = {len(df)}",
    ""
]

for row in group_corr_results:
    stats_lines.extend([
        f"{row['group'].capitalize()}",
        f"  r = {row['pearson_r']:.2f}, p {format_p(row['pearson_p'])}, n = {int(row['n'])}",
        ""
    ])

stats_text = "\n".join(stats_lines)

ax_info.text(
    0.00,
    0.98,
    stats_text,
    va="top",
    ha="left",
    fontsize=8.5,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="0.8",
        alpha=0.95,
    ),
)

# Custom legend in side panel
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=7,
        linestyle="None",
        label="female",
    ),
    Line2D(
        [0], [0],
        marker="^",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=7,
        linestyle="None",
        label="male",
    ),
    Line2D(
        [0], [0],
        color=colors["control"],
        linestyle="--",
        linewidth=1.5,
        label="control fit",
    ),
    Line2D(
        [0], [0],
        color=colors["ifs"],
        linestyle="--",
        linewidth=1.5,
        label="ifs fit",
    ),
]

ax_info.legend(
    handles=legend_handles,
    frameon=False,
    fontsize=8.5,
    loc="lower left",
    bbox_to_anchor=(0.00, 0.02),
    borderaxespad=0.0,
)

# Layout
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_png = fig_dir / "OFT2_LM_locomotor_consistency_two_panel_sideinfo_ID32_ID12_excluded.png"
out_svg = fig_dir / "OFT2_LM_locomotor_consistency_two_panel_sideinfo_ID32_ID12_excluded.svg"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")

plt.show()

print(f"Saved: {out_png}")
print(f"Saved: {out_svg}")


# =============================================================================
# 6. Short interpretation notes for terminal
# =============================================================================
print("\nInterpretation guide:")
print("- Overall Pearson/Spearman: descriptive cross-assay locomotor consistency.")
print("- Group-wise correlations: descriptive only; n per group is small.")
print("- Slope interaction model:")
print("  Key term = z_oft2_distance:C(group)[T.ifs]")
print("  If this term is not statistically reliable, there is no clear evidence")
print("  that the OFT2-LM locomotor relationship differs between IFS and controls.")