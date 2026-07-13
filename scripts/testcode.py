import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# -------------------------
# Load data
# -------------------------
lm = pd.read_csv("data/experiments/EXP-2026-002/processed/LM_processed.csv")

# A priori exclusion
exclude_ids = ["4F_ID32"]
lm = lm[~lm["mouse_id"].isin(exclude_ids)].copy()

# -------------------------
# Choose LM anxiety-like endpoints
# Edit this list to match exact column names in your LM_processed.csv
# -------------------------
endpoints = [
    "time_in_central_alley",
    # "fraction_in_central_alley",
    # "time_in_protected_area",
    # "time_in_narrow_arms",
    # "latency_to_center",
    # "n_transitions",
]

results = []

for endpoint in endpoints:
    required_cols = ["mouse_id", "group", "sex", endpoint, "lm_dist_total"]
    df = lm[required_cols].dropna().copy()

    if len(df) < 10:
        print(f"Skipping {endpoint}: too few complete observations")
        continue

    # Unadjusted model
    m0 = smf.ols(
        f"{endpoint} ~ C(group) * C(sex)",
        data=df
    ).fit(cov_type="HC3")

    # Locomotion-adjusted model
    m1 = smf.ols(
        f"{endpoint} ~ C(group) * C(sex) + lm_dist_total",
        data=df
    ).fit(cov_type="HC3")

    term = "C(group)[T.ifs]"

    beta_unadj = m0.params.get(term, float("nan"))
    p_unadj = m0.pvalues.get(term, float("nan"))

    beta_adj = m1.params.get(term, float("nan"))
    p_adj = m1.pvalues.get(term, float("nan"))

    attenuation = beta_adj - beta_unadj
    percent_change = 100 * (abs(beta_unadj) - abs(beta_adj)) / abs(beta_unadj) if beta_unadj != 0 else float("nan")

    results.append({
        "endpoint": endpoint,
        "n": len(df),
        "IFS_beta_unadjusted": beta_unadj,
        "IFS_p_unadjusted": p_unadj,
        "IFS_beta_locomotion_adjusted": beta_adj,
        "IFS_p_locomotion_adjusted": p_adj,
        "absolute_beta_change": attenuation,
        "percent_attenuation": percent_change,
        "lm_distance_beta": m1.params.get("lm_dist_total", float("nan")),
        "lm_distance_p": m1.pvalues.get("lm_dist_total", float("nan")),
    })

summary = pd.DataFrame(results)

# -------------------------
# Save summary table
# -------------------------
out_dir = Path("outputs/EXP-2026-002/cross_assay/tables")
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "LM_anxiety_endpoints_locomotion_sensitivity.csv"
summary.to_csv(out_csv, index=False)

print("\nLocomotion sensitivity summary:")
print(summary.to_string(index=False))

print(f"\nSaved: {out_csv}")