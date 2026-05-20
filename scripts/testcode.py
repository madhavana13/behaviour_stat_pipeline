import pandas as pd

df = pd.read_csv(r"data\experiments\EXP-2026-002\processed\OFT2_processed.csv")

f1 = "group"
f2 = "sex"

exclude = {"mouse_id", "group", "sex", "assay", "experiment_id"}

for endpoint in df.columns:
    if endpoint in exclude:
        continue

    tmp = df[[endpoint, f1, f2]].copy()
    tmp[endpoint] = pd.to_numeric(tmp[endpoint], errors="coerce")
    tmp = tmp.dropna(subset=[endpoint, f1, f2])

    group_levels = tmp[f1].dropna().unique().tolist()
    sex_levels = tmp[f2].dropna().unique().tolist()

    if tmp.empty or len(group_levels) != 2 or len(sex_levels) != 2:
        print("\nBAD ENDPOINT:", endpoint)
        print("n complete:", len(tmp))
        print("group levels:", group_levels)
        print("sex levels:", sex_levels)
        if not tmp.empty:
            print(pd.crosstab(tmp[f1], tmp[f2]))