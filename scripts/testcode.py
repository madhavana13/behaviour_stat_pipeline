import pandas as pd
from scipy.stats import ttest_rel

df = pd.read_csv("data/processed/VIT_processed.csv")

endpoint_before = "hand__Before__time in right lower quadrant"
endpoint_after = "hand__After__time in right lower quadrant"

for group in df["group"].unique():
    d = df[df["group"] == group].copy()

    before = d[endpoint_before]
    after = d[endpoint_after]

    # paired test
    res = ttest_rel(after, before, nan_policy="omit")

    print(f"\nGroup: {group}")
    print("Mean before:", before.mean())
    print("Mean after:", after.mean())
    print("Mean delta:", (after - before).mean())
    print("Paired t-test:", res)