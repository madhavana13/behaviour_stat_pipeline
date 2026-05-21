import pandas as pd

p = r"outputs\EXP-2026-002\OFT2\tables\OFT2_secondary_two_way_anova.csv"
df = pd.read_csv(p)

print(df[["endpoint", "term", "p_value", "p_adj", "significance_label"]].to_string(index=False))