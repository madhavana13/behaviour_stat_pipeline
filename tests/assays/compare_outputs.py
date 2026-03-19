import pandas as pd
from pathlib import Path

OLD = Path("outputs/tables_reference")
NEW = Path("outputs/tables")

FILES = [
    "VIT_secondary_two_way_anova.csv",
    "VIT_primary_one_sample.csv",
    "VIT_secondary_posthoc_cellmeans.csv",
    # add OFT, LM, VIT equivalents
]

TOL = 1e-10  # numerical tolerance


def compare_file(fname):
    old = pd.read_csv(OLD / fname)
    new = pd.read_csv(NEW / fname)

    print(f"\n=== {fname} ===")

    # 1. Shape check
    if old.shape != new.shape:
        print(f"Shape mismatch: {old.shape} vs {new.shape}")
        return

    # 2. Column check
    if list(old.columns) != list(new.columns):
        print("Column mismatch")
        print("OLD:", old.columns)
        print("NEW:", new.columns)
        return

    # 3. Sort deterministically (CRITICAL)
    sort_cols = [c for c in ["assay", "endpoint", "term", "group"] if c in old.columns]
    old = old.sort_values(sort_cols).reset_index(drop=True)
    new = new.sort_values(sort_cols).reset_index(drop=True)

    # 4. Numeric vs non-numeric split
    num_cols = old.select_dtypes(include="number").columns
    cat_cols = [c for c in old.columns if c not in num_cols]

    # 5. Exact match for categorical
    if not old[cat_cols].equals(new[cat_cols]):
        print("Categorical mismatch")
        diff = old[cat_cols].compare(new[cat_cols])
        print(diff.head())
        return

    # 6. Numeric tolerance check
    diff = (old[num_cols] - new[num_cols]).abs()

    if (diff > TOL).any().any():
        print("Numeric mismatch detected")
        print(diff.max())
    else:
        print("MATCH")


if __name__ == "__main__":
    for f in FILES:
        compare_file(f)