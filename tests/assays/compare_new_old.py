import pandas as pd

old = pd.read_csv("data/processed_reference/NOR_processed.csv")
new = pd.read_csv("data/processed/NOR_processed.csv")

old = old.sort_values(by="mouse_id").reset_index(drop=True)
new = new.sort_values(by="mouse_id").reset_index(drop=True)

old_ids = set(old["mouse_id"])
new_ids = set(new["mouse_id"])

print("Only in new:", sorted(new_ids - old_ids))
print("Only in old:", sorted(old_ids - new_ids))

print(f"Old shape: {old.shape}")
print(f"New shape: {new.shape}")

if list(old.columns) != list(new.columns):
    print("Column mismatch detected.")
    print("Old only:", [c for c in old.columns if c not in new.columns])
    print("New only:", [c for c in new.columns if c not in old.columns])
    raise AssertionError("Column mismatch")

pd.testing.assert_frame_equal(old, new, check_dtype=False)
print("VIT processed outputs match exactly.")