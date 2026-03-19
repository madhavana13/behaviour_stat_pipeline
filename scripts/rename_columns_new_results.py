from pathlib import Path
import pandas as pd

# --- SETTINGS ---
input_dir = Path(r"C:\Users\Anjana\Projects\behaviour_stat_pipeline\data\raw\vit")

# --- RENAME MAP ---
RENAME_MAP = {
    "total_time_in_arena": "time in homecage",
    "total_distance_moved": "total distance moved",
    "mean_speed": "average speed",

    "time_in_tl_quadrant": "time in left upper quadrant",
    "distance_moved_in_tl_quadrant": "distance moved in left upper quadrant",
    "transitions_into_tl_quadrant": "transitions into left upper quadrant",

    "time_in_tr_quadrant": "time in right upper quadrant",
    "distance_moved_in_tr_quadrant": "distance moved in right upper quadrant",
    "transitions_into_tr_quadrant": "transitions into right upper quadrant",

    "time_in_br_quadrant": "time in right lower quadrant",
    "distance_moved_in_br_quadrant": "distance moved in right lower quadrant",
    "transitions_into_br_quadrant": "transitions into right lower quadrant",

    "time_in_bl_quadrant": "time in left lower quadrant",
    "distance_moved_in_bl_quadrant": "distance moved in left lower quadrant",
    "transitions_into_bl_quadrant": "transitions into left lower quadrant",
}

csv_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]

print(f"\nFound {len(csv_files)} CSV files in: {input_dir}\n")

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        original_cols = list(df.columns)

        renamed_cols = [RENAME_MAP.get(c, c) for c in original_cols]
        changed = renamed_cols != original_cols

        if not changed:
            print(f"Skipped (no exact header matches): {csv_path.name}")
            continue

        df.columns = renamed_cols
        df.to_csv(csv_path, index=False)

        changed_pairs = [(old, new) for old, new in zip(original_cols, renamed_cols) if old != new]

        print(f"Updated in place: {csv_path.name}")
        for old, new in changed_pairs:
            print(f"  {old} -> {new}")
        print()

    except Exception as e:
        print(f"Error processing {csv_path.name}: {e}\n")

print("Done.")