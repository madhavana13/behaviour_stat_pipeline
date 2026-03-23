from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

# --- SETTINGS ---
input_dir = Path(r"C:\\Users\\Anjana\\Projects\\behaviour_stat_pipeline\\data\\experiments\\EXP-2025-001\\processed\\VIT_processed.csv")
dry_run = False  # True = preview only, False = overwrite files in place

# -----------------------------------------------------------------------------
# COLUMN NORMALISATION STRATEGY
#
# 1) exact full-column renames
# 2) fragment-level replacements applied anywhere in the header
#
# This is designed for vendor/team outputs where the prefix changes
# (e.g. hand__After__..., did__..., delta_after_minus_before__...)
# but the metric fragment is the same.
# -----------------------------------------------------------------------------

# Exact full-column matches only
EXACT_RENAME_MAP = {
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

# Fragment replacements applied anywhere in the column name.
# Keep these conservative: only fix variants that are clearly the same variable.
FRAGMENT_REPLACEMENTS = [
    ("time in in left lower quadrant", "time in left lower quadrant"),
    ("time in in left upper quadrant", "time in left upper quadrant"),
    ("time in in right lower quadrant", "time in right lower quadrant"),
    ("time in in right upper quadrant", "time in right upper quadrant"),
]


def normalize_column_name(col: str) -> str:
    """Return canonical column name while leaving unknown parts untouched."""
    new_col = EXACT_RENAME_MAP.get(col, col)

    for old_fragment, new_fragment in FRAGMENT_REPLACEMENTS:
        new_col = new_col.replace(old_fragment, new_fragment)

    # optional light cleanup for repeated spaces
    new_col = re.sub(r"\s+", " ", new_col).strip()
    return new_col


if input_dir.is_file():
    csv_files = [input_dir]
elif input_dir.is_dir():
    csv_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv"
    )
else:
    raise ValueError(f"Invalid path: {input_dir}")

print(f"\nFound {len(csv_files)} CSV files in: {input_dir}")
print(f"Mode: {'DRY RUN (no files will be changed)' if dry_run else 'WRITE IN PLACE'}\n")

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        original_cols = list(df.columns)
        renamed_cols = [normalize_column_name(c) for c in original_cols]

        changed_pairs = [
            (old, new)
            for old, new in zip(original_cols, renamed_cols)
            if old != new
        ]

        if not changed_pairs:
            print(f"Skipped (no matching header variants): {csv_path.name}")
            continue

        # Guard against accidental duplicate columns after renaming
        duplicates = pd.Index(renamed_cols)[pd.Index(renamed_cols).duplicated()].unique().tolist()
        if duplicates:
            print(f"WARNING: {csv_path.name} would create duplicate columns after renaming.")
            print("Skipped for safety. Duplicates:")
            for dup in duplicates:
                print(f"  {dup}")
            print()
            continue

        print(f"{'Would update' if dry_run else 'Updated in place'}: {csv_path.name}")
        for old, new in changed_pairs:
            print(f"  {old} -> {new}")
        print()

        if not dry_run:
            df.columns = renamed_cols
            df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"Error processing {csv_path.name}: {e}\n")

print("Done.")
