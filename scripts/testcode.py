import pandas as pd
from behaviour_stat_pipeline.io import load_tracking_csv, merge_subject_metadata, load_subjects

subjects = load_subjects("data/experiments/EXP-2025-001/metadata/subjects.csv")

f1 = "data/experiments/EXP-2025-001/raw/oft/02_HE2_OFT_27A_output_summary.csv"
f2 = "data/experiments/EXP-2025-001/raw/oft/HE2_OFT_Report2_output_summary.csv"

df1 = load_tracking_csv(f1, assay="OFT")
df1 = merge_subject_metadata(df1, subjects, on_missing_subjects="warn_drop")
df1["timepoint"] = "Single"
df1["source_tag"] = "raw_summary"

df2 = load_tracking_csv(f2, assay="OFT")
df2 = merge_subject_metadata(df2, subjects, on_missing_subjects="warn_drop")
df2["timepoint"] = "Single"
df2["source_tag"] = "report"

cols_to_compare = [
    "mouse_id", "source_tag",
    "total_time_in_arena", "total_distance_moved", "mean_speed",
    "time_in_center", "distance_moved_in_center",
    "time_in_periphery", "time_in_corners",
    "supported_rearing_time", "supported_rearing_count",
    "unsupported_rearing_time", "unsupported_rearing_count",
]

combined = pd.concat([df1, df2], ignore_index=True)

for mid in ["27A_ID1", "27A_ID2", "27A_ID3"]:
    print("\n", "=" * 80)
    print(mid)
    print(combined.loc[combined["mouse_id"] == mid, [c for c in cols_to_compare if c in combined.columns]].to_string(index=False))