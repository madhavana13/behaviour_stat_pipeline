import argparse
from pathlib import Path

from behaviour_stat_pipeline.io import load_subjects, load_tracking_csv, merge_subject_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to raw tracking CSV")
    p.add_argument("--subjects", default="data/metadata/subjects.csv", help="Path to subjects.csv")
    p.add_argument("--raw_id_col", default="Unnamed: 0", help="Column containing raw IDs")
    p.add_argument("--out", default="", help="Optional path to save standardized CSV")
    args = p.parse_args()

    subjects = load_subjects(args.subjects)
    df = load_tracking_csv(args.csv, raw_id_col=args.raw_id_col)
    df = merge_subject_metadata(df, subjects)

    print(df[["mouse_id", "group", "sex"]].head(10))
    print(f"\nRows: {len(df)} | Unique mice: {df['mouse_id'].nunique()}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
