import argparse
from pathlib import Path

import yaml

from behaviour_stat_pipeline.assays.registry import RUNNER_REGISTRY, RunContext


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--assay",
        required=True,
        choices=sorted(RUNNER_REGISTRY.keys()),
        help="Which assay pipeline to run",
    )
    ap.add_argument("--root", required=True, help="Root folder containing raw CSV outputs")
    ap.add_argument("--pattern", default="*_output_summary.csv", help="Glob pattern for raw CSVs")
    ap.add_argument("--subjects", default="data/metadata/subjects.csv", help="Path to subjects.csv")
    ap.add_argument("--raw_id_col", default="Unnamed: 0", help="Raw ID column in tracking CSV")
    ap.add_argument("--out", default="", help="Output processed CSV path")
    ap.add_argument("--config", default="", help="Path to YAML config (optional)")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    ctx = RunContext(
        root=Path(args.root),
        pattern=args.pattern,
        subjects_path=args.subjects,
        raw_id_col=args.raw_id_col,
        out=args.out,
        cfg=cfg,
    )

    RUNNER_REGISTRY[args.assay].run(ctx)


if __name__ == "__main__":
    main()