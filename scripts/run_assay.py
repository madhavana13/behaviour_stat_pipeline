import argparse
from pathlib import Path

import yaml

from behaviour_stat_pipeline.assays.registry import RUNNER_REGISTRY, RunContext


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--assay",
        required=True,
        help="Which assay pipeline to run (case-insensitive)",
    )
    ap.add_argument("--root", default="", help="Root folder containing raw CSV outputs")
    ap.add_argument("--pattern", default="", help="Glob pattern for raw CSVs")
    ap.add_argument("--subjects", default="", help="Path to subjects.csv")
    ap.add_argument("--raw_id_col", default="Unnamed: 0", help="Raw ID column in tracking CSV")
    ap.add_argument("--out", default="", help="Output processed CSV path")
    ap.add_argument("--config", default="", help="Path to shared YAML config (optional)")
    ap.add_argument(
        "--experiment-config",
        default="",
        help="Path to experiment YAML config with paths/raw roots",
    )
    return ap


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_assay_inputs(args: argparse.Namespace, shared_cfg: dict, exp_cfg: dict) -> tuple[Path, str, str, str]:
    assay_key = args.assay.lower()

    exp_paths = exp_cfg.get("paths", {})
    exp_raw = exp_cfg.get("raw", {})
    exp_patterns = exp_cfg.get("assay_patterns", {})

    # Precedence:
    # CLI explicit value > experiment config > default fallback
    root = args.root or exp_raw.get(assay_key, "")
    if not root:
        raise ValueError(
            f"No raw root provided for assay '{args.assay}'. "
            f"Use --root or define raw.{assay_key} in experiment config."
        )

    subjects = args.subjects or exp_paths.get("metadata_subjects", "data/metadata/subjects.csv")

    pattern = args.pattern or exp_patterns.get(assay_key, "*_output_summary.csv")

    out = args.out or str(Path(exp_paths.get("processed_dir", "data/processed")) / f"{args.assay.upper()}_processed.csv")

    return Path(root), pattern, subjects, out


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    assay_key = args.assay.upper()

    if assay_key not in RUNNER_REGISTRY:
        raise ValueError(
            f"Invalid assay '{args.assay}'. Must be one of {list(RUNNER_REGISTRY.keys())}"
        )

    args.assay = assay_key

    shared_cfg = load_yaml(args.config)
    exp_cfg = load_yaml(args.experiment_config)

    # For now, pass shared config into the runner because that's where
    # subjects validation is currently read from.
    cfg = shared_cfg.copy()

    root, pattern, subjects, out = resolve_assay_inputs(args, shared_cfg, exp_cfg)

    ctx = RunContext(
        root=root,
        pattern=pattern,
        subjects_path=subjects,
        raw_id_col=args.raw_id_col,
        out=out,
        cfg=cfg,
    )

    RUNNER_REGISTRY[args.assay].run(ctx)


if __name__ == "__main__":
    main()