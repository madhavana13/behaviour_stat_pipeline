import argparse
from pathlib import Path

import yaml

from behaviour_stat_pipeline.assays.registry import RUNNER_REGISTRY, RunContext
from behaviour_stat_pipeline.common.schema import load_schema_aliases
schema_aliases = load_schema_aliases("configs/schema_aliases.yaml")

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

def resolve_assay_identity(args: argparse.Namespace, exp_cfg: dict) -> tuple[str, str, str]:
    """
    Resolve a user-facing assay argument into:
      - assay_instance: output/provenance name, e.g. OFT1
      - assay_type: canonical registered runner name, e.g. OFT
      - session: optional human-readable session label, defaults to assay_instance
    """
    requested = args.assay.upper()
    instances = exp_cfg.get("assay_instances", {}) or {}

    if requested in instances:
        inst = instances[requested] or {}
        assay_type = str(inst.get("assay_type", requested)).upper()
        session = str(inst.get("session", requested))
        return requested, assay_type, session

    return requested, requested, requested

def instance_config(exp_cfg: dict, assay_instance: str) -> dict:
    """assay_instances.<NAME> from the experiment config (case-insensitive name), or {}."""
    instances = exp_cfg.get("assay_instances", {}) or {}
    for name, inst in instances.items():
        if str(name).upper() == assay_instance.upper():
            return dict(inst or {})
    return {}


def resolve_assay_inputs(
    args: argparse.Namespace,
    shared_cfg: dict,
    exp_cfg: dict,
    assay_instance: str,
    assay_type: str,
) -> tuple[Path, str, str, str]:
    instance_key = assay_instance.lower()
    type_key = assay_type.lower()

    exp_paths = exp_cfg.get("paths", {})
    exp_raw = exp_cfg.get("raw", {})
    exp_patterns = exp_cfg.get("assay_patterns", {})
    misc_cfg = shared_cfg.get("misc", {})

    subjects = args.subjects or exp_paths.get("metadata_subjects", "data/metadata/subjects.csv")
    out = args.out or str(
    Path(exp_paths.get("processed_dir", "data/processed"))
    / f"{assay_instance.upper()}_processed.csv"
)

    if type_key == "misc":
        raw_file = (
            args.root
            or (exp_cfg.get("assay_instances", {}).get(assay_instance, {}) or {}).get("raw_path", "")
            or exp_raw.get(instance_key, "")
            or exp_raw.get(type_key, "")
            or misc_cfg.get("raw_file", "")
        )
        if not raw_file:
            raise ValueError(
                "No raw file provided for assay 'MISC'. "
                "Use --root with the path to misc_raw.csv, or define raw.misc in the experiment config, "
                "or misc.raw_file in the shared config."
            )
        return Path(raw_file), "", subjects, out

    instance_cfg = (exp_cfg.get("assay_instances", {}).get(assay_instance, {}) or {})
    root = args.root or instance_cfg.get("raw_path", "") or exp_raw.get(instance_key, "") or exp_raw.get(type_key, "")
    if not root:
        raise ValueError(
            f"No raw root provided for assay '{args.assay}'. "
            f"Use --root or define raw.{assay_key} in experiment config."
        )

    pattern = args.pattern or instance_cfg.get("pattern", "") or exp_patterns.get(instance_key, "") or exp_patterns.get(type_key, "*_output_summary.csv")
    return Path(root), pattern, subjects, out


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    shared_cfg = load_yaml(args.config)
    exp_cfg = load_yaml(args.experiment_config)

    assay_instance, assay_type, session = resolve_assay_identity(args, exp_cfg)

    if assay_type not in RUNNER_REGISTRY:
        raise ValueError(
            f"Invalid assay type '{assay_type}' resolved from '{args.assay}'. "
            f"Must be one of {list(RUNNER_REGISTRY.keys())}"
        )

    # For now, pass shared config into the runner because that's where
    # subjects validation is currently read from.
    cfg = shared_cfg.copy()

    root, pattern, subjects, out = resolve_assay_inputs(
    args,
    shared_cfg,
    exp_cfg,
    assay_instance=assay_instance,
    assay_type=assay_type,
    )

    ctx = RunContext(
        root=root,
        pattern=pattern,
        subjects_path=subjects,
        raw_id_col=args.raw_id_col,
        out=out,
        cfg=cfg,
        schema_aliases=schema_aliases,
        assay_instance=assay_instance,
        session=session,
        instance_cfg=instance_config(exp_cfg, assay_instance),
    )

    RUNNER_REGISTRY[assay_type].run(ctx)


if __name__ == "__main__":
    main()