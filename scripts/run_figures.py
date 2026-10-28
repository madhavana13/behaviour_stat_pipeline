from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


from behaviour_stat_pipeline.figures.runner import run_figures_from_config
from behaviour_stat_pipeline.common.exclusions import load_exclusions
from behaviour_stat_pipeline.common.yaml_io import load_yaml_strict
from behaviour_stat_pipeline.stats.common import (
    MOUSE_ID_COL,
    ConfigError,
    experiment_paths,
    read_groups,
    require_repo_root,
    resolve_config,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return load_yaml_strict(path)


def resolve_assay_identity(assay: str, exp_cfg: dict) -> tuple[str, str]:
    """
    Resolve user-facing assay instance to canonical assay type.

    Returns:
      assay_instance: e.g. OFT2
      assay_type: e.g. OFT
    """
    requested = assay.upper()
    instances = exp_cfg.get("assay_instances", {}) or {}

    if requested in instances:
        inst = instances[requested] or {}
        assay_type = str(inst.get("assay_type", requested)).upper()
        return requested, assay_type

    return requested, requested


def resolve_figures_config_path(
    *,
    stats_config_path: Path,
    figures_config_arg: str,
) -> Path | None:
    """
    Backward-compatible figure config resolution.

    Priority:
      1. Explicit --figures-config
      2. configs/figures.yaml next to stats.yaml, if present
      3. None, meaning use figures block from --config if it still exists
    """
    if figures_config_arg:
        return Path(figures_config_arg)

    sibling = stats_config_path.parent / "figures.yaml"
    if sibling.exists():
        return sibling

    return None


def merge_stats_and_figures_config(
    *,
    stats_cfg: dict[str, Any],
    figures_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the config object expected by figures.runner.

    The figure runner still needs access to stats-side blocks such as
    secondary_two_way_anova and correlations for auto-generated figure specs.
    Therefore we keep the stats config as the base and replace/add only the
    top-level figures block from figures.yaml.
    """
    merged = dict(stats_cfg or {})

    if not figures_cfg:
        return merged

    # Support either:
    #   figures.yaml:
    #     figures: ...
    # or a bare figures config file containing enabled/defaults/specs directly.
    if "figures" in figures_cfg:
        merged["figures"] = figures_cfg["figures"]
    else:
        merged["figures"] = figures_cfg

    # Optional: allow figures.yaml to define/override figure-related paths,
    # e.g. paths.labels_csv, without duplicating all stats paths.
    if "paths" in figures_cfg:
        paths = dict(merged.get("paths", {}) or {})
        paths.update(figures_cfg.get("paths", {}) or {})
        merged["paths"] = paths

    return merged


def main() -> None:
    require_repo_root(Path(__file__).resolve().parents[1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to stats YAML config")
    ap.add_argument(
        "--figures-config",
        default="",
        help=(
            "Path to figures YAML config. If omitted, scripts/run_figures.py "
            "will use configs/figures.yaml next to --config when present."
        ),
    )
    ap.add_argument("--experiment-config", default="", help="Path to experiment YAML config")
    ap.add_argument("--assays", nargs="*", default=None,
                    help="Assays to run figures for, e.g. nor oft2 (default: analysis.assays of the experiment)")
    args = ap.parse_args()

    stats_config_path = Path(args.config)
    exp_cfg = load_yaml(args.experiment_config)
    # Same merged config as run_stats: the experiment's analysis block over stats.yaml
    stats_cfg = resolve_config(load_yaml(stats_config_path), exp_cfg)

    figures_config_path = resolve_figures_config_path(
        stats_config_path=stats_config_path,
        figures_config_arg=args.figures_config,
    )
    figures_cfg = load_yaml(figures_config_path) if figures_config_path else {}

    cfg = merge_stats_and_figures_config(
        stats_cfg=stats_cfg,
        figures_cfg=figures_cfg,
    )

    group_order, group_labels, group_colors, group_errors = read_groups(exp_cfg)
    if group_errors:
        raise ConfigError("Invalid groups in experiment config:\n  - " + "\n  - ".join(group_errors))

    exp_paths = exp_cfg.get("paths", {}) or {}

    exclusions_path = (
        exp_paths.get("metadata_exclusions", "")
        or exp_paths.get("exclusions", "")
    )
    exclusions = load_exclusions(exclusions_path)

    paths = experiment_paths(cfg, exp_cfg, required=True)
    processed_dir = Path(paths["processed_dir"])
    out_dir = Path(paths["out_dir"])
    mouse_id_col = MOUSE_ID_COL

    factors_cfg = cfg.get("factors", {}) or {}
    group_col = (factors_cfg.get("group", {}) or {}).get("column", "group")
    sex_col = (factors_cfg.get("sex", {}) or {}).get("column", "sex")

    written_all = []

    assays = args.assays if args.assays else list(cfg.get("assays", []) or [])
    if not assays:
        raise SystemExit("No assays: pass --assays or set analysis.assays in the experiment config.")

    for requested_assay in assays:
        assay_instance, assay_type = resolve_assay_identity(requested_assay, exp_cfg)

        instance_out_figures = out_dir / assay_instance.upper() / "figures"
        instance_out_figures.mkdir(parents=True, exist_ok=True)

        instance_cfg = (exp_cfg.get("assay_instances", {}) or {}).get(assay_instance.upper(), {}) or {}
        trial_duration_s = instance_cfg.get("trial_duration_s")

        written = run_figures_from_config(
            config=cfg,
            processed_dir=processed_dir,
            out_figures=instance_out_figures,
            assay=assay_instance.lower(),
            assay_config_key=assay_type.lower(),
            output_prefix=assay_instance.upper(),
            mouse_id_col=mouse_id_col,
            group_col=group_col,
            sex_col=sex_col,
            exclusions=exclusions,
            trial_duration_s=float(trial_duration_s) if trial_duration_s is not None else None,
            group_order=group_order,
            group_labels=group_labels,
            group_colors=group_colors,
        )

        written_all.extend(written)

    if written_all:
        print("Wrote figures:")
        for p in written_all:
            print(f" - {p}")
    else:
        print("No figures written: figures disabled or no matching specs.")


if __name__ == "__main__":
    main()
