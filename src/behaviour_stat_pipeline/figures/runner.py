from __future__ import annotations

import pandas as pd
import re

from pathlib import Path
from typing import Any, Dict, List

from .common import PlotStyle, require_columns
from .boxpoints import plot_group_boxpoints
from .scatter import plot_group_scatter


def _make_long_from_mapping(
    df: pd.DataFrame,
    *,
    x_col: str,
    mapping: Dict[str, str],
    id_cols: List[str],
) -> pd.DataFrame:
    """
    Convert wide columns into long:
      mapping: {x_level: wide_column_name}
    Produces columns: id_cols + [x_col, "value"]
    """
    for label, col in mapping.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in processed data.")

    pieces = []
    for label, col in mapping.items():
        tmp = df[id_cols + [col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp[x_col] = str(label)
        pieces.append(tmp)

    return pd.concat(pieces, axis=0, ignore_index=True)

def _auto_specs_from_endpoints(
    *,
    config: Dict[str, Any],
    fig_cfg: Dict[str, Any],
    assay: str,
    df_columns: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate simple boxpoints specs for each endpoint in the chosen endpoint list.

    Generated plots are single-column "Total" plots:
      wide_to_long:
        x_col: ["ALL"]
        mapping: {"ALL": endpoint}

    Multi-column plots (e.g., Familiar vs Novel) are NOT inferred here by design.
    """
    auto = fig_cfg.get("auto_from_endpoints", {}) or {}
    if not auto.get("enabled", False):
        return []

    source = str(auto.get("source", "secondary_two_way_anova")).strip()
    exclude = set([str(x) for x in (auto.get("exclude", []) or [])])

    # Pull endpoints from config
    if source not in config:
        raise ValueError(f"figures.auto_from_endpoints.source='{source}' not found in config")
    endpoints_by_assay = (config.get(source, {}) or {}).get("endpoints", {}) or {}
    endpoints = endpoints_by_assay.get(assay.lower(), []) or []

    if not isinstance(endpoints, list):
        raise ValueError(f"{source}.endpoints.{assay} must be a list")

    # Validate existence in df (fail fast = reproducible)
    missing = [e for e in endpoints if str(e) not in df_columns and str(e) not in exclude]
    if missing:
        raise ValueError(
            f"[figures:auto_from_endpoints] {assay}: endpoints not found in processed CSV columns: {missing}"
        )

    out_specs: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        endpoint = str(endpoint)
        if endpoint in exclude:
            continue

        out_specs.append(
            {
                "name": endpoint,
                "assay": assay.lower(),
                "type": "boxpoints",
                "value_label": endpoint,
                "x_order": [""],
                "wide_to_long": {
                    "x_col": "measure",
                    "mapping": {"": endpoint},
                },
            }
        )

    return out_specs
def _slug(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _resolve_corr_specs_for_figures(cfg: Dict[str, Any], assay: str) -> List[Dict[str, Any]]:
    """
    Resolve correlation specs similarly to stats-side shorthand:
    - apply correlations.defaults
    - replace groupings:null with default_groupings
    - auto-name if missing
    """
    corr = (cfg.get("correlations", {}) or {})
    defaults = (corr.get("defaults", {}) or {})
    default_groupings = (corr.get("default_groupings", [[]]) or [[]])

    specs_raw = corr.get("specs", []) or []
    if not isinstance(specs_raw, list):
        raise TypeError("correlations.specs must be a list")

    resolved: List[Dict[str, Any]] = []
    for i, s in enumerate(specs_raw):
        if not isinstance(s, dict):
            raise TypeError(f"correlations.specs[{i}] must be a dict")

        out = dict(defaults)
        out.update(s)

        out.setdefault("assay", assay.lower())
        out.setdefault("method", "spearman")

        if out.get("groupings", None) is None:
            out["groupings"] = default_groupings

        if not out.get("name"):
            out["name"] = f"{assay.lower()}_{_slug(out['x'])}_vs_{_slug(out['y'])}"

        # minimal required keys
        for k in ["name", "assay", "x", "y"]:
            if k not in out:
                raise ValueError(f"Resolved correlations spec missing key '{k}'")

        resolved.append(out)

    return resolved


def _auto_scatter_specs_from_correlations(
    *,
    config: Dict[str, Any],
    fig_cfg: Dict[str, Any],
    assay: str,
    df_columns: List[str],
) -> List[Dict[str, Any]]:
    auto = fig_cfg.get("auto_from_correlations", {}) or {}
    if not auto.get("enabled", False):
        return []

    fit_enabled = bool(auto.get("fit_enabled", True))
    exclude = set([str(x) for x in (auto.get("exclude", []) or [])])

    corr_specs = _resolve_corr_specs_for_figures(config, assay)
    out_specs: List[Dict[str, Any]] = []

    for cs in corr_specs:
        if cs.get("assay", "").lower() != assay.lower():
            continue

        name = str(cs["name"])
        if name in exclude:
            continue

        x = str(cs["x"])
        y = str(cs["y"])

        # fail fast: figures should not silently skip missing schema
        missing = [c for c in [x, y] if c not in df_columns]
        if missing:
            raise ValueError(f"[figures:auto_from_correlations] {assay}:{name} missing columns: {missing}")

        out_specs.append(
            {
                "name": f"scatter_{name}",
                "assay": assay.lower(),
                "type": "scatter",
                "x": x,
                "y": y,
                "x_label": x,
                "y_label": y,
                "fit": {"enabled": fit_enabled, "by": "pooled"},
            }
        )

    return out_specs

def run_figures_from_config(
    *,
    config: Dict[str, Any],
    processed_dir: Path,
    out_figures: Path,
    assay: str,
    mouse_id_col: str = "mouse_id",
    group_col: str = "group",
    sex_col: str = "sex",
) -> List[Path]:
    """
    Generate figures for one assay based on a config dict.
    Returns list of written figure paths.
    """
    fig_cfg = config.get("figures", {})
    if not fig_cfg or not fig_cfg.get("enabled", False):
        return []

    defaults = fig_cfg.get("defaults", {})

    style = PlotStyle(
        group_edgecolors=defaults.get("group_edgecolors", {}),
        sex_markers=defaults.get("sex_markers", {}),
        group_gap=float(defaults.get("group_gap", 0.6)),
        box_width=float(defaults.get("box_width", 0.45)),
        sex_dodge=float(defaults.get("sex_dodge", 0.06)),
        jitter_sd=float(defaults.get("jitter_sd", 0.18)),
        point_size=float(defaults.get("point_size", 28.0)),
        point_alpha=float(defaults.get("point_alpha", 0.9)),
    )

    formats = defaults.get("formats", ["png"])
    if not isinstance(formats, list) or not formats:
        raise ValueError("figures.defaults.formats must be a non-empty list, e.g. ['png'].")

    for fmt in formats:
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Unsupported format: {fmt}")

    # load processed data
    processed_path = processed_dir / f"{assay.upper()}_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)
    require_columns(df, [mouse_id_col, group_col, sex_col])

    written: List[Path] = []

    # Start with manually defined specs
    specs = fig_cfg.get("specs", []) or []
    if not isinstance(specs, list):
        raise ValueError("figures.specs must be a list")

    # Add auto-generated specs from endpoint lists (single-column Total plots)
    auto_specs = _auto_specs_from_endpoints(
        config=config,
        fig_cfg=fig_cfg,
        assay=assay,
        df_columns=list(df.columns),
    )
    # Add auto-generated scatter specs from correlations.specs
    auto_scatter_specs = _auto_scatter_specs_from_correlations(
        config=config,
        fig_cfg=fig_cfg,
        assay=assay,
        df_columns=list(df.columns),
    )

    manual_names = {str(s.get("name")) for s in specs if isinstance(s, dict) and "name" in s}
    specs.extend([s for s in auto_scatter_specs if str(s.get("name")) not in manual_names])

    # Avoid duplicates: manual specs win if names collide
    manual_names = {str(s.get("name")) for s in specs if isinstance(s, dict) and "name" in s}
    specs.extend([s for s in auto_specs if str(s.get("name")) not in manual_names])

    for spec in specs:
        if str(spec.get("assay", "")).lower() != assay.lower():
            continue
        plot_type = str(spec.get("type", "")).lower()
        if plot_type not in ("boxpoints", "scatter"):
            raise ValueError(f"Unsupported figure type: {spec.get('type')}")

        name = str(spec.get("name", f"{assay}_figure"))

        # Output paths (same for all plot types)
        stem = f"{assay.upper()}_{name}"
        out_paths = [out_figures / f"{stem}.{fmt}" for fmt in formats]

        if plot_type == "boxpoints":
            value_label = spec.get("value_label", None)
            x_order = spec.get("x_order", None)

            # Prefer per-spec order, else fall back to figures.defaults.group_order
            group_order = spec.get("group_order", defaults.get("group_order", None))
            sex_order = spec.get("sex_order", ["female", "male"])

            w2l = spec.get("wide_to_long", None)
            if not w2l:
                raise ValueError(f"Figure spec '{name}' missing wide_to_long.")

            x_col = str(w2l.get("x_col", "x"))
            mapping = w2l.get("mapping", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Figure spec '{name}' wide_to_long.mapping must be a non-empty dict.")

            # Build long data
            id_cols = [mouse_id_col, group_col, sex_col]
            long_df = _make_long_from_mapping(df, x_col=x_col, mapping=mapping, id_cols=id_cols)

            # --- style overrides ---
            style_spec = PlotStyle(
                group_edgecolors=style.group_edgecolors,
                sex_markers=style.sex_markers,
                box_linewidth=style.box_linewidth,
                median_linewidth=style.median_linewidth,
                whisker_linewidth=style.whisker_linewidth,
                cap_linewidth=style.cap_linewidth,
                box_width=float(spec.get("box_width", style.box_width)),
                point_size=float(spec.get("point_size", style.point_size)),
                point_alpha=float(spec.get("point_alpha", style.point_alpha)),
                jitter_sd=float(spec.get("jitter_sd", style.jitter_sd)),
                sex_dodge=float(spec.get("sex_dodge", style.sex_dodge)),
                group_gap=float(spec.get("group_gap", style.group_gap)),
            )

            # --- optional ANOVA-based significance annotation (main effects only) ---
            sig_cfg = (defaults.get("significance", {}) or {}).copy()
            sig_cfg.update(spec.get("significance", {}) or {})

            anova_cfg = (sig_cfg.get("anova_summary", {}) or {}).copy()
            anova_df = None

            if bool(sig_cfg.get("enabled", False)) and bool(anova_cfg.get("enabled", False)):
                table_key = str(anova_cfg.get("table", "secondary_two_way_anova")).strip()
            table_path = out_figures.parent / "tables" / f"{assay.upper()}_{table_key}.csv"

            if table_path.exists():
                anova_df = pd.read_csv(table_path)
            else:
                anova_df = None

            # --- draw plot ---
            anova_endpoint = None
            mapped = list(mapping.values())
            mapped = [str(m) for m in mapped if m is not None]
            if len(set(mapped)) == 1:
                anova_endpoint = mapped[0]

            plot_group_boxpoints(
                df=long_df,
                value_col="value",
                x_col=x_col,
                group_col=group_col,
                sex_col=sex_col,
                mouse_id_col=mouse_id_col,
                assay=assay.upper(),
                endpoint=name,
                anova_endpoint=anova_endpoint,
                out_paths=out_paths,
                x_order=x_order,
                group_order=group_order,
                sex_order=sex_order,
                style=style_spec,
                title=None,
                y_label=value_label,
                show=False,
                anova_df=anova_df,
                anova_cfg=anova_cfg,
            )

            written.extend(out_paths)


        elif plot_type == "scatter":
            x = spec.get("x")
            y = spec.get("y")
            if not x or not y:
                raise ValueError(f"Figure spec '{name}' (scatter) requires x and y.")

            fit_cfg = spec.get("fit", {}) or {}
            fit_enabled = bool(fit_cfg.get("enabled", True))

            for out_path in out_paths:
                plot_group_scatter(
                    df=df,
                    x=str(x),
                    y=str(y),
                    out_path=str(out_path),
                    style=style,
                    group_col=group_col,
                    sex_col=sex_col,
                    group_order=spec.get("group_order", defaults.get("group_order", None)),
                    sex_order=spec.get("sex_order", ["female", "male"]),
                    x_label=spec.get("x_label", None),
                    y_label=spec.get("y_label", None),
                    title=spec.get("title", None),
                    fit_enabled=fit_enabled,
                )

            written.extend(out_paths)

    return written