from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .boxpoints import plot_group_boxpoints
from .common import PlotStyle, require_columns
from .scatter import plot_group_scatter


def _make_long_from_mapping(
    df: pd.DataFrame,
    *,
    x_col: str,
    mapping: Dict[str, str],
    id_cols: List[str],
) -> pd.DataFrame:
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


def _load_label_map(cfg):
    paths = cfg.get("paths", {}) or {}
    p = paths.get("labels_csv", None)
    if not p:
        return None

    df = pd.read_csv(p)
    required = {"assay", "endpoint", "title", "y_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing columns: {missing}")

    df["assay"] = df["assay"].astype(str).str.lower()
    df["endpoint"] = df["endpoint"].astype(str)
    return df


def _label_for_axis(label_map: pd.DataFrame | None, assay: str, endpoint: str) -> str | None:
    if label_map is None:
        return None
    m = label_map[(label_map["assay"] == assay.lower()) & (label_map["endpoint"] == str(endpoint))]
    if m.empty:
        return None
    v = m.iloc[0].get("y_label", None)
    return None if pd.isna(v) else str(v)


def _label_for_title(label_map: pd.DataFrame | None, assay: str, endpoint: str) -> str | None:
    if label_map is None:
        return None
    m = label_map[(label_map["assay"] == assay.lower()) & (label_map["endpoint"] == str(endpoint))]
    if m.empty:
        return None
    v = m.iloc[0].get("title", None)
    return None if pd.isna(v) else str(v)


def _auto_specs_from_endpoints(
    *,
    config: Dict[str, Any],
    fig_cfg: Dict[str, Any],
    assay: str,
    df_columns: List[str],
    label_map=None,
) -> List[Dict[str, Any]]:
    auto = fig_cfg.get("auto_from_endpoints", {}) or {}
    if not auto.get("enabled", False):
        return []

    source = str(auto.get("source", "secondary_two_way_anova")).strip()
    exclude = set([str(x) for x in (auto.get("exclude", []) or [])])

    if source not in config:
        raise ValueError(f"figures.auto_from_endpoints.source='{source}' not found in config")
    endpoints_by_assay = (config.get(source, {}) or {}).get("endpoints", {}) or {}
    endpoints = endpoints_by_assay.get(assay.lower(), []) or []

    if not isinstance(endpoints, list):
        raise ValueError(f"{source}.endpoints.{assay} must be a list")

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

        row = None
        if label_map is not None:
            m = label_map[
                (label_map["assay"] == assay.lower()) &
                (label_map["endpoint"] == endpoint)
            ]
            if not m.empty:
                row = m.iloc[0]

        out_specs.append(
            {
                "name": endpoint,
                "assay": assay.lower(),
                "type": "boxpoints",
                "value_label": (
                    row["y_label"]
                    if row is not None and pd.notna(row["y_label"])
                    else endpoint
                ),
                "title": (
                    False
                    if not bool(fig_cfg.get("auto_from_endpoints", {}).get("show_titles", False))
                    else (
                        row["title"]
                        if row is not None and pd.notna(row["title"])
                        else None
                    )
                ),
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
    label_map=None,
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

        missing = [c for c in [x, y] if c not in df_columns]
        if missing:
            raise ValueError(f"[figures:auto_from_correlations] {assay}:{name} missing columns: {missing}")

        x_lab = _label_for_axis(label_map, assay, x) or x
        y_lab = _label_for_axis(label_map, assay, y) or y

        x_title = _label_for_title(label_map, assay, x) or x
        y_title = _label_for_title(label_map, assay, y) or y
        auto_title = f"{y_title} vs {x_title}"

        out_specs.append(
            {
                "name": f"scatter_{name}",
                "assay": assay.lower(),
                "type": "scatter",
                "x": x,
                "y": y,
                "x_label": x_lab,
                "y_label": y_lab,
                "title": auto_title,
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
    assay_config_key: str | None = None,
    output_prefix: str | None = None,
    mouse_id_col: str = "mouse_id",
    group_col: str = "group",
    sex_col: str = "sex",
) -> List[Path]:
    fig_cfg = config.get("figures", {})
    if not fig_cfg or not fig_cfg.get("enabled", False):
        return []

    defaults = fig_cfg.get("defaults", {}) or {}
    assay_config_key = (assay_config_key or assay).lower()
    output_prefix = output_prefix or assay.upper()
    figure_size_cfg = defaults.get("figure_size", [5.2, 4.2])
    if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
        raise ValueError("figures.defaults.figure_size must be a 2-element list, e.g. [5.2, 4.2]")

    style = PlotStyle(
        group_edgecolors=defaults.get("group_edgecolors", {}),
        sex_markers=defaults.get("sex_markers", {}),
        group_gap=float(defaults.get("group_gap", 0.6)),
        box_linewidth=float(defaults.get("box_linewidth", 1.1)),
        median_linewidth=float(defaults.get("median_linewidth", 1.1)),
        whisker_linewidth=float(defaults.get("whisker_linewidth", 1.0)),
        cap_linewidth=float(defaults.get("cap_linewidth", 1.0)),
        box_width=float(defaults.get("box_width", 0.45)),
        sex_dodge=float(defaults.get("sex_dodge", 0.06)),
        jitter_sd=float(defaults.get("jitter_sd", 0.18)),
        point_size=float(defaults.get("point_size", 28.0)),
        x_gap=float(defaults.get("x_gap", 0.8)),
        point_alpha=float(defaults.get("point_alpha", 0.9)),
        figure_size=(float(figure_size_cfg[0]), float(figure_size_cfg[1])),
        dpi=int(defaults.get("dpi", 300)),
        x_margin=float(defaults.get("x_margin", 0.45)),
        title_pad=float(defaults.get("title_pad", 18.0)),
        group_label_y=float(defaults.get("group_label_y", -0.12)),
        ylim_top_mult=float(defaults.get("ylim_top_mult", 1.38)),
        di_ylim=tuple(defaults.get("di_ylim", [-1.0, 1.0])),
        di_zero_line=bool(defaults.get("di_zero_line", True)),
        di_zero_linewidth=float(defaults.get("di_zero_linewidth", 1.0)),
        di_zero_linestyle=str(defaults.get("di_zero_linestyle", "--")),
        di_ticks=tuple(defaults.get("di_ticks", [-1.0, -0.5, 0.0, 0.5, 1.0])),
        one_sample_text_position=str(defaults.get("one_sample_text_position", "top_left")),
        one_sample_fontsize=float(defaults.get("one_sample_fontsize", 10.0)),
        title_fontsize=float(defaults.get("title_fontsize", 10.0)),
        axis_label_fontsize=float(defaults.get("axis_label_fontsize", 9.0)),
        tick_label_fontsize=float(defaults.get("tick_label_fontsize", 8.0)),
        group_label_fontsize=float(defaults.get("group_label_fontsize", 9.0)),
        annotation_fontsize=float(defaults.get("annotation_fontsize", 9.0)),
        bracket_fontsize=float(defaults.get("bracket_fontsize", 10.0)),
        spine_linewidth=float(defaults.get("spine_linewidth", 1.2)),
        tick_width=float(defaults.get("tick_width", 1.0)),
        tick_length=float(defaults.get("tick_length", 4.0)),
    )

    formats = defaults.get("formats", ["png"])
    if not isinstance(formats, list) or not formats:
        raise ValueError("figures.defaults.formats must be a non-empty list, e.g. ['png'].")

    for fmt in formats:
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Unsupported format: {fmt}")

    processed_path = processed_dir / f"{assay.upper()}_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)
    require_columns(df, [mouse_id_col, group_col, sex_col])

    written: List[Path] = []
    label_map = _load_label_map(config)

    specs = fig_cfg.get("specs", []) or []
    if not isinstance(specs, list):
        raise ValueError("figures.specs must be a list")

    auto_specs = _auto_specs_from_endpoints(
        config=config,
        fig_cfg=fig_cfg,
        assay=assay_config_key,
        df_columns=list(df.columns),
        label_map=label_map,
    )
    auto_scatter_specs = _auto_scatter_specs_from_correlations(
        config=config,
        fig_cfg=fig_cfg,
        assay=assay_config_key,
        df_columns=list(df.columns),
        label_map=label_map,
    )

    manual_names = {str(s.get("name")) for s in specs if isinstance(s, dict) and "name" in s}
    specs.extend([s for s in auto_scatter_specs if str(s.get("name")) not in manual_names])

    manual_names = {str(s.get("name")) for s in specs if isinstance(s, dict) and "name" in s}
    specs.extend([s for s in auto_specs if str(s.get("name")) not in manual_names])

    for spec in specs:
        if str(spec.get("assay", "")).lower() != assay_config_key.lower():
            continue

        plot_type = str(spec.get("type", "")).lower()
        if plot_type not in ("boxpoints", "scatter"):
            raise ValueError(f"Unsupported figure type: {spec.get('type')}")

        name = str(spec.get("name", f"{assay}_figure"))
        stem = f"{output_prefix}_{name}"
        out_paths = [out_figures / f"{stem}.{fmt}" for fmt in formats]

        if plot_type == "boxpoints":
            value_label = spec.get("value_label", None)
            x_order = spec.get("x_order", None)
            group_order = spec.get("group_order", defaults.get("group_order", None))
            sex_order = spec.get("sex_order", ["female", "male"])

            w2l = spec.get("wide_to_long", None)
            if not w2l:
                raise ValueError(f"Figure spec '{name}' missing wide_to_long.")

            x_col = str(w2l.get("x_col", "x"))
            mapping = w2l.get("mapping", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Figure spec '{name}' wide_to_long.mapping must be a non-empty dict.")

            id_cols = [mouse_id_col, group_col, sex_col]
            long_df = _make_long_from_mapping(df, x_col=x_col, mapping=mapping, id_cols=id_cols)

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
                x_gap=float(spec.get("x_gap", style.x_gap)),
                group_gap=float(spec.get("group_gap", style.group_gap)),
                figure_size=tuple(spec.get("figure_size", style.figure_size)),
                dpi=int(spec.get("dpi", style.dpi)),
                x_margin=float(spec.get("x_margin", style.x_margin)),
                title_pad=float(spec.get("title_pad", style.title_pad)),
                group_label_y=float(spec.get("group_label_y", style.group_label_y)),
                ylim_top_mult=float(spec.get("ylim_top_mult", style.ylim_top_mult)),
                di_ylim=tuple(spec.get("di_ylim", style.di_ylim)),
                di_zero_line=bool(spec.get("di_zero_line", style.di_zero_line)),
                di_zero_linewidth=float(spec.get("di_zero_linewidth", style.di_zero_linewidth)),
                di_zero_linestyle=str(spec.get("di_zero_linestyle", style.di_zero_linestyle)),
                di_ticks=tuple(spec.get("di_ticks", style.di_ticks)),
                one_sample_text_position=str(spec.get("one_sample_text_position", style.one_sample_text_position)),
                one_sample_fontsize=float(spec.get("one_sample_fontsize", style.one_sample_fontsize)),
                title_fontsize=float(spec.get("title_fontsize", style.title_fontsize)),
                axis_label_fontsize=float(spec.get("axis_label_fontsize", style.axis_label_fontsize)),
                tick_label_fontsize=float(spec.get("tick_label_fontsize", style.tick_label_fontsize)),
                group_label_fontsize=float(spec.get("group_label_fontsize", style.group_label_fontsize)),
                annotation_fontsize=float(spec.get("annotation_fontsize", style.annotation_fontsize)),
                bracket_fontsize=float(spec.get("bracket_fontsize", style.bracket_fontsize)),
                spine_linewidth=float(spec.get("spine_linewidth", style.spine_linewidth)),
                tick_width=float(spec.get("tick_width", style.tick_width)),
                tick_length=float(spec.get("tick_length", style.tick_length)),
                
            )

            sig_cfg = (defaults.get("significance", {}) or {}).copy()
            sig_cfg.update(spec.get("significance", {}) or {})

            anova_cfg = (sig_cfg.get("anova_summary", {}) or {}).copy()
            anova_df = None

            if bool(sig_cfg.get("enabled", False)) and bool(anova_cfg.get("enabled", False)):
                table_key = str(anova_cfg.get("table", "secondary_two_way_anova")).strip()
                table_path = out_figures.parent / "tables" / f"{output_prefix}_{table_key}.csv"
                if table_path.exists():
                    anova_df = pd.read_csv(table_path)

            anova_endpoint = None
            mapped = [str(m) for m in mapping.values() if m is not None]
            if len(set(mapped)) == 1:
                anova_endpoint = mapped[0]

            one_sample_df = None
            one_sample_cfg = (sig_cfg.get("one_sample_summary", {}) or {}).copy()

            if bool(sig_cfg.get("enabled", False)) and bool(one_sample_cfg.get("enabled", False)):
                table_key = str(one_sample_cfg.get("table", "primary_one_sample")).strip()
                table_path = out_figures.parent / "tables" / f"{output_prefix}_{table_key}.csv"
                if table_path.exists():
                    one_sample_df = pd.read_csv(table_path)

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
                title=spec.get("title", None),
                y_label=value_label,
                show=False,
                anova_df=anova_df,
                anova_cfg=anova_cfg,
                one_sample_df=one_sample_df,
                one_sample_cfg=one_sample_cfg,
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
                    x_label=spec.get("x_label", None) or (_label_for_axis(label_map, assay, str(x)) or str(x)),
                    y_label=spec.get("y_label", None) or (_label_for_axis(label_map, assay, str(y)) or str(y)),
                    title=spec.get("title", None) or (
                        f"{_label_for_title(label_map, assay, str(y)) or str(y)} vs "
                        f"{_label_for_title(label_map, assay, str(x)) or str(x)}"
                    ),
                    fit_enabled=fit_enabled,
                )

            written.extend(out_paths)

    return written