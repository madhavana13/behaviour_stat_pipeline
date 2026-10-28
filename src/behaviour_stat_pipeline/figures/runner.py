from __future__ import annotations

import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .boxpoints import plot_group_boxpoints
from .common import PlotStyle, require_columns
from .km import plot_km
from .scatter import plot_group_scatter
from behaviour_stat_pipeline.stats.endpoints.correlations import resolve_corr_specs
from behaviour_stat_pipeline.common.exclusions import apply_exclusions_for_endpoint
from behaviour_stat_pipeline.stats.common import check_groups_in_data
from behaviour_stat_pipeline.stats.endpoints.time_to_event import build_survival_data

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

        if row is None:
            print(f"WARNING [figures]: no label for {assay.upper()},{endpoint} in labels.csv; axis shows the column name")
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

def _exclusion_summary(applied: pd.DataFrame) -> dict:
    if applied is None or applied.empty:
        return {
            "n_excluded": 0,
            "excluded_mouse_ids": "",
            "exclusion_reasons": "",
        }

    return {
        "n_excluded": applied["mouse_id"].nunique(),
        "excluded_mouse_ids": ";".join(
            sorted(applied["mouse_id"].astype(str).unique())
        ),
        "exclusion_reasons": "; ".join(
            sorted(applied["reason"].astype(str).unique())
        ),
    }


def _apply_plot_exclusions(
    df: pd.DataFrame,
    exclusions: pd.DataFrame | None,
    *,
    assay: str,
    endpoint: str,
    mouse_id_col: str,
) -> tuple[pd.DataFrame, dict]:
    if exclusions is None or exclusions.empty:
        return df.copy(), _exclusion_summary(pd.DataFrame())

    df_plot, applied = apply_exclusions_for_endpoint(
        df,
        exclusions,
        assay=assay.upper(),
        endpoint=endpoint,
        analysis_scope="primary",
        mouse_id_col=mouse_id_col,
    )

    return df_plot, _exclusion_summary(applied)


def _title_for(spec: Dict[str, Any], defaults: Dict[str, Any], fallback: str):
    """
    A title only when asked for: a title string in the figure spec, or defaults.show_title: true
    (then `fallback`). Otherwise False (no title).
    """
    t = spec.get("title", None)
    if isinstance(t, str) and t.strip():
        return t
    if t is False:
        return False
    return fallback if bool(defaults.get("show_title", False)) else False


def _mask_excluded(
    df: pd.DataFrame,
    exclusions: pd.DataFrame | None,
    *,
    assay: str,
    endpoints: List[str],
    mouse_id_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Per-endpoint exclusions for figures that show several endpoints (scatter x/y, multi-panel
    box plots): an excluded mouse's value is blanked only for the endpoint it is excluded from,
    exactly as in the statistics.
    """
    out = df.copy()
    applied_all = []
    if exclusions is not None and not exclusions.empty:
        for ep in dict.fromkeys(endpoints):
            if ep not in out.columns:
                continue
            _, applied = apply_exclusions_for_endpoint(
                out, exclusions, assay=assay.upper(), endpoint=ep,
                analysis_scope="primary", mouse_id_col=mouse_id_col,
            )
            if not applied.empty:
                ids = set(applied["mouse_id"].astype(str))
                out.loc[out[mouse_id_col].astype(str).isin(ids), ep] = float("nan")
                applied_all.append(applied)
    applied = pd.concat(applied_all, ignore_index=True) if applied_all else pd.DataFrame()
    return out, _exclusion_summary(applied)


def _move_superseded(folder: Path, prefix: str, written: List[Path], suffixes=(".png", ".pdf")) -> List[Path]:
    """
    Files of this assay that the current run did not produce are moved to <folder>/_superseded/,
    so old figures never sit next to current ones. Files not named <prefix>_* are left alone.
    """
    keep = {Path(p).resolve() for p in written}
    moved = []
    for f in sorted(folder.glob(f"{prefix}_*")):
        if f.is_file() and f.suffix.lower() in suffixes and f.resolve() not in keep:
            dest = folder / "_superseded"
            dest.mkdir(exist_ok=True)
            target = dest / f.name
            if target.exists():
                target.unlink()
            f.rename(target)
            moved.append(target)
    return moved


def _write_figure_log(log_path: Path, records: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        log_path.write_text("No figures written.\n", encoding="utf-8")
        return

    lines = []
    for r in records:
        lines.append(f"figure_name: {r.get('figure_name', '')}")
        lines.append(f"plot_type: {r.get('plot_type', '')}")
        lines.append(f"endpoint: {r.get('endpoint', '')}")
        lines.append(f"n_rows_processed_pre_exclusion: {r.get('n_rows_processed_pre_exclusion', '')}")
        lines.append(f"n_rows_plotted_after_exclusion: {r.get('n_rows_plotted_after_exclusion', '')}")
        lines.append(f"n_mice_processed_pre_exclusion: {r.get('n_mice_processed_pre_exclusion', '')}")
        lines.append(f"n_mice_plotted_after_exclusion: {r.get('n_mice_plotted_after_exclusion', '')}")
        lines.append(f"n_excluded: {r.get('n_excluded', 0)}")
        lines.append(f"excluded_mouse_ids: {r.get('excluded_mouse_ids', '')}")
        lines.append(f"exclusion_reasons: {r.get('exclusion_reasons', '')}")
        lines.append("output_files:")
        for p in r.get("output_files", []):
            lines.append(f"  - {p}")
        lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")

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

    corr_specs = resolve_corr_specs(config, assay)  # same resolution as the statistics
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


        out_specs.append(
            {
                "name": f"scatter_{name}",
                "assay": assay.lower(),
                "type": "scatter",
                "x": x,
                "y": y,
                "x_label": x_lab,
                "y_label": y_lab,
                "fit": {"enabled": fit_enabled, "by": "pooled"},
            }
        )

    return out_specs


def _auto_km_specs_from_time_to_event(
    *,
    config: Dict[str, Any],
    fig_cfg: Dict[str, Any],
    assay: str,
    df_columns: List[str],
    label_map=None,
) -> List[Dict[str, Any]]:
    """One pooled-sex KM figure per time_to_event endpoint, plus a by-sex version if enabled."""
    auto = fig_cfg.get("auto_from_time_to_event", {}) or {}
    if not auto.get("enabled", False):
        return []

    endpoints = ((config.get("time_to_event", {}) or {}).get("endpoints", {}) or {}).get(assay.lower(), []) or []
    missing = [e for e in endpoints if str(e) not in df_columns]
    if missing:
        raise ValueError(f"[figures:auto_from_time_to_event] {assay}: endpoints not in processed CSV: {missing}")

    # The female | male version only makes sense when sex is a factor of the analysis
    tte_factors = (config.get("time_to_event", {}) or {}).get("factors", ["group", "sex"]) or []
    variants = [False] + ([True] if auto.get("by_sex", True) and "sex" in tte_factors else [])
    out_specs: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        endpoint = str(endpoint)
        for by_sex in variants:
            out_specs.append({
                "name": f"km_{endpoint}" + ("_by_sex" if by_sex else ""),
                "assay": assay.lower(),
                "type": "km",
                "endpoint": endpoint,
                "by_sex": by_sex,
                "x_label": _label_for_axis(label_map, assay, endpoint) or f"{endpoint} (s)",
                "y_label": auto.get("y_label", "Proportion without event"),
                "title": _label_for_title(label_map, assay, endpoint) if auto.get("show_titles", False) else None,
                "figure_size": auto.get("figure_size_by_sex" if by_sex else "figure_size"),
                "linewidth": auto.get("linewidth", 1.2),
                "censor_marker_size": auto.get("censor_marker_size", 5.0),
            })
    return out_specs


_STYLE_FIXED = {"group_edgecolors", "group_labels", "sex_markers"}  # set from the experiment / defaults only


def style_with_overrides(base: PlotStyle, overrides: Dict[str, Any]) -> PlotStyle:
    """
    Copy of `base` with every PlotStyle field present in `overrides` (figures.yaml defaults or a
    figure spec) replaced, converted to the type of the base value. Other keys are ignored here.
    """
    changes = {}
    for f in fields(PlotStyle):
        if f.name in _STYLE_FIXED or f.name not in overrides or overrides[f.name] is None:
            continue
        current, value = getattr(base, f.name), overrides[f.name]
        if isinstance(current, tuple):
            changes[f.name] = tuple(float(v) for v in value)
        elif isinstance(current, bool):
            changes[f.name] = bool(value)
        elif isinstance(current, int):
            changes[f.name] = int(value)
        elif isinstance(current, float):
            changes[f.name] = float(value)
        else:
            changes[f.name] = str(value)
    return replace(base, **changes)


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
    exclusions: pd.DataFrame | None = None,
    trial_duration_s: float | None = None,
    group_order: List[str] | None = None,
    group_labels: Dict[str, str] | None = None,
    group_colors: Dict[str, str] | None = None,
) -> List[Path]:
    """
    group_order / group_labels / group_colors come from the experiment config `groups:` block
    (the only place they are defined); figures.yaml no longer holds group-specific settings.
    """
    fig_cfg = config.get("figures", {})
    if not fig_cfg or not fig_cfg.get("enabled", False):
        return []

    defaults = fig_cfg.get("defaults", {}) or {}
    assay_config_key = (assay_config_key or assay).lower()
    output_prefix = output_prefix or assay.upper()
    figure_size_cfg = defaults.get("figure_size", [5.2, 4.2])
    if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
        raise ValueError("figures.defaults.figure_size must be a 2-element list, e.g. [5.2, 4.2]")

    style = style_with_overrides(
        PlotStyle(
            group_edgecolors=dict(group_colors or {}),
            group_labels=dict(group_labels or {}),
            sex_markers=defaults.get("sex_markers", {}),
        ),
        {**defaults, "figure_size": (float(figure_size_cfg[0]), float(figure_size_cfg[1]))},
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
    if not group_order:
        raise ValueError("Figures need the experiment's groups.order (experiment config).")
    check_groups_in_data(df[group_col], list(group_order), where=f"figures {output_prefix}")
    for key in ("group_order", "group_edgecolors"):
        if key in defaults:
            raise ValueError(f"figures.yaml defaults.{key} is no longer used: set groups in the experiment config")

    written: List[Path] = []
    label_map = _load_label_map(config)
    figure_log_records: list[dict] = []

    specs = fig_cfg.get("specs", []) or []
    if not isinstance(specs, list):
        raise ValueError("figures.specs must be a list")
    specs = [dict(s) for s in specs]  # copy: the config's list must not grow with auto specs
    for i, s in enumerate(specs):
        if not s.get("name"):
            mapped = [str(m) for m in ((s.get("wide_to_long") or {}).get("mapping") or {}).values() if m]
            # A spec for one endpoint takes its name, so it replaces the automatic figure of that endpoint
            s["name"] = mapped[0] if len(set(mapped)) == 1 else f"{s.get('assay', assay)}_figure_{i}"

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

    auto_km_specs = _auto_km_specs_from_time_to_event(
        config=config,
        fig_cfg=fig_cfg,
        assay=assay_config_key,
        df_columns=list(df.columns),
        label_map=label_map,
    )
    manual_names = {str(s.get("name")) for s in specs if isinstance(s, dict) and "name" in s}
    specs.extend([s for s in auto_km_specs if str(s.get("name")) not in manual_names])

    for spec in specs:
        if str(spec.get("assay", "")).lower() != assay_config_key.lower():
            continue
        if "group_order" in spec:
            raise ValueError(
                f"figures.yaml spec {spec.get('name', spec.get('type'))!r} sets group_order; group order comes "
                f"from the experiment config (groups.order) only"
            )

        plot_type = str(spec.get("type", "")).lower()
        if plot_type not in ("boxpoints", "scatter", "km"):
            raise ValueError(f"Unsupported figure type: {spec.get('type')}")

        name = str(spec.get("name", f"{assay}_figure"))
        stem = f"{output_prefix}_{name}"
        out_paths = [out_figures / f"{stem}.{fmt}" for fmt in formats]

        if plot_type == "boxpoints":
            value_label = spec.get("value_label", None)
            x_order = spec.get("x_order", None)
            sex_order = spec.get("sex_order", ["female", "male"])

            w2l = spec.get("wide_to_long", None)
            if not w2l:
                raise ValueError(f"Figure spec '{name}' missing wide_to_long.")

            x_col = str(w2l.get("x_col", "x"))
            mapping = w2l.get("mapping", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Figure spec '{name}' wide_to_long.mapping must be a non-empty dict.")

            id_cols = [mouse_id_col, group_col, sex_col]
            mapped_endpoints = [str(m) for m in mapping.values() if m is not None]
            unique_mapped_endpoints = sorted(set(mapped_endpoints))

            if len(unique_mapped_endpoints) == 1:
                exclusion_endpoint = unique_mapped_endpoints[0]
                df_plot, excl_info = _apply_plot_exclusions(
                    df,
                    exclusions,
                    assay=assay,
                    endpoint=exclusion_endpoint,
                    mouse_id_col=mouse_id_col,
                )
            else:
                # several endpoints in one figure: exclusions apply per endpoint (as in the statistics)
                exclusion_endpoint = ";".join(unique_mapped_endpoints)
                df_plot, excl_info = _mask_excluded(
                    df, exclusions, assay=assay, endpoints=unique_mapped_endpoints, mouse_id_col=mouse_id_col,
                )

            long_df = _make_long_from_mapping(
                df_plot,
                x_col=x_col,
                mapping=mapping,
                id_cols=id_cols,
            )

            style_spec = style_with_overrides(style, spec)

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
                title=_title_for(spec, defaults, _label_for_title(label_map, assay, anova_endpoint or "") or name),
                y_label=value_label,
                show=False,
                anova_df=anova_df,
                anova_cfg=anova_cfg,
                one_sample_df=one_sample_df,
                one_sample_cfg=one_sample_cfg,
            )

            written.extend(out_paths)
            figure_log_records.append({
                "figure_name": name,
                "plot_type": "boxpoints",
                "endpoint": exclusion_endpoint,
                "n_rows_processed_pre_exclusion": len(df),
                "n_rows_plotted_after_exclusion": len(df_plot),
                "n_mice_processed_pre_exclusion": df[mouse_id_col].nunique(),
                "n_mice_plotted_after_exclusion": df_plot[mouse_id_col].nunique(),
                **excl_info,
                "output_files": [str(p) for p in out_paths],
            })

        elif plot_type == "scatter":
            x = spec.get("x")
            y = spec.get("y")
            if not x or not y:
                raise ValueError(f"Figure spec '{name}' (scatter) requires x and y.")

            fit_cfg = spec.get("fit", {}) or {}
            fit_enabled = bool(fit_cfg.get("enabled", True))

            exclusion_endpoint = f"{x};{y}"
            df_plot, excl_info = _mask_excluded(
                df, exclusions, assay=assay, endpoints=[str(x), str(y)], mouse_id_col=mouse_id_col,
            )
            scatter_style = style_with_overrides(
                style, {"figure_size": spec.get("figure_size", defaults.get("scatter_figure_size", [3.2, 3.0]))}
            )

            for out_path in out_paths:
                plot_group_scatter(
                    df=df_plot,
                    x=str(x),
                    y=str(y),
                    out_path=str(out_path),
                    style=scatter_style,
                    group_col=group_col,
                    sex_col=sex_col,
                    group_order=group_order,
                    sex_order=spec.get("sex_order", ["female", "male"]),
                    x_label=spec.get("x_label", None) or (_label_for_axis(label_map, assay, str(x)) or str(x)),
                    y_label=spec.get("y_label", None) or (_label_for_axis(label_map, assay, str(y)) or str(y)),
                    title=_title_for(spec, defaults, (
                        f"{_label_for_title(label_map, assay, str(y)) or str(y)} vs "
                        f"{_label_for_title(label_map, assay, str(x)) or str(x)}"
                    )),
                    fit_enabled=fit_enabled,
                )

            written.extend(out_paths)
            figure_log_records.append({
                "figure_name": name,
                "plot_type": "scatter",
                "endpoint": exclusion_endpoint,
                "n_rows_processed_pre_exclusion": len(df),
                "n_rows_plotted_after_exclusion": len(df_plot),
                "n_mice_processed_pre_exclusion": df[mouse_id_col].nunique(),
                "n_mice_plotted_after_exclusion": df_plot[mouse_id_col].nunique(),
                **excl_info,
                "output_files": [str(p) for p in out_paths],
            })

        elif plot_type == "km":
            endpoint = spec.get("endpoint")
            if not endpoint:
                raise ValueError(f"Figure spec '{name}' (km) requires endpoint.")
            endpoint = str(endpoint)

            df_plot, excl_info = _apply_plot_exclusions(
                df,
                exclusions,
                assay=assay,
                endpoint=endpoint,
                mouse_id_col=mouse_id_col,
            )
            # Same censoring rule as the statistics
            surv = build_survival_data(
                df_plot,
                endpoint,
                trial_duration_s=trial_duration_s,
                duration_col=(config.get("time_to_event", {}) or {}).get("duration_col") or None,
            ).dropna(subset=[group_col, sex_col])

            tests_path = out_figures.parent / "tables" / f"{output_prefix}_time_to_event.csv"
            tests_df = pd.read_csv(tests_path) if tests_path.exists() else None
            if tests_df is None and not spec.get("by_sex", False):
                print(f"WARNING [{output_prefix}]: {tests_path} not found; KM figure '{name}' has no test annotation. "
                      f"Run run_stats.py first.")

            plot_km(
                surv,
                out_paths=out_paths,
                style=style,
                group_col=group_col,
                sex_col=sex_col,
                group_order=group_order,
                sex_order=spec.get("sex_order", ["female", "male"]),
                by_sex=bool(spec.get("by_sex", False)),
                x_label=str(spec.get("x_label", "Time (s)")),
                y_label=str(spec.get("y_label", "Proportion without event")),
                title=spec.get("title", None) or None,
                tests=tests_df,
                endpoint=endpoint,
                linewidth=float(spec.get("linewidth", 1.2)),
                censor_marker_size=float(spec.get("censor_marker_size", 5.0)),
                figure_size=tuple(spec["figure_size"]) if spec.get("figure_size") else None,
            )

            written.extend(out_paths)
            figure_log_records.append({
                "figure_name": name,
                "plot_type": "km",
                "endpoint": exclusion_endpoint,
                "n_rows_processed_pre_exclusion": len(df),
                "n_rows_plotted_after_exclusion": len(df_plot),
                "n_mice_processed_pre_exclusion": df[mouse_id_col].nunique(),
                "n_mice_plotted_after_exclusion": df_plot[mouse_id_col].nunique(),
                **excl_info,
                "output_files": [str(p) for p in out_paths],
            })
    log_path = out_figures.parent / "logs" / f"{output_prefix}_figures_summary.txt"
    _write_figure_log(log_path, figure_log_records)
    for moved in _move_superseded(out_figures, output_prefix, written):
        print(f"NOTE [{output_prefix}]: not produced by this run, moved to {moved}")
    
    return written