from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .common import PlotStyle, require_columns, infer_order
from .boxpoints import plot_group_boxpoints


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

    out = pd.concat(pieces, axis=0, ignore_index=True)
    return out


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
        group_edgecolors=defaults.get("group_edgecolors", {"tail": "black", "cup": "orange"}),
        sex_markers=defaults.get("sex_markers", {"female": "o", "male": "^"}),
    )
    formats = defaults.get("formats", ["png", "pdf"])

    written: List[Path] = []

    # load processed data
    processed_path = processed_dir / f"{assay.upper()}_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    require_columns(df, [mouse_id_col, group_col, sex_col])

    specs = fig_cfg.get("specs", [])
    for spec in specs:
        if str(spec.get("assay", "")).lower() != assay.lower():
            continue
        if str(spec.get("type", "")).lower() != "boxpoints":
            raise ValueError(f"Unsupported figure type: {spec.get('type')}")

        name = str(spec.get("name", f"{assay}_figure"))
        value_label = spec.get("value_label", None)
        x_order = spec.get("x_order", None)
        group_order = spec.get("group_order", None)
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

        # Output filenames
        # one figure per spec (contains all x levels in that mapping)
        stem = f"{assay.upper()}_{name}"
        for fmt in formats:
            if fmt not in ("png", "pdf"):
                raise ValueError(f"Unsupported format: {fmt}")

        out_png = out_figures / f"{stem}.png"
        out_pdf = out_figures / f"{stem}.pdf"

        plot_group_boxpoints(
            long_df,
            value_col="value",
            x_col=x_col,
            group_col=group_col,
            sex_col=sex_col,
            mouse_id_col=mouse_id_col,
            assay=assay.upper(),
            endpoint=name,
            out_png=out_png,
            out_pdf=out_pdf,
            x_order=x_order,
            group_order=group_order,
            sex_order=sex_order,
            style=style,
            title=None,
            y_label=value_label,
            show=False,
        )

        written.extend([out_png])

    return written
