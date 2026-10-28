from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import PlotStyle, infer_order


def plot_group_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    out_path: str,
    style: PlotStyle,
    group_col: str = "group",
    sex_col: str = "sex",
    group_order: Optional[Sequence[str]] = None,
    sex_order: Optional[Sequence[str]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title=None,
    fit_enabled: bool = True,
) -> None:
    """
    Scatter plot of y vs x: colour = group, marker = sex (hollow), with a legend for both.
    Optional pooled OLS fit line (descriptive only; the statistics are in the correlation table).
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"scatter missing columns: x='{x}' or y='{y}'")

    d = df[[x, y, group_col, sex_col]].dropna().copy()

    groups = infer_order(d[group_col], group_order)
    sexes = infer_order(d[sex_col], sex_order or ["female", "male"])

    fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)

    if style.transparent_background:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

    # pooled fit line (simple OLS), drawn first so points sit on top
    if fit_enabled and len(d) >= 3:
        X = d[x].to_numpy(dtype=float)
        Y = d[y].to_numpy(dtype=float)
        b, a = np.polyfit(X, Y, deg=1)  # y = a + b x
        xgrid = np.linspace(np.nanmin(X), np.nanmax(X), 100)
        ax.plot(xgrid, a + b * xgrid, color="0.45", linewidth=style.spine_linewidth * 1.4, zorder=1)

    # points: marker by sex, edgecolor by group
    for g in groups:
        dg = d[d[group_col] == g]
        edge = style.group_edgecolors.get(str(g), "black")

        for sx in sexes:
            dgs = dg[dg[sex_col] == sx]
            if dgs.empty:
                continue
            marker = style.sex_markers.get(str(sx), "o")
            ax.scatter(
                dgs[x].to_numpy(),
                dgs[y].to_numpy(),
                s=style.point_size * 1.6,
                alpha=style.point_alpha,
                marker=marker,
                edgecolors=edge,
                facecolors="none",
                linewidths=1.0,
                zorder=2,
            )

    # legend: groups by colour, sexes by marker (identity never by colour alone)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markerfacecolor=style.group_edgecolors.get(str(g), "black"),
               markeredgecolor="none", markersize=5, label=style.group_labels.get(str(g), str(g).capitalize()))
        for g in groups
    ] + [
        Line2D([], [], linestyle="none", marker=style.sex_markers.get(str(sx), "o"), markerfacecolor="none",
               markeredgecolor=style.text_color, markersize=5, label=str(sx).capitalize())
        for sx in sexes
    ]
    ax.legend(handles=handles, frameon=False, fontsize=style.annotation_fontsize, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, borderaxespad=0.2)

    ax.set_xlabel(x_label or x, fontsize=style.axis_label_fontsize, color=style.text_color)
    ax.set_ylabel(y_label or y, fontsize=style.axis_label_fontsize, color=style.text_color)
    if title:
        ax.set_title(title, fontsize=style.title_fontsize, color=style.text_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.axis_color)
        ax.spines[side].set_linewidth(style.spine_linewidth)

    ax.tick_params(
        color=style.axis_color,
        labelcolor=style.text_color,
        width=style.tick_width,
        length=style.tick_length,
        labelsize=style.tick_label_fontsize,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=style.dpi,
        transparent=style.transparent_background,
        bbox_inches="tight",
    )
    plt.close(fig)
