from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    title: Optional[str] = None,
    fit_enabled: bool = True,
) -> None:
    """
    Scatter plot of y vs x with group color + sex marker aesthetics.
    Friday-safe: pooled OLS fit line (no CI).
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
                s=style.point_size,
                alpha=style.point_alpha,
                marker=marker,
                edgecolors=edge,
                facecolors="none",
                linewidths=1.2,
            )

    # pooled fit line (simple OLS)
    if fit_enabled and len(d) >= 3:
        X = d[x].to_numpy(dtype=float)
        Y = d[y].to_numpy(dtype=float)
        b, a = np.polyfit(X, Y, deg=1)  # y = a + b x
        xgrid = np.linspace(np.nanmin(X), np.nanmax(X), 100)
        ygrid = a + b * xgrid
        ax.plot(xgrid, ygrid, linewidth=1.5)

    ax.set_xlabel(x_label or x, color=style.text_color)
    ax.set_ylabel(y_label or y, color=style.text_color)
    if title:
        ax.set_title(title, color=style.text_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style.axis_color)
    ax.spines["bottom"].set_color(style.axis_color)

    ax.tick_params(
        color=style.axis_color,
        labelcolor=style.text_color,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=style.dpi,
        transparent=style.transparent_background,
        bbox_inches="tight",
    )
    plt.close(fig)
