from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import PlotStyle, infer_order, require_columns
from .common import jitter_from_ids



def plot_group_boxpoints(
    df: pd.DataFrame,
    *,
    value_col: str,
    x_col: str,
    group_col: str,
    sex_col: str,
    mouse_id_col: Optional[str],
    assay: str,
    endpoint: str,
    out_paths: List[Path],
    x_order: Optional[List[str]] = None,
    group_order: Optional[List[str]] = None,
    sex_order: Optional[List[str]] = None,
    style: Optional[PlotStyle] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Draw Prism-like box+whisker plots with individual points:
    - edgecolor of box = group color
    - points overlaid with sex-specific markers
    - x-axis categories from x_col (e.g., Familiar/Novel)
    - groups shown as separated blocks with a gap (like your example)
    """
    if style is None:
        raise ValueError("plot_group_boxpoints requires a PlotStyle (pass from config).")
    
    if mouse_id_col is None or str(mouse_id_col).strip() == "":
       raise ValueError("mouse_id_col is required for stable jittered points.")

    require_columns(df, [value_col, x_col, group_col, sex_col])
    d = df[[value_col, x_col, group_col, sex_col] + ([mouse_id_col] if mouse_id_col else [])].copy()
    d = d.dropna(subset=[value_col, x_col, group_col, sex_col])

    # Orders (deterministic)
    x_levels = infer_order(d[x_col].astype(str).tolist(), preferred=x_order)
    g_levels = infer_order(d[group_col].astype(str).tolist(), preferred=group_order)
    s_levels = infer_order(d[sex_col].astype(str).tolist(), preferred=sex_order)

    if len(x_levels) < 1 or len(g_levels) < 1:
        raise ValueError("No levels found for x or group after dropping NAs.")

    # Layout: for each group, plot the x_levels as adjacent boxes, and leave a gap between groups
    n_x = len(x_levels)

    positions = {}
    xticks = []
    xticklabels = []

    base = 1.0

    if n_x == 1:
        # Simple layout: groups side-by-side for single x category (e.g. Total)
        x = x_levels[0]
        for gi, g in enumerate(g_levels):
            pos = base + gi * (1.0 + style.group_gap)
            positions[(g, x)] = pos
            xticks.append(pos)
            xticklabels.append(str(x))  # keep "Total" under each group
    else:
        # Block layout: x categories within each group, with group_gap between blocks
        group_block_width = n_x + 0.4
        for gi, g in enumerate(g_levels):
            block_start = base + gi * (group_block_width + style.group_gap)
            for xi, x in enumerate(x_levels):
                pos = block_start + xi
                positions[(g, x)] = pos
                xticks.append(pos)
                xticklabels.append(str(x))


    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=300)

    # Draw boxes per (group, x)
    for g in g_levels:
        edge = style.group_edgecolors.get(str(g).lower(), "black")
        for x in x_levels:
            sub = d[(d[group_col].astype(str) == str(g)) & (d[x_col].astype(str) == str(x))]
            y = sub[value_col].astype(float).to_numpy()
            pos = positions[(g, x)]

            if y.size == 0:
                continue

            bp = ax.boxplot(
                [y],
                positions=[pos],
                widths=style.box_width,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
            )

            # box styling: white fill, colored edge
            for box in bp["boxes"]:
                box.set(facecolor="white", edgecolor=edge, linewidth=style.box_linewidth)
            for med in bp["medians"]:
                med.set(color=edge, linewidth=style.median_linewidth)
            for whisk in bp["whiskers"]:
                whisk.set(color=edge, linewidth=style.whisker_linewidth)
            for cap in bp["caps"]:
                cap.set(color=edge, linewidth=style.cap_linewidth)

            # Overlay points, sex-specific markers.


            # plot per sex to get marker shapes
            # small deterministic dodge so male/female don't sit on top of each other

            for s in s_levels:
                sub_s = sub[sub[sex_col].astype(str) == str(s)]
                if sub_s.empty:
                    continue
                ys = sub_s[value_col].astype(float).to_numpy()

                if mouse_id_col:
                    mids_s = sub_s[mouse_id_col].astype(str).tolist()
                    jit_s = jitter_from_ids(mids_s, sd=style.jitter_sd)

                    sex_dodge = style.sex_dodge
                    offset = (-sex_dodge / 2.0) if str(s).lower() == "female" else (sex_dodge / 2.0)
                    xjs = pos + jit_s + offset

                ax.scatter(
                    xjs,
                    ys,
                    s=style.point_size,
                    marker= style.sex_markers.get(str(s).lower(), "o"),
                    alpha=style.point_alpha,
                    c="black",
                    edgecolors="black",
                    linewidths=0.0,
                    zorder=3,
                )


    # x-axis ticks/labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)

    # Add group labels under blocks (optional but helpful)
    # Place group label centered under each block
    ymin, ymax = ax.get_ylim()
    y_text = ymin - 0.08 * (ymax - ymin)
    for gi, g in enumerate(g_levels):
        block_positions = [positions[(g, x)] for x in x_levels]
        center = float(np.mean(block_positions))
        ax.text(center, y_text, str(g).capitalize(), ha="center", va="top", fontsize=10)

    # Labels / title
    ax.set_title(title or f"{assay} — {endpoint}", fontsize=12)
    ax.set_ylabel(y_label or endpoint)
    ax.set_xlabel("")

    # Aesthetic: Prism-like axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2)

    fig.tight_layout()

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
