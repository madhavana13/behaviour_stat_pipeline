from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import PlotStyle, infer_order, jitter_from_ids, require_columns

def _is_di_endpoint(endpoint: str) -> bool:
    e = str(endpoint).strip().lower()
    return e.startswith("di_") or e.endswith("_di") or e == "di"

def _draw_corner_text(ax, text: str, position: str = "top_left", fontsize: float = 10.0) -> None:
    if not text:
        return

    if position == "top_right":
        x, ha = 0.98, "right"
    else:
        x, ha = 0.02, "left"

    ax.text(
        x,
        0.98,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize,
    )

def _collect_one_sample_labels(
    one_sample_df: pd.DataFrame,
    *,
    endpoint: str,
    endpoint_col: str,
    group_col: str,
    label_col: str,
) -> dict:
    out = {}
    sub = one_sample_df[one_sample_df[endpoint_col].astype(str) == str(endpoint)].copy()
    if sub.empty:
        return out

    for _, r in sub.iterrows():
        g = str(r[group_col])
        out[g] = _stars_from_pvalue(r[label_col])
    return out  

def _parse_term_to_key(term: str) -> str:
    """
    Map statsmodels term strings to compact keys.
    """
    t = str(term)
    if ':C(Q("sex"))' in t and 'C(Q("group"))' in t:
        return "interaction"
    if 'C(Q("group"))' in t:
        return "group"
    if 'C(Q("sex"))' in t:
        return "sex"
    return ""


def _stars_from_label(label: str) -> str:
    """
    Convert stored significance_label into stars.
    Accepts:
      'ns', '<0.05', '<0.01', '<0.001', '<0.0001'
    Also accepts direct star strings.
    """
    s = str(label).strip().lower()
    if s in ("ns", "", "nan", "none"):
        return ""
    if "<0.0001" in s:
        return "****"
    if "<0.001" in s:
        return "***"
    if "<0.01" in s:
        return "**"
    if "<0.05" in s:
        return "*"
    if set(s) <= set("*"):
        return s
    return ""

def _stars_from_pvalue(p) -> str:
    if pd.isna(p):
        return "ns"
    p = float(p)
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def _label_from_significance(label: str) -> str:
    """
    Return stars if significant, otherwise 'ns'.
    """
    stars = _stars_from_label(label)
    return stars if stars else "ns"


def _draw_main_effect_bracket(
    *,
    ax,
    x1: float,
    x2: float,
    label: str,
    y_pad_frac: float = 0.12,
    h_frac: float = 0.035,
    title_buffer_frac: float = 0.12,
    lw: float = 1.2,
) -> None:
    """
    Draw a bracket from x1 to x2 at the top of the plot with a star label.
    """
    if not label:
        return

    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    if yr <= 0:
        yr = 1.0

    y = y1 + y_pad_frac * yr
    h = h_frac * yr

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=lw,
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        y + h + 0.01 * yr,
        label,
        ha="center",
        va="bottom",
    )

    ax.set_ylim(y0, y + h + title_buffer_frac * yr)


def _draw_anova_text_lines(
    ax,
    lines: List[str],
    position: str = "top_right",
    fontsize: float = 10,
) -> None:
    if not lines:
        return

    if position == "top_left":
        x, ha = 0.00, "left"
    else:
        x, ha = 1.00, "right"

    text = "\n".join(lines)
    ax.text(
        x,
        0.98,  # outside plotting area
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize,
        clip_on=False,
    )


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
    anova_endpoint: Optional[str] = None,
    show: bool = False,
    anova_df: Optional[pd.DataFrame] = None,
    anova_cfg: Optional[dict] = None,
    one_sample_df: Optional[pd.DataFrame] = None,
    one_sample_cfg: Optional[dict] = None,
) -> None:
    """
    Draw Prism-like box+whisker plots with individual points:
    - edgecolor of box = group color
    - points overlaid with sex-specific markers
    - x-axis categories from x_col
    - groups shown as separated blocks with a gap
    """
    if style is None:
        raise ValueError("plot_group_boxpoints requires a PlotStyle (pass from config).")

    if mouse_id_col is None or str(mouse_id_col).strip() == "":
        raise ValueError("mouse_id_col is required for stable jittered points.")

    require_columns(df, [value_col, x_col, group_col, sex_col])

    keep_cols = [value_col, x_col, group_col, sex_col]
    if mouse_id_col:
        keep_cols.append(mouse_id_col)

    d = df[keep_cols].copy()
    d = d.dropna(subset=[value_col, x_col, group_col, sex_col])

    x_levels = infer_order(d[x_col].astype(str).tolist(), preferred=x_order)
    g_levels = infer_order(d[group_col].astype(str).tolist(), preferred=group_order)
    s_levels = infer_order(d[sex_col].astype(str).tolist(), preferred=sex_order)

    if len(x_levels) < 1 or len(g_levels) < 1:
        raise ValueError("No levels found for x or group after dropping NAs.")

    n_x = len(x_levels)

    positions = {}
    xticks: List[float] = []
    xticklabels: List[str] = []

    base = 0.0

    if n_x == 1:
        # true center-to-center spacing controlled directly by group_gap
        step = style.group_gap
        x = x_levels[0]
        for gi, g in enumerate(g_levels):
            pos = base + gi * step
            positions[(g, x)] = pos
            xticks.append(pos)
            xticklabels.append(str(x))
    else:
        # within-group spacing = x_gap
        # between-group spacing = group_gap
        group_block_width = (n_x - 1) * style.x_gap
        for gi, g in enumerate(g_levels):
            block_start = base + gi * (group_block_width + style.group_gap)
            for xi, x in enumerate(x_levels):
                pos = block_start + xi * style.x_gap
                positions[(g, x)] = pos
                xticks.append(pos)
                xticklabels.append(str(x))

    center_shift = (max(xticks) + min(xticks)) / 2
    xticks = [x - center_shift for x in xticks]
    positions = {(g, x): pos - center_shift for (g, x), pos in positions.items()}

    fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)

    for g in g_levels:
        edge = style.group_edgecolors.get(str(g).lower(), "black")

        for x in x_levels:
            sub = d[
                (d[group_col].astype(str) == str(g))
                & (d[x_col].astype(str) == str(x))
            ]
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

            for box in bp["boxes"]:
                box.set(facecolor="white", edgecolor=edge, linewidth=style.box_linewidth)
            for med in bp["medians"]:
                med.set(color=edge, linewidth=style.median_linewidth)
            for whisk in bp["whiskers"]:
                whisk.set(color=edge, linewidth=style.whisker_linewidth)
            for cap in bp["caps"]:
                cap.set(color=edge, linewidth=style.cap_linewidth)

            for s in s_levels:
                sub_s = sub[sub[sex_col].astype(str) == str(s)]
                if sub_s.empty:
                    continue

                ys = sub_s[value_col].astype(float).to_numpy()
                mids_s = sub_s[mouse_id_col].astype(str).tolist()
                jit_s = jitter_from_ids(mids_s, sd=style.jitter_sd)

                sex_dodge = style.sex_dodge
                offset = (-sex_dodge / 2.0) if str(s).lower() == "female" else (sex_dodge / 2.0)
                xjs = pos + jit_s + offset

                ax.scatter(
                    xjs,
                    ys,
                    s=style.point_size,
                    marker=style.sex_markers.get(str(s).lower(), "o"),
                    alpha=style.point_alpha,
                    c="black",
                    edgecolors="black",
                    linewidths=0.0,
                    zorder=3,
                )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)

    for g in g_levels:
        block_positions = [positions[(g, x)] for x in x_levels]
        center = float(np.mean(block_positions))
        label = "IFS" if str(g).lower() == "ifs" else str(g).capitalize()

        ax.text(
            center,
            style.group_label_y,
            label,
            ha="center",
            va="top",
            fontsize=10,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    ax.set_title(title or f"{assay} — {endpoint}", fontsize=12, pad=style.title_pad)
    ax.set_ylabel(y_label or endpoint)
    ax.set_xlabel("")

    plot_key = anova_endpoint if anova_endpoint else endpoint
    is_di = _is_di_endpoint(plot_key)

    # --- ANOVA annotation (group / sex / interaction) ---
    if (
        anova_cfg
        and bool(anova_cfg.get("enabled", False))
        and anova_df is not None
        and not anova_df.empty
    ):
        endpoint_col = str(anova_cfg.get("endpoint_col", "endpoint"))
        term_col = str(anova_cfg.get("term_col", "term"))
        label_col = str(anova_cfg.get("label_col", "significance_label"))

        display_cfg = anova_cfg.get("display", {}) or {}
        group_mode = str(display_cfg.get("group", "bracket")).lower()
        sex_mode = str(display_cfg.get("sex", "text")).lower()
        interaction_mode = str(display_cfg.get("interaction", "text")).lower()

        text_position = str(anova_cfg.get("position", "top_right"))
        text_fontsize = float(anova_cfg.get("fontsize", 10))

        key = plot_key
        sub = anova_df[anova_df[endpoint_col].astype(str) == str(key)]

        text_lines: List[str] = []

        if not sub.empty and term_col in sub.columns and label_col in sub.columns:
            labels_by_term = {"group": None, "sex": None, "interaction": None}

            for _, r in sub.iterrows():
                k = _parse_term_to_key(r[term_col])
                if k in labels_by_term and labels_by_term[k] is None:
                    labels_by_term[k] = str(r[label_col])

            # group
            g_label = labels_by_term["group"]
            g_stars = _label_from_significance(g_label) if g_label is not None else None
            if g_stars is not None:
                if group_mode == "bracket" and g_stars != "ns" and len(g_levels) >= 2 and len(x_levels) == 1:
                    x = x_levels[0]
                    g1 = g_levels[0]
                    g2 = g_levels[-1]
                    x1 = float(positions[(g1, x)])
                    x2 = float(positions[(g2, x)])
                    _draw_main_effect_bracket(
                        ax=ax,
                        x1=min(x1, x2),
                        x2=max(x1, x2),
                        label=g_stars,
                        y_pad_frac=float(anova_cfg.get("y_pad_frac", 0.05)),
                        h_frac=float(anova_cfg.get("h_frac", 0.03)),
                        title_buffer_frac=float(anova_cfg.get("title_buffer_frac", 0.12)),
                        lw=float(anova_cfg.get("linewidth", 1.2)),
                    )
                elif group_mode == "text" or (group_mode == "bracket" and len(x_levels) != 1):
                    text_lines.append(f"group: {g_stars}")

            # sex
            s_label = labels_by_term["sex"]
            s_stars = _label_from_significance(s_label) if s_label is not None else None
            if s_stars is not None and sex_mode == "text":
                text_lines.append(f"sex: {s_stars}")

            # interaction
            i_label = labels_by_term["interaction"]
            i_stars = _label_from_significance(i_label) if i_label is not None else None
            if i_stars is not None and interaction_mode == "text":
                text_lines.append(f"interaction: {i_stars}")

        _draw_anova_text_lines(
            ax,
            text_lines,
            position=text_position,
            fontsize=text_fontsize,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=1)

    if is_di and one_sample_cfg and bool(one_sample_cfg.get("enabled", False)) and one_sample_df is not None and not one_sample_df.empty:
        endpoint_col = str(one_sample_cfg.get("endpoint_col", "endpoint"))
        group_col_os = str(one_sample_cfg.get("group_col", "group"))
        label_col = str(one_sample_cfg.get("label_col", "t_p_adj_holm"))

        labels = _collect_one_sample_labels(
            one_sample_df,
            endpoint=plot_key,
            endpoint_col=endpoint_col,
            group_col=group_col_os,
            label_col=label_col,
        )

        if labels:
            ordered = []
            for g in g_levels:
                lab = labels.get(str(g))
                if lab is not None:
                    ordered.append(f"{str(g)}: {lab}")
            if ordered:
                _draw_corner_text(
                    ax,
                    "\n".join(ordered),
                    position=style.one_sample_text_position,
                    fontsize=style.one_sample_fontsize,
                )

    # --- DI-specific axis handling ---
    if is_di:
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            color="grey",
            alpha=0.8,
            zorder=1,
        )
    else:
        ymax = float(d[value_col].max())
        top = ymax * style.ylim_top_mult if ymax > 0 else 1.0
        ax.set_ylim(0, top)

    xmin = min(xticks) - style.x_margin
    xmax = max(xticks) + style.x_margin
    ax.set_xlim(xmin, xmax)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)