from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.duration.survfunc import SurvfuncRight

from .common import PlotStyle, infer_order


def _group_display(g: str) -> str:
    # Same convention as boxpoints group labels
    return "IFS" if str(g).lower() == "ifs" else str(g).capitalize()


def km_steps(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaplan-Meier step coordinates (for where='post') plus censoring marks.

    Returns t, s (starting at t=0, S=1 and extended to the last follow-up time)
    and the (time, S) positions of censored animals.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if event.sum() > 0:
        sf = SurvfuncRight(time, event)
        ev_t, ev_s = np.asarray(sf.surv_times, float), np.asarray(sf.surv_prob, float)
    else:
        ev_t, ev_s = np.array([]), np.array([])

    t = np.concatenate([[0.0], ev_t])
    s = np.concatenate([[1.0], ev_s])
    last = float(time.max()) if time.size else 0.0
    if last > t[-1]:
        t = np.append(t, last)
        s = np.append(s, s[-1])

    cens_t = np.sort(time[event == 0])
    # A censored animal sits on the curve at its censoring time (after any event at that time)
    idx = np.searchsorted(t, cens_t, side="right") - 1
    cens_s = s[idx] if cens_t.size else np.array([])
    return t, s, cens_t, cens_s


def _annotation(tests: Optional[pd.DataFrame], endpoint: str, group_col: str) -> Optional[str]:
    if tests is None or tests.empty:
        return None
    sub = tests[tests["endpoint"] == endpoint].set_index("term")
    if group_col not in sub.index:
        return None
    row = sub.loc[group_col]
    p_col = "p_adj" if "p_adj" in sub.columns else "p_value"
    method = str(row.get("correction_method", "none")) if p_col == "p_adj" else "none"
    suffix = "" if method.lower() == "none" else f" ({method.capitalize()})"

    def fmt_p(p: float) -> str:
        return ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") + suffix

    if np.isfinite(row["hazard_ratio"]):
        lines = [
            f"HR {row['hazard_ratio']:.2f} [{row['hr_ci_low']:.2f}, {row['hr_ci_high']:.2f}]",
            f"Cox {fmt_p(row[p_col])}",
        ]
    else:  # more than two groups: omnibus test; per-level HRs are in the table
        lines = [f"Cox LR (df={int(row['lr_df'])}) {fmt_p(row[p_col])}"]
    logrank = sub[sub["model"] == "log-rank"]
    if not logrank.empty:
        lines.append(f"log-rank {fmt_p(logrank.iloc[0][p_col])}")
    return "\n".join(lines)


def _style_axes(ax, style: PlotStyle) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.axis_color)
        ax.spines[side].set_linewidth(style.spine_linewidth)
    ax.tick_params(
        color=style.axis_color, labelcolor=style.text_color,
        width=style.tick_width, length=style.tick_length, labelsize=style.tick_label_fontsize,
    )
    ax.set_ylim(-0.02, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(left=0)


def plot_km(
    d: pd.DataFrame,
    *,
    out_paths: Sequence[Path],
    style: PlotStyle,
    group_col: str = "group",
    sex_col: str = "sex",
    group_order: Optional[Sequence[str]] = None,
    sex_order: Optional[Sequence[str]] = None,
    by_sex: bool = False,
    x_label: str = "Time (s)",
    y_label: str = "Proportion without event",
    title: Optional[str] = None,
    tests: Optional[pd.DataFrame] = None,
    endpoint: Optional[str] = None,
    linewidth: float = 1.2,
    censor_marker_size: float = 5.0,
    figure_size: Optional[tuple[float, float]] = None,
) -> None:
    """
    Kaplan-Meier curves by group; censored animals marked with '|'.

    `d` must already contain `_time` and `_event` (from stats.endpoints.time_to_event.
    build_survival_data), so the figure uses exactly the censoring rule of the analysis.
    With by_sex=True, one panel per sex (shared y axis); otherwise sexes are pooled
    and the Cox/log-rank result for group is printed in the panel.
    """
    groups = infer_order(d[group_col], group_order)
    panels = infer_order(d[sex_col], sex_order or ["female", "male"]) if by_sex else [None]

    size = figure_size or style.figure_size
    fig, axes = plt.subplots(1, len(panels), figsize=size, dpi=style.dpi, sharey=True, squeeze=False)
    axes = axes[0]
    if style.transparent_background:
        fig.patch.set_alpha(0.0)

    for ax, panel in zip(axes, panels):
        if style.transparent_background:
            ax.set_facecolor("none")
        dp = d if panel is None else d[d[sex_col].astype(str) == str(panel)]
        for g in groups:
            dg = dp[dp[group_col].astype(str) == str(g)]
            if dg.empty:
                continue
            color = style.group_edgecolors.get(str(g), "black")
            t, s, ct, cs = km_steps(dg["_time"].to_numpy(), dg["_event"].to_numpy())
            n_cens = int((dg["_event"] == 0).sum())
            label = f"{_group_display(g)} (n={len(dg)}" + (f", {n_cens} censored)" if n_cens else ")")
            ax.step(t, s, where="post", color=color, linewidth=linewidth, label=label)
            if ct.size:
                ax.plot(ct, cs, linestyle="none", marker="|", markersize=censor_marker_size,
                        markeredgewidth=linewidth, color=color)

        _style_axes(ax, style)
        ax.set_xlabel(x_label, fontsize=style.axis_label_fontsize, color=style.text_color)
        if panel is not None:
            ax.set_title(str(panel).capitalize(), fontsize=style.title_fontsize, color=style.text_color)
        ax.legend(frameon=False, fontsize=style.annotation_fontsize, loc="upper right",
                  handlelength=1.4, borderaxespad=0.2)

    axes[0].set_ylabel(y_label, fontsize=style.axis_label_fontsize, color=style.text_color)

    if not by_sex and endpoint is not None:
        text = _annotation(tests, endpoint, group_col)
        if text:
            axes[0].text(0.97, 0.62, text, transform=axes[0].transAxes, ha="right", va="top",
                         fontsize=style.annotation_fontsize, color=style.text_color)

    if title:
        fig.suptitle(title, fontsize=style.title_fontsize, color=style.text_color)

    fig.tight_layout()
    for p in out_paths:
        fig.savefig(p, dpi=style.dpi, transparent=style.transparent_background, bbox_inches="tight")
    plt.close(fig)
