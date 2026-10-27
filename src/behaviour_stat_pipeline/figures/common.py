from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PlotStyle:
    # aesthetics driven by config
    group_edgecolors: Dict[str, str]
    sex_markers: Dict[str, str]
    # display label per group value (experiment config groups.labels)
    group_labels: Dict[str, str] = field(default_factory=dict)

    # box aesthetics
    box_linewidth: float = 1.1
    median_linewidth: float = 1.1
    whisker_linewidth: float = 1.0
    cap_linewidth: float = 1.0
    box_width: float = 0.45
    x_gap: float = 0.8

    # point aesthetics
    point_size: float = 28.0
    point_alpha: float = 0.9
    jitter_sd: float = 0.18
    sex_dodge: float = 0.06

    # optional debugging labels for individual points
    show_point_ids: bool = False
    point_id_fontsize: float = 6.0
    point_id_dx: float = 0.015
    point_id_dy: float = 0.0

    # layout
    group_gap: float = 0.6
    figure_size: Tuple[float, float] = (5.2, 4.2)
    dpi: int = 300
    x_margin: float = 0.45
    title_pad: float = 18.0
    group_label_y: float = -0.12
    ylim_top_mult: float = 1.38
    box_face_alpha: float = 0.18
    transparent_background: bool = False
    axis_color: str = "black"
    text_color: str = "black"
    
    # typography
    title_fontsize: float = 10.0
    axis_label_fontsize: float = 9.0
    tick_label_fontsize: float = 8.0
    group_label_fontsize: float = 9.0
    annotation_fontsize: float = 9.0
    bracket_fontsize: float = 10.0

    # axes
    spine_linewidth: float = 1.2
    tick_width: float = 1.0
    tick_length: float = 4.0

    #DI_NOR
    di_ylim: Tuple[float, float] = (-1.0, 1.0)
    di_zero_line: bool = True
    di_zero_linewidth: float = 1.0
    di_zero_linestyle: str = "--"
    di_ticks: Tuple[float, float, float, float, float] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    one_sample_text_position: str = "top_left"
    one_sample_fontsize: float = 10.0


def jitter_from_ids(ids, sd: float) -> np.ndarray:
    out = []
    for s in ids:
        h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
        u = int(h[:8], 16) / 0xFFFFFFFF
        out.append((u - 0.5) * 2 * sd)
    return np.array(out, dtype=float)


def require_columns(df, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_order(values, preferred: Optional[List[str]] = None) -> List[str]:
    obs = sorted({str(v) for v in values if v is not None and str(v) != "nan"})
    if preferred:
        pref = [str(x) for x in preferred]
        out = [x for x in pref if x in obs]
        out += [x for x in obs if x not in out]
        return out
    return obs