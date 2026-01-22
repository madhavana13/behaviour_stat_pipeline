from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class PlotStyle:
    # aesthetics driven by config
    group_edgecolors: Dict[str, str]  # e.g. {"tail": "black", "cup": "orange"}
    sex_markers: Dict[str, str]       # e.g. {"female": "o", "male": "^"}

    # box aesthetics
    box_linewidth: float = 2.0
    median_linewidth: float = 2.0
    whisker_linewidth: float = 2.0
    cap_linewidth: float = 2.0
    box_width: float = 0.45
    x_gap: float = 0.8

    # point aesthetics
    point_size: float = 28.0
    point_alpha: float = 0.9
    jitter_sd: float = 0.18
    sex_dodge: float = 0.06

    # layout
    group_gap: float = 0.6

import hashlib
import numpy as np

def jitter_from_ids(ids, sd: float) -> np.ndarray:
    out = []
    for s in ids:
        h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
        u = int(h[:8], 16) / 0xFFFFFFFF  # [0,1]
        out.append((u - 0.5) * 2 * sd)   # [-sd, sd]
    return np.array(out, dtype=float)


def require_columns(df, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_order(values, preferred: Optional[List[str]] = None) -> List[str]:
    """Deterministic ordering: preferred order filtered to observed; else sorted unique."""
    obs = sorted({str(v) for v in values if v is not None and str(v) != "nan"})
    if preferred:
        pref = [str(x) for x in preferred]
        out = [x for x in pref if x in obs]
        # append any remaining observed levels not specified
        out += [x for x in obs if x not in out]
        return out
    return obs
