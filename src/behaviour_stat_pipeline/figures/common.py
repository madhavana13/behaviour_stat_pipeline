from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class PlotStyle:
    group_edgecolors: Dict[str, str]  # e.g. {"tail": "black", "cup": "orange"}
    sex_markers: Dict[str, str]       # e.g. {"female": "o", "male": "^"}
    box_linewidth: float = 2.0
    median_linewidth: float = 2.0
    whisker_linewidth: float = 2.0
    cap_linewidth: float = 2.0
    point_size: float = 24.0
    point_alpha: float = 0.9
    jitter_sd: float = 0.06  # jitter in x units
    seed: int = 1337

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
