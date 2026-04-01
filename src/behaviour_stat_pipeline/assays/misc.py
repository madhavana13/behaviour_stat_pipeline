from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key


def read_misc_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MISC raw file not found: {path}")
    return pd.read_csv(path)


def qc_misc(df: pd.DataFrame) -> None:
    if "mouse_id" not in df.columns:
        raise ValueError("MISC raw must contain a 'mouse_id' column.")
    assert_unique_key(df, ["mouse_id"], where="MISC raw")


def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {where}: {missing}")


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def derive_misc_metrics(
    df: pd.DataFrame,
    *,
    derived_metrics: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Generic config-driven derivations for arbitrary mouse-level extra data.

    Supported kinds:
      - sum
      - mean
      - difference
      - ratio
      - normalize_by
      - pi
      - di

    All original columns are preserved.
    """
    out = df.copy()

    if derived_metrics:
        for spec in derived_metrics:
            if "name" not in spec or "kind" not in spec:
                raise ValueError(f"Each MISC derived metric needs 'name' and 'kind': {spec}")

            name = str(spec["name"])
            kind = str(spec["kind"]).strip().lower()

            if kind == "sum":
                cols = list(spec["columns"])
                _require_columns(out, cols, where=f"MISC metric '{name}'")
                out[name] = out[cols].sum(axis=1)

            elif kind == "mean":
                cols = list(spec["columns"])
                _require_columns(out, cols, where=f"MISC metric '{name}'")
                out[name] = out[cols].mean(axis=1)

            elif kind == "difference":
                left = str(spec["left"])
                right = str(spec["right"])
                _require_columns(out, [left, right], where=f"MISC metric '{name}'")
                out[name] = out[left] - out[right]

            elif kind == "ratio":
                numerator = str(spec["numerator"])
                denominator = str(spec["denominator"])
                _require_columns(out, [numerator, denominator], where=f"MISC metric '{name}'")
                out[name] = _safe_divide(out[numerator], out[denominator])

            elif kind == "normalize_by":
                value = str(spec["value"])
                by = str(spec["by"])
                _require_columns(out, [value, by], where=f"MISC metric '{name}'")
                out[name] = _safe_divide(out[value], out[by])

            elif kind == "pi":
                preferred = str(spec["preferred"])
                other = str(spec["other"])
                _require_columns(out, [preferred, other], where=f"MISC metric '{name}'")
                denom = (out[preferred] + out[other]).replace(0, np.nan)
                out[name] = 100.0 * out[preferred] / denom

            elif kind == "di":
                preferred = str(spec["preferred"])
                other = str(spec["other"])
                _require_columns(out, [preferred, other], where=f"MISC metric '{name}'")
                denom = (out[preferred] + out[other]).replace(0, np.nan)
                out[name] = (out[preferred] - out[other]) / denom

            else:
                raise ValueError(
                    f"Unknown MISC derived metric kind '{kind}' for metric '{name}'."
                )

    out["assay"] = "MISC"
    out["timepoint"] = "Single"
    return out
def apply_misc_masks(
    df: pd.DataFrame,
    *,
    masks: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Apply config-driven masking rules while preserving provenance columns.

    Example rule:
      - when: flag_rct
        equals: 1
        targets: [sucrose_consumption, normal_consumption]
    """
    out = df.copy()

    if not masks:
        return out

    for spec in masks:
        when = str(spec["when"])
        targets = list(spec["targets"])

        if when not in out.columns:
            raise ValueError(f"MISC mask refers to missing column: '{when}'")

        _require_columns(out, targets, where=f"MISC mask on '{when}'")

        if "equals" in spec:
            value = spec["equals"]
            mask = out[when] == value
        elif "not_equals" in spec:
            value = spec["not_equals"]
            mask = out[when] != value
        else:
            # default: truthy flag
            col = out[when]
            if pd.api.types.is_bool_dtype(col):
                mask = col.fillna(False)
            else:
                mask = col.fillna(0).astype(str).str.strip().isin(
                    ["1", "true", "True", "TRUE", "yes", "Yes", "YES"]
                )

        out.loc[mask, targets] = np.nan

    return out

def summarize_misc(df: pd.DataFrame) -> str:
    derived_cols = [c for c in df.columns if c not in {"mouse_id", "group", "sex", "assay", "timepoint"}]
    return f"MISC summary: n_rows={len(df)}, n_mice={df['mouse_id'].nunique()}, n_columns={len(derived_cols)}"