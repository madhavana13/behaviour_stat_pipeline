"""
SIT (social interaction test) assay helpers.

Protocol:
- FAM: 2 min in the open field with an EMPTY mesh cage against one wall.
- TEST: the same arena with a target mouse in the mesh cage, 5 min.

Discrimination (like NOR, familiar -> empty cage, novel -> target):
- FAM (empty cage, whole 2-min session) vs the FIRST 2 MIN of TEST (target present). The 2-min
  TEST window comes from the export ("... (first 2 min)" columns), not from the pipeline.
- Each side is a fraction of tracked time, because tracked time differs between the sessions
  (FAM ~110 s of a nominal 120 s): FAM time / FAM total_time_in_arena, and TEST first-2-min time /
  min(test_window_s, TEST total_time_in_arena). The DI is computed from these fractions.
- Computed for the interaction zone (di_iz) and for cage exploration (di_explore).

Outputs:
- One row per mouse_id, all raw measures kept with fam_/test_ prefixes.
- fam_<measure>_fraction, test_<measure>_first2min_fraction (the two sides of each DI).
- target_id (when target_col is configured): the subject-metadata column naming the target mouse,
  for use as a random effect (each target meets several test mice).
- di_iz, pi_iz, di_explore, pi_explore (positive DI = more time near the target than near the
  empty cage).
- Advisory flags (no rows dropped): low time near the cage, short recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from behaviour_stat_pipeline.assays.nor import _prefix_non_id_columns, _safe_di, _safe_pi
from behaviour_stat_pipeline.common.qc import assert_unique_key
from behaviour_stat_pipeline.common.schema import normalize_headers

ID_COLS = ["mouse_id", "group", "sex", "timepoint"]

# Canonical column names (schema_aliases.yaml -> sit)
IZ_TIME = "time_in_interaction_zone"
EXPLORE_TIME = "time_exploring_cage"
TRACKED_TIME = "total_time_in_arena"
TEST_WINDOW_SUFFIX = "_first2min"   # TEST measures restricted to the first 2 min

# side of the index -> (FAM column, TEST column), before the fam_/test_ prefix
INDEX_MEASURES = {
    "iz": (IZ_TIME, IZ_TIME + TEST_WINDOW_SUFFIX),
    "explore": (EXPLORE_TIME, EXPLORE_TIME + TEST_WINDOW_SUFFIX),
}


@dataclass(frozen=True)
class SITConfig:
    """
    Configuration for SIT processing (from assay_instances.SIT in the experiment config).

    min_explore_s:
        Advisory threshold: FAM + TEST (first 2 min) interaction-zone time below this is flagged.
    min_tracked_fraction:
        Advisory threshold: a session whose tracked time is below this fraction of the median
        tracked time for that session is flagged (e.g. a recording that stopped early).
    test_window_s:
        Length of the TEST window behind the "(first 2 min)" columns.
    target_col:
        Subject-metadata column naming the target mouse; copied to target_id. None -> not used.
    """
    min_explore_s: float = 20.0
    min_tracked_fraction: float = 0.9
    test_window_s: float = 120.0
    target_col: str | None = None


def qc_sit(fam: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, df in (("FAM", fam), ("TEST", test)):
        missing_ids = set(ID_COLS) - set(df.columns)
        if missing_ids:
            raise ValueError(f"SIT QC failed ({name}): missing required columns: {sorted(missing_ids)}")
        assert_unique_key(df, ["mouse_id"], where=f"SIT {name} (session-level)")
        for col in ["raw_id", "source_file"]:
            if col not in df.columns:
                raise ValueError(f"SIT QC failed ({name}): missing provenance column '{col}'")

    needed = {
        "FAM": [TRACKED_TIME] + [f for f, _ in INDEX_MEASURES.values()],
        "TEST": [TRACKED_TIME] + [t for _, t in INDEX_MEASURES.values()],
    }
    for name, df in (("FAM", fam), ("TEST", test)):
        missing = [c for c in needed[name] if c not in df.columns]
        if missing:
            raise ValueError(
                f"SIT QC failed ({name}): no columns {missing}. Check the export headers against "
                f"configs/schema_aliases.yaml -> sit."
            )


def derive_sit_metrics(
    fam: pd.DataFrame,
    test: pd.DataFrame,
    *,
    schema_aliases: dict,
    cfg: SITConfig,
) -> pd.DataFrame:
    """Return one row per mouse_id × timepoint (runner sets timepoint="Single")."""
    fam, fam_rename_log, fam_suspicious = normalize_headers(fam, assay="sit", schema=schema_aliases)
    test, test_rename_log, test_suspicious = normalize_headers(test, assay="sit", schema=schema_aliases)

    qc_sit(fam, test)

    fam = fam.rename(columns={"raw_id": "raw_id_fam", "source_file": "source_file_fam"})
    test = test.rename(columns={"raw_id": "raw_id_test", "source_file": "source_file_test"})
    fam = _prefix_non_id_columns(fam, "fam_", id_cols=ID_COLS)
    test = _prefix_non_id_columns(test, "test_", id_cols=ID_COLS)

    merged = fam.merge(test, on=ID_COLS, how="inner", validate="one_to_one")

    # Fractions of tracked time; a TEST recording shorter than the window only covers what was tracked
    fam_tracked = merged[f"fam_{TRACKED_TIME}"].replace(0, float("nan"))
    test_tracked = merged[f"test_{TRACKED_TIME}"].clip(upper=cfg.test_window_s).replace(0, float("nan"))
    for name, (fam_col, test_col) in INDEX_MEASURES.items():
        empty = merged[f"fam_{fam_col}_fraction"] = merged[f"fam_{fam_col}"] / fam_tracked
        target = merged[f"test_{test_col}_fraction"] = merged[f"test_{test_col}"] / test_tracked
        merged[f"di_{name}"] = _safe_di(target, empty)
        merged[f"pi_{name}"] = _safe_pi(target, empty)

    if cfg.target_col:
        fam_t, test_t = f"fam_{cfg.target_col}", f"test_{cfg.target_col}"
        if fam_t not in merged.columns:
            raise ValueError(f"SIT: target_col '{cfg.target_col}' is not a subject-metadata column")
        differs = merged[fam_t].astype(str) != merged[test_t].astype(str)
        if differs.any():
            raise ValueError(f"SIT: target differs between FAM and TEST rows for {merged.loc[differs, 'mouse_id'].tolist()}")
        if merged[fam_t].isna().any():
            raise ValueError(f"SIT: no target recorded for {merged.loc[merged[fam_t].isna(), 'mouse_id'].tolist()}")
        merged["target_id"] = merged[fam_t].astype(str)

    assert_unique_key(merged, ["mouse_id", "timepoint"], where="SIT processed")
    return add_sit_sanity_flags(merged, cfg=cfg)


def add_sit_sanity_flags(df: pd.DataFrame, *, cfg: SITConfig) -> pd.DataFrame:
    """Advisory flags only (no rows dropped)."""
    out = df.copy()
    fam_iz, test_iz = INDEX_MEASURES["iz"]
    out["iz_total_time"] = out[f"fam_{fam_iz}"] + out[f"test_{test_iz}"]
    out["iz_low_explore"] = out["iz_total_time"] < float(cfg.min_explore_s)
    for session in ("fam", "test"):
        tracked = out[f"{session}_{TRACKED_TIME}"]
        out[f"{session}_short_recording"] = tracked < cfg.min_tracked_fraction * tracked.median()
    return out


def summarize_sit_sanity(df: pd.DataFrame, *, cfg: SITConfig, max_ids: int = 12) -> str:
    def _ids(mask: pd.Series) -> list[str]:
        ids = df.loc[mask, "mouse_id"].astype(str).tolist()
        return ids[:max_ids] + ([f"+{len(ids) - max_ids} more"] if len(ids) > max_ids else [])

    n = len(df)
    if n == 0:
        return "SIT sanity: no rows."
    lines = ["SIT sanity:"]
    for session in ("fam", "test"):
        tracked = df[f"{session}_{TRACKED_TIME}"]
        short = df[f"{session}_short_recording"]
        lines.append(
            f"- {session.upper()} tracked time: min {tracked.min():.1f}s, median {tracked.median():.1f}s; "
            f"< {cfg.min_tracked_fraction:g} x median: {int(short.sum())}/{n} | ids={_ids(short)}"
        )
    low = df["iz_low_explore"]
    lines.append(
        f"- FAM + TEST (first 2 min) interaction-zone time < {cfg.min_explore_s:g}s: "
        f"{int(low.sum())}/{n} | ids={_ids(low)}"
    )
    nan_di = df["di_iz"].isna() | df["di_explore"].isna()
    if nan_di.any():
        lines.append(f"- DI undefined (zero time on both sides): {int(nan_di.sum())}/{n} | ids={_ids(nan_di)}")
    return "\n".join(lines)


PRIMARY_OUTCOMES: list[str] = [
    "di_iz",
    "pi_iz",
    "di_explore",
    "pi_explore",
]
