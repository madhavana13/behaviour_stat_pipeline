"""
NOR (Novel Object Recognition) assay helpers.

Goal (automated and audit-friendly):
- Determine familiar object TYPE from FAM session:
    - if flask columns are non-zero => familiar_type = "flask"
    - if lego_tower columns are non-zero => familiar_type = "lego_tower"
    - if both are non-zero => raise (tracking / labeling issue)
- Use familiar_type to label TEST measures as familiar vs novel.

Outputs (Option A):
- One row per mouse_id (timepoint treated as "Single" at runner level)
- Derived metrics for BOTH:
    - exploration (8 cm; "time exploring ...")
    - interaction (2 cm; "time in interaction zone of ...")

Assumptions:
- FAM file contains flask_0, flask_1, lego_tower_0, lego_tower_1 columns. 
- TEST file contains flask_0 and lego_tower_0 columns. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key


FamiliarType = Literal["flask", "lego_tower"]


# Canonical column names in your exports
FAM_EXPLORE_FLASK = ["time exploring flask_0", "time exploring flask_1"]
FAM_EXPLORE_TOWER = ["time exploring lego_tower_0", "time exploring lego_tower_1"]
FAM_IZ_FLASK = ["time in interaction zone of flask_0", "time in interaction zone of flask_1"]
FAM_IZ_TOWER = ["time in interaction zone of lego_tower_0", "time in interaction zone of lego_tower_1"]

TEST_EXPLORE_FLASK = "time exploring flask_0"
TEST_EXPLORE_TOWER = "time exploring lego_tower_0"
TEST_IZ_FLASK = "time in interaction zone of flask_0"
TEST_IZ_TOWER = "time in interaction zone of lego_tower_0"


REQUIRED_ID_COLS = {"mouse_id", "group", "sex", "timepoint"}
REQUIRED_SESSION_COL = "session"  # runner sets this to "FAM"/"TEST"


@dataclass(frozen=True)
class NORConfig:
    """
    Configuration for NOR processing.

    presence_rule:
        "nonzero" means any >0 in the relevant object columns implies presence.
        We raise if BOTH object types look present in FAM.
    """
    presence_rule: str = "nonzero"  # currently only supported value


def qc_nor(fam: pd.DataFrame, test: pd.DataFrame) -> None:
    missing_fam = REQUIRED_ID_COLS - set(fam.columns)
    missing_test = REQUIRED_ID_COLS - set(test.columns)
    if missing_fam:
        raise ValueError(f"NOR QC failed (FAM): missing required columns: {sorted(missing_fam)}")
    if missing_test:
        raise ValueError(f"NOR QC failed (TEST): missing required columns: {sorted(missing_test)}")

    if REQUIRED_SESSION_COL not in fam.columns or REQUIRED_SESSION_COL not in test.columns:
        raise ValueError("NOR QC failed: runner must add a 'session' column with values 'FAM'/'TEST'.")

    # Uniqueness: these exports are session-level (one row per mouse per file).
    assert_unique_key(fam, ["mouse_id"], where="NOR FAM (session-level)")
    assert_unique_key(test, ["mouse_id"], where="NOR TEST (session-level)")

    # Required object columns
    fam_required = set(FAM_EXPLORE_FLASK + FAM_EXPLORE_TOWER + FAM_IZ_FLASK + FAM_IZ_TOWER)
    test_required = {TEST_EXPLORE_FLASK, TEST_EXPLORE_TOWER, TEST_IZ_FLASK, TEST_IZ_TOWER}

    miss_fam_obj = sorted(fam_required - set(fam.columns))
    miss_test_obj = sorted(test_required - set(test.columns))
    if miss_fam_obj:
        raise ValueError(f"NOR QC failed (FAM): missing object columns: {miss_fam_obj}")
    if miss_test_obj:
        raise ValueError(f"NOR QC failed (TEST): missing object columns: {miss_test_obj}")
    
    for col in ["raw_id", "source_file"]:
        if col not in fam.columns:
            raise ValueError(f"NOR QC failed (FAM): missing provenance column '{col}'")
        if col not in test.columns:
            raise ValueError(f"NOR QC failed (TEST): missing provenance column '{col}'")

def _infer_familiar_type_from_fam_row(row: pd.Series) -> FamiliarType:
    flask_any = (row[FAM_EXPLORE_FLASK + FAM_IZ_FLASK] > 0).any()
    tower_any = (row[FAM_EXPLORE_TOWER + FAM_IZ_TOWER] > 0).any()

    if flask_any and tower_any:
        raise ValueError(
            f"NOR: Both flask and lego_tower appear non-zero in FAM for mouse_id={row['mouse_id']}. "
            "This suggests a tracking/labeling issue."
        )
    if not flask_any and not tower_any:
        raise ValueError(
            f"NOR: Neither flask nor lego_tower appears non-zero in FAM for mouse_id={row['mouse_id']}. "
            "Cannot infer familiar object type."
        )
    return "flask" if flask_any else "lego_tower"


def _safe_di(novel: pd.Series, familiar: pd.Series) -> pd.Series:
    denom = novel + familiar
    out = (novel - familiar) / denom.replace(0, np.nan)
    return out


def _safe_pi(novel: pd.Series, familiar: pd.Series) -> pd.Series:
    denom = novel + familiar
    out = novel / denom.replace(0, np.nan)
    return out
def add_nor_sanity_flags(df: pd.DataFrame, *, min_total_s: float = 20.0) -> pd.DataFrame:
    """
    Add non-destructive sanity flags for NOR.

    Flags are advisory only (no rows dropped):
    - *_no_explore: total == 0
    - *_low_explore: total < min_total_s (default 20s)
    """
    out = df.copy()

    out["test_total_explore_time"] = out["test_novel_explore_time"] + out["test_familiar_explore_time"]
    out["test_total_iz_time"] = out["test_novel_iz_time"] + out["test_familiar_iz_time"]

    out["explore_no_explore"] = out["test_total_explore_time"] == 0
    out["iz_no_explore"] = out["test_total_iz_time"] == 0

    out["explore_low_explore"] = out["test_total_explore_time"] < float(min_total_s)
    out["iz_low_explore"] = out["test_total_iz_time"] < float(min_total_s)

        # FAM: total time with the familiar object (already aligned in processed table)
    out["fam_total_explore_time"] = out["fam_familiar_explore_time"]
    out["fam_total_iz_time"] = out["fam_familiar_iz_time"]

    out["fam_explore_no_explore"] = out["fam_total_explore_time"] == 0
    out["fam_iz_no_explore"] = out["fam_total_iz_time"] == 0

    out["fam_explore_low_explore"] = out["fam_total_explore_time"] < float(min_total_s)
    out["fam_iz_low_explore"] = out["fam_total_iz_time"] < float(min_total_s)

    return out


def summarize_nor_sanity(df: pd.DataFrame, *, min_total_s: float = 20.0, max_ids: int = 12) -> str:
    def _ids(mask: pd.Series) -> list[str]:
        return df.loc[mask, "mouse_id"].astype(str).tolist()

    def _fmt_ids(ids: list[str]) -> str:
        if not ids:
            return "[]"
        head = ids[:max_ids]
        suffix = "" if len(ids) <= max_ids else f" (+{len(ids) - max_ids} more)"
        return f"{head}{suffix}"

    n = len(df)
    if n == 0:
        return "NOR sanity: no rows."

    # TEST
    test_total_explore = df["test_total_explore_time"]
    test_total_iz = df["test_total_iz_time"]

    test_no_explore_ids = _ids(test_total_explore == 0)
    test_low_explore_ids = _ids(test_total_explore < float(min_total_s))

    test_no_iz_ids = _ids(test_total_iz == 0)
    test_low_iz_ids = _ids(test_total_iz < float(min_total_s))

    # FAM
    fam_total_explore = df["fam_total_explore_time"]
    fam_total_iz = df["fam_total_iz_time"]

    fam_no_explore_ids = _ids(fam_total_explore == 0)
    fam_low_explore_ids = _ids(fam_total_explore < float(min_total_s))

    fam_no_iz_ids = _ids(fam_total_iz == 0)
    fam_low_iz_ids = _ids(fam_total_iz < float(min_total_s))

    return (
        "NOR sanity:\n"
        f"- TEST exploration total == 0s: {len(test_no_explore_ids)}/{n} | ids={_fmt_ids(test_no_explore_ids)}\n"
        f"- TEST exploration total < {min_total_s:g}s: {len(test_low_explore_ids)}/{n} | ids={_fmt_ids(test_low_explore_ids)}\n"
        f"- TEST interaction-zone total == 0s: {len(test_no_iz_ids)}/{n} | ids={_fmt_ids(test_no_iz_ids)}\n"
        f"- TEST interaction-zone total < {min_total_s:g}s: {len(test_low_iz_ids)}/{n} | ids={_fmt_ids(test_low_iz_ids)}\n"
        "\n"
        f"- FAM familiar exploration == 0s: {len(fam_no_explore_ids)}/{n} | ids={_fmt_ids(fam_no_explore_ids)}\n"
        f"- FAM familiar exploration < {min_total_s:g}s: {len(fam_low_explore_ids)}/{n} | ids={_fmt_ids(fam_low_explore_ids)}\n"
        f"- FAM familiar IZ == 0s: {len(fam_no_iz_ids)}/{n} | ids={_fmt_ids(fam_no_iz_ids)}\n"
        f"- FAM familiar IZ < {min_total_s:g}s: {len(fam_low_iz_ids)}/{n} | ids={_fmt_ids(fam_low_iz_ids)}"
    )


def derive_nor_metrics(
    fam: pd.DataFrame,
    test: pd.DataFrame,
    cfg: NORConfig | None = None,
) -> pd.DataFrame:
    """
    Return one row per mouse_id × timepoint (runner sets timepoint="Single").

    Adds (both exploration + interaction-zone versions):
      - fam_familiar_explore_time
      - fam_familiar_iz_time
      - test_familiar_explore_time, test_novel_explore_time
      - test_familiar_iz_time,      test_novel_iz_time
      - di_explore, pi_explore
      - di_iz,      pi_iz
      - familiar_type  ("flask" or "lego_tower")
    """
    cfg = cfg or NORConfig()
    if cfg.presence_rule != "nonzero":
        raise ValueError(f"Unsupported NORConfig.presence_rule: {cfg.presence_rule}")

    qc_nor(fam, test)

    fam = fam.copy()
    test = test.copy()

    # Preserve provenance for auditability
    fam = fam.rename(columns={"raw_id": "raw_id_fam", "source_file": "source_file_fam"})
    test = test.rename(columns={"raw_id": "raw_id_test", "source_file": "source_file_test"})

    flask_any = (fam[FAM_EXPLORE_FLASK + FAM_IZ_FLASK] > 0).any(axis=1)
    tower_any = (fam[FAM_EXPLORE_TOWER + FAM_IZ_TOWER] > 0).any(axis=1)

    if (flask_any & tower_any).any():
        bad_ids = fam.loc[(flask_any & tower_any), "mouse_id"].astype(str).tolist()
        raise ValueError(
            "NOR: Both flask and lego_tower appear non-zero in FAM for some mice (tracking/labeling issue). "
            f"mouse_id examples: {bad_ids[:20]}"
        )

    if (~flask_any & ~tower_any).any():
        bad_ids = fam.loc[(~flask_any & ~tower_any), "mouse_id"].astype(str).tolist()
        raise ValueError(
            "NOR: Neither flask nor lego_tower appears non-zero in FAM for some mice; cannot infer familiar type. "
            f"mouse_id examples: {bad_ids[:20]}"
        )

    fam["familiar_type"] = np.where(flask_any, "flask", "lego_tower")

    # Collapse FAM identical objects into a single familiar-time measure
    fam["fam_familiar_explore_time"] = np.where(
        fam["familiar_type"] == "flask",
        fam[FAM_EXPLORE_FLASK].sum(axis=1),
        fam[FAM_EXPLORE_TOWER].sum(axis=1),
    )
    fam["fam_familiar_iz_time"] = np.where(
        fam["familiar_type"] == "flask",
        fam[FAM_IZ_FLASK].sum(axis=1),
        fam[FAM_IZ_TOWER].sum(axis=1),
    )

    # Merge fam + test (inner join keeps only mice with both sessions)
    id_cols = ["mouse_id", "group", "sex", "timepoint"]
    keep_fam = id_cols + ["raw_id_fam", "source_file_fam","familiar_type", "fam_familiar_explore_time", "fam_familiar_iz_time"]
    keep_test = id_cols + ["raw_id_test", "source_file_test", TEST_EXPLORE_FLASK, TEST_EXPLORE_TOWER, TEST_IZ_FLASK, TEST_IZ_TOWER]

    merged = fam[keep_fam].merge(
        test[keep_test],
        on=id_cols,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_test"),
    )

    # Label TEST familiar vs novel based on familiar_type
    merged["test_familiar_explore_time"] = np.where(
        merged["familiar_type"] == "flask",
        merged[TEST_EXPLORE_FLASK],
        merged[TEST_EXPLORE_TOWER],
    )
    merged["test_novel_explore_time"] = np.where(
        merged["familiar_type"] == "flask",
        merged[TEST_EXPLORE_TOWER],
        merged[TEST_EXPLORE_FLASK],
    )

    merged["test_familiar_iz_time"] = np.where(
        merged["familiar_type"] == "flask",
        merged[TEST_IZ_FLASK],
        merged[TEST_IZ_TOWER],
    )
    merged["test_novel_iz_time"] = np.where(
        merged["familiar_type"] == "flask",
        merged[TEST_IZ_TOWER],
        merged[TEST_IZ_FLASK],
    )

    # Indices (both definitions)
    merged["di_explore"] = _safe_di(
        merged["test_novel_explore_time"],
        merged["test_familiar_explore_time"],
    )
    merged["pi_explore"] = _safe_pi(
        merged["test_novel_explore_time"],
        merged["test_familiar_explore_time"],
    )

    merged["di_iz"] = _safe_di(
        merged["test_novel_iz_time"],
        merged["test_familiar_iz_time"],
    )
    merged["pi_iz"] = _safe_pi(
        merged["test_novel_iz_time"],
        merged["test_familiar_iz_time"],
    )

    # Project invariant: processed must be unique at mouse_id × timepoint
    assert_unique_key(merged, ["mouse_id", "timepoint"], where="NOR processed")
    merged = add_nor_sanity_flags(merged, min_total_s=20.0)

    return merged


PRIMARY_OUTCOMES: list[str] = [
    "di_explore",
    "pi_explore",
    "di_iz",
    "pi_iz",
]

