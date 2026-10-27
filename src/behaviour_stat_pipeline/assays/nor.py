"""
NOR (Novel Object Recognition) assay helpers.

Goal (automated and audit-friendly):
- The experiment config names the two object types (assay_instances.NOR.objects,
  e.g. [flask, lego_tower]). Nothing about the objects is hardcoded.
- Determine the familiar object TYPE from the FAM session:
    - the object type whose columns are non-zero in FAM is familiar
    - if both or neither are non-zero => raise (tracking / labeling issue)
- Use familiar_type to label TEST measures as familiar vs novel.

Outputs (Option A):
- One row per mouse_id (timepoint treated as "Single" at runner level)
- Derived metrics for BOTH:
    - exploration (8 cm; "time exploring ...")
    - interaction (2 cm; "time in interaction zone of ...")

Assumptions:
- Columns are named time_exploring_<object>_<copy> and
  time_in_interaction_zone_of_<object>_<copy> (copy = 0, 1, ...), for both objects.
- FAM: all copies of the familiar object are summed.
- TEST: one copy of each object carries data (empty copies are ignored; data in
  more than one copy of an object is an error).

Note:
- Processed output preserves ALL raw measures by prefixing:
    - FAM columns become fam_<original>
    - TEST columns become test_<original>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from behaviour_stat_pipeline.common.qc import assert_unique_key
from behaviour_stat_pipeline.common.schema import normalize_headers

# Column name templates in the tracking exports (after schema normalisation)
EXPLORE_TEMPLATE = "time_exploring_{obj}_"
IZ_TEMPLATE = "time_in_interaction_zone_of_{obj}_"


REQUIRED_ID_COLS = {"mouse_id", "group", "sex", "timepoint"}
REQUIRED_SESSION_COL = "session"  # runner sets this to "FAM"/"TEST"

@dataclass(frozen=True)
class NORConfig:
    """
    Configuration for NOR processing (from assay_instances.NOR in the experiment config).

    objects:
        The two object types used, as they appear in the column names.
    min_explore_s:
        Advisory threshold for low-exploration flags (no rows are dropped).
    presence_rule:
        "nonzero" means any >0 in the relevant object columns implies presence.
        We raise if BOTH object types look present in FAM.
    """
    objects: tuple[str, str]
    min_explore_s: float = 20.0
    presence_rule: str = "nonzero"  # currently only supported value

    def __post_init__(self):
        if len(self.objects) != 2 or len(set(self.objects)) != 2:
            raise ValueError(f"NOR needs exactly two different object types, got {list(self.objects)}")


def object_columns(columns, template: str, obj: str) -> list[str]:
    """All copies of an object's measure, e.g. time_exploring_flask_0, _1, ... (sorted by copy)."""
    pattern = re.compile("^" + re.escape(template.format(obj=obj)) + r"(\d+)$")
    found = [(int(m.group(1)), c) for c in columns if (m := pattern.match(c))]
    return [c for _, c in sorted(found)]


def _required_object_columns(df: pd.DataFrame, objects, where: str, prefix: str = "") -> dict:
    """(template, object) -> columns (with `prefix`); raises if any object has no columns."""
    unprefixed = [c[len(prefix):] for c in df.columns if c.startswith(prefix)]
    out, missing = {}, []
    for template in (EXPLORE_TEMPLATE, IZ_TEMPLATE):
        for obj in objects:
            cols = object_columns(unprefixed, template, obj)
            if not cols:
                missing.append(template.format(obj=obj) + "<n>")
            out[(template, obj)] = [prefix + c for c in cols]
    if missing:
        raise ValueError(
            f"NOR QC failed ({where}): no columns for {missing}. Check assay_instances.NOR.objects "
            f"in the experiment config against the export (configured objects: {list(objects)})."
        )
    return out


def _single_test_copy(test: pd.DataFrame, cols: list[str]) -> str:
    """The one copy column of an object that carries data in TEST; error if several do."""
    with_data = [c for c in cols if test[c].notna().any()]
    if len(with_data) > 1:
        raise ValueError(
            f"NOR (TEST): several copies of the same object carry data: {with_data}. "
            "The test session should have one copy of the familiar and one of the novel object."
        )
    return with_data[0] if with_data else cols[0]


def qc_nor(fam: pd.DataFrame, test: pd.DataFrame, cfg: NORConfig) -> None:
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

    # Required object columns (raises naming what is missing)
    _required_object_columns(fam, cfg.objects, "FAM")
    _required_object_columns(test, cfg.objects, "TEST")

    for col in ["raw_id", "source_file"]:
        if col not in fam.columns:
            raise ValueError(f"NOR QC failed (FAM): missing provenance column '{col}'")
        if col not in test.columns:
            raise ValueError(f"NOR QC failed (TEST): missing provenance column '{col}'")

def derive_nor_metrics(
    fam: pd.DataFrame,
    test: pd.DataFrame,
    *,
    schema_aliases: dict,
    cfg: NORConfig,
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
      - familiar_type  (one of cfg.objects)
      Also preserves ALL raw measures by prefixing them (fam_/test_).
    """
    if cfg.presence_rule != "nonzero":
        raise ValueError(f"Unsupported NORConfig.presence_rule: {cfg.presence_rule}")

    fam, fam_rename_log, fam_suspicious = normalize_headers(
        fam,
        assay="nor",
        schema=schema_aliases,
    )
    test, test_rename_log, test_suspicious = normalize_headers(
        test,
        assay="nor",
        schema=schema_aliases,
    )

    qc_nor(fam, test, cfg)

    fam = fam.copy()
    test = test.copy()

    # Preserve normalization audit trail if you want it downstream
    fam.attrs["schema_rename_log"] = fam_rename_log
    fam.attrs["schema_suspicious_columns"] = fam_suspicious
    test.attrs["schema_rename_log"] = test_rename_log
    test.attrs["schema_suspicious_columns"] = test_suspicious

    # Preserve provenance for auditability
    fam = fam.rename(columns={"raw_id": "raw_id_fam", "source_file": "source_file_fam"})
    test = test.rename(columns={"raw_id": "raw_id_test", "source_file": "source_file_test"})

    id_cols = ["mouse_id", "group", "sex", "timepoint"]

    # Prefix all non-ID columns (including provenance + session) to keep everything and avoid collisions
    fam = _prefix_non_id_columns(fam, "fam_", id_cols=id_cols)
    test = _prefix_non_id_columns(test, "test_", id_cols=id_cols)

    obj_a, obj_b = cfg.objects
    fam_cols = _required_object_columns(fam, cfg.objects, "FAM", prefix="fam_")
    test_cols = _required_object_columns(test, cfg.objects, "TEST", prefix="test_")

    # Infer familiar object from FAM session
    a_any = (fam[fam_cols[(EXPLORE_TEMPLATE, obj_a)] + fam_cols[(IZ_TEMPLATE, obj_a)]] > 0).any(axis=1)
    b_any = (fam[fam_cols[(EXPLORE_TEMPLATE, obj_b)] + fam_cols[(IZ_TEMPLATE, obj_b)]] > 0).any(axis=1)

    if (a_any & b_any).any():
        bad_ids = fam.loc[(a_any & b_any), "mouse_id"].astype(str).tolist()
        raise ValueError(
            f"NOR: Both {obj_a} and {obj_b} appear non-zero in FAM for some mice "
            "(tracking/labeling issue). "
            f"mouse_id examples: {bad_ids[:20]}"
        )

    if (~a_any & ~b_any).any():
        bad_ids = fam.loc[(~a_any & ~b_any), "mouse_id"].astype(str).tolist()
        raise ValueError(
            f"NOR: Neither {obj_a} nor {obj_b} appears non-zero in FAM for some mice; "
            "cannot infer familiar type. "
            f"mouse_id examples: {bad_ids[:20]}"
        )

    fam["fam_familiar_type"] = np.where(a_any, obj_a, obj_b)

    # Collapse FAM identical objects (all copies) into a single familiar-time measure
    fam["fam_familiar_explore_time"] = np.where(
        fam["fam_familiar_type"] == obj_a,
        fam[fam_cols[(EXPLORE_TEMPLATE, obj_a)]].sum(axis=1),
        fam[fam_cols[(EXPLORE_TEMPLATE, obj_b)]].sum(axis=1),
    )
    fam["fam_familiar_iz_time"] = np.where(
        fam["fam_familiar_type"] == obj_a,
        fam[fam_cols[(IZ_TEMPLATE, obj_a)]].sum(axis=1),
        fam[fam_cols[(IZ_TEMPLATE, obj_b)]].sum(axis=1),
    )

    # TEST: one copy per object carries data (the others are empty in the export)
    test_explore_a = _single_test_copy(test, test_cols[(EXPLORE_TEMPLATE, obj_a)])
    test_explore_b = _single_test_copy(test, test_cols[(EXPLORE_TEMPLATE, obj_b)])
    test_iz_a = _single_test_copy(test, test_cols[(IZ_TEMPLATE, obj_a)])
    test_iz_b = _single_test_copy(test, test_cols[(IZ_TEMPLATE, obj_b)])

    merged = fam.merge(
        test,
        on=id_cols,
        how="inner",
        validate="one_to_one",
    )

    merged["familiar_type"] = merged["fam_familiar_type"]

    # Label TEST familiar vs novel based on familiar_type
    is_a = merged["familiar_type"] == obj_a
    merged["test_familiar_explore_time"] = np.where(is_a, merged[test_explore_a], merged[test_explore_b])
    merged["test_novel_explore_time"] = np.where(is_a, merged[test_explore_b], merged[test_explore_a])
    merged["test_familiar_iz_time"] = np.where(is_a, merged[test_iz_a], merged[test_iz_b])
    merged["test_novel_iz_time"] = np.where(is_a, merged[test_iz_b], merged[test_iz_a])

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

    assert_unique_key(merged, ["mouse_id", "timepoint"], where="NOR processed")

    merged = add_nor_sanity_flags(merged, min_total_s=cfg.min_explore_s)
    return merged

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

def _prefix_non_id_columns(df: pd.DataFrame, prefix: str, id_cols: list[str]) -> pd.DataFrame:
    """Prefix all columns not in id_cols with `prefix`, to avoid collisions after merge."""
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in id_cols}
    return df.rename(columns=rename)

PRIMARY_OUTCOMES: list[str] = [
    "di_explore",
    "pi_explore",
    "di_iz",
    "pi_iz",
]