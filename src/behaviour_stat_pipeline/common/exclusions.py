from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


REQUIRED_EXCLUSION_COLUMNS = {
    "mouse_id",
    "assay",
    "endpoint",
    "exclude_from",
    "reason",
}


def load_exclusions(path: str | Path | None) -> pd.DataFrame:
    """
    Load an optional exclusion file.

    Schema
    ------
    mouse_id      : canonical mouse identifier
    assay         : assay instance/type or '*' wildcard
    endpoint      : endpoint name or '*' wildcard
    exclude_from  : 'primary', 'sensitivity', or 'all'
    reason        : non-empty free-text reason

    Returns an empty dataframe with the expected columns if no file is supplied.
    A supplied path that does not exist is an error (a typo must not silently
    disable all exclusions).
    """
    cols = ["mouse_id", "assay", "endpoint", "exclude_from", "reason"]

    if path is None or str(path).strip() == "":
        return pd.DataFrame(columns=cols)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exclusion file configured but not found: {path}")

    exclusions = pd.read_csv(path)

    missing = REQUIRED_EXCLUSION_COLUMNS - set(exclusions.columns)
    if missing:
        raise ValueError(
            f"Exclusion file is missing required columns: {sorted(missing)}. "
            f"Observed columns: {exclusions.columns.tolist()}"
        )

    exclusions = exclusions[cols].copy()

    for col in ["mouse_id", "assay", "endpoint", "exclude_from", "reason"]:
        exclusions[col] = exclusions[col].astype(str).str.strip()

    if exclusions["mouse_id"].eq("").any():
        raise ValueError("Exclusion file contains blank mouse_id values.")

    if exclusions["assay"].eq("").any():
        raise ValueError("Exclusion file contains blank assay values.")

    if exclusions["endpoint"].eq("").any():
        raise ValueError("Exclusion file contains blank endpoint values.")

    if exclusions["reason"].eq("").any():
        raise ValueError("Exclusion file contains blank reason values.")

    allowed_scopes = {"primary", "sensitivity", "all"}
    observed_scopes = set(exclusions["exclude_from"].str.lower())
    bad_scopes = observed_scopes - allowed_scopes
    if bad_scopes:
        raise ValueError(
            f"Invalid exclude_from values in exclusion file: {sorted(bad_scopes)}. "
            f"Allowed values: {sorted(allowed_scopes)}"
        )

    exclusions["exclude_from"] = exclusions["exclude_from"].str.lower()
    exclusions["assay"] = exclusions["assay"].str.upper()

    return exclusions


def exclusions_for_endpoint(
    exclusions: pd.DataFrame,
    *,
    assay: str,
    endpoint: str,
    analysis_scope: Literal["primary", "sensitivity"],
) -> pd.DataFrame:
    """
    Select exclusions applying to one assay endpoint and analysis scope.
    Supports '*' wildcard for assay and endpoint.
    """
    if exclusions.empty:
        return exclusions.copy()

    assay = str(assay).upper()
    endpoint = str(endpoint)

    mask = (
        exclusions["assay"].isin([assay, "*"])
        & exclusions["endpoint"].isin([endpoint, "*"])
        & exclusions["exclude_from"].isin([analysis_scope, "all"])
    )

    return exclusions.loc[mask].copy()


def apply_exclusions_for_endpoint(
    df: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    assay: str,
    endpoint: str,
    analysis_scope: Literal["primary", "sensitivity"] = "primary",
    mouse_id_col: str = "mouse_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply endpoint-specific exclusions.

    Returns
    -------
    filtered_df
        Data after removing excluded mouse_ids for this assay/endpoint/scope.
    applied
        Exclusion rows that were actually relevant to this endpoint.
    """
    if mouse_id_col not in df.columns:
        raise KeyError(f"Dataframe missing mouse id column: {mouse_id_col}")

    applied = exclusions_for_endpoint(
        exclusions,
        assay=assay,
        endpoint=endpoint,
        analysis_scope=analysis_scope,
    )

    if applied.empty:
        return df.copy(), applied

    excluded_ids = set(applied["mouse_id"].astype(str))
    keep = ~df[mouse_id_col].astype(str).isin(excluded_ids)

    return df.loc[keep].copy(), applied