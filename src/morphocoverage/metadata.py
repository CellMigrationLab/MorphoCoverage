"""MorphoBricks metadata conventions shared across repositories."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

MORPHOBRICKS_SHARED_COLUMNS = (
    "condition",
    "repeat",
    "image_file",
    "mask_file",
    "patch_file",
)
MORPHOBRICKS_GROUP_COLUMNS = ("condition", "repeat")
MORPHOBRICKS_FOV_CANDIDATES = ("image_file", "fov_id")


def _tables(objects: pd.DataFrame, fovs: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
    tables = [("objects", objects)]
    if fovs is not None:
        tables.append(("fovs", fovs))
    return tables


def check_canonical_spellings(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    explicit_columns: Sequence[str],
) -> None:
    """Reject near-miss spellings such as ``Repeat`` that would bypass defaults.

    A column named ``Repeat`` (as emitted by some MorphoBricks tools) would
    otherwise be ignored by the canonical ``repeat`` handling, silently pooling
    biological repeats. Columns the caller named explicitly are accepted.
    """
    canonical = {*MORPHOBRICKS_GROUP_COLUMNS, *MORPHOBRICKS_FOV_CANDIDATES}
    for table_name, table in _tables(objects, fovs):
        for column in table.columns:
            if not isinstance(column, str) or column in explicit_columns:
                continue
            if column.lower() in canonical and column != column.lower():
                raise ValueError(
                    f"{table_name} has column {column!r}, a non-canonical spelling of "
                    f"MorphoBricks column {column.lower()!r}. Rename it to "
                    f"{column.lower()!r} or name it explicitly via fov_col/group_cols."
                )


def _canonical_presence(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    column: str,
) -> bool:
    """Return whether a canonical column is present, requiring both linked tables to agree."""
    in_objects = column in objects.columns
    if fovs is None:
        return in_objects
    in_fovs = column in fovs.columns
    if in_objects != in_fovs:
        present, absent = ("objects", "fovs") if in_objects else ("fovs", "objects")
        raise ValueError(
            f"MorphoBricks column {column!r} is present in {present} but absent from "
            f"{absent}. Add it to both linked tables so FOV identity and grouping are "
            "unambiguous."
        )
    return in_objects


def resolve_fov_col(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    fov_col: str | None,
) -> str:
    """Resolve the FOV identifier, preferring the MorphoBricks image_file contract.

    ``image_file`` identifies the acquired FOV: one image file, including a
    z/t/c stack, is one FOV. ``fov_id`` is used only when no ``image_file``
    column exists. A candidate present in only one linked table is an error,
    never a reason to fall back to the next candidate.
    """
    if fov_col is not None:
        if not isinstance(fov_col, str):
            raise TypeError("fov_col must be a column name")
        return fov_col

    for candidate in MORPHOBRICKS_FOV_CANDIDATES:
        if _canonical_presence(objects, fovs, candidate):
            return candidate

    raise ValueError(
        "Could not resolve an FOV identifier automatically. "
        "Provide fov_col explicitly, or supply the MorphoBricks 'image_file' "
        "column (preferred) or a 'fov_id' column."
    )


def resolve_group_cols(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve analysis groups using canonical MorphoBricks columns when implicit.

    None means use canonical condition/repeat columns. An explicit empty
    sequence means pool intentionally. A canonical column present in only one
    linked table is an error rather than a silent reason to pool.
    """
    if group_cols is not None:
        if isinstance(group_cols, str):
            raise TypeError("group_cols must be a sequence of column names, not a string")
        return tuple(group_cols)

    return tuple(
        column
        for column in MORPHOBRICKS_GROUP_COLUMNS
        if _canonical_presence(objects, fovs, column)
    )


def resolve_identity_cols(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Sequence[str],
    fov_col: str,
) -> tuple[str, ...]:
    """Resolve the columns that, together with ``fov_col``, identify one FOV.

    Identity always includes the analysis group columns and any canonical
    condition/repeat columns, even when those are pooled. Pooling changes which
    FOVs are evaluated together; it never merges distinct FOVs that reuse a
    name (e.g. ``img_001.tif`` in every repeat folder).
    """
    identity = list(group_cols)
    for column in MORPHOBRICKS_GROUP_COLUMNS:
        if column in identity or column == fov_col:
            continue
        if _canonical_presence(objects, fovs, column):
            identity.append(column)
    return tuple(identity)
