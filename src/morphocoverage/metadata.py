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


def resolve_fov_col(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    fov_col: str | None,
) -> str:
    """Resolve the FOV identifier, preferring the MorphoBricks image_file contract."""
    if fov_col is not None:
        return fov_col

    for candidate in MORPHOBRICKS_FOV_CANDIDATES:
        if candidate not in objects.columns:
            continue
        if fovs is None or candidate in fovs.columns:
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

    None means use canonical condition/repeat columns that are present in both
    linked tables. An explicit empty sequence means pool intentionally.
    """
    if group_cols is not None:
        return tuple(group_cols)

    resolved = []
    for column in MORPHOBRICKS_GROUP_COLUMNS:
        if column not in objects.columns:
            continue
        if fovs is not None and column not in fovs.columns:
            continue
        resolved.append(column)
    return tuple(resolved)
