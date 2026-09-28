"""The plane rule of the input contract (private): z, c and t planes are not fields.

A z-stack, a set of channels or a time series at one stage position is one
field of view. These checks, applied by the validator (``schema``) and the live
session (``online``), refuse or warn when plane files would be counted as
fields, and keep plane coordinates out of the features.

Shared-private: every name here is used by ``schema``, ``online`` and
``_data_rules``; ``schema`` re-exports them.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import pandas as pd

from .metadata import (
    WITHIN_FOV_METADATA_COLUMNS,
)


def _within_fov_axes(objects: pd.DataFrame, feature_cols: Sequence[str]) -> list[str]:
    """The z/c/t-style columns that describe acquisition within a position."""
    return [
        column
        for column in WITHIN_FOV_METADATA_COLUMNS
        if column in objects.columns and column not in feature_cols
    ]


def _check_within_fov_axes(
    objects: pd.DataFrame,
    fov_key: tuple[str, ...],
    feature_cols: Sequence[str],
    *,
    fov_col_was_chosen: bool,
    fovs: pd.DataFrame | None = None,
) -> None:
    """Refuse FOV identities that would silently count planes, channels or frames as FOVs.

    - A position spread over several ``image_file`` values must say, on every
      row, which plane/channel/frame each file is (``z``/``c``/``t``), and
      those values must tell the files apart.
    - When the FOV column was resolved automatically to a file-level
      identifier and each FOV holds a single value of a ``z``/``c``/``t``
      column (on the observations or in the FOV table) that varies across
      FOVs, each "FOV" looks like one plane of a position. The caller must
      supply ``position_id`` or name ``fov_col`` (``_check_planes_as_fields``).
    """
    key = list(fov_key)
    fov_col = key[-1]
    axes = _within_fov_axes(objects, feature_cols)
    if "image_file" in objects.columns and fov_col != "image_file":
        files = objects.groupby(key, dropna=False)["image_file"].transform("nunique")
        split = objects.loc[files > 1]
        if len(split):
            if not axes:
                raise ValueError(
                    f"a {fov_col!r} maps to multiple image_file values, but no z/c/t "
                    "metadata says which plane, channel or time point each file is"
                )
            if split[axes].isna().all(axis=1).any():
                raise ValueError(
                    "rows from a position spread over several files are missing z/c/t "
                    "metadata; document the within-position coordinate on every row"
                )
            per_plane = split.groupby([*key, *axes], dropna=False)["image_file"].nunique()
            if (per_plane > 1).any():
                example = per_plane[per_plane > 1].index[0]
                raise ValueError(
                    f"z/c/t values {axes} do not tell the files of one position apart "
                    f"(e.g. {example}); each file must be a distinct plane/channel/frame"
                )
    if fov_col_was_chosen and fov_col in ("image_file", "fov_id"):
        _check_planes_as_fields(objects, fovs, key, axes, feature_cols)


def _check_planes_as_fields(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    key: list[str],
    axes: list[str],
    feature_cols: Sequence[str],
) -> None:
    """Refuse (or, with one row per field, warn about) file-level "FOVs" that look like planes.

    A ``z``/``c``/``t`` column is read from the observations, or from the FOV
    table when the observations do not carry it. When every field holds a
    single value of it and the value varies across fields, each field may be
    one plane, channel or frame of a position. With several observations in a
    field that is refused. With one row per field (per-plane summaries) the
    pattern is also what one summary per real position with a varying ``z``
    would look like, so it is a warning, not a refusal.

    In the FOV table a ``z`` or ``t`` may also be a stage coordinate of the
    position (an autofocus height, an acquisition time), distinct in every
    field. Planes of positions share their values (``z`` 0, 1, 2 at every
    position), so a FOV-table column is read as planes only when one of its
    values recurs in several fields.
    """
    fov_col = key[-1]
    sources = [(axis, objects) for axis in axes]
    if fovs is not None and all(column in fovs.columns for column in key):
        sources += [
            (axis, fovs)
            for axis in WITHIN_FOV_METADATA_COLUMNS
            if axis in fovs.columns
            and axis not in objects.columns
            and axis not in feature_cols
            and fovs[axis].dropna().duplicated().any()
        ]
    if not sources:
        return
    several = bool((objects.groupby(key, dropna=False)[fov_col].size() > 1).any())
    for axis, table in sources:
        per_fov = table.groupby(key, dropna=False)[axis].nunique(dropna=False)
        if not ((per_fov == 1).all() and table[axis].nunique(dropna=False) > 1):
            continue
        found = "" if table is objects else " (in the FOV table)"
        if several:
            raise ValueError(
                f"every {fov_col!r} holds a single {axis!r} value{found} that varies across "
                f"{fov_col!r} values, so each may be one plane/channel/frame of a "
                "position rather than a spatial FOV. Supply 'position_id', or pass "
                f"fov_col={fov_col!r} explicitly if each really is one position."
            )
        warnings.warn(
            f"every {fov_col!r} has one row and its own {axis!r} value{found}, so each may be "
            "one plane/channel/frame of a position counted as a spatial FOV. Supply "
            f"'position_id', or pass fov_col={fov_col!r} explicitly if each really is one "
            "position.",
            UserWarning,
            stacklevel=4,
        )
        return


# Why a within-field coordinate is never a feature, and what to do instead (both rule sets).
PLANE_FEATURE_ADVICE = (
    "every plane, channel and time point of a position belongs to one field, so its "
    "coordinate is metadata, never a phenotype. Leave it out of feature_cols; to compare "
    "a feature between planes, turn it into one feature per plane with "
    "morphocoverage.planes_as_features (or Setup(per_plane=('z',)))"
)


def plane_features(feature_cols: Sequence[object]) -> list:
    """The within-field coordinates (``z``, ``z0``, ``z1``, ``c``, ``t``) named as features."""
    return [f for f in dict.fromkeys(feature_cols) if f in WITHIN_FOV_METADATA_COLUMNS]
