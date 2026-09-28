"""The plane rule of the input contract (private): z, c and t planes are not fields.

A z-stack, a set of channels or a time series at one stage position is one
field of view. These checks, applied by the validator (``schema``) and the live
session (``online``), refuse or warn when plane files would be counted as
fields (a ``z``/``c``/``t`` column, or file names that differ only by a plane
token), refuse the planes of one field counted as new observations of its
objects (``_check_planes_within_fields``) and several stage positions counted
as one field (``_check_positions_within_fields``), and keep plane coordinates
out of the features.

Shared-private: every name here is used by ``schema``, ``online`` and
``_data_rules``; ``schema`` re-exports them.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import pandas as pd

from ._wording import count
from .metadata import (
    WITHIN_FOV_METADATA_COLUMNS,
    position_like,
)

# Observation units whose rows each measure one plane, channel or frame (not an
# object), so that a field holding rows of several planes is what they describe.
PLANE_LEVEL_UNITS = frozenset({"slice", "plane", "frame"})

# A plane, channel or frame token in a file name: ``_z03``, ``-t1``, ``_ch2``,
# ``_plane4`` between separators, and the Operetta/Harmony ``p01``, ``-ch1`` and
# ``sk1`` of ``r01c01f01p01-ch1sk1fk1fl1.tiff``. A bare ``c`` is not a token: ``_C02``
# is as often plate well C02 (no plate row is T or Z).
_PLANE_TOKEN = re.compile(
    r"(?:(?<=[_\-. ])|^)(?:z|t|ch|plane|slice|frame|tp)_?\d+(?=[_\-. ]|$)"
    r"|(?<=f\d\d)p\d+(?=-ch\d)|-ch\d+(?=sk\d)|(?<=\d)sk\d+(?=fk\d)",
    re.IGNORECASE,
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
    if fov_col_was_chosen and fov_col == "image_file":
        listed = fovs is not None and all(column in fovs.columns for column in key)
        _check_plane_file_names(fovs if listed else objects, key)


def _check_plane_file_names(table: pd.DataFrame, key: list[str]) -> None:
    """Warn when ``image_file`` names differ only by a plane, channel or frame token.

    ``pos_z01.tif`` and ``pos_z02.tif`` in one group are then two fields
    although they may be two planes of one stage position, when the plane is
    written only in the file name (no ``z`` column says so). A warning, not a
    refusal: a token can also be part of a real position's name.
    """
    files = table.loc[:, key].drop_duplicates()
    names = files[key[-1]].tolist()
    if not names or not all(isinstance(name, str) for name in names):
        return
    stems = [_PLANE_TOKEN.sub("", name) for name in names]
    collapsed = files.assign(_stem=stems).groupby([*key[:-1], "_stem"], dropna=False, sort=False)[
        key[-1]
    ]
    per_stem = collapsed.nunique()
    if not (per_stem > 1).any():
        return
    # Two names, and no count: the live session checks every field so far, and an unchanged
    # message is shown once.
    first, second = sorted(collapsed.unique()[per_stem > 1].iloc[0])[:2]
    warnings.warn(
        f"the image_file names that differ only by a plane, channel or frame token ({first!r} "
        f"and {second!r}, ...) are counted as separate fields, so each may be one "
        "plane/channel/frame of a position rather than a spatial FOV. Supply 'position_id' "
        "(the same for every plane of one stage position), or pass fov_col='image_file' "
        "explicitly if each file really is one position.",
        UserWarning,
        stacklevel=4,
    )


def _check_planes_within_fields(
    objects: pd.DataFrame,
    fov_key: tuple[str, ...],
    feature_cols: Sequence[str],
    observation_unit: str,
) -> None:
    """Refuse a field whose rows come from several planes, channels or frames (D18).

    An object measured in each of five z-planes of one field is one object,
    not five: counted as five, every rule on objects (the JCB cell count, the
    object spread) sees five times the cells it has. Unless the declared unit
    is a plane-level measurement (``PLANE_LEVEL_UNITS``), a ``z``/``c``/``t``
    column (never a feature) with several values in one field is refused,
    pointing to ``planes_as_features``, a single-plane filter or a plane-level
    unit.
    """
    if observation_unit in PLANE_LEVEL_UNITS or objects.empty:
        return
    key = list(fov_key)
    for axis in _within_fov_axes(objects, feature_cols):
        per_field = objects.groupby(key, dropna=False, sort=False)[axis].nunique()
        several = per_field[per_field > 1]
        if several.empty:
            continue
        field = several.index[0]
        values = field if isinstance(field, tuple) else (field,)
        where = ", ".join(f"{c}={v!r}" for c, v in zip(key, values, strict=True))
        split = "z" if axis in ("z0", "z1") else axis
        raise ValueError(
            f"{count(len(several), 'field')} {'holds' if len(several) == 1 else 'hold'} rows "
            f"of several {axis!r} values (the field "
            f"{where} holds {count(int(several.iloc[0]), axis + ' value')}), so an object "
            f"measured in every plane, channel or frame would count once per plane "
            f"({count(len(objects), 'row')} as {observation_unit} observations). A field is one "
            "stage position and its planes are not new observations. Turn each feature into "
            f"one per plane with morphocoverage.planes_as_features(..., axes=({split!r},)), "
            f"keep one plane (objects[objects[{axis!r}] == ...]), or, if each row measures a "
            "whole plane, declare observation_unit='slice'. A column that is an object's own "
            f"position (a 3D centroid) needs another name than {axis!r}."
        )


def _check_positions_within_fields(
    objects: pd.DataFrame,
    fov_key: tuple[str, ...],
    feature_cols: Sequence[str],
    *,
    fov_col_was_chosen: bool,
) -> None:
    """Refuse an automatically chosen field that holds several stage positions.

    A multi-position file (one ``.nd2`` or ``.czi`` per well, its positions a
    ``series``, ``scene`` or ``site`` column) is several fields under one
    ``image_file``: counted as one, the well's positions are pooled into one
    field. A position-like column (``metadata.position_like``) that is neither
    a feature nor part of the field key and varies within a field is refused;
    ``position_id``, or an explicit ``fov_col``, says what a field is.
    """
    if not fov_col_was_chosen:
        return
    key = list(fov_key)
    candidates = [
        column
        for column in objects.columns
        if position_like(column) and column not in feature_cols and column not in key
    ]
    for column in candidates:
        try:
            per_field = objects.groupby(key, dropna=False, sort=False)[column].nunique()
        except TypeError:  # lists or arrays: not a position index
            continue
        several = per_field[per_field > 1]
        if several.empty:
            continue
        raise ValueError(
            f"{count(len(several), key[-1] + ' value')} "
            f"{'holds' if len(several) == 1 else 'hold'} several {column!r} values (up to "
            f"{int(several.max())}), so one field identifier covers several stage positions, as "
            "in a multi-position file (.nd2, .czi, .lif). Supply 'position_id', one value per "
            f"stage position (for example the file and the {column!r} together); if {column!r} "
            "is not a stage position, rename it, or pass fov_col explicitly."
        )


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
    "morphocoverage.planes_as_features, then declare the split names (area_z0, ...)"
)


def plane_features(feature_cols: Sequence[object]) -> list:
    """The within-field coordinates (``z``, ``z0``, ``z1``, ``c``, ``t``) named as features."""
    return [f for f in dict.fromkeys(feature_cols) if f in WITHIN_FOV_METADATA_COLUMNS]
