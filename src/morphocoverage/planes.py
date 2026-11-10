"""Features per plane: one observation per object, one feature per plane (opt-in).

A field is one XY stage position, however many z-planes, channels or time
points were acquired there, and a feature extractor may measure each object
once per plane. The planes are never fields and a plane coordinate is never a
feature: ``validate_input_tables`` refuses ``z``, ``c`` and ``t`` in
``feature_cols``. When the question is how an object differs between planes
(its area at the top and at the bottom of a stack), ``planes_as_features``
makes that explicit: each declared feature becomes one feature per plane
value (``area`` measured at ``z`` 0, 1 and 2 becomes ``area_z0``,
``area_z1`` and ``area_z2``), with one row per object per field. The field
stays one FOV, and the FOV table is unchanged.

Objects are matched across planes by their identifier within the field
(``object_col``, ``label_id`` by default): the same label must name the same
object in every plane, as a 3D segmentation, or a 2D one tracked across
planes, provides. Labels that restart in each plane without being tracked do
not name one object, and nothing in the table can tell; the pivot refuses
only what it can see: a missing identifier, a label twice in one plane of a
field, or an object missing from a plane (kept as a missing value, NaN, on
request: ``allow_missing_planes=True``, for objects that span a few planes of
a stack or stacks of different depth). CellPatcher writes a patch's plane as
``z0`` (with ``z1 = z0 + 1``); without a ``z`` column that is the ``z`` axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._checks import is_number
from ._repeat_alias import refuse_repeat_alias
from ._wording import count
from .metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    column_names,
    drop_empty_optional_columns,
    repeated_names,
    resolve_fov_col,
)
from .schema import feature_dtype_problem

# The within-field axes a feature can be split along.
PLANE_AXES = ("z", "c", "t")


def planes_as_features(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None = None,
    *,
    features: Sequence[str],
    axes: Sequence[str] = ("z",),
    object_col: str = "label_id",
    fov_col: str | None = None,
    allow_missing_planes: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """The observation table with each feature split per plane, and the new feature names.

    ``objects`` holds one row per object per plane: the field identity (the
    MorphoBricks ``condition``, ``biological_repeat``, ``technical_repeat`` present, and
    ``fov_col`` or the identifier ``validate_input_tables`` would choose,
    with ``fovs`` read as it reads it), ``object_col``, the ``axes`` columns
    and the ``features``. The result holds one row per object per field, in
    order of first appearance: the field identity, ``object_col``, every
    other column that is constant over the object's planes (a column that
    varies between them, such as a per-plane ``image_file``, describes a
    plane and is dropped), and one column per feature and plane value, named
    ``<feature>_<axis><value>`` (``area_z0``; ``area_z0_c1`` for two axes),
    feature by feature in the order of ``features`` and planes in sorted
    order. The second value lists those names, to pass as ``feature_cols``.

    The ``z`` axis is read from CellPatcher's ``z0`` when the table has no
    ``z`` column (or with ``axes=("z0",)``); the names still say ``z``
    (``area_z0``), and a patch that spans several planes (``z1 - z0`` other
    than 1) is refused.

    Refused, with the reason: ``object_col``, an axis or a feature that is not
    a column; a missing ``object_col`` or axis value; an axis written as text
    in some rows and as numbers in others; a feature that is not numeric; a
    label twice in one plane of one field (objects cannot be matched across
    planes one to one); an object missing from a plane that other objects
    were measured in, unless ``allow_missing_planes=True`` (opt-in), which
    keeps its features in that plane as missing values (NaN) that every
    analysis then treats as it treats any missing value; a new name that is
    already a column. The former name ``repeat`` is refused, naming
    ``biological_repeat``.
    """
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame when provided")
    refuse_repeat_alias(objects=objects, fovs=fovs, fov_col=fov_col)
    features = list(column_names(features, "features") or ())
    axes = _plane_columns(objects, list(column_names(axes, "axes") or ()))
    if not isinstance(allow_missing_planes, bool):
        raise TypeError("allow_missing_planes must be True or False")
    _check_arguments(objects, features, axes, object_col)

    explicit = [fov_col] if fov_col is not None else []
    table, listed = drop_empty_optional_columns(objects, fovs, explicit)
    field_col = resolve_fov_col(table, listed, fov_col)
    if field_col not in table.columns:
        raise ValueError(f"the field identifier {field_col!r} is not a column of objects")
    identity = [c for c in MORPHOBRICKS_GROUP_COLUMNS if c in table.columns and c != field_col]
    field = [*identity, field_col]
    taken = {*field, object_col}
    for name, columns in (("axes", axes), ("features", features)):
        overlap = [c for c in columns if c in taken]
        if overlap:
            raise ValueError(f"{name} {overlap} identify the field or the object; name others")

    planes = _planes(table, axes)
    _check_one_row_per_plane(table, field, object_col, axes, planes, allow_missing_planes)
    wide, names = _wide(table, field, object_col, axes, features, planes)
    kept = [c for c in table.columns if c not in {*field, object_col, *axes, *features}]
    constant = _constant_columns(table, [*field, object_col], kept)
    clashes = [n for n in names if n in constant]
    if clashes:
        raise ValueError(
            f"the per-plane feature names {clashes} are already columns of objects; rename them"
        )
    rows = table.drop_duplicates([*field, object_col]).loc[:, [*field, object_col, *constant]]
    result = rows.merge(wide, on=[*field, object_col], how="left", validate="one_to_one")
    return result.reset_index(drop=True), names


def _plane_columns(objects: pd.DataFrame, axes: list) -> list:
    """The columns the ``axes`` are read from: CellPatcher's ``z0`` for ``z`` without a ``z``.

    Refuses a ``z0`` whose patches span several planes (``z1 - z0`` other than 1).
    """
    columns = [
        "z0" if axis == "z" and "z" not in objects.columns and "z0" in objects.columns else axis
        for axis in axes
    ]
    if "z0" in columns and "z1" in objects.columns:
        span = pd.to_numeric(objects["z1"], errors="coerce") - pd.to_numeric(
            objects["z0"], errors="coerce"
        )
        if (span.dropna() != 1).any():
            raise ValueError(
                "some patches span several planes (z1 - z0 is not 1), so z0 does not name one "
                "plane; split them per plane first (one row per plane, z1 = z0 + 1)"
            )
    return columns


def _check_arguments(objects: pd.DataFrame, features: list, axes: list, object_col: object) -> None:
    """The named columns exist, the axes are within-field axes, the features numbers."""
    if not features:
        raise ValueError("features must name at least one feature column to split per plane")
    if not axes:
        raise ValueError("axes must name at least one of 'z', 'c', 't'")
    for name, names in (("features", features), ("axes", axes)):
        repeated = repeated_names(names)
        if repeated:
            raise ValueError(f"{name} names {repeated[0]!r} more than once")
    unknown = [a for a in axes if a not in (*PLANE_AXES, "z0")]
    if unknown:
        raise ValueError(f"axes must be among {list(PLANE_AXES)} (or 'z0'), not {unknown}")
    _check_object_col(objects, object_col)
    for axis in axes:
        _check_axis(objects, axis)
    for feature in features:
        if feature not in objects.columns:
            raise ValueError(f"feature {feature!r} is not a column of objects")
        if feature_dtype_problem(objects[feature]) is not None:
            raise TypeError(f"feature {feature!r} must be real numbers")
        if feature in axes or feature == object_col:
            raise ValueError(f"feature {feature!r} is also an axis or the object identifier")


def _check_object_col(objects: pd.DataFrame, object_col: object) -> None:
    """The column naming each object within its field: a name, present, never missing."""
    if not isinstance(object_col, str):
        raise TypeError("object_col must be the name of the column identifying each object")
    if object_col not in objects.columns:
        raise ValueError(
            f"objects has no {object_col!r} column, so objects cannot be matched across planes; "
            "name the column that identifies each object within its field with object_col"
        )
    if objects[object_col].isna().any():
        raise ValueError(
            f"objects[{object_col!r}] is missing in some rows, so those objects cannot be "
            "matched across planes"
        )


def _check_axis(objects: pd.DataFrame, axis: str) -> None:
    """An axis column: present, never missing, and no plane written both as a number and
    as text."""
    if axis not in objects.columns:
        raise ValueError(f"objects has no {axis!r} column to split the features along")
    if objects[axis].isna().any():
        raise ValueError(f"objects[{axis!r}] is missing in some rows, so their plane is unknown")
    # A number written as text beside numbers ('0' against 0) names one plane twice; a
    # label such as 'max' (a projection) beside numbers is a plane of its own.
    is_text = objects[axis].map(lambda value: isinstance(value, str)).astype(bool)
    as_number = pd.to_numeric(objects[axis].where(is_text), errors="coerce")
    numeric_text = is_text & as_number.notna()
    if numeric_text.any() and not is_text.all():
        written = objects.loc[numeric_text, axis].iloc[0]
        number = objects.loc[~is_text, axis].iloc[0]
        raise ValueError(
            f"objects[{axis!r}] is text in some rows and a number in others ({written!r} "
            f"against {_label(number)}), so one plane would have two names; read the "
            "column as numbers in every file"
        )


def _planes(table: pd.DataFrame, axes: list) -> list[tuple]:
    """Every combination of axis values in the table, sorted (as text when types mix)."""
    combos = list(table.loc[:, axes].drop_duplicates().itertuples(index=False, name=None))
    try:
        return sorted(combos)
    except TypeError:  # fallback: types that cannot be compared: sorted as text
        return sorted(combos, key=lambda combo: tuple(map(str, combo)))


def _check_one_row_per_plane(
    table: pd.DataFrame,
    field: list,
    object_col: str,
    axes: list,
    planes: list,
    allow_missing: bool = False,
) -> None:
    """Every object has at most one row in each plane of its field, and one in every plane
    of the table unless ``allow_missing``."""
    key = [*field, object_col]
    twice = table.duplicated([*key, *axes], keep=False)
    if twice.any():
        example = table.loc[twice, [*key, *axes]].iloc[0]
        raise ValueError(
            f"{_where(example, field, object_col)} has several rows at "
            f"{_at(example[axes].tolist(), axes)}, so objects cannot be matched across planes "
            f"one to one; {object_col!r} must name one object of a field in each plane"
        )
    counts = table.groupby(key, sort=False, dropna=False).size()
    short = counts[counts < len(planes)]
    single = bool(len(short)) and bool(
        table.groupby(field, sort=False, dropna=False)[axes].nunique().max().max() == 1
    )
    if len(short) and (single or not allow_missing):
        first = short.index[0]
        values = first if isinstance(first, tuple) else (first,)
        example = pd.Series(dict(zip(key, values, strict=True)))
        rows = table
        for column, value in zip(key, values, strict=True):
            rows = rows[rows[column].isna()] if pd.isna(value) else rows[rows[column] == value]
        seen = set(rows.loc[:, axes].itertuples(index=False, name=None))
        missing = next(plane for plane in planes if plane not in seen)
        hint = (
            f" Each field holds a single plane: the field identifier {field[-1]!r} names planes, "
            "not stage positions; supply 'position_id' or name fov_col."
            if single
            else ""
        )
        raise ValueError(
            f"{_where(example, field, object_col)} has no row at {_at(list(missing), axes)}, "
            f"which other objects were measured in ({count(len(short), 'object')} miss a "
            f"plane); every object must be measured in every plane, or pass "
            f"allow_missing_planes=True to keep a missing plane as NaN.{hint}"
        )


def _wide(
    table: pd.DataFrame, field: list, object_col: str, axes: list, features: list, planes: list
) -> tuple[pd.DataFrame, list[str]]:
    """One row per object, one column per feature and plane, and the new names in order."""
    # CellPatcher's z0 is the z axis, and named so.
    names_of = ["z" if axis == "z0" else axis for axis in axes]
    labels = {
        plane: "_".join(f"{a}{_label(v)}" for a, v in zip(names_of, plane)) for plane in planes
    }
    if len(set(labels.values())) != len(labels):
        raise ValueError(
            f"plane values {list(labels)} give the same name twice (1 and 1.0?); make the "
            f"{axes} values distinct as text"
        )
    key = [*field, object_col]
    # Keys as object columns, so a missing group value (NaN) stays a key rather than a
    # dropped row, and every key column keeps its values.
    stacked = table.loc[:, [*key, *axes, *features]]
    wide = stacked.pivot_table(
        index=key, columns=axes, values=features, aggfunc="first", dropna=False, sort=False
    )
    names: list[str] = []
    columns: dict[str, pd.Series] = {}
    for feature in features:
        for plane in planes:
            column = (feature, *plane)
            name = f"{feature}_{labels[plane]}"
            names.append(name)
            columns[name] = wide[column] if column in wide.columns else np.nan
    result = pd.DataFrame(columns, index=wide.index).reset_index()
    return result, names


def _constant_columns(table: pd.DataFrame, key: list, columns: list) -> list:
    """The ``columns`` that hold one value over each object's planes."""
    if not columns:
        return []
    grouped = table.groupby(key, sort=False, dropna=False)
    constant = []
    for column in columns:
        try:
            spread = grouped[column].nunique(dropna=False)
        except TypeError:  # fallback: lists or arrays: not comparable, so not carried
            continue
        if bool((spread <= 1).all()):
            constant.append(column)
    return constant


def _label(value: Any) -> str:
    """A plane value as it appears in a feature name: ``0`` for 0, 0.0 or ``np.int64(0)``."""
    if isinstance(value, np.generic):
        value = value.item()
    if is_number(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _where(row: pd.Series, field: list, object_col: str) -> str:
    """``label_id 3 of field (condition=A, position_id=p1)``."""
    identity = ", ".join(f"{c}={row[c]}" for c in field)
    return f"{object_col} {row[object_col]} of field ({identity})"


def _at(values: list, axes: list) -> str:
    """``z=0`` or ``z=0, c=1``."""
    return ", ".join(f"{a}={_label(v)}" for a, v in zip(axes, values))


__all__ = ["PLANE_AXES", "planes_as_features"]
