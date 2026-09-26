"""MorphoBricks metadata conventions shared across repositories."""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd

MORPHOBRICKS_SHARED_COLUMNS = (
    "condition",
    "repeat",
    "technical_repeat",
    "position_id",
    "image_file",
    "mask_file",
    "patch_file",
)
# Default analysis groups and FOV identity, outermost first: the condition, the
# biological repeat and, when present, the technical repeat (a well, a dish, a
# coverslip within a repeat; optional in the shared contract). Each is used
# when its column exists in the linked tables.
MORPHOBRICKS_GROUP_COLUMNS = ("condition", "repeat", "technical_repeat")
CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN = MORPHOBRICKS_GROUP_COLUMNS
# Group columns the shared contract makes optional.
OPTIONAL_GROUP_COLUMNS = (TECHNICAL_REPEAT_COLUMN,)
# Optional shared columns that a producer writing every column (CellPatcher
# does) leaves empty when it has no value: position_id when one file is one
# field, technical_repeat when there are no wells. An entirely empty one is
# read as absent; a partly empty one is still refused as ambiguous.
EMPTY_MEANS_ABSENT = (TECHNICAL_REPEAT_COLUMN, "position_id")

# Scientific FOV identity is the independently selected spatial acquisition
# position. position_id is preferred when storage layout and position do not
# map 1:1. image_file remains a convenient fallback when one file contains
# exactly one position; fov_id is the generic fallback.
MORPHOBRICKS_FOV_CANDIDATES = ("position_id", "image_file", "fov_id")

# Coordinates/dimensions nested inside one spatial position. These are
# acquisition metadata, never phenotype features in MorphoCoverage.
WITHIN_FOV_METADATA_COLUMNS = ("z", "z0", "z1", "c", "t")


def column_names(columns: object, name: str) -> tuple | None:
    """``columns`` read once, as a tuple of column names; ``None`` stays ``None``.

    Any iterable of names works (a list, a generator, a ``pd.Index``, a numpy
    array), and is read exactly once, so a generator is not consumed by a first
    look and then seen as empty (which would pool every group). A single string
    is refused rather than read letter by letter, as is anything that is not a
    collection of hashable names.
    """
    if columns is None:
        return None
    if isinstance(columns, str):
        raise TypeError(
            f"{name} must be a sequence of column names, not a string; "
            f"pass a list: {name}=[{columns!r}]"
        )
    if not isinstance(columns, Iterable):
        raise TypeError(f"{name} must be a sequence of column names, not {type(columns).__name__}")
    # numpy names (from an array) as Python ones, so group labels print as plain names.
    names = tuple(n.item() if isinstance(n, np.generic) else n for n in columns)
    unusable = [n for n in names if not isinstance(n, Hashable)]
    if unusable:
        raise TypeError(f"{name} must hold column names; {unusable[0]!r} cannot be one")
    return names


def empty_values(values: pd.Series) -> pd.Series:
    """Which entries hold no value: missing, or text that is blank (``""``, ``"  "``).

    Vectorised where it can be, since the validator runs this on every identity
    column. Blanks are searched for in text, object and categorical columns;
    other values in an object column (numbers, dates, booleans) are never blank.
    """
    missing = values.isna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        blank = [c for c in values.cat.categories if isinstance(c, str) and not c.strip()]
        return missing | values.isin(blank) if blank else missing
    if values.dtype == object or pd.api.types.is_string_dtype(values.dtype):
        try:
            blank = values.str.strip().eq("")  # NaN for entries that are not text
        except AttributeError:
            # An object column with no text at all (dates, Python ints, booleans):
            # pandas refuses the text accessor, so the entries are tested one by one.
            blank = values.map(lambda value: isinstance(value, str) and not value.strip())
        missing = missing | blank.fillna(False).astype(bool)
    return missing


def column_has_values(table: pd.DataFrame, column: str) -> bool:
    """Whether ``table`` has ``column`` with at least one value in it."""
    return column in table.columns and not bool(empty_values(table[column]).all())


def drop_empty_optional_columns(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    explicit_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The tables without the ``EMPTY_MEANS_ABSENT`` columns that hold no value in any of them.

    A column named explicitly (``group_cols``, ``fov_col``) is kept, so an
    empty column the caller asked for is still refused downstream.
    """
    for column in EMPTY_MEANS_ABSENT:
        if column in explicit_columns:
            continue
        present = [t for t in (objects, fovs) if t is not None and column in t.columns]
        if present and not any(column_has_values(t, column) for t in present):
            if column in objects.columns:
                objects = objects.drop(columns=column)
            if fovs is not None and column in fovs.columns:
                fovs = fovs.drop(columns=column)
    return objects, fovs


def _tables(objects: pd.DataFrame, fovs: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
    """The linked tables present, with the names used in error messages."""
    tables = [("objects", objects)]
    if fovs is not None:
        tables.append(("fovs", fovs))
    return tables


def _squashed(name: str) -> str:
    """Lowercase, no separators, no plural ``s``: ``Position_IDs`` becomes ``positionid``."""
    return re.sub(r"[\s_-]+", "", name.lower()).removesuffix("s")


# The canonical names by their squashed form, for ``canonical_spelling``.
_CANONICAL_BY_SQUASHED = {
    _squashed(name): name for name in (*MORPHOBRICKS_GROUP_COLUMNS, *MORPHOBRICKS_FOV_CANDIDATES)
}


def canonical_spelling(column: object) -> str | None:
    """The MorphoBricks column that ``column`` misspells, or ``None``.

    A near miss differs only in case, in spaces, hyphens, camelCase or no
    separator for underscores, in a plural ``s``, in surrounding whitespace,
    or in a byte-order mark that a spreadsheet wrote before the first header
    (``Repeat``, ``technical repeat``, ``technicalRepeat``, ``PositionID``,
    ``Conditions``, ``condition ``, ``\ufeffcondition``). Such a column would
    otherwise be ignored and its groups pooled without a word. The exact name
    is not a near miss.
    """
    if not isinstance(column, str):
        return None
    meant = _CANONICAL_BY_SQUASHED.get(_squashed(column.replace("\ufeff", "")))
    return meant if meant is not None and meant != column else None


def check_canonical_spellings(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    explicit_columns: Sequence[str],
) -> None:
    """Reject near-miss spellings such as Repeat that would bypass defaults."""
    for table_name, table in _tables(objects, fovs):
        for column in table.columns:
            if column in explicit_columns:
                continue
            meant = canonical_spelling(column)
            if meant is not None:
                raise ValueError(
                    f"{table_name} has column {column!r}, a non-canonical spelling of "
                    f"MorphoBricks column {meant!r}. Rename it to "
                    f"{meant!r} or name it explicitly via fov_col/group_cols."
                )


def _canonical_presence(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    column: str,
) -> bool:
    """Return whether a canonical identity/group column is present in linked tables."""
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
    """Resolve the spatial acquisition-position identifier.

    Scientific FOV identity means one independently chosen XY acquisition
    position, not one z-plane, channel, time point, or physical file.

    Resolution order when fov_col is omitted:
    1. position_id -- preferred explicit spatial-position identifier;
    2. image_file -- valid when one file corresponds to exactly one position;
    3. fov_id -- generic fallback.

    A candidate present in only one linked table is an error, never a reason
    to fall back to the next candidate.
    """
    if fov_col is not None:
        if not isinstance(fov_col, str):
            raise TypeError("fov_col must be a column name")
        return fov_col

    for candidate in MORPHOBRICKS_FOV_CANDIDATES:
        if _canonical_presence(objects, fovs, candidate):
            return candidate

    raise ValueError(
        "Could not resolve a spatial FOV/position identifier automatically. "
        "Provide fov_col explicitly, or supply 'position_id' (preferred), "
        "'image_file' when one file equals one position, or 'fov_id'."
    )


def resolve_group_cols(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve analysis groups using canonical MorphoBricks columns when implicit."""
    if group_cols is not None:
        return column_names(group_cols, "group_cols")

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
    """Resolve columns that, together with fov_col, identify one spatial FOV."""
    identity = list(group_cols)
    for column in MORPHOBRICKS_GROUP_COLUMNS:
        if column in identity or column == fov_col:
            continue
        if _canonical_presence(objects, fovs, column):
            identity.append(column)
    return tuple(identity)
