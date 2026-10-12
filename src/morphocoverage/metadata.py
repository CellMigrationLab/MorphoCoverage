"""MorphoBricks metadata conventions shared across repositories.

Column resolution (the FOV identifier, the analysis groups, the identity
columns), near-miss spellings of the shared columns and of the within-field
coordinates, and the optional shared columns a producer leaves empty.

Shared-private: ``_tables`` (the linked tables with their names in messages; used
by ``schema``, and by ``_data_rules`` with its own names).
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._spelling import typing_error_of

MORPHOBRICKS_SHARED_COLUMNS = (
    "condition",
    "biological_repeat",
    "technical_repeat",
    "position_id",
    "image_file",
    "mask_file",
    "patch_file",
)
# Default analysis groups and FOV identity, outermost first: the condition, the
# biological repeat and, when present, the technical repeat (a well, a dish, a
# coverslip within a repeat; optional in the shared contract). Each is used
# when its column exists in the linked tables. The biological repeat's former
# name, ``repeat``, is still read, with a warning (``_former_repeat``).
MORPHOBRICKS_GROUP_COLUMNS = ("condition", "biological_repeat", "technical_repeat")
CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN = MORPHOBRICKS_GROUP_COLUMNS
# Optional: what each condition acts on (the gene two siRNAs silence), so the report can say
# whether independent reagents for one target agree. Not a grouping column.
TARGET_COLUMN = "target"
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
    is refused rather than read letter by letter, as is a mapping (whose keys
    alone would be read) and anything that is not a collection of hashable names.
    """
    if columns is None:
        return None
    if isinstance(columns, str):
        raise TypeError(
            f"{name} must be a sequence of column names, not a string; "
            f"pass a list: {name}=[{columns!r}]"
        )
    if isinstance(columns, (set, frozenset)):
        # A set has no fixed order: the feature matrix and the group order would
        # then change from one Python process to the next.
        raise TypeError(
            f"{name} must be an ordered sequence of column names, not a set, whose order "
            f"changes between runs; pass a list, e.g. {name}=sorted(...)"
        )
    if isinstance(columns, Mapping):  # its keys alone: {"area": False} would select "area"
        raise TypeError(f"{name} must be a sequence of column names, not a mapping")
    if not isinstance(columns, Iterable):
        raise TypeError(f"{name} must be a sequence of column names, not {type(columns).__name__}")
    # numpy names (from an array) as Python ones, so group labels print as plain names.
    names = tuple(n.item() if isinstance(n, np.generic) else n for n in columns)
    unusable = [n for n in names if not isinstance(n, Hashable)]
    if unusable:
        raise TypeError(f"{name} must hold column names; {unusable[0]!r} cannot be one")
    return names


def repeated_names(names: Sequence[Hashable]) -> list:
    """The names that occur more than once in ``names``, each once, in order of first repeat."""
    seen: set = set()
    repeated: dict = {}
    for name in names:
        if name in seen:
            repeated[name] = None
        seen.add(name)
    return list(repeated)


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
        except AttributeError:  # fallback: an object column with no text: tested one by one below
            # An object column with no text at all (dates, Python ints, booleans):
            # pandas refuses the text accessor, so the entries are tested one by one.
            blank = values.map(lambda value: isinstance(value, str) and not value.strip())
        missing = missing | blank.fillna(False).astype(bool)
    return missing


def column_has_values(table: pd.DataFrame, column: str) -> bool:
    """Whether ``table`` has ``column`` with at least one value in it."""
    return column in table.columns and not bool(empty_values(table[column]).all())


def empty_optional_columns(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    explicit_columns: Sequence[str],
) -> dict[str, list[str]]:
    """The ``EMPTY_MEANS_ABSENT`` columns read as absent, with the tables they are empty in.

    Keys are the columns, values the tables (``"objects"``, ``"fovs"``) in
    which the column exists but holds no value in any row; a table without
    rows counts only when no table has a value of the column. A column named
    explicitly (``group_cols``, ``fov_col``) is never listed.
    """
    empty: dict[str, list[str]] = {}
    for column in EMPTY_MEANS_ABSENT:
        if column in explicit_columns:
            continue
        present = [(name, t) for name, t in _tables(objects, fovs) if column in t.columns]
        where = [name for name, t in present if not column_has_values(t, column)]
        if len(where) < len(present):
            # The other table has values: a table without rows (a live session's
            # first field, an empty subset) says nothing about the column.
            where = [name for name, t in present if name in where and len(t)]
        if where:
            empty[column] = where
    return empty


def drop_empty_optional_columns(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    explicit_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The tables without the ``EMPTY_MEANS_ABSENT`` columns that hold no value in them.

    Such a column is read as absent from each table that holds no value of it
    (``empty_optional_columns``): from both when it is empty in both, from one
    table only when the other has values. A producer that writes every column
    (CellPatcher writes ``technical_repeat`` and ``position_id``) leaves the
    feature table's copy empty while the FOV table knows the wells or the
    positions: the FOV table's values are then used, attached to the
    observations like any FOV-level group column (``technical_repeat``), or,
    for ``position_id``, through the file identifier both tables hold
    (``image_file``, else ``fov_id``) when each of its values is one
    position (``_attach_position_ids``). A column named explicitly
    (``group_cols``, ``fov_col``) is kept, so an empty column the caller
    asked for is still refused downstream.
    """
    for column, where in empty_optional_columns(objects, fovs, explicit_columns).items():
        if "objects" in where:
            objects = objects.drop(columns=column)
        if "fovs" in where:
            assert fovs is not None
            fovs = fovs.drop(columns=column)
        if column == "position_id" and where == ["objects"]:
            objects = _attach_position_ids(objects, fovs)
    return objects, fovs


def _attach_position_ids(objects: pd.DataFrame, fovs: pd.DataFrame | None) -> pd.DataFrame:
    """``objects`` with the FOV table's ``position_id``, when the join is unambiguous.

    The join runs through the first of ``image_file`` and ``fov_id`` that both
    tables hold, together with the group columns both tables hold (so file
    names that restart in each repeat are matched within their repeat). It is
    made only when every observation's file is listed and each listed file
    has one ``position_id``; otherwise ``objects`` is returned as it is and
    ``position_id``, now only in the FOV table, is refused with the other
    errors.
    """
    if fovs is None or "position_id" not in fovs.columns:
        return objects
    joins = [
        c for c in MORPHOBRICKS_FOV_CANDIDATES[1:] if c in objects.columns and c in fovs.columns
    ]
    if not joins:
        return objects
    key = [c for c in MORPHOBRICKS_GROUP_COLUMNS if c in objects.columns and c in fovs.columns]
    key.append(joins[0])
    try:
        mapping = fovs.loc[:, [*key, "position_id"]].drop_duplicates()
        listed = pd.MultiIndex.from_frame(objects[key]).isin(pd.MultiIndex.from_frame(fovs[key]))
    except TypeError:  # fallback: identifiers that cannot be compared (lists) are refused later
        return objects
    if mapping.duplicated(key).any() or not listed.all():
        return objects
    return objects.merge(mapping, on=key, how="left", validate="many_to_one").set_axis(
        objects.index
    )


def attach_mapping(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    column: str,
    identity: Sequence[str],
    fov_col: str,
) -> tuple[list[str], pd.DataFrame]:
    """How an FOV-table-only ``column`` is attached to the observations: the join key and
    the FOV table's (key, ``column``) pairs.

    The key is ``fov_col``. When one ``fov_col`` value has several values of
    ``column`` (a field name that restarts in each repeat, ``f1`` in every
    plate), the identity columns (other than ``column``) that both tables hold
    are added in front: ``f1`` still names one field within its repeat, so a
    condition listed only in the FOV table is attached through (repeat,
    field). The caller refuses the join when the pairs still repeat a key.
    """
    key = [fov_col]
    mapping = fovs.loc[:, [fov_col, column]].drop_duplicates()
    if mapping.duplicated(fov_col).any():
        shared = [
            c
            for c in dict.fromkeys(identity)
            if c not in (column, fov_col) and c in objects.columns and c in fovs.columns
        ]
        if shared:
            key = [*shared, fov_col]
            mapping = fovs.loc[:, [*key, column]].drop_duplicates()
    return key, mapping


def _tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    names: tuple[str, str] = ("objects", "fovs"),
) -> list[tuple[str, pd.DataFrame]]:
    """The linked tables present, each with its name in messages (``names``)."""
    tables = [(names[0], objects)]
    if fovs is not None:
        tables.append((names[1], fovs))
    return tables


# A copy of a column that pandas renamed on reading ("condition.1" for a second "condition").
_PANDAS_COPY = re.compile(r"\.\d+$")


def _squashed(name: str) -> str:
    """Lowercase, no separators (spaces, ``_``, ``-``, ``.``), no plural ``s`` and no pandas
    copy suffix: ``Position_IDs`` becomes ``positionid``, ``technical.repeat``
    ``technicalrepeat`` and ``Repeat.1`` ``repeat``."""
    return re.sub(r"[\s_.-]+", "", _PANDAS_COPY.sub("", name).lower()).removesuffix("s")


def _plain_spelling(name: str) -> str:
    """Lowercase, without whitespace or a pandas copy suffix: ``Z``, `` z`` and ``z.1`` are ``z``.

    The within-field coordinates are single letters (and ``z0``, ``z1``), so
    only case and spaces are forgiven: ``zs`` or ``c_`` stay other columns.
    """
    return re.sub(r"\s+", "", _PANDAS_COPY.sub("", name)).lower()


# The canonical names by their squashed form, for ``canonical_spelling``, with the
# short and former names of the repeats: ``bio_repeat`` or ``tech_rep`` would
# otherwise be an ordinary column, and its repeats pooled. The exact former name
# ``repeat`` is read as ``biological_repeat`` before any spelling is checked
# (``_former_repeat``); its near misses (``Repeat``, ``repeats``) are refused.
_CANONICAL_BY_SQUASHED = (
    {_squashed(name): name for name in (*MORPHOBRICKS_GROUP_COLUMNS, *MORPHOBRICKS_FOV_CANDIDATES)}
    | {
        _squashed(f"{prefix}{word}"): column
        for column, prefixes in (
            (REPEAT_COLUMN, ("bio", "biol", "biological")),
            (TECHNICAL_REPEAT_COLUMN, ("tec", "tech", "technical")),
        )
        for prefix in prefixes
        for word in ("repeat", "rep", "replicate")
    }
    | {"repeat": REPEAT_COLUMN}
)
# Other names of the within-field coordinates, by their squashed form: a plane,
# channel or frame column under any of them would be read as an ordinary column,
# so one file per plane would count as one field per plane.
_PLANE_SYNONYMS = {
    **dict.fromkeys(("plane", "zplane", "slice", "zslice", "zindex"), "z"),
    **dict.fromkeys(("channel", "cindex", "channelnumber"), "c"),
    **dict.fromkeys(("frame", "timepoint", "tindex", "timeframe"), "t"),
}
# The prefix CellProfiler gives every metadata column (``Metadata_Z``, ``Metadata_condition``).
_METADATA_PREFIX = re.compile(r"^metadata[\s_.-]*", re.IGNORECASE)


def canonical_spelling(column: object) -> str | None:
    """The MorphoBricks column or within-field coordinate that ``column`` misspells, or ``None``.

    A near miss of a shared column differs only in case, in spaces, hyphens,
    dots, camelCase or no separator for underscores, in a plural ``s``, in
    surrounding whitespace, in a byte-order mark that a spreadsheet wrote
    before the first header, or in the ``.1`` pandas gives a second column of
    the same name (``Biological_Repeat``, ``technical repeat``, ``technical.repeat``,
    ``technicalRepeat``, ``PositionID``, ``Conditions``, ``condition ``,
    ``\ufeffcondition``, ``Repeat.1``), or is a short or former name of a repeat
    (``bio_repeat``, ``biological replicate``, ``tec_repeat``, ``tech_rep``; ``Repeat``
    or ``repeats`` for the former ``repeat``). A near miss of a within-field
    coordinate (``z``, ``z0``, ``z1``, ``c``, ``t``) differs only in case or
    whitespace (``Z``, ``T ``), or is one of its other names (``plane``,
    ``slice``, ``channel``, ``frame``, ``timepoint``, ...). Either may carry
    CellProfiler's ``Metadata_`` prefix (``Metadata_Z``, ``Metadata_condition``).
    A typing error of a shared column (``biological_repat``, ``techincal_repeat``,
    ``conditon``, ``image_flie``) is a near miss too: one edit (a letter added, missing,
    changed, or two swapped), or two for the repeats' longer names. Such a column would
    otherwise be ignored: its groups pooled, or its planes counted as fields, without a word.
    The exact name is not a near miss.
    """
    if not isinstance(column, str):
        return None
    cleaned = column.replace("\ufeff", "").replace("\u200b", "")
    # CellProfiler's Metadata_ prefix is forgiven only before a name that is then a near miss.
    unprefixed = _METADATA_PREFIX.sub("", cleaned) or cleaned
    meant = _CANONICAL_BY_SQUASHED.get(_squashed(unprefixed))
    if meant is None and _plain_spelling(unprefixed) in WITHIN_FOV_METADATA_COLUMNS:
        meant = _plain_spelling(unprefixed)
    if meant is None:
        meant = _PLANE_SYNONYMS.get(_squashed(unprefixed))
    if meant is None:
        meant = typing_error_of(_squashed(unprefixed), _TYPO_TARGETS)
    return meant if meant is not None and meant != column else None


# The shared columns whose typing errors are caught (``biological_repat``, ``conditon``),
# by their squashed form; ``fov_id`` is too short to tell a typo from another name.
_TYPO_TARGETS = {
    _squashed(name): name
    for name in (*MORPHOBRICKS_GROUP_COLUMNS, *MORPHOBRICKS_FOV_CANDIDATES)
    if len(_squashed(name)) >= 8
}


# Names of a stage-position index inside a multi-position file (an .nd2 or .czi
# series, a scene, a site), by their squashed form, with or without ``Metadata_``.
_POSITION_NAMES = frozenset(
    _squashed(name)
    for name in ("series", "scene", "site", "position", "stage_position", "multipoint")
)


def position_like(column: object) -> bool:
    """Whether ``column`` names a stage position within a file (``series``, ``Metadata_Site``).

    The exact MorphoBricks identifiers (``position_id``) are not position-like:
    they are the field identifier itself.
    """
    if not isinstance(column, str) or column in MORPHOBRICKS_FOV_CANDIDATES:
        return False
    return _squashed(_METADATA_PREFIX.sub("", column) or column) in _POSITION_NAMES


def near_miss_advice(column: str, meant: str) -> str:
    """How to fix a near-miss spelling ``column`` of ``meant`` (``canonical_spelling``)."""
    if meant in WITHIN_FOV_METADATA_COLUMNS:
        return (
            f"Rename it to {meant!r} (the within-field coordinates are read under their "
            "lowercase names only), "
            "or, if it is not a plane, channel or time coordinate, give it another name."
        )
    if _PANDAS_COPY.search(column):
        base = _PANDAS_COPY.sub("", column)
        rename = f" (named {meant!r})" if base != meant else ""
        return (
            f"pandas names the second of two {base!r} columns {column!r}: keep one of them"
            f"{rename}, or name it in group_cols/fov_col to use it as it is."
        )
    return (
        f"Did you mean {meant!r}? Rename it to {meant!r}, or name it in group_cols/fov_col "
        "to use it as it is."
    )


def check_canonical_spellings(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    explicit_columns: Sequence[str],
) -> None:
    """Reject near-miss spellings such as Repeat, or Z, that would bypass defaults."""
    for table_name, table in _tables(objects, fovs):
        for column in table.columns:
            if column in explicit_columns:
                continue
            meant = canonical_spelling(column)
            if meant is not None:
                kind = (
                    "within-field coordinate"
                    if meant in WITHIN_FOV_METADATA_COLUMNS
                    else "MorphoBricks column"
                )
                raise ValueError(
                    f"{table_name} has column {column!r}, a non-canonical spelling of "
                    f"{kind} {meant!r}. {near_miss_advice(column, meant)}"
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

    The first candidate present is used, even when its values are missing
    (they are then refused), except ``position_id`` when it is empty in every
    row (``drop_empty_optional_columns`` has then removed it). A candidate
    present in only one linked table is an error, never a reason to fall back
    to the next candidate.
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
    """Resolve analysis groups using canonical MorphoBricks columns when implicit.

    Explicit ``group_cols`` must name each column once.
    """
    if group_cols is not None:
        names = column_names(group_cols, "group_cols")
        repeated = repeated_names(names)
        if repeated:
            raise ValueError(
                f"group_cols names {repeated[0]!r} more than once; name each group column once"
            )
        return names

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


def sorted_labels(values: Iterable[Any], first: Any = None) -> list:
    """The distinct ``values`` in sorted order (by type, then text, when types mix), with
    ``first`` (the control condition) first when present: an order of groups for
    sentences and figures that never follows the row order of a table."""
    distinct = list(dict.fromkeys(values))
    try:
        ordered = sorted(distinct)
    except TypeError:  # fallback: types that cannot be compared: sorted by type, then text
        ordered = sorted(distinct, key=lambda value: (type(value).__name__, str(value)))
    if first is None or first not in ordered:
        return ordered
    return [first, *(value for value in ordered if value != first)]


def unambiguous_index(table: pd.DataFrame) -> pd.DataFrame:
    """``table`` without the index level names that are also column names.

    ``set_index([...], drop=False)`` leaves ``condition`` both an index level
    and a column, which pandas refuses to group by as ambiguous. The labels are
    kept (results aligned with the caller's rows keep their index); only the
    clashing level names are dropped.
    """
    names = list(table.index.names)
    if not any(name is not None and name in table.columns for name in names):
        return table
    table = table.copy(deep=False)
    table.index = table.index.set_names(
        [None if name is not None and name in table.columns else name for name in names]
    )
    return table


def sorted_by_key(table: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """``table`` in the order of its ``key`` values (a stable sort), whatever their dtype.

    The canonical order that seeded permutations and splits start from. A
    categorical column is sorted by its values, not by the order of its
    categories, which is not data: reordering the categories (or adding
    unused ones) leaves the order unchanged.
    """

    def by_value(values: pd.Series) -> pd.Series:
        if isinstance(values.dtype, pd.CategoricalDtype):
            return values.astype(object)
        return values

    return table.sort_values(list(key), kind="stable", key=by_value)


__all__ = [
    "CONDITION_COLUMN",
    "EMPTY_MEANS_ABSENT",
    "MORPHOBRICKS_FOV_CANDIDATES",
    "MORPHOBRICKS_GROUP_COLUMNS",
    "MORPHOBRICKS_SHARED_COLUMNS",
    "OPTIONAL_GROUP_COLUMNS",
    "REPEAT_COLUMN",
    "TARGET_COLUMN",
    "TECHNICAL_REPEAT_COLUMN",
    "WITHIN_FOV_METADATA_COLUMNS",
    "attach_mapping",
    "canonical_spelling",
    "check_canonical_spellings",
    "column_has_values",
    "column_names",
    "drop_empty_optional_columns",
    "empty_optional_columns",
    "empty_values",
    "near_miss_advice",
    "position_like",
    "repeated_names",
    "resolve_fov_col",
    "resolve_group_cols",
    "resolve_identity_cols",
    "sorted_by_key",
    "sorted_labels",
    "unambiguous_index",
]
