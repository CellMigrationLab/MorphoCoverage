"""Input traps (private): tables that validate but are probably not what the caller meant.

Rows a producer marked as not valid (``warn_invalid_rows``, a warning of the
validator, so the live session and every workflow see it), and the
``check_tables`` warnings about the layout of the tables: a column that looks
like an analysis group but is not one (``_check_undeclared_groups``), a well
that holds no observation in any field (``_check_unimaged_technical_repeats``),
a plate well named in two conditions of one repeat (``_check_well_names``),
and multi-position files that repeat a ``label_id`` (``_check_container_files``).
Each warns; none refuses, since each pattern can also be real.

Shared-private: ``warn_invalid_rows`` is used by ``schema``, the ``_check_*``
rules by ``data_check``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from ._data_problems import Collector, _examples, _names, _value
from ._wording import count
from .metadata import (
    CONDITION_COLUMN,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_SHARED_COLUMNS,
    REPEAT_COLUMN,
    TECHNICAL_REPEAT_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
)

# Files that usually hold several stage positions (series, scenes, points).
CONTAINER_EXTENSIONS = (".nd2", ".czi", ".lif", ".vsi", ".ims", ".oib", ".lsm")
# A plate well: a row letter (A to AF) and a column number, zero-padded or not.
_WELL = re.compile(r"(?:A[A-F]|[A-P])0*\d{1,2}", re.IGNORECASE)


class InvalidRowsWarning(UserWarning):
    """Observations a producer marked as not valid (CellPatcher's ``valid=False``)."""


def warn_invalid_rows(objects: pd.DataFrame, taken: Sequence[str], stacklevel: int) -> None:
    """Warn when a ``valid`` column (CellPatcher's) marks observations as not valid.

    CellPatcher keeps a patch it could not cut (``reason`` 'patch outside
    image') with ``valid=False``. Such rows are still observations to the
    analysis, counted and measured, so the caller is told; they are never
    dropped silently. ``taken`` are the columns with a role (key, features):
    a ``valid`` column among them is not read as a flag.
    """
    if "valid" not in objects.columns or "valid" in taken:
        return
    invalid = (
        objects["valid"]
        .map(
            lambda v: (
                (isinstance(v, (bool, np.bool_)) and not v)
                or (isinstance(v, str) and v.strip().casefold() == "false")
            )
        )
        .astype(bool)
    )
    if not invalid.any():
        return
    reasons = ""
    if "reason" in objects.columns:
        shown = [repr(r) for r in objects.loc[invalid, "reason"].dropna().unique()[:3]]
        reasons = f" (reason {', '.join(shown)})" if shown else ""
    warnings.warn(
        f"{count(int(invalid.sum()), 'observation')} have valid=False{reasons}, as CellPatcher "
        "marks a patch it could not cut, yet they are counted as observations and their "
        "features used. Drop them first (objects[objects['valid'] != False]) unless they are "
        "real measurements.",
        InvalidRowsWarning,
        stacklevel=stacklevel,
    )


def _text_column(values: pd.Series) -> bool:
    """Whether ``values`` holds labels as text (text, object or categorical of text)."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(object)
    elif values.dtype != object and not pd.api.types.is_string_dtype(values.dtype):
        return False
    present = values.dropna()
    return bool(len(present)) and bool(present.map(lambda v: isinstance(v, str)).all())


def _field_values(table: pd.DataFrame, key: list[str], column: str) -> pd.DataFrame | None:
    """One row per field with its ``column`` value, or ``None`` when it varies in a field."""
    try:
        pairs = table.loc[:, [*key, column]].astype(object).drop_duplicates()
    except TypeError:  # lists or arrays: not labels
        return None
    return None if pairs.duplicated(key).any() else pairs


def _check_undeclared_groups(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    key: Sequence[str],
    groups: Sequence[str],
    feature_cols: Sequence[str],
) -> None:
    """Text columns that split analysis groups into blocks of fields, undeclared.

    A column constant within each field that, within one analysis group,
    takes several values with at least one shared by several fields, looks
    like a group column the analysis pools: ``replicate``, ``well``,
    ``Bedingung``, CellProfiler's ``Metadata_Well``. It may be a field property
    too (a quality flag, a date), so this is a warning, once per column; the
    FOV table, which lists every field, is read first.
    """
    key = list(key)
    taken = {
        *key,
        *feature_cols,
        *MORPHOBRICKS_SHARED_COLUMNS,
        *MORPHOBRICKS_FOV_CANDIDATES,
        *WITHIN_FOV_METADATA_COLUMNS,
        "label_id",
    }
    groups = [g for g in groups if g in key]
    reported: set[Hashable] = set()
    tables = [("FOV table", fovs)] if fovs is not None else []
    for name, table in [*tables, ("observation table", objects)]:
        if any(k not in table.columns for k in key):
            continue
        for column in table.columns:
            if column in taken or column in reported or not _text_column(table[column]):
                continue
            pairs = _field_values(table, key, column)
            if pairs is None or pairs[column].nunique() < 2:
                continue
            sizes = pairs.groupby([*groups, column], dropna=False, sort=False).size()
            per_group = (
                sizes.groupby(level=list(range(len(groups))), dropna=False, sort=False)
                if groups
                else sizes.groupby(np.zeros(len(sizes)))
            )
            splits = (per_group.size() > 1) & (per_group.max() > 1)
            if not splits.any():
                continue
            reported.add(column)
            values = pairs[column].dropna().drop_duplicates().tolist()
            first = splits.index[splits][0]
            first = first if isinstance(first, tuple) else (first,)
            within = (
                "the fields of one analysis group ("
                + ", ".join(f"{g}={_value(v)}" for g, v in zip(groups, first, strict=True))
                + ")"
                if groups
                else "the pooled fields (no analysis group is declared)"
            )
            c.warning(
                f"Column {column!r} looks like a group column: it is constant within each field "
                f"and splits {within} into {count(int(per_group.size()[splits].iloc[0]), 'block')}"
                ", but it is not an analysis group, so those fields are pooled.",
                f"{name}, column {column!r}: {_names(values)}.",
                "If it names a group (a well, a plate, a repeat), declare it: rename it to "
                "'technical_repeat' (a well or dish within a repeat) or 'biological_repeat', or "
                "name it in group_cols. If it only describes the field (a quality flag, a date), "
                "nothing needs to change.",
            )


def _check_unimaged_technical_repeats(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, key: Sequence[str]
) -> None:
    """Wells of the FOV table with no observation in any of their fields.

    A FOV table copied from the plate map lists wells that were never imaged;
    their fields would count as acquired, empty fields: effort, and technical
    repeats in the overview.
    """
    if fovs is None or TECHNICAL_REPEAT_COLUMN not in key:
        return
    well = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN) if c in key] + [TECHNICAL_REPEAT_COLUMN]
    if any(k not in objects.columns or k not in fovs.columns for k in well):
        return
    try:
        seen = set(objects.loc[:, well].astype(object).itertuples(index=False, name=None))
        listed = fovs.loc[:, well].astype(object)
        unseen = pd.Series(
            [row not in seen for row in listed.itertuples(index=False, name=None)],
            index=fovs.index,
        )
    except TypeError:  # lists: reported with the field identity
        return
    if not unseen.any():
        return
    wells = listed.loc[unseen].drop_duplicates()
    first = wells.drop_duplicates(well).index
    c.warning(
        f"{count(len(wells), 'technical repeat')} of the FOV table "
        f"{'holds' if len(wells) == 1 else 'hold'} no observation in any field.",
        f"FOV table, {_examples(fovs, pd.Series(fovs.index.isin(first), index=fovs.index), well)}.",
        "A whole well or dish without one observation was probably never imaged (a FOV table "
        "copied from the plate map?) or failed; every listed field counts as effort, so leave "
        "out the wells that were not imaged and mark the failed ones.",
    )


def _check_well_names(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, identity: Sequence[str]
) -> None:
    """A plate well named in several conditions of one repeat.

    One well holds one condition, so ``B02`` in the control and in the drug of
    one repeat is two technical repeats to the analysis (it is identified with
    its condition), and probably a plate-map error. Read only when the
    technical repeats look like plate wells (a row letter and a column
    number, over at least two rows), so dishes ``d1``, ``d2`` that restart in
    each condition are not flagged.
    """
    if TECHNICAL_REPEAT_COLUMN not in identity or CONDITION_COLUMN not in identity:
        return
    table = fovs if fovs is not None else objects
    within = [REPEAT_COLUMN] if REPEAT_COLUMN in identity else []
    columns = [*within, TECHNICAL_REPEAT_COLUMN, CONDITION_COLUMN]
    if any(k not in table.columns for k in columns):
        return
    try:
        pairs = table.loc[:, columns].astype(object).drop_duplicates()
    except TypeError:
        return
    names = pairs[TECHNICAL_REPEAT_COLUMN]
    wells = names.map(lambda v: isinstance(v, str) and _WELL.fullmatch(v.strip()) is not None)
    rows = {v.strip()[0].casefold() for v in names[wells]}
    if not wells.all() or len(rows) < 2:
        return
    shared = pairs.duplicated([*within, TECHNICAL_REPEAT_COLUMN], keep=False)
    if not shared.any():
        return
    c.warning(
        "A plate well is named in several conditions of one repeat, so each is analysed as a "
        "technical repeat of its own.",
        f"{'FOV' if fovs is not None else 'observation'} table, "
        f"{_examples(pairs, shared, columns)}.",
        "One well holds one condition: check the plate map (a well typed under the wrong "
        "condition); if these are different plates or dishes, give them different names.",
    )


def _check_container_files(c: Collector, objects: pd.DataFrame, key: Sequence[str]) -> None:
    """Multi-position files (``.nd2``, ``.czi``) as fields whose ``label_id`` repeats.

    One such file per well holds several stage positions; with no
    ``position_id`` (and no ``series`` column, which the validator reads)
    each file is one field, and labels that restart in each position repeat.
    """
    fov_col = key[-1]
    if fov_col != "image_file" or "label_id" not in objects.columns:
        return
    names = objects[fov_col]
    container = names.map(
        lambda name: isinstance(name, str) and name.casefold().endswith(CONTAINER_EXTENSIONS)
    )
    if not container.any():
        return
    try:
        twice = objects.duplicated([*key, "label_id"], keep="first")
    except TypeError:
        return
    twice &= container & objects["label_id"].notna()
    if twice.any():
        c.warning(
            "Fields named by multi-position files repeat a label_id, so each file may hold "
            "several stage positions counted as one field.",
            f"observation table, {_examples(objects, twice, [*key, 'label_id'])}.",
            "A .nd2, .czi or .lif file often holds several positions (series, scenes); give each "
            "position its own 'position_id' (for example the file and the series together).",
        )
