"""The rules of ``data_check.check_tables``, one function per kind of problem (private).

Each rule reads the tables, records problems on a ``Collector`` in plain words
(what is wrong, where, how to fix it) and never raises for bad data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metadata import (
    EMPTY_MEANS_ABSENT,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    canonical_spelling,
    empty_values,
)

ERROR = "error"
WARNING = "warning"
EXAMPLES = 3


@dataclass(frozen=True)
class Problem:
    """One finding: ``level`` (error or warning), what is wrong, where, and how to fix it."""

    level: str
    title: str
    where: str
    fix: str


def _count(n: int, word: str) -> str:
    """``1 error``, ``2 errors``."""
    return f"{n} {word}{'s' if n != 1 else ''}"


def _value(value: Any) -> str:
    """A cell as a person reads it: text quoted, numbers plain, missing as empty."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "empty"
    return repr(value) if isinstance(value, str) else str(value)


def _examples(table: pd.DataFrame, rows: pd.Series, columns: Sequence[str]) -> str:
    """How many rows ``rows`` marks, and the values of ``columns`` in the first few."""
    marked = table.index[rows.to_numpy()]
    shown = []
    for label in marked[:EXAMPLES]:
        values = ", ".join(f"{c}={_value(table.at[label, c])}" for c in columns)
        shown.append(f"row {label!r} ({values})")
    more = f" and {len(marked) - EXAMPLES} more" if len(marked) > EXAMPLES else ""
    return f"{_count(len(marked), 'row')}: " + "; ".join(shown) + more


def _names(values: Sequence[Any]) -> str:
    """A short, quoted list of names."""
    shown = ", ".join(repr(v) for v in list(values)[:8])
    return shown + (f" and {len(values) - 8} more" if len(values) > 8 else "")


class Collector:
    """Accumulates problems and understood facts while the checks run."""

    def __init__(self) -> None:
        """Start with no finding."""
        self.problems: list[Problem] = []
        self.understood: list[str] = []

    def error(self, title: str, where: str, fix: str) -> None:
        """Record a problem that stops the analysis."""
        self.problems.append(Problem(ERROR, title, where, fix))

    def warning(self, title: str, where: str, fix: str) -> None:
        """Record a problem that does not stop the analysis."""
        self.problems.append(Problem(WARNING, title, where, fix))

    def has_errors(self) -> bool:
        """Whether an error was recorded."""
        return any(p.level == ERROR for p in self.problems)


def _tables(objects: pd.DataFrame, fovs: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
    """The linked tables, named as the messages name them."""
    named = [("observation table", objects)]
    if fovs is not None:
        named.append(("FOV table", fovs))
    return named


def _check_spellings(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, explicit: Sequence[str]
) -> None:
    """Near-miss spellings of the MorphoBricks columns (``Repeat`` for ``repeat``)."""
    for name, table in _tables(objects, fovs):
        for column in table.columns:
            if column in explicit:
                continue
            meant = canonical_spelling(column)
            if meant is not None:
                c.error(
                    f"Column {column!r} looks like the MorphoBricks column {meant!r} "
                    "but is spelt differently, so it would be ignored.",
                    f"{name}, column {column!r}.",
                    f"Rename it to {meant!r}, or name it in group_cols/fov_col to use it as it is.",
                )


def _resolve_fov_col(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, fov_col: str | None
) -> str | None:
    """The field identifier, or ``None`` after recording why none can be used."""
    if fov_col is not None:
        missing = [name for name, t in _tables(objects, fovs) if fov_col not in t.columns]
        if missing:
            c.error(
                f"The field identifier fov_col={fov_col!r} is not a column of the "
                f"{' and the '.join(missing)}.",
                f"columns present: {_names(list(objects.columns))}.",
                "Pass the name of the column that identifies each acquired field.",
            )
            return None
        return fov_col
    for candidate in MORPHOBRICKS_FOV_CANDIDATES:
        in_objects = candidate in objects.columns
        in_fovs = fovs is not None and candidate in fovs.columns
        if not in_objects and not in_fovs:
            continue
        if fovs is not None and in_objects != in_fovs:
            present, absent = (
                ("observation table", "FOV table")
                if in_objects
                else ("FOV table", "observation table")
            )
            c.error(
                f"The field identifier {candidate!r} is in the {present} but not in the {absent}.",
                f"{present}, column {candidate!r}.",
                f"Add {candidate!r} to the {absent} so each observation can be matched to its "
                "field.",
            )
            return None
        return candidate
    c.error(
        "No column identifies the acquired fields.",
        f"observation table columns: {_names(list(objects.columns))}.",
        "Add 'position_id' (one value per stage position), or 'image_file' when one file is "
        "one field, or name your column with fov_col=...",
    )
    return None


def _resolve_groups(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Sequence[str] | None,
    fov_col: str,
) -> tuple[list[str], list[str]]:
    """The analysis groups and the identity columns, after checking both tables hold them."""
    if group_cols is not None:
        groups = list(group_cols)
    else:
        groups = [
            g
            for g in MORPHOBRICKS_GROUP_COLUMNS
            if g in objects.columns or (fovs is not None and g in fovs.columns)
        ]
    identity = list(groups) + [
        g
        for g in MORPHOBRICKS_GROUP_COLUMNS
        if g not in groups
        and g != fov_col
        and (g in objects.columns or (fovs is not None and g in fovs.columns))
    ]
    for column in identity:
        in_objects = column in objects.columns
        in_fovs = fovs is not None and column in fovs.columns
        if not in_objects and not in_fovs:
            c.error(
                f"The group column {column!r} is in neither table.",
                f"group_cols={list(group_cols or [])}.",
                f"Add {column!r} to the FOV table, or leave it out of group_cols.",
            )
        elif (
            in_objects and fovs is not None and not in_fovs and column in MORPHOBRICKS_GROUP_COLUMNS
        ):
            c.error(
                f"{column!r} is in the observation table but not in the FOV table.",
                f"observation table, column {column!r}.",
                f"Add {column!r} to the FOV table too, so fields without observations keep their "
                "group.",
            )
    return groups, identity


def _attach(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    fov_col: str,
    identity: Sequence[str],
) -> pd.DataFrame:
    """Copy FOV-table-only group columns onto the observations, when unambiguous.

    The copied columns are marked in ``enriched.attrs["attached"]``, so a missing
    value is reported once, in the FOV table it came from.
    """
    if fovs is None or fov_col not in fovs.columns:
        return objects
    enriched = objects.copy()
    enriched.attrs["attached"] = []
    copied = [col for col in identity if col not in objects.columns and col in fovs.columns]
    unknown = ~objects[fov_col].isin(fovs[fov_col]) if fov_col in objects.columns else None
    if copied and unknown is not None and unknown.any():
        # Their group values cannot be copied, so the field check below would not see them.
        c.error(
            "Some observations belong to fields that are not in the FOV table.",
            f"observation table, {_examples(objects, unknown, [fov_col])}.",
            f"Add those fields to the FOV table, or check that {fov_col!r} is spelt the same way "
            "in both tables.",
        )
    for column in identity:
        if column in objects.columns or column not in fovs.columns:
            continue
        mapping = fovs.loc[:, [fov_col, column]].drop_duplicates()
        reused = mapping[fov_col].duplicated(keep=False)
        if reused.any():
            c.error(
                f"{column!r} is only in the FOV table, and the same {fov_col!r} has several "
                f"values of {column!r}, so observations cannot be matched to a {column!r}.",
                f"FOV table, {_examples(mapping, reused, [fov_col, column])}.",
                f"Add {column!r} to the observation table, or make {fov_col!r} unique across "
                f"{column!r} values.",
            )
            continue
        attached = [*enriched.attrs["attached"], column]
        enriched = enriched.merge(mapping, on=fov_col, how="left").set_axis(enriched.index)
        enriched.attrs["attached"] = attached
    return enriched


def _check_features(
    c: Collector, objects: pd.DataFrame, feature_cols: Sequence[str], taken: set[str]
) -> None:
    """Feature columns: named, distinct, present, numeric, finite; missing values reported."""
    if not feature_cols:
        c.error(
            "No feature column was named.",
            "feature_cols is empty.",
            "Name the numeric phenotype columns, e.g. feature_cols=['area'] (or Setup.readout).",
        )
        return
    duplicates = sorted({f for f in feature_cols if list(feature_cols).count(f) > 1})
    if duplicates:
        c.error(
            "A feature column is named twice.",
            f"feature_cols: {_names(duplicates)}.",
            "Name each feature once.",
        )
    overlap = sorted(set(feature_cols) & taken)
    if overlap:
        c.error(
            "A column is used both as a feature and as a group or field identifier.",
            f"columns {_names(overlap)}.",
            "Give each column one role: remove it from feature_cols or from group_cols/fov_col.",
        )
    missing = [f for f in dict.fromkeys(feature_cols) if f not in objects.columns]
    if missing:
        close = {f: _close_column(f, objects.columns) for f in missing}
        hints = [f"{found!r} for {f!r}" for f, found in close.items() if found is not None]
        c.error(
            f"{_count(len(missing), 'feature column')} named in feature_cols "
            f"{'is' if len(missing) == 1 else 'are'} not in the observation table.",
            f"missing: {_names(missing)}"
            + (f"; the table has {', '.join(hints)}" if hints else "")
            + ".",
            "Check the spelling (spaces and case count), or add the columns to the observation "
            "table.",
        )
    for feature in dict.fromkeys(feature_cols):
        if feature not in objects.columns:
            continue
        values = objects[feature]
        if not pd.api.types.is_numeric_dtype(values):
            as_numbers = pd.to_numeric(values, errors="coerce")
            bad = as_numbers.isna() & values.notna()
            c.error(
                f"Feature {feature!r} is not numeric.",
                "observation table, "
                + (_examples(objects, bad, [feature]) if bad.any() else "column type")
                + ".",
                "Convert the column to numbers (a decimal point, not a comma; empty cells for "
                "missing values).",
            )
            continue
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        infinite = pd.Series(np.isinf(numbers), index=objects.index)
        if infinite.any():
            c.error(
                f"Feature {feature!r} contains infinite values.",
                f"observation table, {_examples(objects, infinite, [feature])}.",
                "Replace infinite values by missing values (NaN) or fix their computation, for "
                "example a division by zero.",
            )
        nan = pd.Series(np.isnan(numbers), index=objects.index)
        if nan.any():
            c.warning(
                f"Feature {feature!r} is missing in {100 * nan.mean():.1f}% of observations.",
                f"observation table, {_examples(objects, nan, [feature])}.",
                "Missing values are kept and never imputed; check they are expected (for "
                "example, a measurement that failed).",
            )


def _close_column(name: str, columns: Sequence[Any]) -> Any | None:
    """A column that differs from ``name`` only in case or surrounding whitespace."""
    target = name.strip().casefold()
    for column in columns:
        if isinstance(column, str) and column != name and column.strip().casefold() == target:
            return column
    return None


def _loose(value: Any) -> str:
    """A value compared loosely: without surrounding spaces or case, a number by its value."""
    text = str(value).strip().casefold()
    try:
        return repr(float(text))
    except ValueError:
        return text


def _near_values(objects: pd.DataFrame, fovs: pd.DataFrame, key: Sequence[str]) -> str:
    """Why observation keys miss the FOV table, when it is spaces, case or number against text."""
    hints = []
    for column in key:
        seen = pd.Series(objects[column].dropna().unique())
        listed = pd.Series(fovs[column].dropna().unique())
        absent = seen[~seen.isin(listed)]
        if absent.empty:
            continue
        by_text = {_loose(v): v for v in listed}
        pairs = [(v, by_text[_loose(v)]) for v in absent if _loose(v) in by_text][:EXAMPLES]
        if pairs:
            shown = "; ".join(f"{_value(a)} against {_value(b)}" for a, b in pairs)
            mixed = any(isinstance(a, str) != isinstance(b, str) for a, b in pairs)
            why = (
                "is text in one table and a number in the other"
                if mixed
                else "differs only in spaces or case"
            )
            hints.append(f"'{column}' {why} ({shown})")
    return (" In these rows " + "; ".join(hints) + ".") if hints else ""


def _check_copies(c: Collector, objects: pd.DataFrame) -> None:
    """Rows that repeat another row exactly, or a ``label_id`` twice in one field."""
    copies = objects.duplicated(keep="first")
    if copies.any():
        c.warning(
            f"{_count(int(copies.sum()), 'observation')} {'is' if copies.sum() == 1 else 'are'} "
            "an exact copy of another row.",
            f"observation table, {_examples(objects, copies, list(objects.columns[:4]))}.",
            "A file read or appended twice counts every cell twice; remove the copies, or keep "
            "them if identical rows are real.",
        )
        return
    if "label_id" not in objects.columns or c.has_errors():
        # A misspelt group column already breaks field identity; labels would repeat.
        return
    field = [
        col
        for col in (*MORPHOBRICKS_GROUP_COLUMNS, *MORPHOBRICKS_FOV_CANDIDATES)
        if col in objects.columns
    ]
    twice = objects.duplicated([*field, "label_id"], keep="first") & objects["label_id"].notna()
    if field and twice.any():
        c.warning(
            f"{_count(int(twice.sum()), 'label_id')} {'appears' if twice.sum() == 1 else 'appear'} "
            "twice in the same field.",
            f"observation table, {_examples(objects, twice, [*field, 'label_id'])}.",
            "A label names one segmented object of a field; check for a file appended twice or "
            "a field identifier shared by two images.",
        )


def _check_key_values(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, key: Sequence[str]
) -> None:
    """Every key column is complete in both tables; a copied column is checked at its source."""
    attached = set(objects.attrs.get("attached", []))
    for name, table in _tables(objects, fovs):
        for column in key:
            if column not in table.columns or (table is objects and column in attached):
                continue
            empty = empty_values(table[column])
            if not empty.any():
                continue
            role = "field identifier" if column == key[-1] else "group column"
            fix = f"Fill {column!r} in every row of the {name}."
            if column in EMPTY_MEANS_ABSENT:
                fix += " If it does not apply to this dataset, leave it empty in every row instead."
            c.error(
                f"The {role} {column!r} is empty in some rows, so their field or group is unknown.",
                f"{name}, {_examples(table, empty, [column])}.",
                fix,
            )


def _check_fov_table(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame, key: Sequence[str]
) -> None:
    """One row per field, and every observation's field listed (rows with a complete key)."""
    key = [k for k in key if k in objects.columns and k in fovs.columns]
    if not key:
        return
    objects = objects[objects[key].notna().all(axis=1)]
    fovs = fovs[fovs[key].notna().all(axis=1)]
    repeated = fovs.duplicated(key, keep=False)
    if repeated.any():
        c.error(
            "The FOV table lists a field more than once.",
            f"FOV table, {_examples(fovs, repeated, key)}.",
            "Keep exactly one row per acquired field (identified by " + ", ".join(key) + ").",
        )
    listed = set(zip(*(fovs[k].tolist() for k in key)))
    observed = pd.Series(
        [row not in listed for row in zip(*(objects[k].tolist() for k in key))],
        index=objects.index,
    )
    if observed.any():
        c.error(
            "Some observations belong to fields that are not in the FOV table.",
            f"observation table, {_examples(objects, observed, key)}."
            + _near_values(objects[observed], fovs, key),
            "Add those fields to the FOV table, or check that condition, repeat and the field "
            "identifier are spelt the same way in both tables.",
        )
    seen = set(zip(*(objects[k].tolist() for k in key)))
    empty = pd.Series(
        [row not in seen for row in zip(*(fovs[k].tolist() for k in key))], index=fovs.index
    )
    if empty.any():
        c.warning(
            f"{_count(int(empty.sum()), 'field')} of the FOV table "
            f"{'holds' if int(empty.sum()) == 1 else 'hold'} no observation.",
            f"FOV table, {_examples(fovs, empty, key)}.",
            "An empty field is a real result and counts as effort; if an image failed "
            "(unreadable, failed segmentation), mark it instead of listing it as empty.",
        )


def _check_control(c: Collector, fovs: pd.DataFrame, control: Mapping[str, Any]) -> None:
    """The declared control exists, and in every repeat."""
    missing = [k for k in control if k not in fovs.columns]
    if missing:
        c.error(
            "The control names a column that is not in the FOV table.",
            f"control={dict(control)}, missing: {_names(missing)}.",
            "Name the control by FOV-table columns, e.g. control={'condition': 'MOCK'}.",
        )
        return
    is_control = np.ones(len(fovs), dtype=bool)
    for column, value in control.items():
        is_control &= (fovs[column] == value).to_numpy()
    if not is_control.any():
        values = fovs[next(iter(control))].drop_duplicates().tolist()
        c.error(
            f"No field matches the control {dict(control)}.",
            f"values present: {_names(values)}.",
            "Check the spelling of the control condition.",
        )
        return
    if REPEAT_COLUMN in fovs.columns:
        with_control = set(fovs.loc[is_control, REPEAT_COLUMN])
        lacking = [r for r in fovs[REPEAT_COLUMN].drop_duplicates() if r not in with_control]
        if lacking:
            c.warning(
                f"{_count(len(lacking), 'repeat')} {'has' if len(lacking) == 1 else 'have'} no "
                f"control field.",
                f"repeats without {dict(control)}: {_names(lacking)}.",
                "Normalising to the control and comparing conditions work within a repeat; add "
                "the control to those repeats or leave them out.",
            )
