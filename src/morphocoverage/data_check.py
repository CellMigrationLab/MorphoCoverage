"""Check a dataset before analysing it, and say in plain words what is wrong and how to fix it.

``check_tables`` runs every check of ``validate_input_tables`` without stopping
at the first problem, and a few more that do not stop the analysis but are
often mistakes. It returns a ``DataCheck`` whose ``summary()`` lists, for each
problem, what is wrong, where (the table, the column, how many rows and a few
example rows) and how to fix it, together with what was understood about the
tables. ``validate_input_tables`` and every workflow attach the same summary
to the error they raise, so the full list reaches the user whichever way the
data were loaded. The format itself is described in ``docs/data_format.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metadata import (
    EMPTY_MEANS_ABSENT,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    column_has_values,
    drop_empty_optional_columns,
)

ERROR = "error"
WARNING = "warning"
EXAMPLES = 3
FORMAT_DOC = "docs/data_format.md"


@dataclass(frozen=True)
class Problem:
    """One finding: ``level`` (error or warning), what is wrong, where, and how to fix it."""

    level: str
    title: str
    where: str
    fix: str


@dataclass(frozen=True)
class DataCheck:
    """Every problem found, and what was understood about the tables (see the module)."""

    problems: tuple[Problem, ...]
    understood: tuple[str, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> list[Problem]:
        """Problems that stop the analysis."""
        return [p for p in self.problems if p.level == ERROR]

    @property
    def warnings(self) -> list[Problem]:
        """Problems that do not stop the analysis but are probably mistakes."""
        return [p for p in self.problems if p.level == WARNING]

    @property
    def ok(self) -> bool:
        """Whether the tables can be analysed (warnings may remain)."""
        return not self.errors

    def summary(self) -> str:
        """What was understood, then every error and warning with where and how to fix it."""
        n_errors, n_warnings = len(self.errors), len(self.warnings)
        verdict = (
            "the tables can be analysed"
            if self.ok
            else "the tables cannot be analysed until the errors are fixed"
        )
        lines = [
            (
                f"MorphoCoverage data check: {_count(n_errors, 'error')}, "
                f"{_count(n_warnings, 'warning')}; {verdict}."
            )
        ]
        if self.understood:
            lines += ["", "What was understood"] + [f"  - {u}" for u in self.understood]
        for title, level in (
            ("Errors (must be fixed)", ERROR),
            ("Warnings (the analysis runs, but check these)", WARNING),
        ):
            found = [p for p in self.problems if p.level == level]
            if not found:
                continue
            lines += ["", title]
            for number, problem in enumerate(found, start=1):
                lines.append(f"  {number}. {problem.title}")
                lines.append(f"     Where: {problem.where}")
                lines.append(f"     Fix: {problem.fix}")
        lines += ["", f"The expected format, column by column, is in {FORMAT_DOC}."]
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        """Raise ``ValueError`` with the summary when any error was found."""
        if not self.ok:
            raise ValueError(self.summary())


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


class _Collector:
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

    def result(self) -> DataCheck:
        """The check so far."""
        return DataCheck(tuple(self.problems), tuple(self.understood))


def _tables(objects: pd.DataFrame, fovs: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
    """The linked tables, named as the messages name them."""
    named = [("observation table", objects)]
    if fovs is not None:
        named.append(("FOV table", fovs))
    return named


def _check_spellings(
    c: _Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, explicit: Sequence[str]
) -> None:
    """Near-miss spellings of the MorphoBricks columns (``Repeat`` for ``repeat``)."""
    canonical = {*MORPHOBRICKS_GROUP_COLUMNS, *MORPHOBRICKS_FOV_CANDIDATES}
    for name, table in _tables(objects, fovs):
        for column in table.columns:
            if not isinstance(column, str) or column in explicit:
                continue
            if column.lower() in canonical and column != column.lower():
                c.error(
                    f"Column {column!r} looks like the MorphoBricks column {column.lower()!r} "
                    "but is spelt differently, so it would be ignored.",
                    f"{name}, column {column!r}.",
                    f"Rename it to {column.lower()!r}, or name it in group_cols/fov_col to use it "
                    "as it is.",
                )


def _resolve_fov_col(
    c: _Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, fov_col: str | None
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
    c: _Collector,
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
    c: _Collector,
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
    c: _Collector, objects: pd.DataFrame, feature_cols: Sequence[str], taken: set[str]
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
        c.error(
            f"{_count(len(missing), 'feature column')} named in feature_cols "
            f"{'is' if len(missing) == 1 else 'are'} not in the observation table.",
            f"missing: {_names(missing)}.",
            "Check the spelling, or add the columns to the observation table.",
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


def _check_key_values(
    c: _Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, key: Sequence[str]
) -> None:
    """Every key column is complete in both tables; a copied column is checked at its source."""
    attached = set(objects.attrs.get("attached", []))
    for name, table in _tables(objects, fovs):
        for column in key:
            if column not in table.columns or (table is objects and column in attached):
                continue
            empty = table[column].isna() | (table[column].astype(str).str.strip() == "")
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
    c: _Collector, objects: pd.DataFrame, fovs: pd.DataFrame, key: Sequence[str]
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
            f"observation table, {_examples(objects, observed, key)}.",
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


def _check_control(c: _Collector, fovs: pd.DataFrame, control: Mapping[str, Any]) -> None:
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


def check_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None = None,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    control: Mapping[str, Any] | None = None,
) -> DataCheck:
    """Check the observation and FOV tables and report every problem (see the module).

    Arguments are those of ``validate_input_tables``; ``control`` (for
    example ``{"condition": "MOCK"}``) also checks that the control exists in
    every repeat. Never raises for bad data: the problems are in the result.
    """
    from .schema import _validate_input_tables  # the validator attaches this check to its errors

    c = _Collector()
    if not isinstance(objects, pd.DataFrame):
        c.error(
            "The observation table is not a table.",
            f"objects is a {type(objects).__name__}.",
            "Pass a pandas DataFrame, e.g. pd.read_csv('cells.csv').",
        )
        return c.result()
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        c.error(
            "The FOV table is not a table.",
            f"fovs is a {type(fovs).__name__}.",
            "Pass a pandas DataFrame with one row per acquired field.",
        )
        return c.result()
    if isinstance(feature_cols, str):
        c.error(
            "feature_cols is a single string, not a list of column names.",
            f"feature_cols={feature_cols!r}.",
            f"Pass a list: feature_cols=[{feature_cols!r}].",
        )
        feature_cols = [feature_cols]
    feature_cols = list(feature_cols)
    explicit = [*(group_cols or ()), *([fov_col] if fov_col is not None else [])]

    for column in EMPTY_MEANS_ABSENT:
        empty_everywhere = [
            t for _, t in _tables(objects, fovs) if column in t.columns
        ] and not any(column_has_values(t, column) for _, t in _tables(objects, fovs))
        if empty_everywhere and column not in explicit:
            c.understood.append(f"{column!r} is empty in every row, so it is read as absent.")
    objects, fovs = drop_empty_optional_columns(objects, fovs, explicit)
    _check_spellings(c, objects, fovs, explicit)

    c.understood.insert(
        0,
        f"{len(objects)} observations ({observation_unit or 'objects, unit not declared'})"
        + (f"; {len(fovs)} rows in the FOV table." if fovs is not None else "; no FOV table."),
    )
    if fovs is None:
        c.warning(
            "No FOV table was given.",
            "fovs=None.",
            "Pass the list of every acquired field, including fields with no observation; "
            "without it, empty fields are missing and every sampling answer is optimistic.",
        )
    if observation_unit is None:
        c.warning(
            "The observation unit is not declared.",
            "observation_unit=None.",
            "Say what one row is, e.g. observation_unit='cell', 'nucleus' or 'patch'; the report "
            "otherwise says 'object' and records that the unit was assumed.",
        )

    resolved = _resolve_fov_col(c, objects, fovs, fov_col)
    taken: set[str] = set()
    if resolved is not None:
        groups, identity = _resolve_groups(c, objects, fovs, group_cols, resolved)
        taken = {resolved, *identity}
        key = [*identity, resolved]
        c.understood.append(
            f"A field is identified by {resolved!r}"
            + (f" within {', '.join(identity)}." if identity else ".")
        )
        c.understood.append(
            "Analysis groups: "
            + (", ".join(groups) if groups else "none (all fields pooled)")
            + "."
        )
        objects = _attach(c, objects, fovs, resolved, identity)
        _check_key_values(c, objects, fovs, key)
        if fovs is not None:
            _check_fov_table(c, objects, fovs, key)
    _check_features(c, objects, feature_cols, taken)
    c.understood.append(
        f"{_count(len(feature_cols), 'feature column')}: {_names(feature_cols)}."
        if feature_cols
        else "No feature column."
    )
    if control is not None and fovs is not None:
        _check_control(c, fovs, control)
    if c.result().ok:
        # The remaining rules (planes, channels or frames counted as fields) run as the
        # validator runs them.
        try:
            _validate_input_tables(
                objects,
                feature_cols=feature_cols,
                fov_col=fov_col,
                fovs=fovs,
                group_cols=group_cols,
                observation_unit=observation_unit,
            )
        except (ValueError, TypeError) as exc:
            c.error(
                str(exc)[0].upper() + str(exc)[1:] + ".",
                "the within-field coordinates (z, c, t) and the field identifier.",
                "Give every plane, channel or frame of one stage position the same "
                "'position_id', and a z/c/t value that tells its files apart.",
            )
    return c.result()


__all__ = ["DataCheck", "Problem", "check_tables"]
