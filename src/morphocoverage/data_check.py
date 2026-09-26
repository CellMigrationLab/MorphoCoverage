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

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ._data_rules import (
    ERROR,
    WARNING,
    Collector,
    Problem,
    _attach,
    _check_control,
    _check_copies,
    _check_features,
    _check_fov_table,
    _check_key_values,
    _check_spellings,
    _count,
    _names,
    _resolve_fov_col,
    _resolve_groups,
    _tables,
)
from .metadata import (
    EMPTY_MEANS_ABSENT,
    column_has_values,
    column_names,
    drop_empty_optional_columns,
)
from .schema import ValidatedInput, _validate_input_tables, normalize_observation_unit

FORMAT_DOC = "docs/data_format.md"


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
    c = Collector()
    if not isinstance(objects, pd.DataFrame):
        c.error(
            "The observation table is not a table.",
            f"objects is a {type(objects).__name__}.",
            "Pass a pandas DataFrame, e.g. pd.read_csv('cells.csv').",
        )
        return DataCheck(tuple(c.problems), tuple(c.understood))
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        c.error(
            "The FOV table is not a table.",
            f"fovs is a {type(fovs).__name__}.",
            "Pass a pandas DataFrame with one row per acquired field.",
        )
        return DataCheck(tuple(c.problems), tuple(c.understood))
    for name, table in (("observation table", objects), ("FOV table", fovs)):
        repeated = [] if table is None else list(table.columns[table.columns.duplicated()].unique())
        if repeated:
            c.error(
                f"The {name} has more than one column with the same name.",
                f"{name}, columns {_names(repeated)}.",
                "Give every column its own name (a table joined side by side, e.g. with "
                "pd.concat(axis=1), can repeat one).",
            )
    if c.has_errors():
        # Every other rule reads columns by name, which is ambiguous here.
        return DataCheck(tuple(c.problems), tuple(c.understood))
    # Read once (a generator is consumed by reading); None when absent or already reported.
    features = _column_list(c, feature_cols, "feature_cols")
    features_readable = features is not None or feature_cols is None
    feature_cols = list(features or ())
    group_cols = _column_list(c, group_cols, "group_cols")
    unit_given = observation_unit is not None
    if unit_given:
        try:
            normalize_observation_unit(observation_unit)
        except ValueError:
            c.error(
                "The observation unit is not a word.",
                f"observation_unit={observation_unit!r}.",
                "Say what one row is in words, e.g. observation_unit='cell', 'nucleus' or 'patch'.",
            )
            observation_unit = None
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
            "without it, empty fields are missing and every sampling answer is optimistic. "
            "The workflows (report, optimise, benchmark, plots) need it; "
            "schema.fov_table_from_observations rebuilds one, without the empty fields.",
        )
    if objects.empty:
        c.warning(
            "The observation table has no rows.",
            "0 observations.",
            "Nothing was measured, so no feature can be analysed; check that the right file was "
            "read (a live session may start empty).",
        )
    if not unit_given:
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
        # After the group columns are attached, so the same label in two wells is not a copy.
        _check_copies(c, objects, key)
        _check_key_values(c, objects, fovs, key)
        if fovs is not None:
            _check_fov_table(c, objects, fovs, key)
    if features_readable:
        _check_features(c, objects, feature_cols, taken)
    c.understood.append(
        f"{_count(len(feature_cols), 'feature column')}: {_names(feature_cols)}."
        if feature_cols
        else "No feature column."
    )
    if control is not None:
        # Without a FOV table, the fields seen in the observations are all that is known.
        _check_control(c, objects if fovs is None else fovs, control, fovs is None)
    if not c.has_errors():
        # The remaining rules (planes, channels or frames counted as fields) run as the
        # validator runs them; its warnings become warnings of this check.
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _validate_input_tables(
                    objects,
                    feature_cols=feature_cols,
                    fov_col=fov_col,
                    fovs=fovs,
                    group_cols=group_cols,
                    observation_unit=observation_unit,
                )
            for warning in caught:
                if not issubclass(warning.category, UserWarning):
                    warnings.warn(warning.message, warning.category, stacklevel=2)
                    continue
                # The validator's only warning: planes that may be counted as fields.
                message = str(warning.message)
                c.warning(
                    message[0].upper() + message[1:],
                    "the within-field coordinates (z, c, t) and the field identifier.",
                    "Give every plane, channel or frame of one stage position the same "
                    "'position_id'; if each row really is one stage position, pass fov_col "
                    "explicitly.",
                )
        except (ValueError, TypeError) as exc:
            message = str(exc).split("\n\n")[0].rstrip(".")
            planes = any(w in message for w in ("plane", "z/c/t", "channel"))
            c.error(
                message[0].upper() + message[1:] + ".",
                "the within-field coordinates (z, c, t) and the field identifier."
                if planes
                else "found by validate_input_tables.",
                "Give every plane, channel or frame of one stage position the same "
                "'position_id', and a z/c/t value that tells its files apart."
                if planes
                else "Correct what the message names; every rule is in docs/data_format.md.",
            )
    return DataCheck(tuple(c.problems), tuple(c.understood))


def _column_list(c: Collector, columns: object, name: str) -> list | None:
    """``columns`` as a list of names, read once, or ``None`` (not given, or unreadable).

    A single string is reported and then read as one name, so the other rules
    still run; anything else that is not a collection of names is reported.
    """
    if isinstance(columns, str):
        c.error(
            f"{name} is a single string, not a list of column names.",
            f"{name}={columns!r}.",
            f"Pass a list: {name}=[{columns!r}].",
        )
        return [columns]
    try:
        names = column_names(columns, name)
    except TypeError as error:
        c.error(
            f"{name} is not a list of column names.",
            f"{name}={columns!r}: {error}.",
            f"Pass a list of column names, e.g. {name}=['area'].",
        )
        return None
    return None if names is None else list(names)


def validate_input_tables(
    objects: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    fovs: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> ValidatedInput:
    """Validate linked observation/FOV tables using MorphoBricks conventions.

    See ``schema._validate_input_tables`` for the rules. When the tables are refused,
    the error (same type, first line unchanged) carries the full
    ``check_tables`` report: every problem, where, and how to fix it.
    """
    arguments = {
        # An iterator is read once here, so that the report sees the same names.
        "feature_cols": tuple(feature_cols) if isinstance(feature_cols, Iterator) else feature_cols,
        "fov_col": fov_col,
        "fovs": fovs,
        "group_cols": tuple(group_cols) if isinstance(group_cols, Iterator) else group_cols,
        "observation_unit": observation_unit,
    }
    try:
        return _validate_input_tables(objects, **arguments)
    except (ValueError, TypeError) as exc:
        try:
            report = check_tables(objects, **arguments).summary()
        except Exception:  # noqa: BLE001 -- the report is a courtesy; never hide the real error
            raise exc from None
        raise type(exc)(f"{exc}\n\n{report}") from exc


__all__ = ["DataCheck", "Problem", "check_tables", "validate_input_tables"]
