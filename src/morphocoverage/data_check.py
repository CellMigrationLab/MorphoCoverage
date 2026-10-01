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
    _check_label_variants,
    _check_spellings,
    _examples,
    _names,
    _resolve_fov_col,
    _resolve_groups,
)
from ._former_repeat import FORMER_REPEAT_COLUMN, Reader, one_warning
from ._input_traps import (
    InvalidRowsWarning,
    _check_container_files,
    _check_undeclared_groups,
    _check_unimaged_technical_repeats,
    _check_well_names,
)
from ._style import CAUTION, PROBLEM, Status, check_detail, headline, wrap
from ._wording import count
from .metadata import (
    column_names,
    drop_empty_optional_columns,
    empty_optional_columns,
)
from .schema import (
    MissingFovTableWarning,
    ValidatedInput,
    _validate_input_tables,
    normalize_observation_unit,
    unhashable_values,
)

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

    @property
    def status(self) -> Status:
        """❌ with an error, ⚠️ with a warning, ✅ with neither."""
        if self.errors:
            return Status.PROBLEM
        return Status.CAUTION if self.warnings else Status.OK

    def summary(self, detail: str = "short") -> str:
        """What was understood, then every error and warning and how to fix it.

        ``"short"`` (the default) gives each problem's title and fix; ``"full"``
        also where it is (the table, the column, a few example rows). Errors
        raised for refused tables carry the full form.
        """
        short = check_detail(detail) == "short"
        n_errors, n_warnings = len(self.errors), len(self.warnings)
        verdict = (
            "the tables can be analysed"
            if self.ok
            else "the tables cannot be analysed until the errors are fixed"
        )
        tally = f"{count(n_errors, 'error')}, {count(n_warnings, 'warning')}; {verdict}"
        lines = [wrap(headline(self.status, "MorphoCoverage data check", tally), 0, hang=3)]
        if self.understood:
            lines += ["", "What was understood"]
            lines += [wrap(f"- {u}", 2, hang=4) for u in self.understood]
        for title, level, marker in (
            ("Errors (must be fixed)", ERROR, PROBLEM),
            ("Warnings (the analysis runs, but check these)", WARNING, CAUTION),
        ):
            found = [p for p in self.problems if p.level == level]
            if not found:
                continue
            lines += ["", title]
            for number, problem in enumerate(found, start=1):
                lines.append(wrap(f"{marker} {number}. {problem.title}", 2, hang=7))
                if not short:
                    lines.append(wrap(f"Where: {problem.where}", 7))
                lines.append(wrap(f"Fix: {problem.fix}", 7))
        closing = f"The expected format, column by column, is in {FORMAT_DOC}."
        if short and self.problems:
            closing = f"Where each problem is, with example rows: summary(detail='full'). {closing}"
        lines += ["", wrap(closing)]
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        """Raise ``ValueError`` with the summary when any error was found."""
        if not self.ok:
            raise ValueError(self.summary(detail="full"))


@one_warning
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

    Arguments are those of ``validate_input_tables`` (``fovs`` may be passed
    by position here); ``control`` (for example ``{"condition": "MOCK"}``)
    also checks that the control exists in every repeat. Never raises for bad
    data: the problems are in the result. Beyond the validator's rules (and
    its warnings) it warns about what is often a mistake: no FOV table, no
    declared unit, exact copies of rows, a ``label_id`` twice in one field,
    fields without observations, wells without an observation in any field,
    missing feature values, group labels that differ only in spaces, case,
    zero padding or number against text, a plate well named in two
    conditions of one repeat, an undeclared text column that splits analysis
    groups into blocks of fields (a well or a plate under another name),
    multi-position files that repeat a ``label_id``, and repeats without a
    control. A table, ``group_cols``, ``fov_col`` or ``control`` that uses the
    former name ``repeat`` is read as ``biological_repeat``, with a
    ``DeprecationWarning``; a table with both, different, is an error.
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
    try:
        objects, fovs, group_cols, fov_col, control = _read_former_repeat(
            objects, fovs, group_cols, fov_col, control
        )
    except ValueError as error:
        c.error(
            f"The former name {FORMER_REPEAT_COLUMN!r} and the biological repeat's column "
            "hold different values.",
            f"{error}.",
            "Keep only 'biological_repeat', the biological repeat, and rename or remove the "
            f"{FORMER_REPEAT_COLUMN!r} column (or control key).",
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
    fov_col_usable = fov_col is None or isinstance(fov_col, str)
    if not fov_col_usable:
        c.error(
            "fov_col is not a column name.",
            f"fov_col={fov_col!r}.",
            "Name the one column that identifies each acquired field, e.g. fov_col='position_id'.",
        )
    explicit = [*(group_cols or ()), *([fov_col] if fov_col_usable and fov_col else [])]

    _understand_empty_columns(c, objects, fovs, explicit)
    objects, fovs = drop_empty_optional_columns(objects, fovs, explicit)
    _check_spellings(c, objects, fovs, explicit)

    unit = (
        normalize_observation_unit(observation_unit)
        if observation_unit is not None
        else "objects, unit not declared"
    )
    c.understood.insert(
        0,
        f"{count(len(objects), 'observation')} ({unit})"
        + (
            f"; {count(len(fovs), 'row')} in the FOV table."
            if fovs is not None
            else "; no FOV table."
        ),
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

    resolved = _resolve_fov_col(c, objects, fovs, fov_col) if fov_col_usable else None
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
        if _identity_hashable(c, objects, fovs, key):
            objects = _attach(c, objects, fovs, resolved, identity)
            # After the group columns are attached, so the same label in two wells is not a copy.
            _check_copies(c, objects, key, feature_cols)
            _check_key_values(c, objects, fovs, key)
            _check_label_variants(c, objects, fovs, identity)
            _check_well_names(c, objects, fovs, identity)
            _check_undeclared_groups(c, objects, fovs, key, groups, feature_cols)
            _check_container_files(c, objects, key)
            if fovs is not None:
                _check_fov_table(c, objects, fovs, key)
                _check_unimaged_technical_repeats(c, objects, fovs, key)
    if features_readable:
        _check_features(c, objects, feature_cols, taken)
    c.understood.append(
        f"{count(len(feature_cols), 'feature column')}: {_names(feature_cols)}."
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
                if issubclass(warning.category, MissingFovTableWarning):
                    continue  # said above in this report's words ("No FOV table was given.")
                if not issubclass(warning.category, UserWarning):
                    warnings.warn(warning.message, warning.category, stacklevel=2)
                    continue
                message = str(warning.message)
                if issubclass(warning.category, InvalidRowsWarning):
                    c.warning(
                        message[0].upper() + message[1:],
                        "observation table, column 'valid'.",
                        "Keep only the valid rows, or rename the column if it means something "
                        "else.",
                    )
                    continue
                # The validator's other warnings: planes that may be counted as fields.
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
            within = "planes_as_features" in message  # planes of one field as observations
            c.error(
                message[0].upper() + message[1:] + ".",
                "the within-field coordinates (z, c, t) and the field identifier."
                if planes
                else "found by validate_input_tables.",
                "Keep one row per object: pivot the planes with planes_as_features, or keep one "
                "plane; declare observation_unit='slice' if each row measures a whole plane."
                if within
                else "Give every plane, channel or frame of one stage position the same "
                "'position_id', and a z/c/t value that tells its files apart."
                if planes
                else "Correct what the message names; every rule is in docs/data_format.md.",
            )
    return DataCheck(tuple(c.problems), tuple(c.understood))


def _read_former_repeat(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Any,
    fov_col: Any,
    control: Any,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Any, Any, Any]:
    """The arguments with the former name ``repeat`` read as ``biological_repeat``, warning
    once (``_former_repeat``); ``ValueError`` when a table or ``control`` holds both, different."""
    former = Reader()
    objects, fovs = former.table(objects, "objects"), former.table(fovs, "fovs")
    group_cols = former.columns(group_cols, "group_cols")
    fov_col, control = former.column(fov_col, "fov_col"), former.mapping(control, "control")
    former.warn()
    return objects, fovs, group_cols, fov_col, control


def _understand_empty_columns(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, explicit: Sequence[str]
) -> None:
    """Say which optional shared columns are read as absent, and where (as the validator)."""
    for column, where in empty_optional_columns(objects, fovs, explicit).items():
        in_both = fovs is not None and column in objects.columns and column in fovs.columns
        if len(where) == 2 or not in_both:
            c.understood.append(f"{column!r} is empty in every row, so it is read as absent.")
        elif where == ["objects"]:
            c.understood.append(
                f"{column!r} is empty in every row of the observation table, so the FOV "
                "table's values are used."
            )
        else:
            c.understood.append(
                f"{column!r} is empty in every row of the FOV table, so it is read as absent there."
            )


def _identity_hashable(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, key: Sequence[str]
) -> bool:
    """Whether the identity columns hold labels; lists or arrays are reported, and the
    rules that match fields are then skipped, since they cannot compare such values."""
    usable = True
    for name, table in (("observation table", objects), ("FOV table", fovs)):
        if table is None:
            continue
        for column in key:
            if column not in table.columns:
                continue
            bad = unhashable_values(table[column])
            if bad.any():
                usable = False
                c.error(
                    f"{column!r} holds values such as lists, which cannot identify a field or "
                    "group.",
                    f"{name}, {_examples(table, bad, [column])}.",
                    "Store one text or number per row (for example the file name, not a list "
                    "of files).",
                )
    return usable


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


@one_warning
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

    ``objects`` holds one row per phenotype observation (the name is kept for
    backwards compatibility; rows may be cells, nuclei, patches, slices, ...),
    ``fovs`` one row per acquired field, fields without observations included.
    The rules, all described column by column in ``docs/data_format.md``:

    - ``feature_cols`` are named explicitly, once each, in an ordered sequence
      (not a set), and hold real numbers; NaN is kept (never imputed or
      dropped), infinity and complex values are refused. A within-field
      coordinate (``z``, ``z0``, ``z1``, ``c``, ``t``) is never a feature
      (``planes_as_features`` turns a feature into one per plane).
    - The field identifier is ``fov_col``, or the first of ``position_id``,
      ``image_file`` and ``fov_id`` present in the tables; the analysis groups
      are ``group_cols``, or the MorphoBricks ``condition``,
      ``biological_repeat`` and ``technical_repeat`` present, and all of these
      are part of the field identity. Identity values are complete labels (no
      missing or blank value, no list). Near-miss spellings
      (``Biological_Repeat``, ``bio_repeat``, ``technical.repeat``, ``Z``) are
      refused unless named explicitly. The former name ``repeat`` (a column,
      or in ``group_cols``/``fov_col``) is read as ``biological_repeat`` with a
      ``DeprecationWarning``; a table with both is refused unless they are
      identical.
    - An optional shared column (``position_id``, ``technical_repeat``) empty
      in every row of a table is read as absent from it; when only the
      observation table's copy is empty, the FOV table's values are used.
      Group columns held only by the FOV table are attached to the
      observations through the field identifier (with the other group
      columns when a field name restarts in each repeat).
    - A position spread over several files must tell its files apart by
      ``z``/``c``/``t``; with an automatically chosen file-level identifier,
      fields that each hold one plane of a varying ``z``/``c``/``t`` are
      refused (a warning with one row per field).
    - ``fovs`` lists every observation's field exactly once. Without it the
      FOV table is rebuilt from the observations, with a warning and
      ``ValidatedInput.fovs_from_observations=True``: fields without
      observations are then missing.
    - ``observation_unit`` says what one row is; when omitted, rows are
      assumed to be objects and results record
      ``observation_unit_declared=False``.

    When the tables are refused, the error (same type, first line unchanged)
    carries the full ``check_tables`` report: every problem, where, and how
    to fix it.
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
            report = check_tables(objects, **arguments).summary(detail="full")
        except Exception:  # noqa: BLE001 -- the report is a courtesy; never hide the real error
            raise exc from None
        raise type(exc)(f"{exc}\n\n{report}") from exc


__all__ = ["DataCheck", "Problem", "check_tables", "validate_input_tables"]
