"""Check a dataset before analysing it, and say in plain words what is wrong and how to fix it.

``check_tables`` runs every check of ``validate_input_tables`` without stopping
at the first problem, and a few more that do not stop the analysis but are
often mistakes. It returns a ``DataCheck`` whose ``summary()`` lists, for each
problem, what is wrong, where (the table, the column, how many rows and a few
example rows) and how to fix it, together with what was understood about the
tables. ``validate_input_tables`` and every workflow attach the same summary
to the error they raise, so the full list reaches the user whichever way the
data were loaded. The format itself is described in ``docs/common/data_format.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ._data_check_steps import (
    _check_as_validator,
    _check_fields,
    _check_repeated_columns,
    _not_tables,
    _read_arguments,
    _report_former_repeat,
    _understand_empty_columns,
    _understand_sizes,
)
from ._data_rules import (
    ERROR,
    WARNING,
    Collector,
    Problem,
    _check_control,
    _check_spellings,
    _names,
    _resolve_fov_col,
)
from ._feature_rules import _check_features
from ._former_repeat import Reader, one_warning
from ._style import CAUTION, PROBLEM, Status, check_detail, headline, wrap
from ._wording import count
from .metadata import drop_empty_optional_columns
from .schema import ValidatedInput, _validate_input_tables

FORMAT_DOC = "docs/common/data_format.md"


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
    if _not_tables(c, objects, fovs):
        return _found(c)
    try:
        objects, fovs, group_cols, fov_col, control = _read_former_repeat(
            objects, fovs, group_cols, fov_col, control
        )
    except ValueError as error:  # fallback: reported as a problem of the check
        _report_former_repeat(c, error)
        return _found(c)
    _check_repeated_columns(c, objects, fovs)
    if c.has_errors():
        # Every other rule reads columns by name, which is ambiguous here.
        return _found(c)
    arguments = _read_arguments(c, feature_cols, group_cols, observation_unit, fov_col)
    feature_cols = arguments.feature_cols
    explicit = arguments.explicit

    _understand_empty_columns(c, objects, fovs, explicit)
    objects, fovs = drop_empty_optional_columns(objects, fovs, explicit)
    _check_spellings(c, objects, fovs, explicit)
    _understand_sizes(c, objects, fovs, arguments)

    resolved = _resolve_fov_col(c, objects, fovs, fov_col) if arguments.fov_col_usable else None
    taken: set[str] = set()
    if resolved is not None:
        objects, taken = _check_fields(c, objects, fovs, arguments, resolved)
    if arguments.features_readable:
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
        _check_as_validator(c, objects, fovs, arguments, fov_col)
    return _found(c)


def _found(c: Collector) -> DataCheck:
    """The check's result: the problems and what was understood, as collected."""
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
    The rules, all described column by column in ``docs/common/data_format.md``:

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
