"""The steps of ``data_check.check_tables`` (private), in the order it runs them.

Each step records problems on a ``Collector`` in plain words, and what was understood
about the tables, and never raises for bad data: the tables are tables, the arguments
readable, the sizes and the declared unit, the fields and their groups, then the rules of
the validator itself, its warnings and errors in this check's words.

Shared-private: every name here is used by ``data_check`` alone.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import pandas as pd

from ._data_rules import (
    Collector,
    _attach,
    _check_copies,
    _check_fov_table,
    _check_key_values,
    _check_label_variants,
    _examples,
    _names,
    _resolve_groups,
)
from ._former_repeat import FORMER_REPEAT_COLUMN
from ._input_traps import (
    InvalidRowsWarning,
    _check_container_files,
    _check_undeclared_groups,
    _check_unimaged_technical_repeats,
    _check_well_names,
)
from ._wording import count
from .metadata import column_names, empty_optional_columns
from .schema import (
    MissingFovTableWarning,
    _validate_input_tables,
    normalize_observation_unit,
    unhashable_values,
)


class _Arguments(NamedTuple):
    """The column arguments of ``check_tables`` as read once (see ``_read_arguments``)."""

    feature_cols: list
    features_readable: bool
    group_cols: list | None
    observation_unit: str | None
    unit_given: bool
    fov_col_usable: bool
    explicit: list


def _not_tables(c: Collector, objects: Any, fovs: Any) -> bool:
    """Whether the observation table, or a FOV table given, is not a DataFrame (reported)."""
    if not isinstance(objects, pd.DataFrame):
        c.error(
            "The observation table is not a table.",
            f"objects is a {type(objects).__name__}.",
            "Pass a pandas DataFrame, e.g. pd.read_csv('cells.csv').",
        )
        return True
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        c.error(
            "The FOV table is not a table.",
            f"fovs is a {type(fovs).__name__}.",
            "Pass a pandas DataFrame with one row per acquired field.",
        )
        return True
    return False


def _report_former_repeat(c: Collector, error: ValueError) -> None:
    """A table or ``control`` holding the former name and the new one, different."""
    c.error(
        f"The former name {FORMER_REPEAT_COLUMN!r} and the biological repeat's column "
        "hold different values.",
        f"{error}.",
        "Keep only 'biological_repeat', the biological repeat, and rename or remove the "
        f"{FORMER_REPEAT_COLUMN!r} column (or control key).",
    )


def _check_repeated_columns(c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None) -> None:
    """A table with two columns of one name."""
    for name, table in (("observation table", objects), ("FOV table", fovs)):
        repeated = [] if table is None else list(table.columns[table.columns.duplicated()].unique())
        if repeated:
            c.error(
                f"The {name} has more than one column with the same name.",
                f"{name}, columns {_names(repeated)}.",
                "Give every column its own name (a table joined side by side, e.g. with "
                "pd.concat(axis=1), can repeat one).",
            )


def _read_arguments(
    c: Collector, feature_cols: Any, group_cols: Any, observation_unit: Any, fov_col: Any
) -> _Arguments:
    """The column arguments read once and checked: a feature or group list that cannot be
    read, an observation unit that is not a word, a ``fov_col`` that is not a name."""
    # Read once (a generator is consumed by reading); None when absent or already reported.
    features = _column_list(c, feature_cols, "feature_cols")
    features_readable = features is not None or feature_cols is None
    feature_cols = list(features or ())
    group_cols = _column_list(c, group_cols, "group_cols")
    unit_given = observation_unit is not None
    if unit_given:
        try:
            normalize_observation_unit(observation_unit)
        except ValueError:  # fallback: reported as a problem of the check
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
    return _Arguments(
        feature_cols,
        features_readable,
        group_cols,
        observation_unit,
        unit_given,
        fov_col_usable,
        explicit,
    )


def _understand_sizes(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, arguments: _Arguments
) -> None:
    """The tables' sizes and the unit, understood first; no FOV table, no observation and
    no declared unit warned about."""
    observation_unit = arguments.observation_unit
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
    if not arguments.unit_given:
        c.warning(
            "The observation unit is not declared.",
            "observation_unit=None.",
            "Say what one row is, e.g. observation_unit='cell', 'nucleus' or 'patch'; the report "
            "otherwise says 'object' and records that the unit was assumed.",
        )


def _check_fields(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    arguments: _Arguments,
    resolved: str,
) -> tuple[pd.DataFrame, set[str]]:
    """The groups and the field identity understood, then every rule that matches fields;
    the observations with the FOV table's group columns attached, and the columns taken by
    the identity."""
    groups, identity = _resolve_groups(c, objects, fovs, arguments.group_cols, resolved)
    taken = {resolved, *identity}
    key = [*identity, resolved]
    c.understood.append(
        f"A field is identified by {resolved!r}"
        + (f" within {', '.join(identity)}." if identity else ".")
    )
    c.understood.append(
        "Analysis groups: " + (", ".join(groups) if groups else "none (all fields pooled)") + "."
    )
    if _identity_hashable(c, objects, fovs, key):
        feature_cols = arguments.feature_cols
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
    return objects, taken


def _check_as_validator(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    arguments: _Arguments,
    fov_col: Any,
) -> None:
    """The remaining rules (planes, channels or frames counted as fields) run as the
    validator runs them; its warnings become warnings of this check, its error an error."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_input_tables(
                objects,
                feature_cols=arguments.feature_cols,
                fov_col=fov_col,
                fovs=fovs,
                group_cols=arguments.group_cols,
                observation_unit=arguments.observation_unit,
            )
        _report_validator_warnings(c, caught)
    except (ValueError, TypeError) as exc:  # fallback: reported as a problem of the check
        _report_validator_error(c, exc)


def _report_validator_warnings(c: Collector, caught: list[warnings.WarningMessage]) -> None:
    """The validator's warnings in this check's words; any warning that is not a
    ``UserWarning`` is warned again, at the caller of ``check_tables``."""
    for warning in caught:
        if issubclass(warning.category, MissingFovTableWarning):
            continue  # said above in this report's words ("No FOV table was given.")
        if not issubclass(warning.category, UserWarning):
            # stacklevel 4: past this helper, _check_as_validator and check_tables, so the
            # warning names the same caller as when check_tables warned it itself.
            warnings.warn(warning.message, warning.category, stacklevel=4)
            continue
        message = str(warning.message)
        if issubclass(warning.category, InvalidRowsWarning):
            c.warning(
                message[0].upper() + message[1:],
                "observation table, column 'valid'.",
                "Keep only the valid rows, or rename the column if it means something else.",
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


def _report_validator_error(c: Collector, exc: Exception) -> None:
    """The validator's error, its first paragraph, with where to look and how to fix it."""
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
        else "Correct what the message names; every rule is in docs/common/data_format.md.",
    )


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
    except TypeError as error:  # fallback: reported as a problem of the check
        c.error(
            f"{name} is not a list of column names.",
            f"{name}={columns!r}: {error}.",
            f"Pass a list of column names, e.g. {name}=['area'].",
        )
        return None
    return None if names is None else list(names)
