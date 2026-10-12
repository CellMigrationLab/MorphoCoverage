"""The former name of the biological-repeat column, still read with a warning (private).

MorphoBricks renamed its shared biological-repeat column from ``repeat`` to
``biological_repeat`` (``metadata.REPEAT_COLUMN``). A table, a column name
(``group_cols``, ``fov_col``, ``repeat_col``, ...) or a control mapping that
still uses ``repeat`` is read as ``biological_repeat`` where input enters the
package (the validator, ``check_tables``, the workflows, the live session),
with one ``DeprecationWarning`` per call: the caller's tables are never
modified, and every result names the new column. A table that holds both
columns is refused unless they are identical, when ``biological_repeat`` is
used. The former name will be removed in a later version.

``Reader`` collects what one call read under the former name and warns once
(``read_former``, ``read_fields`` and ``renamed_tables`` are its short forms);
``one_warning`` makes the functions it decorates share that single warning
with every function they call (an analysis that validates several times, a
live poll that adds several fields).
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from .metadata import REPEAT_COLUMN

FORMER_REPEAT_COLUMN = "repeat"

_Function = TypeVar("_Function", bound=Callable[..., Any])
# Within a call decorated by ``one_warning``: whether the former name was warned about.
_SCOPE: ContextVar[list[bool] | None] = ContextVar("former_repeat_scope", default=None)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _is_former(name: object) -> bool:
    return isinstance(name, str) and name == FORMER_REPEAT_COLUMN


def _differing_rows(former: pd.Series, new: pd.Series) -> np.ndarray:
    """Which rows hold different values in the two columns (missing values alike)."""
    old = former.astype(object).reset_index(drop=True)
    current = new.astype(object).reset_index(drop=True)
    return (old.ne(current) & ~(old.isna() & current.isna())).to_numpy(dtype=bool)


class Reader:
    """What one call reads under the former name ``repeat``; ``warn`` says it once."""

    def __init__(self) -> None:
        self.places: list[str] = []
        self.notes: list[str] = []

    def table(self, table: Any, name: str) -> Any:
        """``table`` with its ``repeat`` column named ``biological_repeat`` (a new table).

        Anything that is not a DataFrame, or has no ``repeat`` column, is
        returned as it is. With both columns, the former is dropped when they
        are identical and the table refused (``ValueError``) when they differ.
        """
        if not isinstance(table, pd.DataFrame) or FORMER_REPEAT_COLUMN not in table.columns:
            return table
        self.places.append(name)
        both = REPEAT_COLUMN in table.columns
        columns = table.columns
        if not both or columns.duplicated().any():
            # A column named twice is refused by name after the renaming, as any other.
            return table.rename(columns={FORMER_REPEAT_COLUMN: REPEAT_COLUMN})
        former, new = table[FORMER_REPEAT_COLUMN], table[REPEAT_COLUMN]
        differ = _differing_rows(former, new)
        if differ.any():
            first = int(differ.argmax())
            raise ValueError(
                f"{name} has both {FORMER_REPEAT_COLUMN!r} (the former name of "
                f"{REPEAT_COLUMN!r}) and {REPEAT_COLUMN!r}, with different values (row "
                f"{table.index[first]!r}: {former.iloc[first]!r} against {new.iloc[first]!r}); "
                f"keep only {REPEAT_COLUMN!r}, the biological repeat, and rename or remove the "
                "other column"
            )
        self.notes.append(
            f"{name} holds both columns with the same values, so {REPEAT_COLUMN!r} is used"
        )
        return table.drop(columns=FORMER_REPEAT_COLUMN)

    def column(self, name: Any, argument: str) -> Any:
        """One column name, ``biological_repeat`` for ``repeat``."""
        if not _is_former(name):
            return name
        self.places.append(argument)
        return REPEAT_COLUMN

    def columns(self, names: Any, argument: str) -> Any:
        """A sequence of column names with ``repeat`` read as ``biological_repeat``.

        A string, a set or anything that is not a collection of names is
        returned as it is, for the caller's own check to refuse; an iterator is
        read once and returned as a tuple.
        """
        if names is None or isinstance(names, (str, set, frozenset, Mapping)):
            return names
        if not isinstance(names, Iterable):
            return names
        if isinstance(names, Iterator):
            names = tuple(names)
        if not any(_is_former(name) for name in names):
            return names
        self.places.append(argument)
        translated = [REPEAT_COLUMN if _is_former(name) else name for name in names]
        return translated if isinstance(names, list) else tuple(translated)

    def mapping(self, mapping: Any, argument: str) -> Any:
        """A mapping of column names to values with the key ``repeat`` read as the new one.

        With both keys, the former is dropped when both give the same value
        and the mapping refused (``ValueError``) when they differ.
        """
        if not isinstance(mapping, Mapping) or FORMER_REPEAT_COLUMN not in mapping:
            return mapping
        self.places.append(argument)
        if REPEAT_COLUMN in mapping:
            if mapping[REPEAT_COLUMN] != mapping[FORMER_REPEAT_COLUMN]:
                raise ValueError(
                    f"{argument} names both {FORMER_REPEAT_COLUMN!r} (the former name of "
                    f"{REPEAT_COLUMN!r}) and {REPEAT_COLUMN!r}, with different values; name "
                    f"only {REPEAT_COLUMN!r}"
                )
            return {k: v for k, v in mapping.items() if not _is_former(k)}
        return {(REPEAT_COLUMN if _is_former(k) else k): v for k, v in mapping.items()}

    def read(self, value: Any, name: str) -> Any:
        """``value`` read by its kind: a table (``table``), a mapping of columns to values
        (``mapping``), one column name (``column``) or a sequence of them (``columns``)."""
        if isinstance(value, pd.DataFrame):
            return self.table(value, name)
        if isinstance(value, Mapping):
            return self.mapping(value, name)
        if isinstance(value, str):
            return self.column(value, name)
        return self.columns(value, name)

    def warn(self) -> None:
        """One ``DeprecationWarning`` naming every place the former name was read, if any.

        Inside a call decorated by ``one_warning``, only the first reader that
        found the former name warns.
        """
        if not self.places:
            return
        scope = _SCOPE.get()
        if scope is not None:
            if scope:
                return
            scope.append(True)
        places = ", ".join(dict.fromkeys(self.places))
        notes = "".join(f"; {note}" for note in self.notes)
        warnings.warn(
            f"{FORMER_REPEAT_COLUMN!r} is the former name of {REPEAT_COLUMN!r}; rename the "
            f"column to {REPEAT_COLUMN!r} in {places}. It is read as {REPEAT_COLUMN!r} for "
            f"now{notes}; it will be removed in a later version.",
            DeprecationWarning,
            stacklevel=caller_stacklevel(_in_package),
        )


def one_warning(function: _Function) -> _Function:
    """``function``, with at most one former-name warning for the whole call (see the module)."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _SCOPE.get() is not None:
            return function(*args, **kwargs)
        token = _SCOPE.set([])
        try:
            return function(*args, **kwargs)
        finally:
            _SCOPE.reset(token)

    return wrapper  # type: ignore[return-value]


def read_former(**arguments: Any) -> tuple[Any, ...]:
    """The ``arguments`` with the former name read as the new one, in order, warning once.

    Each value is read by its kind (``Reader.read``); the keyword names the
    argument in the warning.
    """
    reader = Reader()
    values = tuple(reader.read(value, name) for name, value in arguments.items())
    reader.warn()
    return values


def read_fields(instance: Any, *names: str) -> None:
    """The fields ``names`` of a frozen dataclass read with the former name as the new one.

    Set in place (``object.__setattr__``, as in ``__post_init__``), warning once;
    the warning names each field with its class (``Setup.control``).
    """
    reader = Reader()
    for name in names:
        value = reader.read(getattr(instance, name), f"{type(instance).__name__}.{name}")
        object.__setattr__(instance, name, value)
    reader.warn()


def renamed_tables(
    objects: Any, fovs: Any, names: tuple[str, str] = ("objects", "fovs")
) -> tuple[Any, Any]:
    """Both linked tables with ``repeat`` read as ``biological_repeat``, warning once."""
    reader = Reader()
    objects, fovs = reader.table(objects, names[0]), reader.table(fovs, names[1])
    reader.warn()
    return objects, fovs


def in_package_file(frame: Any) -> bool:
    """Whether ``frame`` runs code of a package file."""
    return os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR)


def _in_package(frame: Any) -> bool:
    """Whether ``frame`` runs package code: a package file, or the ``__init__`` that
    ``dataclasses`` generates for a package class (``Setup``), which has no file."""
    if in_package_file(frame):
        return True
    instance = frame.f_locals.get("self") if frame.f_code.co_name == "__init__" else None
    generated = frame.f_code.co_filename.startswith("<")
    return generated and type(instance).__module__.startswith("morphocoverage.")


def caller_stacklevel(inside: Callable[[Any], bool] = in_package_file) -> int:
    """The ``stacklevel`` that makes a warning name the first caller outside the package,
    ``inside`` telling the package's frames (by default, those of its files).

    Called by the function that warns. A warning raised through several
    package functions (``run_strategies`` validating, an analysis validating
    twice) then points at the user's line, so it is shown once per line.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and inside(frame):
        frame = frame.f_back
        level += 1
    return level


__all__ = [
    "FORMER_REPEAT_COLUMN",
    "Reader",
    "one_warning",
    "read_fields",
    "read_former",
    "renamed_tables",
]
