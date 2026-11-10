"""The former name of the biological-repeat column, refused (private).

MorphoBricks named the shared biological-repeat column ``repeat`` before it became
``biological_repeat`` (``metadata.REPEAT_COLUMN``). Owner ruling 14 (2026-11-10): no
backward compatibility is kept before the first pipeline release, so a table, a file's table,
a column name (``group_cols``, ``fov_col``, ``repeat_col``, ...) or a control mapping that
uses ``repeat`` is refused, naming ``biological_repeat``. Nothing is renamed or translated.

Shared-private: called where tables and column arguments enter (the validator, ``check_tables``,
the workflows, the live session, and the public functions that read a column argument).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .metadata import REPEAT_COLUMN

_FORMER_REPEAT_COLUMN = "repeat"


def _holds_former_repeat(value: object) -> bool:
    """Whether ``value`` (a table, a mapping of columns, a column name or a list of names)
    names the column ``repeat``; an iterator or set is left unread."""
    if isinstance(value, pd.DataFrame):
        return _FORMER_REPEAT_COLUMN in value.columns
    if isinstance(value, Mapping):
        return _FORMER_REPEAT_COLUMN in value
    if isinstance(value, str):
        return value == _FORMER_REPEAT_COLUMN
    if isinstance(value, (list, tuple)):
        return any(isinstance(name, str) and name == _FORMER_REPEAT_COLUMN for name in value)
    return False


def refuse_repeat_alias(**arguments: Any) -> None:
    """Refuse (``ValueError``) a table, column name, list of names or mapping of columns that
    uses ``repeat``, the former name of ``biological_repeat``.

    Each keyword names the argument in the message. The caller's values are not changed.
    """
    for name, value in arguments.items():
        if _holds_former_repeat(value):
            raise ValueError(refused_repeat_message(name))


def refused_repeat_message(where: str) -> str:
    """The message that refuses the former name ``repeat`` in ``where`` and names the column
    to use."""
    return (
        f"{where} uses {_FORMER_REPEAT_COLUMN!r}, which is no longer accepted: the biological "
        f"repeat's column is {REPEAT_COLUMN!r}. Rename the column to {REPEAT_COLUMN!r} (a file "
        "written with the old name must be written again)."
    )
