"""The problems a data check reports, and how they are written (private).

Shared-private: ``Problem``, ``Collector`` and the formatting helpers are used
by ``_data_rules`` and ``data_check``; ``_data_rules`` re-exports them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._wording import count, listing

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


def _value(value: Any) -> str:
    """A cell as a person reads it: text quoted, numbers plain, missing as empty."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):  # NaN, NA, NaT
        return "empty"
    return repr(value) if isinstance(value, str) else str(value)


def _examples(table: pd.DataFrame, rows: pd.Series, columns: Sequence[str]) -> str:
    """How many rows ``rows`` marks, and the values of ``columns`` in the first few.

    Rows are read by position, so an index that repeats a label still shows
    each row's own values.
    """
    positions = np.flatnonzero(rows.to_numpy())
    marked = table.index[positions]
    shown = []
    for position in positions[:EXAMPLES]:
        values = ", ".join(f"{c}={_value(table[c].iloc[position])}" for c in columns)
        label = table.index[position]
        label = label.item() if isinstance(label, np.generic) else label
        shown.append(f"row {label!r} ({values})")
    more = f" and {len(marked) - EXAMPLES} more" if len(marked) > EXAMPLES else ""
    return f"{count(len(marked), 'row')}: " + "; ".join(shown) + more


def _names(values: Sequence[Any]) -> str:
    """A short, quoted list of names: ``'a', 'b' and 'c'``, ``... and 3 more`` beyond 8."""
    return listing([repr(v) for v in values], limit=8)


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
