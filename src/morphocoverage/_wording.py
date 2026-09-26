"""Shared wording of the summaries (private): percentages, counts, fields and groups.

Every summary formats numbers through these helpers, so the same number is
written the same way everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def percent(value: float) -> str:
    """A share as a short percentage; small non-zero shares keep one significant digit."""
    return f"{100 * value:.0f}%" if value >= 0.01 or value == 0 else f"{100 * value:.1g}%"


def number(value: float) -> str:
    """A count or median as a person reads it: ``7``, ``7.5``, ``1234567``; one decimal at most."""
    value = round(float(value), 1)
    return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"


def count(n: float, word: str, plural: str | None = None) -> str:
    """``1 field``, ``3 fields``, ``2.5 fields`` (``plural`` defaults to ``word + 's'``)."""
    many = plural if plural is not None else f"{word}s"
    written = number(n)  # singular when the number as written is 1 (1.04 reads "1 field")
    return f"{written} {word if written == '1' else many}"


def fields(n: float) -> str:
    """``1 field``, ``12 fields``, ``2.5 fields``."""
    return count(n, "field")


def duration(seconds: float) -> str:
    """``45 s``, ``12 min``, ``1.5 h``: never ``0 min`` for a short time."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{number(seconds / 3600)} h"


def ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``11th``, ``22nd``."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def where(group: Mapping[str, Any]) -> str:
    """``condition=MOCK, repeat=1A``, or a pooled label for an empty group."""
    return ", ".join(f"{k}={v}" for k, v in group.items()) or "all FOVs (pooled)"


def rows_of(frame: pd.DataFrame, group: Mapping[str, Any]) -> pd.DataFrame:
    """The rows of ``frame`` belonging to ``group`` (a missing value matches a missing one)."""
    mask = pd.Series(True, index=frame.index)
    for column, value in group.items():
        missing = value is None or (isinstance(value, float) and np.isnan(value))
        mask &= frame[column].isna() if missing else frame[column].eq(value)
    return frame.loc[mask]


__all__ = ["count", "duration", "fields", "number", "ordinal", "percent", "rows_of", "where"]
