"""Shared wording and seeding helpers for the retrospective summaries (private)."""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural  # noqa: F401  (re-exported for the summaries)


def percent(value: float) -> str:
    """A share as a short percentage; small non-zero shares keep one significant digit."""
    return f"{100 * value:.0f}%" if value >= 0.01 or value == 0 else f"{100 * value:.1g}%"


def fields(count: float) -> str:
    """``1 field``, ``12 fields``."""
    return "1 field" if count == 1 else f"{count:g} fields"


def where(group: Mapping[str, Any]) -> str:
    """``condition=MOCK, repeat=1A``, or a pooled label for an empty group."""
    return ", ".join(f"{k}={v}" for k, v in group.items()) or "all FOVs (pooled)"


def group_identity(group: Mapping[str, Any]) -> int:
    """A stable integer for a group, so its random draws never depend on other groups."""
    return zlib.crc32(json.dumps(sorted(group.items()), default=str).encode())


def rows_of(frame: pd.DataFrame, group: Mapping[str, Any]) -> pd.DataFrame:
    """The rows of ``frame`` belonging to ``group``."""
    for column, value in group.items():
        frame = frame[frame[column] == value]
    return frame


def sample_word(hierarchy: Mapping[str, Any]) -> str:
    """What a sample is called: a technical repeat when it is the MorphoBricks column."""
    return "technical repeat" if hierarchy.get("sample") == TECHNICAL_REPEAT_COLUMN else "sample"


def unit_word(hierarchy: Mapping[str, Any], unit: str) -> str:
    """The word for an independent unit named by its column."""
    if unit == hierarchy.get("sample"):
        return sample_word(hierarchy)
    if unit == hierarchy.get("biological_repeat"):
        return "repeat"
    return unit
