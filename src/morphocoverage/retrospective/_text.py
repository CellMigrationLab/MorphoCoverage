"""Shared wording and seeding helpers for the retrospective summaries (private)."""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

_PLURALS = {"nucleus": "nuclei", "object": "objects", "fov_summary": "field summaries"}


def percent(value: float) -> str:
    """A share as a short percentage; small non-zero shares keep one significant digit."""
    return f"{100 * value:.0f}%" if value >= 0.01 or value == 0 else f"{100 * value:.1g}%"


def fields(count: float) -> str:
    """``1 field``, ``12 fields``."""
    return "1 field" if count == 1 else f"{count:g} fields"


def where(group: Mapping[str, Any]) -> str:
    """``condition=MOCK, repeat=1A``, or a pooled label for an empty group."""
    return ", ".join(f"{k}={v}" for k, v in group.items()) or "all FOVs (pooled)"


def unit_plural(observation_unit: str) -> str:
    """Plural of a declared observation unit, e.g. ``nuclei``; undeclared rows are objects."""
    return _PLURALS.get(observation_unit, f"{observation_unit}s")


def group_identity(group: Mapping[str, Any]) -> int:
    """A stable integer for a group, so its random draws never depend on other groups."""
    return zlib.crc32(json.dumps(sorted(group.items()), default=str).encode())


def rows_of(frame: pd.DataFrame, group: Mapping[str, Any]) -> pd.DataFrame:
    """The rows of ``frame`` belonging to ``group``."""
    for column, value in group.items():
        frame = frame[frame[column] == value]
    return frame
