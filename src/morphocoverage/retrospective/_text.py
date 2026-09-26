"""Wording helpers of the retrospective summaries (private).

The shared formatting and seeding helpers live in ``morphocoverage._wording``
and ``morphocoverage._seeds`` and are re-exported here for the summaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._seeds import group_identity  # noqa: F401  (re-exported for the summaries)
from .._wording import fields, number, percent, rows_of, where  # noqa: F401  (re-exported)
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural  # noqa: F401  (re-exported for the summaries)


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
