"""Wording helpers of the retrospective summaries (private).

The shared formatting and seeding helpers live in ``morphocoverage._wording``
and ``morphocoverage._seeds`` and are re-exported here for the summaries.
Two rules every summary applies alike also live here, written once: the
median over runs in which a run that never got there counts as beyond its
own stream (``median_beyond``), and the sentences naming the groups an
analysis skipped (``skipped_lines``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._seeds import group_identity  # noqa: F401  (re-exported for the summaries)
from .._wording import fields, number, percent, plural, rows_of, where  # noqa: F401  (re-exported)
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural  # noqa: F401  (re-exported for the summaries)

_ORDER_PHRASES = {
    "recorded": "in the recorded order",
    "permuted": "in random orders",
    "dispersed": "spreading the same positions out",
}


def order_phrase(order: str, *, capital: bool = False) -> str:
    """Where a replay ran, as a sentence opens: ``in random orders``, ``In random orders``."""
    phrase = _ORDER_PHRASES.get(order, order)
    return phrase[:1].upper() + phrase[1:] if capital else phrase


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


def column_word(column: object) -> str:
    """A column's own name as a word: ``plate``, ``technical repeat`` (underscores as spaces)."""
    return str(column).replace("_", " ")


def median_beyond(values: Sequence[Any] | pd.Series) -> float | None:
    """The median over runs, a missing value (a run that never got there) counting as beyond all.

    A run that never reached a point is beyond its own stream, whatever that
    stream's length, so it counts as larger than any reached value. ``None``
    when half or more of the runs never got there, or there are no runs.
    """
    series = pd.Series(list(values), dtype=object)
    filled = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not len(filled):
        return None
    ordered = np.sort(np.where(np.isnan(filled), np.inf, filled))
    middle = ordered[(len(ordered) - 1) // 2 : len(ordered) // 2 + 1]
    if not np.isfinite(middle).all():
        return None  # np.median would interpolate inf - inf and warn
    return float(middle.mean())


def skipped_lines(skipped: pd.DataFrame | None, keys: Sequence[str]) -> list[str]:
    """``<group>: skipped, <reason>.`` for every group an analysis could not run."""
    if skipped is None or skipped.empty or "reason" not in skipped.columns:
        return []
    shown = [k for k in keys if k in skipped.columns]
    return [
        f"{where({k: row[k] for k in shown})}: skipped, {row['reason']}."
        for _, row in skipped.iterrows()
    ]
