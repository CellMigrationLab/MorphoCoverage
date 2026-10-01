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
from .._style import group_label, rule_label
from .._wording import (  # noqa: F401  (re-exported)
    column_word,
    fields,
    number,
    percent,
    plural,
    rows_of,
    where,
)
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural  # noqa: F401  (re-exported for the summaries)

_ORDER_PHRASES = {
    "recorded": "in the recorded order",
    "permuted": "in random orders",
    "dispersed": "spreading the same positions out",
}


# The same order kinds in a word, for a table's rows or columns.
_ORDER_NAMES = {"recorded": "recorded", "permuted": "random", "dispersed": "spread out"}


def order_name(order: str) -> str:
    """An order kind in a word, for a table: ``recorded``, ``random``, ``spread out``."""
    return _ORDER_NAMES.get(order, order)


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


def skipped_labels(skipped: pd.DataFrame | None, keys: Sequence[str]) -> list[str]:
    """``MOCK · 1A: skipped, <reason>.``: ``skipped_lines`` with the groups by their values."""
    if skipped is None or skipped.empty or "reason" not in skipped.columns:
        return []
    shown = [k for k in keys if k in skipped.columns]
    return [
        f"{group_label({k: row[k] for k in shown})}: skipped, {row['reason']}."
        for _, row in skipped.iterrows()
    ]


def rule_names(settings: Mapping[str, Any], detail: str = "short") -> dict[str, str]:
    """Each replayed rule's readable name by config id (``_style.rule_label``).

    From the replay's ``rule_labels``; for a replay recorded before them, from
    its strategies' configurations and kinds. ``detail="full"`` adds the
    strategy's own name with its settings and the config id.
    """
    labels = dict(settings.get("rule_labels") or {})
    kinds = settings.get("strategy_kinds", {})
    unit = settings.get("observation_unit")
    for config, configuration in settings.get("strategies", {}).items():
        if config not in labels and config in kinds:
            row = {"strategy": kinds[config], "policies": configuration.get("policies", [])}
            labels[config] = rule_label(row, unit)
    if detail == "short":
        return labels
    names = settings.get("strategy_names", {})
    return {
        config: f"{label} ({names.get(config, kinds.get(config, config))}, config {config})"
        for config, label in labels.items()
    }
