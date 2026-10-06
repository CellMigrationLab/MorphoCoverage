"""Sub-experiments and the correction within them (private).

A screen often runs as several sub-experiments: each target's reagents imaged with their own
control in their own biological repeats. The tests of one sub-experiment share its control
and its repeats; the tests of two sub-experiments share nothing. Correcting every test of
the report as one family would let the size of the screen decide what each sub-experiment
can show, so the p-values are corrected (Benjamini-Hochberg, as the owner decided,
2026-10-01) within each sub-experiment: the conditions linked by a biological repeat they
were both compared in (``experiments``; the whole report when every condition shares the
repeats).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .._control import VS
from .._seeds import group_key
from .._wording import features_counted, listing
from ..statistics import benjamini_hochberg

ALPHA = 0.05


def experiments(repeats: Mapping[Any, Iterable[Any]]) -> dict[Any, int]:
    """Each condition's sub-experiment, numbered from 0 in the order of ``repeats``: the
    conditions linked by a shared biological repeat (``repeats`` maps a condition to the
    repeats it was compared with the control in)."""
    parent = {condition: condition for condition in repeats}

    def root(condition: Any) -> Any:
        while parent[condition] != condition:
            parent[condition] = parent[parent[condition]]
            condition = parent[condition]
        return condition

    first: dict[Any, Any] = {}
    for condition, held in repeats.items():
        for repeat in held:
            if repeat in first:
                parent[root(condition)] = root(first[repeat])
            else:
                first[repeat] = condition
    numbers: dict[Any, int] = {}
    return {c: numbers.setdefault(root(c), len(numbers)) for c in repeats}


def repeat_experiments(
    groups: Iterable[Mapping[str, Any]],
    control: set[Any],
    condition_col: str,
    repeat_col: str,
) -> dict[Any, int]:
    """Each biological repeat's sub-experiment, from the analysis ``groups`` (dicts of their
    column values) and the keys (``_seeds.group_key``) of the ``control`` groups: the repeats
    linked by a condition other than the control (``experiments``). A repeat holding only the
    control belongs to none and is left out, so it stays comparable with every repeat; empty
    without both columns or without a condition beside the control."""
    held: dict[Any, list[Any]] = {}
    for group in groups:
        if condition_col not in group or repeat_col not in group:
            return {}
        if group_key(dict(group)) not in control:
            held.setdefault(group[condition_col], []).append(group[repeat_col])
    family = experiments(held)
    numbers: dict[Any, int] = {}
    for condition, members in held.items():
        for repeat in members:
            numbers.setdefault(repeat, family[condition])
    return numbers


def not_compared(conditions: list[str], control: str, word: str = "biological repeat") -> str:
    """``Not compared: cond3 (no biological repeat holds both it and ctrl).``: the conditions
    imaged in no repeat beside the control, which no test of question 9 can include. A
    comparison's label names its own reference (``THBS5_SiFBN2 vs THBS5_SiNC``), so it is
    said alone: ``(no biological repeat holds both)`` (owner, 2026-10-05)."""
    if conditions and all(VS in str(c) for c in conditions):
        if len(conditions) == 1:
            return f"Not compared: {conditions[0]} (no {word} holds both)."
        return f"Not compared: {listing(conditions, 4)} (for each, no {word} holds both)."
    if len(conditions) == 1:
        return f"Not compared: {conditions[0]} (no {word} holds both it and {control})."
    return f"Not compared: {listing(conditions, 4)} (none shares a {word} with {control})."


def bridges(repeats: Mapping[Any, Iterable[Any]]) -> dict[Any, int]:
    """The conditions that join otherwise separate sets of repeats into one sub-experiment,
    each with the number of sets it joins: without it, the other conditions of its
    sub-experiment would fall into two or more (``repeats`` as for ``experiments``). The
    sub-experiments are as ``experiments`` defines them; this only says why one is whole."""
    family = experiments(repeats)
    found: dict[Any, int] = {}
    for condition in repeats:
        others = {
            c: list(held)
            for c, held in repeats.items()
            if c != condition and family[c] == family[condition]
        }
        parts = len(set(experiments(others).values())) if others else 0
        if parts > 1:
            found[condition] = parts
    return found


def same_experiment(blocks: Mapping[Any, int], first: Any, second: Any) -> bool:
    """Whether two repeats may be compared: the same sub-experiment, or either in none."""
    return first not in blocks or second not in blocks or blocks[first] == blocks[second]


def corrected(p_values: pd.Series, family: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg q-values of ``p_values``, within each ``family`` (NaN kept)."""
    q = np.full(len(p_values), np.nan)
    values = p_values.to_numpy(dtype=float)
    labels = family.to_numpy()
    for label in pd.unique(labels):
        mine = labels == label
        q[mine] = benjamini_hochberg(values[mine])
    return q


def family_sizes(p_values: pd.Series, family: pd.Series) -> np.ndarray:
    """Per test, how many tests its family holds (those with a p-value)."""
    tested = p_values.notna().to_numpy()
    labels = family.to_numpy()
    sizes = {label: int(tested[labels == label].sum()) for label in pd.unique(labels)}
    return np.array([sizes[label] for label in labels])


def family_words(settings: Mapping[str, Any], own: bool = False) -> str:
    """Where a q-value was corrected, in words: within each sub-experiment (``own``: the
    condition's), over every feature and condition of it when the report shares one family
    across its features (``settings["family_features"]``, ``workflows._families``; a
    labelled state's share among them, ``family_share``, counted apart)."""
    where = "the condition's sub-experiment" if own else "each sub-experiment"
    features = settings.get("family_features")
    if features and features > 1:
        counted = features_counted(int(features), settings.get("family_share"), ", ")
        return f"within {where}, over its {counted} and their conditions"
    return f"within {where}"


__all__ = [
    "ALPHA",
    "bridges",
    "corrected",
    "experiments",
    "family_sizes",
    "family_words",
    "not_compared",
    "repeat_experiments",
    "same_experiment",
]
