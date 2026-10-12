"""Sub-experiments and the correction within them (private).

A screen often runs as several sub-experiments: each target's reagents imaged with their own
control in their own biological repeats. The tests of one sub-experiment share its control
and its repeats; the tests of two sub-experiments share nothing. Correcting every test of
the report as one family would let the size of the screen decide what each sub-experiment
can show, so the p-values are corrected (Benjamini-Hochberg, as the owner decided,
2026-10-01) within each sub-experiment: the conditions linked by a biological repeat they
were both compared in (``experiments``; the whole report when every condition shares the
repeats).

The Setup may name the families instead (``Setup(experiment=...)``, audit 10, owner,
2026-10-06): each declared experiment is one family (``declared``), so a set imaged in the
same repeats as another but answering another question, an untransfected pair beside an
siRNA set, is not corrected with it, and a screen declared as one experiment is corrected as
one; a condition no declaration names keeps its sub-experiment. When tests of two or more
families share one control label (a screen whose targets were imaged in their own repeats
against the same control), the screen-wide q, over every test against that label, is given
beside each verdict (``screen_wide``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .._connected import connected_groups
from .._control import VS
from .._seeds import group_key
from .._wording import features_counted, listing
from ..statistics import ALPHA, benjamini_hochberg


def experiments(repeats: Mapping[Any, Iterable[Any]]) -> dict[Any, int]:
    """Each condition's sub-experiment, numbered from 0 in the order of ``repeats``: the
    conditions linked by a shared biological repeat (``repeats`` maps a condition to the
    repeats it was compared with the control in)."""
    first: dict[Any, Any] = {}
    links = []
    for condition, held in repeats.items():
        for repeat in held:
            if repeat in first:
                links.append((condition, first[repeat]))
            else:
                first[repeat] = condition
    group = connected_groups(list(repeats), links)
    numbers: dict[Any, int] = {}
    return {c: numbers.setdefault(group[c], len(numbers)) for c in repeats}


def declared(numbers: Mapping[Any, int], families: Mapping[Any, Any] | None) -> dict[Any, int]:
    """Each condition's correction family, numbered from 0 in order: its declared experiment
    (``families``: condition -> experiment; see the module), else its sub-experiment
    (``numbers``, ``experiments``)."""
    if not families:
        return dict(numbers)
    codes: dict[Any, int] = {}
    keys = {
        c: ("declared", families[c]) if c in families else ("derived", n)
        for c, n in numbers.items()
    }
    return {c: codes.setdefault(key, len(codes)) for c, key in keys.items()}


def screen_wide(p_values: pd.Series, controls: pd.Series, family: pd.Series) -> np.ndarray:
    """The screen-wide q of each test (see the module): Benjamini-Hochberg over every test
    against its control label when those tests fall in two or more families; NaN for a
    label whose tests are one family, where the family's q is the screen's."""
    families = family.to_numpy()
    finite = np.isfinite(p_values.to_numpy(dtype=float))
    return _within(
        p_values,
        controls.astype(str).to_numpy(),
        lambda mine: len(pd.unique(families[mine & finite])) > 1,
    )


def _within(
    p_values: pd.Series, labels: np.ndarray, applies: Callable[[np.ndarray], bool] | None = None
) -> np.ndarray:
    """Benjamini-Hochberg q-values of ``p_values`` within each label of ``labels``, for the
    labels whose rows ``applies`` accepts (all by default); NaN elsewhere and NaN kept."""
    q = np.full(len(p_values), np.nan)
    values = p_values.to_numpy(dtype=float)
    for label in pd.unique(labels):
        mine = labels == label
        if applies is None or applies(mine):
            q[mine] = benjamini_hochberg(values[mine])
    return q


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
    return _within(p_values, family.to_numpy())


def family_sizes(p_values: pd.Series, family: pd.Series) -> np.ndarray:
    """Per test, how many tests its family holds (those with a p-value)."""
    tested = p_values.notna().to_numpy()
    labels = family.to_numpy()
    sizes = {label: int(tested[labels == label].sum()) for label in pd.unique(labels)}
    return np.array([sizes[label] for label in labels])


def family_noun(settings: Mapping[str, Any] | None, plural: bool = False) -> str:
    """``sub-experiment``, or ``experiment`` when the Setup declares the correction families
    (``settings["declared_experiments"]``, ``Setup.experiment``; owner, 2026-10-06)."""
    word = "experiment" if (settings or {}).get("declared_experiments") else "sub-experiment"
    return f"{word}s" if plural else word


def family_words(settings: Mapping[str, Any], own: bool = False) -> str:
    """Where a q-value was corrected, in words: within each sub-experiment (``own``: the
    condition's), over every feature and condition of it when the report shares one family
    across its features (``settings["family_features"]``, ``workflows._families``; a
    labelled state's share among them, ``family_share``, counted apart)."""
    where = f"the condition's {family_noun(settings)}" if own else f"each {family_noun(settings)}"
    features = settings.get("family_features")
    if features and features > 1:
        counted = features_counted(int(features), settings.get("family_share"), ", ")
        return f"within {where}, over its {counted} and their conditions"
    return f"within {where}"


__all__ = [
    "ALPHA",
    "bridges",
    "corrected",
    "declared",
    "experiments",
    "family_noun",
    "family_sizes",
    "family_words",
    "not_compared",
    "repeat_experiments",
    "same_experiment",
    "screen_wide",
]
