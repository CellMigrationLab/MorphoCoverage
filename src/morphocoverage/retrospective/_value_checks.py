"""Two checks of the values themselves, for "The data" (private; ``overview``; owner,
2026-10-02).

- **Fields more alike than chance**: in each analysis group with at least ``MIN_FIELDS``
  fields of at least two observations, a one-way analysis of variance of the feature over
  its fields, read in its lower tail: the field means agree more closely than observations
  drawn at random would let them (``F`` below what chance gives, p < ``ALIKE_P``). Values
  normalised per image (each image to its own mean), rounded to few levels or copied make
  fields alike; real fields of one well vary at least as much as random draws (a design
  effect of at least 1). A feature is named when at least half of its tested groups show it.
- **Few distinct values**: a feature whose most common value holds at least ``ONE_VALUE`` of
  its non-missing observations, or that takes at most ``FEW_LEVELS`` values with the most
  common holding at least ``COMMON_LEVEL`` (a fraction of pixels most patches do not have,
  counted in a few pixels). It stays in the analysis; a mean, d or test on so few levels
  rests on the observations that differ from the most common value.

Each is a measured property of these data, so it is a caution (⚠️), never a standing
assumption, and is said once, naming the features. A third check, of features that repeat
one measurement (copies, exact affine relations, near copies), is ``_feature_relations``
(owner, 2026-10-05).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .._wording import count, listing, percent
from ..evaluator import iter_analysis_groups
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..schema import ValidatedInput, fov_positions, observation_words
from ._feature_relations import relation_checks

# The fewest fields of two or more observations a group needs for the comparison.
MIN_FIELDS = 3
# The lower-tail p under which a group's field means are more alike than chance.
ALIKE_P = 0.001
# The share of observations holding one value from which a feature has few values.
ONE_VALUE = 0.9
# At most this many values, the most common holding COMMON_LEVEL, is also few values.
FEW_LEVELS = 25
COMMON_LEVEL = 1 / 3
# Features named at most in a sentence.
_LISTED = 4


def _alike_p(values: np.ndarray, positions: np.ndarray) -> float:
    """The lower-tail p of the one-way analysis of variance of ``values`` over their fields
    (``positions``), on the fields of two or more observations; NaN when it cannot be run."""
    keep = ~np.isnan(values) & (positions >= 0)
    values, positions = values[keep], positions[keep]
    sizes = np.bincount(positions)
    used = np.isin(positions, np.flatnonzero(sizes >= 2))
    values, positions = values[used], positions[used]
    fields = np.unique(positions)
    k, n = len(fields), len(values)
    if k < MIN_FIELDS or n - k < 1:
        return np.nan
    codes = np.searchsorted(fields, positions)
    sizes = np.bincount(codes).astype(float)
    means = np.bincount(codes, weights=values) / sizes
    within = float(((values - means[codes]) ** 2).sum()) / (n - k)
    if within <= 0:
        return np.nan
    between = float((sizes * (means - values.mean()) ** 2).sum()) / (k - 1)
    return float(stats.f.cdf(between / within, k - 1, n - k))


def alike_fields(data: ValidatedInput) -> str | None:
    """The sentence naming the features whose fields are more alike than chance (see the
    module); ``None`` when none is."""
    found: dict[str, tuple[int, int]] = {}
    word = "group"
    for group, group_data in iter_analysis_groups(data):
        positions = fov_positions(group_data, group_data.fovs)
        if group and list(group)[-1] == TECHNICAL_REPEAT_COLUMN:
            word = "well"
        for feature in data.feature_cols:
            p = _alike_p(group_data.feature_values(feature), positions)
            if np.isnan(p):
                continue
            alike, tested = found.get(feature, (0, 0))
            found[feature] = (alike + int(p < ALIKE_P), tested + 1)
    named = [(f, a, t) for f, (a, t) in found.items() if a and 2 * a >= t]
    if not named:
        return None
    words = observation_words(data)[1]
    listed = listing(
        [f"{feature} (in {alike} of {count(tested, word)})" for feature, alike, tested in named],
        _LISTED,
    )
    return (
        f"Fields more alike than chance: {listed}: their field means agree more closely than "
        f"{words} drawn at random would let them (p < {ALIKE_P:g}). Values normalised per "
        "image, rounded or copied do this; if they were normalised per image, the variation "
        "between fields that the report measures has been removed."
    )


def few_values(data: ValidatedInput) -> str | None:
    """The sentence naming the features that take few distinct values (see the module);
    ``None`` when none does."""
    words = observation_words(data)[1]
    named = []
    for feature in data.feature_cols:
        values = data.feature_values(feature)
        values = values[~np.isnan(values)]
        if not len(values):
            continue
        levels, counts = np.unique(values, return_counts=True)
        share = float(counts.max()) / len(values)
        if share >= ONE_VALUE or (len(levels) <= FEW_LEVELS and share >= COMMON_LEVEL):
            common = f"{levels[counts.argmax()]:.3g} in {percent(share)} of {words}"
            named.append(f"{feature} ({count(len(levels), 'value')}, {common})")
    if not named:
        return None
    return (
        f"Few distinct values: {listing(named, _LISTED)}: kept in the analysis; a mean, d or "
        f"test on so few levels rests on the {words} that differ from the most common value."
    )


def value_checks(data: ValidatedInput) -> tuple[str, ...]:
    """The checks' sentences, those that found something: fields more alike than chance,
    few distinct values, then one per group of related features (``_feature_relations``)."""
    found = tuple(text for text in (alike_fields(data), few_values(data)) if text)
    return (*found, *relation_checks(data))


__all__ = [
    "ALIKE_P",
    "COMMON_LEVEL",
    "FEW_LEVELS",
    "MIN_FIELDS",
    "ONE_VALUE",
    "alike_fields",
    "few_values",
    "value_checks",
]
