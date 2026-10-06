"""The definition of new's own false-alarm rate at a group's size (owner, 2026-10-06, audit 10).

Whether a field "shows a new state" depends on the definition of new (its radius, transform,
``min_other_fovs`` and ``min_novel_observations``), on the number of fields and observations
per field, and on the number of features: with 17 features and one image per field, a radius
of ``sqrt(17)`` called 99% of the fields of a homogeneous population new with 5 fields, 92%
with 10 and 81% with 15 (audit 10B). A list of "new states" from such a definition says
nothing about the biology.

``null_rate`` simulates the definition on a homogeneous population of the group's size: the
same fields with the same observation counts, each observation drawn from one standard
normal distribution over the same number of features, judged by the same estimator, its
transform (the robust standardisation of the default definition, fitted on the other fields
in each fold) included, so the scale of the features does not matter. It is the share of
the fields the estimator calls new over ``NULL_DRAWS`` draws at most (in batches of
``_BATCH``, stopping once the share is clearly below or above ``NOT_INFORMATIVE``), seeded
by the group's size and the definition, so it is reproducible and cached
(``functools.lru_cache`` on the size, the features and the definition). Independent Gaussian
features are the reference of "homogeneous": correlated or heavy-tailed real features differ,
so it is a floor of the false alarms, not their estimate. Without a transform the radius is
in the features' own units, which a simulation cannot reproduce: no rate.

``not_informative`` gives the sentence a report prints in place of a list of fields when that
rate is above ``NOT_INFORMATIVE`` (20%): "not informative with 8 fields in 17 features".
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .._wording import count, listing, percent
from .support_novelty import SupportNoveltyEstimator

# Above this share of a homogeneous population's fields called new, a list of new states is
# not informative (owner, 2026-10-06, audit 10).
NOT_INFORMATIVE = 0.20
# The simulated populations, at most, and per batch: a batch whose share lies far from the
# threshold ends the simulation.
NULL_DRAWS = 20
_BATCH = 5
_CLEARLY = (0.05, 0.5)

# The definitions met, by their parameters, so the cache keys on text alone.
_DEFINITIONS: dict[str, SupportNoveltyEstimator] = {}


def _definition_key(novelty: SupportNoveltyEstimator) -> str:
    """The definition's parameters as text (its identity for the cache and the seed)."""
    return json.dumps(novelty.parameters, sort_keys=True, default=str)


@lru_cache(maxsize=512)
def _simulated(key: str, counts: tuple[int, ...], names: tuple[str, ...]) -> float | None:
    """The share of fields called new over the simulated homogeneous populations."""
    novelty = _DEFINITIONS[key]
    positions = np.repeat(np.arange(len(counts)), counts)
    seed = zlib.crc32(json.dumps([key, counts, names]).encode())
    rng = np.random.default_rng(seed)
    flagged = trials = 0
    for draw in range(NULL_DRAWS):
        matrix = rng.standard_normal((len(positions), len(names)))
        fields = novelty.new_state_fields(matrix, positions, names)
        if fields is None:
            continue
        flagged += len(fields)
        trials += len(counts)
        if (draw + 1) % _BATCH == 0 and not _CLEARLY[0] <= flagged / trials <= _CLEARLY[1]:
            break
    return flagged / trials if trials else None


def null_rate(
    novelty: SupportNoveltyEstimator, counts: Sequence[int], columns: Sequence[str]
) -> float | None:
    """The share of a homogeneous population's fields ``novelty`` calls new, at ``counts``
    observations per field (fields without any left out) in ``columns`` (see the module);
    ``None`` without a transform or when the definition cannot judge that many fields."""
    if novelty.transform is None:
        return None
    sizes = tuple(sorted(int(c) for c in counts if int(c) > 0))
    if not sizes:
        return None
    key = _definition_key(novelty)
    _DEFINITIONS.setdefault(key, novelty)
    return _simulated(key, sizes, tuple(str(c) for c in columns))


def not_informative(
    novelty: SupportNoveltyEstimator, counts: Sequence[int], columns: Sequence[str]
) -> tuple[str | None, float | None]:
    """``(reason, rate)``: the reason a list of new states is not informative at this size,
    ``None`` when the null rate is at most ``NOT_INFORMATIVE`` (or unknown), and the rate."""
    rate = null_rate(novelty, counts, columns)
    if rate is None or rate <= NOT_INFORMATIVE:
        return None, rate
    fields = sum(1 for c in counts if int(c) > 0)
    reason = (
        f"not informative with {count(fields, 'field')} in {count(len(columns), 'feature')}: "
        f"this definition of new calls {percent(rate)} of the fields of a homogeneous "
        "population of this size new"
    )
    return reason, rate


def uninformative_line(entries: Sequence[dict]) -> str | None:
    """The groups where new states are not listed, in one sentence (owner, 2026-10-06: one
    line, not one per group): ``New states are not listed in 9 groups (a, b, c and 6 more):
    not informative with 15 to 27 fields in 10 features each: ...``; ``None`` without any.
    ``entries``: the inspection's ``uninformative_novelty``, each with its ``label``,
    ``reason``, ``rate``, ``fields`` and ``features``."""
    if not entries:
        return None
    labels = [str(e.get("label", e["group"])) for e in entries]
    if len(entries) == 1:
        return f"New states are not listed in {labels[0]}: {entries[0]['reason']}"
    fields = sorted({int(e["fields"]) for e in entries})
    features = sorted({int(e["features"]) for e in entries})
    rates = sorted(float(e["rate"]) for e in entries)
    size = f"{fields[0]}" if len(fields) == 1 else f"{fields[0]} to {fields[-1]}"
    width = f"{features[0]}" if len(features) == 1 else f"{features[0]} to {features[-1]}"
    rate = (
        percent(rates[0])
        if percent(rates[0]) == percent(rates[-1])
        else f"{percent(rates[0])} to {percent(rates[-1])}"
    )
    return (
        f"New states are not listed in {len(entries)} groups ({listing(labels, 3)}): not "
        f"informative with {size} fields in {width} features each, where this definition of "
        f"new calls {rate} of the fields of a homogeneous population of these sizes new"
    )


__all__ = ["NOT_INFORMATIVE", "NULL_DRAWS", "not_informative", "null_rate", "uninformative_line"]
