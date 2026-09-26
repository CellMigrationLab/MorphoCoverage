"""Statistics shared by the analyses, isolated so they are written once and tested once.

Every function takes plain arrays and returns plain numbers, so it can be
read and checked on its own. The field-aware versions take the objects of an
acquisition as a list of arrays, one per field, because that is how a
microscope produced them (the Foundation of the report): whole fields are the
unit that is resampled, drawn or shuffled, never individual objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def cohen_d(first: np.ndarray, second: np.ndarray) -> float:
    """Cohen's d with the pooled sample variance, as the lab's Plot&Stats notebook computes it.

    ``(mean(first) - mean(second)) / sqrt(pooled variance)``; NaN when a side
    has fewer than two values or the pooled variance is zero.
    """
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = ((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return float((first.mean() - second.mean()) / np.sqrt(pooled))


def jcs_effect_size(treated: np.ndarray, control: np.ndarray) -> float:
    """The JCS effect size (Marcotti et al. 2026, Eqn 2): treated mean minus control mean,
    over the control standard deviation; NaN when the control SD is undefined or zero."""
    if len(control) < 2 or len(treated) == 0:
        return float("nan")
    sd = control.std(ddof=1)
    return float((treated.mean() - control.mean()) / sd) if sd > 0 else float("nan")


def field_sample(fields: Sequence[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` objects as a microscope stopping at ``n`` has them: whole fields in random order,
    with a random part of the last field. Needs at least ``n`` objects in total."""
    taken, count = [], 0
    for i in rng.permutation(len(fields)):
        field = fields[i]
        need = n - count
        if len(field) >= need:
            taken.append(rng.choice(field, size=need, replace=False))
            break
        taken.append(field)
        count += len(field)
    return np.concatenate(taken)


def fields_holding(
    fields: Sequence[np.ndarray], n: int, rng: np.random.Generator, draws: int = 50
) -> float:
    """Mean number of fields, in random order, whose objects reach ``n`` objects."""
    sizes = np.array([len(f) for f in fields])
    counts = []
    for _ in range(draws):
        cumulative = np.cumsum(sizes[rng.permutation(len(sizes))])
        reached = np.flatnonzero(cumulative >= n)
        counts.append(int(reached[0]) + 1 if len(reached) else len(sizes))
    return float(np.mean(counts))


def bootstrap_fields(
    fields: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """``statistic`` of all objects, and its percentile interval from resampling whole fields.

    Each resample draws as many fields as were acquired, with replacement, and
    pools their objects; the interval is the ``confidence`` percentile range
    of the statistic over ``n_boot`` resamples. Fields without objects take
    part in the resampling (they were acquired) and contribute no value. NaN
    bounds when fewer than two fields hold objects.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    values = [np.asarray(f, dtype=float) for f in fields]
    pooled = np.concatenate(values) if values else np.array([])
    point = float(statistic(pooled)) if len(pooled) else float("nan")
    if sum(len(f) > 0 for f in values) < 2:
        return point, float("nan"), float("nan")
    draws = np.empty(n_boot)
    k = len(values)
    for b in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        sample = np.concatenate([values[i] for i in chosen])
        draws[b] = statistic(sample) if len(sample) else np.nan
    defined = draws[np.isfinite(draws)]
    if len(defined) < 2:
        return point, float("nan"), float("nan")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(defined, [tail, 100 - tail])
    return point, float(low), float(high)


__all__ = ["bootstrap_fields", "cohen_d", "field_sample", "fields_holding", "jcs_effect_size"]
