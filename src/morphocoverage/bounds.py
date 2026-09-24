"""Uncertainty bounds for probabilities estimated from a few FOVs.

A leave-one-FOV-out rate of zero after two FOVs says little: two homogeneous
fields cannot show that a third will add nothing. ``upper_bound`` turns ``x``
novel FOVs out of ``k`` into a one-sided Clopper-Pearson upper confidence
bound on the probability that another FOV is novel, which stays high until
many FOVs have been seen (0.78 for 0 of 2, 0.45 for 0 of 5, 0.14 for 0 of 20,
at 95%).

The bound treats the ``k`` held-out outcomes as independent Bernoulli trials.
They are not quite: each FOV is judged against the others, and neighbouring
FOVs can be correlated, so the bound is approximate. It is the exact binomial
bound, not an asymptotic one, so it stays valid-sized for small ``k``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def upper_bound(successes: int, trials: int, confidence: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial probability.

    ``stats.beta.ppf(confidence, successes + 1, trials - successes)``; 1 when
    every trial succeeded. ``trials`` must be positive.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if successes == trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def extra_trials_for_bound(
    successes: int,
    trials: int,
    target: float,
    confidence: float,
    *,
    keep_rate: bool,
    limit: int = 100_000,
) -> int | None:
    """Fewest further trials after which ``upper_bound`` is at most ``target``.

    ``keep_rate=False`` assumes none of the further trials succeeds, the
    fewest trials that could possibly suffice. ``keep_rate=True`` assumes
    successes keep arriving at the observed rate ``successes / trials``; then
    the bound never falls below ``target`` if that rate is at least
    ``target``, and ``None`` is returned, as it is when ``limit`` further
    trials are not enough.
    """
    upper_bound(successes, trials, confidence)  # validates the counts and confidence
    if not 0 < target < 1:
        raise ValueError("target must be a probability between 0 and 1")
    rate = successes / trials
    if keep_rate and rate >= target:
        return None
    total = trials + np.arange(limit + 1)
    hits = np.round(rate * total) if keep_rate else np.full(len(total), successes, dtype=float)
    hits = np.minimum(hits, total - 1)  # all-success prefixes have bound 1 anyway
    bounds = stats.beta.ppf(confidence, hits + 1, total - hits)
    bounds[hits >= total] = 1.0
    if successes == trials:
        bounds[0] = 1.0
    met = np.flatnonzero(bounds <= target)
    return int(met[0]) if len(met) else None


def check_confidence(confidence: object) -> float:
    """A confidence level strictly between 0 and 1, as a float."""
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise TypeError("confidence must be a number")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(confidence)


__all__ = ["check_confidence", "extra_trials_for_bound", "upper_bound"]
