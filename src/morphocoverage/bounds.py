"""Uncertainty bounds for probabilities estimated from a few FOVs.

A leave-one-FOV-out rate of zero after two FOVs says little: two homogeneous
fields cannot show that a third will add nothing. ``upper_bound`` turns ``x``
novel FOVs out of ``k`` into a one-sided Clopper-Pearson upper confidence
bound on the probability that another FOV is novel, which stays high until
many FOVs have been seen (0.78 for 0 of 2, 0.45 for 0 of 5, 0.14 for 0 of 20,
at 95%).

For ``k`` independent Bernoulli trials this is the exact binomial bound,
valid for small ``k``. Leave-one-FOV-out outcomes are not independent: each
FOV is judged against the others, so the outcomes share almost all their
reference data, and neighbouring FOVs can be correlated. Applied to them the
confidence level is **nominal**: the bound is an approximate upper bound
whose actual coverage must be checked empirically
(``morphocoverage.calibration``), and it should be reported as a nominal
bound rather than as "at most X with 95% confidence". For zero successes in
``k`` independent, representative FOVs (a state no FOV has shown) it is exact.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._checks import is_integer, is_number


def _check_counts(successes: int, trials: int, confidence: float) -> tuple[int, int]:
    """Validated binomial counts (numpy integers accepted) as Python ints."""
    if not is_integer(trials) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if not is_integer(successes):
        raise TypeError("successes must be an integer")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return int(successes), int(trials)


def upper_bound(successes: int, trials: int, confidence: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial probability.

    ``stats.beta.ppf(confidence, successes + 1, trials - successes)``; 1 when
    every trial succeeded. ``trials`` must be positive.
    """
    successes, trials = _check_counts(successes, trials, confidence)
    if successes == trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def lower_bound(successes: int, trials: int, confidence: float) -> float:
    """One-sided Clopper-Pearson lower bound on a binomial probability.

    ``stats.beta.ppf(1 - confidence, successes, trials - successes + 1)``; 0
    when no trial succeeded (the mirror of ``upper_bound``).
    """
    successes, trials = _check_counts(successes, trials, confidence)
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(1 - confidence, successes, trials - successes + 1))


def interval(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval: each end a one-sided bound at ``(1 + confidence) / 2``.

    Exact, and conservative, for independent trials; at 0 or ``trials``
    successes the closed end is 0 or 1 and the other end keeps half the
    error rate, as the equal-tailed interval does.
    """
    _check_counts(successes, trials, confidence)
    level = (1 + confidence) / 2
    return lower_bound(successes, trials, level), upper_bound(successes, trials, level)


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
    successes, trials = _check_counts(successes, trials, confidence)
    if not is_integer(limit) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if not 0 < target < 1:
        raise ValueError("target must be a probability between 0 and 1")
    rate = successes / trials
    if keep_rate and rate >= target:
        return None
    total = trials + np.arange(int(limit) + 1)
    if keep_rate:
        # The further trials keep the rate: the successes are the observed ones plus
        # the rate's share of the new trials, rounded half up so they never decrease.
        hits = successes + np.floor(rate * (total - trials) + 0.5)
    else:
        hits = np.full(len(total), successes, dtype=float)
    hits = np.minimum(hits, total - 1)  # all-success prefixes have bound 1 anyway
    bounds = stats.beta.ppf(confidence, hits + 1, total - hits)
    if successes == trials:
        bounds[0] = 1.0
    # A bound can wobble with the rounding of ``hits``; the answer is a number of
    # trials after which it stays at most ``target``, so every later bound must be too.
    above = np.flatnonzero(bounds > target)
    first = 0 if not len(above) else int(above[-1]) + 1
    return first if first <= limit else None


def check_confidence(confidence: object) -> float:
    """A confidence level strictly between 0 and 1, as a float."""
    if not is_number(confidence):
        raise TypeError("confidence must be a number")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(confidence)


__all__ = ["check_confidence", "extra_trials_for_bound", "interval", "lower_bound", "upper_bound"]
