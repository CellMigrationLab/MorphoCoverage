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


def check_confidence(confidence: object) -> float:
    """A confidence level strictly between 0 and 1, as a float."""
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise TypeError("confidence must be a number")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(confidence)


__all__ = ["check_confidence", "upper_bound"]
