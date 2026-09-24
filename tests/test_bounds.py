"""Tests for the Clopper-Pearson upper bound."""

import pytest
from scipy import stats

from morphocoverage.bounds import check_confidence, upper_bound


@pytest.mark.parametrize("x, k", [(0, 1), (0, 2), (1, 5), (3, 10), (7, 20)])
def test_matches_scipys_exact_binomial_interval(x, k):
    # A one-sided 95% upper bound is the upper end of the two-sided 90% interval.
    expected = stats.binomtest(x, k).proportion_ci(0.90, method="exact").high
    assert upper_bound(x, k, 0.95) == pytest.approx(expected)


def test_zero_successes_has_the_closed_form():
    for k in (1, 2, 5, 20):
        assert upper_bound(0, k, 0.95) == pytest.approx(1 - 0.05 ** (1 / k))
    assert upper_bound(4, 4, 0.95) == 1.0


@pytest.mark.parametrize(
    "args", [(0, 0, 0.95), (3, 2, 0.95), (-1, 2, 0.95), (0, 2, 1.0), (0, True, 0.95)]
)
def test_invalid_input_is_refused(args):
    with pytest.raises((TypeError, ValueError)):
        upper_bound(*args)


def test_confidence_must_be_a_probability():
    assert check_confidence(0.9) == 0.9
    for bad in (0, 1, 1.5, True, "0.9"):
        with pytest.raises((TypeError, ValueError)):
            check_confidence(bad)


def test_extra_trials_reach_the_target_and_not_one_fewer():
    from morphocoverage.bounds import extra_trials_for_bound

    # No success in 2 trials: 29 in all are needed for a 95% bound below 10%.
    assert extra_trials_for_bound(0, 2, 0.1, 0.95, keep_rate=False) == 27
    for x, k, target in ((0, 2, 0.1), (2, 28, 0.1), (3, 10, 0.2), (2, 2, 0.5)):
        extra = extra_trials_for_bound(x, k, target, 0.95, keep_rate=False)
        assert upper_bound(x, k + extra, 0.95) <= target
        if extra:
            assert upper_bound(x, k + extra - 1, 0.95) > target


def test_extra_trials_at_the_current_rate():
    from morphocoverage.bounds import extra_trials_for_bound

    # A rate already at or above the target can never give a bound below it.
    assert extra_trials_for_bound(3, 20, 0.1, 0.95, keep_rate=True) is None
    assert extra_trials_for_bound(2, 20, 0.1, 0.95, keep_rate=True) is None
    extra = extra_trials_for_bound(1, 20, 0.1, 0.95, keep_rate=True)
    total = 20 + extra
    assert upper_bound(round(total / 20), total, 0.95) <= 0.1
    # Keeping the rate never needs fewer trials than assuming no further success.
    assert extra >= extra_trials_for_bound(1, 20, 0.1, 0.95, keep_rate=False)
    # Already below the target: nothing more is needed.
    assert extra_trials_for_bound(0, 40, 0.1, 0.95, keep_rate=True) == 0


@pytest.mark.parametrize("target", [0, 1, 1.5, -0.1])
def test_extra_trials_need_a_probability_target(target):
    from morphocoverage.bounds import extra_trials_for_bound

    with pytest.raises(ValueError):
        extra_trials_for_bound(0, 5, target, 0.95, keep_rate=False)
