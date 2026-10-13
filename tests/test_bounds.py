"""Tests for the Clopper-Pearson upper bound."""

import numpy as np
import pytest
from _helpers_retrospective import BY_WELL, _repeat_curve
from scipy import stats

from morphocoverage.bounds import (
    check_confidence,
    extra_trials_for_bound,
    interval,
    lower_bound,
    upper_bound,
)
from morphocoverage.retrospective import protocol, protocol_reliability


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


def test_j9_e12_one_clopper_pearson_lower_bound_and_interval():
    assert lower_bound(0, 10, 0.95) == 0.0
    assert lower_bound(3, 10, 0.95) == pytest.approx(stats.beta.ppf(0.05, 3, 8))
    assert lower_bound(3, 10, 0.95) == pytest.approx(1 - upper_bound(7, 10, 0.95))
    low, high = interval(3, 10, 0.95)
    assert low == pytest.approx(stats.beta.ppf(0.025, 3, 8))
    assert high == pytest.approx(stats.beta.ppf(0.975, 4, 7))
    assert interval(10, 10, 0.95) == (pytest.approx(0.025 ** (1 / 10)), 1.0)


def test_b14_bounds_accept_numpy_integers():
    assert upper_bound(np.int64(0), np.int64(5), 0.95) == upper_bound(0, 5, 0.95)
    assert extra_trials_for_bound(
        np.int32(1), np.int64(10), 0.2, 0.95, keep_rate=False
    ) == extra_trials_for_bound(1, 10, 0.2, 0.95, keep_rate=False)


def test_b10_keep_rate_answer_is_a_count_after_which_the_bound_stays_met():
    for successes, trials, target in ((1, 10, 0.3), (2, 25, 0.2), (1, 7, 0.4), (3, 50, 0.12)):
        extra = extra_trials_for_bound(successes, trials, target, 0.95, keep_rate=True)
        assert extra is not None
        rate = successes / trials
        for more in range(extra, extra + 200):
            hits = successes + int(np.floor(rate * more + 0.5))
            assert upper_bound(min(hits, trials + more), trials + more, 0.95) <= target + 1e-12


def test_f5_protocol_interval_is_never_narrower_than_the_exact_one():
    def interval_of(n_fail):
        curve = {
            (f"R{u}", f"W{w}"): {4: 0.5 if u * 4 + w < n_fail else 0.95}
            for u in range(5)
            for w in range(4)
        }
        row = protocol_reliability(
            _repeat_curve(curve), BY_WELL, threshold=0.9, strata=["condition"]
        ).table.iloc[0]
        return row["share_low"], row["share_high"]

    all_low, all_high = interval_of(0)
    assert (all_low, all_high) == pytest.approx(interval(5, 5, 0.95))
    for n_fail in (1, 2, 4):
        low, high = interval_of(n_fail)
        assert 0 <= low <= all_low and high <= 1
    # The exact interval takes whole counts and is Clopper-Pearson's (issue #95).
    assert protocol._exact(3, 5) == pytest.approx(interval(3, 5, 0.95))
