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
