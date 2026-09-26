"""The shared statistics: written once, checked against hand calculations."""

import numpy as np
import pytest

from morphocoverage.statistics import (
    bootstrap_fields,
    cohen_d,
    field_sample,
    fields_holding,
    jcs_effect_size,
)


def test_cohen_d_is_the_pooled_sd_effect_size():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])
    assert cohen_d(b, a) == pytest.approx(2.0 / np.sqrt(5 / 3))
    assert np.isnan(cohen_d(np.array([1.0]), a))
    assert np.isnan(cohen_d(np.full(3, 2.0), np.full(3, 2.0)))


def test_jcs_effect_size_is_the_difference_over_the_control_sd():
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treated = control + 2.0
    assert jcs_effect_size(treated, control) == pytest.approx(2.0 / control.std(ddof=1))
    assert np.isnan(jcs_effect_size(treated, np.array([1.0])))
    assert np.isnan(jcs_effect_size(treated, np.full(4, 2.0)))


def test_field_sample_takes_whole_fields_and_a_part_of_the_last():
    rng = np.random.default_rng(0)
    fields = [np.full(10, i, dtype=float) for i in range(5)]
    sample = field_sample(fields, 25, rng)
    assert len(sample) == 25
    counts = np.unique(sample, return_counts=True)[1]
    # two whole fields of 10 and five objects of a third
    assert sorted(counts.tolist()) == [5, 10, 10]


def test_fields_holding_counts_the_fields_that_reach_n():
    rng = np.random.default_rng(0)
    fields = [np.zeros(10)] * 6
    assert fields_holding(fields, 25, rng) == 3.0
    assert fields_holding(fields, 1000, rng) == 6.0  # never reached: every field


def test_bootstrap_fields_brackets_the_point_and_needs_two_fields():
    rng = np.random.default_rng(0)
    fields = [rng.normal(i, 0.1, 20) for i in range(6)]
    point, low, high = bootstrap_fields(fields, np.median, n_boot=300, rng=rng)
    assert point == pytest.approx(np.median(np.concatenate(fields)))
    assert low <= point <= high and high - low > 0.5  # field-to-field spread shows
    single = [fields[0], np.array([])]
    one, low_one, high_one = bootstrap_fields(single, np.median, n_boot=50, rng=rng)
    assert np.isfinite(one) and np.isnan(low_one) and np.isnan(high_one)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_fields(fields, np.median, n_boot=10, rng=rng, confidence=1.5)
