"""Regressions for the third audit (docs/audit/README.md): the shared foundations."""

import numpy as np
import pytest
from scipy import stats

from morphocoverage import statistics
from morphocoverage._checks import check_seed
from morphocoverage._seeds import group_identity, group_key, split_seed
from morphocoverage._wording import count, listing, percent, plural
from morphocoverage.bounds import check_confidence


def test_a6_e19_percent_never_reads_0_or_100_for_a_share_between_and_rounds_half_up():
    assert percent(199 / 200) == "99.5%" and percent(0.999) == "99.9%"
    assert percent(1 / 5000) == "0.02%" and percent(0.003) == "0.3%"
    assert percent(0.125) == "13%" and percent(0.025) == "3%"
    assert percent(0) == "0%" and percent(1) == "100%"
    assert percent(float("nan")) == "undefined"


def test_j7_one_plural_helper():
    assert [plural(w) for w in ("nucleus", "colony", "branch", "patch", "cell")] == [
        "nuclei",
        "colonies",
        "branches",
        "patches",
        "cells",
    ]
    assert plural("organoids") == "organoids" and plural("fov_summary") == "field summaries"
    assert count(1, "nucleus") == "1 nucleus" and count(3, "colony") == "3 colonies"
    assert listing(["a", "b", "c"]) == "a, b and c"
    assert listing(list("abcdefg"), limit=3) == "a, b, c and 4 more"


def test_a10_b11_group_identity_does_not_depend_on_the_numpy_type():
    assert group_identity({"repeat": np.int64(1)}) == group_identity({"repeat": 1})
    # Audit 4 (owner decision on P9-2): a retyped label is the same group for seeding.
    assert group_identity({"repeat": 1}) == group_identity({"repeat": "1"})
    assert group_identity({"repeat": 1}) != group_identity({"repeat": 2})


def test_j1_group_key_tells_every_group_apart():
    a = {"condition": "plate24_wellI03_KO", "repeat": "2024_01_15"}
    b = {"condition": "plate15_wellP15_siGENE", "repeat": "2024_01_15"}
    assert group_identity(a) == group_identity(b)  # a real crc32 collision
    assert group_key(a) != group_key(b)
    assert group_key({"repeat": np.int64(1)}) == group_key({"repeat": 1})


def test_f6_split_seeds_of_neighbouring_random_states_do_not_overlap():
    first = {split_seed(0, s) for s in range(20)}
    second = {split_seed(1, s) for s in range(20)}
    assert not first & second and len(first) == 20
    assert split_seed(np.int64(3), np.int64(2)) == split_seed(3, 2)


def test_b8_seeds_and_b9_confidence_accept_numpy_and_refuse_negatives():
    assert check_seed(np.int64(7)) == 7
    with pytest.raises(ValueError, match="non-negative"):
        check_seed(-1)
    assert check_confidence(np.float32(0.95)) == pytest.approx(0.95)


def test_e2_intervals_are_widened_for_the_number_of_units():
    # One field per well, no objects apart: the percentile half-widths times
    # sqrt(k / (k - 1)) t_{k-1} / z.
    rng = np.random.default_rng(0)
    wells = [[np.array([x])] for x in rng.normal(size=6)]
    point, low, high = statistics.bootstrap_nested(
        wells, np.mean, n_boot=2000, rng=np.random.default_rng(1)
    )
    draws_rng = np.random.default_rng(1)
    pooled = [w[0] for w in wells]
    draws = [
        np.mean(np.concatenate([pooled[i] for i in draws_rng.integers(0, 6, 6)]))
        for _ in range(2000)
    ]
    p_low, p_high = np.percentile(draws, [2.5, 97.5])
    widen = np.sqrt(6 / 5) * stats.t.ppf(0.975, 5) / stats.norm.ppf(0.975)
    assert low == pytest.approx(point - (point - p_low) * widen)
    assert high == pytest.approx(point + (p_high - point) * widen)


def test_e1_a_well_of_empty_fields_is_not_a_unit():
    filled = [np.arange(10.0) + i for i in range(4)]
    alone = statistics.bootstrap_nested([filled], np.mean, n_boot=300, rng=np.random.default_rng(2))
    with_empty = statistics.bootstrap_nested(
        [filled, [np.array([]), np.array([])]], np.mean, n_boot=300, rng=np.random.default_rng(2)
    )
    assert with_empty == alone and with_empty[2] > with_empty[1]


def test_e11_variance_components_refuse_codes_that_do_not_nest_and_non_finite_values():
    values = np.random.default_rng(0).normal(size=24)
    wells = np.repeat([0, 1], 12)
    fields_within = np.tile(np.repeat([0, 1, 2], 4), 2)
    with pytest.raises(ValueError, match="do not nest"):
        statistics.nested_variance_components(values, [wells, fields_within])
    with pytest.raises(ValueError, match="finite"):
        statistics.nested_variance_components(np.r_[values[:-1], np.inf], [wells])
    with pytest.raises(ValueError, match="cannot be told apart"):
        statistics.nested_variance_components(values, [wells, wells])


def test_e14_field_sampling_and_jcb_count_edges():
    rng = np.random.default_rng(0)
    assert len(statistics.field_sample([], 0, rng)) == 0
    assert statistics.fields_holding([np.zeros(3)], 0, rng) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        statistics.field_sample([np.zeros(3)], -1, rng)
    for cv, target in ((-0.2, 0.05), (0.1, -0.05), (float("nan"), 0.05), (0.1, 0.0)):
        with pytest.raises(ValueError):
            statistics.jcb_required_objects(cv, target)


def test_c17_descriptors_live_in_statistics():
    from morphocoverage.strategies import stabilization

    assert stabilization.describe is statistics.describe
    assert statistics.describe(np.array([1.0, 2.0, 3.0, 4.0]))["iqr"] == pytest.approx(1.5)
