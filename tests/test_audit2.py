"""Regressions for the second function audit (docs/audit/README.md), one test per finding."""

import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

from morphocoverage import statistics
from morphocoverage._wording import count, duration, fields, number, percent, rows_of
from morphocoverage.bounds import extra_trials_for_bound, interval, lower_bound, upper_bound

# --- Shared foundations -------------------------------------------------------


def test_j13_shared_statistics_are_defined_once_and_re_exported():
    from morphocoverage.outcomes import srs_relative_rmse
    from morphocoverage.retrospective.effort import nested_variance_components
    from morphocoverage.strategies.rare_planning import first_detection_fovs, fovs_to_detect
    from morphocoverage.strategies.scalar_precision import sd_upper_bound_factor

    assert srs_relative_rmse is statistics.srs_relative_rmse
    assert nested_variance_components is statistics.nested_variance_components
    assert fovs_to_detect is statistics.fovs_to_detect
    assert first_detection_fovs is statistics.first_detection_fovs
    assert sd_upper_bound_factor is statistics.sd_upper_bound_factor


def test_j11_j19_sampling_unit_and_schema_import_without_cycles():
    code = (
        "import sys, morphocoverage.sampling_unit, morphocoverage.schema\n"
        "assert not [m for m in sys.modules if m.startswith('morphocoverage.retrospective')]\n"
        "import morphocoverage.schema as s\n"
        "from morphocoverage.data_check import validate_input_tables\n"
        "assert s.validate_input_tables is validate_input_tables\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_h4_constant_feature_has_no_spread_and_no_effect_size():
    constant = np.full(20, 0.1)
    assert statistics.spread_is_zero(constant.std(ddof=1), constant)
    assert np.isnan(statistics.cohen_d(constant, constant + 1e-17))
    assert np.isnan(statistics.jcs_effect_size(constant * 2, constant))
    assert statistics.mean_is_zero(np.mean([1e-17, -1e-17 + 1e-30]), np.array([1.0, -1.0]))


def test_e8_field_sampling_refuses_more_objects_than_there_are():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="asked for 30 objects"):
        statistics.field_sample([np.zeros(10), np.zeros(10)], 30, rng)
    with pytest.raises(ValueError, match="asked for 5 objects"):
        statistics.field_sample([], 5, rng)


def test_c8_jcb_count_is_not_raised_by_floating_point_error():
    assert (0.07 / 0.01) ** 2 > 49
    assert statistics.jcb_required_objects(0.07, 0.01) == 49
    assert statistics.jcb_required_objects(0.071, 0.01) == 51


def test_c9_sd_bound_validates_its_arguments_and_accepts_numpy_ints():
    assert statistics.sd_upper_bound_factor(np.int64(9)) == statistics.sd_upper_bound_factor(9)
    with pytest.raises(ValueError, match="confidence"):
        statistics.sd_upper_bound_factor(9, 1.5)
    with pytest.raises(ValueError, match="df"):
        statistics.sd_upper_bound_factor(2.5)


def test_e13_srs_relative_rmse_refuses_impossible_inputs():
    assert statistics.srs_relative_rmse(4.0, 2.0, 10, 100) == pytest.approx(np.sqrt(0.4 * 0.9) / 2)
    for arguments in ((4.0, 2.0, 0, 100), (4.0, 2.0, 200, 100), (-1.0, 2.0, 10, 100)):
        with pytest.raises(ValueError):
            statistics.srs_relative_rmse(*arguments)
    with pytest.raises(ValueError, match="zero mean"):
        statistics.srs_relative_rmse(4.0, 0.0, 10, 100)


def test_e5_nested_bootstrap_keeps_fields_within_a_drawn_well():
    # Two wells whose fields are identical within each well: resampling wells only
    # can give just three means (both of one well, or one of each).
    wells = [[np.full(5, 1.0), np.full(5, 1.0)], [np.full(5, 3.0), np.full(5, 3.0)]]
    draws = set()
    for seed in range(20):
        point, low, high = statistics.bootstrap_nested(
            wells, np.mean, n_boot=200, rng=np.random.default_rng(seed)
        )
        draws.update({low, high})
        assert point == 2.0
    assert draws <= {1.0, 2.0, 3.0}


def test_e5_nested_bootstrap_with_one_well_is_the_field_bootstrap():
    fields_of_well = [np.arange(5.0) + i for i in range(6)]
    nested = statistics.bootstrap_nested(
        [fields_of_well], np.mean, n_boot=300, rng=np.random.default_rng(1)
    )
    flat = statistics.bootstrap_fields(
        fields_of_well, np.mean, n_boot=300, rng=np.random.default_rng(1)
    )
    assert nested == flat


def test_d7_design_effect_and_cluster_size():
    assert statistics.cluster_size(np.array([10, 10, 0])) == 10.0
    assert statistics.cluster_size(np.array([5, 15])) == pytest.approx((25 + 225) / 20)
    assert statistics.design_effect(1.0, 1.0, 10.0) == pytest.approx(5.5)
    assert statistics.design_effect(-1.0, 2.0, 10.0) == pytest.approx(1.0)
    assert statistics.design_effect(0.0, 0.0, 10.0) is None


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


def test_j16_wording_helpers():
    assert percent(0.003) == "0.3%" and percent(0.5) == "50%" and percent(0) == "0%"
    assert number(7.0) == "7" and number(7.25) == "7.2" and number(1234567) == "1234567"
    assert fields(1) == "1 field" and fields(2.5) == "2.5 fields"
    assert fields(1.047) == "1 field" and fields(0.96) == "1 field" and fields(1.06) == "1.1 fields"
    assert count(1, "batch", "batches") == "1 batch" and count(3, "batch", "batches") == "3 batches"
    assert duration(20) == "20 s" and duration(600) == "10 min" and duration(5400) == "1.5 h"


def test_j16_rows_of_matches_missing_values():
    import pandas as pd

    frame = pd.DataFrame({"condition": ["A", None, "B"], "value": [1, 2, 3]})
    assert rows_of(frame, {"condition": None})["value"].tolist() == [2]
    assert rows_of(frame, {"condition": "B"})["value"].tolist() == [3]
