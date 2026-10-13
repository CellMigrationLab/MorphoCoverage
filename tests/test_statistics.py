"""The shared statistics: written once, checked against hand calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import CONTROL, _well_tables, _well_tables_with_empty_fields
from _helpers_strategies import _area_tables, _ordered_acquisition, _run, _x_run
from scipy import stats

from morphocoverage import ScalarPrecisionStrategy, statistics
from morphocoverage.plots import plots_of_data
from morphocoverage.sampling_unit import (
    condition_contrast,
    feature_ranking,
    fov_variability,
    object_rules,
)
from morphocoverage.sampling_unit.object_rules import object_rule_check
from morphocoverage.statistics import (
    _interval,
    bootstrap_fields,
    bootstrap_nested,
    cluster_size,
    cohen_d,
    design_effect,
    field_sample,
    fields_holding,
    jcb_required_objects,
    jcs_effect_size,
    nested_variance_components,
    object_weighted_mean_se,
    paired_t,
    repeat_effect,
    repeat_values,
)
from morphocoverage.strategies import FOVBootstrapStrategy, stabilization
from morphocoverage.strategies.rare_planning import first_detection_fovs, fovs_to_detect
from morphocoverage.strategies.scalar_precision import required_objects


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
    with pytest.raises(ValueError, match="asked for 1000 objects"):
        fields_holding(fields, 1000, rng)  # never reached: refused, not every field


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


def test_bootstrap_nested_widens_for_a_unit_shift_and_equals_fields_for_one_unit():
    from morphocoverage.statistics import bootstrap_fields, bootstrap_nested

    rng = np.random.default_rng(0)
    low = [rng.normal(0, 0.1, 20) for _ in range(6)]
    high = [rng.normal(1, 0.1, 20) for _ in range(6)]
    point, a, b = bootstrap_nested([low, high], np.mean, n_boot=300, rng=np.random.default_rng(1))
    _, c, d = bootstrap_fields([*low, *high], np.mean, n_boot=300, rng=np.random.default_rng(1))
    assert point == pytest.approx(np.mean(np.concatenate([*low, *high])))
    assert a < point < b and (b - a) > 1.5 * (d - c)  # the well-to-well shift widens it
    one = bootstrap_nested([low], np.mean, n_boot=100, rng=np.random.default_rng(2))
    assert one == bootstrap_fields(low, np.mean, n_boot=100, rng=np.random.default_rng(2))
    assert np.isnan(bootstrap_nested([[low[0]]], np.mean, n_boot=10, rng=rng)[1])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_nested([low, high], np.mean, n_boot=10, rng=rng, confidence=1.5)


def test_benjamini_hochberg_adjusts_in_rank_order_and_keeps_nan():
    from morphocoverage.statistics import benjamini_hochberg

    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert q[:3].tolist() == pytest.approx([0.03, 0.04, 0.04]) and np.isnan(q[3])
    assert benjamini_hochberg(np.array([0.5])).tolist() == [0.5]
    assert np.isnan(benjamini_hochberg(np.array([np.nan]))).all()
    assert benjamini_hochberg(np.array([0.9, 0.95])).max() <= 1.0


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
    # can give just three means (both of one well, or one of each), so each end of
    # the percentile interval is 1, 2 or 3 before the small-sample widening (audit 3,
    # E-2), which multiplies its distance from the point by one factor.
    wells = [[np.full(5, 1.0), np.full(5, 1.0)], [np.full(5, 3.0), np.full(5, 3.0)]]
    widen = np.sqrt(2) * stats.t.ppf(0.975, 1) / stats.norm.ppf(0.975)
    ends = set()
    for seed in range(20):
        point, low, high = statistics.bootstrap_nested(
            wells, np.mean, n_boot=200, rng=np.random.default_rng(seed)
        )
        assert point == 2.0
        ends.update({round((point - low) / widen, 9), round((high - point) / widen, 9)})
    assert ends <= {0.0, 1.0}


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


def test_c8_c9_required_objects_delegates_to_the_shared_rounded_count():
    assert required_objects(0.07, 0.01) == statistics.jcb_required_objects(0.07, 0.01) == 49
    for bad in (1.0, 1.5, 0.0):
        with pytest.raises(ValueError, match="confidence"):
            statistics.sd_upper_bound_factor(5, bad)


def _skewed_fields():
    """Crowded fields of one state, sparse fields of another, and empty fields."""
    rng = np.random.default_rng(3)
    sizes = [3, 5, 80, 120, 4, 6, 150, 2, 0, 0, 0, 0]
    means = [10, 10, 13, 13, 10, 10, 13, 10, 10, 10, 10, 10]
    return [rng.normal(m, 2, n) for n, m in zip(sizes, means, strict=True)]


def test_c14_two_field_level_estimands_named_apart_and_the_size_warning():
    fields = _skewed_fields()
    sums = np.array([f.sum() for f in fields])
    counts = np.array([len(f) for f in fields], dtype=float)
    ratio_mean, ratio_se, filled = statistics.object_weighted_mean_se(sums, counts)
    means = np.array([f.mean() for f in fields if len(f)])
    field_mean, field_se, n_means = statistics.field_weighted_mean_se(means)
    assert filled == n_means == 8
    assert ratio_mean == pytest.approx(sums.sum() / counts.sum())
    residual = np.sum((sums - ratio_mean * counts) ** 2)
    assert ratio_se == pytest.approx(math.sqrt(8 / 7 * residual) / counts.sum())
    assert field_mean == pytest.approx(means.mean())
    assert field_se == pytest.approx(means.std(ddof=1) / math.sqrt(8))
    # Equal fields: both estimands and their SEs agree.
    equal = [np.array([1.0, 2.0, 3.0]) + j for j in range(4)]
    ratio = statistics.object_weighted_mean_se(np.array([f.sum() for f in equal]), np.full(4, 3.0))
    assert ratio[:2] == pytest.approx(
        statistics.field_weighted_mean_se(np.array([f.mean() for f in equal]))[:2]
    )

    # object_rule_check's field rule uses the field-weighted estimand.
    objects = pd.DataFrame(
        [{"image_file": f"f{j}", "area": v} for j, f in enumerate(fields) for v in f]
    )
    fovs = pd.DataFrame({"image_file": [f"f{j}" for j in range(len(fields))]})
    [row] = object_rule_check(
        objects, fovs, feature="area", targets=[0.05], n_draws=20
    ).table.to_dict("records")
    per_field_sd = field_se * math.sqrt(8) / ratio_mean
    assert row["fov_rule_fovs"] == statistics.jcb_required_objects(per_field_sd, 0.05)

    # ScalarPrecision (object-weighted ratio estimator) warns that size and phenotype relate.
    result = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert result.diagnostics["field_size_correlation"] > 0.5
    assert "field size and field mean are related" in result.diagnostics["field_size_warning"]
    rng = np.random.default_rng(1)
    independent = [rng.normal(10 + rng.normal(), 2, n) for n in rng.poisson(30, 20)]
    objects, fovs = _area_tables(independent)
    result = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert "field_size_warning" not in result.diagnostics
    assert "field_size_correlation" in result.diagnostics


def test_h10_pooled_repeats_resample_the_repeats_first():
    rng = np.random.default_rng(0)
    # Ten repeats far apart, each one well of tight fields.
    repeats = [[[rng.normal(3 * r, 0.1, 10) for _ in range(6)]] for r in range(10)]
    fields = [f for repeat in repeats for unit in repeat for f in unit]
    # Pooled repeats resample the repeats alone, each with all its fields (audit 3, H-2).
    units = [[f for unit in repeat for f in unit] for repeat in repeats]
    _, low, high = bootstrap_nested(units, np.mean, n_boot=400, rng=np.random.default_rng(1))
    _, f_low, f_high = bootstrap_fields(fields, np.mean, n_boot=400, rng=np.random.default_rng(1))
    # Repeat means have SD 9: the mean of 10 has SE 2.9, a 95% interval about 11 wide.
    # Resampling the 60 fields instead treats them as independent: about sqrt(6) narrower.
    assert high - low > 9 and high - low > 2 * (f_high - f_low)
    spec = [("ctrl", f"r{r}", "w", 3, float(r), 0.2) for r in range(10)]
    objects, fovs = _well_tables(spec, per_field=5)
    title = plots_of_data(objects, fovs, features=["x"], n_boot=20)["x"][0].axes[0].figure.caption
    assert "resampling whole biological repeats with all their fields" in title


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


def test_c17_the_paper_descriptors_are_computed_once_in_statistics():
    assert stabilization.describe is statistics.describe
    assert object_rules.describe is statistics.describe
    assert stabilization.PAPER_DESCRIPTORS is statistics.PAPER_DESCRIPTORS


def test_e1_empty_wells_do_not_narrow_the_nested_interval():
    rng = np.random.default_rng(0)
    filled = [rng.normal(10, 2, 30) for _ in range(6)]
    empty = [np.array([]) for _ in range(3)]
    alone = bootstrap_fields(filled, np.mean, n_boot=400, rng=np.random.default_rng(1))
    with_empty = bootstrap_nested(
        [filled, empty, empty], np.mean, n_boot=400, rng=np.random.default_rng(1)
    )
    assert with_empty == alone
    assert with_empty[2] - with_empty[1] > 0


def test_e2_intervals_are_widened_for_few_units():
    draws = np.random.default_rng(0).normal(0, 1, 20_000)
    for units in (3, 5, 10):
        _, low, high = _interval(0.0, draws, units, 0.95)
        raw_low, raw_high = np.percentile(draws, [2.5, 97.5])
        t_over_z = stats.t.ppf(0.975, units - 1) / stats.norm.ppf(0.975)
        widen = math.sqrt(units / (units - 1)) * t_over_z
        assert low == pytest.approx(raw_low * widen) and high == pytest.approx(raw_high * widen)


def test_e10_the_ratio_se_docstring_compares_relative_spreads():
    doc = object_weighted_mean_se.__doc__
    assert "relative to the mean, 0.013" in doc and "0.044" in doc
    rng = np.random.default_rng(3)
    sizes = [3, 5, 80, 120, 4, 6, 150, 2, 0, 0, 0, 0]
    centres = [10, 10, 13, 13, 10, 10, 13, 10, 10, 10, 10, 10]
    fields = [rng.normal(m, 2, n) for n, m in zip(sizes, centres, strict=True)]
    sums = np.array([f.sum() for f in fields])
    counts = np.array([len(f) for f in fields], float)
    mean, se, _ = object_weighted_mean_se(sums, counts)
    assert se / mean == pytest.approx(0.013, abs=0.0005)
    filled = np.flatnonzero(counts > 0)
    picks = filled[np.random.default_rng(0).integers(0, len(filled), (20_000, len(filled)))]
    spread = np.std(sums[picks].sum(1) / counts[picks].sum(1)) / mean
    assert spread == pytest.approx(0.044, abs=0.003)


def test_e11_variance_components_refuse_designs_they_cannot_estimate():
    values = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match="do not nest"):
        nested_variance_components(
            values, [np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0, 1])]
        )
    with pytest.raises(ValueError, match="finite"):
        nested_variance_components(np.array([1.0, np.inf, 2.0]), [np.array([0, 0, 1])])
    with pytest.raises(ValueError, match="cannot be told apart"):
        nested_variance_components(values, [np.arange(8)])
    assert "can be negative" in nested_variance_components.__doc__


def test_e12_cluster_size_weights_crowded_fields():
    assert cluster_size(np.array([2, 0, 4])) == pytest.approx((4 + 16) / 6)
    assert cluster_size(np.array([5, 5, 5])) == 5
    assert math.isnan(cluster_size(np.array([0, 0])))


def test_e13_the_nested_bootstrap_resamples_the_top_level_only():
    # However the fields of a unit are split into lower levels, only whole top units
    # are drawn, so the interval is the same.
    rng = np.random.default_rng(0)
    units = [[rng.normal(u, 1, 20) for _ in range(4)] for u in range(5)]
    merged = [[np.concatenate(unit)] for unit in units]
    first = bootstrap_nested(units, np.mean, n_boot=300, rng=np.random.default_rng(4))
    second = bootstrap_nested(merged, np.mean, n_boot=300, rng=np.random.default_rng(4))
    assert first == pytest.approx(second)


def test_e14_small_helpers_validate_their_arguments():
    rng = np.random.default_rng(0)
    fields = [np.arange(3.0), np.arange(4.0)]
    assert field_sample(fields, 0, rng).size == 0
    assert fields_holding(fields, 0, rng) == 0
    with pytest.raises(ValueError):
        field_sample(fields, -1, rng)
    with pytest.raises(ValueError):
        fields_holding(fields, -1, rng)
    for arguments in ((float("nan"), 0.05), (0.1, 0.0), (-0.2, 0.05), (0.1, -0.05)):
        with pytest.raises(ValueError):
            jcb_required_objects(*arguments)


def test_h17_contrast_and_ranking_share_one_test_and_variability_the_design_effect():
    spec = [
        (c, "r1", f"{c}{i}", 5, 2.0 if c == "drug" else 0.0, 0.2)
        for c in ("ctrl", "drug")
        for i in range(4)  # 4 wells a side: a floor of 2/70, so both run the test
    ]
    objects, fovs = _well_tables_with_empty_fields(spec)
    contrast = condition_contrast(objects, fovs, feature="x", control=CONTROL, object_counts=[])
    ranking = feature_ranking(objects, fovs, features=["x"], control=CONTROL)
    columns = ["d_fields", "permutation_unit", "distinct_assignments", "p_value", "p_floor"]
    for column in columns:
        assert contrast.table[column].iloc[0] == ranking.tests[column].iloc[0]
    row = fov_variability(objects, fovs, feature="x").groups.iloc[0]
    sizes = objects.groupby("position_id").size().to_numpy()
    assert row["cluster_size"] == pytest.approx(cluster_size(sizes))
    assert row["design_effect"] == pytest.approx(
        design_effect(row["v_fov"], row["v_obs"], row["cluster_size"])
    )


def test_p4_4_the_bootstrap_stopping_se_is_widened_for_few_fields():
    assert statistics.small_sample_widening(5) == math.sqrt(5 / 4) * stats.t.ppf(
        0.975, 4
    ) / stats.norm.ppf(0.975)
    rng = np.random.default_rng(2)
    values = [list(rng.normal(10 + rng.normal(0, 0.5), 1, 15)) for _ in range(4)] + [[]]
    objects, fovs = _ordered_acquisition(values)
    strategy = FOVBootstrapStrategy("centroid", 0.5, None, 400, 0, 2, features=("x",))
    [result] = _x_run(objects, fovs, [strategy])
    diagnostics = result.evidence.diagnostics
    # 4 of the 5 fields hold objects: those are the resampled units of the widening.
    assert diagnostics["resampling_units"] == 4
    assert diagnostics["small_sample_widening"] == statistics.small_sample_widening(4)
    assert diagnostics["small_sample_widening"] > 1.8
    se = result.evidence.value("fov_bootstrap_centroid_max_se")
    assert diagnostics["max_component_se"] == se
    # The code multiplies by 1 / sqrt(2 (B - 1)): equal to the division up to the last bit.
    assert diagnostics["monte_carlo_se"] == pytest.approx(se / math.sqrt(2 * 399), rel=1e-12)
    # The first two fields no longer stop the rule at a tolerance the raw SE met.
    two = _ordered_acquisition([[10.0, 10.2, 9.9], [10.1, 10.3, 10.0]])
    [early] = _x_run(*two, [FOVBootstrapStrategy("centroid", 0.1, None, 400, 0, 2)])
    raw = early.evidence.value("fov_bootstrap_centroid_max_se") / (
        statistics.small_sample_widening(2)
    )
    assert raw < 0.1 and early.rule_decision == "continue"


def test_d5_the_one_occurrence_closed_form_equals_the_plan_everywhere():
    grid = np.round(np.arange(0.01, 0.99, 0.01), 2)
    for p in grid:
        for beta in grid:
            assert first_detection_fovs(p, beta) == fovs_to_detect(p, beta, 1)


@pytest.mark.parametrize("values", ["normalised", "raw"])
def test_the_t_interval_of_three_repeats_covers_its_level(values):
    rng = np.random.default_rng(11)
    truth, covered, reps = 0.3, 0, 4000
    for _ in range(reps):
        effect = repeat_effect(values, rng.normal(truth, 0.4, size=3))
        target = np.exp(truth) if values == "normalised" else truth
        covered += effect["ci_low"] <= target <= effect["ci_high"]
    # 95% nominal; the Monte Carlo SE of the share is about 0.0035.
    assert 0.935 <= covered / reps <= 0.965


def test_one_rule_for_the_values_the_paired_test_compares():
    values, ratios, fold = repeat_values([2.0, 4.0], [1.0, 2.0])
    assert values == "normalised" and fold == pytest.approx(2.0)
    assert np.allclose(ratios, np.log(2.0))
    values, _, mean = repeat_values([1.0, -1.0], [0.5, 0.5])
    assert values == "raw" and mean == pytest.approx(-0.5)
    assert np.isnan(paired_t(np.array([0.1, 0.1, 0.1])))


def test_benjamini_hochberg_keeps_nan_and_matches_the_definition():
    from morphocoverage.statistics import benjamini_hochberg

    p = np.array([0.01, np.nan, 0.04, 0.03, 1.0])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    assert np.allclose(q[[0, 2, 3, 4]], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 1.0])
