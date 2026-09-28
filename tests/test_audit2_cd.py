"""Regressions for the second audit, groups C (scalar precision, JCB planning, descriptor
baselines) and D (distances, incidence coverage, transforms), one test per finding."""

import math
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from morphocoverage import (
    DescriptorPlateauStrategy,
    ObjectiveConfig,
    PlannedObjectCountStrategy,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    plan_objects_to_average,
    run_strategies,
    statistics,
    validate_input_tables,
)
from morphocoverage.distances import frechet_distance
from morphocoverage.multivariate_outcomes import CentroidError, FrechetError
from morphocoverage.sampling_unit.object_rules import object_rule_check
from morphocoverage.strategies import (
    DescriptorSpreadEstimator,
    FixedAreaStrategy,
    FOVBootstrapEstimator,
    ScalarPrecisionEstimator,
    _descriptor_resamples,
)
from morphocoverage.strategies.incidence_coverage import (
    chao2,
    expected_new_states,
    incidence_coverage,
)
from morphocoverage.strategies.jcb_planning import ObjectCountPlan
from morphocoverage.strategies.scalar_precision import (
    fov_averaging_cv,
    required_fovs,
    required_objects,
)
from morphocoverage.transforms import PCA, Chain, Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _tables(values_per_fov, feature="area"):
    rows = [
        {"image_file": f"f{j}.tif", feature: value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["image_file", feature])
    fovs = pd.DataFrame(
        {
            "image_file": [f"f{j}.tif" for j in range(len(values_per_fov))],
            "order": range(1, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _run(objects, fovs, strategy, features=("area",)):
    [result] = run_strategies(
        objects, fovs=fovs, strategies=[strategy], feature_cols=list(features), objective=POPULATION
    )
    return result


def _replay(objects, fovs, strategy, incremental=False, features=("area",)):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=list(features),
        acquisition_order_col="order",
        objective=POPULATION,
        incremental=incremental,
    )


def _estimate_input(values_per_fov, features=("f0",)):
    """A ValidatedInput from per-FOV (n_i, d) arrays."""
    rows = [
        {"position_id": f"p{i:03d}", **dict(zip(features, row, strict=True))}
        for i, values in enumerate(values_per_fov)
        for row in np.asarray(values, dtype=float).reshape(len(values), -1)
    ]
    objects = pd.DataFrame(rows, columns=["position_id", *features])
    fovs = pd.DataFrame({"position_id": [f"p{i:03d}" for i in range(len(values_per_fov))]})
    return validate_input_tables(
        objects, fovs=fovs, feature_cols=list(features), fov_col="position_id", group_cols=[]
    )


# --- Group C: scalar precision ------------------------------------------------


def test_c1_empty_fields_leave_the_fov_unit_cv_and_its_bound_unchanged():
    sums, counts = np.array([30.0, 50.0, 44.0]), np.array([20.0, 20.0, 18.0])
    alone = fov_averaging_cv(sums, counts)
    for empties in (1, 8, 38):
        padded = fov_averaging_cv(np.r_[sums, np.zeros(empties)], np.r_[counts, np.zeros(empties)])
        assert padded["cv_mean"] == pytest.approx(alone["cv_mean"])
        assert padded["cv_mean_upper_95"] == pytest.approx(alone["cv_mean_upper_95"])
        assert padded["fovs_with_objects"] == 3
    # The factor counts the 3 fields with objects; the bound's degrees of freedom are
    # Kish's effective number of those fields minus one (audit 4, Q3).
    mean = sums.sum() / counts.sum()
    residual = np.sum((sums - mean * counts) ** 2)
    assert alone["cv_mean"] == pytest.approx(math.sqrt(3 / 2 * residual) / counts.sum() / mean)
    kish = counts.sum() ** 2 / (counts**2).sum()
    assert alone["cv_mean_upper_95"] == pytest.approx(
        alone["cv_mean"] * statistics.effective_sd_upper_bound_factor(kish - 1)
    )
    # Empty fields are still effort: the same CV over more acquired fields asks for more.
    assert required_fovs(alone["cv_mean"], 13, 0.05) > required_fovs(alone["cv_mean"], 3, 0.05)

    # Through the strategy: appending empty fields to an acquisition does not move the CV,
    # while the pilot projection still divides by every acquired field.
    objects, fovs = _tables([[1.0, 2.0], [3.0, 3.5], [2.0, 2.5]])
    padded_objects, padded_fovs = _tables([[1.0, 2.0], [3.0, 3.5], [2.0, 2.5], [], [], []])
    strategy = ScalarPrecisionStrategy("area", 0.05, "fov", "upper_95")
    assert _run(padded_objects, padded_fovs, strategy).evidence.value(
        "averaging_cv"
    ) == pytest.approx(_run(objects, fovs, strategy).evidence.value("averaging_cv"))
    piloted = ScalarPrecisionStrategy("area", 0.05, "fov", "point", pilot_fovs=3, order_col="order")
    result = _run(padded_objects, padded_fovs, piloted)
    pilot_cv = result.diagnostics["pilot"]["cv_mean"]
    assert result.evidence.value("pilot_projected_averaging_cv") == pytest.approx(
        pilot_cv * math.sqrt(3 / 6)
    )


def test_c2_a_plan_whose_samples_are_constant_is_refused():
    pilot = pd.DataFrame(
        {
            "condition": ["ctrl"] * 4 + ["drug"] * 4,
            "repeat": ["r1", "r1", "r2", "r2"] * 2,
            "image_file": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "n_nuclei": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="within-sample SD .* is zero"):
        plan_objects_to_average(
            pilot,
            feature="n_nuclei",
            target_cv=0.05,
            sd_estimate="upper_95",
            sample_cols=["condition", "repeat"],
            control={"condition": "ctrl"},
        )
    # A hand-made plan of 0 objects cannot drive the strategy either.
    fields = {name: None for name in ObjectCountPlan.__dataclass_fields__}
    fields.update(sample_cols=("condition",), control=(), pilot_samples=(), required_objects=0)
    with pytest.raises(ValueError, match="requires no object"):
        PlannedObjectCountStrategy(ObjectCountPlan(**fields))


def _area_plan():
    rng = np.random.default_rng(0)
    pilot = pd.DataFrame(
        {
            "condition": ["ctrl"] * 20 + ["drug"] * 20,
            "repeat": "p",
            "image_file": [f"p{i // 5}" for i in range(40)],
            "area": rng.normal(100, 30, 40),
        }
    )
    return plan_objects_to_average(
        pilot,
        feature="area",
        target_cv=0.05,
        sd_estimate="point",
        sample_cols=["condition", "repeat"],
        control={"condition": "ctrl"},
    )


def test_c3_the_planned_count_counts_measured_values_of_the_planned_feature():
    plan = _area_plan()
    n = plan.required_objects
    area = np.random.default_rng(1).normal(100, 30, n)
    area[: n // 2] = np.nan
    objects = pd.DataFrame(
        {
            "condition": "ctrl",
            "repeat": "e",
            "image_file": [f"e{i % 4}" for i in range(n)],
            "area": area,
        }
    )
    fovs = pd.DataFrame(
        {"condition": "ctrl", "repeat": "e", "image_file": [f"e{j}" for j in range(4)]}
    ).assign(order=range(4))
    last = _replay(objects, fovs, PlannedObjectCountStrategy(plan))[-1].result
    assert last.n_observations == n
    assert last.evidence.value("measured_objects") == n - n // 2
    assert last.diagnostics["objects_without_value"] == n // 2
    assert last.decision == "continue"
    # The planned feature must be evaluated: another feature is not counted instead.
    with pytest.raises(ValueError, match="'area' is not one of the explicitly selected"):
        _replay(
            objects.drop(columns="area").assign(other=1.0),
            fovs,
            PlannedObjectCountStrategy(plan),
            features=("other",),
        )


def test_c4_a_categorical_pilot_order_is_refused_rather_than_sorted_two_ways():
    objects, fovs = _tables([[5.0, 6.0], [7.0, 3.0], [9.0, 4.0], [5.0, 5.5]])
    fovs["pilot_order"] = pd.Categorical(
        ["late", "early", "mid", "first"],
        categories=["first", "early", "mid", "late"],
        ordered=True,
    )
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=2, order_col="pilot_order"
    )
    for incremental in (False, True):
        with pytest.raises(TypeError, match="categorical"):
            _replay(objects, fovs, strategy, incremental=incremental)
    # The same order as numbers: full and incremental replay pick the same pilot.
    fovs["pilot_order"] = fovs["pilot_order"].cat.codes
    full, incremental = _replay(objects, fovs, strategy), _replay(objects, fovs, strategy, True)
    for a, b in zip(full, incremental, strict=True):
        pilot_a, pilot_b = a.result.diagnostics.get("pilot"), b.result.diagnostics.get("pilot")
        assert (pilot_a is None) == (pilot_b is None)
        if pilot_a is not None:
            assert pilot_a["cv_objects"] == pytest.approx(pilot_b["cv_objects"])


def test_c5_descriptor_plateau_refuses_a_missing_order_value():
    objects, fovs = _tables([[1.0, 2.0], [10.0, 11.0], [3.0, 2.5], [2.0, 1.5]])
    fovs["acq"] = [1.0, np.nan, 2.0, 3.0]
    strategy = DescriptorPlateauStrategy("area", 100.0, 1, "acq", descriptors=("mean",))
    with pytest.raises(ValueError, match="'acq' has missing values; the descriptor-plateau"):
        _run(objects, fovs, strategy)


def test_c6_a_constant_feature_has_no_design_effect_on_either_path():
    objects, fovs = _tables([[0.1] * 7, [0.1] * 3, [0.1] * 11, [0.1] * 5])
    strategy = ScalarPrecisionStrategy("area", 0.05, "fov", "point")
    full, incremental = _replay(objects, fovs, strategy), _replay(objects, fovs, strategy, True)
    for a, b in zip(full, incremental, strict=True):
        assert "ratio_design_effect" not in a.result.diagnostics
        assert "ratio_design_effect" not in b.result.diagnostics
        assert set(a.result.diagnostics) == set(b.result.diagnostics)


def test_c7_the_pilot_is_documented_and_reported_as_reselected_by_its_order():
    # Pilot fields of equal size: 2 effective fields (Kish) are needed (audit 4, Q3).
    objects, fovs = _tables([[7.0, 8.0], [], [4.0, 6.0], [5.5, 6.5]])
    fovs["plan_rank"] = [3, 4, 1, 2]
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "fov", "point", pilot_fovs=2, order_col="plan_rank"
    )
    steps = [
        item.result.evidence.quantities["pilot_projected_averaging_cv"]
        for item in _replay(objects, fovs, strategy)
    ]
    assert steps[1].availability == "not_applicable"
    assert "changes only if a later FOV sorts before a pilot FOV" in steps[1].reason
    assert steps[2].availability == "available"
    doc = " ".join(ScalarPrecisionEstimator.__doc__.split())
    assert "Only when ``order_col`` is the acquisition order is the pilot fixed" in doc
    assert "another order re-selects the pilot" in " ".join(ScalarPrecisionStrategy.__doc__.split())


def test_c8_c9_required_objects_delegates_to_the_shared_rounded_count():
    assert required_objects(0.07, 0.01) == statistics.jcb_required_objects(0.07, 0.01) == 49
    for bad in (1.0, 1.5, 0.0):
        with pytest.raises(ValueError, match="confidence"):
            statistics.sd_upper_bound_factor(5, bad)


def test_c10_pilot_fovs_must_be_an_integer():
    with pytest.raises(ValueError, match="pilot_fovs must be an integer"):
        ScalarPrecisionStrategy("area", 0.05, "object", "point", pilot_fovs=2.0, order_col="order")
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=np.int64(2), order_col="order"
    )
    assert type(strategy.pilot_fovs) is int


def test_c11_fixed_area_checks_its_area_column_on_construction():
    with pytest.raises(TypeError, match="area_col"):
        FixedAreaStrategy(100.0, "")
    objects, fovs = _tables([[1.0, 2.0], [3.0]])
    fovs["area_um2"] = ["60", "70"]
    with pytest.raises(ValueError, match="not numbers"):
        _run(objects, fovs, FixedAreaStrategy(100.0, "area_um2"))


def test_c12_a_blank_sample_column_in_the_pilot_is_refused_with_a_reason():
    pilot = pd.DataFrame(
        {
            "condition": ["ctrl"] * 6 + ["drug"] * 6,
            "repeat": "r1",
            "well": ["A1", "A1", "A1", "A2", "A2", np.nan] + ["B1"] * 3 + ["B2"] * 3,
            "image_file": [f"p{i}" for i in range(12)],
            "area": [10, 12, 11, 9, 13, 30, 20, 22, 21, 19, 23, 18.0],
        }
    )
    with pytest.raises(ValueError, match=r"sample_cols \['well'\] have missing values"):
        plan_objects_to_average(
            pilot,
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=["condition", "repeat", "well"],
            control={"condition": "ctrl"},
        )


def test_c13_a_zero_mean_so_far_is_insufficient_as_documented():
    objects, fovs = _tables([[0.0, 0.0], [0.0, 0.0]])
    result = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert result.evidence.quantities["averaging_cv"].availability == "insufficient_data"
    doc = " ".join(ScalarPrecisionEstimator.__doc__.split())
    assert "or a zero mean so far (only zeros), give ``insufficient_data``" in doc


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
    objects, fovs = _tables(independent)
    result = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert "field_size_warning" not in result.diagnostics
    assert "field_size_correlation" in result.diagnostics


def test_c15_the_scalar_precision_design_effect_has_its_own_name():
    rng = np.random.default_rng(0)
    objects, fovs = _tables([rng.normal(100 + rng.normal(0, 15), 20, 30) for _ in range(8)])
    result = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert "design_effect" not in result.diagnostics
    object_cv = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert result.diagnostics["ratio_design_effect"] == pytest.approx(
        (result.evidence.value("averaging_cv") / object_cv.evidence.value("averaging_cv")) ** 2
    )
    assert "ratio_design_effect" in ScalarPrecisionEstimator.__doc__


def test_c16_required_fovs_is_rounded_like_the_object_count():
    assert required_fovs(0.07, 1, 0.01) == 49
    assert required_fovs(0.1, 4, 0.05) == 16


def test_c17_object_resampling_is_chunked_and_gives_the_same_spreads(monkeypatch):
    rng = np.random.default_rng(0)
    objects = pd.DataFrame(
        {
            "image_file": rng.integers(0, 50, 5000).astype(str),
            "area": rng.gamma(4, 50, 5000),
        }
    )
    data = validate_input_tables(
        objects, fovs=pd.DataFrame({"image_file": np.arange(50).astype(str)}), feature_cols=["area"]
    )
    estimator = DescriptorSpreadEstimator("area", "object", 100, 0)
    whole = estimator.estimate(data, objective=POPULATION)
    monkeypatch.setattr(_descriptor_resamples, "_MAX_RESAMPLED_VALUES", 12_000)
    tracemalloc.start()
    chunked = estimator.estimate(data, objective=POPULATION)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert chunked.diagnostics["resample_spreads"] == whole.diagnostics["resample_spreads"]
    # 100 resamples of 5000 values would hold 4 MB of values and 4 MB of indices at once.
    assert peak < 3_000_000


# --- Group D: distances, incidence coverage, transforms -------------------------


def test_d1_frechet_distance_scales_with_the_units():
    rng = np.random.default_rng(1)
    x, y = rng.normal(0.0, 1.0, (200, 2)), rng.normal(0.5, 2.0, (200, 2))
    reference = frechet_distance(x, y)
    for unit in (1e-3, 1e-7, 1e-12):
        assert frechet_distance(x * unit, y * unit) == pytest.approx(reference * unit, rel=1e-6)
    assert frechet_distance(x, x) == 0.0
    assert frechet_distance(x * 1e-9, x * 1e-9) == 0.0
    fields = [rng.normal(400e-12, 50e-12, (30, 1)) for _ in range(5)] + [
        rng.normal(700e-12, 50e-12, (30, 1)) for _ in range(5)
    ]
    full, stop = (_estimate_input(f, ("area_m2",)) for f in (fields, fields[:5]))
    assert FrechetError(("area_m2",), False).evaluate(full, stop) > 1e-10


def test_d2_coverage_uses_the_bias_corrected_unseen_states_when_no_doubleton():
    incidences, units = np.array([1, 1, 1, 4]), 5
    unseen = (units - 1) / units * 3 * 2 / 2
    a = units * unseen / (units * unseen + 3)
    assert incidence_coverage(incidences, units) == pytest.approx(1 - 3 / 7 * a)
    assert incidence_coverage(incidences, units) == pytest.approx(0.6571, abs=1e-4)
    # Coverage, Chao2 and the expected new states agree: E[new] = (U / T)(1 - C).
    for incidences, units in (([1, 1, 1, 4], 5), ([1, 1, 2, 3, 5], 6), ([1, 1, 1, 3], 10)):
        incidences = np.array(incidences)
        implied = incidences.sum() / units * (1 - incidence_coverage(incidences, units))
        assert expected_new_states(incidences, units) == pytest.approx(implied)
        assert chao2(incidences, units) >= len(incidences)
    # With Q2 > 0 it is Chao and Jost's own form, unchanged.
    assert incidence_coverage(np.array([5, 2, 1, 1]), 5) == pytest.approx(1 - (2 / 9) * (8 / 10))
    # One singleton and no doubleton: nothing unseen is estimated, so coverage is 1.
    assert incidence_coverage(np.array([1, 3, 3]), 4) == 1.0


def test_d3_a_pca_summary_of_one_standardized_feature_is_never_defined():
    rng = np.random.default_rng(4)
    fields = [
        np.array([[1.0]]),
        np.array([[1.0]]),
        rng.normal(size=(3, 1)),
        rng.normal(size=(3, 1)),
    ]
    estimator = FOVBootstrapEstimator("eigenspectrum", Standardize(), 200, 0, 2)
    for k in range(1, len(fields) + 1):
        quantity = estimator.estimate(_estimate_input(fields[:k]), objective=POPULATION)
        assert quantity.quantities[estimator.quantity].availability == "not_applicable"
    assert Standardize().output_width(3) == 3
    assert PCA(2).output_width(3) == 2 and PCA(2).output_width(1) is None
    assert Chain([Standardize(), PCA(2)]).output_width(4) == 2


def test_d4_pca_keeps_independent_features_on_very_different_scales():
    rng = np.random.default_rng(7)
    matrix = np.c_[rng.normal(0, 1e8, 500), rng.normal(0, 0.05, 500)]
    fitted = PCA(2).fit_matrix(matrix, ("area_nm2", "ecc"))
    assert not isinstance(fitted, str)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    reason = PCA(2).fit_matrix(np.c_[a, 2 * a, -a], ("x", "y", "z"))
    assert isinstance(reason, str) and "floating-point rounding" in reason
    # A constant column with a large mean is rounding noise after centring.
    reason = PCA(2).fit_matrix(np.c_[a * 1e-6, np.full(4, 1e6 + 0.1)], ("x", "y"))
    assert isinstance(reason, str)


def test_d6_outcome_standardization_is_the_package_transform(monkeypatch):
    rng = np.random.default_rng(0)
    full = _estimate_input([rng.normal(3, 2, (10, 2)) for _ in range(5)], ("a", "b"))
    stop = _estimate_input([rng.normal(3, 2, (10, 2)) for _ in range(2)], ("a", "b"))
    calls = []
    original = Standardize.fit_matrix

    def recording(self, matrix, names):
        calls.append(names)
        return original(self, matrix, names)

    monkeypatch.setattr(Standardize, "fit_matrix", recording)
    value = CentroidError(("a", "b"), True).evaluate(full, stop)
    assert calls == [("a", "b")]
    reference = np.column_stack([full.feature_values(f) for f in ("a", "b")])
    sample = np.column_stack([stop.feature_values(f) for f in ("a", "b")])
    centre, scale = reference.mean(axis=0), reference.std(axis=0, ddof=1)
    expected = np.sqrt(np.mean(((sample - centre) / scale).mean(0) ** 2))
    assert value == pytest.approx(expected)
