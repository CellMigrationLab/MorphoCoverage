"""Regressions for the fourth audit, group D (stopping rules), one test per finding.

Q3 (P3-1, P4-3): the per-field JCB bound on Kish's effective number of fields; P4-1: a
readout constant so far; P4-4 (D13): the widened bootstrap SE; P4-5, P4-6, P4-8 (D14),
P4-11 and P9-4 (canonical order), P4-12, P4-13, D2 (P3-7, P4-7), P3-11 and P2-12.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from morphocoverage import (
    ObjectiveConfig,
    OnlineSession,
    RareTarget,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    plan_objects_to_average,
    run_strategies,
    statistics,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.strategies import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    FOVBootstrapStrategy,
    PlannedObjectCountStrategy,
    RareIncidenceStrategy,
    ScalarPrecisionEstimator,
    ScalarPrecisionStrategy,
    _averaging_cv,
    rare_incidence,
    scalar_precision,
    stabilization,
)
from morphocoverage.strategies.scalar_precision import fov_averaging_cv
from morphocoverage.sufficiency import sufficiency_report

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _acquisition(values_per_fov, feature="x"):
    """Objects and an FOV table (with an acquisition order) from values per FOV."""
    rows = [
        {"position_id": f"p{j}", feature: value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["position_id", feature])
    fovs = pd.DataFrame(
        {
            "position_id": [f"p{j}" for j in range(len(values_per_fov))],
            "acq": np.arange(1.0, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _replay(objects, fovs, strategies, incremental=False, objective=POPULATION):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=["x"],
        acquisition_order_col="acq",
        objective=objective,
        fov_col="position_id",
        incremental=incremental,
    )


def _run(objects, fovs, strategies, objective=POPULATION, **kw):
    return run_strategies(
        objects,
        fovs=fovs,
        strategies=strategies,
        feature_cols=["x"],
        objective=objective,
        fov_col="position_id",
        **kw,
    )


def _words(text):
    return " ".join(text.split())


def _quantity(result, name):
    return result.evidence.quantities[name]


def test_p3_1_the_per_field_bound_takes_kish_degrees_of_freedom():
    sums, counts = np.array([300.0, 520.0, 60.0, 90.0]), np.array([30.0, 50.0, 5.0, 10.0])
    found = fov_averaging_cv(sums, counts)
    kish = counts.sum() ** 2 / (counts**2).sum()
    assert found["effective_fovs"] == statistics.kish_effective_fields(counts)
    assert found["effective_fovs"] == round(kish, 9) and 2 < kish < 4
    assert found["cv_mean_upper_95"] == pytest.approx(
        found["cv_mean"] * statistics.effective_sd_upper_bound_factor(kish - 1)
    )
    # Wider than the former k' - 1 = 3 degrees of freedom, since the sizes are unequal.
    assert found["cv_mean_upper_95"] > found["cv_mean"] * statistics.sd_upper_bound_factor(3)
    # Equal sizes: exactly the former bound, and the real-df factor meets the integer one.
    equal = fov_averaging_cv(np.array([30.0, 52.0, 44.0]), np.array([10.0, 10.0, 10.0]))
    assert equal["effective_fovs"] == 3
    assert equal["cv_mean_upper_95"] == equal["cv_mean"] * statistics.sd_upper_bound_factor(2)
    assert statistics.effective_sd_upper_bound_factor(7) == statistics.sd_upper_bound_factor(7)
    assert math.isnan(
        fov_averaging_cv(np.array([1000.0, 5.0]), np.array([10.0, 1.0]))["cv_mean_upper_95"]
    )
    # The docstring names the scope of its former coverage claim.
    doc = _words(_averaging_cv.fov_averaging_cv.__doc__)
    assert "93-98% of the time for a nominal 95% only when field sizes were Poisson" in doc
    assert "94-97%" not in doc


def test_p4_3_one_dominant_field_is_not_enough_data_for_the_per_field_rule():
    rng = np.random.default_rng(0)
    crowded, sparse = list(rng.normal(100, 20, 1000)), list(rng.normal(88, 20, 5))
    even = [list(rng.normal(70, 14, 20)) for _ in range(4)]
    objects, fovs = _acquisition([crowded, sparse, *even])
    rules = [ScalarPrecisionStrategy("x", 0.05, "fov", sd) for sd in ("point", "upper_95")]
    full, incremental = _replay(objects, fovs, rules), _replay(objects, fovs, rules, True)
    for a, b in zip(full, incremental, strict=True):
        assert a.result.evidence == b.result.evidence
    for item in full:
        if item.step < 2:
            continue
        quantity = _quantity(item.result, "averaging_cv")
        assert quantity.availability == "insufficient_data"
        assert quantity.reason.startswith("fewer than 2 effective FOVs (Kish: 1.")
        assert item.result.rule_decision == "continue"
    at_two = [r for r in full if r.step == 2]
    assert len(at_two) == 2 and all(
        _quantity(r.result, "averaging_cv").reason
        == "fewer than 2 effective FOVs (Kish: 1.01); one FOV holds most of the objects"
        for r in at_two
    )
    # Without the crowded field the same fields give a value.
    objects, fovs = _acquisition([sparse, *even])
    [result] = _run(objects, fovs, rules[1:])
    assert _quantity(result, "averaging_cv").availability == "available"


def test_p4_1_a_readout_constant_so_far_is_not_enough_data_for_the_jcs_rules_and_bootstrap():
    rng = np.random.default_rng(1)
    saturated = [[4095.0] * 12 for _ in range(5)]
    varied = [list(np.minimum(4095.0, rng.normal(3000, 600, 12))) for _ in range(3)]
    objects, fovs = _acquisition(saturated + varied)
    rules = [
        DescriptorSpreadStrategy("x", 50.0, "object", 50, 0),
        DescriptorSpreadStrategy("x", 50.0, "fov", 50, 0),
        DescriptorPlateauStrategy("x", 50.0, 2, "acq"),
        FOVBootstrapStrategy("centroid", 50.0, None, 100, 0, 2, features=("x",)),
        FOVBootstrapStrategy("quantiles", 50.0, None, 100, 0, 2, (0.5,), features=("x",)),
    ]
    results = _replay(objects, fovs, rules)
    for item in results:
        if item.step <= 5:
            assert item.result.rule_decision == "continue", (item.step, item.result.strategy)
            for quantity in item.result.evidence.quantities.values():
                assert quantity.availability == "insufficient_data"
    constant = {
        r.result.strategy: _quantity(r.result, next(iter(r.result.evidence.quantities))).reason
        for r in results
        if r.step == 5
    }
    assert constant == {
        "descriptor_spread_object": "no spread among the objects so far",
        "descriptor_spread_fov": "no spread among the objects so far",
        "descriptor_plateau": "no spread among the objects so far",
        "fov_bootstrap_centroid": "no spread between the FOV means of x so far",
        "fov_bootstrap_quantiles": "no spread among the values of x so far",
    }
    # Once a field differs every rule has a value again.
    last = [r for r in results if r.step == len(fovs)]
    assert all(
        q.availability == "available" for r in last for q in r.result.evidence.quantities.values()
    )


def test_p4_4_the_bootstrap_stopping_se_is_widened_for_few_fields():
    assert statistics.small_sample_widening(5) == math.sqrt(5 / 4) * stats.t.ppf(
        0.975, 4
    ) / stats.norm.ppf(0.975)
    rng = np.random.default_rng(2)
    values = [list(rng.normal(10 + rng.normal(0, 0.5), 1, 15)) for _ in range(4)] + [[]]
    objects, fovs = _acquisition(values)
    strategy = FOVBootstrapStrategy("centroid", 0.5, None, 400, 0, 2, features=("x",))
    [result] = _run(objects, fovs, [strategy])
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
    two = _acquisition([[10.0, 10.2, 9.9], [10.1, 10.3, 10.0]])
    [early] = _run(*two, [FOVBootstrapStrategy("centroid", 0.1, None, 400, 0, 2)])
    raw = early.evidence.value("fov_bootstrap_centroid_max_se") / (
        statistics.small_sample_widening(2)
    )
    assert raw < 0.1 and early.rule_decision == "continue"


def test_p4_5_a_plateau_stop_is_worded_as_change_over_the_window():
    rng = np.random.default_rng(3)
    objects, fovs = _acquisition([list(rng.normal(10, 1, 30)) for _ in range(6)])
    rule = DescriptorPlateauStrategy("x", 5.0, 2, "acq")
    [result] = _run(objects, fovs, [rule])
    assert result.rule_decision == "stop"
    caution = result.evidence.diagnostics["warning"]
    assert caution.startswith(
        "the tolerance bounds how much the descriptors changed over the last 2 FOVs with "
        "objects, not their error against the well"
    )
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["x"],
        strategies=[rule],
        objective=POPULATION,
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="acq",
        fov_col="position_id",
    )
    answer = report.enough["answer"].iloc[0]
    assert "yes, its threshold was met by the 6 acquired FOVs; " in answer
    # A standing assumption (#139): stated in the answer, never as a caution.
    assert f"{caution[0].upper()}{caution[1:]}" in answer and "Caution" not in answer
    # Stated once in both summaries (their lines are wrapped, so compare the words).
    for detail in ("short", "full"):
        assert caution in " ".join(report.summary(detail=detail).split())
    assert "not that they are within ``tolerance`` of the well's" in _words(
        DescriptorPlateauStrategy.__doc__
    )
    assert "40% of stops in homogeneous wells" in _words(stabilization.__doc__)


def test_p4_6_the_per_field_spread_needs_five_fields_with_objects():
    rng = np.random.default_rng(4)
    objects, fovs = _acquisition([list(rng.normal(10, 1, 10)) for _ in range(6)] + [[]])
    results = _replay(objects, fovs, [DescriptorSpreadStrategy("x", 1.0, "fov", 50, 0)])
    reasons = {r.step: _quantity(r.result, "max_location_spread").reason for r in results}
    for step in (2, 3, 4):
        assert reasons[step].startswith("fewer than 5 acquired FOVs with objects")
    assert all(reasons[step] is None for step in (5, 6, 7))
    assert stabilization.MIN_SPREAD_FOVS == 5


def test_p4_8_rare_incidence_states_its_false_stop_rates():
    doc = _words(rare_incidence.__doc__)
    for rate in ("5.2%", "7.7%", "9.6%", "13.9%", "18.3%", "20.9%", "55-69%"):
        assert rate in doc
    assert "not recalibrated" in doc
    objects, fovs = _acquisition([[1.0, 2.0]] * 3)
    objects["state"] = False
    target = RareTarget(
        fov_incidence=0.05, detection_probability=0.95, min_fov_occurrences=2, phenotype="state"
    )
    objective = ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=target)
    [result] = _run(objects, fovs, [RareIncidenceStrategy("state", 1, 1, 0.05)], objective)
    caution = result.evidence.diagnostics["warning"]
    assert caution.startswith("a posterior risk, not a detection guarantee")
    assert "7.7%" in caution and "runs of 5 neighbouring FOVs" in caution


def _well(seed, n_fovs=8, per_fov=12):
    rng = np.random.default_rng(seed)
    means = rng.normal(10, 1.5, n_fovs)
    return _acquisition([list(rng.normal(m, 2, per_fov)) for m in means])


def test_p4_11_the_fov_table_order_does_not_change_the_resampled_rules():
    objects, fovs = _well(5)
    rules = [
        DescriptorSpreadStrategy("x", 1.0, "fov", 100, 0),
        FOVBootstrapStrategy("centroid", 0.3, None, 200, 0, 3, features=("x",)),
        FOVBootstrapStrategy("quantiles", 0.3, None, 50, 0, 3, (0.25, 0.75), features=("x",)),
    ]
    base = _run(objects, fovs, rules)
    for seed in range(4):
        shuffled_fovs = fovs.sample(frac=1.0, random_state=seed)
        shuffled_objects = objects.sample(frac=1.0, random_state=seed)
        again = _run(shuffled_objects, shuffled_fovs, rules)
        for a, b in zip(base, again, strict=True):
            # The same draws; only floating-point summation order may differ.
            for name, quantity in a.evidence.quantities.items():
                assert b.evidence.quantities[name].value == pytest.approx(
                    quantity.value, rel=1e-12
                ), (a.strategy, name)


def test_p9_4_object_row_order_within_fields_does_not_change_the_live_stop():
    objects, fovs = _well(6, n_fovs=10, per_fov=15)
    rule = DescriptorSpreadStrategy("x", 0.6, "object", 100, 0)
    base = _replay(objects, fovs, [rule])
    reordered = (
        objects.assign(_key=np.random.default_rng(0).random(len(objects)))
        .sort_values(["position_id", "_key"])
        .drop(columns="_key")
    )
    again = _replay(reordered, fovs, [rule])
    assert [r.result.evidence.quantities for r in base] == [
        r.result.evidence.quantities for r in again
    ]
    assert [r.result.rule_decision for r in base] == [r.result.rule_decision for r in again]


def test_p4_12_the_cv_criterion_asks_for_background_subtracted_values():
    doc = _words(scalar_precision.__doc__)
    assert "intensities must be background-subtracted" in doc
    assert "exchangeable draws of the well's positions" in doc


def test_p4_13_a_point_estimate_from_few_fields_carries_a_caution():
    rng = np.random.default_rng(7)
    values = [list(rng.normal(10 + rng.normal(0, 0.3), 1, 20)) for _ in range(5)]
    point = ScalarPrecisionStrategy("x", 0.05, "fov", "point")
    upper = ScalarPrecisionStrategy("x", 0.05, "fov", "upper_95")
    [few, bounded] = _run(*_acquisition(values[:3]), [point, upper])
    assert few.evidence.diagnostics["warning"].startswith(
        "the point estimate from 3 FOVs with objects is optimistic; the paper advises the "
        "upper-95 bound for small samples"
    )
    assert "warning" not in bounded.evidence.diagnostics
    [enough] = _run(*_acquisition(values), [point])
    assert "warning" not in enough.evidence.diagnostics


def _mixture(rng, n):
    """90% of cells around 10, 10% around 40: a minority state."""
    return list(np.where(rng.random(n) < 0.1, rng.normal(40, 4, n), rng.normal(10, 2, n)))


def test_p3_7_a_readout_far_from_normal_carries_a_caution_on_the_per_object_bound():
    rng = np.random.default_rng(8)
    mixed = [_mixture(rng, 40) for _ in range(4)]
    normal = [list(rng.normal(10, 2, 40)) for _ in range(4)]
    upper = ScalarPrecisionStrategy("x", 0.05, "object", "upper_95")
    point = ScalarPrecisionStrategy("x", 0.05, "object", "point")
    [flagged, unflagged] = _run(*_acquisition(mixed), [upper, point])
    caution = flagged.evidence.diagnostics["warning"]
    assert "the upper-95 SD bound assumes normal values, and these objects" in caution
    # The point SD is not a bound: only the 50% CV caution.
    assert "normal values" not in unflagged.evidence.diagnostics["warning"]
    [plain] = _run(*_acquisition(normal), [upper])
    assert "warning" not in plain.evidence.diagnostics
    # The incremental path merges the higher moments to the same caution.
    objects, fovs = _acquisition(mixed)
    full, incremental = _replay(objects, fovs, [upper]), _replay(objects, fovs, [upper], True)
    assert [r.result.evidence.diagnostics.get("warning") for r in full] == [
        r.result.evidence.diagnostics.get("warning") for r in incremental
    ]
    n, m2, m3, m4 = _averaging_cv._central_sums(np.concatenate([np.array(v) for v in mixed]))
    pooled = np.concatenate([np.array(v) for v in mixed])
    assert f"excess kurtosis {stats.kurtosis(pooled, bias=False):.1f}" in caution
    assert _averaging_cv._shape_warning(n, m2, m3, m4, "objects") in caution


def _pilot(values_by_sample):
    rows = [
        {
            "condition": condition,
            "biological_repeat": "r1",
            "image_file": f"{condition}{j}",
            "x": value,
        }
        for condition, samples in values_by_sample.items()
        for j, values in enumerate(samples)
        for value in values
    ]
    return pd.DataFrame(rows)


def test_p4_7_the_plan_carries_cautions_for_values_far_from_normal_and_a_cv_above_50():
    rng = np.random.default_rng(9)
    settings = {
        "feature": "x",
        "target_cv": 0.05,
        "sample_cols": ["condition", "image_file"],
        "control": {"condition": "ctrl"},
    }
    mixed = _pilot({"ctrl": [_mixture(rng, 30) for _ in range(3)]})
    plan = plan_objects_to_average(mixed, sd_estimate="upper_95", **settings)
    assert any(w.startswith("the upper-95 SD bound assumes normal values") for w in plan.warnings)
    assert "warnings" not in plan.parameters  # the cautions do not define the plan
    point = plan_objects_to_average(mixed, sd_estimate="point", **settings)
    assert not any("normal values" in w for w in point.warnings)
    wide = _pilot({"ctrl": [list(rng.exponential(10.0, 30)) for _ in range(3)]})
    assert any(
        "exceeds 50% CV of the control mean" in w
        for w in plan_objects_to_average(wide, sd_estimate="point", **settings).warnings
    )
    normal = _pilot({"ctrl": [list(rng.normal(10, 1, 30)) for _ in range(3)]})
    assert plan_objects_to_average(normal, sd_estimate="upper_95", **settings).warnings == ()
    # The planned rule shows the plan's cautions beside its evidence.
    treated = pd.DataFrame(
        {"condition": "drug", "biological_repeat": "r1", "image_file": "d0", "x": [1.0, 2.0, 3.0]}
    )
    [result] = run_strategies(
        treated,
        strategies=[PlannedObjectCountStrategy(plan)],
        feature_cols=["x"],
        objective=POPULATION,
    )
    assert result.evidence.diagnostics["warning"] == "plan: " + "; ".join(plan.warnings)


def test_p3_11_the_field_size_check_says_no_warning_is_not_evidence():
    phrase = "no warning is not evidence that"
    assert phrase in _words(_averaging_cv._field_size_check.__doc__)
    assert phrase in _words(ScalarPrecisionEstimator.__doc__)


def test_p2_12_a_missing_value_names_its_field_and_object_and_what_to_do():
    rng = np.random.default_rng(10)
    values = [list(rng.normal(10, 1, 6)) for _ in range(5)]
    values[1][2] = float("nan")
    values[3][0] = float("nan")
    objects, fovs = _acquisition(values)
    rule = ScalarPrecisionStrategy("x", 0.05, "fov", "upper_95")
    full, incremental = _replay(objects, fovs, [rule]), _replay(objects, fovs, [rule], True)
    expected = (
        "missing values in the selected scalar feature: object 3 of FOV 'p1' has none{}; the "
        "rule stays undefined for this group until the value is filled in, since values are "
        "never imputed or dropped"
    )
    for a, b in zip(full, incremental, strict=True):
        qa, qb = _quantity(a.result, "averaging_cv"), _quantity(b.result, "averaging_cv")
        assert qa == qb
        if a.step >= 2:
            assert qa.availability == "not_applicable"
            others = " (2 FOVs hold one)" if a.step >= 4 else ""
            assert qa.reason == expected.format(others)
    # The live session says the same, field by field.
    session = OnlineSession(strategies=[rule], feature_cols=["x"], objective=POPULATION)
    for _, fov in fovs.iterrows():
        rows = objects.loc[objects["position_id"].eq(fov["position_id"])]
        [live] = session.add_fov(rows.copy(), fov.to_dict())
    assert _quantity(live, "averaging_cv").reason == expected.format(" (2 FOVs hold one)")
