"""Tests for the JCB/Zweifach-style scalar precision baseline."""

from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _three
from _helpers_strategies import (
    PERMUTED_NINE,
    READOUT,
    _acquisition,
    _area_replay,
    _area_run,
    _area_tables,
    _area_values_tables,
    _joined_words,
    _live,
    _mixture,
    _named_scenario_report,
    _ordered_acquisition,
    _quantity,
    _raster_well,
    _report_both_orders,
    _run,
    _run_one,
    _summary,
    _words,
    _x_replay,
    _x_run,
)
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS
from scipy import stats

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    OnlineSession,
    SamplingObjective,
    ScalarPrecisionEstimator,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    run_strategies,
    statistics,
    validate_input_tables,
)
from morphocoverage._style import Status
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evidence import Availability
from morphocoverage.objectives import RareTarget
from morphocoverage.results import Decision
from morphocoverage.retrospective import microscopist_report
from morphocoverage.retrospective.report_status import enough_status
from morphocoverage.strategies import (
    DescriptorPlateauStrategy,
    RareIncidenceStrategy,
    _averaging_cv,
    base,
    baselines,
    fov_bootstrap,
    jcb_planning,
    scalar_precision,
    stabilization,
)
from morphocoverage.strategies._averaging_cv import _field_size_check
from morphocoverage.strategies.scalar_precision import (
    fov_averaging_cv,
    object_averaging_cv,
    required_fovs,
    required_objects,
    sd_upper_bound_factor,
)
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import Setup

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _cv(result, quantity="averaging_cv"):
    return result.evidence.value(quantity)


def _evaluate(objects, fovs, strategy, feature_cols=("area",)):
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[strategy],
        feature_cols=list(feature_cols),
        objective=POPULATION,
    )
    return result


def _tables(values_per_fov):
    rows = [
        {"image_file": f"f{j}.tif", "area": value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["image_file", "area"])
    fovs = pd.DataFrame(
        {
            "image_file": [f"f{j}.tif" for j in range(len(values_per_fov))],
            "order": range(1, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


# --- The published criterion -----------------------------------------------------


def test_cell_averaging_cv_matches_hand_calculation():
    estimate = object_averaging_cv(np.array([1.0, 2.0, 3.0, 4.0]))
    cv_objects = np.std([1, 2, 3, 4], ddof=1) / 2.5
    assert estimate["cv_objects"] == pytest.approx(cv_objects)
    assert estimate["cv_mean"] == pytest.approx(cv_objects / 2)


def test_object_unit_required_cells_follows_cv_squared_over_target_squared():
    objects, fovs = _tables([[1.0, 2.0], [3.0, 4.0]])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    cv_objects = np.std([1, 2, 3, 4], ddof=1) / 2.5
    assert result.diagnostics["cv_objects"] == pytest.approx(cv_objects)
    assert required_objects(cv_objects, 0.05) == math.ceil((cv_objects / 0.05) ** 2)
    assert _cv(result) == pytest.approx(cv_objects / 2)
    assert result.decision == "continue"
    assert result.diagnostics["assumes_spatial_uniformity"] is True


def test_object_unit_stops_exactly_when_averaging_cv_reaches_target():
    # 100 cells, mean 1, sample SD 0.2 * sqrt(100 / 99): CV_mean = 0.020101.
    values = np.tile([0.8, 1.2], 50)
    objects, fovs = _tables([values])
    stop = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.0202, "object", "point"))
    go_on = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.0200, "object", "point"))
    assert _cv(stop) == pytest.approx(0.2 * math.sqrt(100 / 99) / 10)
    assert (stop.decision, go_on.decision) == ("stop", "continue")
    # The estimate does not depend on the target; only the policy does.
    assert stop.evidence == go_on.evidence
    assert stop.config_id != go_on.config_id
    assert required_objects(stop.diagnostics["cv_objects"], 0.0202) <= 100


@pytest.mark.parametrize(
    ("df", "lower_quantile"),
    [(1, 0.000982069), (3, 0.215795), (9, 2.700389), (19, 8.906516), (29, 16.047072)],
)
def test_sd_upper_bound_uses_the_chi_square_table(df, lower_quantile):
    # Printed chi-square 2.5% quantiles; the bound is sqrt(df / quantile).
    assert sd_upper_bound_factor(df) == pytest.approx(math.sqrt(df / lower_quantile), rel=1e-5)


def _cells_with_cv(n, cv):
    """n positive values with mean 1 and sample CV exactly ``cv``."""
    z = np.linspace(-1, 1, n)
    z = (z - z.mean()) / np.std(z, ddof=1)
    return 1 + cv * z


def test_reproduces_the_papers_worked_example():
    # Zweifach 2024: 20 cells, point SD 17.8% CV, 95% upper bound 26.0% CV; at a
    # 2.5% target average 51 (point) or 108 (upper bound) cells, at 5% 13 or 27.
    # The printed 17.8% is rounded. The four counts, with a chi-square bound on
    # 19 df, and the printed 26.0% bound together require SD in about
    # [17.767%, 17.789%], so 17.78% is used.
    objects, fovs = _tables([_cells_with_cv(20, 0.1778)])
    estimate = object_averaging_cv(objects["area"].to_numpy())
    assert round(estimate["cv_objects"], 3) == 0.178
    assert round(estimate["cv_objects_upper_95"], 3) == 0.260
    required = {}
    for target in (0.025, 0.05):
        for sd_estimate in ("point", "upper_95"):
            strategy = ScalarPrecisionStrategy("area", target, "object", sd_estimate)
            result = _evaluate(objects, fovs, strategy)
            suffix = "_upper_95" if sd_estimate == "upper_95" else ""
            cv_objects = result.diagnostics["cv_objects" + suffix]
            required[target, sd_estimate] = required_objects(cv_objects, target)
    assert required == {
        (0.025, "point"): 51,
        (0.025, "upper_95"): 108,
        (0.05, "point"): 13,
        (0.05, "upper_95"): 27,
    }


def test_upper_bound_is_more_conservative_and_converges_with_more_cells():
    assert sd_upper_bound_factor(3) > sd_upper_bound_factor(19) > sd_upper_bound_factor(999) > 1
    assert sd_upper_bound_factor(999) == pytest.approx(1, abs=0.05)


def test_large_between_cell_sd_carries_the_papers_warning():
    objects, fovs = _tables([_cells_with_cv(4, 0.6)])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert "50% of their mean" in result.diagnostics["warning"]
    objects, fovs = _tables([_cells_with_cv(4, 0.3)])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert "warning" not in result.diagnostics


# --- MorphoCoverage FOV-unit variant ---------------------------------------------


def test_fov_averaging_cv_matches_hand_calculation():
    # FOV sums 2 and 6 over 2 objects each; m = 2, residuals -2 and 2.
    estimate = fov_averaging_cv(np.array([2.0, 6.0]), np.array([2.0, 2.0]))
    assert estimate["cv_mean"] == pytest.approx(math.sqrt(2 / 1 * 8 / 16) / 2)


def test_zero_object_fovs_are_sampled_units_in_the_fov_variant():
    # An empty FOV has no residual and no between-FOV information: the k' / (k' - 1)
    # factor counts the 2 FOVs with objects (audit 2, C-1), and the FOV is still effort.
    with_empty = fov_averaging_cv(np.array([2.0, 6.0, 0.0]), np.array([2.0, 2.0, 0.0]))
    assert with_empty["cv_mean"] == pytest.approx(math.sqrt(2 / 1 * 8 / 16) / 2)

    objects, fovs = _tables([[1.0, 1.0], [3.0, 3.0], []])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert _cv(result) == pytest.approx(with_empty["cv_mean"])
    assert result.n_fovs == 3


def test_fov_variant_is_not_yet_defined_when_fov_means_do_not_differ():
    # Identical FOVs carry no between-FOV variance: a CV of 0 would say nothing about
    # averaging noise, so the FOV unit waits for FOV means that differ (audit 3, C-1).
    objects, fovs = _tables([[1.0, 3.0]] * 4)
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    quantity = result.evidence.quantities["averaging_cv"]
    assert quantity.availability == "insufficient_data"
    assert quantity.reason == "no spread among the FOV means so far"


def test_spatial_clustering_inflates_only_the_fov_variant():
    # Same cells, two arrangements: mixed within every FOV, or sorted by FOV.
    values = np.tile([1.0, 3.0], 40)
    mixed = _tables(values.reshape(8, 10))
    clustered = _tables(np.sort(values).reshape(8, 10))
    cell_mixed = _cv(_evaluate(*mixed, ScalarPrecisionStrategy("area", 0.05, "object", "point")))
    cell_clustered = _cv(
        _evaluate(*clustered, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    )
    fov_mixed = _evaluate(*mixed, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    fov_clustered = _evaluate(*clustered, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert cell_mixed == pytest.approx(cell_clustered)
    assert _cv(fov_clustered) > 2 * cell_clustered
    assert fov_clustered.diagnostics["ratio_design_effect"] > 5
    # Mixed FOVs all have the same mean: no between-FOV spread yet (audit 3, C-1).
    assert fov_mixed.evidence.quantities["averaging_cv"].availability == "insufficient_data"


def test_neighbour_correlation_scenario_shows_design_effect():
    scenario = SCENARIOS["neighbour_correlation"]()
    objects = scenario.objects.assign(feature_a=scenario.objects["feature_a"] + 20)
    fov = _evaluate(
        objects,
        scenario.fovs,
        ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
        feature_cols=FEATURE_COLS,
    )
    homogeneous = SCENARIOS["homogeneous"]()
    objects_h = homogeneous.objects.assign(feature_a=homogeneous.objects["feature_a"] + 20)
    fov_h = _evaluate(
        objects_h,
        homogeneous.fovs,
        ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
        feature_cols=FEATURE_COLS,
    )
    assert fov.diagnostics["ratio_design_effect"] > 10
    assert 0.2 < fov_h.diagnostics["ratio_design_effect"] < 3


# --- Pilot planning ------------------------------------------------------------


def test_pilot_plan_is_fixed_by_the_first_fovs_in_acquisition_order():
    objects, fovs = _tables([[1.0, 3.0]] * 2 + [[2.0, 2.0]] * 30)
    strategy = ScalarPrecisionStrategy(
        "area", 0.1, "object", "point", pilot_fovs=2, order_col="order"
    )
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=POPULATION,
    )
    # Pilot CV_cells = sd([1,3,1,3]) / 2; required cells fixed from that alone.
    cv_pilot = np.std([1, 3, 1, 3], ddof=1) / 2
    required = math.ceil((cv_pilot / 0.1) ** 2)
    assert results[0].result.decision == "continue"  # pilot incomplete
    projected = results[0].result.evidence.quantities["pilot_projected_averaging_cv"]
    assert projected.availability == "insufficient_data"
    for item in results[1:]:
        assert item.result.diagnostics["pilot"]["cv_objects"] == pytest.approx(cv_pilot)
        assert _cv(item.result, "pilot_projected_averaging_cv") == pytest.approx(
            cv_pilot / math.sqrt(item.result.n_observations)
        )
    first_stop = next(r for r in results if r.result.decision == "stop")
    assert first_stop.result.n_observations == required + (required % 2)  # 2 cells per FOV
    # Stopping at the projected CV is the paper's rule: stop once the planned
    # number of cells has been acquired.
    for item in results[1:]:
        planned = item.result.n_observations >= required
        assert (item.result.decision == "stop") == planned


def test_fov_pilot_stops_when_the_planned_number_of_fovs_is_reached():
    rng = np.random.default_rng(4)
    objects, fovs = _tables([rng.uniform(1, 3, 10) + j % 3 for j in range(40)])
    strategy = ScalarPrecisionStrategy(
        "area", 0.08, "fov", "point", pilot_fovs=4, order_col="order"
    )
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=POPULATION,
    )
    pilot = results[3].result.diagnostics["pilot"]
    planned = required_fovs(pilot["cv_mean"], 4, 0.08)
    assert 4 < planned < 40
    assert [r.step for r in results if r.result.decision == "stop"] == list(range(planned, 41))


def test_pilot_uses_order_column_not_row_order():
    objects, fovs = _tables([[2.0, 2.0], [1.0, 3.0], [1.0, 3.0]])
    fovs["order"] = [3, 1, 2]  # the two variable FOVs were acquired first
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=2, order_col="order"
    )
    result = _evaluate(objects, fovs, strategy)
    assert result.diagnostics["pilot"]["cv_objects"] == pytest.approx(
        np.std([1, 3, 1, 3], ddof=1) / 2
    )


def test_diagnostics_leave_effort_to_the_framework_and_keep_the_pilot_counts():
    objects, fovs = _tables([[2.0, 2.0], [1.0, 3.0], [1.0, 3.0]])
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=2, order_col="order"
    )
    result = _evaluate(objects, fovs, strategy)
    # AGENTS.md section 20: effort of the visible data is the framework's.
    assert "n_objects" not in result.diagnostics and "n_fovs" not in result.diagnostics
    assert (result.n_fovs, result.n_observations) == (3, 6)
    # The pilot's own counts are its provenance: the first two FOVs by order.
    assert (result.diagnostics["pilot"]["n_fovs"], result.diagnostics["pilot"]["n_objects"]) == (
        2,
        4,
    )


# --- Undefined cases and input rules --------------------------------------------


@pytest.mark.parametrize("unit", ["object", "fov"])
def test_non_positive_mean_gives_no_decision(unit):
    objects, fovs = _tables([[-1.0, -2.0], [-3.0, -1.0]])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, unit, "point"))
    assert result.decision is None
    assert "negative" in result.diagnostics["undefined_reason"]


@pytest.mark.parametrize("unit", ["object", "fov"])
def test_negative_values_with_positive_mean_give_no_decision(unit):
    # A positive mean does not make a CV meaningful when values can be negative.
    objects, fovs = _tables([[-5.0, 10.0], [10.0, 10.0]])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, unit, "point"))
    assert result.decision is None and _cv(result) is None
    assert result.evidence.quantities["averaging_cv"].availability == "not_applicable"
    assert "negative values" in result.diagnostics["undefined_reason"]


def test_too_little_data_continues():
    # Insufficient data is not an undefined criterion: more acquisition can fix it.
    objects, fovs = _tables([[1.0]])
    for unit in ("object", "fov"):
        result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, unit, "point"))
        assert result.decision == "continue" and _cv(result) is None
        assert result.evidence.quantities["averaging_cv"].availability == "insufficient_data"


def test_pilot_that_cannot_define_the_requirement_gives_no_decision():
    # The pilot is fixed once complete, so a pilot with one object never recovers.
    objects, fovs = _tables([[1.0], [], [2.0, 3.0]])
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=2, order_col="order"
    )
    result = _evaluate(objects, fovs, strategy)
    assert result.decision is None
    projected = result.evidence.quantities["pilot_projected_averaging_cv"]
    assert projected.availability == "not_applicable" and "pilot" in projected.reason


def test_feature_must_be_an_explicit_feature_column():
    objects, fovs = _tables([[1.0, 2.0]])
    objects["stage_x"] = 5.0
    with pytest.raises(ValueError, match="explicitly selected"):
        _evaluate(objects, fovs, ScalarPrecisionStrategy("stage_x", 0.05, "object", "point"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_cv": 0.0},
        {"target_cv": 1.5},
        {"resampling_unit": "cell"},
        {"sd_estimate": "upper"},
        {"pilot_fovs": 3},
        {"order_col": "order"},
        {"pilot_fovs": 1, "order_col": "order"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    config = {
        "feature": "area",
        "target_cv": 0.05,
        "resampling_unit": "object",
        "sd_estimate": "point",
        **kwargs,
    }
    with pytest.raises(ValueError):
        ScalarPrecisionStrategy(**config)


def test_names_and_parameters_record_the_variant():
    strategy = ScalarPrecisionStrategy(
        "area", 0.025, "fov", "point", pilot_fovs=3, order_col=ORDER_COL
    )
    assert strategy.name == "scalar_precision_fov_pilot"
    upper = ScalarPrecisionStrategy("area", 0.025, "object", "upper_95")
    assert upper.name == "scalar_precision_object_upper95"
    assert strategy.parameters == {
        "feature": "area",
        "target_cv": 0.025,
        "resampling_unit": "fov",
        "sd_estimate": "point",
        "pilot_fovs": 3,
        "order_col": ORDER_COL,
    }


@pytest.mark.parametrize("unit", ["object", "fov"])
@pytest.mark.parametrize("sd_estimate", ["point", "upper_95"])
def test_incremental_pilot_follows_the_order_column_not_arrival(unit, sd_estimate):
    # The pilot is the first FOVs by order_col, which need not be the replay
    # order; incremental and full recomputation must agree, including warm-up.
    rng = np.random.default_rng(5)
    objects, fovs = _tables([rng.uniform(1, 3, 6) + j % 4 for j in range(12)])
    fovs["pilot_rank"] = fovs["order"].to_numpy()[::-1]
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, unit, sd_estimate, pilot_fovs=3, order_col="pilot_rank"
    )

    def replay(incremental):
        return evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[strategy],
            feature_cols=["area"],
            acquisition_order_col="order",
            objective=POPULATION,
            min_fovs=4,
            incremental=incremental,
        )

    full, incremental = replay(False), replay(True)
    assert len(full) == len(incremental) == 9
    for before, after in zip(full, incremental, strict=True):
        assert before.result.decision == after.result.decision
        for name, quantity in before.result.evidence.quantities.items():
            other = after.result.evidence.quantities[name]
            assert quantity.availability == other.availability
            if quantity.value is not None:
                assert other.value == pytest.approx(quantity.value, rel=1e-9)


@pytest.mark.parametrize("unit", ["object", "fov"])
def test_missing_scalar_values_make_the_criterion_not_applicable(unit):
    objects, fovs = _tables([[1.0, float("nan")], [2.0, 3.0]])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, unit, "point"))
    quantity = result.evidence.quantities["averaging_cv"]
    assert quantity.availability == "not_applicable"
    assert result.decision is None
    assert "missing values" in quantity.reason


def test_object_unit_criterion_is_not_applicable_to_non_object_rows():
    objects, fovs = _tables([[1.0, 2.0], [2.0, 3.0]])
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=["area"],
        observation_unit="slice",
    )
    estimator = ScalarPrecisionEstimator("area", "object", "point", pilot_fovs=2, order_col="order")
    evidence = estimator.estimate(data, objective=POPULATION)
    for quantity in evidence.quantities.values():
        assert quantity.availability == "not_applicable"
        assert "(slice)" in quantity.reason
    # Incremental replay falls back to full recomputation, with the same evidence.
    assert estimator.start(data, objective=POPULATION) is None


def test_fov_unit_variant_applies_to_any_observation_unit():
    objects, fovs = _tables([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]])
    strategy = ScalarPrecisionStrategy("area", 0.05, "fov", "point")
    by_unit = {
        unit: run_strategies(
            objects,
            fovs=fovs,
            strategies=[strategy],
            feature_cols=["area"],
            objective=POPULATION,
            observation_unit=unit,
        )[0]
        for unit in ("slice", "cell")
    }
    # The same numbers; only the wording names the rows (slices, cells).
    assert by_unit["slice"].evidence.quantities == by_unit["cell"].evidence.quantities
    assert by_unit["slice"].diagnostics["scalar_parameter"] == "mean of 'area' over slices"
    assert by_unit["slice"].evidence.quantities["averaging_cv"].availability == "available"


@pytest.mark.parametrize("incremental", [False, True])
def test_non_object_rows_do_not_crash_a_replay(incremental):
    objects, fovs = _tables([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]])
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[
            ScalarPrecisionStrategy("area", 0.05, "object", "point"),
            ScalarPrecisionStrategy("area", 0.05, "fov", "point"),
        ],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=POPULATION,
        observation_unit="slices",
        incremental=incremental,
    )
    decisions = {r.result.strategy: r.result.decision for r in results if r.step == 3}
    assert decisions["scalar_precision_object"] is None
    assert decisions["scalar_precision_fov"] is not None
    assert {r.result.observation_unit for r in results} == {"slice"}


def test_c1_empty_fields_leave_the_fov_unit_cv_and_its_bound_unchanged():
    sums, counts = np.array([30.0, 50.0, 44.0, 36.0]), np.array([20.0, 20.0, 18.0, 19.0])
    alone = fov_averaging_cv(sums, counts)
    for empties in (1, 8, 38):
        padded = fov_averaging_cv(np.r_[sums, np.zeros(empties)], np.r_[counts, np.zeros(empties)])
        assert padded["cv_mean"] == pytest.approx(alone["cv_mean"])
        assert padded["cv_mean_upper_95"] == pytest.approx(alone["cv_mean_upper_95"])
        assert padded["fovs_with_objects"] == 4
    # The factor counts the 4 fields with objects; the bound resamples those fields only
    # (a bootstrap of whole fields, #97; owner, 2026-10-05).
    mean = sums.sum() / counts.sum()
    residual = np.sum((sums - mean * counts) ** 2)
    assert alone["cv_mean"] == pytest.approx(math.sqrt(4 / 3 * residual) / counts.sum() / mean)
    assert alone["cv_mean_upper_95"] > alone["cv_mean"]
    # Empty fields are still effort: the same CV over more acquired fields asks for more.
    assert required_fovs(alone["cv_mean"], 13, 0.05) > required_fovs(alone["cv_mean"], 3, 0.05)

    # Through the strategy: appending empty fields to an acquisition does not move the CV,
    # while the pilot projection still divides by every acquired field.
    acquired = [[1.0, 2.0], [3.0, 3.5], [2.0, 2.5], [1.5, 2.5]]
    objects, fovs = _area_tables(acquired)
    padded_objects, padded_fovs = _area_tables([*acquired, [], [], []])
    strategy = ScalarPrecisionStrategy("area", 0.05, "fov", "upper_95")
    bound = _run(objects, fovs, strategy).evidence.value("averaging_cv")
    assert bound is not None
    assert _run(padded_objects, padded_fovs, strategy).evidence.value(
        "averaging_cv"
    ) == pytest.approx(bound)
    piloted = ScalarPrecisionStrategy("area", 0.05, "fov", "point", pilot_fovs=3, order_col="order")
    result = _run(padded_objects, padded_fovs, piloted)
    pilot_cv = result.diagnostics["pilot"]["cv_mean"]
    assert result.evidence.value("pilot_projected_averaging_cv") == pytest.approx(
        pilot_cv * math.sqrt(3 / 7)
    )


def test_c4_a_categorical_pilot_order_is_refused_rather_than_sorted_two_ways():
    objects, fovs = _area_tables([[5.0, 6.0], [7.0, 3.0], [9.0, 4.0], [5.0, 5.5]])
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
            _area_replay(objects, fovs, strategy, incremental=incremental)
    # The same order as numbers: full and incremental replay pick the same pilot.
    fovs["pilot_order"] = fovs["pilot_order"].cat.codes
    full, incremental = (
        _area_replay(objects, fovs, strategy),
        _area_replay(objects, fovs, strategy, True),
    )
    for a, b in zip(full, incremental, strict=True):
        pilot_a, pilot_b = a.result.diagnostics.get("pilot"), b.result.diagnostics.get("pilot")
        assert (pilot_a is None) == (pilot_b is None)
        if pilot_a is not None:
            assert pilot_a["cv_objects"] == pytest.approx(pilot_b["cv_objects"])


def test_c6_a_constant_feature_has_no_design_effect_on_either_path():
    objects, fovs = _area_tables([[0.1] * 7, [0.1] * 3, [0.1] * 11, [0.1] * 5])
    strategy = ScalarPrecisionStrategy("area", 0.05, "fov", "point")
    full, incremental = (
        _area_replay(objects, fovs, strategy),
        _area_replay(objects, fovs, strategy, True),
    )
    for a, b in zip(full, incremental, strict=True):
        assert "ratio_design_effect" not in a.result.diagnostics
        assert "ratio_design_effect" not in b.result.diagnostics
        assert set(a.result.diagnostics) == set(b.result.diagnostics)


def test_c7_the_pilot_is_documented_and_reported_as_reselected_by_its_order():
    # Pilot fields of equal size: 2 effective fields (Kish) are needed (audit 4, Q3).
    objects, fovs = _area_tables([[7.0, 8.0], [], [4.0, 6.0], [5.5, 6.5]])
    fovs["plan_rank"] = [3, 4, 1, 2]
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "fov", "point", pilot_fovs=2, order_col="plan_rank"
    )
    steps = [
        item.result.evidence.quantities["pilot_projected_averaging_cv"]
        for item in _area_replay(objects, fovs, strategy)
    ]
    assert steps[1].availability == "not_applicable"
    assert "changes only if a later FOV sorts before a pilot FOV" in steps[1].reason
    assert steps[2].availability == "available"
    doc = " ".join(ScalarPrecisionEstimator.__doc__.split())
    assert "Only when ``order_col`` is the acquisition order is the pilot fixed" in doc
    assert "another order re-selects the pilot" in " ".join(ScalarPrecisionStrategy.__doc__.split())


def test_c10_pilot_fovs_must_be_an_integer():
    with pytest.raises(ValueError, match="pilot_fovs must be an integer"):
        ScalarPrecisionStrategy("area", 0.05, "object", "point", pilot_fovs=2.0, order_col="order")
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=np.int64(2), order_col="order"
    )
    assert type(strategy.pilot_fovs) is int


def test_c13_a_zero_mean_so_far_is_insufficient_as_documented():
    objects, fovs = _area_tables([[0.0, 0.0], [0.0, 0.0]])
    result = _run(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert result.evidence.quantities["averaging_cv"].availability == "insufficient_data"
    doc = " ".join(ScalarPrecisionEstimator.__doc__.split())
    assert "or a zero mean so far (only zeros), give ``insufficient_data``" in doc


def test_c15_the_scalar_precision_design_effect_has_its_own_name():
    rng = np.random.default_rng(0)
    objects, fovs = _area_tables([rng.normal(100 + rng.normal(0, 15), 20, 30) for _ in range(8)])
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


def _replay(objects, fovs, strategies, incremental=False, feature="area"):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=[feature],
        acquisition_order_col="acq",
        objective=POPULATION,
        fov_col="position_id",
        incremental=incremental,
    )


def test_c1_a_feature_constant_so_far_is_not_yet_enough_data_in_every_variant():
    rng = np.random.default_rng(0)
    # Two saturated fields (255, the camera maximum), then fields that vary widely.
    values = [[255.0] * 6, [255.0] * 5] + [list(rng.uniform(50, 255, 8)) for _ in range(4)]
    objects, fovs = _acquisition(values)
    strategies = [
        ScalarPrecisionStrategy("area", 0.05, unit, sd)
        for unit in ("object", "fov")
        for sd in ("point", "upper_95")
    ]
    full = _replay(objects, fovs, strategies)
    incremental = _replay(objects, fovs, strategies, incremental=True)
    for a, b in zip(full, incremental, strict=True):
        qa, qb = (r.result.evidence.quantities["averaging_cv"] for r in (a, b))
        assert qa.availability == qb.availability and a.result.decision == b.result.decision
        if a.step <= 2:
            unit = a.result.parameters["resampling_unit"]
            assert qa.availability == "insufficient_data", (a.step, a.result.strategy)
            if unit == "object":
                assert qa.reason == "no spread among the objects so far"
            elif a.step == 2:  # one FOV with objects is too few for the FOV unit anyway
                assert qa.reason == "no spread among the FOV means so far"
            assert a.result.decision == "continue"
    # Once the fields differ, every variant has a CV again; the FOV unit's upper-95 bootstrap
    # waits until under 5% of its resamples draw only the two saturated fields (#97).
    fov_upper = "scalar_precision_fov_upper95"
    assert all(
        r.result.evidence.value("averaging_cv") > 0
        for r in full
        if r.step == 5 or (r.step == 3 and r.result.strategy != fov_upper)
    )
    # The observation unit names the rows in the reason.
    [cells] = _area_run(
        *_acquisition([[5.0, 5.0], [5.0]]),
        [ScalarPrecisionStrategy("area", 0.05, "object", "point")],
        observation_unit="cell",
    )
    assert cells.evidence.quantities["averaging_cv"].reason == "no spread among the cells so far"


def _field_size_acquisition():
    """Crowded fields hold small objects, sparse ones large objects (a density effect)."""
    sizes = [150, 120, 100, 90, 5, 4, 3, 3, 2, 2, 140, 3]
    means = [5, 5.5, 6, 6, 20, 21, 22, 20, 21, 23, 5, 22]
    rng = np.random.default_rng(2)
    return _acquisition([list(np.abs(rng.normal(m, 0.5, n))) for n, m in zip(sizes, means)])


def test_c2_the_sufficiency_sentence_carries_the_rules_own_warnings():
    objects, fovs = _field_size_acquisition()
    text = _summary(objects, fovs, [ScalarPrecisionStrategy("area", 0.1, "fov", "point")])
    assert "Caution: field size and field mean are related (Spearman rho -0." in text
    assert "point SD, in the group: field size and field mean are related (Spearman rho -0." in text
    # The paper's advice for a between-object CV above 50% is printed too.
    rng = np.random.default_rng(1)
    wide = _acquisition([list(rng.exponential(10.0, 20)) for _ in range(6)])
    text = _summary(*wide, [ScalarPrecisionStrategy("area", 0.05, "object", "point")])
    assert "Caution: the SD between objects exceeds 50% of their mean" in text
    assert "point SD, in the group: the SD between objects exceeds 50% of their mean" in text


def test_c4_text_and_boolean_order_columns_are_refused_like_the_acquisition_order():
    rng = np.random.default_rng(4)
    objects, fovs = _acquisition([list(rng.normal(10, 2, 5)) for _ in range(12)])
    pilot = ScalarPrecisionStrategy("area", 0.02, "object", "point", pilot_fovs=3, order_col="o")
    plateau = DescriptorPlateauStrategy("area", 0.5, 3, "o")
    for strategy in (pilot, plateau):
        # "1".."12" read from file names: as text, "10" would sort before "2".
        text = fovs.assign(o=(fovs["acq"].astype(int)).astype(str))
        with pytest.raises(ValueError, match="text would sort '10' before '2'"):
            _area_run(objects, text, [strategy])
        with pytest.raises(ValueError, match="holds bool values"):
            _area_run(objects, fovs.assign(o=fovs["acq"] > 6), [strategy])
        with pytest.raises(TypeError, match="categorical"):
            _area_run(objects, fovs.assign(o=pd.Categorical(fovs["acq"])), [strategy])
        # Numbers and times are an order.
        _area_run(objects, fovs.assign(o=fovs["acq"]), [strategy])
        times = pd.Timestamp("2026-01-01") + pd.to_timedelta(fovs["acq"], unit="min")
        _area_run(objects, fovs.assign(o=times), [strategy])
    # The replay refuses the same column with the same rule.
    with pytest.raises(ValueError, match="text would sort '10' before '2'"):
        evaluate_acquisition_prefixes(
            objects,
            fovs.assign(acq=fovs["acq"].astype(int).astype(str)),
            strategies=[pilot.__class__("area", 0.02, "object", "point")],
            feature_cols=["area"],
            acquisition_order_col="acq",
            objective=POPULATION,
            fov_col="position_id",
        )


def test_c7_the_field_size_p_value_is_exact_for_few_fovs():
    counts = np.array([2.0, 3.0, 4.0, 5.0])
    found = _field_size_check(np.array([1.0, 1.1, 1.2, 1.3]) * counts, counts)
    # A perfect rank order of 4 FOVs: 2 of the 24 pairings are as extreme, not p = 0.
    assert found["field_size_correlation"] == pytest.approx(1.0)
    assert found["field_size_correlation_p"] == pytest.approx(2 / 24)
    assert "field_size_warning" not in found
    # The exact p-value is the share of pairings at least as extreme, ties included.
    sizes = np.array([3.0, 3.0, 7.0, 9.0, 12.0, 20.0])
    means = np.array([2.0, 1.5, 1.4, 1.1, 1.1, 0.7])
    rho = stats.spearmanr(sizes, means).statistic
    pairings = [stats.spearmanr(sizes, p).statistic for p in itertools.permutations(means)]
    exact = np.mean(np.abs(pairings) >= abs(rho) - 1e-9)
    found = _field_size_check(means * sizes, sizes)
    assert found["field_size_correlation_p"] == pytest.approx(exact)
    # Under independence the warning rate stays at or below its 5% level.
    rng = np.random.default_rng(0)
    for k in (4, 5, 6):
        warned = 0
        for _ in range(1000):
            c, m = rng.integers(1, 100, k).astype(float), rng.normal(10, 1, k)
            warned += "field_size_warning" in _field_size_check(m * c, c)
        assert warned / 1000 <= 0.05


def test_c9_the_pilot_plan_is_not_claimed_to_stop_exactly_at_required_fovs():
    assert "reaches the target exactly" not in _words(scalar_precision.__doc__)
    # At a rounding boundary the projection needs one FOV more than required_fovs says.
    assert required_fovs(0.05, 2, 0.01) == 50
    assert 0.05 * math.sqrt(2 / 50) > 0.01 >= 0.05 * math.sqrt(2 / 51)
    assert "one FOV later" in required_fovs.__doc__


def test_c11_the_fov_unit_rule_names_the_rows_it_averages():
    rng = np.random.default_rng(4)
    objects, fovs = _acquisition([list(np.abs(rng.normal(10, 8, 12))) for _ in range(8)])
    [result] = _area_run(
        objects,
        fovs,
        [ScalarPrecisionStrategy("area", 0.05, "fov", "point")],
        observation_unit="patch",
    )
    diagnostics = result.evidence.diagnostics
    assert diagnostics["scalar_parameter"] == "mean of 'area' over patches"
    assert diagnostics["warning"].startswith("the SD between patches exceeds 50% of their mean")
    assert "cell" not in diagnostics["warning"]


def test_j12_j16_modules_declare_their_public_and_shared_private_names():
    for module in (
        base,
        baselines,
        scalar_precision,
        stabilization,
        fov_bootstrap,
        jcb_planning,
        _averaging_cv,
    ):
        assert module.__all__, module.__name__
        for name in module.__all__:
            assert not name.startswith("_") and hasattr(module, name), (module.__name__, name)
    assert "Shared-private: ``_field_size_check``" in _averaging_cv.__doc__
    assert "Shared-private: ``_as_ints``" in base.__doc__


def test_e3_the_median_stop_says_how_many_orders_met_it():
    report = _report_both_orders(
        "neighbour_correlation",
        [ScalarPrecisionStrategy("feature_a", 0.01, "fov", "point")],
        orders=PERMUTED_NINE,
    )
    answer = report.enough["answer"].iloc[0]
    stops = report.stops.iloc[0]
    met = int(stops["orders"] - stops["never_stopped"])
    assert 0 < met < 9
    assert f"in {met} of 9 random orders, first met after" in answer
    assert "never in" not in answer and "over 9 orders" not in answer


def test_e6_effort_rules_say_target_reached_not_yes():
    report = _report_both_orders(
        "homogeneous",
        [FixedFOVStrategy(3), ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point")],
    )
    enough = report.enough.set_index("criterion")["sampled_enough"]
    assert enough["fixed_fov"] == "target reached"
    assert enough["scalar_precision_fov"] == "yes"
    assert not pd.isna(report.enough.set_index("criterion")["fovs_beyond_enough"]["fixed_fov"])


def test_p3_1_the_per_field_bound_is_a_whole_field_bootstrap():
    # The Kish/chi-square bound was replaced by a bootstrap of whole fields (#97; owner,
    # 2026-10-05); Kish's effective number of fields still gates the rule.
    sums, counts = np.array([300.0, 520.0, 60.0, 90.0]), np.array([30.0, 50.0, 5.0, 10.0])
    found = fov_averaging_cv(sums, counts)
    kish = counts.sum() ** 2 / (counts**2).sum()
    assert found["effective_fovs"] == statistics.kish_effective_fields(counts)
    assert found["effective_fovs"] == round(kish, 9) and 2 < kish < 4
    assert found["cv_mean_upper_95"] > found["cv_mean"]
    # Seeded: the same fields give the same bound, in any order; another seed moves it a little.
    shuffled = fov_averaging_cv(sums[::-1], counts[::-1])
    assert shuffled["cv_mean_upper_95"] == found["cv_mean_upper_95"]
    other = fov_averaging_cv(sums, counts, random_state=1)["cv_mean_upper_95"]
    assert other == pytest.approx(found["cv_mean_upper_95"], rel=0.2)
    # Two fields: half the resamples draw one field twice, so no bound yet; nor with one
    # dominant field (Kish below 2).
    assert math.isnan(fov_averaging_cv(sums[:2], counts[:2])["cv_mean_upper_95"])
    assert math.isnan(
        fov_averaging_cv(np.array([1000.0, 5.0]), np.array([10.0, 1.0]))["cv_mean_upper_95"]
    )
    # The docstrings name the former bound's coverage and the bootstrap's.
    doc = _joined_words(_averaging_cv.fov_averaging_cv.__doc__)
    assert "covered 70-85% with over-dispersed field sizes" in doc
    assert "Coverage of the true averaging CV" in _joined_words(
        _averaging_cv._bootstrap_upper.__doc__
    )


def test_p4_3_one_dominant_field_is_not_enough_data_for_the_per_field_rule():
    rng = np.random.default_rng(0)
    crowded, sparse = list(rng.normal(100, 20, 1000)), list(rng.normal(88, 20, 5))
    even = [list(rng.normal(70, 14, 20)) for _ in range(4)]
    objects, fovs = _ordered_acquisition([crowded, sparse, *even])
    rules = [ScalarPrecisionStrategy("x", 0.05, "fov", sd) for sd in ("point", "upper_95")]
    full, incremental = _x_replay(objects, fovs, rules), _x_replay(objects, fovs, rules, True)
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
    objects, fovs = _ordered_acquisition([sparse, *even])
    [result] = _x_run(objects, fovs, rules[1:])
    assert _quantity(result, "averaging_cv").availability == "available"


def test_p4_12_the_cv_criterion_asks_for_background_subtracted_values():
    doc = _joined_words(scalar_precision.__doc__)
    assert "intensities must be background-subtracted" in doc
    assert "exchangeable draws of the well's positions" in doc


def test_p4_13_a_point_estimate_from_few_fields_carries_a_caution():
    rng = np.random.default_rng(7)
    values = [list(rng.normal(10 + rng.normal(0, 0.3), 1, 20)) for _ in range(5)]
    point = ScalarPrecisionStrategy("x", 0.05, "fov", "point")
    upper = ScalarPrecisionStrategy("x", 0.05, "fov", "upper_95")
    [few, bounded] = _x_run(*_ordered_acquisition(values[:3]), [point, upper])
    assert few.evidence.diagnostics["warning"].startswith(
        "the point estimate from 3 FOVs with objects is optimistic; the paper advises the "
        "upper-95 bound for small samples"
    )
    assert "warning" not in bounded.evidence.diagnostics
    [enough] = _x_run(*_ordered_acquisition(values), [point])
    assert "warning" not in enough.evidence.diagnostics


def test_p3_7_a_readout_far_from_normal_carries_a_caution_on_the_per_object_bound():
    rng = np.random.default_rng(8)
    mixed = [_mixture(rng, 40) for _ in range(4)]
    normal = [list(rng.normal(10, 2, 40)) for _ in range(4)]
    upper = ScalarPrecisionStrategy("x", 0.05, "object", "upper_95")
    point = ScalarPrecisionStrategy("x", 0.05, "object", "point")
    [flagged, unflagged] = _x_run(*_ordered_acquisition(mixed), [upper, point])
    caution = flagged.evidence.diagnostics["warning"]
    assert "the upper-95 SD bound assumes normal values, and these objects" in caution
    # The point SD is not a bound: only the 50% CV caution.
    assert "normal values" not in unflagged.evidence.diagnostics["warning"]
    [plain] = _x_run(*_ordered_acquisition(normal), [upper])
    assert "warning" not in plain.evidence.diagnostics
    # The incremental path merges the higher moments to the same caution.
    objects, fovs = _ordered_acquisition(mixed)
    full, incremental = _x_replay(objects, fovs, [upper]), _x_replay(objects, fovs, [upper], True)
    assert [r.result.evidence.diagnostics.get("warning") for r in full] == [
        r.result.evidence.diagnostics.get("warning") for r in incremental
    ]
    n, m2, m3, m4 = _averaging_cv._central_sums(np.concatenate([np.array(v) for v in mixed]))
    pooled = np.concatenate([np.array(v) for v in mixed])
    assert f"excess kurtosis {stats.kurtosis(pooled, bias=False):.1f}" in caution
    assert _averaging_cv._shape_warning(n, m2, m3, m4, "objects") in caution


def test_p3_11_the_field_size_check_says_no_warning_is_not_evidence():
    phrase = "no warning is not evidence that"
    assert phrase in _joined_words(_averaging_cv._field_size_check.__doc__)
    assert phrase in _joined_words(ScalarPrecisionEstimator.__doc__)


def test_p2_12_a_missing_value_names_its_field_and_object_and_what_to_do():
    rng = np.random.default_rng(10)
    values = [list(rng.normal(10, 1, 6)) for _ in range(5)]
    values[1][2] = float("nan")
    values[3][0] = float("nan")
    objects, fovs = _ordered_acquisition(values)
    rule = ScalarPrecisionStrategy("x", 0.05, "fov", "upper_95")
    full, incremental = _x_replay(objects, fovs, [rule]), _x_replay(objects, fovs, [rule], True)
    expected = (
        "missing values in the selected scalar feature: object 3 of FOV 'position_id=p1' has "
        "none{}; the rule stays undefined for this group until the value is filled in, since "
        "values are never imputed or dropped"
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


def test_p2_4_object_rule_line_states_its_assumption():
    objects, fovs = _raster_well(drift=0.0)
    session = OnlineSession(
        strategies=[
            ScalarPrecisionStrategy(READOUT, 0.05, "object", "upper_95"),
            ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
        ],
        feature_cols=[READOUT],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )
    first = fovs.iloc[0]
    rows = objects[objects["image_file"].eq(first["image_file"])]
    per_object, per_field = session.add_fov(rows, first.to_dict())
    clause = "treats observations as independent draws: assumes a spatially uniform sample"
    assert clause in per_object.explain()
    assert clause not in per_field.explain()


def test_p2_14_rule_deciding_one_target_reads_decides_only():
    objects, fovs = _raster_well(drift=0.0, rare_rows=2)
    setup = Setup(
        readout=READOUT,
        rare=RareTarget(
            fov_incidence=0.2, detection_probability=0.8, min_fov_occurrences=1, phenotype="rare"
        ),
        rare_label="rare",
        rules=(
            ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
            RareIncidenceStrategy(label_col="rare", prior_a=1.0, prior_b=1.0, max_risk=0.3),
        ),
    )
    last = _live(setup, objects, fovs)[-1]
    population, rare = (r.explain() for r in last.results)
    assert population.startswith("stop (decides the population summary only): averaging_cv")
    assert rare.startswith("stop (decides the rare phenotype only): rare_incidence_risk")
    assert "unsupported" not in population + rare


def test_c1_one_field_with_objects_never_stops_the_per_field_rule():
    spread = list(np.random.default_rng(0).uniform(1, 100, 50))
    for sd in ("point", "upper_95"):
        for empties in (1, 2):
            result = _run_one(
                [spread] + [[]] * empties, ScalarPrecisionStrategy("area", 0.05, "fov", sd)
            )
            assert result.decision is Decision.CONTINUE
            quantity = result.evidence.quantities["averaging_cv"]
            assert quantity.availability is Availability.INSUFFICIENT_DATA
    # Two fields with objects and an empty one still give a value (fields of equal size: 2
    # effective fields, Kish, audit 4, Q3).
    halves = [spread[:25], [], spread[25:]]
    result = _run_one(halves, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert result.evidence.value("averaging_cv") > 0


def test_c3_a_pilot_order_with_missing_values_is_refused():
    objects, fovs = _area_values_tables(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order=[1.0, np.nan, 2.0]
    )
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=2, order_col="order"
    )
    with pytest.raises(ValueError, match="missing values; the pilot needs an order"):
        run_strategies(
            objects, fovs=fovs, strategies=[strategy], feature_cols=["area"], objective=POPULATION
        )


def test_c5_zeros_so_far_are_not_enough_data_rather_than_undefined():
    objects, fovs = _area_values_tables([[0, 0, 0], [0, 0], [1, 2, 0, 3], [2, 1, 1]], acq=range(4))
    strategy = ScalarPrecisionStrategy("area", 0.05, "object", "point")
    steps = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["area"],
        acquisition_order_col="acq",
        objective=POPULATION,
    )
    first = steps[0].result
    assert first.evidence.quantities["averaging_cv"].availability is Availability.INSUFFICIENT_DATA
    assert first.decision is Decision.CONTINUE
    assert steps[2].result.evidence.value("averaging_cv") > 0


def test_c5_negative_values_are_not_applicable_from_the_first_object():
    result = _run_one([[-1.0]], ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert result.evidence.quantities["averaging_cv"].availability is Availability.NOT_APPLICABLE


def test_the_object_unit_jcb_verdict_states_its_independence_assumption():
    # On 3 homogeneous FOVs the published rule is met after 1 FOV; the sentence
    # must say what makes that possible, and the FOV-unit variant must not.
    from morphocoverage import ScalarPrecisionStrategy

    built = SCENARIOS["homogeneous"]()
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[
            ScalarPrecisionStrategy("feature_a", 0.05, "object", "upper_95"),
            ScalarPrecisionStrategy("feature_a", 0.05, "fov", "upper_95"),
        ],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    answers = report.enough.set_index("criterion")["answer"]
    clause = "assumes a spatially uniform sample"
    assert clause in answers["scalar_precision_object_upper95"]
    assert clause not in answers["scalar_precision_fov_upper95"]


def test_a_reason_every_group_shares_is_said_once():
    from morphocoverage import ObjectiveConfig, SamplingObjective
    from morphocoverage.benchmark import AcquisitionOrder
    from morphocoverage.strategies import ScalarPrecisionStrategy
    from morphocoverage.sufficiency import sufficiency_report

    objects, fovs = _three()
    rules = [
        ScalarPrecisionStrategy("x", 0.05, "object", "upper_95"),
        ScalarPrecisionStrategy("x", 0.05, "fov", "upper_95"),
    ]
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["x"],
        strategies=rules,
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        orders=[AcquisitionOrder.permuted(0)],
        observation_unit="track",
    )
    full = " ".join(report.summary(detail="full").split())
    assert full.count("is undefined in every group: the object-unit mean-precision") == 1
    assert "undefined (the object-unit" not in full
    assert "(cell, nucleus, object), and these rows are of another unit (track)" in full


def _table_rules(short: str) -> list[str]:
    """The rule rows of question 2's short table."""
    lines = short.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("rule "))
    pad = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    rows = []
    for line in lines[start + 1 :]:
        if not line.startswith(pad) or line[len(pad)] == " " or not line.strip():
            break
        rows.append(line.strip())
    return rows


def test_the_question_2_headline_counts_the_rules_as_the_table_lists_them():
    report = _named_scenario_report(
        "condition_repeat_groups",
        [
            FixedFOVStrategy(2),
            ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point"),
            ScalarPrecisionStrategy("feature_a", 1e-6, "fov", "point"),
        ],
    )
    rows = _table_rules(report.summary())
    status, text = enough_status(report)
    found = re.match(r"(\d+) of (\d+) rules said enough in every group", text)
    assert found, text
    assert int(found.group(2)) == len(rows) == 3
    everywhere = sum(" 4 of 4 " in f" {row} " for row in rows)
    assert int(found.group(1)) == everywhere == 2
    assert "mean-precision rule, per field, 0.0001%, point SD: in 0 of 4" in text
    assert "cautions below" not in text and status is Status.CAUTION


def test_every_headline_of_the_idr0028_like_report_stays_within_two_lines():
    report = _named_scenario_report(
        "condition_repeat_groups",
        [FixedFOVStrategy(2), ScalarPrecisionStrategy("feature_a", 0.5, "fov", "upper_95")],
    )
    glance = microscopist_report(sufficiency=report).summary().split("\n\n")[1]
    answers = re.split(r"\n(?=  \S)", glance)[1:]
    assert answers and all(len(answer.splitlines()) <= 2 for answer in answers)
