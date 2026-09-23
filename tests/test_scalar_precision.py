"""Tests for the JCB/Zweifach-style scalar precision baseline."""

import math

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionEstimator,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    run_strategies,
    validate_input_tables,
)
from morphocoverage.strategies.scalar_precision import (
    cell_averaging_cv,
    fov_averaging_cv,
    required_fovs,
    required_objects,
    sd_upper_bound_factor,
)

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
    estimate = cell_averaging_cv(np.array([1.0, 2.0, 3.0, 4.0]))
    cv_cells = np.std([1, 2, 3, 4], ddof=1) / 2.5
    assert estimate["cv_cells"] == pytest.approx(cv_cells)
    assert estimate["cv_mean"] == pytest.approx(cv_cells / 2)


def test_object_unit_required_cells_follows_cv_squared_over_target_squared():
    objects, fovs = _tables([[1.0, 2.0], [3.0, 4.0]])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    cv_cells = np.std([1, 2, 3, 4], ddof=1) / 2.5
    assert result.diagnostics["cv_cells"] == pytest.approx(cv_cells)
    assert required_objects(cv_cells, 0.05) == math.ceil((cv_cells / 0.05) ** 2)
    assert _cv(result) == pytest.approx(cv_cells / 2)
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
    assert required_objects(stop.diagnostics["cv_cells"], 0.0202) <= 100


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
    estimate = cell_averaging_cv(objects["area"].to_numpy())
    assert round(estimate["cv_cells"], 3) == 0.178
    assert round(estimate["cv_cells_upper_95"], 3) == 0.260
    required = {}
    for target in (0.025, 0.05):
        for sd_estimate in ("point", "upper_95"):
            strategy = ScalarPrecisionStrategy("area", target, "object", sd_estimate)
            result = _evaluate(objects, fovs, strategy)
            suffix = "_upper_95" if sd_estimate == "upper_95" else ""
            cv_cells = result.diagnostics["cv_cells" + suffix]
            required[target, sd_estimate] = required_objects(cv_cells, target)
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
    assert "50% CV" in result.diagnostics["warning"]
    objects, fovs = _tables([_cells_with_cv(4, 0.3)])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert "warning" not in result.diagnostics


# --- MorphoCoverage FOV-unit variant ---------------------------------------------


def test_fov_averaging_cv_matches_hand_calculation():
    # FOV sums 2 and 6 over 2 objects each; m = 2, residuals -2 and 2.
    estimate = fov_averaging_cv(np.array([2.0, 6.0]), np.array([2.0, 2.0]))
    assert estimate["cv_mean"] == pytest.approx(math.sqrt(2 / 1 * 8 / 16) / 2)


def test_zero_object_fovs_are_sampled_units_in_the_fov_variant():
    with_empty = fov_averaging_cv(np.array([2.0, 6.0, 0.0]), np.array([2.0, 2.0, 0.0]))
    assert with_empty["cv_mean"] == pytest.approx(math.sqrt(3 / 2 * 8 / 16) / 2)

    objects, fovs = _tables([[1.0, 1.0], [3.0, 3.0], []])
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert _cv(result) == pytest.approx(with_empty["cv_mean"])
    assert result.n_fovs == 3


def test_fov_variant_equals_cell_criterion_when_fov_means_do_not_differ():
    # Identical FOVs carry no between-FOV variance.
    objects, fovs = _tables([[1.0, 3.0]] * 4)
    result = _evaluate(objects, fovs, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert _cv(result) == pytest.approx(0.0)


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
    assert fov_clustered.diagnostics["design_effect"] > 5
    assert _cv(fov_mixed) == pytest.approx(0.0)


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
    assert fov.diagnostics["design_effect"] > 10
    assert 0.2 < fov_h.diagnostics["design_effect"] < 3


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
        assert item.result.diagnostics["pilot"]["cv_cells"] == pytest.approx(cv_pilot)
        assert _cv(item.result, "pilot_projected_averaging_cv") == pytest.approx(
            cv_pilot / math.sqrt(item.result.n_objects)
        )
    first_stop = next(r for r in results if r.result.decision == "stop")
    assert first_stop.result.n_objects == required + (required % 2)  # 2 cells per FOV
    # Stopping at the projected CV is the paper's rule: stop once the planned
    # number of cells has been acquired.
    for item in results[1:]:
        planned = item.result.n_objects >= required
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
    assert result.diagnostics["pilot"]["cv_cells"] == pytest.approx(
        np.std([1, 3, 1, 3], ddof=1) / 2
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
        assert "'slice'" in quantity.reason
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
    assert by_unit["slice"].evidence == by_unit["cell"].evidence
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
