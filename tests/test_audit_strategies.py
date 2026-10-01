"""Regressions of the 2026-09-26 function audit: strategies, transforms and outcomes.

Each test names the finding it pins (the first function audit, docs/audit/README.md)
and fails on the code before the fix.
"""

import numpy as np
import pandas as pd
import pytest

from morphocoverage import (
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    plan_objects_to_average,
    run_strategies,
)
from morphocoverage.evidence import Availability
from morphocoverage.results import Decision
from morphocoverage.strategies import FixedAreaStrategy, FOVBootstrapStrategy
from morphocoverage.strategies.rare_planning import first_detection_fovs, fovs_to_detect
from morphocoverage.transforms import PCA, Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)


def _tables(values_per_fov, **fov_columns):
    rows = [
        {"image_file": f"f{j}.tif", "area": v}
        for j, values in enumerate(values_per_fov)
        for v in values
    ]
    objects = pd.DataFrame(rows, columns=["image_file", "area"])
    fovs = pd.DataFrame(
        {"image_file": [f"f{j}.tif" for j in range(len(values_per_fov))], **fov_columns}
    )
    return objects, fovs


def _run(values_per_fov, strategy, **fov_columns):
    objects, fovs = _tables(values_per_fov, **fov_columns)
    [result] = run_strategies(
        objects, fovs=fovs, strategies=[strategy], feature_cols=["area"], objective=POPULATION
    )
    return result


def test_c1_one_field_with_objects_never_stops_the_per_field_rule():
    spread = list(np.random.default_rng(0).uniform(1, 100, 50))
    for sd in ("point", "upper_95"):
        for empties in (1, 2):
            result = _run(
                [spread] + [[]] * empties, ScalarPrecisionStrategy("area", 0.05, "fov", sd)
            )
            assert result.decision is Decision.CONTINUE
            quantity = result.evidence.quantities["averaging_cv"]
            assert quantity.availability is Availability.INSUFFICIENT_DATA
    # Two fields with objects and an empty one still give a value (fields of equal size: 2
    # effective fields, Kish, audit 4, Q3).
    halves = [spread[:25], [], spread[25:]]
    result = _run(halves, ScalarPrecisionStrategy("area", 0.05, "fov", "point"))
    assert result.evidence.value("averaging_cv") > 0


def test_c2_planning_refuses_a_pilot_of_one_object_per_sample():
    objects = pd.DataFrame(
        {
            "condition": ["control", "control", "drug", "drug"],
            "biological_repeat": ["r1", "r2", "r3", "r4"],
            "image_file": ["a", "b", "c", "d"],
            "area": [10.0, 30.0, 5.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="2 or more cells"):
        plan_objects_to_average(
            objects,
            fovs=objects.drop(columns="area"),
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=("condition", "biological_repeat"),
            control={"condition": "control"},
        )


def test_c3_a_pilot_order_with_missing_values_is_refused():
    objects, fovs = _tables([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order=[1.0, np.nan, 2.0])
    strategy = ScalarPrecisionStrategy(
        "area", 0.05, "object", "point", pilot_fovs=2, order_col="order"
    )
    with pytest.raises(ValueError, match="missing values; the pilot needs an order"):
        run_strategies(
            objects, fovs=fovs, strategies=[strategy], feature_cols=["area"], objective=POPULATION
        )


def test_c4_an_infinite_field_area_is_not_summed():
    result = _run([[1.0], [2.0]], FixedAreaStrategy(10.0, "area_um2"), area_um2=[np.inf, 1.0])
    assert result.decision is not Decision.STOP
    assert result.evidence.quantities["imaged_area"].availability is Availability.NOT_APPLICABLE


def test_c5_zeros_so_far_are_not_enough_data_rather_than_undefined():
    objects, fovs = _tables([[0, 0, 0], [0, 0], [1, 2, 0, 3], [2, 1, 1]], acq=range(4))
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
    result = _run([[-1.0]], ScalarPrecisionStrategy("area", 0.05, "object", "point"))
    assert result.evidence.quantities["averaging_cv"].availability is Availability.NOT_APPLICABLE


def test_c6_a_pca_summary_of_one_feature_is_not_applicable_from_the_start():
    rng = np.random.default_rng(0)
    objects = pd.DataFrame({"image_file": [f"f{j}" for j in range(5) for _ in range(4)]})
    objects["x"] = rng.normal(size=20)
    fovs = pd.DataFrame({"image_file": [f"f{j}" for j in range(5)], "acq": range(5)})
    strategy = FOVBootstrapStrategy("eigenspectrum", 0.1, None, 200, 0, 3)
    steps = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["x"],
        acquisition_order_col="acq",
        objective=DISTRIBUTION,
    )
    for step in steps:
        quantity = step.result.evidence.quantities["fov_bootstrap_eigenspectrum_max_se"]
        assert quantity.availability is Availability.NOT_APPLICABLE


def test_d1_a_constant_feature_is_refused_whatever_its_rounding():
    for constant in (1.0, 0.1, 0.7, 123.456):
        matrix = np.c_[[1.0, 2.0, 3.0], np.full(3, constant)]
        assert isinstance(Standardize().fit_matrix(matrix, ("a", "b")), str)
    fitted = Standardize().fit_matrix(np.c_[[1.0, 2.0, 3.0], [0.1, 0.2, 0.4]], ("a", "b"))
    assert not isinstance(fitted, str)


def test_d1_standardised_outcomes_are_undefined_with_a_constant_reference_feature():
    from morphocoverage.multivariate_outcomes import CentroidError
    from morphocoverage.schema import validate_input_tables

    objects = pd.DataFrame(
        {"image_file": ["f1", "f1", "f2", "f2"], "a": [1.0, 2.0, 3.0, 4.0], "b": [0.1] * 4}
    )
    data = validate_input_tables(
        objects, fovs=objects[["image_file"]].drop_duplicates(), feature_cols=["a", "b"]
    )
    assert CentroidError(("a", "b"), True).evaluate(data, data) is None


def test_d2_pca_refuses_components_beyond_the_data_rank():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    reason = PCA(2).fit_matrix(np.c_[a, 2 * a, -a], ("x", "y", "z"))
    assert isinstance(reason, str) and "fewer than 2 dimensions" in reason
    assert not isinstance(PCA(1).fit_matrix(np.c_[a, 2 * a, -a], ("x", "y", "z")), str)


def test_d5_the_one_occurrence_closed_form_equals_the_plan_everywhere():
    grid = np.round(np.arange(0.01, 0.99, 0.01), 2)
    for p in grid:
        for beta in grid:
            assert first_detection_fovs(p, beta) == fovs_to_detect(p, beta, 1)
