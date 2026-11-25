"""Tests for the fixed imaged-area baseline."""

import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_strategies import _area_tables, _run, _run_one
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Decision,
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    validate_input_tables,
)
from morphocoverage.evidence import Availability
from morphocoverage.strategies import FixedAreaStrategy, ImagedAreaEstimator

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _replay(scenario, strategy):
    built = SCENARIOS[scenario]()
    return evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )


def _first_stop(results):
    return next(item.result for item in results if item.result.decision is Decision.STOP)


def test_unequal_fields_make_area_and_fov_count_rules_differ():
    # FOVs alternate between 10,000 and 40,000 um^2, so 100,000 um^2 is
    # reached after 4 FOVs (10k + 40k + 10k + 40k), not 10.
    stop = _first_stop(_replay("unequal_fov_area", FixedAreaStrategy(100_000.0, "area_um2")))
    assert stop.n_fovs == 4
    assert stop.evidence.value("imaged_area") == 100_000.0


def test_zero_observation_fovs_count_as_imaged_area():
    results = _replay("zero_object_fovs", FixedAreaStrategy(50_000.0, "area_um2"))
    assert _first_stop(results).n_fovs == 5


def test_missing_zero_or_negative_areas_are_not_applicable():
    built = SCENARIOS["homogeneous"]()
    estimator = ImagedAreaEstimator("area_um2")
    # A zero-area FOV (external re-audit) would add no effort silently.
    for bad in (np.nan, -1.0, 0.0):
        fovs = built.fovs.copy()
        fovs.loc[0, "area_um2"] = bad
        data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
        quantity = estimator.estimate(data, objective=POPULATION).quantities["imaged_area"]
        assert quantity.availability.value == "not_applicable"


def test_misconfiguration_raises():
    built = SCENARIOS["homogeneous"]()
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    with pytest.raises(ValueError, match="not a column"):
        ImagedAreaEstimator("area").estimate(data, objective=POPULATION)
    objects = pd.DataFrame({"image_file": ["a"], "area_um2": [5.0]})
    fovs = pd.DataFrame({"image_file": ["a"], "area_um2": [5.0]})
    as_feature = validate_input_tables(objects, fovs=fovs, feature_cols=["area_um2"])
    with pytest.raises(ValueError, match="metadata"):
        ImagedAreaEstimator("area_um2").estimate(as_feature, objective=POPULATION)
    for target in (0.0, float("inf"), True):
        with pytest.raises(ValueError):
            FixedAreaStrategy(target, "area_um2")
    with pytest.raises(TypeError):
        ImagedAreaEstimator("")


def test_c11_fixed_area_checks_its_area_column_on_construction():
    with pytest.raises(TypeError, match="area_col"):
        FixedAreaStrategy(100.0, "")
    objects, fovs = _area_tables([[1.0, 2.0], [3.0]])
    fovs["area_um2"] = ["60", "70"]
    with pytest.raises(ValueError, match="not numbers"):
        _run(objects, fovs, FixedAreaStrategy(100.0, "area_um2"))


def test_c4_an_infinite_field_area_is_not_summed():
    result = _run_one([[1.0], [2.0]], FixedAreaStrategy(10.0, "area_um2"), area_um2=[np.inf, 1.0])
    assert result.decision is not Decision.STOP
    assert result.evidence.quantities["imaged_area"].availability is Availability.NOT_APPLICABLE


def test_a_total_area_that_overflows_is_not_applicable_never_a_stop_544():
    # Every area finite and positive, their float64 sum infinite: no measured total, no stop.
    built = SCENARIOS["homogeneous"]()
    fovs = built.fovs.copy()
    fovs["area_um2"] = 1e308
    data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    estimator = ImagedAreaEstimator("area_um2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow warning either
        quantity = estimator.estimate(data, objective=POPULATION).quantities["imaged_area"]
    assert quantity.availability.value == "not_applicable"
    assert "overflow" in quantity.reason
    fovs["area_um2"] = 1e300  # an ordinary finite total is summed as before
    data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    quantity = estimator.estimate(data, objective=POPULATION).quantities["imaged_area"]
    assert quantity.value == pytest.approx(1e300 * len(fovs))
