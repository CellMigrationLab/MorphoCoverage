"""Tests for beta-binomial rare-state incidence evidence."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS
from scipy import stats

from morphocoverage import (
    Decision,
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    validate_input_tables,
)
from morphocoverage.strategies import RareIncidenceEstimator, RareIncidenceStrategy
from morphocoverage.strategies.rare_incidence import incidence_risk
from morphocoverage.strategies.rare_planning import fovs_to_detect


def _objective(**target):
    settings = {"detection_probability": 0.95, "min_fov_occurrences": 1, "phenotype": "state"}
    settings.update(target)
    return ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=RareTarget(**settings))


def _labelled(positive_per_fov, cells_per_fov=5):
    """FOVs in acquisition order; ``positive_per_fov[j]`` rare cells in FOV j."""
    rows = []
    for j, positives in enumerate(positive_per_fov):
        for i in range(cells_per_fov):
            rows.append({"image_file": f"f{j}.tif", "x": float(i), "rare": int(i < positives)})
    objects = pd.DataFrame(rows)
    fovs = pd.DataFrame(
        {
            "image_file": [f"f{j}.tif" for j in range(len(positive_per_fov))],
            "order": range(len(positive_per_fov)),
        }
    )
    return objects, fovs


def _replay(objects, fovs, strategy, objective):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["x"],
        acquisition_order_col="order",
        objective=objective,
    )


def test_risk_is_the_beta_posterior_tail():
    assert incidence_risk(1, 10, 0.2, 2, 0.5, 0.5) == pytest.approx(stats.beta.sf(0.2, 1.5, 9.5))
    assert incidence_risk(2, 10, 0.2, 2, 0.5, 0.5) == 0.0


@pytest.mark.parametrize("p, beta", [(0.3, 0.9), (0.1, 0.95), (0.05, 0.99)])
def test_uniform_prior_without_detections_stops_one_fov_before_the_binomial_plan(p, beta):
    # Beta(1, 1 + k) gives risk (1 - p) ** (k + 1), so the stop is at plan - 1.
    plan = fovs_to_detect(p, beta, 1)
    objects, fovs = _labelled([0] * (plan + 5))
    strategy = RareIncidenceStrategy("rare", 1.0, 1.0, 1 - beta)
    results = _replay(
        objects, fovs, strategy, _objective(fov_incidence=p, detection_probability=beta)
    )
    first_stop = next(item.result for item in results if item.result.decision is Decision.STOP)
    assert first_stop.n_fovs == plan - 1
    assert first_stop.evidence.value("rare_incidence_risk") == pytest.approx((1 - p) ** plan)


def test_recurrence_stops_as_soon_as_enough_fovs_show_the_state():
    objects, fovs = _labelled([0, 1, 0, 0, 1, 0, 0, 0])
    strategy = RareIncidenceStrategy("rare", 1.0, 1.0, 0.05)
    results = _replay(
        objects, fovs, strategy, _objective(fov_incidence=0.05, min_fov_occurrences=2)
    )
    first_stop = next(item.result for item in results if item.result.decision is Decision.STOP)
    assert first_stop.n_fovs == 5
    assert first_stop.diagnostics["recurrent"]


def test_many_labelled_cells_in_one_fov_are_one_positive_fov():
    few, fovs = _labelled([1, 0, 0])
    many, _ = _labelled([5, 0, 0])
    objective = _objective(fov_incidence=0.1, min_fov_occurrences=2)
    estimator = RareIncidenceEstimator("rare", 0.5, 0.5)
    for objects in (few, many):
        data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
        evidence = estimator.estimate(data, objective=objective)
        assert evidence.diagnostics["positive_fovs"] == 1
    assert evidence.value("rare_incidence_risk") > 0


def test_a_detected_but_not_recurrent_state_raises_the_risk():
    objective = _objective(fov_incidence=0.1, min_fov_occurrences=2)
    estimator = RareIncidenceEstimator("rare", 1.0, 1.0)
    risks = []
    for pattern in ([0] * 10, [1] + [0] * 9):
        objects, fovs = _labelled(pattern)
        data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
        risks.append(estimator.estimate(data, objective=objective).value("rare_incidence_risk"))
    assert risks[1] > risks[0]


def test_object_prevalence_targets_and_missing_labels_are_not_applicable():
    objects, fovs = _labelled([0, 0, 1])
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    estimator = RareIncidenceEstimator("rare", 1.0, 1.0)
    prevalence = estimator.estimate(data, objective=_objective(object_prevalence=0.01))
    assert prevalence.quantities["rare_incidence_risk"].availability.value == "not_applicable"
    objects.loc[0, "rare"] = np.nan
    missing = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    result = estimator.estimate(missing, objective=_objective(fov_incidence=0.1))
    assert result.quantities["rare_incidence_risk"].availability.value == "not_applicable"


def test_misuse_raises():
    objects, fovs = _labelled([0, 1])
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    objective = _objective(fov_incidence=0.1)
    with pytest.raises(ValueError, match="not a column"):
        RareIncidenceEstimator("label", 1.0, 1.0).estimate(data, objective=objective)
    with pytest.raises(ValueError, match="phenotype feature"):
        RareIncidenceEstimator("x", 1.0, 1.0).estimate(data, objective=objective)
    bad = validate_input_tables(objects.assign(rare=2), fovs=fovs, feature_cols=["x"])
    with pytest.raises(ValueError, match="booleans or 0/1"):
        RareIncidenceEstimator("rare", 1.0, 1.0).estimate(bad, objective=objective)


def test_population_objectives_are_unsupported():
    built = SCENARIOS["rare_uniform"]()
    results = evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[RareIncidenceStrategy("true_state", 1.0, 1.0, 0.05)],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )
    assert {item.result.decision for item in results} == {Decision.UNSUPPORTED}


@pytest.mark.parametrize(
    "build",
    [
        lambda: RareIncidenceEstimator("", 1.0, 1.0),
        lambda: RareIncidenceEstimator("rare", 0.0, 1.0),
        lambda: RareIncidenceEstimator("rare", 1.0, float("inf")),
        lambda: RareIncidenceStrategy("rare", 1.0, 1.0, 1.0),
    ],
)
def test_invalid_settings_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()
