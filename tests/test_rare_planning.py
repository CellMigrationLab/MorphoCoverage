"""Tests for binomial FOV-incidence planning of a rare target."""

import numpy as np
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS
from scipy import stats

from morphocoverage import (
    Decision,
    EffortPolicy,
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    SamplingStrategy,
    evaluate_acquisition_prefixes,
)
from morphocoverage.strategies import FOVIncidencePlanEstimator, FOVIncidencePlanStrategy
from morphocoverage.strategies.rare_planning import (
    _PlannedFOVPolicy,
    first_detection_fovs,
    fovs_to_detect,
)


def _objective(kind=SamplingObjective.RARE_DISCOVERY, **target):
    settings = {"detection_probability": 0.9, "min_fov_occurrences": 1, "phenotype": "label"}
    settings.update(target)
    return ObjectiveConfig(kind, rare=RareTarget(**settings))


def _replay(strategy, objective, scenario="homogeneous"):
    built = SCENARIOS[scenario]()
    return evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=objective,
    )


@pytest.mark.parametrize("p", [0.5, 0.1, 0.01, 0.001])
@pytest.mark.parametrize("beta", [0.5, 0.8, 0.95, 0.99])
def test_one_occurrence_matches_the_closed_form(p, beta):
    assert fovs_to_detect(p, beta, 1) == first_detection_fovs(p, beta)


@pytest.mark.parametrize("p, beta, r", [(0.01, 0.95, 2), (0.1, 0.95, 3), (0.3, 0.5, 5)])
def test_the_plan_is_the_smallest_sufficient_fov_count(p, beta, r):
    n = fovs_to_detect(p, beta, r)
    assert stats.binom.sf(r - 1, n, p) >= beta
    assert stats.binom.sf(r - 1, n - 1, p) < beta


def test_published_style_example():
    # A state in 1% of FOVs needs 299 independent FOVs for 95% first detection.
    assert fovs_to_detect(0.01, 0.95, 1) == 299


@pytest.mark.parametrize("args", [(0, 0.9, 1), (0.1, 1.0, 1), (0.1, 0.9, 0), (0.1, 0.9, True)])
def test_invalid_plans_are_refused(args):
    with pytest.raises(ValueError):
        fovs_to_detect(*args)


def test_strategy_stops_at_the_planned_fov_count():
    objective = _objective(fov_incidence=0.3)  # ceil(log(0.1) / log(0.7)) = 7
    results = _replay(FOVIncidencePlanStrategy(), objective)
    decisions = [item.result.decision for item in results]
    assert decisions[:6] == [Decision.CONTINUE] * 6
    assert set(decisions[6:]) == {Decision.STOP}
    assert results[0].result.evidence.value("planned_fovs_for_rare_target") == 7
    assert results[0].result.component_decisions == {"rare": Decision.CONTINUE}


def test_object_prevalence_is_never_converted_to_fov_incidence():
    results = _replay(FOVIncidencePlanStrategy(), _objective(object_prevalence=0.01))
    quantity = results[0].result.evidence.quantities["planned_fovs_for_rare_target"]
    assert quantity.availability.value == "not_applicable"
    assert "object prevalence only" in quantity.reason
    assert {item.result.decision for item in results} == {None}


def test_population_objectives_are_unsupported():
    results = _replay(FOVIncidencePlanStrategy(), ObjectiveConfig(SamplingObjective.POPULATION))
    assert {item.result.decision for item in results} == {Decision.UNSUPPORTED}


def test_balanced_objectives_need_the_population_component_too():
    strategy = SamplingStrategy(
        [FOVIncidencePlanEstimator()], [_PlannedFOVPolicy(), EffortPolicy("n_fovs", 10)]
    )
    results = _replay(strategy, _objective(SamplingObjective.BALANCED, fov_incidence=0.3))
    first_stop = next(i for i, item in enumerate(results) if item.result.decision is Decision.STOP)
    assert results[first_stop].result.n_fovs == 10


def test_zero_object_fovs_count_as_sampled_positions():
    objective = _objective(fov_incidence=0.3)
    results = _replay(FOVIncidencePlanStrategy(), objective, "zero_object_fovs")
    first_stop = next(item.result for item in results if item.result.decision is Decision.STOP)
    assert first_stop.n_fovs == 7


def test_spatially_clustered_states_make_the_binomial_plan_anti_conservative():
    # Counterexample for the independence assumption, kept as a regression
    # guard on the documentation: the state occupies runs of 10 neighbouring
    # FOVs, one run per 100 (10% FOV incidence). Acquiring the planned 29
    # contiguous FOVs from a random start detects it far less often than 95%.
    incidence, runs = 0.1, 10
    line = np.zeros(1000, dtype=bool)
    for start in range(0, 1000, 100):
        line[start : start + runs] = True
    assert line.mean() == incidence
    n = fovs_to_detect(incidence, 0.95, 1)
    windows = np.lib.stride_tricks.sliding_window_view(np.concatenate([line, line[:n]]), n)
    detected = windows[: len(line)].any(axis=1).mean()
    assert n == 29 and detected < 0.5
    # Independent FOVs at the same incidence meet the target.
    assert 1 - (1 - incidence) ** n >= 0.95
