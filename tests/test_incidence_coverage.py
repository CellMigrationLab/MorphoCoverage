"""Tests for incidence-based sample coverage and Chao2 completeness."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Decision,
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    validate_input_tables,
)
from morphocoverage.bounds import upper_bound
from morphocoverage.strategies import IncidenceCoverageEstimator, IncidenceCoverageStrategy
from morphocoverage.strategies.incidence_coverage import (
    chao2,
    expected_new_states,
    fovs_with_unique_states,
    good_turing_estimate,
    incidence_coverage,
    incidence_frequencies,
    prob_new_state,
)

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _labelled(labels_per_fov, empty_fovs=0):
    rows = [
        {"image_file": f"f{j}.tif", "state": label, "x": float(i)}
        for j, labels in enumerate(labels_per_fov)
        for i, label in enumerate(labels)
    ]
    names = [f"f{j}.tif" for j in range(len(labels_per_fov) + empty_fovs)]
    return validate_input_tables(
        pd.DataFrame(rows, columns=["image_file", "state", "x"]),
        fovs=pd.DataFrame({"image_file": names}),
        feature_cols=["x"],
    )


def _estimate(data, min_fovs=2):
    return IncidenceCoverageEstimator("state", min_fovs).estimate(data, objective=POPULATION)


def test_worked_example():
    # 5 FOVs; states seen in 5, 2, 1 and 1 FOVs: Q1 = 2, Q2 = 1, U = 9.
    incidences = np.array([5, 2, 1, 1])
    assert incidence_coverage(incidences, 5) == pytest.approx(1 - (2 / 9) * (8 / 10))
    assert chao2(incidences, 5) == pytest.approx(4 + (4 / 5) * 4 / 2)
    # No doubletons: the bias-corrected form (4/5) * Q1 (Q1 - 1) / 2.
    assert chao2(np.array([5, 1, 1, 3]), 5) == pytest.approx(4 + (4 / 5) * 1)


def test_incidences_count_fovs_not_cells():
    labels = np.array(["a", "a", "a", "b", "a", "c"])
    positions = np.array([0, 0, 0, 1, 2, 2])
    assert list(incidence_frequencies(labels, positions)) == [2, 1, 1]


def test_many_cells_of_one_state_in_one_fov_are_one_incidence():
    few = _labelled([["a", "b"], ["a"], ["a", "b"], ["a", "c"]])
    many = _labelled([["a", "b"], ["a"], ["a", "b"], ["a"] + ["c"] * 50])
    for name in ("incidence_coverage", "chao2_completeness"):
        assert _estimate(few).value(name) == _estimate(many).value(name)
    assert _estimate(many).diagnostics["singletons"] == 1


def test_coverage_is_complete_when_every_state_recurs():
    data = _labelled([["a", "b"], ["a", "b"], ["b"]])
    evidence = _estimate(data)
    assert evidence.value("incidence_coverage") == 1.0
    assert evidence.value("chao2_completeness") == 1.0


def test_estimated_coverage_tracks_true_incidence_coverage():
    # Monte Carlo on independent FOVs: state i appears in an FOV with
    # probability pi_i. True coverage is the incidence share of detected
    # states; the estimator should be close to it on average.
    rng = np.random.default_rng(0)
    pi = np.concatenate([np.full(10, 0.6), np.full(20, 0.1), np.full(30, 0.02)])
    n_fovs, estimated, true, observed, richness = 20, [], [], [], []
    for _ in range(400):
        present = rng.random((n_fovs, len(pi))) < pi
        incidences = present.sum(axis=0)
        seen = incidences > 0
        estimated.append(incidence_coverage(incidences[seen], n_fovs))
        true.append(pi[seen].sum() / pi.sum())
        observed.append(seen.sum())
        richness.append(chao2(incidences[seen], n_fovs))
    assert np.mean(estimated) == pytest.approx(np.mean(true), abs=0.01)
    # Not hollow: many rare states stay unseen, so coverage is well below 1.
    assert np.mean(true) < 0.95
    # Chao2 is a lower-bound estimator: above the observed count, at most the truth.
    assert np.mean(observed) < np.mean(richness) <= len(pi)


def test_zero_observation_fovs_are_sampling_units():
    with_empty = _estimate(_labelled([["a"], ["b"], ["a"]], empty_fovs=3))
    assert with_empty.diagnostics["sampling_units"] == 6
    without = _estimate(_labelled([["a"], ["b"], ["a"]]))
    # Q1 = Q2 = 1, U = 3; T enters the formula through (T - 1) Q1 / ((T - 1) Q1 + 2 Q2).
    assert without.value("incidence_coverage") == pytest.approx(1 - (1 / 3) * (2 / 4))
    assert with_empty.value("incidence_coverage") == pytest.approx(1 - (1 / 3) * (5 / 7))


def test_insufficient_not_applicable_and_misuse():
    assert _estimate(_labelled([["a"]])).quantities["incidence_coverage"].availability.value == (
        "insufficient_data"
    )
    missing = _labelled([["a", None], ["a"]])
    assert _estimate(missing).quantities["chao2_completeness"].availability.value == (
        "not_applicable"
    )
    data = _labelled([["a"], ["b"]])
    with pytest.raises(ValueError, match="not a column"):
        IncidenceCoverageEstimator("cell_type", 2).estimate(data, objective=POPULATION)
    with pytest.raises(ValueError, match="phenotype feature"):
        IncidenceCoverageEstimator("x", 2).estimate(data, objective=POPULATION)


def test_strategy_stops_once_coverage_reaches_the_target():
    built = SCENARIOS["rare_uniform"]()
    strategy = IncidenceCoverageStrategy("true_state", "incidence_coverage", 0.95, 3)
    results = evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )
    for item in results:
        quantity = item.result.evidence.quantities["incidence_coverage"]
        if quantity.availability.value == "available":
            expected = Decision.STOP if quantity.value >= 0.95 else Decision.CONTINUE
            assert item.result.rule_decision is expected


@pytest.mark.parametrize(
    "build",
    [
        lambda: IncidenceCoverageEstimator("", 2),
        lambda: IncidenceCoverageEstimator("state", 1),
        lambda: IncidenceCoverageStrategy("state", "coverage", 0.9, 2),
        lambda: IncidenceCoverageStrategy("state", "incidence_coverage", 0.0, 2),
        lambda: IncidenceCoverageStrategy("state", "incidence_coverage", 1.5, 2),
    ],
)
def test_invalid_settings_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()


def test_expected_new_states_formula_and_edge_cases():
    # 5 FOVs; incidences 5, 2, 1, 1: Q1 = 2, Chao2 adds Q0 = 1.6 unseen states.
    incidences = np.array([5, 2, 1, 1])
    q0 = 1.6
    assert expected_new_states(incidences, 5) == pytest.approx(q0 * 2 / (5 * q0 + 2))
    assert expected_new_states(np.array([3, 2]), 5) == 0.0


def test_expected_new_states_track_what_the_next_fov_reveals():
    # Monte Carlo on independent FOVs: the estimate from 20 FOVs against the
    # states the 21st FOV actually adds. Chao2 is a lower bound, so the
    # estimate may undershoot slightly but not overshoot.
    rng = np.random.default_rng(1)
    pi = np.concatenate([np.full(10, 0.6), np.full(20, 0.1), np.full(30, 0.02)])
    estimated, realized = [], []
    for _ in range(2000):
        present = rng.random((21, len(pi))) < pi
        incidences = present[:20].sum(axis=0)
        seen = incidences > 0
        estimated.append(expected_new_states(incidences[seen], 20))
        realized.append((present[20] & ~seen).sum())
    assert np.mean(estimated) == pytest.approx(np.mean(realized), rel=0.1)
    assert np.mean(estimated) <= np.mean(realized) * 1.02


def test_strategy_can_stop_on_expected_new_states():
    strategy = IncidenceCoverageStrategy("true_state", "expected_new_states_next_fov", 0.1, 3)
    assert strategy.policies[0].quantity == "expected_new_states_next_fov"
    evidence = _estimate(_labelled([["a", "b"], ["a"], ["a", "c"]]))
    quantity = evidence.quantities["expected_new_states_next_fov"]
    assert quantity.direction.value == "lower_is_better" and quantity.value > 0
    with pytest.raises(ValueError):
        IncidenceCoverageStrategy("state", "expected_new_states_next_fov", 0.0, 2)


def test_new_state_probability_formula_and_good_turing_estimate():
    # Incidences 5, 2, 1, 1 in 5 FOVs: Q1 = 2, Q0 = 1.6, q = 2 / (5 * 1.6 + 2).
    incidences = np.array([5, 2, 1, 1])
    q = 2 / (5 * 1.6 + 2)
    assert prob_new_state(incidences, 5) == pytest.approx(1 - (1 - q) ** 1.6)
    assert good_turing_estimate(incidences, 5) == pytest.approx(2 / 5)
    assert prob_new_state(np.array([3, 2]), 5) == 0.0
    assert good_turing_estimate(np.array([1, 1, 1]), 2) == 1.0


@pytest.mark.parametrize("n_fovs", [20, 40])
def test_new_state_probability_matches_the_next_fov_in_simulation(n_fovs):
    # Independent FOVs with known incidences: the Chao-model probability
    # matches how often the next FOV really shows a new state from about 20
    # FOVs, and the Good-Turing estimate is not below it on average.
    rng = np.random.default_rng(2)
    pi = np.concatenate([np.full(10, 0.6), np.full(20, 0.1), np.full(30, 0.02)])
    estimated, bound, realized = [], [], []
    for _ in range(2000):
        present = rng.random((n_fovs + 1, len(pi))) < pi
        incidences = present[:n_fovs].sum(axis=0)
        seen = incidences > 0
        estimated.append(prob_new_state(incidences[seen], n_fovs))
        bound.append(good_turing_estimate(incidences[seen], n_fovs))
        realized.append((present[n_fovs] & ~seen).any())
    assert np.mean(estimated) == pytest.approx(np.mean(realized), abs=0.03)
    assert np.mean(bound) >= np.mean(realized) - 0.01


def test_strategy_can_stop_on_the_new_state_probability():
    strategy = IncidenceCoverageStrategy("true_state", "prob_new_state_next_fov", 0.05, 3)
    assert strategy.policies[0].quantity == "prob_new_state_next_fov"
    with pytest.raises(ValueError):
        IncidenceCoverageStrategy("state", "good_turing_new_state_estimate", 1.5, 2)


def test_two_identical_fovs_leave_a_high_new_state_bound():
    # Regression for the small-sample concern: two FOVs showing the same
    # states give model probabilities of 0, yet say little about a third.
    evidence = _estimate(_labelled([["a", "b"], ["a", "b"]]))
    assert evidence.value("prob_new_state_next_fov") == 0.0
    assert evidence.value("good_turing_new_state_estimate") == 0.0
    assert evidence.value("lofo_new_state_fov_fraction") == 0.0
    assert evidence.value("new_state_fov_upper") == pytest.approx(1 - 0.05 ** (1 / 2))


def test_unique_state_fovs_count_fovs_with_a_singleton_state():
    labels = np.array(["a", "a", "b", "c", "c", "d"])
    positions = np.array([0, 1, 1, 2, 2, 2])
    # b is only in FOV 1, c and d only in FOV 2: two FOVs hold a unique state.
    assert fovs_with_unique_states(labels, positions) == 2
    evidence = _estimate(_labelled([["a"], ["a", "b"], ["c", "c", "d"]], empty_fovs=1))
    assert evidence.value("lofo_new_state_fov_fraction") == pytest.approx(2 / 4)


def test_new_state_bound_covers_the_true_probability_in_simulation():
    # Independent FOVs with known incidences: the 95% bound from 8 FOVs lies
    # above the true chance that the next FOV shows a new state in at least
    # about 95% of acquisitions.
    rng = np.random.default_rng(3)
    pi = np.concatenate([np.full(5, 0.6), np.full(10, 0.05)])
    covered = []
    for _ in range(1000):
        present = rng.random((8, len(pi))) < pi
        seen = present.any(axis=0)
        truth = 1 - np.prod(1 - pi[~seen])
        labels, positions = np.nonzero(present.T)
        count = fovs_with_unique_states(labels, positions)
        covered.append(upper_bound(count, 8, 0.95) >= truth)
    assert np.mean(covered) >= 0.93


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
