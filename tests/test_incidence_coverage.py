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
from morphocoverage.strategies import IncidenceCoverageEstimator, IncidenceCoverageStrategy
from morphocoverage.strategies.incidence_coverage import (
    chao2,
    incidence_coverage,
    incidence_frequencies,
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
            assert item.result.decision is expected


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
