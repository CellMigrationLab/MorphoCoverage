"""Tests for leave-one-FOV-out support novelty."""

import json

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS
from scipy.spatial import distance

from morphocoverage import (
    Decision,
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    validate_input_tables,
)
from morphocoverage.evaluator import objects_in_fovs, restrict_input
from morphocoverage.strategies import SupportNoveltyEstimator, SupportNoveltyStrategy
from morphocoverage.strategies.support_novelty import novel_observations
from morphocoverage.transforms import Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _table(values_per_fov, empty_fovs=0):
    """Validated data with one FOV per array in ``values_per_fov`` plus empty FOVs."""
    rows = [
        {"image_file": f"f{j}.tif", "a": a, "b": b}
        for j, values in enumerate(values_per_fov)
        for a, b in values
    ]
    names = [f"f{j}.tif" for j in range(len(values_per_fov) + empty_fovs)]
    return validate_input_tables(
        pd.DataFrame(rows), fovs=pd.DataFrame({"image_file": names}), feature_cols=["a", "b"]
    )


def _novelty(data, epsilon=0.5, n_neighbors=1, transform=None, min_fovs=2):
    estimator = SupportNoveltyEstimator(epsilon, n_neighbors, transform, min_fovs)
    return estimator.estimate(data, objective=POPULATION)


def test_counting_queries_match_brute_force_distances():
    rng = np.random.default_rng(0)
    for _ in range(25):
        counts = rng.integers(1, 12, size=rng.integers(2, 7))
        positions = np.repeat(np.arange(len(counts)), counts)
        values = rng.normal(size=(len(positions), 3))
        epsilon, n_neighbors = rng.uniform(0.3, 1.5), int(rng.integers(1, 4))
        pairwise = distance.cdist(values, values)
        other = positions[:, None] != positions[None, :]
        expected = ((pairwise <= epsilon) & other).sum(axis=1) < n_neighbors
        assert np.array_equal(novel_observations(values, positions, epsilon, n_neighbors), expected)


def test_a_state_seen_in_one_fov_stays_novel_however_many_cells_show_it():
    rng = np.random.default_rng(1)
    common = [rng.normal(0, 0.1, size=(20, 2)) for _ in range(5)]
    rare = rng.normal(5, 0.1, size=(30, 2))
    with_rare = common[:4] + [np.vstack([common[4], rare])]
    fraction = _novelty(_table(with_rare)).quantities["lofo_novel_fraction"].value
    assert fraction == pytest.approx(30 / 130)
    # Twice as many cells of the rare state in the same FOV: still all novel.
    doubled = common[:4] + [np.vstack([common[4], rare, rare])]
    evidence = _novelty(_table(doubled))
    assert evidence.quantities["lofo_novel_fraction"].value == pytest.approx(60 / 160)
    assert evidence.diagnostics["fovs_with_novel_observations"] == 1
    # The same state recurring in a second FOV is covered.
    recurring = common[:3] + [np.vstack([common[3], rare]), np.vstack([common[4], rare])]
    assert _novelty(_table(recurring)).quantities["lofo_novel_fraction"].value == 0


def test_more_neighbours_demand_denser_support():
    rng = np.random.default_rng(2)
    fovs = [rng.normal(size=(15, 2)) for _ in range(6)]
    fractions = [
        _novelty(_table(fovs), n_neighbors=k).quantities["lofo_novel_fraction"].value
        for k in (1, 3, 10)
    ]
    assert fractions == sorted(fractions) and fractions[-1] > fractions[0]


def test_novelty_falls_as_representative_fovs_accumulate():
    built = SCENARIOS["homogeneous"]()
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    ordered = data.fovs.sort_values(ORDER_COL)
    fractions = []
    for k in (3, 8, 20):
        fovs = ordered.iloc[:k]
        prefix = restrict_input(data, objects_in_fovs(data, fovs), fovs)
        result = _novelty(prefix, epsilon=0.3, transform=Standardize())
        fractions.append(result.quantities["lofo_novel_fraction"].value)
    assert fractions[0] > fractions[1] > fractions[2]


def test_zero_observation_fovs_do_not_change_the_fraction():
    rng = np.random.default_rng(3)
    fovs = [rng.normal(size=(10, 2)) for _ in range(4)]
    plain = _novelty(_table(fovs)).quantities["lofo_novel_fraction"].value
    padded = _novelty(_table(fovs, empty_fovs=5)).quantities["lofo_novel_fraction"].value
    assert padded == plain


def test_insufficient_and_not_applicable_cases():
    rng = np.random.default_rng(4)
    one = _novelty(_table([rng.normal(size=(10, 2))], empty_fovs=3))
    assert one.quantities["lofo_novel_fraction"].availability.value == "insufficient_data"
    constant = _table([np.column_stack([np.ones(5), rng.normal(size=5)]) for _ in range(3)])
    unfit = _novelty(constant, transform=Standardize()).quantities["lofo_novel_fraction"]
    assert unfit.availability.value == "insufficient_data" and "transform" in unfit.reason
    missing = _table([rng.normal(size=(5, 2)) for _ in range(3)])
    missing.objects.loc[0, "a"] = np.nan
    assert _novelty(missing).quantities["lofo_novel_fraction"].availability.value == (
        "not_applicable"
    )


def test_strategy_stops_once_novelty_is_within_tolerance():
    built = SCENARIOS["homogeneous"]()
    strategy = SupportNoveltyStrategy(0.5, 0.02, 1, Standardize(), 3)
    results = evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )
    for item in results:
        quantity = item.result.evidence.quantities["lofo_novel_fraction"]
        if quantity.availability.value == "available":
            expected = Decision.STOP if quantity.value <= 0.02 else Decision.CONTINUE
            assert item.result.decision is expected
    assert any(item.result.decision is Decision.STOP for item in results)


@pytest.mark.parametrize(
    "build",
    [
        lambda: SupportNoveltyEstimator(0.0, 1, None, 2),
        lambda: SupportNoveltyEstimator(float("inf"), 1, None, 2),
        lambda: SupportNoveltyEstimator(0.5, 0, None, 2),
        lambda: SupportNoveltyEstimator(0.5, 1, None, 1),
        lambda: SupportNoveltyEstimator(0.5, 1, "standardize", 2),
        lambda: SupportNoveltyStrategy(0.5, 1.0, 1, None, 2),
    ],
)
def test_invalid_settings_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()


def test_parameters_are_serializable_provenance():
    strategy = SupportNoveltyStrategy(0.4, 0.05, 2, Standardize(), 3)
    record = json.loads(json.dumps(strategy.parameters))
    assert record == {
        "epsilon": 0.4,
        "n_neighbors": 2,
        "transform": Standardize().describe(),
        "min_fovs": 3,
        "tolerance": 0.05,
    }
    assert strategy.config_id != SupportNoveltyStrategy(0.3, 0.05, 2, Standardize(), 3).config_id
