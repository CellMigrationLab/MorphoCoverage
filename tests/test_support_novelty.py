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
    run_strategies,
    validate_input_tables,
)
from morphocoverage.evaluator import objects_in_fovs, restrict_input
from morphocoverage.schema import fov_positions
from morphocoverage.strategies import SupportNoveltyEstimator, SupportNoveltyStrategy
from morphocoverage.strategies.support_novelty import novel_against, novel_observations
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


def _novelty(data, epsilon=0.5, min_other_fovs=1, transform=None, min_fovs=2):
    estimator = SupportNoveltyEstimator(epsilon, min_other_fovs, transform, min_fovs)
    return estimator.estimate(data, objective=POPULATION)


def test_counting_queries_match_brute_force_distances():
    rng = np.random.default_rng(0)
    for _ in range(25):
        counts = rng.integers(1, 12, size=rng.integers(2, 7))
        positions = np.repeat(np.arange(len(counts)), counts)
        values = rng.normal(size=(len(positions), 3))
        epsilon, min_other = rng.uniform(0.3, 1.5), int(rng.integers(1, 4))
        near = distance.cdist(values, values) <= epsilon
        expected = np.array(
            [
                len({positions[j] for j in np.flatnonzero(near[i])} - {positions[i]}) < min_other
                for i in range(len(values))
            ]
        )
        assert np.array_equal(novel_observations(values, positions, epsilon, min_other), expected)


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


def test_requiring_more_other_fovs_demands_wider_support():
    rng = np.random.default_rng(2)
    fovs = [rng.normal(size=(15, 2)) for _ in range(6)]
    fractions = [
        _novelty(_table(fovs), min_other_fovs=k).quantities["lofo_novel_fraction"].value
        for k in (1, 3, 5)
    ]
    assert fractions == sorted(fractions) and fractions[-1] > fractions[0]


def test_support_counts_fovs_not_cells_on_both_sides():
    # Regression: counting neighbouring observations let a dense FOV cover a
    # state that a sparse FOV showing the same state could not. The same state
    # in the same two FOVs must now be judged alike, whatever the cell counts.
    rng = np.random.default_rng(5)
    common = [rng.normal(0, 0.1, size=(20, 2)) for _ in range(4)]
    rare = np.array([[5.0, 5.0]])
    fovs = [np.vstack([common[0], rare]), np.vstack([common[1], np.repeat(rare, 10, 0)])]
    fovs += common[2:]
    data = _table(fovs)
    values = data.feature_matrix()
    positions = fov_positions(data, data.fovs)
    rare_rows = values[:, 0] > 4
    for required, novel in ((1, False), (2, True)):
        flags = novel_observations(values, positions, 0.5, required)[rare_rows]
        assert set(flags) == {novel}


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
        lambda: SupportNoveltyEstimator(0.5, 1.5, None, 2),
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
        "min_other_fovs": 2,
        "transform": Standardize().describe(),
        "min_fovs": 3,
        "min_novel_observations": 1,
        "confidence": 0.95,
        "quantity": "lofo_novel_fraction",
        "tolerance": 0.05,
    }
    assert strategy.config_id != SupportNoveltyStrategy(0.3, 0.05, 2, Standardize(), 3).config_id


def test_fov_level_novelty_counts_fovs_and_requires_enough_novel_cells():
    # Five common FOVs, one with a single outlier cell, one with 3 cells of a
    # state no other FOV shows, and two empty FOVs (7 visible FOVs in all).
    rng = np.random.default_rng(6)
    common = [rng.normal(0, 0.1, size=(20, 2)) for _ in range(3)]
    outlier = np.vstack([rng.normal(0, 0.1, size=(20, 2)), [[9.0, 9.0]]])
    state = np.vstack([rng.normal(0, 0.1, size=(20, 2)), rng.normal(5, 0.05, size=(3, 2))])
    data = _table([*common, outlier, state], empty_fovs=2)
    any_novel = _novelty(data).quantities["lofo_novel_fov_fraction"].value
    three = SupportNoveltyEstimator(0.5, 1, None, 2, 3).estimate(data, objective=POPULATION)
    assert any_novel == pytest.approx(2 / 7)
    assert three.value("lofo_novel_fov_fraction") == pytest.approx(1 / 7)


@pytest.mark.parametrize("k, expected", [(2, 1 - 0.05 ** (1 / 2)), (5, 1 - 0.05 ** (1 / 5))])
def test_few_homogeneous_fovs_leave_a_high_upper_bound(k, expected):
    # Identical FOVs: nothing looks novel, yet with only k fields the chance
    # that another is novel cannot be ruled out; the bound says so.
    fov = np.random.default_rng(7).normal(size=(30, 2))
    evidence = _novelty(_table([fov] * k))
    assert evidence.value("lofo_novel_fov_fraction") == 0
    assert evidence.value("lofo_novel_fov_upper") == pytest.approx(expected)


def test_upper_bound_covers_the_true_novelty_probability():
    # Simulated acquisitions where each FOV independently holds a unique
    # state with probability 0.2: the 95% upper bound from k FOVs should lie
    # above 0.2 in at least about 95% of acquisitions.
    rng = np.random.default_rng(8)
    covered = []
    for _ in range(300):
        k = 6
        fovs = []
        for j in range(k):
            cells = rng.normal(0, 0.1, size=(15, 2))
            if rng.random() < 0.2:
                cells = np.vstack([cells, rng.normal(10 + 5 * j, 0.05, size=(2, 2))])
            fovs.append(cells)
        covered.append(_novelty(_table(fovs)).value("lofo_novel_fov_upper") >= 0.2)
    assert np.mean(covered) >= 0.93


def test_new_observations_are_judged_against_the_reference_fovs_only():
    rng = np.random.default_rng(9)
    for _ in range(20):
        counts = rng.integers(1, 8, size=rng.integers(1, 5))
        positions = np.repeat(np.arange(len(counts)), counts)
        reference = rng.normal(size=(len(positions), 2))
        queries = rng.normal(size=(int(rng.integers(0, 10)), 2))
        epsilon, required = rng.uniform(0.2, 1.2), int(rng.integers(1, 3))
        near = distance.cdist(queries, reference) <= epsilon
        expected = [len(set(positions[row])) < required for row in near]
        flags = novel_against(reference, positions, queries, epsilon, required)
        assert list(flags) == expected


def test_a_new_fov_is_novel_only_with_enough_unseen_cells():
    rng = np.random.default_rng(10)
    seen = _table([rng.normal(0, 0.1, size=(20, 2)) for _ in range(4)])
    common = rng.normal(0, 0.1, size=(20, 2))
    one_outlier = np.vstack([rng.normal(0, 0.1, size=(20, 2)), [[9.0, 9.0]]])
    new_state = np.vstack([rng.normal(0, 0.1, size=(20, 2)), rng.normal(5, 0.05, (3, 2))])
    new = _table([common, one_outlier, new_state], empty_fovs=1)
    estimator = SupportNoveltyEstimator(0.5, 1, None, 2, 3)
    assert list(estimator.novel_fovs_against(seen, new)) == [False, False, True, False]
    any_cell = SupportNoveltyEstimator(0.5, 1, None, 2, 1)
    assert list(any_cell.novel_fovs_against(seen, new)) == [False, True, True, False]


def test_new_fovs_use_the_transform_fitted_on_the_seen_fovs():
    # In raw units the new FOV lies far away; standardizing with the seen FOVs'
    # scale must keep it far, whereas refitting on the new FOV would not.
    rng = np.random.default_rng(11)
    seen = _table([rng.normal(0, 1, size=(30, 2)) for _ in range(3)])
    new = _table([rng.normal(20, 1, size=(30, 2))])
    estimator = SupportNoveltyEstimator(1.0, 1, Standardize(), 2, 1)
    assert list(estimator.novel_fovs_against(seen, new)) == [True]
    too_few = _table([rng.normal(size=(5, 2))], empty_fovs=2)
    assert estimator.novel_fovs_against(too_few, new) is None


def test_observation_flags_match_the_estimate():
    rng = np.random.default_rng(12)
    data = _table([rng.normal(size=(15, 2)) for _ in range(5)])
    estimator = SupportNoveltyEstimator(0.4, 1, Standardize(), 2)
    flags = estimator.novel_observation_flags(data)
    evidence = estimator.estimate(data, objective=POPULATION)
    assert flags.sum() == evidence.diagnostics["novel_observations"]
    assert flags.mean() == pytest.approx(evidence.value("lofo_novel_fraction"))
    assert estimator.novel_observation_flags(_table([rng.normal(size=(5, 2))])) is None


def test_new_state_clusters_match_brute_force_single_linkage():
    from scipy.sparse.csgraph import connected_components

    from morphocoverage.strategies.support_novelty import new_state_members

    rng = np.random.default_rng(13)
    for _ in range(30):
        n = int(rng.integers(1, 25))
        values = rng.uniform(0, 3, size=(n, 2))
        positions = rng.integers(0, 3, size=n)
        novel = rng.random(n) < 0.6
        epsilon, size = rng.uniform(0.2, 1.0), int(rng.integers(1, 5))
        expected = np.zeros(n, dtype=bool)
        for fov in np.unique(positions):
            rows = np.flatnonzero(novel & (positions == fov))
            if not len(rows):
                continue
            linked = distance.cdist(values[rows], values[rows]) <= epsilon
            _, labels = connected_components(linked, directed=False)
            expected[rows] = np.bincount(labels)[labels] >= size
        members = new_state_members(values, positions, novel, epsilon, size)
        assert np.array_equal(members, expected)


def test_scattered_outliers_do_not_add_up_to_a_new_state():
    # Regression (audit of this module): three isolated outliers in one FOV
    # counted like a tight 3-cell state in another. Only the cluster counts now.
    rng = np.random.default_rng(14)
    fovs = [rng.normal(0, 0.1, size=(20, 2)) for _ in range(8)]
    fovs[0] = np.vstack([fovs[0], [[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]]])
    fovs[1] = np.vstack([fovs[1], rng.normal((5.0, -5.0), 0.05, size=(3, 2))])
    data = _table(fovs)
    evidence = SupportNoveltyEstimator(0.5, 1, None, 2, 3).estimate(data, objective=POPULATION)
    assert evidence.diagnostics["novel_fovs"] == 1
    assert evidence.value("lofo_novel_fov_fraction") == pytest.approx(1 / 8)
    # Every novel cell still counts for the cell-level fraction.
    assert evidence.diagnostics["novel_observations"] == 6
    counted = SupportNoveltyEstimator(0.5, 1, None, 2, 3).novel_observation_flags(
        data, new_states_only=True
    )
    assert counted.sum() == 3 and set(data.objects.loc[counted, "image_file"]) == {"f1.tif"}
    # A new FOV with scattered outliers is not new either.
    scattered = _table([np.vstack([fovs[2], [[9.0, 0.0], [0.0, 9.0], [-9.0, 0.0]]])])
    estimator = SupportNoveltyEstimator(0.5, 1, None, 2, 3)
    assert list(estimator.novel_fovs_against(_table(fovs[2:]), scattered)) == [False]


def test_cell_weighted_and_fov_weighted_fractions_differ_with_unequal_fovs():
    # External audit: two dense, mutually supporting FOVs of 1,000 cells and
    # eight one-cell FOVs, each in its own unsupported state.
    rows = [(f"d{j}", 0.0, 0.0) for j in range(2) for _ in range(1000)]
    rows += [(f"s{i}", 10.0 * (i + 1), 0.0) for i in range(8)]
    data = validate_input_tables(
        pd.DataFrame(rows, columns=["image_file", "a", "b"]), feature_cols=["a", "b"]
    )
    evidence = SupportNoveltyEstimator(0.5, 1, None, 2).estimate(data, objective=POPULATION)
    assert evidence.value("lofo_novel_fraction") == pytest.approx(8 / 2008)
    assert evidence.value("lofo_mean_fov_novel_fraction") == pytest.approx(0.8)
    assert evidence.value("lofo_novel_fov_fraction") == pytest.approx(0.8)
    # A strategy on the FOV-weighted quantity does not stop; the cell-weighted one does.
    for quantity, stops in (("lofo_novel_fraction", True), ("lofo_mean_fov_novel_fraction", False)):
        strategy = SupportNoveltyStrategy(0.5, 0.01, 1, None, 2, quantity=quantity)
        [result] = run_strategies(
            data.objects, strategies=[strategy], feature_cols=["a", "b"], objective=POPULATION
        )
        assert (result.decision is Decision.STOP) is stops


def test_support_needing_several_cells_matches_brute_force():
    rng = np.random.default_rng(15)
    for _ in range(25):
        counts = rng.integers(1, 12, size=rng.integers(2, 6))
        positions = np.repeat(np.arange(len(counts)), counts)
        values = rng.normal(size=(len(positions), 2))
        queries = rng.normal(size=(int(rng.integers(0, 8)), 2))
        epsilon = rng.uniform(0.3, 1.5)
        others, support = int(rng.integers(1, 3)), int(rng.integers(1, 4))

        def supporters(rows_near, support=support):
            fovs, per_fov = np.unique(rows_near, return_counts=True)
            return set(fovs[per_fov >= support])

        near = distance.cdist(values, values) <= epsilon
        expected = [
            len(supporters(positions[near[i]]) - {positions[i]}) < others
            for i in range(len(values))
        ]
        flags = novel_observations(values, positions, epsilon, others, support)
        assert list(flags) == expected
        near_q = distance.cdist(queries, values) <= epsilon
        expected_q = [len(supporters(positions[row])) < others for row in near_q]
        assert list(novel_against(values, positions, queries, epsilon, others, support)) == (
            expected_q
        )


def test_one_stray_cell_elsewhere_does_not_cancel_a_new_state():
    # Regression (re-audit of this module): one cell near a tight 5-cell state
    # in another FOV used to cover the whole state, and the state covered it.
    rng = np.random.default_rng(16)
    fovs = [rng.normal(0, 0.1, size=(20, 2)) for _ in range(8)]
    fovs[0] = np.vstack([fovs[0], rng.normal((5.0, 5.0), 0.05, size=(5, 2))])
    fovs[1] = np.vstack([fovs[1], [[5.0, 5.0]]])
    estimator = SupportNoveltyEstimator(0.5, 1, None, 2, 3)
    evidence = estimator.estimate(_table(fovs), objective=POPULATION)
    assert evidence.diagnostics["novel_fovs"] == 1
    counted = estimator.novel_observation_flags(_table(fovs), new_states_only=True)
    assert counted.sum() == 5
    # A state that genuinely recurs as a group in another FOV is still covered.
    fovs[1] = np.vstack([fovs[1], rng.normal((5.0, 5.0), 0.05, size=(3, 2))])
    assert estimator.estimate(_table(fovs), objective=POPULATION).diagnostics["novel_fovs"] == 0


def test_a_held_out_fov_does_not_shape_the_geometry_that_judges_it():
    # External audit: fitted on all visible FOVs, the extreme FOV inflated the
    # standard deviation to about 50 and sat only ~2 SD from the others, so it
    # was not novel at epsilon 2.1. Fitted without it, it is ~1,000 SD away.
    objects = pd.DataFrame({"image_file": ["a", "b", "c", "novel"], "x": [-0.1, 0.0, 0.1, 100.0]})
    one = validate_input_tables(objects, feature_cols=["x"])
    estimator = SupportNoveltyEstimator(2.1, 1, Standardize(), 2)
    assert list(estimator.novel_observation_flags(one)) == [False, False, False, True]
    evidence = estimator.estimate(one, objective=POPULATION)
    # Held out, the three ordinary FOVs are judged against folds that include
    # the extreme FOV and are covered by each other; only "novel" is new.
    assert evidence.diagnostics["novel_fovs"] == 1
    assert evidence.diagnostics["transform_folds"] == 4
    # The LOFO verdict matches judging it as a genuinely new FOV.
    seen = validate_input_tables(objects.iloc[:3], feature_cols=["x"])
    new = validate_input_tables(objects.iloc[3:], feature_cols=["x"])
    assert list(estimator.novel_fovs_against(seen, new)) == [True]


def test_nested_folds_that_cannot_be_fitted_make_the_estimate_insufficient():
    # One FOV carries all the spread of a feature: without it the feature is
    # constant, so the fold cannot be standardized.
    objects = pd.DataFrame(
        {"image_file": ["a", "a", "b", "c"], "x": [0.0, 1.0, 5.0, 5.0], "y": [0.0, 1.0, 2.0, 3.0]}
    )
    data = validate_input_tables(objects, feature_cols=["x", "y"])
    quantity = (
        SupportNoveltyEstimator(0.5, 1, Standardize(), 2)
        .estimate(data, objective=POPULATION)
        .quantities["lofo_novel_fov_fraction"]
    )
    assert quantity.availability.value == "insufficient_data"
    assert "without one FOV" in quantity.reason


def test_every_supporting_fov_count_path_matches_one_tree_per_fov():
    """The dense, single-tree and per-FOV paths give the reference per-FOV counts."""
    from scipy.spatial import cKDTree

    from morphocoverage.strategies import support_novelty as sn

    def reference_counts(reference, positions, queries, epsilon, m, exclude):
        supporting = np.zeros(len(queries), dtype=np.int64)
        for fov in np.unique(positions):
            members = reference[positions == fov]
            if len(members) < m or not len(queries):
                continue
            counts = cKDTree(members).query_ball_point(queries, epsilon, return_length=True)
            near = counts >= m
            if exclude is not None:
                near &= exclude != fov
            supporting += near
        return supporting

    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.integers(5, 5000 if trial % 5 == 0 else 300))
        d = int(rng.choice([1, 2, 5, 12]))
        reference = rng.normal(size=(n, d))
        positions = rng.integers(0, int(rng.integers(1, 30)), n)
        same = trial % 2 == 0
        queries = reference if same else rng.normal(size=(int(rng.integers(0, 200)), d))
        epsilon = float(rng.choice([0.2, 0.6, np.sqrt(d)]))
        m = int(rng.integers(1, 4))
        exclude = positions if same else None
        expected = reference_counts(reference, positions, queries, epsilon, m, exclude)
        got = sn._supporting_fovs(reference, positions, queries, epsilon, m, exclude)
        np.testing.assert_array_equal(got, expected)


def test_counting_paths_agree_at_exact_ties(monkeypatch):
    """Epsilon equal to data distances: every path applies one rule, ``<= epsilon``."""
    from scipy.spatial import distance

    from morphocoverage.strategies import support_novelty as sn

    def exact(reference, positions, queries, epsilon, m, exclude):
        within = distance.cdist(queries, reference) <= epsilon
        supporting = np.zeros(len(queries), dtype=np.int64)
        for fov in np.unique(positions):
            near = within[:, positions == fov].sum(axis=1) >= m
            if exclude is not None:
                near &= exclude != fov
            supporting += near
        return supporting

    paths = {
        "dense": {"_DENSE_PAIRS": 10**12},
        "one_tree": {"_DENSE_PAIRS": 0, "_LIST_NEIGHBOURS_UP_TO": 10**9},
        "per_fov": {"_DENSE_PAIRS": 0, "_LIST_NEIGHBOURS_UP_TO": -1},
    }
    rng = np.random.default_rng(3)
    for d in (2, 5):
        reference = rng.normal(size=(120, d))
        positions = rng.integers(0, 8, len(reference))
        # Every epsilon is a distance between two observations: ties by design.
        pairs = distance.pdist(reference)
        for epsilon in rng.choice(pairs, 6, replace=False):
            for m in (1, 2):
                expected = exact(reference, positions, reference, epsilon, m, positions)
                for settings in paths.values():
                    for name, value in settings.items():
                        monkeypatch.setattr(sn, name, value)
                    got = sn._supporting_fovs(
                        reference, positions, reference, epsilon, m, positions
                    )
                    np.testing.assert_array_equal(got, expected)
                    monkeypatch.undo()
            # The leave-one-FOV-out count without listing follows the same rule.
            others = sn._other_fov_observations(reference, positions, epsilon)
            assert np.array_equal(
                others >= 1, exact(reference, positions, reference, epsilon, 1, positions) >= 1
            )


def test_non_finite_transformed_values_raise_on_every_path():
    from morphocoverage.strategies import support_novelty as sn

    reference = np.array([[0.0, 1.0], [np.inf, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        sn._supporting_fovs(reference, np.array([0, 1]), reference, 1.0, 1)
