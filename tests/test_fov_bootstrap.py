"""Tests for the FOV-bootstrap uncertainty estimator and strategy."""

import json

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Decision,
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    run_strategies,
    validate_input_tables,
)
from morphocoverage.evaluator import objects_in_fovs, restrict_input
from morphocoverage.reservoir import spatial_blocks
from morphocoverage.schema import fov_positions
from morphocoverage.strategies import FOVBootstrapEstimator, FOVBootstrapStrategy
from morphocoverage.strategies.fov_bootstrap import (
    centroid_resamples,
    covariance_resamples,
    eigenspectrum_resamples,
    quantile_resamples,
    subspace_angles_to,
)
from morphocoverage.strategies.scalar_precision import fov_averaging_cv
from morphocoverage.transforms import Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _data(scenario="multimodal", features=FEATURE_COLS, **kwargs):
    built = SCENARIOS[scenario]()
    return validate_input_tables(
        built.objects, fovs=built.fovs, feature_cols=list(features), **kwargs
    )


def _estimate(
    data, summary="centroid", transform=None, n=200, seed=0, min_fovs=2, q=None, n_components=None
):
    estimator = FOVBootstrapEstimator(summary, transform, n, seed, min_fovs, q, n_components)
    return estimator.estimate(data, objective=POPULATION).quantities[estimator.quantity]


def test_resample_helpers_match_brute_force_pooling():
    rng = np.random.default_rng(0)
    counts = np.array([3, 0, 5, 2])
    positions = np.repeat(np.arange(4), counts)
    values = rng.normal(size=(len(positions), 2))
    draws = rng.integers(0, 4, size=(6, 4))
    weights = np.zeros((6, 4))
    np.add.at(weights, (np.arange(6)[:, None], draws), 1.0)
    sums = np.array([values[positions == j].sum(axis=0) for j in range(4)])
    products = np.einsum("ni,nj->nij", values, values).reshape(len(values), -1)
    outer = np.array([products[positions == j].sum(axis=0) for j in range(4)])
    centroids = centroid_resamples(counts.astype(float), sums, weights)
    covariances = covariance_resamples(counts.astype(float), sums, outer, weights)
    quantiles = quantile_resamples(values, positions, weights, (0.2, 0.5))
    for b, fovs in enumerate(draws):
        pooled = np.concatenate([values[positions == j] for j in fovs])
        if len(pooled) == 0:
            assert np.isnan(centroids[b]).all() and np.isnan(quantiles[b]).all()
            continue
        assert np.allclose(centroids[b], pooled.mean(axis=0))
        assert np.allclose(quantiles[b], np.quantile(pooled, (0.2, 0.5), axis=0).ravel())
        if len(pooled) >= 2:
            assert np.allclose(covariances[b], np.cov(pooled, rowvar=False).ravel())


def test_centroid_se_matches_the_ratio_estimator_with_equal_fovs():
    # With equal observations per FOV the pooled mean is the mean of FOV
    # means, and the bootstrap variance is the analytic FOV-unit variance
    # times (k - 1) / k, the scalar FOV-unit JCB estimate.
    rng = np.random.default_rng(1)
    k, per_fov = 15, 8
    fov_means = rng.normal(10, 2, size=k)
    objects = pd.DataFrame(
        {
            "image_file": np.repeat([f"f{j}.tif" for j in range(k)], per_fov),
            "area": np.repeat(fov_means, per_fov) + rng.normal(0, 1, k * per_fov),
        }
    )
    data = validate_input_tables(objects, feature_cols=["area"])
    positions = fov_positions(data, data.fovs)
    values = data.feature_values("area")
    analytic = fov_averaging_cv(
        np.bincount(positions, weights=values, minlength=k), np.bincount(positions, minlength=k)
    )
    expected = analytic["cv_mean"] * analytic["mean"] * np.sqrt((k - 1) / k)
    assert _estimate(data, n=20000).value == pytest.approx(expected, rel=0.03)


def test_more_cells_from_the_same_fovs_do_not_reduce_uncertainty():
    # Duplicating every observation doubles the object count but adds no FOVs;
    # FOV-bootstrap uncertainty of the centroid is exactly unchanged.
    built = SCENARIOS["multimodal"]()
    doubled = pd.concat([built.objects, built.objects], ignore_index=True)
    single = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    twice = validate_input_tables(doubled, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    assert _estimate(twice).value == pytest.approx(_estimate(single).value, rel=1e-12)


def test_uncertainty_falls_roughly_as_one_over_root_fovs():
    data = _data("homogeneous")
    ordered = data.fovs.sort_values(ORDER_COL)
    few = ordered.iloc[:5]
    prefix = restrict_input(data, objects_in_fovs(data, few), few)
    ratio = _estimate(prefix, n=2000).value / _estimate(data, n=2000).value
    assert ratio == pytest.approx(np.sqrt(len(ordered) / 5), rel=0.35)


@pytest.mark.parametrize(
    "summary, q", [("centroid", None), ("covariance", None), ("quantiles", (0.1, 0.9))]
)
def test_every_summary_is_seeded_and_positive(summary, q):
    data = _data()
    first = _estimate(data, summary, Standardize(), seed=3, q=q)
    assert first.availability.value == "available" and first.value > 0
    assert _estimate(data, summary, Standardize(), seed=3, q=q) == first
    assert _estimate(data, summary, Standardize(), seed=4, q=q).value != first.value


def test_insufficient_and_not_applicable_cases():
    data = _data()
    first = data.fovs.iloc[:1]
    one_fov = restrict_input(data, objects_in_fovs(data, first), first)
    assert _estimate(one_fov).availability.value == "insufficient_data"
    assert _estimate(data, min_fovs=10**6).availability.value == "insufficient_data"

    built = SCENARIOS["multimodal"]()
    constant = validate_input_tables(
        built.objects.assign(feature_b=1.0), fovs=built.fovs, feature_cols=list(FEATURE_COLS)
    )
    unfit = _estimate(constant, transform=Standardize())
    assert unfit.availability.value == "insufficient_data" and "transform" in unfit.reason

    missing = built.objects.copy()
    missing.loc[0, "feature_a"] = np.nan
    nan_data = validate_input_tables(missing, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    assert _estimate(nan_data).availability.value == "not_applicable"


def test_zero_observation_fovs_are_resampled_as_acquired_units():
    data = _data("zero_object_fovs")
    estimator = FOVBootstrapEstimator("centroid", None, 500, 0, 2)
    evidence = estimator.estimate(data, objective=POPULATION)
    assert evidence.diagnostics["fovs_with_observations"] < len(data.fovs)
    assert evidence.quantities[estimator.quantity].value > 0


@pytest.mark.parametrize(
    "aggregate, name",
    [("max", "fov_bootstrap_centroid_max_se"), ("rms", "fov_bootstrap_centroid_se")],
)
def test_strategy_stops_once_the_standard_error_is_within_tolerance(aggregate, name):
    built = SCENARIOS["homogeneous"]()
    strategy = FOVBootstrapStrategy("centroid", 0.08, Standardize(), 200, 0, 3, aggregate=aggregate)
    results = evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )
    for item in results:
        quantity = item.result.evidence.quantities[name]
        if quantity.availability.value != "available":
            assert item.result.decision is Decision.CONTINUE
        else:
            expected = Decision.STOP if quantity.value <= 0.08 else Decision.CONTINUE
            assert item.result.decision is expected
    assert any(item.result.decision is Decision.STOP for item in results)


@pytest.mark.parametrize(
    "build",
    [
        lambda: FOVBootstrapEstimator("median", None, 10, 0, 2),
        lambda: FOVBootstrapEstimator("centroid", None, 1, 0, 2),
        lambda: FOVBootstrapEstimator("centroid", None, 10, 0, 1),
        lambda: FOVBootstrapEstimator("centroid", None, 10, 0.5, 2),
        lambda: FOVBootstrapEstimator("quantiles", None, 10, 0, 2),
        lambda: FOVBootstrapEstimator("centroid", None, 10, 0, 2, (0.5,)),
        lambda: FOVBootstrapEstimator("quantiles", None, 10, 0, 2, (1.5,)),
        lambda: FOVBootstrapEstimator("centroid", "standardize", 10, 0, 2),
        lambda: FOVBootstrapStrategy("centroid", 0.0, None, 10, 0, 2),
    ],
)
def test_invalid_settings_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()


def test_parameters_are_serializable_provenance():
    strategy = FOVBootstrapStrategy("quantiles", 0.1, Standardize(), 100, 7, 4, [0.25, 0.75])
    record = json.loads(json.dumps(strategy.parameters))
    assert record["quantiles"] == [0.25, 0.75] and record["random_state"] == 7
    assert record["transform"]["name"] == "standardize"
    other = FOVBootstrapStrategy("quantiles", 0.1, Standardize(), 100, 8, 4, [0.25, 0.75])
    assert strategy.config_id != other.config_id


# --- PCA summaries --------------------------------------------------------------


def _rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def test_eigenspectrum_fractions_are_sorted_and_match_numpy():
    rng = np.random.default_rng(0)
    covariances = [np.cov(rng.normal(size=(50, 3)) * [3, 2, 1], rowvar=False) for _ in range(4)]
    fractions = eigenspectrum_resamples(np.array([c.ravel() for c in covariances]), 3)
    for row, cov in zip(fractions, covariances, strict=True):
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        assert np.allclose(row, eigenvalues / eigenvalues.sum())
        assert np.all(np.diff(row) <= 0)


def test_subspace_angle_recovers_a_known_rotation():
    reference = np.diag([4.0, 1.0])
    rotated = [_rotation(t) @ reference @ _rotation(t).T for t in (0.0, 0.2, 0.5)]
    angles = subspace_angles_to(np.array([c.ravel() for c in rotated]), reference, 1)
    assert np.allclose(angles, [0.0, 0.2, 0.5], atol=1e-8)


def _fov_table(values, per_fov=20):
    objects = pd.DataFrame(values, columns=["a", "b"])
    objects["image_file"] = [f"f{i // per_fov}.tif" for i in range(len(values))]
    return validate_input_tables(objects, feature_cols=["a", "b"])


def test_near_equal_eigenvalues_leave_the_subspace_undetermined():
    # With equal variance on both axes the leading direction is arbitrary, so
    # resampled subspaces swing widely; a clear dominant axis is stable.
    rng = np.random.default_rng(2)
    isotropic = _fov_table(rng.normal(size=(400, 2)))
    elongated = _fov_table(rng.normal(size=(400, 2)) * [5.0, 1.0])
    unstable = _estimate(isotropic, "subspace", None, n=200, q=None, n_components=1)
    stable = _estimate(elongated, "subspace", None, n=200, q=None, n_components=1)
    assert unstable.units == "radians"
    assert unstable.value > 0.2
    assert stable.value < 0.1


def test_pca_summaries_need_several_features_and_fewer_components():
    data = _data(features=("feature_a",))
    assert _estimate(data, "eigenspectrum").availability.value == "not_applicable"
    two = _data()
    too_many = _estimate(two, "subspace", n_components=2)
    assert too_many.availability.value == "not_applicable" and "n_components" in too_many.reason
    assert _estimate(two, "eigenspectrum").units == "fraction of total variance"


@pytest.mark.parametrize(
    "build",
    [
        lambda: FOVBootstrapEstimator("subspace", None, 10, 0, 2),
        lambda: FOVBootstrapEstimator("centroid", None, 10, 0, 2, None, 1),
        lambda: FOVBootstrapEstimator("subspace", None, 10, 0, 2, None, 0),
    ],
)
def test_n_components_belongs_to_the_subspace_summary(build):
    with pytest.raises(ValueError):
        build()


# --- spatial block bootstrap ----------------------------------------------------


def _with_blocks(scenario, blocks):
    built = SCENARIOS[scenario]()
    fovs = built.fovs.assign(block=blocks(built.fovs))
    return validate_input_tables(built.objects, fovs=fovs, feature_cols=["feature_a"])


def _block_se(data, block_col="block", min_fovs=2, n=2000):
    estimator = FOVBootstrapEstimator("centroid", None, n, 0, min_fovs, None, None, block_col)
    return estimator.estimate(data, objective=POPULATION).quantities[estimator.quantity]


def test_single_fov_blocks_reproduce_the_fov_bootstrap_exactly():
    data = _with_blocks("neighbour_correlation", lambda fovs: np.arange(len(fovs)))
    assert _block_se(data).value == _block_se(data, block_col=None).value


def test_blocks_widen_uncertainty_when_neighbouring_fovs_are_correlated():
    # The phenotype drifts smoothly with stage position: FOVs within a 300 um
    # block are alike, so resampling single FOVs understates the uncertainty.
    data = _with_blocks(
        "neighbour_correlation", lambda fovs: spatial_blocks(fovs, "stage_x", "stage_y", 300.0)
    )
    assert _block_se(data).value > 1.5 * _block_se(data, block_col=None).value


def test_too_few_blocks_are_insufficient_data():
    data = _with_blocks(
        "neighbour_correlation", lambda fovs: spatial_blocks(fovs, "stage_x", "stage_y", 600.0)
    )
    quantity = _block_se(data, min_fovs=3)
    assert quantity.availability.value == "insufficient_data"
    assert "blocks" in quantity.reason


def test_block_column_must_exist_and_be_complete():
    data = _data("neighbour_correlation", features=("feature_a",))
    with pytest.raises(ValueError, match="not a column"):
        _block_se(data)
    gaps = _with_blocks("neighbour_correlation", lambda fovs: [np.nan] + [1.0] * (len(fovs) - 1))
    with pytest.raises(ValueError, match="missing values"):
        _block_se(gaps)
    strategy = FOVBootstrapStrategy("centroid", 0.1, None, 10, 0, 2, None, None, "block")
    assert strategy.parameters["block_col"] == "block"


@pytest.mark.parametrize(
    "summary, r", [("covariance", None), ("eigenspectrum", None), ("subspace", 1)]
)
def test_covariance_summaries_do_not_depend_on_a_location_shift(summary, r):
    # Regression: a one-pass covariance on raw values lost precision when the
    # mean was large relative to the spread (offset 1e8 gave a 60x SE).
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1000, 2)) * [2.0, 1.0]
    base = _estimate(_fov_table(values, per_fov=50), summary, n_components=r)
    shifted = _estimate(_fov_table(values + 1e8, per_fov=50), summary, n_components=r)
    assert shifted.value == pytest.approx(base.value, rel=1e-6)


@pytest.mark.parametrize("summary, r", [("eigenspectrum", None), ("subspace", 1)])
def test_resamples_without_spread_make_pca_summaries_undefined(summary, r):
    # Regression (external audit): two one-cell FOVs. About half the resamples
    # repeat one FOV and have no spread. Leaving them out reported an
    # eigenspectrum SE of exactly 0, and the subspace took an arbitrary basis.
    objects = pd.DataFrame({"image_file": ["f0", "f1"], "a": [0.0, 1.0], "b": [0.0, 0.0]})
    data = validate_input_tables(objects, feature_cols=["a", "b"])
    estimator = FOVBootstrapEstimator(summary, None, 500, 0, 2, None, r)
    evidence = estimator.estimate(data, objective=POPULATION)
    quantity = evidence.quantities[estimator.quantity]
    assert all(q == quantity for q in evidence.quantities.values())
    assert quantity.availability.value == "insufficient_data"
    assert "too little spread" in quantity.reason
    assert 150 < evidence.diagnostics["degenerate_resamples"] < 350


def test_a_subspace_needs_spread_in_every_leading_direction():
    # Cells spread along one axis only: a 2-dimensional leading subspace has a
    # zero eigenvalue, so its second direction would be arbitrary.
    rng = np.random.default_rng(3)
    rows = [(f"f{j}", x, 0.0, 0.0) for j in range(6) for x in rng.normal(size=10)]
    objects = pd.DataFrame(rows, columns=["image_file", "a", "b", "c"])
    data = validate_input_tables(objects, feature_cols=["a", "b", "c"])
    one = FOVBootstrapEstimator("subspace", None, 50, 0, 2, None, 1).estimate(
        data, objective=POPULATION
    )
    two = FOVBootstrapEstimator("subspace", None, 50, 0, 2, None, 2).estimate(
        data, objective=POPULATION
    )
    assert one.quantities["fov_bootstrap_subspace_angle"].availability.value == "available"
    assert two.quantities["fov_bootstrap_subspace_angle"].availability.value == (
        "insufficient_data"
    )


def test_unequal_blocks_resample_as_clusters_and_report_their_sizes():
    # Documented behaviour (external audit): blocks are drawn with equal
    # probability, as clusters, so unequal occupancy enters the SE. 9 FOVs at 0
    # in one block and 1 FOV at 10 in another: resamples AA, AB, BA, BB give
    # means 0, 1, 1, 10, an SE of sqrt(16.5) ~ 4.06, against about 0.95 for FOVs.
    fovs = pd.DataFrame({"image_file": [f"f{i}" for i in range(10)], "block": ["A"] * 9 + ["B"]})
    objects = pd.DataFrame({"image_file": fovs["image_file"], "x": [0.0] * 9 + [10.0]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    blocks = FOVBootstrapEstimator("centroid", None, 20000, 0, 2, None, None, "block")
    evidence = blocks.estimate(data, objective=POPULATION)
    assert evidence.value("fov_bootstrap_centroid_se") == pytest.approx(np.sqrt(16.5), rel=0.03)
    assert evidence.diagnostics["block_sizes"] == {"min": 1, "max": 9}
    fovs_only = FOVBootstrapEstimator("centroid", None, 20000, 0, 2)
    assert fovs_only.estimate(data, objective=POPULATION).value(
        "fov_bootstrap_centroid_se"
    ) == pytest.approx(0.95, abs=0.03)


@pytest.mark.parametrize("n_stable", [1, 10, 100])
def test_stable_features_do_not_make_the_strategy_stop_earlier(n_stable):
    # External re-audit: the RMS over components falls as perfectly stable
    # features are added, although one feature keeps the same uncertainty.
    # The largest component SE, the default stopping quantity, does not.
    objects = pd.DataFrame(
        {"image_file": [f"f{i}" for i in range(8)], "x": [-4, -3, -2, -1, 1, 2, 3, 4]}
    )
    one = validate_input_tables(objects, feature_cols=["x"])
    stable = [f"stable{j}" for j in range(n_stable)]
    many = validate_input_tables(
        pd.concat([objects, pd.DataFrame(0.0, index=objects.index, columns=stable)], axis=1),
        feature_cols=["x", *stable],
    )
    estimator = FOVBootstrapEstimator("centroid", None, 2000, 0, 2)
    alone = estimator.estimate(one, objective=POPULATION)
    diluted = estimator.estimate(many, objective=POPULATION)
    se = alone.value("fov_bootstrap_centroid_se")
    assert diluted.value("fov_bootstrap_centroid_max_se") == pytest.approx(se)
    assert diluted.value("fov_bootstrap_centroid_se") == pytest.approx(se / np.sqrt(n_stable + 1))
    assert diluted.diagnostics["max_component"] == 0
    tolerance = se * 0.9  # the unstable feature is not yet within tolerance
    for aggregate, stops_with_stable in (("max", False), ("rms", n_stable >= 1)):
        strategy = FOVBootstrapStrategy(
            "centroid", tolerance, None, 2000, 0, 2, aggregate=aggregate
        )
        decisions = [
            run_strategies(
                data.objects,
                strategies=[strategy],
                feature_cols=list(data.feature_cols),
                objective=POPULATION,
            )[0].decision
            for data in (one, many)
        ]
        assert decisions[0] is Decision.CONTINUE
        assert (decisions[1] is Decision.STOP) is stops_with_stable


def test_aggregate_must_be_max_or_rms():
    with pytest.raises(ValueError, match="aggregate"):
        FOVBootstrapStrategy("centroid", 0.1, None, 10, 0, 2, aggregate="mean")
