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


def test_strategy_stops_once_the_standard_error_is_within_tolerance():
    built = SCENARIOS["homogeneous"]()
    strategy = FOVBootstrapStrategy("centroid", 0.08, Standardize(), 200, 0, 3)
    results = evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )
    for item in results:
        quantity = item.result.evidence.quantities["fov_bootstrap_centroid_se"]
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
