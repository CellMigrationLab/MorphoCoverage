"""Tests for the FOV-bootstrap uncertainty estimator and strategy."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from _helpers_strategies import _acquisition, _well, _words, _x_run
from _helpers_tables import _validated_mixed
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
from morphocoverage.evidence import Availability
from morphocoverage.reservoir import spatial_blocks
from morphocoverage.schema import fov_positions
from morphocoverage.statistics import small_sample_widening
from morphocoverage.strategies import (
    DescriptorSpreadStrategy,
    FOVBootstrapEstimator,
    FOVBootstrapStrategy,
    fov_bootstrap,
)
from morphocoverage.strategies._fov_bootstrap_checks import _check
from morphocoverage.strategies.fov_bootstrap import (
    centroid_resamples,
    covariance_resamples,
    eigenspectrum_resamples,
    quantile_resamples,
    subspace_angles_to,
)
from morphocoverage.strategies.scalar_precision import fov_averaging_cv
from morphocoverage.transforms import PCA, Standardize

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
    # The reported SE is the raw one widened for k FOVs (audit 4, P4-4).
    expected *= small_sample_widening(k)
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


@pytest.mark.parametrize("observed_blocks, available", [(2, False), (3, True)])
def test_min_fovs_counts_blocks_with_observations(observed_blocks, available):
    # Regression (#315): five visible blocks, but observations in only two of
    # them, were enough for min_fovs=3, so two blocks were resampled.
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"fov": [f"f{i}" for i in range(10)], "block": np.repeat(range(5), 2)})
    seen = [f"f{i}" for i in range(2 * observed_blocks)]
    objects = pd.DataFrame({"fov": np.repeat(seen, 5), "feature_a": rng.normal(size=5 * len(seen))})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["feature_a"], fov_col="fov")
    quantity = _block_se(data, min_fovs=3, n=200)
    assert (quantity.availability.value == "available") is available
    if not available:
        assert quantity.reason == "fewer than 3 spatial blocks with observations"


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
    objects = pd.DataFrame(rows, columns=["image_file", "a", "b", "d"])
    data = validate_input_tables(objects, feature_cols=["a", "b", "d"])
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
    # means 0, 1, 1, 10, an SE of sqrt(16.5) ~ 4.06, against about 0.95 for FOVs (raw
    # SEs; the reported ones are widened for 2 blocks and 10 FOVs, audit 4, P4-4).
    fovs = pd.DataFrame({"image_file": [f"f{i}" for i in range(10)], "block": ["A"] * 9 + ["B"]})
    objects = pd.DataFrame({"image_file": fovs["image_file"], "x": [0.0] * 9 + [10.0]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    blocks = FOVBootstrapEstimator("centroid", None, 20000, 0, 2, None, None, "block")
    evidence = blocks.estimate(data, objective=POPULATION)
    assert evidence.value("fov_bootstrap_centroid_se") == pytest.approx(
        np.sqrt(16.5) * small_sample_widening(2), rel=0.03
    )
    assert evidence.diagnostics["block_sizes"] == {"min": 1, "max": 9}
    assert evidence.diagnostics["resampling_units"] == 2
    fovs_only = FOVBootstrapEstimator("centroid", None, 20000, 0, 2)
    assert fovs_only.estimate(data, objective=POPULATION).value(
        "fov_bootstrap_centroid_se"
    ) == pytest.approx(0.95 * small_sample_widening(10), abs=0.04)


@pytest.mark.parametrize("n_stable", [1, 10, 100])
def test_stable_features_do_not_make_the_strategy_stop_earlier(n_stable):
    # External re-audit: the RMS over components falls as nearly stable
    # features are added, although one feature keeps the same uncertainty.
    # The largest component SE, the default stopping quantity, does not.
    # (Nearly, not exactly: a feature without spread so far is not yet enough data for
    # the bootstrap, audit 4, P4-1.)
    objects = pd.DataFrame(
        {"image_file": [f"f{i}" for i in range(8)], "x": [-4, -3, -2, -1, 1, 2, 3, 4]}
    )
    one = validate_input_tables(objects, feature_cols=["x"])
    stable = [f"stable{j}" for j in range(n_stable)]
    nearly = np.outer(objects["x"].to_numpy() * 1e-7, np.ones(n_stable))
    many = validate_input_tables(
        pd.concat([objects, pd.DataFrame(nearly, index=objects.index, columns=stable)], axis=1),
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


def test_pca_availability_does_not_depend_on_the_number_of_resamples():
    # External re-audit: with "any degenerate resample" as the rule, more
    # resamples eventually draw one and flip the answer to insufficient_data.
    # Five one-cell FOVs spanning two dimensions repeat one FOV in 5 * 0.2**5
    # = 0.16% of resamples, below the tolerance, at every resample count.
    objects = pd.DataFrame(
        {
            "image_file": [f"f{i}" for i in range(5)],
            "a": [0.0, 1.0, 0.0, 1.0, 0.5],
            "b": [0.0, 0.0, 1.0, 1.0, 0.3],
        }
    )
    data = validate_input_tables(objects, feature_cols=["a", "b"])
    for n in (20, 200, 2000, 20000):
        evidence = FOVBootstrapEstimator("eigenspectrum", None, n, 0, 2).estimate(
            data, objective=POPULATION
        )
        assert evidence.quantities["fov_bootstrap_eigenspectrum_se"].availability.value == (
            "available"
        )
    assert evidence.diagnostics["degenerate_fraction"] == pytest.approx(0.0016, abs=0.001)


def test_the_bootstrap_draws_the_same_blocks_whatever_the_row_order():
    def weights(order: list[int]) -> np.ndarray:
        data = _validated_mixed(order)
        estimator = FOVBootstrapEstimator(
            summary="centroid",
            transform=None,
            n_resamples=20,
            random_state=3,
            min_fovs=2,
            block_col="block",
        )
        return estimator._weights(data)

    forward, backward = weights([0, 1, 2, 3]), weights([3, 2, 1, 0])
    assert np.array_equal(forward, backward[:, ::-1])


def test_c6_a_transform_that_can_never_be_fitted_is_not_applicable_from_the_first_fov():
    rng = np.random.default_rng(1)
    values = [list(rng.normal(10, 2, 5)) for _ in range(6)]
    objects, fovs = _acquisition(values)
    objects["other"] = rng.normal(0, 1, len(objects))

    def replay(strategy):
        return evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[strategy],
            feature_cols=["area", "other"],
            acquisition_order_col="acq",
            objective=POPULATION,
            fov_col="position_id",
        )

    never = replay(FOVBootstrapStrategy("centroid", 0.05, PCA(3), 50, 0, 2))
    for step in never:
        quantity = step.result.evidence.quantities["fov_bootstrap_centroid_max_se"]
        assert quantity.availability == "not_applicable"
        assert "can never be fitted on 2 features" in quantity.reason
        assert step.result.decision is None
    # A transform that can be fitted once more FOVs arrive is not yet enough data.
    later = replay(FOVBootstrapStrategy("centroid", 5.0, PCA(2), 50, 0, 2))
    first = later[0].result.evidence.quantities["fov_bootstrap_centroid_max_se"]
    assert first.availability == "insufficient_data"
    assert later[-1].result.evidence.quantities["fov_bootstrap_centroid_max_se"].value > 0


def _bootstrap_to_analytic(values_per_fov):
    objects, fovs = _acquisition(values_per_fov)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"], fov_col="position_id")
    boot = FOVBootstrapEstimator("centroid", None, 40_000, 0, 2).estimate(
        data, objective=POPULATION
    )
    sums = np.array([sum(v) for v in values_per_fov])
    counts = np.array([len(v) for v in values_per_fov], dtype=float)
    analytic = fov_averaging_cv(sums, counts)
    # The raw bootstrap SE: the reported one is widened for few FOVs (audit 4, P4-4).
    raw = boot.value("fov_bootstrap_centroid_se") / boot.diagnostics["small_sample_widening"]
    return raw / (analytic["cv_mean"] * analytic["mean"])


def test_c8_the_bootstrap_to_ratio_se_factor_holds_only_when_every_fov_holds_objects():
    rng = np.random.default_rng(5)
    # Equal object counts: the ratio estimator is then linear in the FOV means.
    filled = [list(rng.normal(10 + rng.normal(), 2, 10)) for _ in range(8)]
    assert _bootstrap_to_analytic(filled) == pytest.approx(math.sqrt(7 / 8), rel=0.02)
    half_empty = [values if j % 2 == 0 else [] for j, values in enumerate(filled)]
    # With 4 of 8 FOVs empty the ratio is not sqrt((k' - 1) / k') = 0.87: it is larger.
    assert _bootstrap_to_analytic(half_empty) > math.sqrt(3 / 4) * 1.05
    doc = fov_bootstrap.__doc__
    assert "When every visible FOV holds observations" in _words(doc)
    assert "With empty FOVs it does not" in _words(doc)


def test_c15_degenerate_pca_resamples_are_not_counted_again_as_undefined():
    # Two one-object FOVs: no resample is empty, and half repeat one FOV (no spread).
    objects = pd.DataFrame({"position_id": ["p0", "p1"], "a": [1.0, 2.0], "b": [0.5, 0.1]})
    fovs = pd.DataFrame({"position_id": ["p0", "p1"]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["a", "b"], fov_col="position_id")
    estimator = FOVBootstrapEstimator("eigenspectrum", None, 2000, 0, 2)
    diagnostics = estimator.estimate(data, objective=DISTRIBUTION).diagnostics
    assert diagnostics["degenerate_resamples"] > 800
    assert diagnostics["undefined_resamples"] == 0


def test_p4_11_the_fov_table_order_does_not_change_the_resampled_rules():
    objects, fovs = _well(5)
    rules = [
        DescriptorSpreadStrategy("x", 1.0, "fov", 100, 0),
        FOVBootstrapStrategy("centroid", 0.3, None, 200, 0, 3, features=("x",)),
        FOVBootstrapStrategy("quantiles", 0.3, None, 50, 0, 3, (0.25, 0.75), features=("x",)),
    ]
    base = _x_run(objects, fovs, rules)
    for seed in range(4):
        shuffled_fovs = fovs.sample(frac=1.0, random_state=seed)
        shuffled_objects = objects.sample(frac=1.0, random_state=seed)
        again = _x_run(shuffled_objects, shuffled_fovs, rules)
        for a, b in zip(base, again, strict=True):
            # The same draws; only floating-point summation order may differ.
            for name, quantity in a.evidence.quantities.items():
                assert b.evidence.quantities[name].value == pytest.approx(
                    quantity.value, rel=1e-12
                ), (a.strategy, name)


DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)


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


def test_fov_bootstrap_refuses_a_count_that_is_not_an_integer():
    with pytest.raises(TypeError) as error:
        _check("centroid", 2.5, 0, 2, None, None, None)
    assert str(error.value) == "n_resamples must be an integer"
