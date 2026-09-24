"""Tests for the multivariate outcome metrics and the distance primitives they use."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial import distance

from morphocoverage import distances, validate_input_tables
from morphocoverage.multivariate_outcomes import (
    CentroidError,
    CovarianceError,
    EnergyError,
    FrechetError,
    MMDError,
    QuantileError,
    SlicedWassersteinError,
)

FEATURES = ("a", "b")


def _data(values, cells_per_fov=10):
    """A validated group whose rows are ``values`` (n x 2), ``cells_per_fov`` per FOV."""
    values = np.asarray(values, dtype=float)
    objects = pd.DataFrame(values, columns=list(FEATURES))
    objects["image_file"] = [f"f{i // cells_per_fov}.tif" for i in range(len(values))]
    return validate_input_tables(objects, feature_cols=list(FEATURES))


def _normal(n, mean=(0.0, 0.0), sd=(1.0, 1.0), seed=0):
    return np.random.default_rng(seed).normal(mean, sd, size=(n, 2))


# --- distance primitives --------------------------------------------------------


def test_energy_distance_is_the_unbiased_form_of_scipys_statistic():
    x, y = _normal(80, seed=1), _normal(60, mean=(0.5, 0), seed=2)
    # scipy's 1-D energy distance squared is the V-statistic; rebuild it from ours.
    x1, y1 = x[:, :1], y[:, :1]
    v_within_x = distance.pdist(x1).sum() * 2 / len(x1) ** 2
    v_within_y = distance.pdist(y1).sum() * 2 / len(y1) ** 2
    u = distances.energy_distance(x1, y1)
    u_within_x = distance.pdist(x1).mean()
    u_within_y = distance.pdist(y1).mean()
    v = u + u_within_x + u_within_y - v_within_x - v_within_y
    assert v == pytest.approx(stats.energy_distance(x1[:, 0], y1[:, 0]) ** 2)


def test_unbiased_distances_do_not_shrink_with_sample_size_under_the_null():
    # Equal distributions: the U-statistics average zero at any size, so a
    # larger stop is not rewarded for its size alone.
    for metric in (distances.energy_distance, lambda x, y: distances.mmd_squared(x, y, 1.0)):
        small = [metric(_normal(10, seed=s), _normal(200, seed=1000 + s)) for s in range(200)]
        assert abs(np.mean(small)) < 3 * np.std(small) / np.sqrt(len(small))


def test_mmd_matches_a_brute_force_unbiased_estimate():
    x, y, bandwidth = _normal(40, seed=3), _normal(30, mean=(1, 0), seed=4), 0.7

    def kernel(a, b):
        return np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * bandwidth**2))

    kxx, kyy = kernel(x, x), kernel(y, y)
    n, m = len(x), len(y)
    expected = (
        (kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kernel(x, y).mean()
    )
    assert distances.mmd_squared(x, y, bandwidth) == pytest.approx(expected)


def test_frechet_distance_of_one_feature_is_the_gaussian_formula():
    x, y = _normal(50, seed=5)[:, :1], _normal(70, mean=(2, 0), sd=(3, 1), seed=6)[:, :1]
    expected = np.hypot(x.mean() - y.mean(), x.std(ddof=1) - y.std(ddof=1))
    assert distances.frechet_distance(x, y) == pytest.approx(expected)
    assert distances.frechet_distance(x, x) == pytest.approx(0, abs=1e-7)


def test_sliced_wasserstein_along_one_axis_is_scipys_wasserstein_distance():
    x, y = _normal(50, seed=7), _normal(40, mean=(1, 5), seed=8)
    along_a = distances.sliced_wasserstein(x, y, np.array([[1.0, 0.0]]))
    assert along_a == pytest.approx(stats.wasserstein_distance(x[:, 0], y[:, 0]))
    directions = distances.random_directions(2, 5, seed=0)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1)
    assert np.array_equal(directions, distances.random_directions(2, 5, seed=0))


@pytest.mark.parametrize(
    "call",
    [
        lambda: distances.energy_distance(np.zeros((1, 2)), np.zeros((3, 2))),
        lambda: distances.frechet_distance(np.zeros((3, 2)), np.zeros((3, 1))),
        lambda: distances.mmd_squared(np.zeros((3, 2)), np.zeros((3, 2)), 0),
        lambda: distances.random_directions(2, 0, seed=0),
    ],
)
def test_primitives_refuse_invalid_input(call):
    with pytest.raises(ValueError):
        call()


# --- outcome metrics ------------------------------------------------------------

METRICS = [
    CentroidError(FEATURES, True),
    CovarianceError(FEATURES, True),
    QuantileError(FEATURES, True, (0.1, 0.5, 0.9)),
    FrechetError(FEATURES, True),
    EnergyError(FEATURES, True, None, 0),
    MMDError(FEATURES, True, None, 0, 1.0),
    SlicedWassersteinError(FEATURES, True, 10, 0),
]
IDS = [metric.label for metric in METRICS]


@pytest.mark.parametrize("metric", METRICS, ids=IDS)
def test_a_stop_equal_to_the_reference_has_no_error(metric):
    data = _data(_normal(200))
    value = metric.evaluate(data, data)
    # The U-statistics are not exactly zero on identical samples, but small.
    assert value == pytest.approx(0, abs=0.02)


@pytest.mark.parametrize("metric", METRICS, ids=IDS)
def test_a_shifted_stop_has_a_larger_error_than_a_matching_one(metric):
    reference = _data(_normal(400, seed=1))
    matching = _data(_normal(100, seed=2))
    shifted = _data(_normal(100, mean=(2.0, 0.0), sd=(2.0, 1.0), seed=2))
    assert metric.evaluate(reference, shifted) > metric.evaluate(reference, matching)


@pytest.mark.parametrize("metric", METRICS, ids=IDS)
def test_missing_values_empty_stops_and_zero_spread_are_undefined(metric):
    reference = _data(_normal(50))
    assert metric.evaluate(reference, _data(np.empty((0, 2)))) is None
    with_nan = _normal(50)
    with_nan[3, 1] = np.nan
    assert metric.evaluate(reference, _data(with_nan)) is None
    assert metric.evaluate(_data(with_nan), reference) is None
    constant = np.column_stack([np.ones(50), _normal(50)[:, 1]])
    assert metric.evaluate(_data(constant), reference) is None


def test_standardizing_uses_the_reference_scale():
    # Scaling feature a by 100 changes the raw centroid error but not the
    # standardized one, since both samples are scaled by the reference SD.
    reference, stop = _normal(300, seed=1), _normal(40, mean=(0.3, 0.1), seed=2)
    scaled = np.array([100.0, 1.0])
    raw = CentroidError(FEATURES, False)
    standard = CentroidError(FEATURES, True)
    assert raw.evaluate(_data(reference * scaled), _data(stop * scaled)) != pytest.approx(
        raw.evaluate(_data(reference), _data(stop))
    )
    assert standard.evaluate(_data(reference * scaled), _data(stop * scaled)) == pytest.approx(
        standard.evaluate(_data(reference), _data(stop))
    )


def test_centroid_error_is_the_rms_mean_difference():
    reference, stop = _normal(100, seed=1), _normal(30, seed=2)
    expected = np.sqrt(np.mean((stop.mean(axis=0) - reference.mean(axis=0)) ** 2))
    assert CentroidError(FEATURES, False).evaluate(_data(reference), _data(stop)) == pytest.approx(
        expected
    )


def test_subsampling_is_seeded_and_bounded():
    reference, stop = _data(_normal(500, seed=1)), _data(_normal(300, seed=2))
    first = EnergyError(FEATURES, True, 50, 3)
    assert first.evaluate(reference, stop) == first.evaluate(reference, stop)
    assert first.evaluate(reference, stop) != EnergyError(FEATURES, True, 50, 4).evaluate(
        reference, stop
    )
    assert EnergyError(FEATURES, True, 1000, 3).evaluate(reference, stop) == pytest.approx(
        EnergyError(FEATURES, True, None, 3).evaluate(reference, stop)
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: CentroidError("a", True),
        lambda: CentroidError(("a", "a"), True),
        lambda: CentroidError(FEATURES, 1),
        lambda: QuantileError(FEATURES, True, (0.5, 1.5)),
        lambda: EnergyError(FEATURES, True, 1, 0),
        lambda: MMDError(FEATURES, True, None, 0, 0.0),
        lambda: SlicedWassersteinError(FEATURES, True, 0, 0),
        lambda: SlicedWassersteinError(FEATURES, True, 5, 0.5),
    ],
)
def test_invalid_settings_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()


def test_names_and_parameters_are_provenance():
    metric = MMDError(FEATURES, True, 200, 7, 0.5)
    assert metric.name == "mmd_error:a,b"
    assert json.loads(json.dumps(metric.parameters)) == {
        "features": ["a", "b"],
        "standardize": True,
        "max_observations": 200,
        "seed": 7,
        "bandwidth": 0.5,
    }
    assert metric.config_id != MMDError(FEATURES, True, 200, 7, 1.0).config_id


def test_frechet_of_identical_samples_is_exactly_zero():
    # Regression: sqrtm rounding gave ~1e-7 through the square root.
    x = _normal(200, sd=(5.0, 0.1), seed=9)
    assert distances.frechet_distance(x, x) == 0.0


def test_infinite_bandwidth_is_refused():
    with pytest.raises(ValueError):
        distances.mmd_squared(_normal(5), _normal(5), float("inf"))
    with pytest.raises(ValueError):
        MMDError(FEATURES, True, None, 0, float("inf"))
