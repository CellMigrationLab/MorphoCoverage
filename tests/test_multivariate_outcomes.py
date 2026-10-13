"""Tests for the multivariate outcome metrics and the distance primitives they use."""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import _wide
from _helpers_strategies import _estimate_input, _fovs, _validated_values
from scipy import linalg, stats
from scipy.spatial import distance

from morphocoverage import (
    distances,
    multivariate_outcomes,
    outcomes,
    transforms,
    validate_input_tables,
)
from morphocoverage.distances import frechet_distance
from morphocoverage.embedding import robust_components
from morphocoverage.multivariate_outcomes import (
    CentroidError,
    ComponentsError,
    CovarianceError,
    EnergyError,
    FrechetError,
    MMDError,
    QuantileError,
    SlicedWassersteinError,
)
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.workflows import Setup, optimise
from morphocoverage.workflows._fingerprint_space import fingerprint_space

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
IDS = [metric.family for metric in METRICS]


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


@pytest.mark.parametrize("directions", [[[2.0, 0.0]], [[0.0, 0.0]], np.empty((0, 2))])
def test_sliced_wasserstein_requires_unit_directions(directions):
    # Regression (external audit): a doubled direction doubled the distance
    # and a zero direction made separated samples look identical.
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        distances.sliced_wasserstein(x, x + 10, np.asarray(directions))


def test_outcomes_weight_fovs_by_their_observations():
    # Accepted, documented behaviour: a stop missing a sparse FOV with a
    # different phenotype barely registers, because FOVs count by their cells.
    # 1,000 cells at 0 in one FOV and one cell at 10 in another.
    values = np.zeros((1001, 2))
    values[-1] = 10.0
    objects = pd.DataFrame(values, columns=list(FEATURES))
    objects["image_file"] = ["dense.tif"] * 1000 + ["sparse.tif"]
    full = validate_input_tables(objects, feature_cols=list(FEATURES))
    stop = validate_input_tables(objects.iloc[:1000], feature_cols=list(FEATURES))
    assert CentroidError(FEATURES, False).evaluate(full, stop) == pytest.approx(10 / 1001)
    # Equal FOV weights would put the reference centroid at 5, not near 0.
    fov_means = objects.groupby("image_file")[list(FEATURES)].mean()
    assert fov_means.mean().to_numpy() == pytest.approx([5.0, 5.0])


@pytest.mark.parametrize(
    "metric", [EnergyError(FEATURES, False, None, 0), MMDError(FEATURES, False, None, 0, 1.0)]
)
def test_u_statistics_are_biased_down_when_the_reference_contains_the_stop(metric):
    # The default reference, the full acquisition, contains the stop: identical
    # observations in the cross term pull the statistic below zero on average,
    # whereas against disjoint samples of the same distribution it averages zero.
    nested, disjoint = [], []
    for seed in range(40):
        values = _normal(300, seed=seed)
        full, stop = _data(values), _data(values[:60])
        nested.append(metric.evaluate(full, stop))
        disjoint.append(metric.evaluate(_data(values[60:]), stop))
    assert np.mean(nested) < 0
    assert np.mean(nested) < np.mean(disjoint)


def _clustered(rng, n_fovs, per_fov, between):
    """Cells of ``n_fovs`` FOVs with a share ``between`` of the variance between FOVs."""
    offsets = rng.normal(0, np.sqrt(between), size=(n_fovs, 1, 2))
    cells = offsets + rng.normal(0, np.sqrt(1 - between), size=(n_fovs, per_fov, 2))
    return _data(cells.reshape(-1, 2), cells_per_fov=per_fov)


@pytest.mark.parametrize(
    "metric", [EnergyError(FEATURES, False, None, 0), MMDError(FEATURES, False, None, 0, 1.0)]
)
def test_u_statistics_against_an_independent_reservoir_are_zero_centred_only_for_iid_cells(
    metric,
):
    # External audit: the null expectation against a disjoint reservoir of the
    # same population. With independent cells it is about zero; when cells of a
    # FOV share an offset it is positive and shrinks as FOVs are added, so runs
    # must be compared at matched FOV counts, not against zero.
    rng = np.random.default_rng(21)

    def null_mean(between, n_fovs, runs=150):
        return np.mean(
            [
                metric.evaluate(
                    _clustered(rng, 20, 10, between), _clustered(rng, n_fovs, 10, between)
                )
                for _ in range(runs)
            ]
        )

    iid, few, many = null_mean(0.0, 3), null_mean(0.3, 3), null_mean(0.3, 10)
    assert abs(iid) < 0.01
    assert few > many > 0.005
    assert few > 5 * abs(iid)


def test_energy_error_can_be_negative():
    # A stop at 0 and 1 against a reference at 0.4 and 0.6: -0.2, sampling noise
    # around a small discrepancy rather than an error below zero.
    stop = _data([[0.0, 0.0], [1.0, 0.0]], cells_per_fov=1)
    reference = _data([[0.4, 0.0], [0.6, 0.0]], cells_per_fov=1)
    value = EnergyError(FEATURES, False, None, 0).evaluate(reference, stop)
    assert value == pytest.approx(-0.2)


def test_d1_frechet_distance_scales_with_the_units():
    rng = np.random.default_rng(1)
    x, y = rng.normal(0.0, 1.0, (200, 2)), rng.normal(0.5, 2.0, (200, 2))
    reference = frechet_distance(x, y)
    for unit in (1e-3, 1e-7, 1e-12):
        assert frechet_distance(x * unit, y * unit) == pytest.approx(reference * unit, rel=1e-6)
    assert frechet_distance(x, x) == 0.0
    assert frechet_distance(x * 1e-9, x * 1e-9) == 0.0
    fields = [rng.normal(400e-12, 50e-12, (30, 1)) for _ in range(5)] + [
        rng.normal(700e-12, 50e-12, (30, 1)) for _ in range(5)
    ]
    full, stop = (_estimate_input(f, ("area_m2",)) for f in (fields, fields[:5]))
    assert FrechetError(("area_m2",), False).evaluate(full, stop) > 1e-10


def test_d_4_frechet_distance_does_not_depend_on_the_origin():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 3)), rng.normal(0.3, 1.2, size=(30, 3))
    expected = distances.frechet_distance(x, y)
    for offset in (1e3, 1e6, 1e7):
        assert distances.frechet_distance(x + offset, y + offset) == pytest.approx(
            expected, rel=1e-6
        )
    assert distances.frechet_distance(x, x) == 0.0
    assert distances.frechet_distance(x + 1e6, x + 1e6) == 0.0


def test_d_5_covariance_error_of_a_constant_reference_with_rounding_noise_is_undefined():
    constant = np.column_stack([np.full(5, 0.1), np.full(5, 0.7)])
    data = _validated_values([constant, constant], features=("f0", "f1"))
    stop = _fovs(data, [0])
    assert np.linalg.norm(np.cov(data.feature_matrix(), rowvar=False)) > 0  # rounding noise
    assert CovarianceError(("f0", "f1"), False).evaluate(data, stop) is None
    assert CovarianceError(("f0", "f1"), True).evaluate(data, stop) is None


def test_d_10_the_energy_null_figures_state_the_number_of_features():
    doc = multivariate_outcomes.__doc__
    assert "of one feature" in doc and "two" in doc and "independent features" in doc


def test_d_11_frechet_on_singular_covariances_is_accurate_and_silent():
    rng = np.random.default_rng(5)
    for _ in range(50):
        d = int(rng.integers(2, 12))
        x, y = rng.normal(size=(int(rng.integers(2, d + 2)), d)), rng.normal(size=(30, d))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = distances.frechet_distance(x, y)
        cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        values, vectors = np.linalg.eigh(cov_x)
        root = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T
        product = root @ cov_y @ root
        eigenvalues = np.linalg.eigvalsh((product + product.T) / 2)
        cross = np.sqrt(np.clip(eigenvalues, 0, None)).sum()
        squared = np.sum((x.mean(0) - y.mean(0)) ** 2) + np.trace(cov_x + cov_y) - 2 * cross
        assert got == pytest.approx(math.sqrt(squared), rel=1e-9)
    # A non-singular case agrees with scipy's general matrix square root.
    x, y = rng.normal(size=(50, 3)), rng.normal(1, 2, size=(60, 3))
    cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    root = linalg.sqrtm(cov_x)
    trace = np.trace(cov_x + cov_y - 2 * np.real(linalg.sqrtm(root @ cov_y @ root)))
    expected = math.sqrt(np.sum((x.mean(0) - y.mean(0)) ** 2) + trace)
    assert distances.frechet_distance(x, y) == pytest.approx(expected, rel=1e-9)


def test_j_16_modules_list_their_public_names():
    for module in (distances, transforms, outcomes, multivariate_outcomes):
        names = module.__all__
        assert names and all(not name.startswith("_") for name in names)
        assert all(hasattr(module, name) for name in names)


def test_d1_standardised_outcomes_are_undefined_with_a_constant_reference_feature():
    from morphocoverage.multivariate_outcomes import CentroidError
    from morphocoverage.schema import validate_input_tables

    objects = pd.DataFrame(
        {"image_file": ["f1", "f1", "f2", "f2"], "a": [1.0, 2.0, 3.0, 4.0], "b": [0.1] * 4}
    )
    data = validate_input_tables(
        objects, fovs=objects[["image_file"]].drop_duplicates(), feature_cols=["a", "b"]
    )
    assert CentroidError(("a", "b"), True).evaluate(data, data) is None


def test_no_held_back_field_enters_the_fit_of_the_fingerprints_components(monkeypatch):
    merged, fovs, names = _wide(fields=8, cells=5)
    criterion = OutcomeCriterion(CovarianceError(names, True), 0.5, "lower_is_better", "DINO")
    setup = Setup(
        readout="area",
        features=("area", "shape"),
        fingerprint=(criterion,),
        feature_tables={"DINO": names},
        observation_unit="cell",
    )
    pooled = fingerprint_space(merged, fovs, setup).setup.fingerprint[0].metric
    assert isinstance(pooled, ComponentsError)
    calls = []
    evaluate = ComponentsError.evaluate

    def spy(self, full, stopped):
        calls.append((self, full.objects.copy()))
        return evaluate(self, full, stopped)

    monkeypatch.setattr(ComponentsError, "evaluate", spy)
    optimise(
        merged,
        fovs,
        setup,
        fov_counts=(2, 4),
        reservoir_fraction=0.35,
        n_splits=2,
        draws=2,
        permuted_orders=1,
        dispersed_orders=0,
    )
    assert calls
    k = len(pooled.loadings)
    for metric, reference in calls:
        # The held-back fields that judge the stop, and the group they belong to.
        held = set(reference["image_file"])
        same = merged[["condition", "biological_repeat"]].eq(
            reference[["condition", "biological_repeat"]].iloc[0]
        )
        group = merged[same.all(axis=1)]
        training = group[~group["image_file"].isin(held)]
        assert held and not held & set(training["image_file"])
        centre, _scale, loadings = robust_components(training[list(names)].to_numpy(float), k)
        assert np.allclose(metric.loadings, loadings) and np.allclose(metric.centre, centre)
        assert not np.allclose(metric.loadings, pooled.loadings)


def test_a_metric_without_fitted_state_is_its_own_trained_copy():
    metric = CovarianceError(("area", "shape"), True)
    criterion = OutcomeCriterion(metric, 0.5, "lower_is_better")
    assert criterion.trained(None) is criterion  # type: ignore[arg-type]
