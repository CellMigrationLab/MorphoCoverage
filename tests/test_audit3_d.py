"""Regressions of the third audit, group D: novelty, rare states, distances and outcomes.

Each test names the finding it pins (D-1 to D-11 and the fold suspicion) and
fails on the code before the fix. Constructed data only: every test checks a
contract, not how well a strategy performs.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from morphocoverage import (
    ObjectiveConfig,
    SamplingObjective,
    distances,
    multivariate_outcomes,
    outcomes,
    transforms,
    validate_input_tables,
)
from morphocoverage.evaluator import objects_in_fovs, restrict_input
from morphocoverage.multivariate_outcomes import (
    CovarianceError,
    EnergyError,
    MMDError,
    SlicedWassersteinError,
)
from morphocoverage.outcomes import (
    RandomObjectRelativeError,
    RareStateDetected,
    RelativeMeanError,
)
from morphocoverage.strategies import (
    IncidenceCoverageEstimator,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
    support_novelty,
)
from morphocoverage.strategies.rare_incidence import _positive_fovs
from morphocoverage.strategies.support_novelty import UPPER_QUANTITY
from morphocoverage.transforms import PCA, Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _data(values_per_fov, features=("a", "b")):
    """Validated data with one FOV (``f<j>.tif``) per array of ``values_per_fov``."""
    rows = []
    for j, values in enumerate(values_per_fov):
        for row in np.atleast_2d(np.asarray(values, dtype=float)):
            rows.append({"image_file": f"f{j}.tif", **dict(zip(features, row, strict=True))})
    names = [f"f{j}.tif" for j in range(len(values_per_fov))]
    return validate_input_tables(
        pd.DataFrame(rows), fovs=pd.DataFrame({"image_file": names}), feature_cols=list(features)
    )


def _fovs(data, rows):
    """The FOVs at ``rows`` of ``data.fovs`` with their observations."""
    fovs = data.fovs.iloc[list(rows)]
    return restrict_input(data, objects_in_fovs(data, fovs), fovs)


def test_d_1_random_object_error_drops_the_correction_against_a_disjoint_reference():
    rng = np.random.default_rng(0)
    data = _data([rng.normal(10, 1, size=(20, 2)) for _ in range(10)])
    stop, reservoir = _fovs(data, range(7)), _fovs(data, range(7, 10))
    metric = RandomObjectRelativeError("a")
    reference = reservoir.feature_values("a")
    # 140 stop objects against a 60-object reservoir: no longer refused.
    expected = math.sqrt(reference.var(ddof=1) / 140) / abs(reference.mean())
    assert metric.evaluate(reservoir, stop) == pytest.approx(expected)
    # A stop inside the reference keeps the finite-population correction.
    full = data.feature_values("a")
    corrected = math.sqrt(full.var(ddof=1) / 140 * (1 - 140 / 200)) / abs(full.mean())
    assert metric.evaluate(data, stop) == pytest.approx(corrected)


def test_d_2_a_sparse_recurrent_chain_is_new_in_both_fovs_as_documented():
    assert "Support is symmetric" not in support_novelty.__doc__
    assert "not symmetric" in support_novelty.__doc__
    rng = np.random.default_rng(0)
    state = np.column_stack([10 + np.arange(5) * 0.9, np.zeros(5)])
    common = [rng.normal(0, 0.3, size=(40, 2)) for _ in range(6)]
    data = _data(
        [np.vstack([common[0], state]), np.vstack([common[1], state + [0.45, 0]])] + common[2:]
    )
    evidence = SupportNoveltyEstimator(1.0, 1, None, 2, 3).estimate(data, objective=POPULATION)
    assert evidence.diagnostics["novel_fovs"] == 2


def test_d_3_too_few_other_fovs_for_the_support_is_insufficient_not_all_novel():
    rng = np.random.default_rng(1)
    two = _data([rng.normal(size=(50, 2)), rng.normal(size=(50, 2))])
    estimator = SupportNoveltyEstimator(1.0, 2, None, 2)
    evidence = estimator.estimate(two, objective=POPULATION)
    for quantity in evidence.quantities.values():
        assert quantity.availability.value == "insufficient_data"
        assert "fewer than 3 visible FOVs" in quantity.reason
        assert "min_other_fovs=2" in quantity.reason
    assert estimator.novel_observation_flags(two) is None
    three = _data([rng.normal(size=(50, 2)) for _ in range(3)])
    upper = estimator.estimate(three, objective=POPULATION).quantities[UPPER_QUANTITY]
    assert upper.availability.value == "available"


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
    data = _data([constant, constant], features=("f0", "f1"))
    stop = _fovs(data, [0])
    assert np.linalg.norm(np.cov(data.feature_matrix(), rowvar=False)) > 0  # rounding noise
    assert CovarianceError(("f0", "f1"), False).evaluate(data, stop) is None
    assert CovarianceError(("f0", "f1"), True).evaluate(data, stop) is None


def test_d_6_rare_state_detected_counts_like_the_estimator_and_accepts_missing_labels():
    data = _data([[[0, 0], [1, 1]], [[0, 1]], [[2, 2]]])
    data.objects["rare"] = [True, False, False, True]
    metric = RareStateDetected("rare", 2)
    assert metric.evaluate(data, data) == 1.0
    assert metric.evaluate(data, _fovs(data, [0, 1])) == 0.0
    assert _positive_fovs(data, "rare") == 2  # the one count, shared with the estimator
    for missing in (pd.array([True, pd.NA, False, True], dtype="boolean"), [1.0, np.nan, 0, 1]):
        data.objects["rare"] = missing
        assert metric.evaluate(data, data) is None
    data.objects["rare"] = [2, 0, 0, 1]
    with pytest.raises(ValueError, match="booleans or 0/1"):
        metric.evaluate(data, data)


def test_d_7_a_label_names_the_metric_and_the_default_name_is_unchanged():
    narrow = MMDError(("a", "b"), True, 500, 0, 0.5, label="mmd_bw0.5")
    wide = MMDError(("a", "b"), True, 500, 0, 2.0, label="mmd_bw2")
    assert (narrow.name, wide.name) == ("mmd_bw0.5", "mmd_bw2")
    plain = MMDError(("a", "b"), True, 500, 0, 0.5)
    assert plain.name == "mmd_error:a,b"
    assert "label" not in plain.parameters
    assert narrow.parameters["label"] == "mmd_bw0.5"
    # Without a label the config id is the one computed from the settings alone.
    assert plain.config_id == MMDError(("a", "b"), True, 500, 0, 0.5, label=None).config_id
    assert narrow.config_id != plain.config_id
    assert RelativeMeanError("a", label="mean a").name == "mean a"
    assert RelativeMeanError("a").name == "relative_mean_error:a"
    assert RandomObjectRelativeError("a", label="srs a").name == "srs a"
    assert RareStateDetected("rare", 1, label="rare seen").name == "rare seen"
    with pytest.raises(TypeError, match="label"):
        RelativeMeanError("a", label="")


def test_d_8_zero_tolerance_on_the_upper_bound_is_refused():
    with pytest.raises(ValueError, match="never reached"):
        SupportNoveltyStrategy(1.0, 0.0, 1, None, 2, quantity=UPPER_QUANTITY)
    SupportNoveltyStrategy(1.0, 0.0, 1, None, 2)  # the observed fraction can reach 0


def test_d_9_numpy_integers_are_accepted_and_stored_as_python_ints():
    made = [
        SupportNoveltyEstimator(1.0, np.int64(1), None, np.int64(2), np.int64(3)),
        SupportNoveltyStrategy(1.0, 0.1, np.int64(1), None, np.int64(2), np.int64(3)),
        IncidenceCoverageEstimator("s", np.int64(3)),
        PCA(np.int64(2)),
        RareStateDetected("r", np.int64(2)),
        EnergyError(("a",), True, np.int64(500), np.int64(0)),
        SlicedWassersteinError(("a",), True, np.int32(10), np.int64(0)),
    ]
    for item in made:
        for value in item.parameters.values():
            assert not isinstance(value, np.integer)
    assert PCA(np.int64(2)).config_id == PCA(2).config_id
    assert PCA(np.int64(2)).name == "pca2"
    with pytest.raises(TypeError):
        PCA(True)


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


def test_d_susp_a_feature_varying_in_one_fov_is_named_in_the_reason():
    rng = np.random.default_rng(3)
    values = [np.column_stack([rng.normal(size=20), np.zeros(20)]) for _ in range(5)]
    values.append(np.column_stack([rng.normal(size=20), rng.integers(1, 4, 20).astype(float)]))
    data = _data(values)
    evidence = SupportNoveltyEstimator(1.0, 1, Standardize(), 2).estimate(
        data, objective=POPULATION
    )
    for quantity in evidence.quantities.values():
        assert quantity.availability.value == "insufficient_data"
        assert "feature 'b' varies only in FOV f5.tif" in quantity.reason


def test_j_16_modules_list_their_public_names():
    for module in (distances, transforms, outcomes, multivariate_outcomes):
        names = module.__all__
        assert names and all(not name.startswith("_") for name in names)
        assert all(hasattr(module, name) for name in names)
