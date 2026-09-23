"""Tests for the spatial semivariogram diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scenarios import SCENARIOS

from morphocoverage import validate_input_tables
from morphocoverage.spatial import empirical_semivariogram, fov_means, fov_semivariogram


def test_a_linear_trend_gives_half_the_squared_lag():
    x = np.arange(10.0)
    table = empirical_semivariogram(x, np.column_stack([x, np.zeros(10)]), [0.5, 1.5, 2.5, 3.5])
    assert list(table["gamma"]) == pytest.approx([0.5, 2.0, 4.5])
    assert list(table["n_pairs"]) == [9, 8, 7]
    assert list(table["mean_lag"]) == pytest.approx([1.0, 2.0, 3.0])


def test_independent_values_give_a_flat_semivariogram_at_their_variance():
    # Matheron's estimator is unbiased for sigma^2 at every lag; averaged over
    # repeated fields it sits at 4 in every bin.
    gammas = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(100, 2))
        values = rng.normal(0, 2.0, size=100)
        gammas.append(empirical_semivariogram(values, coords, [0, 25, 50, 75, 100])["gamma"])
    assert np.allclose(np.mean(gammas, axis=0), 4.0, rtol=0.03)


def test_several_features_average_their_squared_differences():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    values = np.array([[0.0, 0.0], [2.0, 4.0]])
    table = empirical_semivariogram(values, coords, [0, 2])
    assert table["gamma"].iloc[0] == pytest.approx(0.5 * (4 + 16) / 2)


def test_empty_bins_are_omitted_not_nan():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    table = empirical_semivariogram(np.array([0.0, 1.0, 5.0]), coords, [0, 2, 5, 20])
    assert list(table["lag_low"]) == [0.0, 5.0]
    assert not table["gamma"].isna().any()


def test_neighbour_correlation_rises_with_distance_at_short_lags():
    built = SCENARIOS["neighbour_correlation"]()
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=["feature_a"])
    table = fov_semivariogram(data, "stage_x", "stage_y", [0, 150, 300, 450])
    assert np.all(np.diff(table["gamma"]) > 0)
    homogeneous = SCENARIOS["homogeneous"]()
    flat = validate_input_tables(
        homogeneous.objects, fovs=homogeneous.fovs, feature_cols=["feature_a"]
    )
    flat_table = fov_semivariogram(flat, "stage_x", "stage_y", [0, 150, 300, 450])
    assert flat_table["gamma"].max() < table["gamma"].min() * 2


def test_fov_means_skip_empty_fovs_and_refuse_missing_values():
    objects = pd.DataFrame({"image_file": ["a", "a", "c"], "v": [1.0, 3.0, 5.0]})
    fovs = pd.DataFrame({"image_file": ["a", "b", "c"], "x": [0.0, 1.0, 2.0], "y": 0.0})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["v"])
    means, rows = fov_means(data)
    assert means[:, 0].tolist() == [2.0, 5.0] and rows.tolist() == [0, 2]
    assert fov_semivariogram(data, "x", "y", [0, 3])["n_pairs"].iloc[0] == 1
    objects.loc[0, "v"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fov_means(validate_input_tables(objects, fovs=fovs, feature_cols=["v"]))
    with pytest.raises(ValueError, match="not a column"):
        fov_semivariogram(data, "stage_x", "y", [0, 3])


@pytest.mark.parametrize(
    "values, coords, edges",
    [
        (np.zeros(3), np.zeros((2, 2)), [0, 1]),
        (np.zeros(1), np.zeros((1, 2)), [0, 1]),
        (np.array([0.0, np.nan]), np.zeros((2, 2)), [0, 1]),
        (np.zeros(2), np.zeros((2, 2)), [1, 0]),
        (np.zeros(2), np.zeros((2, 2)), [0]),
    ],
)
def test_invalid_input_is_refused(values, coords, edges):
    with pytest.raises(ValueError):
        empirical_semivariogram(values, coords, edges)
