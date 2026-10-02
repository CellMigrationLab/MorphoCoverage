"""Tests for leakage-safe feature transforms."""

import json

import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables
from morphocoverage.transforms import PCA, Chain, Standardize


def _data(values, features=("a", "b")):
    matrix = np.asarray(values, dtype=float)
    objects = pd.DataFrame(matrix, columns=list(features)).assign(
        fov_id=[f"F{i % 3}" for i in range(len(matrix))]
    )
    return validate_input_tables(objects, feature_cols=list(features))


def test_standardize_uses_visible_mean_and_sd():
    data = _data([[1.0, 10.0], [3.0, 30.0], [5.0, 20.0]])
    fitted = Standardize().fit(data)
    assert fitted.state["mean"] == [3.0, 20.0]
    assert fitted.state["sd"] == [2.0, 10.0]
    np.testing.assert_allclose(fitted.apply(data), [[-1, -1], [0, 1], [1, 0]])
    assert (fitted.n_fit_objects, fitted.n_fit_fovs) == (3, 3)


@pytest.mark.parametrize(
    ("values", "reason"),
    [([[1.0, 2.0]], "fewer than 2"), ([[1.0, 2.0], [1.0, 3.0]], "zero SD")],
)
def test_standardize_refuses_to_guess_a_scale(values, reason):
    fitted = Standardize().fit(_data(values))
    assert not fitted.fitted and reason in fitted.reason
    with pytest.raises(ValueError, match="not fitted"):
        fitted.apply(_data(values))
    assert fitted.provenance()["reason"] == fitted.reason


def test_pca_matches_numpy_and_has_a_deterministic_sign():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 3)) @ np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 0.1]])
    data = _data(values, ("a", "b", "d"))
    fitted = PCA(2).fit(data)
    scores = fitted.apply(data)
    centred = values - values.mean(axis=0)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(centred.T)))[::-1]
    np.testing.assert_allclose(scores.var(axis=0, ddof=1), eigenvalues[:2])
    ratios = fitted.state["explained_variance_ratio"]
    np.testing.assert_allclose(ratios, eigenvalues[:2] / eigenvalues.sum())
    components = np.asarray(fitted.state["components"])
    assert (components[np.arange(2), np.abs(components).argmax(1)] > 0).all()
    # Flipping the data does not flip the reported axes.
    flipped = PCA(2).fit(_data(-values, ("a", "b", "d")))
    np.testing.assert_allclose(flipped.state["components"], components)
    assert fitted.output_names == ("pc1", "pc2")


def test_pca_needs_enough_objects_and_features():
    assert "features" in PCA(3).fit(_data([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]])).reason
    assert "observations" in PCA(2).fit(_data([[1.0, 2.0], [2.0, 1.0]])).reason
    with pytest.raises(ValueError):
        PCA(0)


def test_chain_fits_each_step_on_the_previous_output():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(40, 2)) * [1.0, 100.0]
    data = _data(values)
    chain = Chain([Standardize(), PCA(1)])
    fitted = chain.fit(data)
    standardized = Standardize().fit(data).apply(data)
    expected = PCA(1).fit_matrix(standardized, ("a", "b"))[0]
    assert fitted.state["steps"][1] == expected
    assert chain.name == "standardize>pca1"
    assert "standardize" in Chain([Standardize(), PCA(1)]).fit(_data([[1.0, 2.0]])).reason
    assert chain == Chain([Standardize(), PCA(1)])
    with pytest.raises(ValueError):
        Chain([])


def test_fitted_state_is_json_provenance():
    fitted = Chain([Standardize(), PCA(1)]).fit(_data([[1.0, 2.0], [2.0, 5.0], [4.0, 3.0]]))
    record = json.loads(json.dumps(fitted.provenance()))
    assert record["config_id"] == Chain([Standardize(), PCA(1)]).config_id
    assert record["input_features"] == ["a", "b"]
    assert record["output_names"] == ["pc1"]


def test_config_ids_distinguish_transforms():
    ids = {
        Standardize().config_id,
        PCA(1).config_id,
        PCA(2).config_id,
        Chain([Standardize(), PCA(1)]).config_id,
        Chain([PCA(1), Standardize()]).config_id,
    }
    assert len(ids) == 5


def test_a_fitted_transform_rejects_other_features():
    fitted = Standardize().fit(_data([[1.0, 2.0], [2.0, 5.0]]))
    with pytest.raises(ValueError, match="cannot be applied"):
        fitted.apply(_data([[1.0, 2.0], [2.0, 5.0]], ("a", "d")))


@pytest.mark.parametrize("transform", [Standardize(), PCA(1), Chain([Standardize(), PCA(1)])])
def test_transforms_refuse_missing_values_explicitly(transform):
    data = _data([[1.0, 2.0], [float("nan"), 3.0], [4.0, 5.0]])
    fitted = transform.fit(data)
    assert not fitted.fitted
    assert "missing values in a (1 observation)" in fitted.reason
