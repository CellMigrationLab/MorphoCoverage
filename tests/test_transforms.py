"""Tests for leakage-safe feature transforms."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import _two_tables
from _helpers_strategies import POPULATION, _estimate_input

from morphocoverage import validate_input_tables
from morphocoverage.multivariate_outcomes import CentroidError, EnergyError, SlicedWassersteinError
from morphocoverage.outcomes import RareStateDetected
from morphocoverage.sampling_unit import unusual_objects
from morphocoverage.schema import fov_positions
from morphocoverage.strategies import (
    FOVBootstrapEstimator,
    IncidenceCoverageEstimator,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
)
from morphocoverage.transforms import (
    PCA,
    BlockWeights,
    Chain,
    ColumnWeights,
    RobustStandardize,
    Standardize,
)
from morphocoverage.workflows import default_novelty


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
    assert (fitted.n_fit_observations, fitted.n_fit_fovs) == (3, 3)
    provenance = fitted.provenance()
    assert (provenance["n_fit_observations"], provenance["n_fit_fovs"]) == (3, 3)
    assert "n_fit_objects" not in provenance  # the duplicate key is gone (owner, 2026-11-10)


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


def test_d3_a_pca_summary_of_one_standardized_feature_is_never_defined():
    rng = np.random.default_rng(4)
    fields = [
        np.array([[1.0]]),
        np.array([[1.0]]),
        rng.normal(size=(3, 1)),
        rng.normal(size=(3, 1)),
    ]
    estimator = FOVBootstrapEstimator("eigenspectrum", Standardize(), 200, 0, 2)
    for k in range(1, len(fields) + 1):
        quantity = estimator.estimate(_estimate_input(fields[:k]), objective=POPULATION)
        assert quantity.quantities[estimator.quantity].availability == "not_applicable"
    assert Standardize().output_width(3) == 3
    assert PCA(2).output_width(3) == 2 and PCA(2).output_width(1) is None
    assert Chain([Standardize(), PCA(2)]).output_width(4) == 2


def test_d4_pca_keeps_independent_features_on_very_different_scales():
    rng = np.random.default_rng(7)
    matrix = np.c_[rng.normal(0, 1e8, 500), rng.normal(0, 0.05, 500)]
    fitted = PCA(2).fit_matrix(matrix, ("area_nm2", "ecc"))
    assert not isinstance(fitted, str)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    reason = PCA(2).fit_matrix(np.c_[a, 2 * a, -a], ("x", "y", "z"))
    assert isinstance(reason, str) and "floating-point rounding" in reason
    # A constant column with a large mean is rounding noise after centring.
    reason = PCA(2).fit_matrix(np.c_[a * 1e-6, np.full(4, 1e6 + 0.1)], ("x", "y"))
    assert isinstance(reason, str)


def test_d6_outcome_standardization_is_the_package_transform(monkeypatch):
    rng = np.random.default_rng(0)
    full = _estimate_input([rng.normal(3, 2, (10, 2)) for _ in range(5)], ("a", "b"))
    stop = _estimate_input([rng.normal(3, 2, (10, 2)) for _ in range(2)], ("a", "b"))
    calls = []
    original = Standardize.fit_matrix

    def recording(self, matrix, names):
        calls.append(names)
        return original(self, matrix, names)

    monkeypatch.setattr(Standardize, "fit_matrix", recording)
    value = CentroidError(("a", "b"), True).evaluate(full, stop)
    assert calls == [("a", "b")]
    reference = np.column_stack([full.feature_values(f) for f in ("a", "b")])
    sample = np.column_stack([stop.feature_values(f) for f in ("a", "b")])
    centre, scale = reference.mean(axis=0), reference.std(axis=0, ddof=1)
    expected = np.sqrt(np.mean(((sample - centre) / scale).mean(0) ** 2))
    assert value == pytest.approx(expected)


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


def _debris_table():
    """16 fields; a compact 8-object state at +6 SD in field 3, debris at +60 SD in 8, 11."""
    rng = np.random.default_rng(1)
    rows = []
    for j in range(16):
        xy = rng.normal(size=(30, 2))
        if j == 3:
            xy = np.r_[xy, np.array([6.0, 0.0]) + 0.3 * rng.normal(size=(8, 2))]
        if j in (8, 11):
            xy = np.r_[xy, np.array([60.0, 0.0]) + 3 * rng.normal(size=(20, 2))]
        rows.append(pd.DataFrame({"position_id": f"f{j:02d}", "a": xy[:, 0], "b": xy[:, 1]}))
    objects = pd.concat(rows, ignore_index=True)
    fovs = pd.DataFrame({"position_id": [f"f{j:02d}" for j in range(16)]})
    return validate_input_tables(objects, fovs=fovs, feature_cols=["a", "b"])


def test_p6_2_debris_in_two_fields_does_not_mask_a_new_state():
    data = _debris_table()
    positions = fov_positions(data, data.fovs)

    def new_fields(transform):
        estimator = SupportNoveltyEstimator(np.sqrt(2), 1, transform, 5, 3)
        members = estimator.novel_observation_flags(data, new_states_only=True)
        return set(positions[members].tolist())

    # The mean and SD are widened by the debris, and the state disappears.
    assert 3 not in new_fields(Standardize())
    assert 3 in new_fields(RobustStandardize())
    assert isinstance(default_novelty(["a", "b"]).transform, RobustStandardize)
    # The robust scale is the SD for normal data, and ignores a few extreme values.
    rng = np.random.default_rng(2)
    values = rng.normal(size=(2000, 1))
    state, _ = RobustStandardize().fit_matrix(values, ("a",))
    assert state["scale"][0] == pytest.approx(1.0, abs=0.05)
    spoiled = np.r_[values, np.full((20, 1), 500.0)]
    assert RobustStandardize().fit_matrix(spoiled, ("a",))[0]["scale"][0] < 1.1
    # A feature with a MAD of 0 that still varies (a count) is scaled by its SD.
    counts = np.r_[np.zeros((8, 1)), np.array([[1.0], [3.0]])]
    state, _ = RobustStandardize().fit_matrix(counts, ("n",))
    assert state["sd_scaled"] == ["n"]
    assert state["scale"][0] == pytest.approx(counts.std(ddof=1))


def test_p9_7_near_constant_features_are_refused_and_named():
    rng = np.random.default_rng(6)
    noisy = np.c_[rng.normal(size=50), 7 + 1e-9 * rng.normal(size=50)]
    for transform in (Standardize(), RobustStandardize()):
        reason = transform.fit_matrix(noisy, ("area", "texture"))
        assert isinstance(reason, str) and "['texture']" in reason
    # The unusual-objects message names the feature, not its column index.
    rows, fields = [], []
    for repeat in ("R1", "R2", "R3"):
        for j in range(4):
            name = f"{repeat}f{j}"
            fields.append(
                {
                    "condition": "MOCK",
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "position_id": name,
                }
            )
            for value in rng.normal(size=10):
                rows.append(
                    {
                        "condition": "MOCK",
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                        "position_id": name,
                    }
                    | {"area": value, "texture": 7.0}
                )
    result = unusual_objects(
        pd.DataFrame(rows),
        pd.DataFrame(fields),
        feature_cols=["area", "texture"],
        reference={"condition": "MOCK"},
    )
    statuses = " ".join(result.groups["status"])
    assert "['texture']" in statuses and "['1']" not in statuses


def test_the_novelty_radius_reads_the_square_root_of_the_weighted_inertia():
    z, names = _two_tables()
    blocks = (tuple(names[:4]), tuple(names[4:]))
    state, _ = BlockWeights(blocks, inertia_radius=True).fit_matrix(z, tuple(names))
    plain, _ = BlockWeights(blocks).fit_matrix(z, tuple(names))
    scaled = np.asarray(state["weights"])
    # The declared radius sqrt(d) in the scaled space is sqrt(total inertia) in the MFA one.
    assert (np.var(z * scaled, axis=0, ddof=1)).sum() == pytest.approx(len(names))
    inertia = (np.var(z * np.asarray(plain["weights"]), axis=0, ddof=1)).sum()
    np.testing.assert_allclose(scaled, np.asarray(plain["weights"]) * np.sqrt(6 / inertia))


def test_d1_a_constant_feature_is_refused_whatever_its_rounding():
    for constant in (1.0, 0.1, 0.7, 123.456):
        matrix = np.c_[[1.0, 2.0, 3.0], np.full(3, constant)]
        assert isinstance(Standardize().fit_matrix(matrix, ("a", "b")), str)
    fitted = Standardize().fit_matrix(np.c_[[1.0, 2.0, 3.0], [0.1, 0.2, 0.4]], ("a", "b"))
    assert not isinstance(fitted, str)


def test_d2_pca_refuses_components_beyond_the_data_rank():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    reason = PCA(2).fit_matrix(np.c_[a, 2 * a, -a], ("x", "y", "z"))
    assert isinstance(reason, str) and "fewer than 2 dimensions" in reason
    assert not isinstance(PCA(1).fit_matrix(np.c_[a, 2 * a, -a], ("x", "y", "z")), str)


def test_weight_settings_are_refused_not_coerced_495():
    # float(True) is 1.0 and float("2.0") is 2.0; bool("False") is True.
    for bad in ((True,), ("2.0",), (np.bool_(True),), (1.0, "1")):
        with pytest.raises(ValueError, match="weights"):
            ColumnWeights(bad)
    assert ColumnWeights((1, np.float32(0.5), np.int64(2))).weights == (1.0, 0.5, 2.0)
    for flag in ("False", 1, None):
        with pytest.raises(TypeError, match="inertia_radius"):
            BlockWeights((("a", "b"),), inertia_radius=flag)
    assert BlockWeights((("a",),), inertia_radius=np.bool_(True)).inertia_radius is True
    for blocks in (((1, "b"),), (("",),), ((" ",),)):
        with pytest.raises(TypeError, match="feature names"):
            BlockWeights(blocks)
