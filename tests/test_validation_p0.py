"""P0 of the benchmark (docs/common/validation.md): what is checked by algebra
and deterministic tests, never by simulation.

1. Estimability: the candidate contrast layer (``data/simulations/decisions/candidates/
   fixed_block.py``) says a contrast is estimable exactly when the design's graph says so,
   on connected, disconnected and parent-chain designs and on random designs.
2. Rotation equivariance of every embedding operation the package applies today: a global
   orthogonal change of the embedding's basis must leave distances, p-values and scores
   unchanged (and move centred values by the same rotation). The operations that read the
   coordinates one at a time fail, and are marked ``xfail(strict=True)`` with the reason:
   these are P0's findings, kept in the suite so that a change that fixes one shows as an
   unexpected pass and must remove its mark.
3. Leakage: what a held-back evaluation reads is never fitted on what it holds back.
"""

from __future__ import annotations

import ast
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "simulations"))

from decisions.candidates import fixed_block
from decisions.design import Design, connected_incomplete, disconnected
from decisions.embedding import EmbeddingSpec, generate, random_rotation, rotate

from morphocoverage.embedding import _pca, robust_scale
from morphocoverage.feature_tables import mfa_weights
from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN
from morphocoverage.sampling_unit import _outliers, populations
from morphocoverage.sampling_unit._population_rule import population_rule
from morphocoverage.sampling_unit.whole_table import (
    _table_rows,
    repeat_test,
    shuffle_test,
)
from morphocoverage.separation import _scores, separation
from morphocoverage.workflows import Setup
from morphocoverage.workflows._feature_space import centred_tables

COORDINATEWISE = "P0 finding: {} reads the coordinates one at a time, so a global rotation of "
COORDINATEWISE += "the embedding's basis changes its result (plan v2, principle 4; held for P3)"

# --- 1. Estimability ------------------------------------------------------------------------


def _pairs(conditions: tuple[str, ...]) -> list[dict[str, float]]:
    return [{a: 1.0, b: -1.0} for a, b in combinations(conditions, 2)]


def test_a_complete_design_pairs_every_condition_directly():
    presence = {f"R{r}": ("control", "A", "B") for r in range(4)}
    for weights in _pairs(("control", "A", "B")):
        assert fixed_block.regime(presence, weights) == "direct"


def test_a_connected_incomplete_design_estimates_pairs_that_never_meet_three_times():
    conditions = ("control", "A", "B", "C")
    presence = dict(enumerate(connected_incomplete(conditions, 6)))
    assert fixed_block.regime(presence, {"A": 1.0, "control": -1.0}) == "direct"
    # A and B share 2 repeats: no paired test from 3 repeats, but the model estimates it.
    assert fixed_block.regime(presence, {"A": 1.0, "B": -1.0}) == "model"


def test_a_disconnected_design_says_a_contrast_across_its_sets_is_not_estimable():
    conditions = ("control", "A", "B", "C")
    presence = dict(enumerate(disconnected(conditions, 6)))
    assert fixed_block.regime(presence, {"A": 1.0, "B": -1.0}) == "not estimable"
    assert fixed_block.regime(presence, {"A": 1.0, "control": -1.0}) == "direct"
    # The interaction across the sets is a difference of two within-set contrasts.
    interaction = {"A": 1.0, "control": -1.0, "C": -1.0, "B": 1.0}
    assert fixed_block.regime(presence, interaction) == "model"


def test_a_parent_chain_is_estimable_through_its_bridge_and_not_without_it():
    bridged = {r: ("P", "C") if r < 3 else ("C", "T") for r in range(6)}
    assert fixed_block.regime(bridged, {"T": 1.0, "P": -1.0}) == "model"
    broken = {r: ("P", "C") if r < 3 else ("T", "U") for r in range(6)}
    assert fixed_block.regime(broken, {"T": 1.0, "P": -1.0}) == "not estimable"


def test_a_contrast_without_a_control_is_direct_when_its_conditions_share_repeats():
    presence = {r: ("A", "B") for r in range(4)}
    assert fixed_block.regime(presence, {"A": 1.0, "B": -1.0}) == "direct"


@pytest.mark.parametrize("seed", range(40))
def test_the_algebra_agrees_with_the_graph_on_random_designs(seed):
    rng = np.random.default_rng(seed)
    conditions = [f"c{i}" for i in range(rng.integers(3, 7))]
    presence = {}
    for r in range(rng.integers(2, 7)):
        size = rng.integers(1, len(conditions) + 1)
        presence[f"R{r}"] = tuple(rng.choice(conditions, size=size, replace=False))
    layout = fixed_block.Layout.from_presence(presence)
    held = list(layout.conditions)
    contrasts = _pairs(tuple(held))
    for _ in range(5):
        weights = rng.normal(size=len(held))
        contrasts.append(dict(zip(held, weights - weights.mean(), strict=True)))
    for weights in contrasts:
        algebra = fixed_block.estimable(layout.matrix(), layout.contrast(weights))
        assert algebra == fixed_block.graph_estimable(presence, weights), (presence, weights)


def test_the_model_estimate_is_the_mean_paired_difference_in_a_complete_design():
    rng = np.random.default_rng(3)
    rows = [
        {"repeat": r, "condition": c, "value": rng.normal() + 2 * r}
        for r in range(5)
        for c in ("control", "A", "B")
    ]
    values = pd.DataFrame(rows)
    wide = values.pivot(index="repeat", columns="condition", values="value")
    found = fixed_block.fit_model(values, {"A": 1.0, "control": -1.0})
    assert found["estimate"] == pytest.approx((wide["A"] - wide["control"]).mean())


# --- 2. Rotation equivariance ---------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding() -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    """An anisotropic embedding (8 columns, spectrum j^-0.7 in a random basis), 3 repeats,
    two conditions, and a random rotation."""
    design = Design(repeats=3, fields=6, cells=12.0)
    data = generate(design, EmbeddingSpec(dims=8, shift=0.5), np.random.default_rng(11))
    rotation = random_rotation(8, np.random.default_rng(12))
    return data.objects, data.fovs, list(data.columns), rotation


def _distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def _standardised(values: np.ndarray) -> np.ndarray:
    centre, scale = robust_scale(values)
    return (values - centre) / scale


@pytest.mark.xfail(strict=True, reason=COORDINATEWISE.format("the robust scaling (median, MAD)"))
def test_robust_scaling_keeps_the_distances_under_a_rotation(embedding):
    objects, _, columns, rotation = embedding
    values = objects[columns].to_numpy()
    before = _distances(_standardised(values))
    after = _distances(_standardised(values @ rotation))
    np.testing.assert_allclose(after, before, rtol=1e-6)


@pytest.mark.xfail(strict=True, reason=COORDINATEWISE.format("the centring of the tables"))
def test_the_centring_of_the_tables_moves_with_a_rotation(embedding):
    objects, fovs, columns, rotation = embedding
    setup = Setup(readout="x0", control={"condition": "control"}, feature_tables={"DINO": columns})
    base = objects.assign(x0=1.0)
    centred = centred_tables(base, fovs, setup)[columns].to_numpy()
    turned = centred_tables(rotate(base, columns, rotation), fovs, setup)[columns].to_numpy()
    np.testing.assert_allclose(turned, centred @ rotation, atol=1e-8)


def _meta(objects: pd.DataFrame) -> pd.DataFrame:
    codes = np.unique(objects["position_id"].to_numpy(), return_inverse=True)[1]
    return pd.DataFrame(
        {
            CONDITION_COLUMN: objects[CONDITION_COLUMN].to_numpy(),
            REPEAT_COLUMN: objects[REPEAT_COLUMN].to_numpy(),
            "field": codes,
        }
    )


@pytest.mark.xfail(strict=True, reason=COORDINATEWISE.format("the whole-table test's scaling"))
def test_the_whole_table_test_gives_the_same_p_after_a_rotation(embedding):
    objects, _, columns, rotation = embedding
    values = objects[columns].to_numpy()
    pairs = [("treated", "control")]
    before = _table_rows("t", values, _meta(objects), pairs, 19, 0)[0]["p_repeats"]
    after = _table_rows("t", values @ rotation, _meta(objects), pairs, 19, 0)[0]["p_repeats"]
    assert after == pytest.approx(before, rel=1e-9)


def test_the_whole_table_projection_test_and_shuffle_are_rotation_invariant(embedding):
    _, _, _, rotation = embedding
    rng = np.random.default_rng(5)
    deltas = rng.normal(size=(5, 8)) + 0.3
    assert repeat_test(deltas @ rotation)[3] == pytest.approx(repeat_test(deltas)[3], rel=1e-9)
    means = rng.normal(size=(12, 8))
    side, repeat = np.tile([1, 0], 6), np.repeat(np.arange(3), 4)
    first = shuffle_test(means, side, repeat, 99, np.random.default_rng(1))
    second = shuffle_test(means @ rotation, side, repeat, 99, np.random.default_rng(1))
    assert second == pytest.approx(first, rel=1e-9)


@pytest.mark.xfail(strict=True, reason=COORDINATEWISE.format("the populations' distance space"))
def test_the_populations_distance_space_keeps_its_distances_under_a_rotation(embedding):
    objects, _, columns, rotation = embedding
    values = objects[columns].to_numpy()
    before = _distances(populations._standardised(values)[0])
    after = _distances(populations._standardised(values @ rotation)[0])
    np.testing.assert_allclose(after, before, rtol=1e-6)


def test_the_populations_purity_on_given_points_is_rotation_invariant(embedding):
    objects, _, columns, rotation = embedding
    values = objects[columns].to_numpy()
    member = objects[CONDITION_COLUMN].eq("treated").to_numpy()
    turned = populations._purity(values @ rotation, member)
    assert turned == pytest.approx(populations._purity(values, member))


@pytest.mark.xfail(strict=True, reason=COORDINATEWISE.format("the separation score"))
def test_the_separation_score_is_the_same_after_a_rotation(embedding):
    objects, fovs, columns, rotation = embedding
    before = separation(objects, fovs, spaces={"DINO": columns}).table
    after = separation(rotate(objects, columns, rotation), fovs, spaces={"DINO": columns}).table
    np.testing.assert_allclose(after["share"], before["share"])


def test_the_separation_neighbours_on_given_points_are_rotation_invariant(embedding):
    objects, _, columns, rotation = embedding
    values = objects[columns].to_numpy()
    labels = objects[CONDITION_COLUMN].to_numpy()
    blocks = objects["position_id"].to_numpy()
    first = _scores(values, labels, blocks, 5)
    second = _scores(values @ rotation, labels, blocks, 5)
    np.testing.assert_allclose(second[0], first[0])


@pytest.mark.xfail(strict=True, reason=COORDINATEWISE.format("the unusual-observation score"))
def test_the_unusual_observation_scores_are_the_same_after_a_rotation(embedding):
    objects, _, columns, rotation = embedding
    reference = objects[objects[CONDITION_COLUMN] == "control"]
    tested = objects[objects[CONDITION_COLUMN] == "treated"][columns].to_numpy()
    fields = reference["position_id"].to_numpy()
    groups = reference[REPEAT_COLUMN].to_numpy()
    values = reference[columns].to_numpy()

    def scores(turn: np.ndarray) -> np.ndarray:
        found = _outliers.score(
            values @ turn, fields, groups, tested @ turn, 5, 0.01, np.random.default_rng(0)
        )
        return found[0]

    np.testing.assert_allclose(scores(rotation), scores(np.eye(len(columns))), rtol=1e-9)


def test_the_mfa_weights_and_the_pca_are_rotation_invariant(embedding):
    objects, _, columns, rotation = embedding
    z = objects[columns].to_numpy()
    blocks = {"first": columns[:4], "second": columns[4:]}
    turned = z.copy()
    q = random_rotation(4, np.random.default_rng(2))
    turned[:, 4:] = z[:, 4:] @ q
    np.testing.assert_allclose(
        mfa_weights(turned, columns, blocks), mfa_weights(z, columns, blocks)
    )
    before, _, shares = _pca(z)
    after, _, turned_shares = _pca(z @ rotation)
    np.testing.assert_allclose(np.abs(after), np.abs(before), atol=1e-8)
    assert turned_shares == pytest.approx(shares)


# --- 3. Leakage -----------------------------------------------------------------------------


def test_the_unusual_threshold_is_fitted_on_the_reference_alone():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(120, 3))
    fields, groups = np.repeat(np.arange(12), 10), np.repeat(np.arange(3), 40)
    tested = rng.normal(size=(30, 3))
    first = _outliers.score(reference, fields, groups, tested, 5, 0.01, np.random.default_rng(0))
    moved = tested * 50 + 100
    second = _outliers.score(reference, fields, groups, moved, 5, 0.01, np.random.default_rng(0))
    assert second[1] == first[1]  # the threshold
    assert second[2] == first[2]  # the fitted standardisation


def test_the_population_rules_held_out_check_never_fits_on_the_held_out_rows(monkeypatch):
    from morphocoverage.sampling_unit import _population_rule as module

    rng = np.random.default_rng(6)
    values = rng.normal(size=(200, 3))
    member = values[:, 0] > 1.0
    groups = np.repeat(np.arange(20), 10)
    folds, fitted = [], []
    real_folds, real_fit = module._folds, module._fit

    def record_folds(*args: object) -> tuple:
        found = real_folds(*args)
        folds.extend(found[0])
        return found

    def record_fit(rows: np.ndarray, *args: object) -> tuple:
        fitted.append({tuple(np.round(r, 12)) for r in rows})
        return real_fit(rows, *args)

    monkeypatch.setattr(module, "_folds", record_folds)
    monkeypatch.setattr(module, "_fit", record_fit)
    population_rule(values, member, ["a", "b", "c"], groups=groups)
    assert len(fitted) == len(folds) + 1  # the shown rule on every row, then one per fold
    for test, train in zip(folds, fitted[1:], strict=True):
        assert not {tuple(np.round(r, 12)) for r in values[test]} & train


def test_a_sequential_rule_never_reads_a_field_after_its_stop():
    from decisions import p4_sequential
    from decisions.scalar import Scalar
    from decisions.sequential import sequence

    design = Design(repeats=1, conditions=("control",), fields=40, cells=20.0)
    data = sequence(design, Scalar(), np.random.default_rng(2))
    run = p4_sequential.METHODS[1].run  # cluster_se, deterministic
    context = {"rng": np.random.default_rng(0), "smoke": True, "alpha": 0.05, "replicate": 0}
    found = run(data, context)
    stop = int(found["fields"])
    later = set(data.order[stop:])
    changed = data.objects.copy()
    changed.loc[changed["position_id"].isin(later), "x0"] *= 10
    again = run(type(data)(changed, data.fovs, data.order, data.wells, data.truth,
                           data.truth_field), context)  # fmt: skip
    assert again["fields"] == found["fields"] and again["stopped"] == found["stopped"]


def test_the_package_never_imports_the_validation_extra():
    # The benchmark's comparators (statsmodels, harmonypy) are an optional extra used by the
    # scripts in data/simulations/decisions only (plan v2, "Readiness").
    found = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            found += [
                f"{path.name}: {n}"
                for n in names
                if n.split(".")[0] in {"statsmodels", "harmonypy"}
            ]
    assert not found
