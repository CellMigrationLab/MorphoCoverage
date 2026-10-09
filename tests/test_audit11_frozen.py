"""Frozen result snapshots own their containers (issue 108, #112)."""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from morphocoverage._frozen import FrozenDict, FrozenList, freeze, own, thaw
from morphocoverage.evidence import Evidence, Quantity
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.policies import ThresholdPolicy, redecide
from morphocoverage.results import StrategyResult


def _quantity() -> Quantity:
    return Quantity.available(0.1, direction="lower_is_better", units="x", component="population")


def test_freeze_copies_nested_containers_into_read_only_equal_ones():
    source = {"a": [1, {"b": 2}], "c": (1, [2]), "d": {1, 2}}
    frozen = freeze(source)
    assert frozen == {"a": [1, {"b": 2}], "c": (1, [2]), "d": frozenset({1, 2})}
    source["a"].append(9)
    source["a"][1]["b"] = 99
    assert frozen["a"] == [1, {"b": 2}]  # the caller's later edits do not reach it
    with pytest.raises(TypeError, match="read-only"):
        frozen["a"].append(3)
    with pytest.raises(TypeError, match="read-only"):
        frozen["a"][1]["b"] = 3
    with pytest.raises(TypeError, match="read-only"):
        frozen.update(x=1)


def test_frozen_containers_serialise_pickle_and_copy_like_plain_ones():
    frozen = freeze({"a": [1, 2], "b": {"c": 3}})
    assert json.loads(json.dumps(frozen)) == {"a": [1, 2], "b": {"c": 3}}
    again = pickle.loads(pickle.dumps(frozen))
    assert again == frozen and isinstance(again, FrozenDict) and isinstance(again["a"], FrozenList)
    assert copy.deepcopy(frozen) == frozen
    plain = thaw(frozen)
    plain["a"].append(3)  # a thawed copy is editable and independent
    assert frozen["a"] == [1, 2] and type(plain) is dict


def test_evidence_owns_its_quantities_and_diagnostics():
    quantities = {"score": _quantity()}
    diagnostics = {"note": ["a"], "inner": {"k": 1}}
    evidence = Evidence(quantities, diagnostics)
    quantities["n_fovs"] = "not a Quantity"  # a reserved name and a wrong type, added later
    quantities["other"] = _quantity()
    diagnostics["note"].append("b")
    assert set(evidence.quantities) == {"score"}
    assert evidence.diagnostics["note"] == ["a"]
    with pytest.raises(TypeError, match="read-only"):
        evidence.quantities["n_fovs"] = "x"
    with pytest.raises(TypeError, match="read-only"):
        evidence.diagnostics["inner"]["k"] = 2
    assert evidence.value("score") == 0.1  # ordinary reads are unchanged


def test_evidence_still_validates_what_it_is_given():
    with pytest.raises(ValueError, match="framework-owned"):
        Evidence({"n_fovs": _quantity()})
    with pytest.raises(TypeError, match="must be a Quantity"):
        Evidence({"x": 1.0})


def _result() -> StrategyResult:
    configuration = {"estimators": [{"config_id": "e", "parameters": {"k": 1}}], "policies": []}
    return StrategyResult(
        strategy="s",
        config_id="c",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        evidence=Evidence({"score": _quantity()}),
        decision=None,
        parameters=configuration,
        configuration=configuration,
        group={"condition": "a"},
        n_fovs=3,
        n_observations=9,
    )


def test_a_strategy_result_owns_its_mappings_and_they_do_not_alias_each_other():
    result = _result()
    assert result.parameters == result.configuration
    assert result.parameters is not result.configuration
    with pytest.raises(TypeError, match="read-only"):
        result.configuration["estimators"][0]["parameters"]["k"] = 2
    with pytest.raises(TypeError, match="read-only"):
        result.group["condition"] = "b"
    assert result.parameters["estimators"][0]["parameters"]["k"] == 1


def test_redecide_builds_an_independent_snapshot():
    result = _result()
    again = redecide(result, [ThresholdPolicy("score", 0.2)], name="t")
    assert again.configuration["estimators"] == result.configuration["estimators"]
    assert again.configuration is not result.configuration
    assert again.parameters is not again.configuration
    assert own({}) == {}


def _owned_results():
    import pandas as pd

    from morphocoverage.embedding import Embedding
    from morphocoverage.retrospective.overview import DatasetOverview
    from morphocoverage.sampling_unit.clustering import FOVClustering
    from morphocoverage.transforms import FittedTransform, Standardize

    table = pd.DataFrame({"a": [1, 2]})
    points = pd.DataFrame({"x": [0.0, 1.0]})
    settings = {"k": [1, 2]}
    return [
        (FOVClustering(table, (), settings), settings, "table", table),
        (Embedding(points, points.copy(), (0.5, 0.2), settings), settings, "points", points),
        (DatasetOverview({"n": 1}, table, ("a",), settings), settings, "layout", table),
        (
            FittedTransform(Standardize(), ("a",), 2, 1, {"s": [1.0]}),
            {"s": [1.0]},
            None,
            None,
        ),
    ]


def test_result_families_own_their_settings_and_tables():
    for result, source, frame_name, frame in _owned_results():
        container = result.state if hasattr(result, "state") else result.settings
        key = "s" if hasattr(result, "state") else "k"
        before = list(container[key])
        source[key].append(99)  # the caller edits what it passed in
        assert list(container[key]) == before, type(result).__name__
        with pytest.raises(TypeError, match="read-only"):
            container[key].append(1)
        with pytest.raises(TypeError, match="read-only"):
            container["new"] = 1
        if frame is not None:
            frame.iloc[0, 0] = -1  # the caller edits its table
            assert getattr(result, frame_name).iloc[0, 0] != -1, type(result).__name__
