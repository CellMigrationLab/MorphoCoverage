"""Tests of ``_frozen``: read-only containers that serialise, pickle and copy like plain ones."""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from morphocoverage._frozen import FrozenDict, FrozenList, freeze, thaw


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
