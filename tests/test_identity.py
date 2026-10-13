"""Tests of ``_identity``: identity keys and display labels of groups and fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_tables import _mixed

from morphocoverage import validate_input_tables
from morphocoverage._identity import canonical_codes, display, identity_key
from morphocoverage.sampling_unit._units import resolve_unit


def test_identity_keys_tell_equal_text_apart_and_ignore_numpy_wrappers():
    assert identity_key(1) != identity_key("1") != identity_key(1.0)
    assert identity_key(np.int64(1)) == identity_key(1)
    with pytest.raises(ValueError, match="never missing"):
        identity_key(float("nan"))
    with pytest.raises(TypeError, match="hashable"):
        identity_key([1])


def test_display_labels_add_the_type_only_where_text_collides():
    shown = display([identity_key(1), identity_key("1"), identity_key("a")])
    assert shown[identity_key(1)] == "1 (int)" and shown[identity_key("1")] == "1 (str)"
    assert shown[identity_key("a")] == "a"


def test_blocks_need_not_be_orderable_and_are_numbered_by_identity():
    frame = _mixed([0, 1, 2, 3])["block"]
    codes, keys = canonical_codes(frame)
    shuffled, same_keys = canonical_codes(frame.iloc[[3, 1, 0, 2]].reset_index(drop=True))
    assert keys == same_keys and len(set(codes)) == 4
    assert list(shuffled) == [codes[3], codes[1], codes[0], codes[2]]


def test_a_technical_unit_is_not_claimed_because_two_repeats_print_alike():
    # Repeat 1 and repeat "1" are two sides, each with one well: no side holds two wells.
    fovs = pd.DataFrame(
        {
            "position_id": ["a", "b"],
            "condition": ["c", "c"],
            "biological_repeat": [1, "1"],
            "well": ["w1", "w2"],
        }
    )
    objects = fovs.assign(v=[1.0, 2.0])
    data = validate_input_tables(
        objects=objects,
        fovs=fovs,
        feature_cols=["v"],
        group_cols=["condition", "biological_repeat", "well"],
    )
    described = [
        {"repeat": 1, "condition": "c"},
        {"repeat": "1", "condition": "c"},
    ]
    unit, why = resolve_unit(data, described, "condition", "biological_repeat", None)
    assert unit is None and "one value per condition and repeat" in why


def test_a_composite_fov_key_is_written_out_whole_in_messages():
    from morphocoverage._identity import key_labels

    fovs = pd.DataFrame({"condition": ["a", "a"], "biological_repeat": ["R1", "R2"], "id": [1, 1]})
    assert key_labels(fovs, ["condition", "biological_repeat", "id"]) == [
        "condition=a, biological_repeat=R1, id=1",
        "condition=a, biological_repeat=R2, id=1",
    ]
