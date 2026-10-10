"""Issue 284: the text layer counts and merges groups by identity, not by rendered label."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from morphocoverage.sampling_unit._group_text import count_groups, distinct_labels, namer
from morphocoverage.sampling_unit._unit_text import _by_reason, _spread_cautions
from morphocoverage.sampling_unit.cells_text import _sparse

KEYS = ["condition", "biological_repeat"]
ROWS = [
    {"condition": "a", "biological_repeat": 1, "status": "too few"},
    {"condition": "a", "biological_repeat": "1", "status": "too few"},
]


def test_two_groups_that_print_alike_are_two_groups():
    assert count_groups(ROWS, KEYS) == 2
    labels = distinct_labels(ROWS, KEYS)
    assert len(labels) == 2 and len(set(labels)) == 2
    assert labels[0].endswith("(str, int)") and labels[1].endswith("(str, str)")


def test_ordinary_labels_are_written_plainly():
    rows = [
        {"condition": "a", "biological_repeat": "R1"},
        {"condition": "b", "biological_repeat": "R1"},
    ]
    assert distinct_labels(rows, KEYS) == ["a · repeat R1", "b · repeat R1"]
    assert namer(rows, KEYS)(rows[1]) == "b · repeat R1"


def test_one_group_listed_twice_is_one_group():
    rows = [ROWS[0], dict(ROWS[0])]
    assert count_groups(rows, KEYS) == 1
    assert distinct_labels(rows, KEYS) == ["a · repeat 1"]


def test_skipped_for_one_reason_names_both_alike_groups():
    (line,) = _by_reason(ROWS, KEYS, "not estimated")
    assert "(str, int)" in line and "(str, str)" in line


def test_skipped_groups_of_the_spread_check_stay_distinct():
    skipped = [
        {
            "label": "a · repeat 1",
            "group": "g",
            "observations": 3,
            "identity": [["a", "str"], ["1", "int"]],
        },
        {
            "label": "a · repeat 1",
            "group": "g",
            "observations": 4,
            "identity": [["a", "str"], ["1", "str"]],
        },
    ]
    result = SimpleNamespace(
        at_threshold=pd.DataFrame({"ratio": [1.0]}),
        settings={"threshold": 10, "skipped_groups": skipped},
    )
    cautions = _spread_cautions(result, "cells")
    assert len(cautions) == 2
    assert "(str, int)" in cautions[0] and "(str, str)" in cautions[1]


def test_sparse_repeats_that_print_alike_are_not_merged():
    cells = pd.DataFrame(
        {
            "condition": ["x", "x"],
            "sparse_repeats": [[1], ["1"]],
            "repeats": [3, 3],
            "balanced_repeats": [2, 2],
            "other_control_repeats": [0, 0],
        }
    )
    contrast = SimpleNamespace(
        settings={"repeat_col": "biological_repeat", "observation_unit": "cell"}
    )
    lines = _sparse(contrast, cells)
    assert len(lines) == 2
