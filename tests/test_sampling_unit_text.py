"""The words of the sampling-unit checks: groups, skipped groups, short and full forms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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


def _balance_held_out():

    return _balance([("A", 1), ("B", 2), ("C", 0)]) + _balance(
        [("A", 1), ("B", 2), ("C", 4), ("D", 3)]
    )


def _balance(held_out):
    from morphocoverage.sampling_unit.cells_text import balance_lines

    rows = []
    for name, held in held_out:
        rows.append(
            {
                "condition": name,
                "control_condition": "ctrl",
                "unbalanced": held > 0,
                "spread": 1.1,
                "other_control_repeats": held,
                "control_shift": 2.5,
                "p_balanced": np.nan,
                "sparse_repeats": [],
                "repeats": 3,
                "balanced_repeats": 0,
            }
        )
    cells = pd.DataFrame(rows)
    contrast = SimpleNamespace(
        cells=cells, settings={"repeat_col": "biological_repeat", "control_only_repeats": ["R9"]}
    )
    return balance_lines(contrast)


EXPECTED_BALANCE_HELD_OUT = [
    (
        "Balance: for A and B, the control as provided is pooled from "
        "biological repeats the condition was not imaged in (1 and 2 of them); "
        "no balanced test could be drawn, so the test is given as provided "
        "only where the control comes from the condition's own biological "
        "repeats. The control means differ up to 2.5x between biological "
        "repeats, so a raw test pooling them would compare each condition with "
        "other repeats' level: read the normalised columns. The ctrl of "
        "biological repeat R9, which holds no other condition, is in no "
        "comparison."
    ),
    (
        "Balance: for every condition, the control as provided is pooled from "
        "biological repeats the condition was not imaged in (1 to 4 of them); no "
        "balanced test could be drawn, so the test is given as provided only "
        "where the control comes from the condition's own biological repeats. "
        "The control means differ up to 2.5x between biological repeats, so a "
        "raw test pooling them would compare each condition with other "
        "repeats' level: read the normalised columns. The ctrl of biological "
        "repeat R9, which holds no other condition, is in no comparison."
    ),
]


def test_balance_held_out():
    assert _balance_held_out() == EXPECTED_BALANCE_HELD_OUT


def test_every_summary_is_short_by_default_and_checks_its_detail(
    foundation_checks, controls, states
):
    for result in [*foundation_checks.values(), controls, *states]:
        assert result.summary() == result.summary(detail="short")
        assert result.summary(detail="full") != ""
        with pytest.raises(ValueError, match="short"):
            result.summary(detail="brief")


def test_groups_print_by_their_values_not_as_column_equals_value(
    foundation_checks, controls, states
):
    for result in [*foundation_checks.values(), controls, *states]:
        for detail in ("short", "full"):
            text = result.summary(detail=detail)
            assert "condition=" not in text and "repeat=" not in text, (result, detail)
    assert "MOCK · repeat r1" in foundation_checks["variability"].summary()
