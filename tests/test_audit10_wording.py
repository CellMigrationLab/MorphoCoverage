"""Audit 10 (owner, 2026-10-06): the report's words say what was done, once, and why.

Each test pins one finding of the audit (``A``, ``B`` or ``C`` and its number) on a small
constructed case; the scenarios (``tests/test_scenarios_matrix.py``) guard the same words
on every scenario that runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from morphocoverage.order_check import OrderCheck
from morphocoverage.sampling_unit._units import NOT_INDEPENDENT
from morphocoverage.sampling_unit.contrast import UNREACHABLE
from morphocoverage.sampling_unit.contrast_text import _no_within_test
from morphocoverage.workflows._enough_features import _needed

ONE_WELL = "only one technical repeat (well) with observations on one side, no replication"


def _contrast(statuses: list[str], technical: list[int], control: list[int]) -> SimpleNamespace:
    table = pd.DataFrame({"status": statuses, "technical": technical, "control_technical": control})
    return SimpleNamespace(table=table)


def test_no_test_within_a_repeat_gives_the_status_reason_never_one_well_each_a6_c5_b18():
    # idr0028: the control has one well per repeat, the treatments two.
    said = _no_within_test(_contrast([ONE_WELL] * 2, [2, 2], [1, 1]))
    assert said.startswith("the control has one well in each repeat")
    both = _no_within_test(_contrast([ONE_WELL] * 2, [1, 1], [1, 1]))
    assert both.startswith("each condition and its control have one well in each repeat")
    # Two wells a side: the test cannot reach 0.05; no well column: not recorded.
    assert "too few wells a side" in _no_within_test(_contrast([UNREACHABLE], [2], [2]))
    unknown = _no_within_test(_contrast([NOT_INDEPENDENT], [0], [0]))
    assert unknown == "no well recorded, so treated as one well per condition and repeat"
    for text in (said, unknown):
        assert "each condition has one well" not in text


def test_an_order_check_that_flags_nothing_names_its_threshold_a21():
    check = OrderCheck("area", 20, "order", ("x", "y"), 0.4, 0.037, 0.1, 0.84)
    said = check.caution()
    assert "no trend with acquisition order or spatial autocorrelation flagged at the " in said
    assert "Bonferroni threshold of 0.025 (Spearman p = 0.037; Moran's I permutation p" in said
    assert "no spatial autocorrelation (" not in said


def test_one_comparison_needs_more_wells_a19():
    row = {"needing": 0, "too_small": 2, "between_wells": 1}
    assert _needed(row, "fields") == "2 too small to show; 1 needs more wells or biological repeats"
    assert _needed({**row, "between_wells": 2}, "fields").endswith(
        "2 need more wells or biological repeats"
    )
