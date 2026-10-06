"""Audit 10 (owner, 2026-10-06): the report's words say what was done, once, and why.

Each test pins one finding of the audit (``A``, ``B`` or ``C`` and its number) on a small
constructed case; the scenarios (``tests/test_scenarios_matrix.py``) guard the same words
on every scenario that runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from morphocoverage.sampling_unit._units import NOT_INDEPENDENT
from morphocoverage.sampling_unit.contrast import UNREACHABLE
from morphocoverage.sampling_unit.contrast_text import _no_within_test

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
