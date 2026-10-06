"""Audit 10, group G2: the tests and their families (owner, 2026-10-06).

Each test pins one decision of the audit's report findings on the scenario of
``data/scenarios`` that shows it, or on a small constructed table: what the report says now,
and that it never says the old, misleading thing.
"""

from __future__ import annotations

import sys
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from morphocoverage.sufficiency_short import met_count
from morphocoverage.workflows import Setup, report

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"
sys.path.insert(0, str(SCENARIOS))
try:
    import make_scenarios
finally:
    sys.path.remove(str(SCENARIOS))


@cache
def _scenario_report(name: str) -> Any:
    """``report`` on a scenario's tables, as a user runs it, its warnings kept quiet."""
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), feature_tables=loaded.feature_tables
        )


def test_the_rule_counts_say_how_many_groups_were_judged() -> None:
    # B19: "0 of 40 (1 not defined)" hid 12 groups with too few fields to judge.
    tally = {"met": 0, "groups": 40, "undefined": 1, "waiting": 12}
    assert met_count(tally) == "0 of 27 judged; 12 too few fields; 1 undefined"
    assert met_count({"met": 3, "groups": 4, "undefined": 0, "waiting": 0}) == "3 of 4"
    assert met_count({"met": 0, "groups": 4, "undefined": 0, "waiting": 4}, "group") == (
        "none judged; 4 too few fields"
    )
    # e01: three fields per group, too few for the per-field bootstrap bound in every group.
    text = _scenario_report("e01_two_repeats").summary("full")
    row = next(line for line in text.splitlines() if "mean-precision rule, per field, 5%  " in line)
    assert "none judged; 4 too few fields" in row
    assert " 0 of 4 " not in row
