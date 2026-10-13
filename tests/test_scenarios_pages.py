"""The page of every scenario of ``data/scenarios`` that runs, beside the matrix's subset.

``tests/test_scenarios_matrix.py`` checks every scenario's truth and builds the page for the
covering subset its truth names (``html``). This module builds the page for every other
scenario that is not a refusal, so a design that only one scenario holds (a control in other
repeats than its condition, one condition, a screen) cannot crash ``html_report`` unseen, and
checks the ``table`` column of its CSVs as the matrix does. It is slow (a page each), so it
is marked ``slow``: it runs in the whole suite, and a focused run can leave it out with
``-m "not slow"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _helpers_scenarios import NAMES, _page, _report

from morphocoverage.workflows import Setup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios


def _paged_here(name: str) -> bool:
    """Whether the scenario runs and the matrix does not already build its page."""
    truth = make_scenarios.load(name).truth
    return truth["outcome"] != "refused" and not truth.get("html")


PAGED = [name for name in NAMES if _paged_here(name)]


@pytest.mark.slow
@pytest.mark.parametrize("name", PAGED)
def test_the_page_of_every_scenario_that_runs_is_built(name: str, tmp_path: Path) -> None:
    loaded = make_scenarios.load(name)
    tagged = any(entry["tag"] == "explainable" for entry in loaded.spec["feature_tables"].values())
    truth = {"table_column": tagged, **loaded.truth}
    result = _report(loaded, Setup(**loaded.setup))
    _page(result, loaded, truth, tmp_path)
