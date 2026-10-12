"""Question 10's groups read against the conditions (owner, 2026-10-06, audit 10: A7, A8, C3,
C4, A13): population enrichment is exploratory (⚠️, "more common (exploratory)", never
"significant across repeats" or ✅); a group mostly of one condition is its shift, not a
subpopulation; a group is described against the control it is compared with, on values
normalised to each own control when the experiments have their own controls; the most
different condition is ranked by effect when no q exists; a trend needs 6 repeats and below
3 nothing is tested; a rule on a feature with few values is said; the page suggests the rare
target only for a rare group that is not one condition."""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit.populations import Populations
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._html_next import populations_next
from morphocoverage.workflows._html_render import chip
from morphocoverage.workflows._population_context import (
    EXPERIMENTS_DIFFER,
    NOT_NORMALISED,
    compared_values,
    few_valued,
)
from morphocoverage.workflows._populations import (
    PopulationsResult,
    _enriched,
    _group_parts,
    with_populations,
)

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _run(name: str):
    sys.path.insert(0, str(SCENARIOS))
    try:
        import make_scenarios
    finally:
        sys.path.remove(str(SCENARIOS))
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), feature_tables=loaded.feature_tables
        )
    return loaded, result


@pytest.fixture(scope="module")
def e16():
    """Two experiments, each with its own control, 2-fold apart in intensity; a1 more than
    doubles the area against cA."""
    return _run("e16_acquisition_block")


@pytest.fixture(scope="module")
def a07():
    """One control; condition a shifted 0.8 on the log scale, no subpopulation planted."""
    return _run("a07_tagged_one_norm_cells_tech_tol")


def _section(result) -> str:
    text = result.document.sections["populations"]
    return _flat(text)


def test_a_group_of_one_condition_is_its_shift_and_exploratory(a07):
    loaded, result = a07
    status = result.document.statuses["populations"]
    headline = result.document.headlines["populations"]
    assert status.marker == "⚠️"
    assert "mostly a, 100% of its cells: its shift, not a subpopulation" in headline
    text = _section(result)
    assert "significant across repeats" not in text
    assert "more common in a (exploratory; 94% vs 0%, q" in text
    assert chip(status, "populations", headline).endswith("A condition's shift</span>")
    # The page does not suggest a whole condition as the rare phenotype; it names a few of
    # its fields with their counts.
    page = populations_next(result, loaded.fovs, loaded.objects)
    assert "not a rare phenotype to declare" in page and "for example" in page
    assert "declare it as the rare phenotype" not in page


def test_groups_are_found_on_normalised_values_and_described_against_their_control(e16):
    _, result = e16
    text = _section(result)
    assert EXPERIMENTS_DIFFER in text
    populations = result.results["populations"]
    assert populations.normalised and populations.against
    table = populations.populations.table.set_index("population")
    group = table.loc["G1"]
    # a1's cells against cA's: higher, as question 9 reads them (before: "low area" against
    # every cell of two experiments pooled).
    assert all(d > 0 for _, d in group["defining"])
    assert "mostly a1" in result.document.headlines["populations"]
    assert "Defined by: each group's cells against those of the control they are compared" in text


def _negative_control(objects: pd.DataFrame) -> pd.DataFrame:
    """e16 with cB's intensity negated in its first repeat: a control mean below zero."""
    changed = objects.copy()
    first = changed["biological_repeat"].iloc[0]
    rows = (changed["condition"] == "cB") & (changed["biological_repeat"] == first)
    changed.loc[rows, "intensity_mean"] = -changed.loc[rows, "intensity_mean"]
    return changed


def test_values_that_cannot_be_normalised_are_said_not_swallowed(e16):
    # Audit 12 (B10): a control mean that is not positive leaves the values as given, and
    # the report says so; before, any ValueError was caught and nothing was said.
    loaded, result = e16
    setup = Setup(**loaded.setup)
    objects = _negative_control(loaded.objects)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values, read, found = compared_values(objects, loaded.fovs, setup, objects)
    why = found["not_normalised"]
    assert values is objects and read is setup and not found["normalised"]
    assert why is not None and why.startswith("for the control cB, in biological_repeat=")
    assert "'intensity_mean' is negative" in why
    found = replace(result.results["populations"], not_normalised=why)
    document = with_populations(result.document, found, setup)
    assert f"{NOT_NORMALISED}: {_flat(why)}." in _section(replace(result, document=document))


def test_another_problem_of_the_tables_is_not_swallowed(e16):
    # Only the documented case falls back: a table without a declared feature still raises.
    loaded, _ = e16
    raw = loaded.objects.drop(columns="length")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="length"):
            compared_values(loaded.objects, loaded.fovs, Setup(**loaded.setup), raw)


def _result(tests: list[dict]) -> PopulationsResult:
    table = pd.DataFrame(
        [
            {
                "population": "G1",
                "kind": "group",
                "observations": 40,
                "share": 0.04,
                "defining": [("area", 1.2)],
                "defined_by": "high area (d +1.2)",
                "rule": "",
            }
        ]
    )
    populations = Populations(pd.Series(dtype=object), table, {"observation_unit": "cell"})
    columns = ["feature", "condition", "repeats", "higher", "lower", "d", "p_paired"]
    enrichment = pd.DataFrame(tests, columns=[*columns, "share", "control_share", "q_paired"])
    return PopulationsResult(populations, enrichment, None, "ctrl")


def _test(condition: str, repeats: int, higher: int, d: float, q: float = np.nan) -> dict:
    return {
        "feature": "G1",
        "condition": condition,
        "repeats": repeats,
        "higher": higher,
        "lower": repeats - higher,
        "d": d,
        "p_paired": q,
        "share": 0.1,
        "control_share": 0.02,
        "q_paired": q,
    }


def test_the_most_different_condition_is_ranked_by_effect_without_a_q():
    # Two repeats: no q; the condition of the largest effect leads, not the first row.
    result = _result([_test("LATS1", 2, 2, 0.1), _test("YAP", 2, 2, 4.0)])
    assert _enriched(result, "G1").startswith("YAP: higher in 2 of 2")
    assert _enriched(result, "G1").endswith("not tested")
    assert (
        _group_parts(result, "G1", "cells")[-1]
        == "YAP higher in both repeats (10% vs 2.0%), not tested"
    )


def test_a_trend_needs_six_repeats():
    three = _result([_test("drug", 3, 3, 1.0, q=0.2)])
    assert _group_parts(three, "G1", "cells") == []
    six = _result([_test("drug", 6, 6, 1.0, q=0.2)])
    assert _group_parts(six, "G1", "cells") == [
        "drug higher in every repeat, not significant (10% vs 2.0%, q 0.20)"
    ]


def test_a_rule_on_a_feature_with_few_values_is_said():
    marked = pd.DataFrame({"count": [0.0] * 95 + [1.0] * 5, "area": np.linspace(1, 2, 100)})
    row = {"rule_terms": (("count", ">", 0.5), ("area", "<=", 1.5))}
    assert few_valued(row, marked) == ["count"]
