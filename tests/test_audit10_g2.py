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

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from morphocoverage import (
    Component,
    Decision,
    Effort,
    EffortPolicy,
    Evidence,
    SamplingObjective,
)
from morphocoverage.plots_pairwise import PANEL_KEYS, verdict_note
from morphocoverage.policies import decide
from morphocoverage.sampling_unit.contrast_cells import repeats_to_balance, trusted_test
from morphocoverage.sufficiency_short import met_count
from morphocoverage.workflows import Setup, plots, report

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


@pytest.mark.parametrize(
    "name", ["a10_embed2_several_norm_cells_none_tol", "a21_embed1_one_norm_cells_none_tol"]
)
def test_a_setup_with_no_target_has_no_component_and_nothing_decides_a_stop(name: str) -> None:
    # C12: "a stop needs every declared target (phenotypic fingerprint)" with none declared.
    setup = Setup(**make_scenarios.load(name).setup)
    objective = setup.objective()
    assert objective.objective is SamplingObjective.NO_TARGET
    assert objective.required_components == frozenset()
    said = [line for line in setup.declaration() if line.startswith("Live decision")]
    assert said == ["Live decision: no target: nothing decides a stop."]
    for detail in ("short", "full"):
        text = " ".join(_scenario_report(name).summary(detail).split())
        assert "a stop needs every declared target" not in text
    assert "Live decision: no target: nothing decides a stop." in text
    # Whatever a rule says, no target means no stop.
    stop = EffortPolicy("n_fovs", 1, Component.DISTRIBUTION)
    overall, per_component = decide(Evidence(), Effort(n_fovs=10, n_objects=100), [stop], objective)
    assert per_component == {"distribution": Decision.STOP}
    assert overall is Decision.UNSUPPORTED


def test_panel_c_says_its_q_is_the_figures_and_gives_the_verdicts_q() -> None:
    # A3: panel C's q, adjusted over the figure's pairs, was green where the verdict's q,
    # adjusted over the feature family, was not significant (and the other way round).
    name = "a15_expl_one_cells_tech_rare"
    loaded, result = make_scenarios.load(name), _scenario_report(name)
    discovery = result.discovery
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = plots(loaded.objects, loaded.fovs, result.setup, ranking=discovery)
    tests = discovery.tests
    checked = 0
    for feature, parts in drawn.items():
        for figure in parts:
            mine = tests[tests["feature"].eq(feature) & tests["q_paired"].notna()]
            expected = dict(zip(mine["condition"], mine["q_paired"], strict=True))
            assert figure.pairwise_verdicts == pytest.approx(expected)
            legend = " ".join(figure.legend_artist.get_text().split())
            assert "adjusted over the pairs of this figure only" in legend
            if expected:
                checked += 1
                assert verdict_note(expected) in legend
                assert "Verdict q (the feature's family): " in legend
    assert checked
    assert PANEL_KEYS["C"].endswith("adjusted over this figure only")


def test_a_balanced_test_left_with_one_repeat_gives_way_to_the_test_as_provided() -> None:
    # B5: the image-level "balanced" test kept R3 alone (the other repeats had fewer than 5
    # control images) and the reading quoted it as the condition's image-level q.
    rng = np.random.default_rng(3)
    by_t = {"R1": rng.normal(1, 1, 3), "R2": rng.normal(1, 1, 4), "R3": rng.normal(1, 1, 12)}
    by_c = {"R1": rng.normal(0, 1, 3), "R2": rng.normal(0, 1, 4), "R3": rng.normal(0, 1, 12)}
    assert repeats_to_balance(by_t, {"R1", "R2"}, unbalanced=True) == []
    assert repeats_to_balance(by_t, {"R1"}, unbalanced=True) == ["R2", "R3"]
    test = trusted_test(by_t, by_c, by_c, 199, np.random.default_rng(0))
    assert not test["balanced"]
    assert test["observations"] == 19 and test["control_observations"] == 19
    assert np.isfinite(test["p"])
