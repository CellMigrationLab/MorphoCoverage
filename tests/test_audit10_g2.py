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
import pandas as pd
import pytest

from morphocoverage import (
    Component,
    Decision,
    Effort,
    EffortPolicy,
    Evidence,
    SamplingObjective,
)
from morphocoverage._control import ACROSS, MASTER_RAW
from morphocoverage.plots_pairwise import PANEL_KEYS, verdict_note
from morphocoverage.policies import decide
from morphocoverage.sampling_unit import object_rule_check
from morphocoverage.sampling_unit.contrast_across import across_rows
from morphocoverage.sampling_unit.contrast_cells import repeats_to_balance, trusted_test
from morphocoverage.sampling_unit.contrast_sampling import (
    MIN_FIELDS,
    noncentral_d_interval,
    sampling_row,
)
from morphocoverage.statistics import cohen_d
from morphocoverage.sufficiency_short import met_count
from morphocoverage.workflows import Setup, plots, report
from morphocoverage.workflows._whole_table import across_reason

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


def test_a_control_against_the_master_is_tested_on_the_scale_of_its_fold_change() -> None:
    # B17: the vs_master_raw row printed a fold change (x0.88) but tested the differences.
    rows = pd.DataFrame(
        [
            {
                "kind": MASTER_RAW,
                "label": "cB vs cA (raw)",
                "condition": "cB",
                "reference": "cA",
                "condition_control": None,
                "reference_control": None,
                "family": MASTER_RAW,
            }
        ]
    )
    means = {("cA", r): m for r, m in zip("123", (10.0, 100.0, 1000.0), strict=True)}
    means |= {("cB", r): m for r, m in zip("123", (8.0, 90.0, 880.0), strict=True)}
    [row] = across_rows(means, rows, list("123"), 0).to_dict("records")
    expected = np.log([0.8, 0.9, 0.88])
    assert row["values"] == "normalised"
    assert row["effects"] == pytest.approx(expected)
    assert row["fold_change"] == pytest.approx(np.exp(expected.mean()))
    # A signed feature keeps the difference, and says so.
    signed = {key: value - 50 for key, value in means.items()}
    [raw] = across_rows(signed, rows, list("123"), 0).to_dict("records")
    assert raw["values"] == "raw" and np.isnan(raw["fold_change"])
    assert raw["effects"] == pytest.approx([-2.0, -10.0, -120.0])


def test_each_kind_of_row_across_experiments_gives_its_own_reason_for_no_whole_table_test():
    # B16: "a difference of log ratios has no meaning" was said of the master rows too, which
    # compare raw centres.
    loaded = make_scenarios.load("a23_embed2_several_image_tech_fp_dino")
    setup = Setup(**{**loaded.setup, "control_parents": None})
    assert set(setup.contrast_table["kind"]) >= {ACROSS, MASTER_RAW}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(loaded.objects, loaded.fovs, setup, feature_tables=loaded.feature_tables)
    full = " ".join(result.summary("full").split())
    assert "a difference of log ratios has no meaning in a table such as an embedding" in full
    assert "not the controls with the master" in full
    assert across_reason([MASTER_RAW], features=True) == (
        "The rows across experiments of the contrast table are tested feature by feature "
        "only: the whole-table test compares each condition with its own control, not the "
        "controls with the master."
    )
    assert across_reason(["own_control"], features=True) == ""


def test_the_exact_interval_of_d_covers_at_its_level() -> None:
    # B1: every new interval with a coverage test (AGENTS.md).
    rng = np.random.default_rng(10)
    covered, reps, true = 0, 1500, 0.8
    for _ in range(reps):
        first, second = rng.normal(true, 1, 6), rng.normal(0, 1, 5)
        low, high = noncentral_d_interval(cohen_d(first, second), 6, 5)
        covered += low <= true <= high
    assert 0.93 <= covered / reps <= 0.97
    assert all(np.isnan(noncentral_d_interval(0.5, 1, 6)))


def _fields(values: np.ndarray) -> list[list[np.ndarray]]:
    """One side, no wells: one field per value."""
    return [[np.array([v]) for v in values]]


def test_shown_within_a_repeat_holds_its_false_alarm_rate_and_needs_five_fields() -> None:
    # B1, A16: the percentile bootstrap called a null difference "shown" in 22% of 3-vs-3
    # comparisons and 11-15% from 5 vs 4; one image per field and no wells now gets the
    # exact interval, and fewer than 5 fields a side is not assessable.
    rng = np.random.default_rng(11)
    shown = 0
    for _ in range(400):
        t, c = rng.normal(0, 1, 5), rng.normal(0, 1, 5)
        row = sampling_row(_fields(t), _fields(c), cohen_d(t, c), rng, n_boot=50)
        assert row["interval"] == "noncentral t" and row["assessable"]
        shown += row["resolved"]
    assert shown / 400 <= 0.08
    few = sampling_row(_fields([5.0, 6, 7]), _fields([0.0, 1, 2]), 5.0, rng, n_boot=50)
    assert not few["assessable"] and not few["resolved"]
    assert np.isfinite(few["d_low"])  # kept for the plan of the next experiment
    # Wells or several observations per field: the bootstrap, as before.
    wells = [[np.array([1.0, 2.0]), np.array([1.5, 2.5])]] * 3
    row = sampling_row(wells, wells, 0.1, rng, n_boot=50)
    assert row["interval"] == "bootstrap"


def test_question_2_says_not_assessable_below_five_fields_a_side() -> None:
    # e01: three fields a side in each repeat.
    text = " ".join(_scenario_report("e01_two_repeats").summary("full").split())
    assert f"not assessable: fewer than {MIN_FIELDS} fields a side" in text
    assert "yes (shown)" not in text
    assert "not assessed: too few fields" not in text


def test_the_count_rule_with_whole_fields_says_its_draws_repeat_fields() -> None:
    # A16: "fields taken" 4-18 of 3 acquired fields, the draws with replacement unsaid.
    rng = np.random.default_rng(12)
    rows = []
    for well, fields in (("w1", 3), ("w2", 8)):
        for field in range(fields):
            for label in range(20):
                rows.append(
                    {
                        "condition": "ctrl",
                        "biological_repeat": "r1",
                        "technical_repeat": well,
                        "image_file": f"{well}_{field}.tif",
                        "label_id": label,
                        "x": 10 + rng.normal(0, 1 + field % 3),
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "technical_repeat", "image_file"]]
    result = object_rule_check(
        objects, fovs.drop_duplicates(), feature="x", observation_unit="cell", n_draws=50
    )
    status = dict(zip(result.table["technical_repeat"], result.table["status"], strict=False))
    assert status["w1"] == f"fewer than {MIN_FIELDS} fields with cells"
    assert status["w2"] == "ok"
    short = " ".join(result.summary().split())
    full = " ".join(result.summary("full").split())
    assert "drawn at random with replacement" in short
    assert "with replacement, until that many, so it can exceed the fields acquired" in full
