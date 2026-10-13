"""Tests of ``sampling_unit.contrast_sampling``: fields drawn within a repeat, the d interval."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _helpers_workflows import _scenario_report

from morphocoverage.sampling_unit.contrast_sampling import (
    MIN_FIELDS,
    noncentral_d_interval,
    sampling_row,
)
from morphocoverage.statistics import cohen_d

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"


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


def test_the_moderated_paired_t_borrows_the_familys_variance() -> None:
    # A10: the P1 candidate (limma-style empirical Bayes) for 3 near-identical repeats.
    sys.path.insert(0, str(SCENARIOS.parent / "simulations"))
    try:
        from decisions.candidates import repeat_means
    finally:
        sys.path.remove(str(SCENARIOS.parent / "simulations"))
    rng = np.random.default_rng(15)
    family = {f"f{i}": rng.normal(0, 0.18, 3) for i in range(30)}
    family["steady"] = np.array([0.180, 0.182, 0.184])
    found = repeat_means.moderated_t(family)
    assert repeat_means.t_summary(family["steady"])["p"] < 1e-3
    assert found["steady"]["p"] > 0.01
    assert found["steady"]["prior_df"] > 0
    # Calibrated under the null at 3 repeats: about 5% of the features at p <= 0.05.
    rejected = [
        result["p"] <= 0.05
        for _ in range(200)
        for result in repeat_means.moderated_t(
            {f"f{i}": rng.normal(0, rng.choice([0.1, 0.2]), 3) for i in range(20)}
        ).values()
    ]
    assert 0.03 <= np.mean(rejected) <= 0.07
