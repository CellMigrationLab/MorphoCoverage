"""Tests of ``retrospective.report_status``: the status marker of each question."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from _helpers_retrospective import (
    INSPECTION_NOVELTY,
    _acquisition,
    _adequacy,
    _effort,
    _inspect,
    _mean_curve,
    _plate_protocol,
    _well_values,
)
from _helpers_sampling_unit import _control, _flat_text, _orders, _plate
from _helpers_strategies import _flat

from morphocoverage._style import CAUTION, Status
from morphocoverage.retrospective import (
    AdequacyResult,
    report_status,
    report_status_conditions,
    report_status_plan,
)
from morphocoverage.retrospective.report_status import (
    bought_status,
    controls_status,
    effort_status,
    inspect_status,
    next_time_status,
    order_status,
    unit_status,
)


def _rules(*groups):
    """An object-rule check: one row per (well, status, meets_target)."""
    table = pd.DataFrame(
        [{"well": w, "target": 0.05, "status": s, "meets_target": m} for w, s, m in groups]
    )
    return SimpleNamespace(table=table, group_cols=("well",), settings={})


def _spread(matched, skipped=0):
    """A descriptor-spread check: whole fields settle (``True``) or never do, per well."""
    rows = pd.DataFrame(
        [{"well": f"B{i}", "objects_to_match": 50 if ok else None} for i, ok in enumerate(matched)],
        columns=["well", "objects_to_match"],
    )
    settings = {
        "threshold": 100,
        "skipped_groups": [{"group": f"well=C{i}", "label": f"C{i}"} for i in range(skipped)],
    }
    return SimpleNamespace(at_threshold=rows, group_cols=("well",), settings=settings)


def test_the_foundation_headline_counts_groups_that_could_not_be_checked_apart():
    few = "fewer than 2 fields with cells"
    held = _rules(("A1", "ok", True), ("A2", "ok", True))
    assert unit_status(None, held, None) == (
        Status.OK,
        "the mean-precision count rule still held with whole fields in both groups",
    )
    held = _rules(("A1", "ok", True), ("A2", "ok", True), ("A3", few, None))
    status, text = unit_status(None, held, None)
    assert "both groups" not in text
    assert text == (
        "the mean-precision count rule still held with whole fields in the 2 groups assessed; "
        f"1 more not assessed ({few})"
    )
    missed = _rules(("A1", "ok", False), ("A2", "ok", True), ("A3", few, None))
    assert unit_status(None, missed, None)[1] == (
        "the mean-precision count rule missed with whole fields in 1 of the 2 groups assessed; "
        f"1 more not assessed ({few})"
    )
    assert unit_status(None, None, _spread([True, True])) == (Status.OK, "assessed")
    status, text = unit_status(None, None, _spread([False, True], skipped=1))
    assert status is Status.CAUTION
    assert text == (
        "descriptors from whole fields never settled in 1 of the 2 groups assessed; 1 more not "
        "assessed (fewer than 100 cells)"
    )
    assert unit_status(None, None, _spread([True], skipped=1))[1] == (
        "descriptor spread not assessed in 1 of 2 groups (fewer than 100 cells)"
    )
    # Every group skipped: never merely "assessed".
    assert unit_status(None, None, _spread([], skipped=3))[1] == (
        "descriptor spread not assessed in all 3 groups (fewer than 100 cells)"
    )


def test_next_time_status_without_answers_and_with_a_target_not_judged(monkeypatch):
    protocol = SimpleNamespace(settings={})
    monkeypatch.setattr(report_status_plan, "protocol_answers", lambda protocol: [])
    assert report_status_plan.next_time_status(None, protocol) == (
        report_status.Status.NOT_ASSESSED,
        "not assessed",
    )
    answers = [
        {"label": "the mean", "served": None, "unreached": True, "never": True, "units": 5},
        {"label": "the spread", "served": 3, "units": 5},
    ]
    monkeypatch.setattr(report_status_plan, "protocol_answers", lambda protocol: answers)
    assert report_status_plan.next_time_status(None, protocol) == (
        report_status.Status.CAUTION,
        "3 fields served every sample for 1 of 2 targets; 1 of 2 targets not judged",
    )


def test_enough_status_with_late_stops_and_nothing_new_not_shown(monkeypatch):
    tallies = [
        {"label": "the mean", "data_rule": True, "groups": 2, "met": 2}
        | {"warned": False, "flagged": False, "lost": False},
        {"label": "nothing new", "data_rule": True, "groups": 2, "met": 0}
        | {"warned": False, "flagged": False, "lost": False},
    ]
    monkeypatch.setattr(report_status, "rule_tallies", lambda sufficiency: tallies)
    monkeypatch.setattr(report_status, "has_resolution", lambda contrast: False)
    monkeypatch.setattr(report_status, "_premature", lambda calibration: 0.5)
    monkeypatch.setattr(report_status, "group_word", lambda result: "group")
    assert report_status.enough_status(object()) == (
        report_status.Status.CAUTION,
        (
            "1 of 2 rules said enough in every group; 'nothing new' not yet shown; held-back "
            "fields: some rules stopped too early (up to 50%)"
        ),
    )


def test_conditions_status_from_contrasts_alone_and_from_a_ranking_alone(monkeypatch):
    module, Status = report_status_conditions, report_status.Status
    monkeypatch.setattr(module, "readings_headline", lambda contrast: "")
    monkeypatch.setattr(module, "condition_readings", lambda c: pd.DataFrame({"verdict": []}))
    monkeypatch.setattr(module, "across_tested", lambda contrast: False)
    monkeypatch.setattr(module, "reagents_headline", lambda contrast: "")
    monkeypatch.setattr(module, "secondary_headline", lambda contrast: "")
    monkeypatch.setattr(module, "opposite_signs", lambda *arguments: True)
    monkeypatch.setattr(module, "across_headline", lambda across, col: f"across ({col})")
    table = pd.DataFrame(
        {
            "status": ["ok", "ok"],
            "d_fields": [0.5, -1.2],
            "p_bonferroni": [0.01, 0.2],
            "p_value": [0.01, 0.2],
            "condition": ["a", "b"],
        }
    )
    settings = {"repeat_col": "biological_repeat"}
    contrast = SimpleNamespace(table=table, repeats=None, settings=settings)
    assert module.conditions_status(contrast, None) == (
        Status.CAUTION,
        (
            "2 contrasts, |d| on field means 0.5 to 1.2; 1 of 2 tested differ (Bonferroni "
            "p ≤ 0.05); opposite signs between repeats"
        ),
    )
    contrast = SimpleNamespace(table=table, repeats=pd.DataFrame({"x": [1]}), settings=settings)
    assert module.conditions_status(contrast, None) == (
        Status.CAUTION,
        (
            "across (biological_repeat); within repeats: 1 of 2 tested differ (Bonferroni p ≤ "
            "0.05); opposite signs between repeats"
        ),
    )
    untested = SimpleNamespace(ranking=pd.DataFrame({"score": [1.0, None], "tests": [0, 0]}))
    assert module.conditions_status(None, untested) == (
        Status.CAUTION,
        "2 features not ranked (no contrast could be tested)",
    )
    tested = SimpleNamespace(ranking=pd.DataFrame({"score": [1.0, None], "tests": [1, 0]}))
    assert module.conditions_status(None, tested) == (Status.OK, "1 of 2 features pass the gate")
    empty = SimpleNamespace(ranking=pd.DataFrame())
    assert module.conditions_status(None, empty) == (Status.NOT_ASSESSED, "not assessed")


def test_question_6_is_marked_by_what_was_listed_and_what_could_be_judged():
    objects, fovs = _acquisition()
    listed = _inspect(objects, fovs)
    status, headline = inspect_status(listed)
    assert status is Status.CAUTION
    assert headline == "3 fields to reopen in 1 group (new state in 1, influence in 1, " + (
        "unusual count in 1)"
    )
    quiet = _inspect(objects, fovs, novelty=None, summaries=[], count_ratio=10)
    assert inspect_status(quiet) == (Status.OK, "no field stands out")
    nothing = _inspect(objects, fovs, novelty=None, summaries=[], count_ratio=None)
    assert inspect_status(nothing)[0] is Status.NOT_ASSESSED
    fovs.loc[3, "stage_x"] = np.nan
    partly = _inspect(
        objects, fovs, novelty=None, summaries=[], count_ratio=10, stage_cols=("stage_x", "stage_y")
    )
    assert inspect_status(partly)[0] is Status.CAUTION
    assert inspect_status(None)[0] is Status.NOT_ASSESSED
    assert INSPECTION_NOVELTY is not None


def test_question_8_says_whether_another_order_needed_fewer_fields():
    result = _orders()
    status, headline = order_status(result)
    table = result.table.set_index("order")
    recorded = table.loc["recorded", "median_fovs"]
    others = table.drop("recorded")["median_fovs"].dropna()
    if recorded - others.min() > 1:
        assert headline.startswith("another order needed fewer fields in the group")
        assert status is Status.CAUTION
    else:
        assert headline.startswith("no order beat the recorded one")


def test_the_foundation_status_follows_the_rule_with_whole_fields(foundation_checks):
    status, headline = unit_status(
        foundation_checks["variability"], foundation_checks["object_rules"], None
    )
    assert status is Status.CAUTION
    assert "the mean-precision count rule missed with whole fields in" in headline
    held = dataclasses.replace(
        foundation_checks["object_rules"],
        table=foundation_checks["object_rules"].table.assign(meets_target=True),
    )
    status, headline = unit_status(foundation_checks["variability"], held, None)
    assert status is Status.OK
    assert "still held with whole fields in both groups" in headline
    assert "a mean from whole fields varies" in headline


def test_a_stratum_not_judged_is_named_by_its_column_and_value():
    objects, fovs = _plate()
    keep = ~(fovs["biological_repeat"].eq("P2") & fovs["condition"].eq("KD"))
    fovs = fovs[keep]
    objects = objects[objects["image_file"].isin(fovs["image_file"])]
    result = _control(objects, fovs)
    status, headline = controls_status(result, minimum_detection=0.01)
    assert status is Status.CAUTION
    assert (
        "not judged in biological repeat P2 (the positive control was not imaged here)" in headline
    )
    assert "Not judged after 6 reference fields in biological repeat P2" in _flat_text(
        result.summary()
    )


def test_next_time_markers_follow_their_rules():
    many = ("P1", "P2", "P3", "P4", "P5")
    served = _plate_protocol(_well_values(many, 0.95, 0.99))
    assert next_time_status(None, served)[0] is Status.OK
    assert "5 fields served every sample for the target" in next_time_status(None, served)[1]
    few = next_time_status(None, _plate_protocol(_well_values(("P1", "P2"), 0.95, 0.99)))
    # Few repeats is a usual design: stated in the line, not flagged.
    assert few[0] is Status.OK and "from 2 plates" in few[1]
    # Never served, for lack of fields (not capped by the held-back fields): a problem.
    never = next_time_status(None, _plate_protocol(_well_values(many, 0.5, 0.6)))
    assert never[0] is Status.PROBLEM and "not reached" in never[1]
    # Reliability alone is per group only: never ✅, and the headline says why.
    alone = next_time_status(_mean_curve(_well_values(many, 0.95, 0.99)), None)
    assert alone[0] is Status.CAUTION and "not checked across samples" in alone[1]
    none = next_time_status(_mean_curve(_well_values(many, 0.5, 0.6)), None)
    assert none[0] is Status.PROBLEM


def test_a_group_skipped_by_the_reliability_curve_is_counted_apart():
    """#272: two wells assessed and one skipped (one field) are not "both groups"."""
    curve = _mean_curve(_well_values(("P1",), 0.95, 0.99))
    assert "enough in both groups" in next_time_status(curve, None)[1]
    skipped = pd.DataFrame(
        [{"condition": "ctrl", "plate": "P2", "well": "A1", "fovs": 1, "reason": "only 1 field"}]
    )
    curve = replace(curve, skipped=skipped)
    status, text = next_time_status(curve, None)
    assert status is Status.CAUTION and "both groups" not in text
    assert "enough in the 2 groups assessed for the target; 1 more not assessed" in text


def test_bought_marker_reads_the_confirmed_runs():
    result = _adequacy()
    status, text = bought_status(result)
    assert status in (Status.OK, Status.CAUTION)
    assert text.startswith("held-back fields")
    never = result.realized.assign(sustained_realized=pd.array([None] * len(result.realized)))
    lost = AdequacyResult(result.apparent, never, result.group_cols, result.settings)
    assert bought_status(lost)[0] is Status.CAUTION


def test_effort_short_is_one_table_and_full_keeps_every_sentence():
    result = _effort()
    short, full = result.summary(), result.summary(detail="full")
    assert "ctrl" in short and "condition=" not in short
    assert "only 3 repeats" in _flat(short)
    # Few independent units is the design: a note on how to read the shares, not a caution.
    assert CAUTION not in short and "a leaning is a reading, not a result" in _flat(short)
    assert "cannot tell from these data (only 3 repeats)" in _flat(full)
    assert "of its variance with the acquired design" in _flat(full)
    assert "Allocations compared" in full
    status, headline = effort_status(result)
    assert status is Status.OK and "not established" in headline and "with 3" in headline
