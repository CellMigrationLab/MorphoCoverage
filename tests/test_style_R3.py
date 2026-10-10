"""Readable reports, questions 6, 8 and 9, the declaration and the closing section.

The short form (the default) states each answer in a table or a few lines and every caveat
once; the full form keeps every number the old sentences had; each answer's marker follows
the rules of ``report_status``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from test_retrospective_inspection import NOVELTY, _acquisition, _inspect
from test_retrospective_spatial import _efficiency, _gradient
from test_sampling_unit import _technical, _two_conditions
from test_workflows import _fingerprint, _rare, _setup

from morphocoverage._style import CAUTION, NOT_ASSESSED, PROBLEM, Status
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.retrospective import microscopist_report
from morphocoverage.retrospective.report import DECLARATION_TITLE, LIMITS_TITLE
from morphocoverage.retrospective.report_status import (
    conditions_status,
    inspect_status,
    order_status,
)
from morphocoverage.sampling_unit import condition_contrast, feature_ranking
from morphocoverage.workflows._declaration import declaration_short
from tests.unrecorded import unrecorded

CONTROL = {"condition": "control"}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _widths(text: str) -> list[int]:
    return [len(line) for line in text.splitlines()]


# --- 6. Which fields should I look at? ----------------------------------------------------


def test_the_fields_to_reopen_are_one_table_and_the_full_form_keeps_every_reason():
    objects, fovs = _acquisition()
    fovs.loc[3, "stage_x"] = np.nan  # the stage neighbours cannot be compared
    result = _inspect(objects, fovs, stage_cols=("stage_x", "stage_y"))
    short = result.summary()
    assert short.splitlines()[0].startswith("3 fields to reopen in 1 group")
    # A long reason (an influence) does not fit a table row: one line per field.
    assert "control · repeat R1 · f07: removing it lowers" in short
    for name in ("f03", "f07", "f12"):
        assert name in short
    # Said once, and the reason not judged is a caveat.
    assert short.count("None of these is called an error") == 1
    assert f"{CAUTION} Stage neighbours were not compared in control · repeat R1" in short
    full = result.summary(detail="full")
    assert "Analysis groups: condition · biological repeat." in full
    assert "control · repeat R1: fields to reopen before trusting the result" in full
    for detail in result.fovs["detail"]:
        assert _flat(detail) in _flat(full)
    assert "f07 (stage_x 200.0, stage_y 100.0)" in full
    with pytest.raises(ValueError, match="short"):
        result.summary(detail="long")


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
    assert NOVELTY is not None


# --- 8. Could I have sampled more efficiently? --------------------------------------------


def _orders():
    orders = [AcquisitionOrder.recorded(), AcquisitionOrder.permuted(1)]
    return _efficiency(*_gradient(), [*orders, AcquisitionOrder.dispersed(0)])


def test_the_order_comparison_is_one_table_per_order_kind_and_every_median_in_full():
    result = _orders()
    short = result.summary()
    assert re.search(r"order\s+fields\s+stage travel\s+time\s+never enough", short)
    for name in ("recorded", "random", "spread out"):
        assert re.search(rf"^  {name}\s", short, re.MULTILINE)
    assert short.count("straight-line path") == 1
    full = result.summary(detail="full")
    table = result.table
    for row in table.to_dict("records"):
        if row["median_fovs"] is not None and not np.isnan(row["median_fovs"]):
            fields = f"{row['median_fovs']:g} field"
            assert fields in full
    assert "Analysis groups: condition · biological repeat." in full
    assert "min" in full  # the timing was given
    with pytest.raises(ValueError, match="short"):
        result.summary(detail="brief")


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


# --- 9. Do the conditions differ? --------------------------------------------------------


def _wells():
    return _technical(*_two_conditions(shift=2.0, field_sd=0.3, seed=7, n_wells=5, n_fields=6))


def test_the_contrasts_are_one_table_each_caveat_once_and_every_number_in_full():
    objects, fovs = _wells()
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, n_permutations=400, n_draws=5
    )
    short = result.summary()
    header = r"contrast\s+fields\s+cells\s+tech\.\s+d on cells\s+d on field means"
    assert re.search(header, short)
    assert _flat(short).count("compare means") == 1
    assert _flat(short).count("drawn with replacement") == 1
    full = result.summary(detail="full")
    for row in result.table.to_dict("records"):
        for column in ("d_objects", "d_fields", "d_technical", "effect_full"):
            assert f"{row[column]:.2f}" in full
        assert f"p = {row['p_value']:.3g}" in full
        assert f"{int(row['distinct_assignments'])} distinct assignments" in full
    for row in result.effects.to_dict("records"):
        assert f"{row['median']:.2f} ({row['q25']:.2f} to {row['q75']:.2f})" in full
    assert full.count("compare means") == 1
    assert conditions_status(result, None)[0] is Status.OK


def test_a_contrast_the_design_cannot_test_is_left_out_said_once_not_flagged():
    # One technical repeat a side and two biological repeats: no test within or across
    # repeats. That is a usual pilot design, so the answer is "not tested", not a warning.
    objects, fovs = _technical(*_two_conditions(shift=2.0, field_sd=0.3, n_wells=1))
    result = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    status, headline = conditions_status(result, None)
    assert status is Status.NOT_ASSESSED
    assert "no test (one well on a side)" in headline
    short = result.summary()
    assert " ".join(short.split()).count("only one technical repeat") == 1
    assert "no test" in short and CAUTION not in short


def test_the_ranking_is_a_table_of_the_best_features_with_its_caveats_once():
    objects, fovs = _wells()
    ranking = feature_ranking(
        objects, fovs, features=["y", "x"], control=CONTROL, n_permutations=400
    )
    short = ranking.summary()
    assert "1 of 2 features pass the gate" in short
    assert re.search(r"rank\s+feature\s+\|d\|\s+best contrast\s+q\s+passes in", short)
    assert short.count("distinct assignments") == 1
    full = ranking.summary(detail="full")
    assert "Ranking within repeats (wells as the unit): each of the 2 declared features" in full
    best = ranking.ranking.iloc[0]
    assert f"{best['score']:.2f}" in full and f"{best['q_value']:.3g}" in full
    assert "No contrast passes the gate for 1 feature: y." in full


# --- The report: short sections, the declaration and the closing ---------------------------


def test_the_report_prints_the_short_forms_and_its_caveats_once():
    objects, fovs = _wells()
    contrast = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    inspection = _inspect(*_acquisition())
    report = microscopist_report(contrast=unrecorded(contrast), inspection=unrecorded(inspection))
    short = report.summary()
    # The answer across biological repeats first, then the contrasts within each repeat.
    # The reading per condition leads, then the table across repeats.
    assert report.short["conditions"].startswith("  Reading, condition by condition")
    assert "Across the 2 biological repeats" in report.short["conditions"]
    assert "Within each repeat: x, each condition against the control" in _flat(short)
    assert "user's inference" not in _flat(short)
    assert _flat(short).count("compare means") == 1
    assert report.short["inspect"] in short
    assert all(width <= 100 for width in _widths(report.short["inspect"]))
    full = report.summary(detail="full")
    assert "Feature x, each condition against the control" in full
    # The closing: what was not assessed, in a line each, and the notes as caveats.
    closing = short.split(LIMITS_TITLE)[1]
    # With a contrast, question 2 is answered from it (the difference resolved or not).
    assert f"{NOT_ASSESSED} 1. Can the features see the phenotype? — not assessed" in closing
    assert "2. Did I image enough? — not assessed" not in closing
    # A standing assumption of every answer: listed under "Assumptions:", not marked.
    assert "Assumptions: Every answer holds for the stated features" in _flat(closing)
    assert f"{CAUTION} Every answer holds" not in closing
    limits = report.limits(detail="full")
    # One line per limit pointing to its section, where its need is (owner, 2026-10-06).
    assert "4. How many fields should I image next time?: not assessed, see question 4." in limits
    reasons = report.limits(detail="full", pointers=False)
    assert "4. How many fields should I image next time?: not assessed; needs: run" in reasons
    assert report.limits() == limits


def test_the_closing_marks_a_weak_representation_as_a_problem():
    report = microscopist_report()
    weak = type(report)(
        report.sections,
        report.profile,
        ("Weak representation: fields were recognized in only 50% of checks.",),
        report.inputs,
    )
    assert f"{PROBLEM} Weak representation" in weak.limits(detail="short")


def test_the_declaration_short_form_names_the_rules_and_states_each_caveat_once():
    setup = _setup(fingerprint=(_fingerprint(),), rare=_rare(), rare_label="state")
    text = declaration_short(setup)
    assert "Population summary: the mean of" in text
    assert (
        "Stopping rules: mean-precision rule, per object, 5% and mean-precision rule, per "
        "field, 5% (the default)" in _flat(text)
    )
    assert _flat(text).count("the standard error of the mean is 5% of the mean") == 1
    assert "the binomial plan asks for 14 representative fields" in _flat(text)
    assert all(width <= 101 for width in _widths(text))
    plain = declaration_short(_setup(acquisition_order_col=None, stage_cols=None))
    # Options left at their defaults are standing notes, not cautions.
    assert "No phenotypic fingerprint declared" in _flat(plain) and CAUTION not in plain
    assert "No rare phenotype declared" in _flat(plain)
    assert "the order is not checked" in _flat(plain)
    # The full sentences name each rule in words, with its strategy.
    rules = next(s for s in setup.declaration() if s.startswith("Stopping rules:"))
    assert "mean-precision rule, per object, 5% [scalar_precision_object_upper95 (" in rules


def test_the_workflow_document_prints_the_short_declaration_and_the_full_one_on_demand():
    from test_workflows import _acquisition as _workflow_acquisition

    from morphocoverage.workflows import report

    objects, fovs = _workflow_acquisition()
    result = report(objects, fovs, _setup())
    short = result.summary()
    declared = short.split(DECLARATION_TITLE)[1]
    assert declared.startswith("\n  Population summary:")
    full = result.summary(detail="full")
    assert f"{DECLARATION_TITLE}\n  - Population summary" in full
    assert result.document.short["declaration"] in short
