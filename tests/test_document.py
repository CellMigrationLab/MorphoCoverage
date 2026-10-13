"""Tests of ``workflows._document``: the assembled report, its parts, reasons and settings."""

from __future__ import annotations

import warnings

import pandas as pd
from _helpers_tables import _checks, _continuous_tables, _replace, _rows
from _helpers_workflows import _WELLS, AREA_SETUP, _report_parts, _sictrl_tables, _tables

import morphocoverage.plots as plots_module
import morphocoverage.workflows._document as document_module
import morphocoverage.workflows._shared as shared_module
from morphocoverage._caveats import is_assumption
from morphocoverage._integrity_tables import integrity_by_table
from morphocoverage.retrospective import microscopist_report
from morphocoverage.retrospective.report import MicroscopistReport
from morphocoverage.workflows import Setup
from morphocoverage.workflows._document import (
    _technical_repeats,
    _with_integrity,
    assemble,
    coverage_notes,
    plan_settings,
)


def test_j2_j16_j17_shared_helpers_are_named_listed_and_word_percentages_once():
    assert hasattr(shared_module, "check_workflow_inputs")
    assert not hasattr(shared_module, "check_tables")
    for module in (shared_module, document_module):
        assert module.__all__ and all(hasattr(module, name) for name in module.__all__)
        assert "Shared-private:" in module.__doc__
    settings = {
        "protocol_threshold": 0.999,
        "reservoir_fraction": 0.334,
        "reservoir_fovs": (2, 3),
        "n_splits": 2,
        "draws": 3,
        "permuted_orders": 1,
        "dispersed_orders": 0,
        "random_state": 0,
    }
    text = plan_settings(settings)
    assert "when 99.9% of random acquisitions" in text  # not "100%"
    assert "2 to 3 fields of each group held back (reservoir_fraction 33%" in text


def test_p7_8_findings_are_grouped_capped_and_named_among_the_fields_to_look_at():
    objects, fovs = _continuous_tables(n_fields=40, size=20)
    copied = objects
    for i in range(10):  # every field of well W1 among the first 40 is a copy of one of W0
        if fovs["technical_repeat"].iloc[i] == "W1":
            copied = _replace(copied, fovs, i, _rows(objects, fovs, i - 1))
    result = _checks(copied, fovs)
    full = result.summary(detail="full")
    lines = [line for line in full.splitlines() if line.startswith("  - ")]
    assert len(lines) == 1 and "3 pairs of fields between" in lines[0]
    # The short form: one table row for the check and pair of wells.
    rows = [line for line in result.summary().splitlines() if "copied field" in line]
    assert len(rows) == 1 and "3 pairs" in rows[0]
    document = _with_integrity(microscopist_report(), result)
    assert "name 6 fields in 3 findings" in document.sections["inspect"]
    assert any("Data integrity" in note for note in document.notes)
    assert "Data integrity" in document.limits()


def test_p1_7_the_report_names_the_variance_between_technical_repeats():
    objects, fovs = _tables(_WELLS)
    results, document = _report_parts(
        objects, fovs, ["condition", "biological_repeat", "technical_repeat"]
    )
    document = _technical_repeats(document, results)
    profile = document.sections["profile"]
    assert "Between technical repeats (x): in 2 condition and repeat groups" in profile
    assert "as much as their fields alone predict" in profile
    assert "only more technical repeats reduce (optimise, question 7)" in profile
    assert any("Each 'enough' is per technical repeat" in note for note in document.notes)


def test_p1_8_a_group_that_pools_technical_repeats_says_so_beside_enough():
    objects, fovs = _tables(_WELLS)
    results, document = _report_parts(objects, fovs, ["condition", "biological_repeat"])
    document = _technical_repeats(document, results)
    enough = document.sections["enough"]
    assert "Each analysis group pools the technical repeats of its condition and repeat" in enough
    assert "(2 to 3 per group" in enough and "add 'technical_repeat' to group_cols" in enough
    assert any("pools technical repeats (2 to 3 per group)" in note for note in document.notes)


def test_p1_9_the_limits_say_what_was_not_read_or_not_recorded():
    setup = Setup(readout="x", features=("x", "y"))
    notes = coverage_notes({"variability": object()}, setup)
    assert any("read the readout (x) only" in n and "(y)" in n for n in notes)
    assert any("No stage coordinates:" in n for n in notes)
    assert any("No acquisition order:" in n for n in notes)
    placed = Setup(readout="x", stage_cols=("sx", "sy"), acquisition_order_col="t")
    assert coverage_notes({}, placed) == []
    # Every note is standing: options left at their defaults (no stage coordinates, no
    # acquisition order) are notes on what was not assessed, not cautions about these data.
    kinds = {n.split(" ")[0]: is_assumption(n) for n in notes}
    assert kinds == {"The": True, "No": True}
    # Nothing left unassessed still reads as "on the assumptions stated", never "everything".
    assessed = MicroscopistReport({}, pd.DataFrame(), inputs={})
    closing = assessed.limits()
    assert "Every question could be answered; the assumptions each answer rests on" in closing
    assert "everything" not in closing


def test_issue_151_the_coverage_note_names_only_the_analyses_that_ran():
    from morphocoverage.strategies import FixedFOVStrategy

    setup = Setup(readout="x", features=("x", "y"))
    (note, *_) = coverage_notes({"variability": object()}, setup)
    assert "only, so a heavy tail or a between-well spread in the other declared features (y)" in (
        " ".join(note.split())
    )
    assert "enter" not in note and "novelty" not in note
    ran = {"variability": object(), "unusual": object(), "ranking": object()}
    (note, *_) = coverage_notes(ran, setup)
    assert "(y) enter the unusual cells and the feature ranking, so" in " ".join(note.split())
    custom = Setup(readout="x", features=("x", "y"), rules=(FixedFOVStrategy(4),))
    (note, *_) = coverage_notes(ran, custom)
    assert "stopping rule" not in note
    assert "The Foundation and the inspection's influence" in note


def test_dead_private_surface_stays_removed():
    """#131 (#152, #203, #229, #246, #258): no unused parameter or re-export."""
    import inspect

    from morphocoverage.retrospective import _text, effort, report
    from morphocoverage.retrospective.hierarchy import SamplingHierarchy
    from morphocoverage.workflows._document import protocol_reason

    assert list(inspect.signature(protocol_reason).parameters) == ["hierarchy", "strata", "groups"]
    hierarchy = SamplingHierarchy("position_id", "technical_repeat", "biological_repeat")
    reason = protocol_reason(hierarchy, ["site"], ["condition"])
    assert "add 'technical_repeat', 'biological_repeat', 'site' to group_cols" in reason
    assert protocol_reason(hierarchy, [], ["technical_repeat", "biological_repeat"]) is None
    assert not hasattr(plots_module, "PALETTE")
    assert not any(hasattr(effort, name) for name in ("_se", "_acquired_se"))
    assert not hasattr(_text, "group_identity")
    assert not any(hasattr(report, n) for n in ("_detection_adjusted_bounds", "_false_alarm_rate"))


def test_the_between_well_spread_names_the_wells_and_groups_left_out():
    from _helpers_workflows import _WELLS, _report_parts, _tables

    from morphocoverage.workflows._document import _technical_repeats

    columns = ["condition", "biological_repeat", "technical_repeat"]
    objects, fovs = _tables(_WELLS)
    results, document = _report_parts(objects, fovs, columns)
    ordinary = _technical_repeats(document, results).sections["profile"]
    assert "in 2 condition and repeat groups" in ordinary and "Left out" not in ordinary
    wells = [
        ("ctrl", "r1", "A", 6, 10.0, 0.1),
        ("ctrl", "r1", "B", 6, 14.0, 0.1),
        ("ctrl", "r1", "C", 1, 8.0, 0.1),
        ("drug", "r1", "A", 6, 12.0, 0.1),
        ("drug", "r1", "B", 1, 16.0, 0.1),
        ("solo", "r1", "A", 6, 12.0, 0.1),
    ]
    objects, fovs = _tables(wells)
    results, document = _report_parts(objects, fovs, columns)
    profile = " ".join(_technical_repeats(document, results).sections["profile"].split())
    # The estimate is kept, from the 2 wells of ctrl with 2 or more fields.
    assert "in 1 condition and repeat group with 2 or more technical repeats" in profile
    assert (
        "Left out: technical repeat C of ctrl · repeat r1, with one field, whose field variance "
        "cannot be estimated; drug · repeat r1, with fewer than 2 technical repeats of 2 or "
        "more fields." in profile
    )
    # A group with one well has nothing between wells: the usual design, not named.
    assert "solo" not in profile


def test_assemble_puts_the_integrity_checks_in_the_overview_without_one():
    objects, fovs = _sictrl_tables(fields=4, cells=5)
    objects.loc[5:9, "area"] = objects.loc[0:4, "area"].to_numpy()  # a copied field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integrity = integrity_by_table(objects, fovs, tables={"features": ["area"]})
    document = assemble({"integrity": integrity}, AREA_SETUP)
    assert document.overview == integrity.summary(detail="full")
    assert document.overview.startswith(
        "❌ Data integrity — 1 finding in 2 fields (structure to inspect, not a judgment)\n"
    )
    assert (
        "⚠️ 6. Which fields should I look at? — not assessed; needs inspect_fovs; integrity "
        "names 2 fields"
    ) in document.summary()


def test_a_text_added_to_a_short_section_goes_before_its_caveats_each_once():
    from morphocoverage.workflows._document import _merged

    short = "  - a line\n  ⚠️ First caveat,\n     continued.\n  ⚠️ Second caveat."
    brief = "  - an added line\n  ⚠️ Second caveat.\n  ⚠️ Third caveat."
    assert _merged(short, brief).splitlines() == [
        "  - a line",
        "  - an added line",
        "  ⚠️ First caveat,",
        "     continued.",
        "  ⚠️ Second caveat.",
        "  ⚠️ Third caveat.",
    ]
