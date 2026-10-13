"""Tests of ``_caveats``: the two kinds of caveat and how a section states them."""

from __future__ import annotations

import pandas as pd
import pytest
from _helpers_strategies import _flat, _named_scenario_report

from morphocoverage import ScalarPrecisionStrategy
from morphocoverage._caveats import (
    ASSUMPTIONS,
    assumption,
    assumption_lines,
    caveat_lines,
    has_caution,
    is_assumption,
    once,
    pool_assumptions,
)
from morphocoverage._style import CAUTION, Status
from morphocoverage.retrospective import microscopist_report
from morphocoverage.retrospective.report import MicroscopistReport
from morphocoverage.retrospective.report_status import caveat_status


def test_issue_148_a_caution_answering_a_question_not_assessed_raises_its_status():
    from morphocoverage._caveats import assumption, caveat_lines
    from morphocoverage._style import Status
    from morphocoverage.workflows._document import with_text

    def added(text: str, key: str = "profile", status: Status = Status.OK) -> Status:
        document = MicroscopistReport(
            sections={key: "  needs it"},
            profile=pd.DataFrame(),
            inputs={key: "not assessed"},
            statuses={key: Status.NOT_ASSESSED},
        )
        short = "\n".join(caveat_lines([text], indent=2))
        return with_text(document, key, text, status=status, short_text=short).statuses[key]

    assert added("a caution about these data") is Status.CAUTION
    assert added("a caution about these data", "populations") is Status.OK
    assert added("a caution", status=Status.PROBLEM) is Status.PROBLEM
    assert added(assumption("a standing note")) is Status.OK


def test_an_assumption_is_its_text_and_keeps_its_kind_through_once():
    standing = assumption("the bound is approximate")
    assert standing == "the bound is approximate" and is_assumption(standing)
    assert not is_assumption("the bound is approximate")
    kept = once([standing, " the bound is approximate ", "only 3 repeats", "only 3 repeats"])
    # A text given as a caution too stays a caution: a caution is never hidden.
    assert kept == ["the bound is approximate", "only 3 repeats"]
    assert not is_assumption(kept[0])
    assert is_assumption(once([standing, assumption("x")])[0])


def test_assumptions_are_never_marked_and_are_listed_once_under_their_heading():
    caveats = [
        "only 3 repeats",
        assumption("the next-field bound is approximate"),
        assumption("the next-field bound is approximate "),
        assumption("hold-outs are not independent experiments (they share fields)"),
    ]
    short = "\n".join(caveat_lines(caveats, indent=2))
    assert short.splitlines()[0] == f"  {CAUTION} Only 3 repeats."
    assert short.count(CAUTION) == 1
    assert short.count(ASSUMPTIONS) == 1 and short.count("next-field bound is approximate") == 1
    # Short: one paragraph, each sentence closed (after a parenthesis too).
    assert _flat(short).endswith(
        "Assumptions: The next-field bound is approximate. Hold-outs are not independent "
        "experiments (they share fields)."
    )
    full = caveat_lines(caveats, indent=2, detail="full")
    assert full[1] == f"  {ASSUMPTIONS}"
    assert full[2:] == [
        "    - The next-field bound is approximate.",
        "    - Hold-outs are not independent experiments (they share fields).",
    ]
    assert assumption_lines([]) == [] and caveat_lines([]) == []
    with pytest.raises(ValueError, match="detail"):
        caveat_lines(caveats, detail="long")


def test_a_caveat_that_starts_with_a_name_keeps_its_case():
    # "siCTRL · repeat 1: ..." read "SiCTRL" (audit, 2026-10-02); a plain word still opens
    # with a capital.
    lines = caveat_lines([assumption("siCTRL · repeat 1: no trend"), "drugA wells differ"], 0)
    assert "siCTRL · repeat 1: no trend." in " ".join(lines)
    assert lines[0].endswith("drugA wells differ.")
    assert caveat_lines([assumption("the bound is approximate")], 0) == [
        "Assumptions: The bound is approximate."
    ]
    assert "Dna_intensity" not in " ".join(caveat_lines(["dna_intensity moves"], 0))


def test_a_section_built_from_several_results_states_its_assumptions_once_at_its_end():
    first = "\n".join(
        ["  first result", *caveat_lines(["a caution", assumption("a")], 2, detail="full")]
    )
    second = "\n".join(
        ["  second result", *caveat_lines([assumption("a"), assumption("b")], 2, detail="full")]
    )
    pooled = pool_assumptions(f"{first}\n\n{second}")
    assert pooled.count(ASSUMPTIONS) == 1 and pooled.endswith("  - A.\n    - B.")
    assert pooled.index("second result") < pooled.index(ASSUMPTIONS)
    assert f"{CAUTION} A caution." in pooled
    short_first = "\n".join(["  one", *caveat_lines([assumption("a"), assumption("b")], 2)])
    short_second = "\n".join(["  two", *caveat_lines([assumption("b"), assumption("c")], 2)])
    merged = pool_assumptions(f"{short_first}\n{short_second}")
    assert merged.splitlines()[-1] == "  Assumptions: A. B. C."
    assert pool_assumptions("  no caveat here") == "  no caveat here"


def test_assumptions_never_raise_a_status_and_cautions_about_the_data_do():
    standing = "\n".join(["  answer", *caveat_lines([assumption("a"), assumption("b")], 2)])
    cautioned = "\n".join(["  answer", *caveat_lines(["only 3 repeats", assumption("a")], 2)])
    assert not has_caution(standing) and has_caution(cautioned)
    assert caveat_status(Status.OK, standing) is Status.OK
    assert caveat_status(Status.OK, cautioned) is Status.CAUTION
    assert caveat_status(Status.PROBLEM, cautioned) is Status.PROBLEM
    assert caveat_status(Status.NOT_ASSESSED, cautioned) is Status.NOT_ASSESSED


def test_question_2_is_fine_when_its_only_caveats_are_standing_assumptions():
    report = _named_scenario_report(
        "homogeneous",
        [ScalarPrecisionStrategy("feature_a", 0.5, "fov", "upper_95")],
        readout="feature_a",
        stage_cols=("stage_x", "stage_y"),
    )
    assert not report.final["order_flagged"].any()
    document = microscopist_report(sufficiency=report)
    assert document.statuses["enough"] is Status.OK
    section = document.short["enough"]
    assert CAUTION not in section
    # The assumption is still there, once, under its heading.
    assert section.count(ASSUMPTIONS) == 1
    assert _flat(section).count("still treats the fields as a random sample") == 1
    full = document.sections["enough"]
    assert CAUTION not in full and full.count(ASSUMPTIONS) == 1
