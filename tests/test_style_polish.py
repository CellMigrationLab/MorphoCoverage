"""The readable-reports polish: cautions about these data apart from standing assumptions.

The owner's decision (2026-09-28): ⚠️ (and a section's ⚠️ status) is used only
for a caution found in these data (a measured property, count or condition,
or a missing declaration that kept a check from running). A standing
assumption of a method, true of every dataset, is listed once per section
under a plain "Assumptions:" heading: never dropped, never marked, never
raising a status. These tests hold ``morphocoverage._caveats`` and the
statuses of ``retrospective.report_status`` to that rule, and the "At a
glance" headline of question 2 to the table it summarises.
"""

import re

import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
)
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
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.order_check import NOT_CHECKED, RANDOM_SAMPLE_ASSUMPTION, order_caveat
from morphocoverage.retrospective import microscopist_report
from morphocoverage.retrospective.report_answers import ASSESSED
from morphocoverage.retrospective.report_status import caveat_status, enough_status
from morphocoverage.strategies import DescriptorPlateauStrategy
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.sufficiency_short import rule_tallies

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
ORDERS = [AcquisitionOrder.recorded(), *(AcquisitionOrder.permuted(s) for s in range(3))]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _report(scenario: str, strategies, **kwargs):
    built = SCENARIOS[scenario]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=ORDERS,
        acquisition_order_col="acquisition_order",
        **kwargs,
    )


# --- The two kinds ------------------------------------------------------------------


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


# --- Statuses -----------------------------------------------------------------------


def test_assumptions_never_raise_a_status_and_cautions_about_the_data_do():
    standing = "\n".join(["  answer", *caveat_lines([assumption("a"), assumption("b")], 2)])
    cautioned = "\n".join(["  answer", *caveat_lines(["only 3 repeats", assumption("a")], 2)])
    assert not has_caution(standing) and has_caution(cautioned)
    assert caveat_status(Status.OK, standing) is Status.OK
    assert caveat_status(Status.OK, cautioned) is Status.CAUTION
    assert caveat_status(Status.PROBLEM, cautioned) is Status.PROBLEM
    assert caveat_status(Status.NOT_ASSESSED, cautioned) is Status.NOT_ASSESSED


def test_a_stop_rests_on_an_assumption_only_when_the_order_check_ran_and_found_nothing():
    clear = "order check: no trend with acquisition order (Spearman p = 0.4) over 20 fields; " + (
        f"still {RANDOM_SAMPLE_ASSUMPTION}"
    )
    assert is_assumption(order_caveat(clear))
    assert "every stop still treats" in order_caveat(clear)
    flagged = "caution: the field means of a trend with acquisition order (...)"
    assert not is_assumption(order_caveat(flagged)) and order_caveat(flagged) == flagged
    # No check run: a missing declaration is a caution about these data, and says so.
    unchecked = order_caveat(RANDOM_SAMPLE_ASSUMPTION)
    assert not is_assumption(unchecked) and unchecked.startswith(NOT_CHECKED)
    assert not is_assumption(order_caveat("not tested: fewer than 4 fields with a mean"))


def test_question_2_is_fine_when_its_only_caveats_are_standing_assumptions():
    report = _report(
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


def test_question_2_without_an_order_check_states_a_caution_and_looks_at_it():
    report = _report("homogeneous", [ScalarPrecisionStrategy("feature_a", 0.5, "fov", "upper_95")])
    document = microscopist_report(sufficiency=report)
    section = _flat(document.short["enough"])
    assert f"{CAUTION} The acquisition order was not checked, so every stop treats" in section
    assert document.statuses["enough"] is Status.CAUTION


def test_a_rules_standing_warning_does_not_count_as_a_caution_about_the_data():
    report = _report(
        "homogeneous",
        [DescriptorPlateauStrategy("feature_a", 50.0, 3, "acquisition_order")],
        readout="feature_a",
        stage_cols=("stage_x", "stage_y"),
    )
    warnings = [w for texts in report.final["warning_list"] for w in texts]
    assert warnings and all(is_assumption(w) for w in warnings)
    assert all(t["warned"] == 0 for t in rule_tallies(report))
    section = microscopist_report(sufficiency=report).short["enough"]
    tolerance = "the tolerance bounds how much the descriptors changed"
    assert _flat(section).count(tolerance.capitalize()[:30]) + _flat(section).count(tolerance) == 1
    assert tolerance not in "\n".join(
        line for line in section.splitlines() if line.lstrip().startswith(CAUTION)
    )


# --- Headlines ----------------------------------------------------------------------


def _table_rules(short: str) -> list[str]:
    """The rule rows of question 2's short table."""
    lines = short.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("rule "))
    pad = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    rows = []
    for line in lines[start + 1 :]:
        if not line.startswith(pad) or line[len(pad)] == " " or not line.strip():
            break
        rows.append(line.strip())
    return rows


def test_the_question_2_headline_counts_the_rules_as_the_table_lists_them():
    report = _report(
        "condition_repeat_groups",
        [
            FixedFOVStrategy(2),
            ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point"),
            ScalarPrecisionStrategy("feature_a", 1e-6, "fov", "point"),
        ],
    )
    rows = _table_rules(report.summary())
    status, text = enough_status(report)
    found = re.match(r"(\d+) of (\d+) rules said enough in every group", text)
    assert found, text
    assert int(found.group(2)) == len(rows) == 3
    everywhere = sum(" 4 of 4 " in f" {row} " for row in rows)
    assert int(found.group(1)) == everywhere == 2
    assert "mean-precision rule, per field, 0.0001%, point SD: in 0 of 4" in text
    assert "cautions below" not in text and status is Status.CAUTION


def test_every_headline_of_the_idr0028_like_report_stays_within_two_lines():
    report = _report(
        "condition_repeat_groups",
        [FixedFOVStrategy(2), ScalarPrecisionStrategy("feature_a", 0.5, "fov", "upper_95")],
    )
    glance = microscopist_report(sufficiency=report).summary().split("\n\n")[1]
    answers = re.split(r"\n(?=  \S)", glance)[1:]
    assert answers and all(len(answer.splitlines()) <= 2 for answer in answers)


def test_the_closing_says_every_question_could_be_answered_in_plain_words():
    assert ASSESSED == (
        "Every question could be answered; the assumptions each answer rests on are listed with it."
    )
