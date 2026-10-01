"""The plan of a pilot with one biological repeat and one well per condition: what it can and
cannot say, and what to image next time (questions 4 and 7)."""

from types import SimpleNamespace

import pandas as pd

from morphocoverage._style import Status, among
from morphocoverage.retrospective.report_status_plan import effort_status

ONE_WELL = (
    "a single 'technical_repeat' unit holds observations, so the variance at that level cannot "
    "be estimated; at least 2 units are needed"
)


def _effort(reasons):
    return SimpleNamespace(
        comparisons=pd.DataFrame(),
        skipped=pd.DataFrame({"stratum": list(range(len(reasons))), "reason": reasons}),
        settings={"hierarchy": {"sample": "technical_repeat"}},
    )


def test_one_well_per_condition_says_what_would_answer_question_7():
    status, headline = effort_status(_effort([ONE_WELL] * 4))
    assert status is Status.NOT_ASSESSED
    assert headline == (
        "not estimable with one technical repeat per condition; image at least 2 per condition "
        "next time to answer it"
    )
    # Another reason keeps the plain answer.
    assert effort_status(_effort(["every FOV holds a single observation"]))[1] == "not assessed"


def test_two_of_two_reads_both():
    assert among(2, 2, "target") == "both targets"
    assert among(3, 3, "target") == "all 3 targets"
    assert among(1, 2, "target") == "1 of 2 targets"
