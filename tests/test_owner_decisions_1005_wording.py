"""The owner's decisions of 2026-10-05 on the retrospective report's wording: a post-hoc
"best" allocation or rule family is described, not tested (#243, #244); the
representation-control rate is pooled over its strata, and said so (#253); a declared
planted check that could not run leaves the representation gate not assessed (#270);
assessed and not assessed groups are counted apart in the headlines (#272); and the
between-well spread names what it leaves out (#150)."""

from types import SimpleNamespace

import pandas as pd

from morphocoverage._style import Status
from morphocoverage.retrospective import effort_text, microscopist_report
from morphocoverage.retrospective.controls import ControlResult
from morphocoverage.retrospective.controls_text import (
    NOT_PLANTED,
    pooled_over,
    representation_gate,
)
from morphocoverage.retrospective.frontier import BASELINE, FrontierResult
from morphocoverage.retrospective.frontier_text import frontier_status
from morphocoverage.retrospective.inspection import InspectionResult
from morphocoverage.retrospective.report_enough import detection_line
from morphocoverage.retrospective.report_profile import (
    _detection_adjusted_bounds,
    _new_state_rows,
    _redundancy_rows,
)
from morphocoverage.retrospective.report_status import controls_status, unit_status

# --- #243, #244: a post-hoc selected best is described ---------------------------------------


def _allocations(ratios):
    """Equal-total allocations, each resolved by its own interval, with these SE ratios."""
    return pd.DataFrame(
        {
            "samples": [10 * (i + 2) for i in range(len(ratios))],
            "fovs_per_sample": [4.0] * len(ratios),
            "ratio": ratios,
            "ratio_low": [r - 0.05 for r in ratios],
            "ratio_high": [r - 0.01 for r in ratios],
            "same_total": [True] * len(ratios),
            "resolved": [True] * len(ratios),
            "current_samples": [12] * len(ratios),
            "current_fovs_per_sample": [4.0] * len(ratios),
        }
    )


def test_the_smallest_resolved_allocation_is_described_not_called_the_answer():
    compared = _allocations([0.9, 0.8, 0.85])
    line = effort_text._allocation_line(compared, {}, 12, "biological_repeat", "sample")
    assert "the smallest standard error that resolved here is with 30 samples" in line
    assert "20% smaller than the acquired 12 samples" in line
    assert "would give" not in line and "best" not in line
    answer, reason = effort_text._allocation_cell(compared, {}, 12, "biological_repeat", "sample")
    assert reason is None
    assert answer == "30 samples of about 4 fields: 20% smaller SE, the smallest that resolved here"


def _score(family, saved, low):
    return {
        "family": family,
        "alpha": 0.05,
        "setting": 0.1,
        "effort": 6.0 - saved,
        "risk": 0.04,
        "fields_saved": saved,
        "saved_low": low,
        "saved_high": saved + 1,
        "saved_reached": 1.0,
        "adaptivity": 0.3,
    }


def _frontier(rows):
    base = {**_score(BASELINE, 0.0, None), "setting": 6.0, "effort": 6.0}
    settings = {"label": "the mean", "units": 8, "unit_cols": ["biological_repeat"]}
    return FrontierResult(pd.DataFrame(), pd.DataFrame([base, *rows]), pd.DataFrame(), settings)


def test_the_family_that_saved_most_is_described_when_several_were_scored():
    one = _frontier([_score("saver", 2.0, 0.5)])
    assert frontier_status(one) == (Status.OK, "at ≤5% risk, saver saves 2.0 of 6 fields")
    several = _frontier([_score("saver", 2.0, 0.5), _score("other", 1.0, 0.2)])
    status, text = frontier_status(several)
    assert status is Status.OK
    assert "saver saves the most here, 2.0 of 6 fields" in text
    unresolved = _frontier([_score("saver", 2.0, -0.5), _score("other", 1.0, -0.2)])
    assert (
        "saver may save the most here, 2.0 fields (not resolved)" in frontier_status(unresolved)[1]
    )


# --- #253: the control's rate is pooled over its strata, and said so -------------------------


def _two_plates(settings=None, **rates):
    """A ControlResult over two plates: kind -> ((new, judged) on P1, (new, judged) on P2),
    pooled as ``representation_control`` pools them."""
    table = pd.DataFrame(
        [
            {"plate": plate, "stratum": f"plate={plate}", "candidate": kind}
            | {"judged_new": n, "candidates": t, "not_judged": 0, "rate": n / t}
            for kind, per_plate in rates.items()
            for plate, (n, t) in zip(("P1", "P2"), per_plate, strict=True)
        ]
    )
    pooled = table.groupby("candidate", sort=False)[["judged_new", "candidates"]].sum()
    pooled = pooled.reset_index().assign(not_judged=0)
    pooled["rate"] = pooled["judged_new"] / pooled["candidates"]
    return ControlResult(
        table, pooled, "assessed", ("plate",), {"fovs_seen": 10, **(settings or {})}
    )


def test_a_strong_plate_and_a_weak_one_are_said_to_be_pooled_wherever_the_rate_is_used():
    # P1 sees the control in every check, P2 in half: one pooled rate for both, said so.
    controls = _two_plates(
        positive=((10, 10), (5, 10)),
        positive_lofo=((10, 10), (5, 10)),
        reference=((0, 10), (1, 10)),
        reference_lofo=((0, 10), (1, 10)),
    )
    assert pooled_over(controls) == "pooled over all plates"
    one_plate = ControlResult(
        controls.table.query("plate == 'P1'"), controls.pooled, "assessed", ("plate",), {}
    )
    assert pooled_over(one_plate) == ""
    sufficiency = SimpleNamespace(
        next_fov=pd.DataFrame(
            {"plate": ["P1", "P2"], "unseen_state_max_frequency": [0.1, 0.1], "confidence": 0.95}
        ),
        group_cols=("plate",),
    )
    bounds = _detection_adjusted_bounds(sufficiency, controls)
    assert "of 20 checks, pooled over all plates):" in bounds
    assert "plate=P1: up to" in bounds and "plate=P2: up to" in bounds
    line, _ = detection_line(sufficiency, controls)
    assert "at 95% confidence, pooled over all plates):" in line
    inspection = InspectionResult(
        pd.DataFrame({"reason": ["new state"], "plate": ["P2"], "f": ["w"]}),
        ("plate",),
        {"fov_key": ["plate", "f"], "n_fovs": 40, "n_fovs_judged": 40},
    )
    statement = _new_state_rows(inspection, controls)[0]["statement"]
    assert "the declared reference's false-alarm rate, pooled over all plates, each" in statement
    status, headline = controls_status(controls)
    assert status is Status.PROBLEM
    assert headline == (
        "weak representation: positive-control fields recognized in 75% of checks, pooled over "
        "all plates"
    )
    report = microscopist_report(controls=controls)
    note = next(n for n in report.notes if n.startswith("Weak representation"))
    assert "recognized in only 75% of checks, pooled over all plates (below 90%)" in note
    strong = _two_plates(positive=((10, 10), (9, 10)), reference=((0, 10), (1, 10)))
    headline = (
        "positive-control fields recognized in 95% of checks; false alarms 5%, both pooled "
        "over all plates"
    )
    assert controls_status(strong) == (Status.OK, headline)


# --- #270: a declared planted check that could not run is not replaced by the whole fields --


PLANTED = {"control": {"planted_objects": 5}}


def test_the_planted_gate_follows_the_declaration_and_never_reads_the_whole_fields():
    strong = _two_plates(PLANTED, positive=((9, 10), (9, 10)), planted=((10, 10), (9, 10)))
    assert representation_gate(strong, 0.9) == ("", False)
    weak = _two_plates(PLANTED, positive=((10, 10), (10, 10)), planted=((5, 10), (3, 10)))
    gate, is_weak = representation_gate(weak, 0.9)
    assert is_weak and "ordinary fields with 5 planted positive-control objects" in gate
    assert "only 40% of checks, pooled over all plates" in gate
    # No planting could be judged: not assessed, whether the whole fields are strong or weak.
    for whole in ((10, 10), (2, 10)):
        unplanted = _two_plates(PLANTED, positive=(whole, whole))
        assert representation_gate(unplanted, 0.9) == (NOT_PLANTED, False)
    undeclared = _two_plates(positive=((5, 10), (5, 10)))
    gate, is_weak = representation_gate(undeclared, 0.9)
    assert is_weak and "fields of the declared positive control" in gate


def test_a_report_with_no_planting_says_the_gate_was_not_assessed():
    from test_microscopist_audit import _weak_control_report

    _, sufficiency = _weak_control_report()
    plain = microscopist_report(sufficiency=sufficiency)
    # Whole positive fields recognized in only 20% of checks: never the planted answer.
    controls = _two_plates(PLANTED, positive=((2, 10), (2, 10)), reference=((0, 10), (0, 10)))
    report = microscopist_report(controls=controls, sufficiency=sufficiency)
    assert report.statuses["enough"] is plain.statuses["enough"]
    assert report.headlines["enough"] == (
        f"small-state representation not assessed; {plain.headlines['enough']}"
    )
    assert report.sections["enough"].lstrip().startswith(f"{NOT_PLANTED}.")
    assert f"{Status.NOT_ASSESSED.marker} {NOT_PLANTED}." in report.summary()
    assert not any(n.startswith("Weak representation") for n in report.notes)
    assert f"{NOT_PLANTED}, so the adequacy answers (questions 2, 4 and 5) are not checked" in (
        " ".join(report.notes)
    )


# --- #272 (#259, #278): assessed and not assessed groups are counted apart --------------------


def _redundancy(*groups):
    """One row per group: (group, smallest_lag, redundancy_distance, reason)."""
    return pd.DataFrame(
        [
            {
                "well": well,
                "smallest_lag": lag,
                "largest_lag": None if lag is None else 2000.0,
                "redundancy_distance": distance,
                "beyond_chance": None if lag is None else True,
                "null_share": None if lag is None else 0.01,
                "unit": "µm",
                "reason": reason,
            }
            for well, lag, distance, reason in groups
        ]
    )


ALIKE = ("A1", 100.0, 500.0, None)
UNLIKE = ("A2", 100.0, 100.0, None)
ONE_FIELD = ("A3", None, None, "only 1 field with observations: the semivariogram needs pairs")


def test_a_redundancy_group_with_no_estimate_is_not_counted_as_negative_evidence():
    both = _redundancy_rows(_redundancy(ALIKE, UNLIKE))[0]["statement"]
    assert both.startswith("in 1 of 2 groups, fields closer than about 500 µm")
    assert "not assessed" not in both
    mixed = _redundancy_rows(_redundancy(ALIKE, UNLIKE, ONE_FIELD))[0]["statement"]
    assert mixed.startswith("in 1 of the 2 groups assessed, fields closer than about 500 µm")
    assert mixed.endswith("; 1 more not assessed (too few fields)")
    none = _redundancy_rows(_redundancy(UNLIKE, ONE_FIELD))[0]["statement"]
    assert none.startswith("in the 1 group assessed, no group shows neighbouring fields")
    assert none.endswith("(100 µm); 1 more not assessed (too few fields)")


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
