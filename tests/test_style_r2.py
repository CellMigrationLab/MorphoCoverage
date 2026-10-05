"""Readable reports, questions 4, 5 and 7: short and full summaries, markers and headlines.

The owner's decisions (2026-09-28, BRIEF of the readable-reports change): a
short default with per-group numbers in tables and every caveat once, every
number of the old text kept in ``detail="full"``, and the markers ✅ ⚠️ ❌ ➖
with fixed meanings. These tests hold ``sampling_reliability``,
``protocol_reliability``, ``adequacy_points`` and ``effort_projection``
summaries, and the report's questions 4, 5 and 7, to those rules.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage._style import CAUTION, OK, PROBLEM, Status
from morphocoverage._wording import percent
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    AdequacyResult,
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    adequacy_points,
    effort_projection,
    microscopist_report,
    protocol_reliability,
    replay_acquisitions,
    sampling_reliability,
)
from morphocoverage.retrospective.report_status import (
    bought_status,
    effort_status,
    next_time_status,
)

MEAN_A = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")
MEAN_B = OutcomeCriterion(RelativeMeanError("feature_b"), 0.05, "lower_is_better", "mean b")
HIERARCHY = SamplingHierarchy("image_file", "well", "plate")


def _flat(text):
    return " ".join(text.split())


def _reliability(criteria=(MEAN_A,)):
    built = SCENARIOS["condition_repeat_groups"]()
    return sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=list(criteria),
        fov_counts=[2, 4],
        reservoir_fraction=0.3,
        n_splits=3,
        draws=5,
    )


def _curve(preserved, stream=12, labels=None):
    """A reliability result from {(plate, well): {k: preserved}} (one criterion, "mean")."""
    rows = [
        {
            "condition": "ctrl",
            "plate": plate,
            "well": well,
            "criterion": "mean",
            "fovs": k,
            "preserved": value,
            "stream_fovs": stream,
        }
        for (plate, well), by_k in preserved.items()
        for k, value in by_k.items()
    ]
    settings = {"labels": labels or {"mean": "the mean"}}
    return ReliabilityResult(
        pd.DataFrame(rows), pd.DataFrame(), ("condition", "plate", "well"), settings
    )


def _protocol(preserved, plates=("P1", "P2")):
    return protocol_reliability(_curve(preserved), HIERARCHY, threshold=0.9, strata=["condition"])


def _wells(plates, value_5, value_10):
    return {(p, w): {5: value_5, 10: value_10} for p in plates for w in ("A1", "A2")}


# --- detail -------------------------------------------------------------------------


def test_every_summary_refuses_an_unknown_detail():
    reliability = _reliability()
    protocol = _protocol(_wells(("P1", "P2"), 0.95, 0.99))
    for result in (reliability, protocol):
        with pytest.raises(ValueError, match="short"):
            result.summary(detail="brief")
        assert result.summary() == result.summary(detail="short")
    with pytest.raises(ValueError, match="full"):
        AdequacyResult(pd.DataFrame(), pd.DataFrame(), ()).summary(detail="all")


# --- 4. How many fields next time? --------------------------------------------------


def test_reliability_short_is_one_table_across_groups_and_full_keeps_every_share():
    result = _reliability()
    short, full = result.summary(), result.summary(detail="full")
    groups = result.curve[list(result.group_cols)].drop_duplicates()
    assert f"the range over {len(groups)} groups" in _flat(short)
    assert "condition=" not in short and "condition=" not in full
    # Every caveat once in the short form, and still in the full one.
    for caveat in ("random draws", "conservative by construction", "cannot be read above"):
        assert _flat(short).count(caveat) == 1
        assert caveat in _flat(full)
    # They hold for every curve: standing assumptions, listed once, never marked.
    standing = _flat(short.split("Assumptions:")[1])
    for caveat in ("random draws", "conservative by construction", "cannot be read above"):
        assert caveat in standing
    assert short.count("Assumptions:") == 1 and full.count("Assumptions:") == 1
    # Every share of the curve, and every Monte Carlo SE, is printed in the full form.
    for row in result.curve.itertuples(index=False):
        assert percent(row.preserved) in full
    assert "Monte Carlo SE" in full and "Judged on" in full


def test_protocol_short_answers_per_target_and_states_few_units_once():
    wells = {**_wells(("P1",), 0.95, 0.99), **_wells(("P2",), 0.5, 0.95)}
    result = _protocol(wells)
    short, full = result.summary(), result.summary(detail="full")
    assert "all served from" in short and "10 fields" in short
    assert _flat(short).count("With 2 plates the field count is a guide") == 1
    assert _flat(full).count("With 2 plates the field count is a guide") == 1
    # The full form keeps every count of samples served, per unit, at every depth.
    assert "2 of 4 samples" in full and "4 of 4 samples" in full
    assert "2 of 2, 0 of 2" in full
    assert "Fields needed per sample: median" in full


def test_next_time_markers_follow_their_rules():
    many = ("P1", "P2", "P3", "P4", "P5")
    served = _protocol(_wells(many, 0.95, 0.99))
    assert next_time_status(None, served)[0] is Status.OK
    assert "5 fields served every sample for the target" in next_time_status(None, served)[1]
    few = next_time_status(None, _protocol(_wells(("P1", "P2"), 0.95, 0.99)))
    # Few repeats is a usual design: stated in the line, not flagged.
    assert few[0] is Status.OK and "from 2 plates" in few[1]
    # Never served, for lack of fields (not capped by the held-back fields): a problem.
    never = next_time_status(None, _protocol(_wells(many, 0.5, 0.6)))
    assert never[0] is Status.PROBLEM and "not reached" in never[1]
    # Reliability alone is per group only: never ✅, and the headline says why.
    alone = next_time_status(_curve(_wells(many, 0.95, 0.99)), None)
    assert alone[0] is Status.CAUTION and "not checked across samples" in alone[1]
    none = next_time_status(_curve(_wells(many, 0.5, 0.6)), None)
    assert none[0] is Status.PROBLEM


def test_a_group_skipped_by_the_reliability_curve_is_counted_apart():
    """#272: two wells assessed and one skipped (one field) are not "both groups"."""
    curve = _curve(_wells(("P1",), 0.95, 0.99))
    assert "enough in both groups" in next_time_status(curve, None)[1]
    skipped = pd.DataFrame(
        [{"condition": "ctrl", "plate": "P2", "well": "A1", "fovs": 1, "reason": "only 1 field"}]
    )
    curve = replace(curve, skipped=skipped)
    status, text = next_time_status(curve, None)
    assert status is Status.CAUTION and "both groups" not in text
    assert "enough in the 2 groups assessed for the target; 1 more not assessed" in text


# --- 5. What did the extra fields buy? ----------------------------------------------


def _adequacy():
    built = SCENARIOS["condition_repeat_groups"]()
    replay = replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(3), FixedFOVStrategy(1000)],
        criteria=[MEAN_A, MEAN_B],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        reservoir_fraction=0.3,
        n_splits=2,
    )
    return adequacy_points(replay)


def test_adequacy_short_tables_the_rules_and_full_keeps_every_sentence():
    result = _adequacy()
    short, full = result.summary(), result.summary(detail="full")
    assert "fixed 3 fields" in short and "fixed 1000 fields" in short
    assert "never" in short  # a rule waiting for 1000 fields never says stop
    assert "Each target on its own" in short and "mean b" in short
    assert "condition=" not in short and "condition=" not in full
    assert _flat(short).count("'Reliably' means") == 1
    # The full form: every group's sentence with its medians and tail, and the config ids.
    groups = result.realized[list(result.group_cols)].drop_duplicates()
    assert full.count("when was it enough?") == len(groups)
    assert "in 90% of runs" in full and "config " in full


def test_bought_marker_reads_the_confirmed_runs():
    result = _adequacy()
    status, text = bought_status(result)
    assert status in (Status.OK, Status.CAUTION)
    assert text.startswith("held-back fields")
    never = result.realized.assign(sustained_realized=pd.array([None] * len(result.realized)))
    lost = AdequacyResult(result.apparent, never, result.group_cols, result.settings)
    assert bought_status(lost)[0] is Status.CAUTION


# --- 7. More fields or more samples? ------------------------------------------------


def _effort(repeats=3):
    rng = np.random.default_rng(6)
    rows, fovs = [], []
    for r in range(repeats):
        for s in range(3):
            for f in range(5):
                name = f"r{r}_s{s}_f{f}"
                fovs.append(
                    {
                        "condition": "ctrl",
                        "biological_repeat": f"R{r}",
                        "well": f"W{s}",
                        "image_file": name,
                    }
                )
                values = rng.normal(r * 0.5 + s * 0.3, 1.0, 15)
                rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fov_table = pd.DataFrame(fovs)
    objects = pd.concat(rows, ignore_index=True).merge(fov_table, on="image_file")
    return effort_projection(
        objects,
        fov_table,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat"),
        strata=["condition"],
        n_boot=50,
    )


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


# --- The report -----------------------------------------------------------------------


def test_the_report_gives_questions_4_5_and_7_short_with_their_markers():
    reliability = _reliability()
    protocol = _protocol(_wells(("P1", "P2"), 0.95, 0.99))
    report = microscopist_report(
        reliability=reliability, protocol=protocol, adequacy=_adequacy(), effort=_effort()
    )
    for key in ("next_time", "bought", "effort"):
        assert key in report.short and report.short[key] != report.sections[key]
        assert all(line.startswith("  ") or not line for line in report.short[key].splitlines())
    short = report.summary()
    assert f"{Status.OK.marker} 4. How many fields next time?" in short
    assert f"{Status.OK.marker} 7. More fields or more samples?" in short
    assert report.statuses["bought"] in (Status.OK, Status.CAUTION)
    full = report.summary(detail="full")
    for text in (reliability.summary(detail="full"), protocol.summary(detail="full")):
        assert _flat(text.splitlines()[1]) in _flat(full)
    # Each caveat of question 4 once in the short form.
    section = _flat(report.short["next_time"])
    assert section.count("conservative by construction") == 1
    assert section.count("the field count is a guide") == 1
    assert OK not in report.short["next_time"] and PROBLEM not in report.short["next_time"]
