"""Planning the next experiment from a pilot of one, two or three biological repeats."""

from __future__ import annotations

import sys
import warnings
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _plain

from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_repeats import MIN_REPEATS
from morphocoverage.sampling_unit.repeat_plan import (
    paired_power,
    repeat_plan,
    repeats_needed,
)

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def test_power_grows_with_repeats_and_with_a_smaller_spread():
    low = [paired_power(1.0, 1.0, r, 0.05) for r in range(2, 12)]
    assert all(b >= a for a, b in pairwise(low))
    assert paired_power(1.0, 0.5, 6, 0.05) > paired_power(1.0, 1.0, 6, 0.05)
    # Far in the tail scipy's noncentral t cdf fails: the power stays a number, and right
    # (a simulation of 400 000 pilots gives 0.975; the shifted central t gave 0.99).
    assert paired_power(1.54, 0.538, 7, 0.05 / 12) == pytest.approx(0.975, abs=0.003)
    # Two repeats, a shift of one SD: 0.092 by simulation (the fallback gave 0.20).
    assert paired_power(1.0, 1.0, 2, 0.05) == pytest.approx(0.092, abs=0.003)


def test_the_repeats_needed_follow_the_effect_and_are_bounded():
    assert repeats_needed(2.0, 0.5, 0.05) < repeats_needed(1.0, 0.5, 0.05)
    assert repeats_needed(0.01, 1.0, 0.05) == np.inf
    assert repeats_needed(-2.0, 0.5, 0.05) == repeats_needed(2.0, 0.5, 0.05)


def test_the_repeats_needed_start_at_the_paired_tests_minimum():
    # Audit 10 (C14): two repeats reached the power of a huge effect, but the paired test
    # needs three, so the plan never says 2 (owner, 2026-10-06).
    assert paired_power(50.0, 1.0, 2, 0.05) >= 0.8
    assert repeats_needed(50.0, 1.0, 0.05) == MIN_REPEATS == 3


def _table(d, width, repeats):
    """A per-repeat table planned on Cohen's d: its sampling errors are not finite."""
    return pd.DataFrame(
        {
            "condition": "KD",
            "control_condition": "CTRL",
            "biological_repeat": [f"r{i}" for i in range(repeats)],
            "d_objects": d[:repeats],
            "d_low": [x - width for x in d[:repeats]],
            "d_high": [x + width for x in d[:repeats]],
            "fields": 20,
            "control_fields": 20,
            "mean": 1.0,
            "control_mean": 1.0,
            "log_ratio_se": np.nan,
            "difference_se": np.nan,
            "between_share": 0.0,
        }
    )


@pytest.mark.parametrize("column", ["mean", "control_mean", "log_ratio_se", "difference_se"])
def test_a_table_without_the_paired_tests_values_is_refused_not_planned_on_d(column):
    """Compatibility sweep Q-MC-b: a table without the repeat means or their sampling errors
    was planned on Cohen's d, another statistic than the paired test's; it is refused."""
    table = _table([-0.6, -0.8, -1.0], 0.3, 3)
    with pytest.raises(ValueError, match=f"the contrast table has no {column}: the plan"):
        repeat_plan(table.drop(columns=column))
    # A non-finite sampling error still plans on d: that is the science, not compatibility.
    assert repeat_plan(table).iloc[0]["planned_on"] == "d"


@pytest.mark.parametrize("column", ["d_low", "d_high", "between_share"])
def test_a_table_without_the_interval_of_d_or_its_part_between_wells_is_refused(column):
    """Compatibility sweep, owner decision on #636: without ``d_low`` the plan was silently
    empty, and without ``between_share`` twice the fields halved the whole sampling error;
    both tables are refused. A missing value (no interval in a repeat) is read as before."""
    table = _table([-0.6, -0.8, -1.0], 0.3, 3)
    with pytest.raises(ValueError, match=f"the contrast table has no {column}: the plan"):
        repeat_plan(table.drop(columns=column))
    with pytest.raises(ValueError, match=f"has no {column}"):
        repeat_plan(table.drop(columns=column).iloc[:0])  # refused even with no rows
    gaps = table.assign(between_share=np.nan)
    gaps.loc[2, ["d_low", "d_high"]] = np.nan
    assert repeat_plan(gaps).iloc[0]["repeats"] == 3


def test_more_fields_help_only_when_the_repeats_agree():
    agree = repeat_plan(_table([-0.6, -0.6, -0.6], 0.8, 3)).iloc[0]
    assert agree["between_sd"] == 0
    assert agree["repeats_needed_twice_fields"] < agree["repeats_needed"]
    spread = repeat_plan(_table([-0.2, -1.0, -1.8], 0.2, 3)).iloc[0]
    assert spread["between_sd"] > 0.7
    # The variation between repeats dominates: twice the fields barely change the plan.
    assert spread["repeats_needed"] - spread["repeats_needed_twice_fields"] <= 1


def test_one_repeat_gets_the_fields_but_not_the_variation_between_repeats():
    [row] = repeat_plan(_table([-1.0], 0.5, 1)).to_dict("records")
    assert np.isnan(row["repeats_needed"]) and row["repeats"] == 1
    assert "the variation between repeats is unknown" in row["status"]
    # 20 fields give a half-width of 0.5 for d = -1: about 5 fields would still resolve it.
    assert row["fields_per_repeat"] == 5


def test_the_fields_per_repeat_are_the_most_any_repeat_needs():
    table = _table([-1.0, -0.5, -1.0], 0.5, 3)
    [row] = repeat_plan(table).to_dict("records")
    # Repeat 2 (d -0.5, half-width 0.5) needs all 20; a repeat too small to settle is inf.
    assert row["fields_per_repeat"] == 20
    small = _table([-1.0, -0.05, -1.0], 0.5, 3)
    assert np.isinf(repeat_plan(small).iloc[0]["fields_per_repeat"])


@pytest.fixture(scope="module")
def cdm():
    return catalog.load("cct8_cdm_morphology")


def _contrast(objects, fovs, control):
    return condition_contrast(
        objects, fovs, feature="Area", control=control, observation_unit="cell"
    )


def test_a_pilot_of_two_repeats_gets_a_plan(cdm):
    objects, fovs = cdm.tables
    keep = fovs["biological_repeat"].isin(["1", "2"])
    two = _contrast(objects[objects["biological_repeat"].isin(["1", "2"])], fovs[keep], cdm.control)
    plan = two.plan.set_index("condition")
    assert (plan["repeats"] == 2).all() and (plan["status"] == "ok").all()
    assert (plan["repeats_needed"] >= 2).all()
    text = " ".join(two.summary().split())
    assert "Planning the next experiment on Area (for a new experiment, if the" in text
    assert "corrected over the conditions of its sub-experiment (a follow-up on Area" in text
    # One cell per field here: the advice counts cells, not fields (images); no limiting
    # factor is named, which 2 or 3 repeats cannot tell (owner, 2026-10-02).
    assert "cells a side per repeat" in text and "limited by" not in text
    assert "what limits it" not in text and "add repeats, not" not in text
    full = " ".join(two.summary(detail="full").split())
    assert "limited by" not in full and "which more cells cannot reduce" not in full
    # The repeats a paired test needs lead the plan (the limit is the repeats).
    assert "repeats needed" in text and "with twice the cells" in text


def test_a_pilot_of_one_repeat_is_told_what_a_second_repeat_gives(cdm):
    objects, fovs = cdm.tables
    one = _contrast(
        objects[objects["biological_repeat"] == "1"],
        fovs[fovs["biological_repeat"] == "1"],
        cdm.control,
    )
    text = " ".join(one.summary().split())
    assert "were imaged in one biological repeat: a second repeat shows how much" in text
    assert "cells a side per repeat" in text and "unknown with one repeat" not in text
    assert "⚠️" not in one.summary()


def test_a_repeat_without_an_interval_still_counts_and_is_not_read_as_one_repeat():
    """#305: the plan dropped every repeat without an interval of d (one field a side), so 2
    repeats imaged read as "imaged in one biological repeat" and 3 as 2."""
    from types import SimpleNamespace

    from morphocoverage.sampling_unit.repeat_plan_text import plan_parts

    three = _table([-0.6, -0.8, -1.0], 0.3, 3)
    three.loc[2, ["d_low", "d_high"]] = np.nan
    row = repeat_plan(three).iloc[0]
    assert row["repeats"] == 3 and row["d_mean"] == pytest.approx(-0.8)
    assert row["status"] == "ok" and np.isfinite(row["repeats_needed"])
    two = three.iloc[1:].reset_index(drop=True)
    none = three.assign(condition="KD2", d_low=np.nan, d_high=np.nan)
    plan = repeat_plan(pd.concat([two, none], ignore_index=True)).set_index("condition")
    assert list(plan["repeats"]) == [2, 3]
    assert plan["status"].str.startswith("one biological repeat").sum() == 0
    assert plan["repeats_needed"].isna().all()
    contrast = SimpleNamespace(
        plan=plan.reset_index(),
        table=pd.DataFrame(),
        settings={"feature": "x", "observation_unit": "cell"},
    )
    text = " ".join(" ".join(plan_parts(contrast)[0]).split())
    assert "imaged in one biological repeat" not in text
    assert "KD and KD2: fewer than two of the repeats have two fields a side" in text


def test_power_planning_marks_a_zero_variance_as_degenerate_not_zero_power():
    assert np.isnan(paired_power(0.5, 0.0, 5, 0.05))
    assert np.isnan(repeats_needed(0.5, 0.0, 0.05))
    assert paired_power(1.0, 1.0, 1, 0.05) == 0.0  # no test can run on one repeat
    assert repeats_needed(0.5, 1.0, 0.05) >= 3  # an ordinary case is unchanged


@pytest.mark.parametrize(
    "call",
    [
        lambda: repeats_needed(0.5, 1.0, 5),
        lambda: repeats_needed(0.5, 1.0, 0.05, power=1.5),
        lambda: repeats_needed(np.nan, 1.0, 0.05),
        lambda: repeats_needed(0.5, -1.0, 0.05),
        lambda: repeats_needed("0.5", 1.0, 0.05),
        lambda: repeats_needed(True, 1.0, 0.05),
        lambda: paired_power(0.5, np.inf, 5, 0.05),
        lambda: paired_power(0.5, 1.0, 2.5, 0.05),
        lambda: paired_power(0.5, 1.0, True, 0.05),
    ],
)
def test_malformed_planning_parameters_are_refused(call):
    with pytest.raises(ValueError):
        call()


def _repeat_plan_edges():
    from morphocoverage.sampling_unit.repeat_plan import repeat_plan

    def row(condition, repeat, d, low, high, mean=2.0, se=0.1):
        return {
            "condition": condition,
            "control_condition": "c",
            "biological_repeat": repeat,
            "d_objects": d,
            "d_low": low,
            "d_high": high,
            "fields": 6,
            "control_fields": 6,
            "mean": mean,
            "control_mean": 1.0,
            "log_ratio_se": se,
            "difference_se": se,
            "between_share": 0.2,
        }

    table = pd.DataFrame(
        [
            row("A", "r1", 0.5, np.nan, np.nan),  # no interval anywhere
            row("A", "r2", 0.6, np.nan, np.nan),
            row("B", "r1", 0.5, 0.1, 0.9),  # one interval of two repeats
            row("B", "r2", 0.7, np.nan, np.nan),
            row("C", "r1", 0.5, 0.1, 0.9, mean=2.0, se=0.0),  # identical ratios, no error
            row("C", "r2", 0.5, 0.1, 0.9, mean=2.0, se=0.0),
            row("C", "r3", 0.5, 0.1, 0.9, mean=2.0, se=0.0),
            row("D", "r1", 0.5, 0.1, 0.9, se=np.nan),  # a missing error
            row("D", "r2", np.inf, 0.2, 1.0, se=np.nan),
        ]
    )
    empty = repeat_plan(table.iloc[:0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = repeat_plan(table)
    return (
        list(empty.columns),
        len(empty),
        _plain(plan),
        [(w.category.__name__, str(w.message)) for w in caught],
    )


def _plan_parts_unestimated():
    from morphocoverage.sampling_unit.repeat_plan_text import plan_parts

    plan = pd.DataFrame(
        [
            {
                "condition": "A",
                "control_condition": "c",
                "repeats": 2,
                "d_mean": 0.5,
                "planned_on": "d",
                "change": np.nan,
                "repeats_needed": np.nan,
                "repeats_needed_twice_fields": np.nan,
                "fields_per_repeat": 4.0,
                "status": "1 of 2 repeats with an interval of d",
            },
            {
                "condition": "B",
                "control_condition": "c",
                "repeats": 1,
                "d_mean": 0.4,
                "planned_on": "d",
                "change": np.nan,
                "repeats_needed": np.nan,
                "repeats_needed_twice_fields": np.nan,
                "fields_per_repeat": np.inf,
                "status": "one biological repeat: the variation between repeats is unknown",
            },
        ]
    )
    contrast = SimpleNamespace(
        plan=plan, table=pd.DataFrame(), settings={"feature": "area", "unit": None}
    )
    return plan_parts(contrast, full=True)


EXPECTED_REPEAT_PLAN_EDGES = (
    [
        "condition",
        "control_condition",
        "repeats",
        "d_mean",
        "between_sd",
        "within_se",
        "planned_on",
        "change",
        "fields_per_repeat",
        "limited_by",
        "repeats_needed",
        "repeats_needed_twice_fields",
        "alpha_planned",
        "experiment",
        "status",
    ],
    0,
    [
        {
            "condition": "A",
            "control_condition": "c",
            "repeats": 2,
            "d_mean": 0.55,
            "between_sd": None,
            "within_se": None,
            "planned_on": "d",
            "change": None,
            "fields_per_repeat": None,
            "limited_by": "",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": (
                "0 of 2 repeats with an interval of d (two fields a side): the "
                "variation between repeats is not estimated"
            ),
        },
        {
            "condition": "B",
            "control_condition": "c",
            "repeats": 2,
            "d_mean": 0.6,
            "between_sd": None,
            "within_se": 0.2040853827698616,
            "planned_on": "log ratio",
            "change": 2.0,
            "fields_per_repeat": 4.0,
            "limited_by": "",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": (
                "1 of 2 repeats with an interval of d (two fields a side): the "
                "variation between repeats is not estimated"
            ),
        },
        {
            "condition": "C",
            "control_condition": "c",
            "repeats": 3,
            "d_mean": 0.5,
            "between_sd": 0.0,
            "within_se": 0.2040853827698616,
            "planned_on": "log ratio",
            "change": 2.0,
            "fields_per_repeat": 4.0,
            "limited_by": "within repeats",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": (
                "the estimated variance of the repeat effects is zero: the plan is not "
                "defined there, which is not evidence of no power"
            ),
        },
        {
            "condition": "D",
            "control_condition": "c",
            "repeats": 2,
            "d_mean": np.inf,
            "between_sd": 0.0,
            "within_se": 0.2040853827698616,
            "planned_on": "d",
            "change": None,
            "fields_per_repeat": 4.0,
            "limited_by": "within repeats",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": ("a value the plan needs is missing, so the repeats needed are not planned"),
        },
    ],
    [
        ("RuntimeWarning", "invalid value encountered in subtract"),
        ("RuntimeWarning", "invalid value encountered in subtract"),
    ],
)


EXPECTED_PLAN_PARTS_UNESTIMATED = (
    [
        (
            "  Planning the next experiment on area (for a new experiment, if the "
            "difference and the variation\n    seen here recur; not what the "
            "current result needs): the biological repeats a paired t-test on\n    "
            "the repeat means needs for 80% power, corrected over the conditions "
            "of its sub-experiment (a\n    follow-up on area alone), with the "
            "fields per repeat as now and with twice as many; then the\n    fields "
            "a side each repeat needs to show the difference on its own:"
        ),
        (
            "    condition  change across repeats  mean d per repeat (cells)  "
            "repeats needed  with twice the fields  fields a side per repeat\n    "
            "A vs c     –                      +0.50                      –        "
            "       –                      about 4\n    B vs c     –               "
            "       +0.40                      –               –                   "
            "   more than 4 times the fields"
        ),
        (
            "  A: fewer than two of the repeats have two fields a side or more, so "
            "the variation between repeats\n    is not estimated."
        ),
        (
            "  B was imaged in one biological repeat: a second repeat shows how "
            "much the difference varies\n    between repeats."
        ),
    ],
    [
        (
            "Repeats needed: for a paired t-test at p ≤ 0.05/m, m the conditions "
            "of the sub-experiment (what Benjamini-Hochberg asks at most), on the "
            "values that test compares: each repeat's change (its mean over its "
            "control's, a fold change; the difference of the means for a feature "
            "with a value that is not positive; the table gives their geometric "
            "mean, as the verdict does), its spread between the repeats and its "
            "sampling error within one. All projected from the repeats here, so "
            "the plan is a guide."
        ),
        (
            "The fields a side per repeat are the most any repeat needs, projected "
            "from its interval of d (narrowing as 1/sqrt of the number)."
        ),
    ],
)


def test_repeat_plan_edges():
    assert _repeat_plan_edges() == EXPECTED_REPEAT_PLAN_EDGES


def test_plan_parts_unestimated():
    assert _plan_parts_unestimated() == EXPECTED_PLAN_PARTS_UNESTIMATED
