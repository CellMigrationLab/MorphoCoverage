"""Planning the next experiment from a pilot of one, two or three biological repeats."""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit.contrast import condition_contrast
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
    # Far in the tail scipy's noncentral t can fail: the power stays a number.
    assert paired_power(1.54, 0.538, 7, 0.05 / 12) == pytest.approx(0.99, abs=0.01)


def test_the_repeats_needed_follow_the_effect_and_are_bounded():
    assert repeats_needed(2.0, 0.5, 0.05) < repeats_needed(1.0, 0.5, 0.05)
    assert repeats_needed(0.01, 1.0, 0.05) == np.inf
    assert repeats_needed(-2.0, 0.5, 0.05) == repeats_needed(2.0, 0.5, 0.05)


def _table(d, width, repeats):
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
        }
    )


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
    assert "Planning the next experiment: the biological repeats a paired t-test" in text
    # One cell per field here: the advice counts cells, not fields (images).
    assert "cells a side per repeat" in text and "limited by" in text
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
    assert "cells a side per repeat" in text and "unknown with one repeat" in text
    assert "⚠️" not in one.summary()
