"""Question 9 read condition by condition: one verdict and its evidence (``reading_text``)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.reading_text import (
    CONFIRMED,
    CONSISTENT,
    MIXED,
    ONE_REPEAT,
    condition_readings,
    readings_headline,
)

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def _contrast(repeats, cells=None, targets=None):
    return SimpleNamespace(
        table=pd.DataFrame(),
        settings={"repeat_col": "biological_repeat", "observation_unit": "cell"},
        repeats=pd.DataFrame(repeats),
        cells=pd.DataFrame(cells or []),
        targets=pd.DataFrame(targets or []),
    )


def _row(condition, lower, higher, p_raw, p_norm, n=3):
    return {
        "condition": condition,
        "control_condition": "CTRL",
        "repeats": n,
        "lower": lower,
        "higher": higher,
        "d_mean": -1.0 if lower else 1.0,
        "q_paired_t": p_raw,
        "q_paired_t_normalised": p_norm,
    }


def test_each_verdict_follows_the_direction_and_the_paired_test():
    contrast = _contrast(
        [
            _row("A", 3, 0, 0.03, 0.01),
            _row("B", 0, 3, 0.4, 0.3),
            _row("C", 1, 2, 0.9, 0.9),
            _row("D", 2, 0, np.nan, np.nan, n=2),
        ]
    )
    readings = condition_readings(contrast).set_index("condition")
    assert readings.loc["A", "verdict"] == CONFIRMED
    assert readings.loc["B", "verdict"] == CONSISTENT and readings.loc["B", "direction"] == "higher"
    assert readings.loc["C", "verdict"] == MIXED
    # Two repeats: the same direction, but the paired test cannot run.
    assert readings.loc["D", "verdict"] == CONSISTENT
    assert "the paired test needs 3 biological repeats" in readings.loc["D", "reading"]
    assert "paired t-test on the repeat means q 0.010" in readings.loc["A", "reading"]


def test_the_normalised_paired_test_is_read_and_a_disagreeing_raw_one_named():
    readings = condition_readings(_contrast([_row("A", 3, 0, 0.2, 0.01)]))
    [text] = readings["reading"]
    assert readings["verdict"].iloc[0] == CONFIRMED
    assert "paired t-test on the repeat means q 0.010; on raw values q 0.20" in text


def test_one_repeat_is_its_own_verdict():
    readings = condition_readings(_contrast([_row("A", 1, 0, np.nan, np.nan, n=1)]))
    assert readings["verdict"].iloc[0] == ONE_REPEAT
    assert "lower than the control in its one biological repeat" in readings["reading"].iloc[0]


@pytest.fixture(scope="module")
def glioma():
    data = catalog.load("cct7_cct8_glioma_cdm_tracks")
    return condition_contrast(
        *data.tables, feature="Mean Speed", control=data.control, observation_unit="track"
    )


def test_the_glioma_reading_names_why_the_raw_test_differs(glioma):
    """CCT7 and CCT8 were imaged in separate repeats: the raw pooled test is significant for
    CCT7 only because it pools the CCT8 repeats' control; balanced and normalised, it is not."""
    readings = condition_readings(glioma).set_index("condition")
    assert readings.loc["siCCT7#3", "verdict"] == MIXED
    text = readings.loc["siCCT7#3", "reading"]
    assert "the raw pooled test gives q < 0.001, but it pools the control of repeats" in text
    assert "(balanced, normalised; with no real effect it gives p ≤ 0.05 in about" in text
    assert readings.loc["siCCT8#5", "verdict"] == CONSISTENT
    assert (
        "siCCT8#8 (CCT8) agrees in 3 of 3 biological repeats" in readings.loc["siCCT8#5", "reading"]
    )
    _, headline = conditions_status(glioma, None)
    # Nothing confirmed for the readout: said first, as the caution.
    assert headline.startswith(
        "Mean Speed: none confirmed across repeats; 4 conditions against the control, each "
        "in 3 biological repeats (6 in all): same direction in every repeat, not "
        "confirmed: siCCT8#5 and siCCT8#8 (lower); no consistent direction: siCCT7#3 and "
        "siCCT7#6"
    )
    short = " ".join(glioma.summary().split())
    assert short.startswith(
        "Design: 2 sub-experiments, each with its own siCTRL in its own 3 biological repeats "
        "(siCCT7#3 and siCCT7#6; siCCT8#5 and siCCT8#8). Reading, condition by condition"
    )
    # The control shift between the CCT7 and CCT8 repeats is named where it matters.
    assert "The control means differ up to 1.8x between biological repeats" in short


def test_a_pilot_headline_counts_the_cell_level_test():
    data = catalog.load("cct8_cdm_morphology")
    objects, fovs = data.tables
    one = condition_contrast(
        objects[objects["biological_repeat"] == "1"],
        fovs[fovs["biological_repeat"] == "1"],
        feature="Area",
        control=data.control,
        observation_unit="cell",
    )
    headline = readings_headline(one)
    assert headline.startswith("2 conditions across 1 biological repeat: one repeat:")
    assert (
        "; cell-level test: 0 of 2 at q ≤ 0.05 (with no real effect it gives p ≤ 0.05 in"
        in headline
    )
