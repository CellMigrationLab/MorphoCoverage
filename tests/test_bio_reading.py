"""The report as a cell biologist reads it (owner decisions of 2026-10-01, after a reading of
the four lab reports by a biologist): the verdict says "significant across repeats", the
markers say what was found, the advice names each condition's repeats, the correction
family is counted, and a figure-legend paragraph gives n, the test, q and the change."""

import sys
from pathlib import Path

import pytest

from morphocoverage._style import Status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.legend_text import legend_box
from morphocoverage.workflows._html_render import chip
from morphocoverage.workflows._html_sections import _repeats_advice, what_checked
from morphocoverage.workflows.setup import Setup

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


@pytest.fixture(scope="module")
def glioma():
    data = catalog.load("cct7_cct8_glioma_cdm_tracks")
    contrast = condition_contrast(
        *data.tables, feature="Mean Speed", control=data.control, observation_unit="track"
    )
    return data, contrast


def test_the_advice_names_the_repeats_each_condition_needs(glioma):
    """CCT8 needs about 5 repeats, CCT7 more than 30: one number for all four was wrong."""
    _, contrast = glioma
    advice = _repeats_advice(contrast, "fields")
    assert "siCCT8#5 and siCCT8#8: about 5 biological repeats in all" in advice
    assert "siCCT7#3 and siCCT7#6: more than 30" in advice


def test_the_correction_family_is_counted_per_sub_experiment(glioma):
    data, _ = glioma
    discovery = feature_discovery(
        *data.tables,
        features=list(data.feature_cols),
        control=data.control,
        n_permutations=99,
        observation_unit="track",
    )
    text = " ".join(discovery.summary().split())
    assert "Benjamini-Hochberg within each sub-experiment: 26 tests in each of 2" in text


def test_the_legend_box_gives_n_the_test_the_family_and_the_change(glioma):
    _, contrast = glioma
    box = legend_box(contrast, "tracks")
    assert box.startswith("For a figure legend (Mean Speed): n = 3 independent biological")
    assert "per sub-experiment (6 in all), one well per condition in each" in box
    assert "2 tests in each of 2 sub-experiments, each with its own control" in box
    assert "siCCT8#5: mean 35% lower than the control, d -1.06, q 0.051 (p 0.049)" in box


def test_one_repeat_gets_a_descriptive_legend_and_no_test():
    data = catalog.load("cct8_cdm_morphology")
    objects, fovs = data.tables
    one = condition_contrast(
        objects[objects["biological_repeat"] == "1"],
        fovs[fovs["biological_repeat"] == "1"],
        feature="Area",
        control=data.control,
        observation_unit="cell",
    )
    box = legend_box(one, "cells")
    assert "n = 1 independent biological repeat" in box
    assert "there is no test across repeats: describe the result as one experiment" in box


def test_the_markers_say_what_was_found():
    assert "Not yet" in chip(Status.CAUTION, "enough", "to compare ...: shown in 0 of 1 comparison")
    assert "Partly" in chip(Status.CAUTION, "enough", "shown in 2 of 6 comparisons")
    assert "Enough" in chip(Status.OK, "enough", "shown in 2 of 2 comparisons")
    assert "No conclusion (n = 1)" in chip(
        Status.NOT_ASSESSED, "conditions", "x, 1 condition in one biological repeat (n = 1, so"
    )
    assert "Differs" in chip(Status.OK, "conditions", "significant across repeats: a")
    trend = "Area, ...: same direction in every repeat, not significant: a (lower)"
    assert "Trend only" in chip(Status.CAUTION, "conditions", trend)
    assert "Nothing flagged" in chip(Status.OK, "inspect", "no field stands out")


def test_a_rule_that_does_not_apply_is_not_listed_as_checked():
    setup = Setup(readout="x", features=("x",), observation_unit="track")
    lines = what_checked(setup, ["mean-precision rule, per track, 5%"])
    checked = next(line for line in lines if line.startswith("Stopping rules checked"))
    assert "per track" not in checked
    assert any(line.startswith("Not applicable to tracks: mean-precision rule") for line in lines)


def test_any_feature_significant_across_repeats_answers_yes():
    """CDM: Round, AR, Major and Feret are significant across repeats: the conditions differ.
    The report is neutral: Area, the readout optimise and live target, is not singled out."""
    from morphocoverage.retrospective.report_status import conditions_status
    from morphocoverage.sampling_unit.discovery import feature_discovery

    data = catalog.load("cct8_cdm_morphology")
    contrast = condition_contrast(
        *data.tables, feature="Area", control=data.control, observation_unit="cell"
    )
    discovery = feature_discovery(
        *data.tables,
        features=list(data.feature_cols),
        control=data.control,
        n_permutations=99,
        observation_unit="cell",
    )
    status, headline = conditions_status(contrast, None, discovery)
    assert status is Status.OK
    assert headline.startswith(
        "4 of 10 features significant across repeats (Round, AR, Major and Feret)"
    )
    # Neutral: no feature is singled out as the readout.
    assert "Area" not in headline
    # Without the other features, the readout alone is a trend.
    assert conditions_status(contrast, None)[0] is Status.CAUTION
