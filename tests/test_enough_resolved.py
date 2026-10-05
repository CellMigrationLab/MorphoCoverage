"""Question 2 with a control: do the fields acquired resolve the difference from the control?"""

import sys
from pathlib import Path

import numpy as np
import pytest

from morphocoverage._style import CAUTION, Status
from morphocoverage.retrospective.report_enough import enough_short
from morphocoverage.retrospective.report_status import enough_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_sampling import PRACTICAL, sampling_row
from morphocoverage.statistics import cohen_d

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog
from test_sampling_unit import _technical, _two_conditions


def _fields(rng, n_fields, mean, per_field=5):
    return [[rng.normal(mean, 1.0, per_field) for _ in range(n_fields)]]


def test_a_large_difference_is_resolved_and_none_is_too_small():
    rng = np.random.default_rng(0)
    t, c = _fields(rng, 20, 2.0), _fields(rng, 20, 0.0)
    d = float(np.mean(np.concatenate(t[0])) - np.mean(np.concatenate(c[0])))
    big = sampling_row(t, c, d, np.random.default_rng(1))
    assert big["resolved"] and big["d_low"] > 0 and np.isnan(big["fields_to_resolve"])
    # No difference: the interval spans 0, and the projected fields are far beyond practical.
    same = sampling_row(t, _fields(rng, 20, 2.0), 0.01, np.random.default_rng(1))
    assert not same["resolved"] and same["too_small"]
    assert same["fields_to_resolve"] > PRACTICAL * 20


def test_a_moderate_difference_gets_the_fields_that_would_resolve_it():
    rng = np.random.default_rng(3)
    t, c = _fields(rng, 10, 0.8, per_field=1), _fields(rng, 10, 0.0, per_field=1)
    d = cohen_d(np.concatenate(t[0]), np.concatenate(c[0]))  # 0.65 in this draw
    row = sampling_row(t, c, d, np.random.default_rng(2))
    assert not row["resolved"] and not row["too_small"]
    assert 10 < row["fields_to_resolve"] <= PRACTICAL * 10


def test_the_fields_to_resolve_count_the_fields_without_observations():
    """The projection counted only the fields with observations, so a side where half the
    fields acquired were empty was told half the fields it would acquire (#289)."""
    rng = np.random.default_rng(3)
    t, c = _fields(rng, 10, 0.8, per_field=1), _fields(rng, 10, 0.0, per_field=1)
    d = cohen_d(np.concatenate(t[0]), np.concatenate(c[0]))
    row = sampling_row(t, c, d, np.random.default_rng(2))
    empty = [np.empty(0)] * 10
    halved = sampling_row([t[0] + empty], [c[0] + empty], d, np.random.default_rng(2))
    assert halved["d_low"] == row["d_low"] and halved["d_high"] == row["d_high"]
    assert halved["fields_to_resolve"] == pytest.approx(2 * row["fields_to_resolve"], abs=1)
    assert halved["too_small"] == row["too_small"]


def test_too_few_fields_give_no_answer():
    rng = np.random.default_rng(0)
    row = sampling_row(_fields(rng, 1, 2.0), _fields(rng, 5, 0.0), 2.0, rng)
    assert np.isnan(row["d_low"]) and not row["resolved"]


def test_question_2_leads_with_the_resolution_when_a_control_is_declared():
    data = catalog.load("cct8_cdm_morphology")
    contrast = condition_contrast(
        *data.tables, feature="Area", control=data.control, observation_unit="cell"
    )
    status, headline = enough_status(None, None, contrast)
    assert status is Status.CAUTION
    assert headline.startswith(
        "to compare with the control, within single repeats: shown in 2 of 6 comparisons"
    )
    # One cell per field: the advice counts cells.
    assert "would need about" in headline and "cells a side" in headline
    short = " ".join(enough_short(None, None, None, contrast=contrast).split())
    assert "Within each repeat, does the difference from the control show?" in short
    assert "about 42 cells a side" in short


def test_every_difference_resolved_is_ok_and_the_precision_cautions_stay_in_the_full_form():
    objects, fovs = _technical(*_two_conditions(shift=2.0, field_sd=0.3, n_wells=1))
    contrast = condition_contrast(
        objects, fovs, feature="x", control={"condition": "control"}, n_draws=5
    )
    assert contrast.table["resolved"].all()
    status, headline = enough_status(None, None, contrast)
    assert status is Status.OK
    assert headline == (
        "to compare with the control, within single repeats: shown in 2 of 2 comparisons"
    )
    assert CAUTION not in enough_short(None, None, None, contrast=contrast)


@pytest.mark.parametrize("dataset_id", ["cct8_filopodia_screen"])
def test_the_screens_hits_are_resolved_in_every_repeat(dataset_id):
    data = catalog.load(dataset_id)
    contrast = condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )
    table = contrast.table
    hits = table["condition"].str.startswith(("siCCT8", "siCSNK1A1", "siSCRIB"))
    assert table.loc[hits, "resolved"].all()
