"""Tests of ``_control``: the declared control and comparisons, resolved in one place."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _tables as _discovery_tables

from morphocoverage._control import ACROSS, MASTER_RAW, NO_CONTROL, check_contrast_table
from morphocoverage.sampling_unit.contrast_across import across_rows
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._guided_controls import contrast_lines
from morphocoverage.workflows._whole_table import across_reason

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios


def test_a_control_against_the_master_is_tested_on_the_scale_of_its_fold_change() -> None:
    # B17: the vs_master_raw row printed a fold change (x0.88) but tested the differences.
    rows = pd.DataFrame(
        [
            {
                "kind": MASTER_RAW,
                "label": "cB vs cA (raw)",
                "condition": "cB",
                "reference": "cA",
                "condition_control": None,
                "reference_control": None,
                "family": MASTER_RAW,
            }
        ]
    )
    means = {("cA", r): m for r, m in zip("123", (10.0, 100.0, 1000.0), strict=True)}
    means |= {("cB", r): m for r, m in zip("123", (8.0, 90.0, 880.0), strict=True)}
    [row] = across_rows(means, rows, list("123"), 0).to_dict("records")
    expected = np.log([0.8, 0.9, 0.88])
    assert row["values"] == "normalised"
    assert row["effects"] == pytest.approx(expected)
    assert row["fold_change"] == pytest.approx(np.exp(expected.mean()))
    # A signed feature keeps the difference, and says so.
    signed = {key: value - 50 for key, value in means.items()}
    [raw] = across_rows(signed, rows, list("123"), 0).to_dict("records")
    assert raw["values"] == "raw" and np.isnan(raw["fold_change"])
    assert raw["effects"] == pytest.approx([-2.0, -10.0, -120.0])


def test_each_kind_of_row_across_experiments_gives_its_own_reason_for_no_whole_table_test():
    # B16: "a difference of log ratios has no meaning" was said of the master rows too, which
    # compare raw centres.
    loaded = make_scenarios.load("a23_embed2_several_image_tech_fp_dino")
    setup = Setup(**{**loaded.setup, "control_parents": None})
    assert set(setup.contrast_table["kind"]) >= {ACROSS, MASTER_RAW}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(loaded.objects, loaded.fovs, setup, feature_tables=loaded.feature_tables)
    full = " ".join(result.summary("full").split())
    assert "a difference of log ratios has no meaning in a table such as an embedding" in full
    assert "not the controls with the master" in full
    assert across_reason([MASTER_RAW], features=True) == (
        "The rows across experiments of the contrast table are tested feature by feature "
        "only: the whole-table test compares each condition with its own control, not the "
        "controls with the master."
    )
    assert across_reason(["own_control"], features=True) == ""


def test_without_a_control_requested_contrasts_are_said_not_yet_supported():
    objects, fovs = _discovery_tables(n_repeats=3)
    result = report(objects, fovs, Setup(readout="area", observation_unit="cell"))
    said = " ".join(result.summary().split())
    sentence = (
        "not applicable: no control is declared: comparisons are available only for explicitly "
        "requested contrasts estimable from the repeat design, which the report does not "
        "support yet"
    )
    assert sentence in said
    section = " ".join(result.document.sections["conditions"].split())
    assert section.count("explicitly requested contrasts") == 1  # one sentence, its section
    assert "descriptive only" not in said and "described, not tested" not in said
    assert contrast_lines(None, None, None) == [f"{NO_CONTROL[:1].upper()}{NO_CONTROL[1:]}."]


def test_check_contrast_table_refusals():
    with pytest.raises(TypeError) as error:
        check_contrast_table([1])
    assert str(error.value) == "the contrast table must be a pandas DataFrame (CONTRAST_COLUMNS)"
    with pytest.raises(ValueError) as error:
        check_contrast_table(pd.DataFrame({"kind": []}))
    assert str(error.value) == "the contrast table needs the columns ['condition', 'reference']"
    with pytest.raises(ValueError) as error:
        check_contrast_table(pd.DataFrame(columns=["kind", "condition", "reference"]))
    assert str(error.value) == "the contrast table holds no row"
    with pytest.raises(ValueError) as error:
        check_contrast_table(pd.DataFrame({"kind": ["x"], "condition": ["A"], "reference": ["B"]}))
    assert str(error.value) == (
        "contrast kind 'x' is not one of ['own_control', 'vs_master_raw', 'across_normalised']"
    )
    with pytest.raises(ValueError) as error:
        check_contrast_table(
            pd.DataFrame({"kind": ["own_control"], "condition": ["A"], "reference": ["A"]})
        )
    assert str(error.value) == "a contrast compares 'A' with itself"
    twice = pd.DataFrame(
        {
            "kind": ["own_control", "vs_master_raw"],
            "condition": ["A", "B"],
            "reference": ["C", "C"],
            "label": ["same", "same"],
        }
    )
    with pytest.raises(ValueError) as error:
        check_contrast_table(twice)
    assert str(error.value) == "the contrast table holds 'same' twice"
