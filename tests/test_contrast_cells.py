"""The cell-level test (Plot&Stats) with its balance check (question 9, ``contrast_cells``)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_cells import (
    ALPHA,
    MIN_BALANCED,
    cell_randomization,
    cell_table,
    chance_without_effect,
    effective_per_well,
    normalised,
    sparse,
)
from morphocoverage.sampling_unit.contrast_repeats import across_repeats

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def test_the_chance_without_an_effect_matches_the_simulation():
    """No variation between repeats: the nominal 5%; 30 cells a well at s = 0.25 and 0.5: the
    15-17% and 32-36% of data/simulations/test_choice.py (normalised to the control)."""
    assert chance_without_effect(30, 0.0) == pytest.approx(ALPHA)
    assert 0.14 < chance_without_effect(30, 0.25) < 0.18
    assert 0.33 < chance_without_effect(30, 0.5) < 0.40
    # More cells a well make it worse, not better.
    assert chance_without_effect(100, 0.25) > chance_without_effect(30, 0.25)


def test_the_randomization_test_is_on_cohens_d_over_the_observations():
    rng = np.random.default_rng(0)
    same = np.arange(20.0)
    d, p = cell_randomization(same, same.copy(), 199, rng)
    assert d == pytest.approx(0.0) and p == pytest.approx(1.0)
    d, p = cell_randomization(same + 100, same, 199, rng)
    assert d > 5 and p == pytest.approx(1 / 200)


def test_a_repeat_too_small_to_balance_to_is_named():
    assert sparse({"1": 100, "2": 100, "3": 20}) == ["3"]
    assert sparse({"1": 100, "2": 30}) == []
    assert sparse({"1": 6, "2": MIN_BALANCED - 1}) == ["2"]
    # One repeat holding many more leaves the others ordinary: against the median, not the
    # largest (data/simulations/unbalanced.py).
    assert sparse({"1": 240, "2": 30, "3": 30}) == []


def _side(treated, control, other_control_repeats=0):
    """A condition's side: (cells, shift) per repeat it was imaged in, and the control's cells
    per repeat, plus control repeats it was not imaged in (30 cells each)."""
    rng = np.random.default_rng(1)
    by_t = {r: 10 + rng.normal(shift, 1, n) for r, (n, shift) in treated.items()}
    by_c = {r: 10 + rng.normal(0, 1, n) for r, n in control.items()}
    others = {f"other{i}": 10 + rng.normal(0, 1, 30) for i in range(other_control_repeats)}
    return {
        "treated_by_repeat": by_t,
        "control_by_repeat": by_c,
        "control_all": {**by_c, **others},
        "control_condition": "CTRL",
    }


def _cells(**sides):
    rng_for = lambda condition, values: np.random.default_rng(0)
    return cell_table(sides, pd.DataFrame(), 199, rng_for).set_index("condition")


def test_balanced_repeats_are_tested_as_provided_only():
    even = _side({"1": (30, 0.5), "2": (32, 0.5), "3": (28, 0.5)}, {"1": 30, "2": 30, "3": 31})
    row = _cells(KD=even).loc["KD"]
    assert not row["unbalanced"] and row["spread"] < 1.25
    assert np.isnan(row["p_balanced"]) and row["p_cells"] < 0.05
    # One repeat: the variation between repeats is unknown, so the chance is a range.
    assert row["chance_low"] < row["chance_high"]


def test_one_repeat_holding_more_than_a_quarter_more_is_balanced():
    """The effect sits in the repeat holding most cells: pooled, it dominates d; balanced,
    every repeat weighs the same."""
    uneven = _side({"1": (120, 1.0), "2": (40, 0.0), "3": (40, 0.0)}, {"1": 40, "2": 40, "3": 40})
    row = _cells(KD=uneven).loc["KD"]
    assert row["unbalanced"] and row["spread"] == pytest.approx(3.0)
    assert (row["balanced_cells"], row["balanced_control_cells"]) == (40, 40)
    assert row["d_cells"] > 0.6 and 0.1 < row["d_balanced"] < row["d_cells"] - 0.2


def test_a_control_from_repeats_the_condition_lacks_is_unbalanced():
    others = _side({"1": (30, 0.5), "2": (30, 0.5), "3": (30, 0.5)}, {"1": 30, "2": 30, "3": 30}, 3)
    row = _cells(KD=others).loc["KD"]
    assert row["unbalanced"] and row["other_control_repeats"] == 3
    assert row["p_balanced"] > 0 and row["p_balanced_bonferroni"] == row["p_balanced"]


def test_a_sparse_repeat_is_flagged_and_the_others_are_balanced():
    """Repeat 3 holds 8 cells against 100: balancing down to it would keep 8 of every
    repeat, so it is named and left out of the balanced test, which balances repeats 1-2."""
    tiny = _side({"1": (100, 0.5), "2": (60, 0.5), "3": (8, 0.5)}, {"1": 90, "2": 90, "3": 90})
    row = _cells(KD=tiny).loc["KD"]
    assert row["unbalanced"] and row["sparse_repeats"] == ["3"]
    assert row["balanced_repeats"] == 2 and row["balanced_cells"] == 60
    assert row["p_balanced"] < 0.05 and row["p_balanced_normalised"] < 0.05
    # The test as provided keeps every cell, the sparse repeat's too.
    assert row["observations"] == 168 and row["p_cells"] < 0.05


def test_a_sparse_control_repeat_is_left_out_of_both_sides():
    tiny = _side({"1": (40, 0.5), "2": (40, 0.5), "3": (40, 0.5)}, {"1": 40, "2": 40, "3": 4})
    row = _cells(KD=tiny).loc["KD"]
    assert row["sparse_repeats"] == ["3"] and row["balanced_repeats"] == 2


def test_with_every_repeat_sparse_nothing_is_balanced():
    tiny = _side({"1": (4, 0.5), "2": (3, 0.5)}, {"1": 4, "2": 3}, 1)
    row = _cells(KD=tiny).loc["KD"]
    assert row["unbalanced"] and row["balanced_repeats"] == 0 and np.isnan(row["p_balanced"])


@pytest.fixture(scope="module")
def screen():
    data = catalog.load("cct8_filopodia_screen")
    return condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )


def test_the_screen_is_given_as_provided_and_balanced(screen):
    """Each condition's control is pooled from 18 repeats against its own 3: MAP4K4#5 reads
    d -0.10 pooled, and above its own repeats' control once balanced, as in every repeat."""
    cells = screen.cells.set_index("condition")
    assert cells["unbalanced"].all() and (cells["other_control_repeats"] == 15).all()
    assert cells.loc["siMAP4K4#5", "d_cells"] < 0 < cells.loc["siMAP4K4#5", "d_balanced"]
    assert (screen.repeats.set_index("condition").loc["siMAP4K4#5", "higher"]) == 3
    joined = " ".join(screen.summary().split())
    assert "Cell-level test (as in Plot&Stats)" in joined
    assert "the control is pooled from 15 biological repeats the condition was not imaged in" in (
        joined
    )
    assert "p ≤ 0.05 without an effect" in joined and "⚠️" not in screen.summary()
    status, headline = conditions_status(screen, None)
    assert status.name == "OK"
    assert headline.endswith(
        "cell-level test, raw / normalised: 9 / 12 of 12 at Bonferroni p ≤ 0.05 (balanced over "
        "the repeats: 10 / 11)"
    )
    # Normalised to its own repeats' control, MAP4K4#5 is above it as in every repeat.
    assert cells.loc["siMAP4K4#5", "d_cells_normalised"] > 0


def test_a_pilot_of_one_repeat_leads_with_the_cell_level_test():
    data = catalog.load("cct8_cdm_morphology")
    objects, fovs = data.tables
    one = condition_contrast(
        objects[objects["biological_repeat"] == "1"],
        fovs[fovs["biological_repeat"] == "1"],
        feature="Area",
        control=data.control,
        observation_unit="cell",
    )
    text = " ".join(one.summary().split())
    assert text.startswith("Cell-level test (as in Plot&Stats)") and "Balance" not in text
    assert (one.cells["chance_low"] < one.cells["chance_high"]).all()
    assert "with one repeat s is unknown and taken as 0.25–0.5" in text
    assert "⚠️" not in one.summary()


def test_normalising_divides_by_the_own_repeats_control_mean():
    values = {"1": np.array([2.0, 4.0]), "2": np.array([30.0])}
    controls = {"1": np.array([1.0, 3.0]), "2": np.array([10.0, 20.0])}
    scaled = normalised(values, controls)
    assert np.allclose(scaled["1"], [1.0, 2.0]) and np.allclose(scaled["2"], [2.0])
    # No fold change from a control mean at or below 0.
    assert normalised(values, {**controls, "2": np.array([-1.0, 1.0])}) is None


def test_a_batch_scale_shared_by_both_wells_is_removed_by_normalising():
    """Repeat 2 is imaged twice as bright and holds most cells: pooled raw values mix the
    scales, the normalised values do not."""
    rng = np.random.default_rng(3)
    by_t = {"1": 10 + rng.normal(0, 1, 30), "2": 2 * (10 + rng.normal(0, 1, 30))}
    by_c = {"1": 10 + rng.normal(0, 1, 30), "2": 2 * (10 + rng.normal(0, 1, 30))}
    side = {
        "treated_by_repeat": by_t,
        "control_by_repeat": by_c,
        "control_all": {**by_c, "3": 2 * (10 + rng.normal(0, 1, 90))},
        "control_condition": "CTRL",
    }
    row = _cells(KD=side).loc["KD"]
    assert row["d_cells"] < -0.4 and abs(row["d_cells_normalised"]) < 0.3


def test_the_paired_t_on_normalised_means_is_on_log_ratios():
    table = pd.DataFrame(
        {
            "condition": "KD",
            "control_condition": "CTRL",
            "mean": [5.0, 10.0, 20.0],
            "control_mean": [10.0, 20.0, 41.0],
            "d_objects": [-1.0, -1.0, -1.0],
        }
    )
    [row] = across_repeats(table, "biological_repeat", 0).to_dict("records")
    # Ratios of about 0.5 in every repeat: the log ratios hardly vary, so p is tiny; the raw
    # differences (-5, -10, -21) vary a lot.
    assert row["p_paired_t_normalised"] < 0.001 < row["p_paired_t"]
    signed = table.assign(control_mean=[10.0, -20.0, 41.0])
    [row] = across_repeats(signed, "biological_repeat", 0).to_dict("records")
    assert np.isnan(row["p_paired_t_normalised"]) and row["p_paired_t"] > 0


def test_the_report_names_a_very_small_repeat_once_for_its_conditions():
    data = catalog.load("cct7_cct8_glioma_cdm_tracks")
    objects, fovs = data.tables
    small = objects["biological_repeat"] == "CCT8_3"
    cut = objects[small].groupby("condition", group_keys=False).head(5)
    contrast = condition_contrast(
        pd.concat([objects[~small], cut]),
        fovs,
        feature="Mean Speed",
        control=data.control,
        observation_unit="track",
    )
    cells = contrast.cells.set_index("condition")
    assert cells.loc["siCCT8#5", "sparse_repeats"] == ["CCT8_3"]
    assert cells.loc["siCCT8#5", "balanced_repeats"] == 2
    joined = " ".join(contrast.summary().split())
    assert joined.count("holds very few tracks") == 1
    assert "siCCT8#5 and siCCT8#8: biological repeat CCT8_3 holds very few tracks" in joined
    assert "the balanced test balances the other 2 biological repeats" in joined


def test_a_dominant_repeat_raises_the_effective_cells_per_well():
    """data/simulations/unbalanced.py: 120/30/30 cells give 31% false positives at s = 0.25
    normalised, against 16% for 30/30/30; the formula at the effective 90 cells gives 32%."""
    assert effective_per_well([30, 30, 30]) == pytest.approx(30)
    assert effective_per_well([120, 30, 30]) == pytest.approx(90)
    assert chance_without_effect(effective_per_well([120, 30, 30]), 0.25) == pytest.approx(
        0.3155, abs=0.001
    )


def test_the_balanced_test_carries_its_own_chance():
    uneven = _side({"1": (120, 0.5), "2": (40, 0.5), "3": (40, 0.5)}, {"1": 40, "2": 40, "3": 40})
    row = _cells(KD=uneven).loc["KD"]
    # As provided the dominant repeat weighs most; balanced keeps 40 of each.
    assert row["per_well"] == pytest.approx((effective_per_well([120, 40, 40]) + 40) / 2)
    assert row["chance_balanced_low"] < row["chance_low"]
    assert row["chance_balanced_high"] < row["chance_high"]
