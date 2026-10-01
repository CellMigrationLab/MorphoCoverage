"""The contrast across biological repeats: the repeat as the replicate (question 9)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_repeats import (
    MIN_REPEATS,
    across_repeats,
    sign_flip_test,
)

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def test_the_sign_flip_test_is_exact_and_states_its_floor():
    rng = np.random.default_rng(0)
    p, floor, exact = sign_flip_test(np.array([-1.0, -2.0, -3.0]), rng)
    assert exact and p == pytest.approx(0.25) and floor == pytest.approx(0.25)
    p, floor, _ = sign_flip_test(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]), rng)
    assert p == pytest.approx(2 / 32) and floor == pytest.approx(2 / 32)
    # Mixed signs: 6 of the 8 assignments are at least as extreme as |mean(-1, 2, -3)|.
    p, _, _ = sign_flip_test(np.array([-1.0, 2.0, -3.0]), rng)
    assert p == pytest.approx(6 / 8)


def _table(differences, control=10.0):
    return pd.DataFrame(
        {
            "condition": "KD",
            "control_condition": "CTRL",
            "mean": [control + d for d in differences],
            "control_mean": control,
            "d_objects": [d / 2 for d in differences],
        }
    )


def test_across_repeats_counts_the_direction_and_pairs_by_repeat():
    result = across_repeats(_table([-2.0, -3.0, -1.0]), "biological_repeat", 0)
    [row] = result.to_dict("records")
    assert (row["repeats"], row["lower"], row["higher"]) == (3, 3, 0)
    assert row["mean_difference"] == pytest.approx(-2.0)
    assert row["d_mean"] == pytest.approx(-1.0)
    assert (row["d_min"], row["d_max"]) == (pytest.approx(-1.5), pytest.approx(-0.5))
    # Three repeats: the sign-flip test's smallest p is 0.25, so it cannot help and is left
    # out; the paired t-test is run.
    assert np.isnan(row["p_sign_flip"]) and row["p_sign_flip_floor"] == pytest.approx(0.25)
    assert 0 < row["p_paired_t"] < 0.1 and row["status"] == "ok"


def test_the_sign_flip_test_runs_once_it_can_reach_the_threshold():
    """Six repeats: a floor of 2/64 below 0.05, so the exact test is run (one condition)."""
    six = _table([-2.0, -3.0, -1.0, -2.5, -1.5, -2.0])
    [row] = across_repeats(six, "biological_repeat", 0).to_dict("records")
    assert row["p_sign_flip"] == pytest.approx(2 / 64) and row["p_sign_flip_exact"]
    # Two conditions at six repeats: 2/64 x 2 is above 0.05, so neither is run.
    two = pd.concat([_table([-2.0, -3.0, -1.0, -2.5, -1.5, -2.0])] * 2).reset_index(drop=True)
    two.loc[6:, "condition"] = "OE"
    assert across_repeats(two, "biological_repeat", 0)["p_sign_flip"].isna().all()


def test_fewer_than_three_repeats_get_the_direction_and_no_test():
    [row] = across_repeats(_table([-2.0, -3.0]), "biological_repeat", 0).to_dict("records")
    assert row["lower"] == 2 and np.isnan(row["p_sign_flip"]) and np.isnan(row["p_paired_t"])
    assert f"fewer than {MIN_REPEATS} biological repeats" in row["status"]
    # One repeat, or no repeat column: no table at all.
    assert across_repeats(_table([-2.0]), "biological_repeat", 0).empty
    assert across_repeats(_table([-2.0, -3.0, -1.0]), None, 0).empty


def test_bonferroni_is_over_the_conditions_tested():
    table = pd.concat([_table([-2.0, -3.0, -1.0]), _table([1.0, 2.5, 2.0]).assign(condition="OE")])
    result = across_repeats(table, "biological_repeat", 0).set_index("condition")
    assert result.loc["OE", "higher"] == 3
    corrected = np.minimum(1.0, result["p_paired_t"] * 2)
    assert np.allclose(result["p_paired_t_bonferroni"], corrected)


@pytest.fixture(scope="module")
def screen():
    data = catalog.load("cct8_filopodia_screen")
    return condition_contrast(
        *data.tables,
        feature="Filopodia number",
        control=data.control,
        observation_unit="cell",
    )


def test_the_screen_is_on_one_side_of_its_sets_control_in_every_repeat(screen):
    """Hypothesis 4 of the card: MAP4K4 and NF2 above, the other four genes below, 3 of 3."""
    repeats = screen.repeats.set_index("condition")
    assert (repeats["repeats"] == 3).all()
    higher = {c for c in repeats.index if repeats.loc[c, "higher"] == 3}
    lower = {c for c in repeats.index if repeats.loc[c, "lower"] == 3}
    assert higher == {"siMAP4K4#5", "siMAP4K4#6", "siNF2#7", "siNF2#17"}
    assert len(lower) == 8 and not higher & lower
    # Within a repeat there is one coverslip a side: no test there, one across repeats.
    assert (screen.table["status"] != "ok").all()
    assert repeats["p_paired_t"].notna().all()
    assert repeats["p_sign_flip"].isna().all()  # a floor of 0.25 with 3 repeats: left out


def test_question_9_leads_with_the_answer_across_repeats(screen):
    short = screen.summary()
    joined = " ".join(short.split())
    assert "Across the 3 biological repeats (each repeat's value is the mean of its cells" in joined
    assert "lower in 3 of 3" in short and "higher in 3 of 3" in short
    # 36 per-repeat contrasts go to the full form, with the sign-flip test.
    assert "36 contrasts, one per condition and repeat" in short
    assert "sign-flip p" not in short.split("Within each repeat")[0]
    full = screen.summary(detail="full")
    assert "sign-flip" not in full and "siCCT8#5 vs siCTRL · siCCT8_1" in full
    # Three repeats and one coverslip per condition is the usual design: notes, not warnings.
    assert "⚠️" not in short
    assert "the paired t-test on the repeat means (as in SuperPlots)" in joined
    assert "Within a repeat the conditions are described by their effect sizes" in joined
    status, headline = conditions_status(screen, None)
    assert status.name == "OK"
    assert headline == (
        "12 conditions across 3 biological repeats: same direction in every repeat, not "
        "confirmed: siCCT8#5, siCCT8#8, siCSNK1A1#11, siCSNK1A1#5 and 4 more (lower); "
        "siMAP4K4#5, siMAP4K4#6, siNF2#17 and siNF2#7 (higher); reagents agree in every repeat "
        "for 6 of 6 targets"
    )


def test_two_repeats_and_one_coverslip_are_described_not_flagged(screen):
    """Two repeats: no test across or within them. A pilot design: described, not flagged."""
    table = screen.table[screen.table["biological_repeat"].str.endswith(("_1", "_2"))]
    two = type(screen)(
        table, screen.effects, screen.settings, across_repeats(table, "biological_repeat", 0)
    )
    status, headline = conditions_status(two, None)
    assert status.name == "NOT_ASSESSED"
    assert "12 conditions across 2 biological repeats: same direction in every repeat" in headline
    assert "within repeats: not tested (one technical repeat on a side)" in headline
    assert "⚠️" not in two.summary()
    assert "the paired test across repeats needs 3" in " ".join(two.summary().split())
