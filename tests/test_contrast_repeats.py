"""The contrast across biological repeats: the repeat as the replicate (question 9)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_repeats import (
    FEW_REPEATS,
    MIN_REPEATS,
    NO_SPREAD,
    across_repeats,
    sign_flip_test,
    untested_reason,
)
from morphocoverage.statistics import benjamini_hochberg
from morphocoverage.workflows import Setup, report

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


def test_the_sign_flip_floor_counts_only_differences_that_are_not_zero():
    """A zero difference flips to itself: with 1 of 6 at 0, p can reach 2/32, not 2/64,
    so the test cannot reach 0.05 and is left out (#98)."""
    rng = np.random.default_rng(0)
    differences = np.array([-2.0, -3.0, 0.0, -2.5, -1.5, -2.0])
    p, floor, exact = sign_flip_test(differences, rng)
    assert exact and p == pytest.approx(2 / 32) and floor == pytest.approx(2 / 32)
    assert sign_flip_test(np.zeros(4), rng)[:2] == (1.0, 1.0)
    [row] = across_repeats(_table(list(differences)), "biological_repeat", 0).to_dict("records")
    assert np.isnan(row["p_sign_flip"]) and row["p_sign_flip_floor"] == pytest.approx(2 / 32)


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


def _oe(repeats):
    return _table([1.0, 2.5, 2.0]).assign(condition="OE", biological_repeat=repeats)


def test_q_is_benjamini_hochberg_within_each_sub_experiment():
    """Conditions imaged in the same repeats are one sub-experiment, corrected together;
    conditions in repeats of their own are corrected apart."""
    kd = _table([-2.0, -3.0, -1.0]).assign(biological_repeat=["a", "b", "c"])
    shared = pd.concat([kd, _oe(["a", "b", "c"])])
    result = across_repeats(shared, "biological_repeat", 0).set_index("condition")
    assert result.loc["OE", "higher"] == 3 and result["experiment"].nunique() == 1
    assert np.allclose(result["q_paired_t"], benjamini_hochberg(result["p_paired_t"].to_numpy()))
    apart = pd.concat([kd, _oe(["d", "e", "f"])])
    result = across_repeats(apart, "biological_repeat", 0).set_index("condition")
    assert result["experiment"].nunique() == 2
    assert np.allclose(result["q_paired_t"], result["p_paired_t"])


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
    # Nothing is tested within a repeat: the per-repeat table goes to the full form.
    assert "No test within a single repeat: each condition and its control have one well" in joined
    assert "sign-flip p" not in short.split("Within each repeat")[0]
    full = screen.summary(detail="full")
    assert "sign-flip" not in full and "siCCT8#5 vs siCTRL, repeat siCCT8_1" in full
    # Three repeats and one coverslip per condition is the usual design: notes, not warnings.
    assert "⚠️" not in short
    assert "The paired t-test on the repeat means treats each biological repeat" in joined
    assert "Within a repeat the conditions are described by their effect sizes" in joined
    status, headline = conditions_status(screen, None)
    assert status.name == "OK"
    assert headline == (
        "Filopodia number, 12 conditions against the control, each in 3 biological repeats "
        "(18 in all): significant across repeats: siCCT8#5, siCCT8#8, siCSNK1A1#11, "
        "siCSNK1A1#5, siSCRIB#6 and siSCRIB#7 (lower); siMAP4K4#5 (higher); not significant "
        "across repeats: siMINK1#10 and siMINK1#3 (lower); siMAP4K4#6, "
        "siNF2#17 and siNF2#7 (higher); the reagents of a target go the same way in every "
        "repeat for 6 of 6 targets (described, not tested)"
    )


def test_two_repeats_and_one_coverslip_are_described_not_flagged(screen):
    """Two repeats: no test across or within them. A pilot design: described, not flagged."""
    table = screen.table[screen.table["biological_repeat"].str.endswith(("_1", "_2"))]
    two = type(screen)(
        table, screen.effects, screen.settings, across_repeats(table, "biological_repeat", 0)
    )
    status, headline = conditions_status(two, None)
    assert status.name == "NOT_ASSESSED"
    assert "each in 2 biological repeats (12 in all): not tested across repeats" in headline
    assert "within repeats: no test (one well on a side)" in headline
    assert "⚠️" not in two.summary()
    assert "the paired test across repeats needs 3" in " ".join(two.summary().split())


def _identical_effects(repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every repeat holds the same change (x2) exactly: the repeat effects have no spread."""
    rows, fovs = [], []
    for condition, level in (("ctrl", 10.0), ("drug", 20.0)):
        for r in range(repeats):
            for f in range(5):
                pid = f"{condition}{r}{f}"
                fovs.append(
                    {"position_id": pid, "condition": condition, "biological_repeat": f"R{r}"}
                )
                for k in range(4):
                    rows.append({**fovs[-1], "x": level + (k - 1.5), "y": 1.0 + (f + k) % 3})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def test_the_reason_is_read_from_the_repeats_held_not_from_a_missing_p():
    assert untested_reason(2, float("nan")) == FEW_REPEATS
    assert untested_reason(3, float("nan")) == NO_SPREAD
    assert untested_reason(5, 0.03) is None


@pytest.mark.parametrize("repeats", [3, 4])
def test_identical_effects_are_never_described_as_needing_three_repeats(repeats):
    objects, fovs = _identical_effects(repeats)
    setup = Setup(
        readout="x", features=("x", "y"), control={"condition": "ctrl"}, observation_unit="cell"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    across = result.results["contrast"].repeats
    assert list(across["status"]) == [NO_SPREAD]
    text = " ".join(result.summary(detail="full").split())
    assert "the paired test across repeats needs 3" not in text
    assert "has the same change in every biological repeat" in text


def _across_one_repeat():
    from morphocoverage.sampling_unit.contrast_repeats import across_repeats

    table = pd.DataFrame(
        {
            "condition": ["A", "B"],
            "control_condition": ["c", "c"],
            "biological_repeat": ["r1", "r1"],
            "mean": [2.0, 3.0],
            "control_mean": [1.0, 1.0],
            "d_objects": [0.5, 0.7],
        }
    )
    out = across_repeats(table, "biological_repeat", 0)
    return list(out.columns), len(out)


def _repeat_cautions_flat():
    from morphocoverage.sampling_unit.repeats_text import repeat_cautions

    repeats = pd.DataFrame(
        [
            {
                "condition": "A",
                "status": "the repeat effects are identical, so the paired test has "
                "no variance to test with",
                "repeats": 3,
                "p_paired_t": np.nan,
                "q_paired_t_normalised": np.nan,
                "p_sign_flip": np.nan,
            },
            {
                "condition": "B",
                "status": "the repeat effects are identical, so the paired test has "
                "no variance to test with",
                "repeats": 4,
                "p_paired_t": np.nan,
                "q_paired_t_normalised": np.nan,
                "p_sign_flip": np.nan,
            },
        ]
    )
    contrast = SimpleNamespace(repeats=repeats, settings={"repeat_col": "biological_repeat"})
    return repeat_cautions(contrast, full=False), repeat_cautions(contrast, full=True)


EXPECTED_ACROSS_ONE_REPEAT = (
    [
        "condition",
        "control_condition",
        "repeats",
        "lower",
        "higher",
        "mean_difference",
        "fold_change",
        "d_mean",
        "d_min",
        "d_max",
        "p_sign_flip",
        "q_sign_flip",
        "p_sign_flip_floor",
        "p_sign_flip_exact",
        "p_paired_t",
        "q_paired_t",
        "p_paired_t_normalised",
        "q_paired_t_normalised",
        "experiment",
        "status",
        "repeat_contrasts",
        "ci_low",
        "ci_high",
        "q_screen",
    ],
    0,
)


EXPECTED_REPEAT_CAUTIONS_FLAT = (
    [
        (
            "A and B have the same change in every biological repeat, so the "
            "paired test has no variance to test with: no p is given, and the "
            "change is described by its direction and size."
        )
    ],
    [
        (
            "A and B have the same change in every biological repeat, so the "
            "paired test has no variance to test with: no p is given, and the "
            "change is described by its direction and size."
        )
    ],
)


def test_across_one_repeat():
    assert _across_one_repeat() == EXPECTED_ACROSS_ONE_REPEAT


def test_repeat_cautions_flat():
    assert _repeat_cautions_flat() == EXPECTED_REPEAT_CAUTIONS_FLAT


def test_a_sign_flip_test_reads_the_same_at_any_scale():
    from morphocoverage.sampling_unit.contrast_repeats import sign_flip_test

    for scale in (1.0, 1e-12, 1e-13):
        p, floor, exact = sign_flip_test(scale * np.arange(1.0, 8.0), np.random.default_rng(0))
        assert exact and p == pytest.approx(2 / 2**7) and floor == pytest.approx(2 / 2**7)
