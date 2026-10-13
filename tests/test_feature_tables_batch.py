"""Batch normalisation of the other feature tables (owner, 2026-10-06): each repeat centred
on the median of the per-condition medians of the conditions in every repeat (audit 8; before,
on all its observations in a balanced design, else on its control); with normalize declared,
the populations, unusual observations and separation read the centred values, never a test."""

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import _batch, _setup

from morphocoverage.separation import lift
from morphocoverage.workflows._feature_space import (
    analysed_tables,
    centred_tables,
    centring_line,
    separation_in_report,
)
from morphocoverage.workflows._whole_table import whole_table_in_report


def _per_condition_centre(rows: pd.DataFrame, conditions: list[str]) -> float:
    """The median of the per-condition medians of ``d0`` over ``conditions``."""
    return float(np.median([rows.loc[rows["condition"] == c, "d0"].median() for c in conditions]))


def test_a_balanced_design_is_centred_on_the_median_of_its_condition_medians():
    merged, fovs, names = _batch()
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    for _, rows in merged.groupby("biological_repeat"):
        centre = _per_condition_centre(rows, ["ctrl", "a", "b"])
        np.testing.assert_allclose(centred.loc[rows.index, "d0"], rows["d0"] - centre)
    line = centring_line(merged, fovs, setup)
    assert line.startswith(
        "Batch: the other tables centred per biological repeat on the median of the "
        "per-condition medians of ctrl, a and b, the conditions in every repeat"
    )
    assert "the centred map reads them, never a test." in line


def test_an_unbalanced_design_is_centred_on_the_conditions_in_every_repeat():
    merged, fovs, names = _batch(drop_ctrl_r3=True)
    setup = _setup(names, normalize="control_mean")
    centred = centred_tables(merged, fovs, setup)
    for _, rows in merged.groupby("biological_repeat"):
        # a is not imaged in R3: ctrl and b set the centre of every repeat (audit 8).
        centre = _per_condition_centre(rows, ["ctrl", "b"])
        np.testing.assert_allclose(centred.loc[rows.index, "d0"], rows["d0"] - centre)
    line = centring_line(merged, fovs, setup)
    assert "per-condition medians of ctrl and b, the conditions in every repeat" in line
    # Unusual observations read the explainable features here (audit 8B, seed 6).
    assert "the populations and the separation of the conditions read them" in line


def test_the_centre_does_not_depend_on_how_many_cells_a_condition_holds():
    # Audit 8 A3: 5 times the cells of a in R1 moved the all-observation median.
    merged, fovs, names = _batch(batch=0.0, effect=3.0)
    setup = _setup(names)
    more = merged[(merged["condition"] == "a") & (merged["biological_repeat"] == "R1")]
    copies = [more.assign(label_id=more["label_id"] + 100 * (i + 1)) for i in range(4)]
    crowded = pd.concat([merged, *copies], ignore_index=True)
    centred = centred_tables(crowded, fovs, setup)
    in_r1 = crowded["biological_repeat"] == "R1"
    centre = (crowded.loc[in_r1, "d0"] - centred.loc[in_r1, "d0"]).iloc[0]
    assert centre == pytest.approx(_per_condition_centre(crowded[in_r1], ["ctrl", "a", "b"]))


def test_without_a_condition_in_every_repeat_each_sub_experiment_is_centred_apart():
    merged, fovs, names = _batch()
    gone = merged["condition"].isin(["ctrl", "a"]) & merged["biological_repeat"].eq("R3")
    gone |= merged["condition"].eq("b") & merged["biological_repeat"].ne("R3")
    merged = merged[~gone].reset_index(drop=True)
    fovs = fovs[fovs["image_file"].isin(merged["image_file"])].reset_index(drop=True)
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    for repeat, rows in merged.groupby("biological_repeat"):
        conditions = ["b"] if repeat == "R3" else ["ctrl", "a"]
        centre = _per_condition_centre(rows, conditions)
        np.testing.assert_allclose(centred.loc[rows.index, "d0"], rows["d0"] - centre)
    line = centring_line(merged, fovs, setup)
    assert (
        "the conditions in every repeat of its sub-experiment: ctrl and a in R1 and R2; b in R3"
        in line
    )


def test_a_sub_experiment_with_no_condition_in_all_its_repeats_is_not_centred_and_named():
    merged, fovs, names = _batch()
    # R1 holds ctrl and a, R2 a and b, R3 b and ctrl: linked, but none in all three.
    gone = (
        (merged["condition"].eq("b") & merged["biological_repeat"].eq("R1"))
        | (merged["condition"].eq("ctrl") & merged["biological_repeat"].eq("R2"))
        | (merged["condition"].eq("a") & merged["biological_repeat"].eq("R3"))
    )
    merged = merged[~gone].reset_index(drop=True)
    fovs = fovs[fovs["image_file"].isin(merged["image_file"])].reset_index(drop=True)
    setup = _setup(names)
    assert centred_tables(merged, fovs, setup) is None
    assert centring_line(merged, fovs, setup).startswith(
        "Batch: the other tables are not centred, as no condition is in every repeat of a "
        "sub-experiment; they are read as given."
    )


def test_a_repeat_with_no_condition_to_centre_on_is_left_out_and_named_once():
    merged, fovs, names = _batch()
    # R1 and R2 keep ctrl, a and b; R3, R4 and R5 hold x and y, y and z, z and x (linked, no
    # condition in all three), relabelled copies of the other repeats.
    keep = merged[merged["biological_repeat"].isin(["R1", "R2"])]
    parts = [keep]
    for repeat, source, renamed in (
        ("R3", "R3", {"ctrl": "x", "a": "y"}),
        ("R4", "R2", {"ctrl": "y", "a": "z"}),
        ("R5", "R1", {"ctrl": "z", "a": "x"}),
    ):
        rows = merged[merged["biological_repeat"].eq(source) & merged["condition"].ne("b")]
        rows = rows.assign(
            biological_repeat=repeat,
            condition=rows["condition"].map(renamed),
            image_file=rows["image_file"] + repeat,
        )
        parts.append(rows)
    merged = pd.concat(parts, ignore_index=True)
    fovs = merged[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    out = merged["biological_repeat"].isin(["R3", "R4", "R5"])
    assert centred.loc[out, "d0"].isna().all() and centred.loc[~out, "d0"].notna().all()
    line = centring_line(merged, fovs, setup)
    assert "R3, R4 and R5 are left out of the centred values, as no condition is in every" in line


def test_a_planted_batch_is_removed_and_a_condition_effect_survives():
    merged, fovs, names = _batch()
    setup = _setup(names)
    raw = separation_in_report(merged, fovs, setup, ["DINO"]).spaces.iloc[0]
    centred = centred_tables(merged, fovs, setup)
    after = separation_in_report(centred, fovs, setup, ["DINO"]).spaces.iloc[0]
    assert lift(raw["repeat_share"], raw["repeat_chance"]) > 0.8
    assert lift(after["repeat_share"], after["repeat_chance"]) < 0.15
    # The condition effect survives: as far from chance as within each repeat on raw values.
    before = lift(raw["share"], raw["chance"])
    assert lift(after["share"], after["chance"]) > max(0.1, 0.8 * before)


def test_normalize_sends_the_centred_tables_to_the_analyses_never_to_the_tests():
    merged, fovs, names = _batch()
    plain, normalised = _setup(names), _setup(names, normalize="control_mean")
    assert analysed_tables(merged, fovs, plain) is merged
    centred = analysed_tables(merged, fovs, normalised)
    np.testing.assert_allclose(centred["d0"], centred_tables(merged, fovs, plain)["d0"])
    raw = whole_table_in_report(merged, fovs, plain).tests
    again = whole_table_in_report(merged, fovs, normalised).tests
    pd.testing.assert_frame_equal(raw, again)


def test_a_sub_experiment_of_one_condition_says_centring_erases_its_difference():
    """Audit 10 (C33; owner, 2026-10-06): in e02 each sub-experiment holds one condition,
    which sets its centre, so the centred view erases the difference between them; one line
    says so."""
    import sys
    import warnings
    from pathlib import Path

    from morphocoverage.workflows import Setup
    from morphocoverage.workflows._feature_space import attached, centring_line

    folder = Path(__file__).resolve().parents[1] / "data" / "scenarios"
    sys.path.insert(0, str(folder))
    try:
        import make_scenarios
    finally:
        sys.path.remove(str(folder))
    loaded = make_scenarios.load("e02_control_in_other_repeats")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        objects, setup, _ = attached(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), loaded.feature_tables
        )
    line = centring_line(objects, loaded.fovs, setup)
    assert (
        "ctrl in 2026_01_05 and 2026_01_12; a in 2026_01_19 and 2026_01_26, so the centred " in line
    )
    assert "view shows no difference between these conditions" in line
