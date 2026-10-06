"""Batch normalisation of the other feature tables (owner, 2026-10-06): each repeat centred
on all its observations in a balanced design, else on its control; with normalize declared,
the populations, unusual observations and separation read the centred values, never a test."""

import numpy as np
import pandas as pd
from test_feature_tables_1006 import _cells, _table

from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.separation import lift
from morphocoverage.workflows import Setup
from morphocoverage.workflows._feature_space import (
    analysed_tables,
    balanced,
    centred_tables,
    centring_line,
    separation_in_report,
)
from morphocoverage.workflows._whole_table import whole_table_in_report


def _batch(drop_ctrl_r3: bool = False, effect: float = 1.5, batch: float = 3.0):
    """3 conditions in 3 repeats; DINO of 6 columns with a shift per repeat (a batch) and
    condition ``a`` shifted on the first two columns."""
    objects, fovs, rng = _cells(fields=4, cells=8)
    if drop_ctrl_r3:
        keep = ~((objects["condition"] == "a") & (objects["biological_repeat"] == "R3"))
        objects = objects[keep].reset_index(drop=True)
        fovs = fovs[~((fovs["condition"] == "a") & (fovs["biological_repeat"] == "R3"))]
        fovs = fovs.reset_index(drop=True)
    values = rng.normal(size=(len(objects), 6))
    shifts = {r: rng.normal(size=6) * batch for r in ("R1", "R2", "R3")}
    values += np.vstack([shifts[r] for r in objects["biological_repeat"]])
    values[objects["condition"].eq("a").to_numpy(), :2] += effect
    table = _table(objects, values, "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    merged["x0"] = 1.0
    names = tuple(f"d{j}" for j in range(6))
    return merged, fovs, names


def _setup(names, **kwargs) -> Setup:
    return Setup(
        readout="x0",
        control={"condition": "ctrl"},
        feature_tables={"DINO": names},
        observation_unit="cell",
        **kwargs,
    )


def test_a_balanced_design_is_centred_on_all_observations_of_each_repeat():
    merged, fovs, names = _batch()
    assert balanced(fovs)
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    medians = merged.groupby("biological_repeat")["d0"].transform("median")
    np.testing.assert_allclose(centred["d0"], merged["d0"] - medians)
    line = centring_line(merged, fovs, setup)
    assert line.startswith("Batch: the other tables centred on the median of all the cells")
    assert "the centred map reads them, never a test." in line


def test_an_unbalanced_design_is_centred_on_its_control():
    merged, fovs, names = _batch(drop_ctrl_r3=True)
    assert not balanced(fovs)
    setup = _setup(names, normalize="control_mean")
    centred = centred_tables(merged, fovs, setup)
    for repeat, rows in merged.groupby("biological_repeat"):
        control = rows[rows["condition"] == "ctrl"]["d0"].median()
        np.testing.assert_allclose(centred.loc[rows.index, "d0"], rows["d0"] - control)
    line = centring_line(merged, fovs, setup)
    assert "on the median of the control in their biological repeat" in line
    assert "the populations, unusual observations and the separation read them" in line


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
