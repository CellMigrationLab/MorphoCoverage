"""Tests of ``separation``: how far the conditions, and the repeats, separate in a feature space."""

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import KEY, _batch, _cells, _table

from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.separation import Separation, separation
from morphocoverage.workflows._html_map_text import comparison_table


def test_one_well_per_condition_says_a_well_difference_reads_as_a_condition_separation():
    merged, fovs, names = _batch()
    spaces = {"DINO": names}
    unknown = separation(merged, fovs, spaces=spaces).well_line()
    assert unknown.startswith("The wells are not given, so a difference between the wells")
    one = fovs.assign(technical_repeat=fovs["condition"] + fovs["biological_repeat"])
    assert separation(merged, one, spaces=spaces).well_line() == (
        "Each condition is imaged in one well per biological repeat, no within-repeat "
        "technical replication, so a difference between wells reads here as a separation of "
        "the conditions."
    )
    field = fovs["image_file"].str.extract(r"_(\d+)\.tif$")[0].astype(int)
    two = one.assign(technical_repeat=one["technical_repeat"] + (field % 2).astype(str))
    assert separation(merged, two, spaces=spaces).well_line() == ""


def test_a_field_effect_does_not_read_as_conditions_separating():
    # Audit B, finding 3: 99% against 33% at a field ICC of 0.2 with no condition effect.
    objects, fovs, rng = _cells(fields=6, cells=20)
    n = len(objects)
    field = objects.groupby(KEY[:3], sort=False).ngroup().to_numpy()
    shifts = rng.normal(size=(field.max() + 1, 60))
    values = np.sqrt(0.8) * rng.normal(size=(n, 60)) + np.sqrt(0.2) * shifts[field]
    merged = attach_feature_tables(objects, fovs, {"T": _table(objects, values, "t")}).objects
    found = separation(merged, fovs, spaces={"T": [f"t{j}" for j in range(60)]})
    row = found.spaces.iloc[0]
    assert abs(row["share"] - row["chance"]) < 0.05
    assert abs(row["repeat_share"] - row["repeat_chance"]) < 0.05


def test_a_perfect_separation_of_few_images_reads_100_percent():
    # Audit B, finding 4: six images a condition and repeat, k = 10: a ceiling of 50%.
    objects, fovs, rng = _cells(fields=6, cells=1)
    objects = objects.drop(columns="label_id")
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    centre = objects["condition"].map({"ctrl": 0.0, "a": 10.0, "b": 20.0}).to_numpy()
    for j in range(3):
        objects[f"x{j}"] = centre + rng.normal(size=len(objects))
    found = separation(objects, fovs, spaces={"X": ["x0", "x1", "x2"]})
    assert found.spaces.loc[0, "share"] == pytest.approx(1.0)
    by_condition = found.table[found.table["score"] == "condition"]
    assert by_condition["k"].eq(5).all()  # the five other images of its condition


def test_repeats_separating_more_than_conditions_are_said_plainly():
    objects, fovs, rng = _cells(fields=4, cells=10)
    n = len(objects)
    repeat = objects["biological_repeat"].map({"R1": 0.0, "R2": 6.0, "R3": 12.0}).to_numpy()
    batch = rng.normal(size=(n, 8)) + repeat[:, None]
    merged = attach_feature_tables(objects, fovs, {"B": _table(objects, batch, "b")}).objects
    merged["x0"] = rng.normal(size=n)
    for j in range(1, 3):
        merged[f"x{j}"] = rng.normal(size=n)
    found = separation(
        merged, fovs, spaces={"B": [f"b{j}" for j in range(8)], "X": ["x0", "x1", "x2"]}
    )
    assert found.repeats_first("B") and not found.repeats_first("X")
    line = found.line("B")
    assert "Within a condition, 100% share the repeat" in line
    assert line.endswith("Here the repeats separate more than the conditions.")
    page = comparison_table(found)
    assert "The repeats separate more than the conditions in B" in page


def test_the_combined_space_is_shown_apart_under_feature_table():
    spaces = pd.DataFrame(
        [
            {"space": "Combined", "features": 9, "share": 0.9, "chance": 0.3, "repeats": 3},
            {"space": "Explainable", "features": 3, "share": 0.4, "chance": 0.3, "repeats": 3},
            {"space": "DINO", "features": 6, "share": 0.8, "chance": 0.3, "repeats": 3},
        ]
    ).assign(repeat_share=0.3, repeat_chance=0.3, conditions=3, left_out=0, single=0)
    found = Separation(pd.DataFrame(), spaces, {})
    lines = found.comparison().splitlines()
    assert lines[0].split()[:2] == ["feature", "table"]
    assert [line.split()[0] for line in lines[1:]] == ["DINO", "Explainable", "Combined"]
    page = comparison_table(found)
    assert "<th>feature table</th>" in page and "Combined: all the tables at once" in page
    assert page.index("DINO") < page.index("Explainable") < page.index("Combined")
