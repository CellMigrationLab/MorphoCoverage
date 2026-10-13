"""Tests of ``retrospective.overview``: what is in the dataset, and the checks of its values."""

from __future__ import annotations

import numpy as np
from _helpers_embedding import _several
from _helpers_retrospective import _homogeneous_tables
from _helpers_sampling_unit import _three
from _helpers_tables import FOVS, OBJECTS, _plate_tables, _quiet
from scenarios import FEATURE_COLS

from morphocoverage import integrity_checks
from morphocoverage.retrospective import dataset_overview, inspect_fovs
from morphocoverage.retrospective._value_checks import alike_fields, few_values
from morphocoverage.schema import validate_input_tables


def test_a_generator_is_consumed_exactly_once():
    expected = _quiet(lambda: dataset_overview(OBJECTS, FOVS, feature_cols=["x", "y"]))
    once = _quiet(lambda: dataset_overview(OBJECTS, FOVS, feature_cols=(c for c in ["x", "y"])))
    assert once.features == expected.features == ("x", "y")
    report = _quiet(lambda: integrity_checks(OBJECTS, FOVS, feature_cols=(c for c in ["x", "y"])))
    assert report is not None
    inspected = _quiet(
        lambda: inspect_fovs(
            OBJECTS,
            FOVS,
            feature_cols=(c for c in ["x", "y"]),
            novelty_cols=(c for c in ["x"]),
            fov_col="position_id",
        )
    )
    assert inspected is not None


def test_p5_9_overview_names_the_declared_groups():
    cells, fovs = _plate_tables()
    renamed = {"condition": "Treatment", "biological_repeat": "Plate", "technical_repeat": "Well"}
    summary = dataset_overview(
        cells.rename(columns=renamed),
        fovs.rename(columns=renamed),
        feature_cols=["area"],
        group_cols=["Treatment", "Plate", "Well"],
        observation_unit="cell",
    )
    full = summary.summary(detail="full")
    assert "all fields pooled" not in full
    assert "the analysis groups are Treatment, Plate, Well (8 groups), each analysed apart" in full
    short = " ".join(summary.summary().split())
    assert "8 analysis groups (Treatment · Plate · Well), each analysed apart" in short


def test_b6_a_condition_label_is_never_capitalised():
    objects, fovs, _ = _several()
    overview = dataset_overview(objects, fovs, feature_cols=["area"], observation_unit="cell")
    assert "\ncB in 2 of the 3 repeats." in overview.summary(detail="full")


def test_fields_normalised_per_image_are_more_alike_than_chance():
    objects, fovs = _three()
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    assert alike_fields(data) is None
    per_image = objects.assign(
        x=objects["x"] / objects.groupby("position_id")["x"].transform("mean")
    )
    data = validate_input_tables(per_image, fovs=fovs, feature_cols=["x"])
    assert "Fields more alike than chance: x (in 6 of 6 groups)" in alike_fields(data)


def test_a_feature_of_few_values_is_named():
    objects, fovs = _three()
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    assert few_values(data) is None
    sparse = objects.assign(x=np.where(np.arange(len(objects)) % 10 == 0, 1e-5, 0.0))
    data = validate_input_tables(sparse, fovs=fovs, feature_cols=["x"], observation_unit="patch")
    assert "Few distinct values: x (2 values, 0 in 90% of patches)" in few_values(data)


def test_overview_full_names_planes_and_a_missing_condition_or_repeat_column():
    objects, fovs = _homogeneous_tables()
    planes = objects.assign(z=objects.groupby("image_file").cumcount() % 3)
    text = dataset_overview(
        planes, fovs, feature_cols=list(FEATURE_COLS), observation_unit="slice"
    ).summary(detail="full")
    assert text.splitlines()[2] == (
        "'z' takes 3 values per field: slices of several planes, channels or frames of one "
        "field (declared as plane-level rows)."
    )
    no_condition = dataset_overview(
        objects.drop(columns="condition"),
        fovs.drop(columns="condition"),
        feature_cols=list(FEATURE_COLS),
    ).summary(detail="full")
    assert "\nno condition column; 1 repeat (biological_repeat: R1).\n" in no_condition
    no_repeat = dataset_overview(
        objects.drop(columns="biological_repeat"),
        fovs.drop(columns="biological_repeat"),
        feature_cols=list(FEATURE_COLS),
    ).summary(detail="full")
    assert (
        "\n1 condition (condition: control); no repeat column (one implicit repeat).\n" in no_repeat
    )
