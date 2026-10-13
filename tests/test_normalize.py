"""Normalising features to the control condition's mean, repeat by repeat."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage.data_check import validate_input_tables
from morphocoverage.normalize import control_means, normalize_to_control
from morphocoverage.outcomes import RandomObjectRelativeError, RelativeMeanError
from morphocoverage.workflows import Setup

GROUPS = ["condition", "biological_repeat", "well"]


def _tables():
    rows, fovs = [], []
    for repeat, level in (("R1", 2.0), ("R2", 4.0)):
        for condition, factor in (("control", 1.0), ("treated", 1.5)):
            name = f"{repeat}_{condition}"
            fovs.append(
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "well": "w",
                    "image_file": name,
                }
            )
            for j in range(4):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "well": "w",
                        "image_file": name,
                        "a": level * factor + (j - 1.5) * 0.01,
                        "b": 1.0,
                        "keep": j,
                    }
                )
    return pd.DataFrame(rows, index=range(100, 100 + len(rows))), pd.DataFrame(fovs)


def test_every_feature_is_divided_by_its_repeats_control_mean():
    objects, fovs = _tables()
    means = control_means(
        objects, fovs, feature_cols=["a", "b"], control={"condition": "control"}, group_cols=GROUPS
    )
    assert means["biological_repeat"].tolist() == ["R1", "R2"]
    assert means["a"].tolist() == pytest.approx([2.0, 4.0])
    normalized = normalize_to_control(
        objects, fovs, feature_cols=["a", "b"], control={"condition": "control"}, group_cols=GROUPS
    )
    assert normalized.index.equals(objects.index)
    assert list(normalized.columns) == list(objects.columns)
    assert normalized["keep"].tolist() == objects["keep"].tolist()  # other columns untouched
    by = normalized.groupby(["biological_repeat", "condition"])["a"].mean()
    assert [by[("R1", "control")], by[("R2", "control")]] == pytest.approx([1.0, 1.0])
    assert [by[("R1", "treated")], by[("R2", "treated")]] == pytest.approx([1.5, 1.5])
    assert (normalized["b"] == 1.0).all()


def test_without_a_repeat_column_the_whole_table_is_one_repeat():
    objects, fovs = _tables()
    objects, fovs = (
        objects.drop(columns="biological_repeat"),
        fovs.drop(columns="biological_repeat"),
    )
    normalized = normalize_to_control(
        objects,
        fovs,
        feature_cols=["a"],
        control={"condition": "control"},
        group_cols=["condition", "well"],
    )
    assert normalized.loc[normalized["condition"] == "control", "a"].mean() == pytest.approx(1.0)


def test_a_repeat_without_control_or_a_zero_mean_raises():
    objects, fovs = _tables()
    missing = fovs[~((fovs["biological_repeat"] == "R2") & (fovs["condition"] == "control"))]
    with pytest.raises(ValueError, match="no control objects .* biological_repeat=R2"):
        normalize_to_control(
            objects[objects["image_file"].isin(missing["image_file"])],
            missing,
            feature_cols=["a"],
            control={"condition": "control"},
            group_cols=GROUPS,
        )
    zero = objects.copy()
    zero.loc[zero["condition"] == "control", "b"] = 0.0
    with pytest.raises(ValueError, match="zero or undefined"):
        normalize_to_control(
            zero, fovs, feature_cols=["b"], control={"condition": "control"}, group_cols=GROUPS
        )
    with pytest.raises(ValueError, match="condition column"):
        normalize_to_control(
            objects, fovs, feature_cols=["a"], control={"well": "w"}, group_cols=GROUPS
        )


def _control_tables(control_level, treated_level):
    """Two repeats, a control and a treated condition, three fields of four objects each."""
    rows = []
    for repeat in ("r1", "r2"):
        for condition, level in (("ctrl", control_level), ("drug", treated_level)):
            for field_index in range(3):
                rows += [
                    {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "position_id": f"{repeat}_{condition}_{field_index}",
                        "x": value,
                    }
                    for value in level(repeat)
                ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_e1_a_negative_control_mean_is_refused_naming_feature_and_repeat():
    objects, fovs = _control_tables(
        lambda r: [-2.0, -2.1, -1.9, -2.0], lambda r: [-1.0, -1.1, -0.9, -1.0]
    )
    with pytest.raises(ValueError, match=r"repeat=r1: 'x' is negative .*flips the sign"):
        normalize_to_control(objects, fovs, feature_cols=["x"], control={"condition": "ctrl"})


def test_e2_an_explicit_repeat_column_that_is_absent_raises_instead_of_pooling():
    objects, fovs = _control_tables(lambda r: [1.0, 2.0, 1.5, 1.5], lambda r: [3.0] * 4)
    objects, fovs = (
        objects.rename(columns={"biological_repeat": "plate"}),
        fovs.rename(columns={"biological_repeat": "plate"}),
    )
    with pytest.raises(ValueError, match="repeat_col 'Plate' is not a column"):
        control_means(
            objects, fovs, feature_cols=["x"], control={"condition": "ctrl"}, repeat_col="Plate"
        )
    # The default name may be absent (one implicit repeat), and None asks for that.
    for repeat_col in ({}, {"repeat_col": None}):
        means = control_means(
            objects, fovs, feature_cols=["x"], control={"condition": "ctrl"}, **repeat_col
        )
        assert means["biological_repeat"].tolist() == [None]
    assert control_means(
        objects, fovs, feature_cols=["x"], control={"condition": "ctrl"}, repeat_col="plate"
    )["biological_repeat"].tolist() == ["r1", "r2"]


def test_e3_a_mean_that_is_rounding_noise_counts_as_zero():
    objects, fovs = _control_tables(lambda r: [0.1, 0.2, -0.3, 0.0], lambda r: [1.0] * 4)
    assert 0 < abs(objects.loc[objects["condition"] == "ctrl", "x"].mean()) < 1e-15
    with pytest.raises(ValueError, match="'x' is zero"):
        normalize_to_control(objects, fovs, feature_cols=["x"], control={"condition": "ctrl"})
    table = pd.DataFrame({"position_id": ["p0", "p0", "p1", "p1"], "x": [0.1, 0.2, -0.3, 0.0]})
    positions = pd.DataFrame({"position_id": ["p0", "p1"]})
    full = validate_input_tables(table, fovs=positions, feature_cols=["x"])
    stop = validate_input_tables(
        table[table["position_id"] == "p0"], fovs=positions.iloc[:1], feature_cols=["x"]
    )
    assert RelativeMeanError("x").evaluate(full, stop) is None
    assert RandomObjectRelativeError("x").evaluate(full, stop) is None


def _signed_feature():
    """A log-ratio centred near 0 whose control mean is positive in both repeats."""
    rng = np.random.default_rng(2)
    rows, fields = [], []
    for repeat in ("r1", "r2"):
        for condition, centre in (("MOCK", 0.05), ("TREAT", 0.30)):
            for f in range(6):
                name = f"{condition}_{repeat}_{f}"
                fields.append(
                    {"image_file": name, "condition": condition, "biological_repeat": repeat}
                )
                rows += [
                    {
                        "image_file": name,
                        "condition": condition,
                        "biological_repeat": repeat,
                        "x": v,
                    }
                    for v in rng.normal(centre, 0.3, 40)
                ]
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_e7_a_control_with_negative_values_is_normalised_with_a_warning():
    objects, fovs = _signed_feature()
    with pytest.warns(UserWarning, match="negative values of 'x'"):
        normalised = normalize_to_control(
            objects, fovs, feature_cols=["x"], control={"condition": "MOCK"}
        )
    assert "not a multiple of the control" in normalised.attrs["normalization_note"]
    means = control_means(objects, fovs, feature_cols=["x"], control={"condition": "MOCK"})
    assert list(means["negative_features"]) == [("x",), ("x",)]
    setup = Setup(readout="x", control={"condition": "MOCK"}, normalize="control_mean")
    assert "Caution: the control objects hold negative values of 'x'" in (
        setup.normalized_sentence(fovs, objects)
    )
    # A positive feature is normalised without a word.
    positive = objects.assign(x=objects["x"] + 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = normalize_to_control(
            positive, fovs, feature_cols=["x"], control={"condition": "MOCK"}
        )
    assert "normalization_note" not in quiet.attrs
    assert "Caution" not in setup.normalized_sentence(fovs, positive)


def test_e16_control_objects_counts_the_values_the_mean_rests_on():
    objects, fovs = _signed_feature()
    objects = objects.assign(x=objects["x"] + 5)
    first = objects.index[(objects.condition == "MOCK") & (objects.biological_repeat == "r1")][:3]
    objects.loc[first, "x"] = np.nan
    means = control_means(objects, fovs, feature_cols=["x"], control={"condition": "MOCK"})
    assert list(means["control_objects"]) == [240 - 3, 240]
