"""Normalising features to the control condition's mean, repeat by repeat."""

import pandas as pd
import pytest

from morphocoverage.normalize import control_means, normalize_to_control

GROUPS = ["condition", "repeat", "well"]


def _tables():
    rows, fovs = [], []
    for repeat, level in (("R1", 2.0), ("R2", 4.0)):
        for condition, factor in (("control", 1.0), ("treated", 1.5)):
            name = f"{repeat}_{condition}"
            fovs.append({"condition": condition, "repeat": repeat, "well": "w", "image_file": name})
            for j in range(4):
                rows.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
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
    assert means["repeat"].tolist() == ["R1", "R2"]
    assert means["a"].tolist() == pytest.approx([2.0, 4.0])
    normalized = normalize_to_control(
        objects, fovs, feature_cols=["a", "b"], control={"condition": "control"}, group_cols=GROUPS
    )
    assert normalized.index.equals(objects.index)
    assert list(normalized.columns) == list(objects.columns)
    assert normalized["keep"].tolist() == objects["keep"].tolist()  # other columns untouched
    by = normalized.groupby(["repeat", "condition"])["a"].mean()
    assert [by[("R1", "control")], by[("R2", "control")]] == pytest.approx([1.0, 1.0])
    assert [by[("R1", "treated")], by[("R2", "treated")]] == pytest.approx([1.5, 1.5])
    assert (normalized["b"] == 1.0).all()


def test_without_a_repeat_column_the_whole_table_is_one_repeat():
    objects, fovs = _tables()
    objects, fovs = objects.drop(columns="repeat"), fovs.drop(columns="repeat")
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
    missing = fovs[~((fovs["repeat"] == "R2") & (fovs["condition"] == "control"))]
    with pytest.raises(ValueError, match="no control objects .* repeat=R2"):
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
