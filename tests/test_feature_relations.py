"""Declared features that repeat one measurement are named once (owner, 2026-10-05)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morphocoverage import check_tables
from morphocoverage.retrospective import dataset_overview
from morphocoverage.retrospective._feature_relations import (
    NAME,
    NEAR_R,
    related_features,
    relation_checks,
)
from morphocoverage.retrospective.report_status import overview_status
from morphocoverage.schema import validate_input_tables


def _tables(extra: dict[str, np.ndarray] | None = None, cells: int = 240, seed: int = 0):
    """Two conditions in three repeats, 4 fields each, with an unrelated ``base`` and
    ``other`` and the ``extra`` columns."""
    rng = np.random.default_rng(seed)
    fields = np.arange(cells) % 24
    objects = pd.DataFrame(
        {
            "condition": np.where(fields % 2, "drug", "ctrl"),
            "biological_repeat": [f"r{f // 8}" for f in fields],
            "position_id": [f"p{f}" for f in fields],
            "base": rng.normal(10.0, 2.0, cells),
            "other": rng.normal(5.0, 1.0, cells),
        }
    )
    for name, values in (extra or {}).items():
        objects[name] = values
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def _data(objects: pd.DataFrame, fovs: pd.DataFrame, features: list[str]):
    return validate_input_tables(objects, fovs=fovs, feature_cols=features)


def _kinds(data) -> list[tuple[str, str, str]]:
    return [(x, y, found["kind"]) for g in related_features(data) for x, y, found in g["relations"]]


def test_unrelated_features_are_not_named():
    objects, fovs = _tables()
    data = _data(objects, fovs, ["base", "other"])
    assert related_features(data) == []
    assert relation_checks(data) == ()


def test_an_identical_column_is_a_copy():
    objects, fovs = _tables()
    objects["copy"] = objects["base"]
    data = _data(objects, fovs, ["base", "other", "copy"])
    assert _kinds(data) == [("base", "copy", "copy")]
    (sentence,) = relation_checks(data)
    assert sentence == (
        f"{NAME}: copy is a copy of base: one measurement counted twice in each test family; "
        "keep one of them."
    )


def test_one_minus_a_feature_is_derived():
    objects, fovs = _tables()
    objects["share"] = objects["base"] / 20
    objects["rest"] = 1 - objects["share"]
    (sentence,) = relation_checks(_data(objects, fovs, ["share", "other", "rest"]))
    assert "rest = 1 − share:" in sentence


def test_a_change_of_units_is_derived():
    objects, fovs = _tables()
    objects["base_nm"] = objects["base"] * 1000
    data = _data(objects, fovs, ["base", "base_nm"])
    ((_, _, found),) = related_features(data)[0]["relations"]
    assert found["kind"] == "derived" and found["b"] == 0
    assert "base_nm = 1000 × base:" in relation_checks(data)[0]


def test_a_near_copy_is_named_with_its_correlation():
    rng = np.random.default_rng(1)
    objects, fovs = _tables()
    objects["near"] = objects["base"] + rng.normal(0, 0.06, len(objects))
    data = _data(objects, fovs, ["base", "near"])
    ((_, _, found),) = related_features(data)[0]["relations"]
    assert found["kind"] == "near" and NEAR_R <= found["r"] < 1
    (sentence,) = relation_checks(data)
    assert "base and near correlate at r = 0.99" in sentence
    assert "close to one measurement, counted twice" in sentence


def test_correlated_features_at_r_095_are_not_named():
    rng = np.random.default_rng(2)
    objects, fovs = _tables()
    noise = rng.normal(0, 1.0, len(objects))
    standard = (objects["base"] - objects["base"].mean()) / objects["base"].std()
    objects["related"] = 0.95 * standard + np.sqrt(1 - 0.95**2) * noise
    data = _data(objects, fovs, ["base", "related"])
    assert abs(np.corrcoef(objects["base"], objects["related"])[0, 1] - 0.95) < 0.02
    assert related_features(data) == []


def test_relations_are_judged_on_the_rows_where_both_are_finite():
    objects, fovs = _tables()
    objects["copy"] = objects["base"].where(np.arange(len(objects)) % 7 != 0)
    objects["base"] = objects["base"].where(np.arange(len(objects)) % 5 != 0)
    objects["scaled"] = (objects["other"] * 3).where(np.arange(len(objects)) % 3 != 0)
    data = _data(objects, fovs, ["base", "other", "copy", "scaled"])
    assert _kinds(data) == [("base", "copy", "copy"), ("other", "scaled", "derived")]
    assert len(relation_checks(data)) == 2


def test_constant_features_are_left_to_the_few_values_check():
    objects, fovs = _tables()
    objects["one"] = 1.0
    objects["also_one"] = 1.0
    assert related_features(_data(objects, fovs, ["base", "one", "also_one"])) == []


def test_a_group_of_four_is_said_once():
    """The owner's ECM data: two copies of porosity, itself 1 - HDM."""
    objects, fovs = _tables()
    objects["HDM"] = objects["base"] / 20
    objects["Porosity (mean)"] = 1 - objects["HDM"]
    objects["Porosity X (mean)"] = objects["Porosity (mean)"]
    objects["Porosity Y (mean)"] = objects["Porosity (mean)"]
    features = ["HDM", "other", "Porosity (mean)", "Porosity X (mean)", "Porosity Y (mean)"]
    (group,) = related_features(_data(objects, fovs, features))
    assert group["features"] == [f for f in features if f != "other"]
    checked = check_tables(objects, fovs, feature_cols=features, observation_unit="cell")
    overview = dataset_overview(objects, fovs, feature_cols=features, data_check=checked)
    named = [check for check in overview.value_checks if check.startswith(NAME)]
    assert named == [
        (
            f"{NAME}: Porosity X (mean) and Porosity Y (mean) are copies of Porosity (mean), "
            "and Porosity (mean) = 1 − HDM: one measurement counted 4 times in each test "
            "family; keep one of them."
        )
    ]
    assert sum(NAME in line for line in overview.short_lines()) == 1
    _, glance = overview_status(overview)
    assert "values: duplicated or derived features" in glance


def test_two_groups_are_two_sentences_named_once_at_a_glance():
    objects, fovs = _tables()
    objects["copy"] = objects["base"]
    objects["other_um"] = objects["other"] / 1000
    features = ["base", "other", "copy", "other_um"]
    checked = check_tables(objects, fovs, feature_cols=features, observation_unit="cell")
    overview = dataset_overview(objects, fovs, feature_cols=features, data_check=checked)
    assert sum(check.startswith(NAME) for check in overview.value_checks) == 2
    _, glance = overview_status(overview)
    assert glance.count("duplicated or derived features") == 1


def test_hundreds_of_features_are_checked_at_once():
    rng = np.random.default_rng(3)
    objects, fovs = _tables(cells=2000)
    names = [f"f{i}" for i in range(300)]
    extra = pd.DataFrame(rng.normal(size=(len(objects), 300)), columns=names)
    extra["f299"] = -2 * extra["f7"] + 5
    objects = pd.concat([objects, extra], axis=1)
    data = _data(objects, fovs, list(extra.columns))
    assert _kinds(data) == [("f7", "f299", "derived")]
    assert "f299 = 5 − 2 × f7:" in relation_checks(data)[0]
