"""A ValidatedInput built by hand cannot masquerade as validated (issue 117)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage import ValidatedInput, validate_input_tables
from morphocoverage.evaluator import restrict_input


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fovs = pd.DataFrame({"condition": ["a", "a", "b"], "position_id": ["f1", "f2", "f3"]})
    objects = fovs.loc[[0, 0, 1, 2]].reset_index(drop=True).assign(x=[1.0, 2.0, 3.0, 4.0])
    return objects, fovs


def _fields(**changes):
    objects, fovs = _tables()
    base = {
        "objects": objects,
        "fovs": fovs,
        "feature_cols": ("x",),
        "fov_col": "position_id",
        "group_cols": ("condition",),
        "fov_key": ("condition", "position_id"),
    }
    return {**base, **changes}


def test_a_hand_built_valid_input_is_accepted_and_equals_the_validated_one():
    objects, fovs = _tables()
    built = ValidatedInput(**_fields())
    validated = validate_input_tables(objects=objects, fovs=fovs, feature_cols=["x"])
    assert built.fov_key == validated.fov_key and built.feature_cols == validated.feature_cols
    assert np.array_equal(built.feature_matrix(), validated.feature_matrix())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"feature_cols": ("missing",)}, "lacks the column"),
        ({"fov_key": ("condition", "gone"), "fov_col": "gone"}, "lacks the column"),
        ({"fov_key": ("condition", "position_id", "condition")}, "repeats a column"),
        ({"fov_col": "condition"}, "must end with the FOV column"),
        ({"group_cols": ("batch",)}, "must be part of fov_key"),
        ({"feature_cols": ()}, "feature_cols is empty"),
        ({"feature_cols": ("position_id",)}, "both a feature and part of the FOV identity"),
        ({"observation_unit": "Cells"}, "not canonical"),
        ({"observation_unit_declared": 1}, "must be True or False"),
        ({"feature_cols": ["x"]}, "must be a tuple"),
    ],
)
def test_direct_construction_checks_the_declared_columns_and_unit(changes, message):
    with pytest.raises(ValueError, match=message):
        ValidatedInput(**_fields(**changes))


def test_duplicated_fov_rows_and_unlisted_fovs_are_refused():
    _, fovs = _tables()
    with pytest.raises(ValueError, match="exactly one row per acquired FOV"):
        ValidatedInput(**_fields(fovs=pd.concat([fovs, fovs.iloc[[0]]], ignore_index=True)))
    with pytest.raises(ValueError, match="absent from the supplied fovs"):
        ValidatedInput(**_fields(fovs=fovs.iloc[:2]))


def test_non_numeric_infinite_and_missing_identity_are_refused():
    objects, _ = _tables()
    with pytest.raises(ValueError, match="infinite"):
        ValidatedInput(**_fields(objects=objects.assign(x=[1.0, np.inf, 3.0, 4.0])))
    with pytest.raises((ValueError, TypeError)):
        ValidatedInput(**_fields(objects=objects.assign(x=["a", "b", "c", "d"])))
    with pytest.raises(ValueError, match="missing"):
        ValidatedInput(**_fields(objects=objects.assign(position_id=["f1", None, "f2", "f3"])))


def test_the_frameworks_own_subsets_are_trusted_and_keep_their_state():
    objects, fovs = _tables()
    data = validate_input_tables(objects=objects, fovs=fovs, feature_cols=["x"])
    subset = restrict_input(data, data.objects.iloc[:2], data.fovs.iloc[:1])
    assert len(subset.fovs) == 1 and subset.feature_cols == data.feature_cols
    copy = data.with_tables(feature_cols=("x",))
    assert copy is not data and copy._feature_cache == {}
