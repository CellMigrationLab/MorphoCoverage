import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables


def _objects():
    return pd.DataFrame(
        {
            "condition": ["control", "control", "control"],
            "repeat": ["R1", "R1", "R1"],
            "fov_id": ["F1", "F1", "F2"],
            "object_id": [1, 2, 3],
            "area": [10.0, 11.0, 20.0],
            "texture": [0.1, 0.2, 0.9],
            "stage_x": [0.0, 0.0, 100.0],
        }
    )


def test_validates_explicit_feature_columns():
    result = validate_input_tables(
        _objects(),
        feature_cols=["area", "texture"],
        fov_col="fov_id",
        group_cols=["condition", "repeat"],
    )
    assert result.feature_cols == ("area", "texture")
    assert list(result.fovs["fov_id"]) == ["F1", "F2"]


def test_does_not_infer_numeric_metadata_as_features():
    result = validate_input_tables(_objects(), feature_cols=["area"], fov_col="fov_id")
    assert result.feature_cols == ("area",)
    assert "stage_x" not in result.feature_cols
    assert "object_id" not in result.feature_cols


def test_requires_fov_column():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_input_tables(_objects().drop(columns="fov_id"), feature_cols=["area"], fov_col="fov_id")


def test_requires_features_to_be_numeric():
    objects = _objects()
    objects["phenotype"] = ["A", "A", "B"]
    with pytest.raises(TypeError, match="must be numeric"):
        validate_input_tables(objects, feature_cols=["phenotype"], fov_col="fov_id")


def test_rejects_non_finite_feature_values():
    objects = _objects()
    objects.loc[1, "area"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_input_tables(objects, feature_cols=["area"], fov_col="fov_id")


def test_rejects_metadata_used_as_feature():
    with pytest.raises(ValueError, match="group columns cannot also"):
        validate_input_tables(
            _objects(),
            feature_cols=["area", "repeat"],
            fov_col="fov_id",
            group_cols=["repeat"],
        )


def test_fov_table_preserves_zero_object_fov():
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control", "control"],
            "repeat": ["R1", "R1", "R1"],
            "fov_id": ["F1", "F2", "F3"],
            "stage_x": [0.0, 100.0, 200.0],
        }
    )
    result = validate_input_tables(
        _objects(),
        feature_cols=["area", "texture"],
        fov_col="fov_id",
        fovs=fovs,
        group_cols=["condition", "repeat"],
    )
    assert set(result.fovs["fov_id"]) == {"F1", "F2", "F3"}
    assert "F3" not in set(result.objects["fov_id"])


def test_rejects_object_fov_missing_from_fov_table():
    fovs = pd.DataFrame({"condition": ["control"], "repeat": ["R1"], "fov_id": ["F1"]})
    with pytest.raises(ValueError, match="absent from the supplied fovs table"):
        validate_input_tables(
            _objects(),
            feature_cols=["area"],
            fov_col="fov_id",
            fovs=fovs,
            group_cols=["condition", "repeat"],
        )


def test_allows_same_fov_id_in_different_groups():
    objects = pd.DataFrame({"repeat": ["R1", "R2"], "fov_id": ["F1", "F1"], "area": [10.0, 20.0]})
    fovs = pd.DataFrame({"repeat": ["R1", "R2"], "fov_id": ["F1", "F1"]})
    result = validate_input_tables(
        objects,
        feature_cols=["area"],
        fov_col="fov_id",
        fovs=fovs,
        group_cols=["repeat"],
    )
    assert len(result.fovs) == 2


def test_rejects_duplicate_fov_rows_within_group():
    fovs = pd.DataFrame(
        {"condition": ["control", "control"], "repeat": ["R1", "R1"], "fov_id": ["F1", "F1"]}
    )
    with pytest.raises(ValueError, match="exactly one row"):
        validate_input_tables(
            _objects(),
            feature_cols=["area"],
            fov_col="fov_id",
            fovs=fovs,
            group_cols=["repeat"],
        )
