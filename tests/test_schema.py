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
        validate_input_tables(
            _objects().drop(columns="fov_id"), feature_cols=["area"], fov_col="fov_id"
        )


def test_requires_features_to_be_numeric():
    objects = _objects()
    objects["phenotype"] = ["A", "A", "B"]
    with pytest.raises(TypeError, match="must be numeric"):
        validate_input_tables(objects, feature_cols=["phenotype"], fov_col="fov_id")


def test_allows_missing_feature_values_but_rejects_infinity():
    objects = _objects()
    objects.loc[1, "area"] = np.nan
    result = validate_input_tables(objects, feature_cols=["area"], fov_col="fov_id")
    assert np.isnan(result.feature_values("area")[1])

    objects.loc[1, "area"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_input_tables(objects, feature_cols=["area"], fov_col="fov_id")


def test_records_generic_observation_semantics():
    result = validate_input_tables(
        _objects(),
        feature_cols=["area"],
        fov_col="fov_id",
        observation_unit="nucleus",
    )
    assert result.observation_unit == "nucleus"
    assert result.n_observations == len(_objects())
    pd.testing.assert_frame_equal(result.observations, result.objects)

    with pytest.raises(ValueError, match="observation_unit"):
        validate_input_tables(
            _objects(),
            feature_cols=["area"],
            fov_col="fov_id",
            observation_unit=" ",
        )


@pytest.mark.parametrize(
    ("given", "canonical"),
    [
        ("cell", "cell"),
        ("Cells", "cell"),
        (" nuclei ", "nucleus"),
        ("Objects", "object"),
        ("slices", "slice"),
        ("FOV summary", "fov_summary"),
        ("fov-summaries", "fov_summary"),
        ("embedding", "embedding"),
        ("tile", "tile"),
    ],
)
def test_observation_unit_is_normalized_to_one_label(given, canonical):
    # "cells" and "Cell" must not count as a different unit from "cell".
    result = validate_input_tables(
        _objects(), feature_cols=["area"], fov_col="fov_id", observation_unit=given
    )
    assert result.observation_unit == canonical


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


def test_nullable_numeric_features_preserve_missing_values_as_nan():
    objects = _objects()
    objects["area"] = pd.Series([10.0, pd.NA, 20.0], dtype="Float64")
    result = validate_input_tables(objects, feature_cols=["area"], fov_col="fov_id")
    values = result.feature_values("area")
    assert values.dtype == float
    assert np.isnan(values[1])


def test_group_metadata_can_live_only_in_fov_table_when_fov_id_is_unique():
    objects = pd.DataFrame(
        {
            "position_id": ["P1", "P1", "P2"],
            "area": [1.0, 2.0, 3.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "drug"],
            "repeat": ["R1", "R1"],
            "position_id": ["P1", "P2"],
        }
    )
    result = validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    assert result.group_cols == ("condition", "repeat")
    assert list(result.objects["condition"]) == ["control", "control", "drug"]
    assert list(result.objects["repeat"]) == ["R1", "R1", "R1"]


def test_group_metadata_is_not_guessed_when_fov_id_is_reused():
    objects = pd.DataFrame(
        {
            "image_file": ["same.tif", "same.tif"],
            "area": [1.0, 2.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "drug"],
            "repeat": ["R1", "R1"],
            "image_file": ["same.tif", "same.tif"],
        }
    )
    with pytest.raises(ValueError, match="cannot attach.*condition"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])


def test_the_cached_feature_matrix_is_a_fresh_copy_and_can_be_cleared():
    objects = pd.DataFrame({"image_file": ["a", "a", "b"], "x": [1.0, 2.0, 3.0]})
    data = validate_input_tables(
        objects, fovs=pd.DataFrame({"image_file": ["a", "b"]}), feature_cols=["x"]
    )
    first = data.feature_matrix()
    first[:] = -1.0  # a caller editing its array must not change the input
    assert data.feature_matrix().ravel().tolist() == [1.0, 2.0, 3.0]
    data.objects.loc[0, "x"] = 10.0  # editing the table in place
    data.clear_cache()
    assert data.feature_matrix().ravel().tolist() == [10.0, 2.0, 3.0]


def test_feature_problems_are_reported_in_column_order():
    fovs = pd.DataFrame({"image_file": ["a"]})
    both = pd.DataFrame({"image_file": ["a"], "x": [np.inf], "label": ["s"]})
    with pytest.raises(ValueError, match="'x' contains infinite values"):
        validate_input_tables(both, fovs=fovs, feature_cols=["x", "label"])
    with pytest.raises(TypeError, match="'label' must be numeric"):
        validate_input_tables(both, fovs=fovs, feature_cols=["label", "x"])
    missing = pd.DataFrame({"image_file": ["a", "zz"], "x": [1.0, 2.0]})
    with pytest.raises(
        ValueError, match=r"absent from the supplied fovs table, e.g. \[\('zz',\)\]"
    ):
        validate_input_tables(missing, fovs=fovs, feature_cols=["x"])
