from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _well_tables_with_empty_fields
from _helpers_tables import _four_cell_tables, _validated_mixed

import morphocoverage.plots as plots_module
from morphocoverage import ValidatedInput, schema, validate_input_tables
from morphocoverage.evaluator import restrict_input
from morphocoverage.plots import plots_of_data
from morphocoverage.schema import _validate_input_tables, canonical_fields


def _objects():
    return pd.DataFrame(
        {
            "condition": ["control", "control", "control"],
            "biological_repeat": ["R1", "R1", "R1"],
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
        group_cols=["condition", "biological_repeat"],
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
            feature_cols=["area", "biological_repeat"],
            fov_col="fov_id",
            group_cols=["biological_repeat"],
        )


def test_fov_table_preserves_zero_object_fov():
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control", "control"],
            "biological_repeat": ["R1", "R1", "R1"],
            "fov_id": ["F1", "F2", "F3"],
            "stage_x": [0.0, 100.0, 200.0],
        }
    )
    result = validate_input_tables(
        _objects(),
        feature_cols=["area", "texture"],
        fov_col="fov_id",
        fovs=fovs,
        group_cols=["condition", "biological_repeat"],
    )
    assert set(result.fovs["fov_id"]) == {"F1", "F2", "F3"}
    assert "F3" not in set(result.objects["fov_id"])


def test_rejects_object_fov_missing_from_fov_table():
    fovs = pd.DataFrame({"condition": ["control"], "biological_repeat": ["R1"], "fov_id": ["F1"]})
    with pytest.raises(ValueError, match="absent from the supplied fovs table"):
        validate_input_tables(
            _objects(),
            feature_cols=["area"],
            fov_col="fov_id",
            fovs=fovs,
            group_cols=["condition", "biological_repeat"],
        )


def test_allows_same_fov_id_in_different_groups():
    objects = pd.DataFrame(
        {"biological_repeat": ["R1", "R2"], "fov_id": ["F1", "F1"], "area": [10.0, 20.0]}
    )
    fovs = pd.DataFrame({"biological_repeat": ["R1", "R2"], "fov_id": ["F1", "F1"]})
    result = validate_input_tables(
        objects,
        feature_cols=["area"],
        fov_col="fov_id",
        fovs=fovs,
        group_cols=["biological_repeat"],
    )
    assert len(result.fovs) == 2


def test_rejects_duplicate_fov_rows_within_group():
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control"],
            "biological_repeat": ["R1", "R1"],
            "fov_id": ["F1", "F1"],
        }
    )
    with pytest.raises(ValueError, match="exactly one row"):
        validate_input_tables(
            _objects(),
            feature_cols=["area"],
            fov_col="fov_id",
            fovs=fovs,
            group_cols=["biological_repeat"],
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
            "biological_repeat": ["R1", "R1"],
            "position_id": ["P1", "P2"],
        }
    )
    result = validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    assert result.group_cols == ("condition", "biological_repeat")
    assert list(result.objects["condition"]) == ["control", "control", "drug"]
    assert list(result.objects["biological_repeat"]) == ["R1", "R1", "R1"]


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
            "biological_repeat": ["R1", "R1"],
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


def test_a_copy_with_other_tables_never_shares_the_cache():
    # OnlineSession joins each new FOV with dataclasses.replace; a shared cache
    # would return the first FOV's features for every later prefix.
    from dataclasses import replace

    from morphocoverage.schema import fov_positions

    objects = pd.DataFrame({"image_file": ["a", "b", "b"], "x": [1.0, 2.0, 3.0]})
    fovs = pd.DataFrame({"image_file": ["a", "b"]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    first = replace(data, objects=objects.iloc[:1], fovs=fovs.iloc[:1])
    assert first.feature_matrix().ravel().tolist() == [1.0]
    assert fov_positions(first, first.fovs).tolist() == [0]
    joined = replace(first, objects=objects, fovs=fovs)
    assert joined.feature_matrix().ravel().tolist() == [1.0, 2.0, 3.0]
    assert fov_positions(joined, joined.fovs).tolist() == [0, 1, 1]


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


def test_an_observation_of_an_unlisted_fov_is_named_when_groups_come_from_fovs():
    objects = pd.DataFrame({"image_file": ["a", "zz"], "x": [1.0, 2.0]})
    fovs = pd.DataFrame({"image_file": ["a"], "condition": ["c"], "biological_repeat": ["r"]})
    with pytest.raises(ValueError, match=r"absent from the supplied fovs table.*\['zz'\]"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["x"])


def test_the_canonical_fov_order_does_not_depend_on_row_order_when_ids_print_alike():
    forward, backward = _validated_mixed([0, 1, 2, 3]), _validated_mixed([3, 2, 1, 0])
    first = [forward.fovs["position_id"].iloc[i] for i in canonical_fields(forward)]
    second = [backward.fovs["position_id"].iloc[i] for i in canonical_fields(backward)]
    assert first == second


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


def test_direct_construction_runs_the_validators_role_and_plane_rules():
    """A hand-built input is refused where the validator refuses (#422): a plane coordinate as
    a feature, an object measured in two planes of one field, an identity list, or FOV ids
    that equality merges."""
    objects, fovs = _tables()
    with pytest.raises(ValueError, match="within-field coordinate 'z'"):
        ValidatedInput(**_fields(objects=objects.assign(z=0.0), feature_cols=("x", "z")))
    planes = objects.assign(z=[0, 1, 0, 0])
    with pytest.raises(ValueError, match="several 'z' values"):
        ValidatedInput(**_fields(objects=planes))
    assert ValidatedInput(**_fields(objects=planes, observation_unit="slice")).n_observations == 4
    listed = objects.assign(position_id=pd.Series([["f1"], ["f1"], ["f2"], ["f3"]]))
    with pytest.raises(TypeError, match="lists"):
        ValidatedInput(**_fields(objects=listed))
    typed = pd.Series([True, True, 2, 3], dtype=object)
    with pytest.raises(ValueError, match="equal as numbers"):
        ValidatedInput(
            **_fields(
                objects=objects.assign(position_id=typed),
                fovs=fovs.assign(position_id=pd.Series([1, 2, 3], dtype=object)),
            )
        )


@pytest.mark.parametrize(
    ("observed", "listed"),
    [(True, 1), (1, 1.0), (1, "1"), (False, 0), (0.0, -0.0)],
)
def test_a_field_is_never_matched_to_another_typed_identifier(observed, listed):
    """bool True is not int 1, int 1 not float 1.0 or text "1" (#420): the validator and
    check_tables refuse such identifiers instead of placing the observation in that field."""
    from morphocoverage.data_check import check_tables

    objects = pd.DataFrame({"fov_id": pd.Series([observed], dtype=object), "x": [2.0]})
    fovs = pd.DataFrame({"fov_id": pd.Series([listed], dtype=object)})
    with pytest.raises(ValueError, match="equal as numbers|absent from the supplied fovs"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    assert not check_tables(objects, fovs, feature_cols=["x"]).ok


def test_a_field_name_reused_in_two_repeats_stays_two_fields():
    """The typed check refuses only values equality merges; f1 of R1 and R2 are two fields."""
    repeats = pd.DataFrame({"biological_repeat": ["R1", "R2"], "position_id": ["f1", "f1"]})
    data = validate_input_tables(repeats.assign(x=[1.0, 2.0]), fovs=repeats, feature_cols=["x"])
    assert schema.fov_positions(data, data.fovs).tolist() == [0, 1]
    with pytest.raises(ValueError, match="control character"):
        validate_input_tables(
            repeats.assign(x=[1.0, 2.0], position_id=["f\x1f1", "f1"]),
            fovs=repeats.assign(position_id=["f\x1f1", "f1"]),
            feature_cols=["x"],
        )


def test_the_frameworks_own_subsets_are_trusted_and_keep_their_state():
    objects, fovs = _tables()
    data = validate_input_tables(objects=objects, fovs=fovs, feature_cols=["x"])
    subset = restrict_input(data, data.objects.iloc[:2], data.fovs.iloc[:1])
    assert len(subset.fovs) == 1 and subset.feature_cols == data.feature_cols
    copy = data.with_tables(feature_cols=("x",))
    assert copy is not data and copy._feature_cache == {}


def test_j3_j9_j11_plots_use_the_shared_helpers():
    source = inspect.getsource(plots_module)
    assert "sampling_unit" not in source.split('"""', 2)[2]  # no private import from there
    assert not hasattr(plots_module, "_technical_count")
    assert not hasattr(plots_module, "_ordinal")
    assert plots_module.fov_values is schema.fov_values
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        plots_of_data(
            *_well_tables_with_empty_fields([("ctrl", "r1", "w", 2, 0, 0)]),
            features=["x"],
            n_boot=True,
        )


def test_the_validator_refuses_tables_that_are_not_tables():
    objects, _ = _four_cell_tables()
    with pytest.raises(TypeError) as error:
        _validate_input_tables([1], feature_cols=["area"])
    assert str(error.value) == "objects must be a pandas DataFrame"
    with pytest.raises(TypeError) as error:
        _validate_input_tables(objects, feature_cols=["area"], fovs=[1])
    assert str(error.value) == "fovs must be a pandas DataFrame when provided"
