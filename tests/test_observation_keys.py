"""How a feature table's rows are matched to observations (``_observation_keys``).

Guards against: a feature row of FOV ``"1"`` attached to the observation of FOV ``1``, two
composite keys joined into one text by a separator inside a value, an observation without a
``label_id`` or ``patch_id`` matched as the text ``"nan"``, and MorphoFeature's patch identity
(``patch_set``, ``object_set``, ``parent_id``) refused, or read as a feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import _tables

from morphocoverage.feature_tables import attach_feature_tables

FIELD = ["condition", "biological_repeat", "image_file"]


def _one_field(fov: object, other: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    objects = pd.DataFrame({"position_id": pd.Series([fov], dtype=object), "area": [10.0]})
    table = pd.DataFrame({"position_id": pd.Series([other], dtype=object), "d0": [999.0]})
    return objects, table


@pytest.mark.parametrize(("fov", "other"), [(1, "1"), (1.5, "1.5"), ("True", True)])
def test_a_feature_row_of_another_typed_fov_is_not_attached(fov: object, other: object) -> None:
    """FOV 1 and FOV "1" are two fields (#420): the row matches no observation."""
    objects, table = _one_field(fov, other)
    with pytest.raises(ValueError, match="matching no observation"):
        attach_feature_tables(objects, None, {"dino": table})
    same, own = _one_field(fov, fov)
    assert attach_feature_tables(same, None, {"dino": own}).objects["d0"].tolist() == [999.0]


def test_a_separator_inside_a_value_does_not_join_two_keys() -> None:
    """("a\\x1fb", "c") and ("a", "b\\x1fc") are two observations, each matched to its row."""
    objects = pd.DataFrame(
        {"condition": ["a\x1fb", "a"], "position_id": ["c", "b\x1fc"], "area": [1.0, 2.0]}
    )
    table = objects[["condition", "position_id"]].iloc[::-1].assign(d0=[20.0, 10.0])
    attached = attach_feature_tables(objects, None, {"dino": table}).objects
    assert attached["d0"].tolist() == [10.0, 20.0]


def test_a_separator_inside_a_within_field_identifier_does_not_join_two_keys() -> None:
    """The within-field columns are matched as untyped text, so only the escaping keeps
    label ``"a\\x1fb"`` of patch ``"c"`` apart from label ``"a"`` of patch ``"b\\x1fc"``."""
    objects = pd.DataFrame(
        {
            "position_id": ["f1", "f1"],
            "label_id": ["a\x1fb", "a"],
            "patch_id": ["c", "b\x1fc"],
            "area": [1.0, 2.0],
        }
    )
    table = objects[["position_id", "label_id", "patch_id"]].iloc[::-1].assign(d0=[20.0, 10.0])
    attached = attach_feature_tables(objects, None, {"dino": table}).objects
    assert attached["d0"].tolist() == [10.0, 20.0]


def test_a_feature_table_writing_the_field_key_with_another_type_is_told_so() -> None:
    """A CSV read of a feature table turns repeat ``"1"`` into the number 1: its rows print
    like the observations', so the refusal names the column and both types (#420)."""
    objects = pd.DataFrame(
        {
            "biological_repeat": ["1", "2"],
            "position_id": ["001", "001"],
            "label_id": [1, 1],
            "area": [1.0, 2.0],
        }
    )
    table = objects[["biological_repeat", "position_id", "label_id"]].assign(
        biological_repeat=[1, 2], d0=[3.0, 4.0]
    )
    with pytest.raises(ValueError, match="writes 'biological_repeat' as int where") as error:
        attach_feature_tables(objects, None, {"dino": table})
    assert "observation table writes it as str" in str(error.value)
    assert "dtype={'biological_repeat': str}" in str(error.value)
    assert "'position_id' as" not in str(error.value)  # its types agree


def test_whole_numbers_written_as_decimals_still_match_within_a_field() -> None:
    """The CSV compatibility of label_id is kept: 1.0 in the feature table is label 1."""
    objects, fovs, embedding = _tables()
    embedding["label_id"] = embedding["label_id"].astype(float)
    result = attach_feature_tables(objects, fovs, {"dino": embedding})
    assert result.incomplete == {"dino": 0}


@pytest.mark.parametrize("missing", [np.nan, None, pd.NA, "", "   "])
def test_a_partly_missing_label_id_is_refused_in_the_observations(missing: object) -> None:
    """A label_id empty in some observations is not an identity (#596)."""
    objects, fovs, embedding = _tables()
    objects["label_id"] = objects["label_id"].astype(object)
    objects.loc[0, "label_id"] = missing
    with pytest.raises(ValueError, match="'label_id' in 1 row"):
        attach_feature_tables(objects, fovs, {"dino": embedding})


def test_a_partly_missing_patch_id_is_refused_in_a_feature_table() -> None:
    """Missing in some rows of a feature table: refused, never matched as text (#596)."""
    objects, fovs, _ = _tables()
    objects["patch_id"] = [f"p{k}" for k in objects["label_id"]]
    table = objects[[*FIELD, "label_id", "patch_id"]].assign(e0=1.0)
    table["patch_id"] = table["patch_id"].astype(object)
    table.loc[2, "patch_id"] = np.nan
    with pytest.raises(ValueError, match="no 'patch_id' in 1 row"):
        attach_feature_tables(objects, fovs, {"dino": table})


def test_missing_and_the_text_nan_are_never_one_identity() -> None:
    """A missing label_id in an observation cannot match a literal "nan" of a table (#596)."""
    objects, fovs, embedding = _tables()
    objects["label_id"] = objects["label_id"].astype(object)
    objects.loc[0, "label_id"] = np.nan
    embedding["label_id"] = embedding["label_id"].astype(object)
    embedding.loc[embedding.index[-1], "label_id"] = "nan"
    with pytest.raises(ValueError, match="'label_id' in 1 row"):
        attach_feature_tables(objects, fovs, {"dino": embedding})


def test_ids_repeated_across_fields_are_valid_and_twice_in_a_field_are_not() -> None:
    """label_id 1 in every field is fine; twice in one field is refused."""
    objects, fovs, embedding = _tables()
    assert attach_feature_tables(objects, fovs, {"dino": embedding}).identity[-1] == "label_id"
    twice = objects.assign(label_id=1)
    with pytest.raises(ValueError, match="share theirs"):
        attach_feature_tables(twice, fovs, {"dino": embedding})


def test_an_empty_label_id_everywhere_is_still_read_as_absent() -> None:
    """Grid patches (#404): a label_id empty in every observation is absent, not partial."""
    objects, fovs, embedding = _tables(cells=1)
    objects["label_id"] = np.nan
    table = embedding.drop(columns="label_id")
    assert attach_feature_tables(objects, fovs, {"dino": table}).identity == tuple(FIELD)


def _patches() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """MorphoFeature's square-cover patches: the full D84 identity in both tables."""
    objects, fovs, _ = _tables(cells=1)
    objects = objects.drop(columns="label_id")
    objects["patch_set"] = "tiles_squares"
    objects["patch_id"] = [f"{f}/t0000/sq0" for f in objects["image_file"]]
    objects["parent_id"] = [f"{f}/whole" for f in objects["image_file"]]
    table = objects[[*FIELD, "patch_set", "patch_id", "parent_id"]].assign(e0=1.0, e1=2.0)
    return objects, fovs, table.iloc[::-1].reset_index(drop=True)


def test_the_full_patch_identity_is_accepted_in_a_feature_table() -> None:
    """patch_set and parent_id beside the match key are checked, never features (#603)."""
    objects, fovs, table = _patches()
    result = attach_feature_tables(objects, fovs, {"embed": (table, "non-explainable")})
    assert result.columns == {"embed": ("e0", "e1")}


def test_patch_identity_that_disagrees_with_the_observation_is_refused() -> None:
    """A row naming another patch set or parent than its observation's is refused (#603)."""
    objects, fovs, table = _patches()
    table.loc[0, "parent_id"] = "elsewhere"
    with pytest.raises(ValueError, match="'parent_id' values that differ"):
        attach_feature_tables(objects, fovs, {"embed": table})


def test_patch_identity_the_observations_lack_is_never_a_feature() -> None:
    """A numeric parent_id in a feature table alone was attached as a feature; refused."""
    objects, fovs, embedding = _tables()
    with pytest.raises(ValueError, match="metadata columns"):
        attach_feature_tables(objects, fovs, {"dino": embedding.assign(parent_id=7)})
