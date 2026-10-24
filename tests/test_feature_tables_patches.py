"""Patch tables (#404): ``patch_id`` identifies an observation within its field, after
``label_id``, so grid and random patches (several per field, an empty ``label_id`` as CellPatcher
writes it) and a feature table that carries the patch id (MorphoRelate's
``features_patches.csv``) attach to their observations. A ``label_id`` or ``patch_id`` empty in
every row is read as absent; one with values must be in every table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import _tables

from morphocoverage.feature_tables import attach_feature_tables, observation_identity

FIELD = ["condition", "biological_repeat", "image_file"]


def _grid_patches() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three grid patches a field with an empty ``label_id`` column, as CellPatcher writes
    them; the patch ids restart in every field, as CellPatcher's positional ids repeat across
    the folders of a tree run. The feature table is keyed by the field and ``patch_id`` only."""
    objects, fovs, _ = _tables(cells=1)
    rows = [
        {**row, "label_id": np.nan, "patch_id": f"image_0000_grid_{p:05d}", "area": i * 3.0 + p}
        for i, row in enumerate(objects[FIELD].to_dict("records"))
        for p in range(3)
    ]
    patches = pd.DataFrame(rows)
    features = patches[[*FIELD, "patch_id"]].copy()
    features["mean_intensity"] = patches["area"] * 2.0
    return patches, fovs, features.iloc[::-1].reset_index(drop=True)


def test_grid_patches_with_an_empty_label_id_are_identified_by_the_field_and_patch_id() -> None:
    patches, fovs, features = _grid_patches()
    assert observation_identity(patches, fovs) == (*FIELD, "patch_id")
    result = attach_feature_tables(patches, fovs, {"morphorelate": (features, "explainable")})
    assert result.columns == {"morphorelate": ("mean_intensity",)}
    pd.testing.assert_series_equal(
        result.objects["mean_intensity"], patches["area"] * 2.0, check_names=False
    )


def test_an_empty_label_id_in_the_feature_table_is_read_as_absent_too() -> None:
    patches, fovs, features = _grid_patches()
    features["label_id"] = np.nan  # MorphoRelate's grid-patch table writes the column empty
    result = attach_feature_tables(patches, fovs, {"morphorelate": features})
    assert result.columns == {"morphorelate": ("mean_intensity",)}


def test_cell_mode_patches_carry_label_id_and_patch_id() -> None:
    objects, fovs, embedding = _tables()
    objects["patch_id"] = [f"image_0000_cell_{k:05d}" for k in objects["label_id"]]
    keys = objects[[*FIELD, "label_id", "patch_id"]]
    table = keys.merge(embedding, on=[*FIELD, "label_id"])
    result = attach_feature_tables(objects, fovs, {"morphorelate": (table, "explainable")})
    assert result.identity == (*FIELD, "label_id", "patch_id")
    assert result.columns == {"morphorelate": ("e0", "e1", "e2")}


def test_a_table_keyed_by_label_id_only_is_refused_beside_patch_observations() -> None:
    """Cell-mode patches carry ``label_id`` and ``patch_id``: an embedding keyed by
    ``label_id`` alone cannot say which patch a row is, and the message says why."""
    objects, fovs, embedding = _tables()
    objects["patch_id"] = [f"image_0000_cell_{k:05d}" for k in objects["label_id"]]
    with pytest.raises(ValueError) as error:
        attach_feature_tables(objects, fovs, {"DINO": embedding})
    message = str(error.value)
    assert "lacks the identity columns ['patch_id']" in message
    assert "patch_id has values in the observation table, so every table must carry it" in message


def test_an_empty_patch_id_is_read_as_absent() -> None:
    objects, fovs, embedding = _tables()
    objects["patch_id"] = ""  # a producer that writes every column, with no patches
    assert "patch_id" not in observation_identity(objects, fovs)
    result = attach_feature_tables(objects, fovs, {"DINO": embedding})
    assert result.columns == {"DINO": ("e0", "e1", "e2")}


def test_a_feature_table_without_the_patch_id_of_grid_patches_is_refused() -> None:
    patches, fovs, features = _grid_patches()
    with pytest.raises(ValueError, match="lacks the identity columns \\['patch_id'\\]"):
        attach_feature_tables(patches, fovs, {"t": features.drop(columns="patch_id")})


def test_shared_identities_name_both_columns_that_tell_observations_apart() -> None:
    patches, fovs, features = _grid_patches()
    patches["patch_id"] = "image_0000_grid_00000"
    with pytest.raises(ValueError, match="label_id for objects, patch_id for patches"):
        attach_feature_tables(patches, fovs, {"t": features})


def test_patch_id_is_never_a_feature_of_another_table() -> None:
    objects, fovs, embedding = _tables()
    embedding["patch_id"] = 1.0  # numeric, still metadata
    with pytest.raises(ValueError, match="metadata columns"):
        attach_feature_tables(objects, fovs, {"t": embedding})
