"""Patch tables (#404): ``patch_id`` identifies an observation within its field, after
``label_id``, so grid and random patches (no label, several per field) and a feature table that
carries the patch id (MorphoRelate's ``features_patches.csv``) attach to their observations."""

from __future__ import annotations

import pandas as pd
import pytest
from _helpers_feature_tables import _tables

from morphocoverage.feature_tables import attach_feature_tables, observation_identity

FIELD = ["condition", "biological_repeat", "image_file"]


def _grid_patches() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three grid patches a field, without label_id; patch ids restart in every field, as
    CellPatcher's positional ids repeat across the leaves of a tree run."""
    objects, fovs, _ = _tables(cells=1)
    objects = objects.drop(columns="label_id")
    rows = [
        {**row, "patch_id": f"image_0000_grid_{p:05d}", "area": float(i * 3 + p)}
        for i, row in enumerate(objects[FIELD].to_dict("records"))
        for p in range(3)
    ]
    patches = pd.DataFrame(rows)
    features = patches[[*FIELD, "patch_id"]].copy()
    features["mean_intensity"] = patches["area"] * 2.0
    return patches, fovs, features.iloc[::-1].reset_index(drop=True)


def test_grid_patches_are_identified_by_the_field_key_and_patch_id() -> None:
    patches, fovs, features = _grid_patches()
    assert observation_identity(patches, fovs) == (*FIELD, "patch_id")
    result = attach_feature_tables(patches, fovs, {"morphorelate": (features, "explainable")})
    assert result.columns == {"morphorelate": ("mean_intensity",)}
    pd.testing.assert_series_equal(
        result.objects["mean_intensity"], patches["area"] * 2.0, check_names=False
    )


def test_cell_mode_patches_carry_label_id_and_patch_id() -> None:
    objects, fovs, embedding = _tables()
    objects["patch_id"] = [f"image_0000_cell_{k:05d}" for k in objects["label_id"]]
    keys = objects[[*FIELD, "label_id", "patch_id"]]
    table = keys.merge(embedding, on=[*FIELD, "label_id"])
    result = attach_feature_tables(objects, fovs, {"morphorelate": (table, "explainable")})
    assert result.identity == (*FIELD, "label_id", "patch_id")
    assert result.columns == {"morphorelate": ("e0", "e1", "e2")}


def test_a_table_without_the_patch_id_of_its_observations_is_refused() -> None:
    patches, fovs, features = _grid_patches()
    with pytest.raises(ValueError, match="patch_id"):
        attach_feature_tables(patches, fovs, {"t": features.drop(columns="patch_id")})


def test_patch_id_is_never_a_feature_of_another_table() -> None:
    objects, fovs, embedding = _tables()
    embedding["patch_id"] = 1.0  # numeric, still metadata
    with pytest.raises(ValueError, match="metadata columns"):
        attach_feature_tables(objects, fovs, {"t": embedding})
