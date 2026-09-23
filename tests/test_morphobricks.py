import pandas as pd
import pytest

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    run_strategies,
    validate_input_tables,
)


def _morphobricks_tables():
    objects = pd.DataFrame(
        {
            "condition": ["control", "control", "drug", "drug"],
            "repeat": ["R1", "R2", "R1", "R1"],
            "image_file": ["a.tif", "b.tif", "c.tif", "d.tif"],
            "mask_file": ["a_mask.tif", "b_mask.tif", "c_mask.tif", "d_mask.tif"],
            "patch_file": ["a.png", "b.png", "c.png", "d.png"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control", "drug", "drug"],
            "repeat": ["R1", "R2", "R1", "R1"],
            "image_file": ["a.tif", "b.tif", "c.tif", "d.tif"],
        }
    )
    return objects, fovs


def test_morphobricks_defaults_resolve_image_file_and_condition_repeat():
    objects, fovs = _morphobricks_tables()
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=["area"],
    )
    assert data.fov_col == "image_file"
    assert data.group_cols == ("condition", "repeat")


def test_run_strategies_keeps_morphobricks_groups_separate_by_default():
    objects, fovs = _morphobricks_tables()
    results = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )

    assert len(results) == 3
    groups = {(r.group["condition"], r.group["repeat"]) for r in results}
    assert groups == {("control", "R1"), ("control", "R2"), ("drug", "R1")}

    by_group = {
        (r.group["condition"], r.group["repeat"]): (r.n_fovs, r.decision)
        for r in results
    }
    assert by_group[("control", "R1")] == (1, "continue")
    assert by_group[("control", "R2")] == (1, "continue")
    assert by_group[("drug", "R1")] == (2, "stop")


def test_explicit_empty_group_cols_allows_intentional_pooling():
    objects, fovs = _morphobricks_tables()
    results = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedFOVStrategy(4)],
        feature_cols=["area"],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        group_cols=[],
    )
    assert len(results) == 1
    assert results[0].group == {}
    assert results[0].n_fovs == 4
    assert results[0].decision == "stop"


def test_explicit_fov_col_overrides_morphobricks_default():
    objects, _ = _morphobricks_tables()
    objects["fov_id"] = ["F1", "F2", "F3", "F4"]
    data = validate_input_tables(
        objects,
        feature_cols=["area"],
        fov_col="fov_id",
        group_cols=[],
    )
    assert data.fov_col == "fov_id"


def test_position_id_is_preferred_over_image_file_for_fov_identity():
    objects = pd.DataFrame(
        {
            "condition": ["control"] * 4,
            "repeat": ["R1"] * 4,
            "position_id": ["P1", "P1", "P2", "P2"],
            "image_file": [
                "P1_z000.tif",
                "P1_z001.tif",
                "P2_z000.tif",
                "P2_z001.tif",
            ],
            "z": [0, 1, 0, 1],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control"],
            "repeat": ["R1", "R1"],
            "position_id": ["P1", "P2"],
            "acquisition_order": [1, 2],
        }
    )

    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    assert data.fov_col == "position_id"
    assert data.fov_key == ("condition", "repeat", "position_id")
    assert len(data.fovs) == 2


def test_z_c_t_are_within_fov_metadata_not_phenotype_features():
    objects = pd.DataFrame(
        {
            "position_id": ["P1", "P1"],
            "z": [0, 1],
            "c": [0, 0],
            "t": [0, 0],
            "area": [1.0, 2.0],
        }
    )
    fovs = pd.DataFrame({"position_id": ["P1"]})

    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    assert data.fov_col == "position_id"
    for metadata_col in ("z", "c", "t"):
        with pytest.raises(ValueError, match="within-FOV acquisition metadata"):
            validate_input_tables(
                objects,
                fovs=fovs,
                feature_cols=["area", metadata_col],
            )
