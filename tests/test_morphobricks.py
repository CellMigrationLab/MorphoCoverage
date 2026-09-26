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

    by_group = {(r.group["condition"], r.group["repeat"]): (r.n_fovs, r.decision) for r in results}
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


def test_z_c_t_can_be_preserved_as_within_fov_metadata():
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
    assert {"z", "c", "t"} <= set(data.objects.columns)
    assert data.feature_cols == ("area",)


def test_short_name_c_can_still_be_an_explicit_scientific_feature():
    objects = pd.DataFrame({"fov_id": ["F1", "F2"], "c": [1.0, 2.0]})
    data = validate_input_tables(objects, feature_cols=["c"])
    assert data.feature_cols == ("c",)


def test_multi_file_position_requires_within_fov_axis_metadata():
    objects = pd.DataFrame(
        {
            "position_id": ["P1", "P1"],
            "image_file": ["P1_part1.tif", "P1_part2.tif"],
            "area": [1.0, 2.0],
        }
    )
    fovs = pd.DataFrame({"position_id": ["P1"]})

    with pytest.raises(ValueError, match="multiple image_file.*no z/c/t"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])


def test_multi_file_position_rejects_rows_missing_axis_metadata():
    objects = pd.DataFrame(
        {
            "position_id": ["P1", "P1"],
            "image_file": ["P1_z000.tif", "P1_z001.tif"],
            "z": [0, None],
            "area": [1.0, 2.0],
        }
    )
    fovs = pd.DataFrame({"position_id": ["P1"]})

    with pytest.raises(ValueError, match="missing z/c/t"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])


# --- Planes, channels and frames must never silently become FOVs -------------


def _per_plane_files():
    """Two positions, each saved as one file per z-plane, with no position_id."""
    return pd.DataFrame(
        {
            "image_file": ["p1_z0.tif", "p1_z0.tif", "p1_z1.tif", "p2_z0.tif", "p2_z1.tif"],
            "z": [0, 0, 1, 0, 1],
            "area": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_per_plane_files_are_not_silently_counted_as_fovs():
    with pytest.raises(ValueError, match="one plane/channel/frame of a position"):
        validate_input_tables(_per_plane_files(), feature_cols=["area"])


def test_per_plane_files_are_accepted_with_position_id_or_an_explicit_fov_col():
    with_position = _per_plane_files().assign(position_id=["p1", "p1", "p1", "p2", "p2"])
    assert len(validate_input_tables(with_position, feature_cols=["area"]).fovs) == 2
    # Naming the column explicitly is the caller's statement that files are positions.
    explicit = validate_input_tables(
        _per_plane_files(), feature_cols=["area"], fov_col="image_file"
    )
    assert len(explicit.fovs) == 4


def test_a_z_stack_per_file_is_one_fov():
    stacks = pd.DataFrame(
        {"image_file": ["a.tif", "a.tif", "b.tif", "b.tif"], "z": [0, 1, 0, 1], "area": [1.0] * 4}
    )
    assert len(validate_input_tables(stacks, feature_cols=["area"]).fovs) == 2


def test_within_position_axes_must_tell_the_files_apart():
    # Two files of one position both claim z = 0: the axis does not identify them.
    objects = pd.DataFrame(
        {
            "position_id": ["P1"] * 4,
            "image_file": ["a.tif", "a.tif", "b.tif", "b.tif"],
            "z": [0, 0, 0, 0],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="do not tell the files of one position apart"):
        validate_input_tables(objects, feature_cols=["area"])
    objects["z"] = [0, 0, 1, 1]
    assert len(validate_input_tables(objects, feature_cols=["area"]).fovs) == 1


def test_multi_file_check_applies_to_any_fov_column_name():
    objects = pd.DataFrame(
        {"pos": ["P1", "P1"], "image_file": ["a.tif", "b.tif"], "area": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="multiple image_file values"):
        validate_input_tables(objects, feature_cols=["area"], fov_col="pos")


def test_technical_repeat_is_a_default_group_and_part_of_fov_identity_when_present():
    objects, fovs = _morphobricks_tables()
    objects["technical_repeat"] = ["w1", "w1", "w1", "w2"]
    fovs["technical_repeat"] = ["w1", "w1", "w1", "w2"]
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    assert data.group_cols == ("condition", "repeat", "technical_repeat")
    assert data.fov_key == ("condition", "repeat", "technical_repeat", "image_file")
    # Named explicitly, groups may leave it out; it still names the FOV.
    pooled = validate_input_tables(
        objects, fovs=fovs, feature_cols=["area"], group_cols=["condition", "repeat"]
    )
    assert pooled.group_cols == ("condition", "repeat")
    assert "technical_repeat" in pooled.fov_key
    with pytest.raises(ValueError, match="non-canonical spelling"):
        validate_input_tables(
            objects.rename(columns={"technical_repeat": "Technical_Repeat"}),
            fovs=fovs.rename(columns={"technical_repeat": "Technical_Repeat"}),
            feature_cols=["area"],
        )


def test_an_all_empty_technical_repeat_column_is_read_as_absent():
    """CellPatcher writes every column, so a run without wells leaves technical_repeat empty."""
    import numpy as np

    objects, fovs = _morphobricks_tables()
    for empty in (None, np.nan, ""):
        data = validate_input_tables(
            objects.assign(technical_repeat=empty),
            fovs=fovs.assign(technical_repeat=empty),
            feature_cols=["area"],
        )
        assert data.group_cols == ("condition", "repeat")
        assert "technical_repeat" not in data.fov_key
        assert "technical_repeat" not in data.objects.columns
    # Empty in the object table only (a feature table that kept the column).
    only_objects = validate_input_tables(
        objects.assign(technical_repeat=None), fovs=fovs, feature_cols=["area"]
    )
    assert only_objects.group_cols == ("condition", "repeat")
    # Partly empty is ambiguous and still refused.
    partial = ["w1", None, "w1", "w2"]
    with pytest.raises(ValueError, match="missing group values"):
        validate_input_tables(
            objects.assign(technical_repeat=partial),
            fovs=fovs.assign(technical_repeat=partial),
            feature_cols=["area"],
        )
    # Named explicitly, an empty column is kept and refused.
    with pytest.raises(ValueError, match="missing group values"):
        validate_input_tables(
            objects.assign(technical_repeat=None),
            fovs=fovs.assign(technical_repeat=None),
            feature_cols=["area"],
            group_cols=["condition", "repeat", "technical_repeat"],
        )


def test_an_all_empty_position_id_falls_back_to_image_file():
    """CellPatcher writes position_id empty when one file is one field."""
    objects, fovs = _morphobricks_tables()
    data = validate_input_tables(objects.assign(position_id=None), fovs=fovs, feature_cols=["area"])
    assert data.fov_col == "image_file"
    both = validate_input_tables(
        objects.assign(position_id=""), fovs=fovs.assign(position_id=""), feature_cols=["area"]
    )
    assert both.fov_col == "image_file" and "position_id" not in both.objects.columns
    # Named as the FOV column, an empty position_id is kept and refused.
    with pytest.raises(ValueError):
        validate_input_tables(
            objects.assign(position_id=None),
            fovs=fovs.assign(position_id=None),
            feature_cols=["area"],
            fov_col="position_id",
        )
