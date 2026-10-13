"""Tests of ``planes``: z, c and t planes are not fields, and ``planes_as_features``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_tables import _agree, _error_titles, _four_cell_tables, _planes

from morphocoverage import planes_as_features, validate_input_tables
from morphocoverage.workflows import Setup


def test_a3_plane_coordinates_are_never_features_and_the_pivot_splits_them():
    cells = _planes()
    fovs = cells[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    for plane in ("z", "c", "t", "z0", "z1"):
        table = cells.assign(**{plane: cells["z"]})
        accepted, check = _agree(table, fovs, feature_cols=["area", plane])
        assert not accepted and "within-field coordinate" in _error_titles(check)[0]
        assert "planes_as_features" in check.errors[0].fix
    table, names = planes_as_features(cells, fovs, features=["area"])
    assert names == ["area_z0", "area_z1", "area_z2"]
    assert len(table) == 8 and "z" not in table.columns and "image_file" not in table.columns
    first = cells[(cells.position_id == "p0") & (cells.label_id == 1)].sort_values("z")
    assert table.loc[0, names].tolist() == first["area"].tolist()
    data = validate_input_tables(table, fovs=fovs, feature_cols=names, observation_unit="cell")
    assert len(data.fovs) == 2 and data.n_observations == 8  # the field stays one FOV
    with pytest.raises(ValueError, match="cannot be matched across planes"):
        planes_as_features(cells.drop(columns="label_id"), fovs, features=["area"])
    with pytest.raises(ValueError, match="one to one"):
        planes_as_features(pd.concat([cells, cells.iloc[:1]]), fovs, features=["area"])
    with pytest.raises(ValueError, match="has no row at z=0"):
        planes_as_features(cells.iloc[1:], fovs, features=["area"])
    # Per-plane files with no position identifier: each "field" is one plane.
    files = cells.drop(columns="position_id")
    with pytest.raises(ValueError, match="names planes, not stage positions"):
        planes_as_features(
            files, files[["condition", "biological_repeat", "image_file"]], features=["area"]
        )
    # The split names are what a Setup declares: pivot first, then set up.
    setup = Setup(readout="area_z0", features=tuple(names[1:]), observation_unit="cell")
    assert setup.columns == names
    assert setup.prepared(table, fovs) is table


def test_p5_10_planes_as_features_missing_planes_and_cellpatcher_z0():
    rows = []
    for position in ("p0", "p1"):
        depth = 3 if position == "p0" else 4  # stacks of different depth
        for label in (1, 2):
            planes = range(depth) if label == 1 else range(1, 3)  # label 2 spans two planes
            rows += [
                {
                    "condition": "A",
                    "biological_repeat": "r1",
                    "position_id": position,
                    "label_id": label,
                    "z": z,
                    "area": 10.0 * label + z,
                }
                for z in planes
            ]
    stack = pd.DataFrame(rows)
    fovs = stack[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    with pytest.raises(ValueError, match="allow_missing_planes=True"):
        planes_as_features(stack, fovs, features=["area"])
    wide, names = planes_as_features(stack, fovs, features=["area"], allow_missing_planes=True)
    assert names == ["area_z0", "area_z1", "area_z2", "area_z3"]
    assert len(wide) == 4 and int(wide[names].isna().sum().sum()) == 16 - 11
    assert wide.loc[(wide.position_id == "p0") & (wide.label_id == 1), "area_z3"].isna().all()
    # CellPatcher's z0 is the z axis; a patch spanning planes is refused.
    patches = stack.drop(columns="z").assign(z0=stack["z"], z1=stack["z"] + 1)
    wide_z0, names_z0 = planes_as_features(
        patches, fovs, features=["area"], allow_missing_planes=True
    )
    assert names_z0 == names and wide_z0[names].equals(wide[names])
    with pytest.raises(ValueError, match="span several planes"):
        planes_as_features(patches.assign(z1=patches["z0"] + 2), fovs, features=["area"])
    # Text against numbers is named as such; a label such as 'max' stays a plane.
    mixed = stack.astype({"z": object})
    mixed.loc[mixed.position_id == "p0", "z"] = mixed.loc[mixed.position_id == "p0", "z"].map(str)
    with pytest.raises(ValueError, match="text in some rows and a number in others"):
        planes_as_features(mixed, fovs, features=["area"], allow_missing_planes=True)
    with pytest.raises(TypeError, match="True or False"):
        planes_as_features(stack, fovs, features=["area"], allow_missing_planes="yes")


@pytest.mark.parametrize(
    ("arguments", "kind", "message"),
    [
        (
            {"features": []},
            ValueError,
            "features must name at least one feature column to split per plane",
        ),
        (
            {"features": ["area"], "axes": []},
            ValueError,
            "axes must name at least one of 'z', 'c', 't'",
        ),
        ({"features": ["area", "area"]}, ValueError, "features names 'area' more than once"),
        (
            {"features": ["area"], "axes": ["q"]},
            ValueError,
            "axes must be among ['z', 'c', 't'] (or 'z0'), not ['q']",
        ),
        (
            {"features": ["area"], "object_col": 3},
            TypeError,
            "object_col must be the name of the column identifying each object",
        ),
        (
            {"features": ["area"], "object_col": "cell"},
            ValueError,
            (
                "objects has no 'cell' column, so objects cannot be matched across planes; "
                "name the column that identifies each object within its field with object_col"
            ),
        ),
        (
            {"features": ["area"], "axes": ["c"]},
            ValueError,
            "objects has no 'c' column to split the features along",
        ),
        ({"features": ["nope"]}, ValueError, "feature 'nope' is not a column of objects"),
        ({"features": ["condition"]}, TypeError, "feature 'condition' must be real numbers"),
        ({"features": ["z"]}, ValueError, "feature 'z' is also an axis or the object identifier"),
    ],
)
def test_planes_as_features_refusals(arguments, kind, message):
    objects, fovs = _four_cell_tables()
    with pytest.raises(kind) as error:
        planes_as_features(objects, fovs, **arguments)
    assert str(error.value) == message


def test_planes_as_features_refuses_a_missing_plane():
    objects, fovs = _four_cell_tables()
    with pytest.raises(ValueError) as error:
        planes_as_features(objects.assign(z=[0, np.nan, 0, 0]), fovs, features=["area"])
    assert str(error.value) == "objects['z'] is missing in some rows, so their plane is unknown"
