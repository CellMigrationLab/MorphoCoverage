"""Tests of ``metadata``: the MorphoBricks columns, their near-miss spellings and the groups."""

import datetime

import pandas as pd
import pytest
from _helpers_tables import (
    _BASE,
    _agree,
    _cells,
    _error_titles,
    _fields,
    _planes,
    _plate_tables,
    _six_cell_tables,
    _warning_titles,
)

from morphocoverage.metadata import canonical_spelling, column_names, empty_values


def test_a1_object_columns_without_text_are_never_blank():
    for values in (
        pd.Series([1, 2], dtype=object),
        pd.Series([datetime.date(2024, 1, 5)] * 2),
        pd.Series([True, False], dtype=object),
    ):
        assert empty_values(values).tolist() == [False, False]
    assert empty_values(pd.Series(["x", " ", None, 3], dtype=object)).tolist() == [
        False,
        True,
        True,
        False,
    ]
    dates = [datetime.date(2024, 1, 5)] * 2 + [datetime.date(2024, 2, 1)] * 2
    objects = _cells(biological_repeat=dates, position_id=["p1", "p2", "p1", "p2"])
    accepted, _ = _agree(objects, objects[["condition", "biological_repeat", "position_id"]])
    assert accepted


def test_a2_a_blank_category_is_a_missing_group_value():
    objects = _cells(condition=pd.Categorical(["drug", "drug", " ", " "]))
    assert empty_values(objects["condition"]).tolist() == [False, False, True, True]
    accepted, check = _agree(objects, _fields(objects))
    assert not accepted
    assert any("'condition' is empty in some rows" in t for t in _error_titles(check))


def test_a3_a_bare_string_or_a_non_collection_is_refused_by_both():
    objects = _cells()
    with pytest.raises(TypeError, match="not a string"):
        column_names("condition", "group_cols")
    for arguments in ({"group_cols": "condition"}, {"feature_cols": "area"}, {"group_cols": 3}):
        accepted, check = _agree(objects, _fields(objects), **arguments)
        assert not accepted and check.errors


@pytest.mark.parametrize(
    ("column", "meant"),
    [
        ("Conditions", "condition"),
        ("repeats", "biological_repeat"),
        ("PositionID", "position_id"),
        ("positionId", "position_id"),
        ("technicalRepeat", "technical_repeat"),
        ("TechnicalRepeats", "technical_repeat"),
        ("ImageFile", "image_file"),
        ("FOVID", "fov_id"),
        ("Repeat", "biological_repeat"),
        ("technical repeat", "technical_repeat"),
    ],
)
def test_a21_plurals_and_camel_case_are_near_misses(column, meant):
    assert canonical_spelling(column) == meant
    objects = _BASE.assign(**{column: "x"})
    fovs = _fields(_BASE).assign(**{column: "x"})
    accepted, check = _agree(objects, fovs)
    assert not accepted and any(column in t for t in _error_titles(check))


def test_a21_exact_names_and_unrelated_columns_are_not_near_misses():
    for column in (
        "condition",
        "biological_repeat",
        "image_file",
        "well",
        "area",
        "position",
        "reps",
    ):
        assert canonical_spelling(column) is None


def test_a1_upper_case_plane_coordinates_are_near_misses():
    # 'Channel' is a near miss of 'c' since audit 4 (P5-2: other names of the coordinates).
    assert [canonical_spelling(n) for n in ("Z", "T ", " c", "Z0", "z1", "zs", "Channel")] == [
        "z",
        "t",
        "c",
        "z0",
        None,
        None,
        "c",
    ]
    cells = _planes().drop(columns="position_id").rename(columns={"z": "Z"})
    fovs = cells[["condition", "biological_repeat", "image_file", "Z"]].drop_duplicates()
    accepted, check = _agree(cells, fovs)
    # 'Z' was ignored, so six plane files counted as six fields; now it is refused.
    assert not accepted and "within-field coordinate 'z'" in _error_titles(check)[0]


def test_a2_dots_and_pandas_copies_are_near_misses():
    assert canonical_spelling("technical.repeat") == "technical_repeat"
    assert (
        canonical_spelling("Repeat.1") == "biological_repeat"
        and canonical_spelling("repeat.1") == "biological_repeat"
    )
    cells, fovs = _six_cell_tables()
    cells["technical.repeat"] = ["w1", "w1", "w2", "w3", "w3", "w4"]
    fovs["technical.repeat"] = ["w1", "w2", "w3", "w4"]
    accepted, check = _agree(cells, fovs)
    assert not accepted and "'technical_repeat'" in _error_titles(check)[0]
    # Named explicitly, it is used as it is.
    accepted, _ = _agree(
        cells, fovs, group_cols=["condition", "biological_repeat", "technical.repeat"]
    )
    assert accepted


def test_p5_7_undeclared_group_columns_are_warned_about():
    cells, fovs = _plate_tables()
    assert canonical_spelling("Metadata_condition") == "condition"
    assert canonical_spelling("Metadata_Repeat") == "biological_repeat"
    renamed = {"biological_repeat": "replicate"}
    accepted, check = _agree(cells.rename(columns=renamed), fovs.rename(columns=renamed))
    assert accepted
    assert any("'replicate' looks like a group column" in t for t in _warning_titles(check))
    # Declared, or a column that only restates a group (a plate per repeat), says nothing.
    accepted, check = _agree(
        cells.rename(columns=renamed),
        fovs.rename(columns=renamed),
        group_cols=["condition", "replicate", "technical_repeat"],
    )
    assert not any("looks like a group" in t for t in _warning_titles(check))
    plate = fovs.assign(
        plate=fovs["biological_repeat"].map({"2026_01_01": "P1", "2026_01_02": "P2"})
    )
    assert not any("looks like a group" in t for t in _warning_titles(_agree(cells, plate)[1]))
    # A file name, one per field, is not a group.
    masks = fovs.assign(mask_name="m_" + fovs["image_file"])
    assert not any("looks like a group" in t for t in _warning_titles(_agree(cells, masks)[1]))
