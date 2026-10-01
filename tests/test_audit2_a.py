"""Regressions for the second audit, group A (validation, check_tables, objectives, provenance).

One test per finding, ``test_a<n>_...``. Many findings were places where
``check_tables`` and ``validate_input_tables`` disagreed or crashed; each such
test also checks that the two agree (``_agree``).
"""

import datetime
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import morphocoverage as mc
from morphocoverage import check_tables, validate_input_tables
from morphocoverage.metadata import canonical_spelling, column_names, empty_values
from morphocoverage.provenance import config_hash
from morphocoverage.results import combine_decisions


def _cells(**columns) -> pd.DataFrame:
    """Four cells in two fields of one condition and repeat, with ``columns`` added."""
    table = pd.DataFrame(
        {
            "condition": ["a"] * 4,
            "biological_repeat": ["r1"] * 4,
            "position_id": ["p1", "p1", "p2", "p2"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return table.assign(**columns)


def _fields(objects: pd.DataFrame) -> pd.DataFrame:
    return objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()


def _agree(objects, fovs, **arguments):
    """Run both; ``check_tables`` never raises and accepts exactly what the validator accepts."""
    arguments = {"feature_cols": ["area"], "observation_unit": "cell", **arguments}
    check = check_tables(objects, fovs, **arguments)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_input_tables(objects, fovs=fovs, **arguments)
        accepted = True
    except (ValueError, TypeError):
        accepted = False
    assert check.ok == accepted, check.summary()
    return accepted, check


def _titles(check, level="error"):
    return [p.title for p in check.problems if p.level == level]


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
    assert any("'condition' is empty in some rows" in t for t in _titles(check))


@pytest.mark.parametrize(
    "make",
    [
        lambda names: (n for n in names),
        lambda names: pd.Index(names),
        lambda names: np.array(names),
        lambda names: tuple(names),
    ],
    ids=["generator", "index", "array", "tuple"],
)
def test_a3_group_and_feature_columns_are_read_once_whatever_their_type(make):
    objects = _cells(condition=["a", "a", "b", "b"], position_id=["p1", "p2", "p1", "p2"])
    fovs = _fields(objects)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=make(["area"]),
        group_cols=make(["condition", "biological_repeat"]),
        observation_unit="cell",
    )
    assert data.group_cols == ("condition", "biological_repeat") and data.feature_cols == ("area",)
    assert all(type(c) is str for c in (*data.group_cols, *data.feature_cols))
    check = check_tables(
        objects,
        fovs,
        feature_cols=make(["area"]),
        group_cols=make(["condition", "biological_repeat"]),
        observation_unit="cell",
    )
    assert check.ok and "Analysis groups: condition, biological_repeat." in check.understood
    results = mc.run_strategies(
        objects,
        fovs=fovs,
        group_cols=make(["condition", "biological_repeat"]),
        strategies=[mc.FixedFOVStrategy(10)],
        feature_cols=["area"],
        objective=mc.ObjectiveConfig(),
        observation_unit="cell",
    )
    assert [dict(r.group) for r in results] == [
        {"condition": "a", "biological_repeat": "r1"},
        {"condition": "b", "biological_repeat": "r1"},
    ]


def test_a3_a_bare_string_or_a_non_collection_is_refused_by_both():
    objects = _cells()
    with pytest.raises(TypeError, match="not a string"):
        column_names("condition", "group_cols")
    for arguments in ({"group_cols": "condition"}, {"feature_cols": "area"}, {"group_cols": 3}):
        accepted, check = _agree(objects, _fields(objects), **arguments)
        assert not accepted and check.errors


@pytest.mark.parametrize(
    "arguments",
    [{"feature_cols": [7]}, {"feature_cols": [0, 1, 2]}, {"feature_cols": None}],
    ids=["integer name", "integer names, one missing", "none"],
)
def test_a4_check_tables_reports_integer_or_absent_feature_names(arguments):
    objects = _cells().join(pd.DataFrame({0: [0.1, 0.2, 0.3, 0.4], 1: [1.0, 2.0, 3.0, 4.0]}))
    accepted, check = _agree(objects, _fields(objects), **arguments)
    assert not accepted and check.errors
    accepted, _ = _agree(objects, _fields(objects), feature_cols=[0, 1])
    assert accepted


def test_a5_a_column_of_lists_is_accepted_by_both():
    objects = _cells(embedding=[[0.1, 0.2]] * 4)
    accepted, check = _agree(objects, _fields(objects))
    assert accepted and not any("copy" in t for t in _titles(check, "warning"))
    copied = pd.concat([objects, objects.iloc[[0]]])
    _, check = _agree(copied, _fields(objects))
    assert any("exact copy" in t for t in _titles(check, "warning"))


@pytest.mark.parametrize("column", ["area", "position_id"])
def test_a6_a_column_name_used_twice_is_refused_before_anything_else(column):
    base = _cells()
    objects = pd.concat([base, base[[column]]], axis=1)
    with pytest.raises(ValueError, match="more than one column named"):
        validate_input_tables(
            objects, fovs=_fields(base), feature_cols=["area"], observation_unit="cell"
        )
    accepted, check = _agree(objects, _fields(base))
    assert not accepted
    assert _titles(check) == ["The observation table has more than one column with the same name."]


def test_a7_labels_restarting_in_each_named_field_are_not_repeats():
    objects = pd.DataFrame(
        {
            "condition": ["a"] * 4,
            "biological_repeat": ["r1"] * 4,
            "site": ["s1", "s1", "s2", "s2"],
            "label_id": [1, 2, 1, 2],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fovs = objects[["condition", "biological_repeat", "site"]].drop_duplicates()
    accepted, check = _agree(objects, fovs, fov_col="site")
    assert accepted and not any("label_id" in t for t in _titles(check, "warning"))
    twice = objects.assign(label_id=[1, 1, 1, 2])
    _, check = _agree(twice, fovs, fov_col="site")
    assert any("label_id" in t for t in _titles(check, "warning"))


def test_a8_a_named_group_missing_from_the_fov_table_is_said_so():
    objects = pd.DataFrame(
        {"plate": ["P1", "P1", "P2", "P2"], "position_id": ["p1", "p2"] * 2, "area": [1.0] * 4}
    )
    fovs = pd.DataFrame({"position_id": ["p1", "p2", "p1", "p2"]})
    accepted, check = _agree(objects, fovs, group_cols=["plate"])
    assert not accepted
    assert _titles(check) == ["'plate' is in the observation table but not in the FOV table."]


def test_a9_planes_counted_as_fields_are_found_in_the_fov_table_and_warned_about():
    files = [f"pos{p}_z{z}.tif" for p in (1, 2) for z in range(3)]
    z = [0, 1, 2, 0, 1, 2]
    fovs = pd.DataFrame({"condition": "a", "biological_repeat": "r1", "image_file": files})
    cells = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r1",
            "image_file": np.repeat(files, 2),
            "area": 1.0,
        }
    )
    # z only in the FOV table, several cells per plane file: refused, as with z on the rows.
    accepted, check = _agree(cells, fovs.assign(z=z))
    assert not accepted and "in the FOV table" in check.errors[0].title
    # One row per plane file (a per-plane summary): kept, with a warning.
    summary = fovs.assign(z=z, area=np.arange(6.0))
    with pytest.warns(UserWarning, match="plane/channel/frame"):
        validate_input_tables(summary, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    accepted, check = _agree(summary, fovs)
    assert accepted and any("plane/channel/frame" in t for t in _titles(check, "warning"))
    # Named explicitly, or with a real position identifier, there is nothing to say.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_input_tables(
            summary, fovs=fovs, feature_cols=["area"], fov_col="image_file", observation_unit="cell"
        )


def test_a10_the_control_is_checked_without_a_fov_table_and_must_name_a_column():
    objects = _cells(condition=["a", "a", "b", "b"])
    check = check_tables(
        objects, None, feature_cols=["area"], observation_unit="cell", control={"condition": "M"}
    )
    assert "No field matches the control {'condition': 'M'}." in _titles(check)
    for control in ({}, ["condition"]):
        check = check_tables(
            objects,
            _fields(objects),
            feature_cols=["area"],
            observation_unit="cell",
            control=control,
        )
        assert "The control does not name a column and its value." in _titles(check)


def test_a11_missing_values_of_any_kind_read_empty():
    objects = _cells(condition=pd.array(["a", pd.NA, "a", "a"], dtype="string"))
    _, check = _agree(objects, _fields(objects).dropna())
    wheres = " ".join(p.where for p in check.errors)
    assert "condition=empty" in wheres and "<NA>" not in wheres


def test_a12_the_same_number_written_twice_is_not_called_spaces_or_case():
    objects = _cells(position_id=["1", "1", "2", "2"])
    fovs = _fields(objects).assign(position_id=["1.0", "2.0"])
    _, check = _agree(objects, fovs)
    where = next(p.where for p in check.errors if "not in the FOV table" in p.title)
    assert "writes the same number differently" in where and "spaces or case" not in where


def test_a13_a_rare_missing_share_is_not_printed_as_zero():
    area = np.arange(5000.0)
    area[0] = np.nan
    objects = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "position_id": [f"p{i % 10}" for i in range(5000)],
        }
    ).assign(area=area)
    _, check = _agree(objects, _fields(objects))
    assert "Feature 'area' is missing in 0.02% of observations." in _titles(check, "warning")


def test_a14_a_repeated_index_label_shows_each_rows_own_value():
    objects = _cells(area=[1.0, np.inf, 3.0, 4.0]).set_axis([0, 0, 1, 1])
    _, check = _agree(objects, _fields(objects))
    (problem,) = [p for p in check.errors if "infinite" in p.title]
    assert problem.where == "observation table, 1 row: row 0 (area=inf)."


def test_a15_the_effort_is_worded_in_the_singular_for_one():
    objects = _cells().iloc[[0]]
    (result,) = mc.run_strategies(
        objects,
        fovs=_fields(objects),
        strategies=[mc.FixedFOVStrategy(5)],
        feature_cols=["area"],
        objective=mc.ObjectiveConfig(),
        observation_unit="cell",
    )
    assert result.explain().endswith("after 1 FOV, 1 cell")


def test_a16_numpy_numbers_configure_targets_and_policies_like_python_ones():
    rare = mc.RareTarget(
        detection_probability=np.float64(0.95),
        min_fov_occurrences=np.int64(2),
        phenotype="label",
        object_prevalence=np.float32(0.5),
    )
    assert type(rare.min_fov_occurrences) is int and type(rare.object_prevalence) is float
    assert rare == mc.RareTarget(
        detection_probability=0.95, min_fov_occurrences=2, phenotype="label", object_prevalence=0.5
    )
    policy = mc.EffortPolicy("n_fovs", np.int64(10))
    assert type(policy.target) is int
    assert policy.config_id == mc.EffortPolicy("n_fovs", 10).config_id
    strategy = mc.FixedFOVStrategy(np.int32(4))
    assert type(strategy.target_fovs) is int
    assert strategy.config_id == mc.FixedFOVStrategy(4).config_id
    with pytest.raises(TypeError):
        mc.FixedFOVStrategy(np.float64(4.0))
    with pytest.raises(TypeError):
        mc.EffortPolicy("n_fovs", True)


def test_a17_config_ids_tell_tuples_from_lists_and_number_keys_from_text():
    assert config_hash("x", {1: "a"}) != config_hash("x", {"1": "a"})
    assert config_hash("x", {"k": (1, 2)}) != config_hash("x", {"k": [1, 2]})
    assert config_hash("x", {1: "a", "b": 2}) == config_hash("x", {"b": 2, 1: "a"})
    # Parameters JSON already keeps apart keep the id earlier versions gave them.
    parameters = {"target": 3, "names": ["a", "b"], "nested": {"z": None, "y": 1.5}}
    payload = json.dumps(
        {"identity": "x", "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert config_hash("x", parameters) == hashlib.sha256(payload.encode()).hexdigest()[:16]
    with pytest.raises(TypeError):
        config_hash("x", {"__tuple__": 1})


def test_a18_no_decision_to_combine_is_refused_not_a_stop():
    with pytest.raises(ValueError, match="at least one decision"):
        combine_decisions([])
    assert combine_decisions([mc.Decision.STOP]) is mc.Decision.STOP


def test_a19_the_observation_index_is_kept_when_groups_come_from_the_fov_table():
    objects = pd.DataFrame(
        {"position_id": ["p1", "p2"], "area": [1.0, 2.0]}, index=["cellA", "cellB"]
    )
    fovs = pd.DataFrame(
        {"condition": ["a", "b"], "biological_repeat": ["r", "r"], "position_id": ["p1", "p2"]}
    )
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert list(data.objects.index) == ["cellA", "cellB"]
    assert list(data.objects["condition"]) == ["a", "b"]


# Every case where the two rule sets disagreed (A-1 ... A-9, A-19, A-21): the tables and
# the arguments. The same answer is required from both, whether accepted or refused.
_BASE = _cells()
_AGREEMENT_CASES = {
    "clean": (_BASE, _fields(_BASE), {}),
    "dates as repeats": (
        _BASE.assign(biological_repeat=[datetime.date(2024, 1, 5)] * 4),
        _fields(_BASE.assign(biological_repeat=[datetime.date(2024, 1, 5)] * 4)),
        {},
    ),
    "blank category": (
        _BASE.assign(condition=pd.Categorical(["a", "a", " ", " "])),
        _fields(_BASE.assign(condition=pd.Categorical(["a", "a", " ", " "]))),
        {},
    ),
    "group_cols array": (_BASE, _fields(_BASE), {"group_cols": np.array(["condition"])}),
    "integer feature missing": (_BASE, _fields(_BASE), {"feature_cols": [7]}),
    "feature_cols None": (_BASE, _fields(_BASE), {"feature_cols": None}),
    "list column": (_BASE.assign(embedding=[[1, 2]] * 4), _fields(_BASE), {}),
    "duplicate column": (pd.concat([_BASE, _BASE[["area"]]], axis=1), _fields(_BASE), {}),
    "named field, labels restart": (
        _BASE.assign(label_id=[1, 2, 1, 2]),
        _fields(_BASE),
        {"fov_col": "position_id"},
    ),
    "named group missing from fovs": (
        _BASE.assign(plate=["P1", "P1", "P2", "P2"]),
        _fields(_BASE),
        {"group_cols": ["plate"]},
    ),
    "plural group column": (
        _BASE.rename(columns={"condition": "Conditions"}),
        _fields(_BASE).rename(columns={"condition": "Conditions"}),
        {},
    ),
    "camelCase named explicitly": (
        _BASE.rename(columns={"position_id": "PositionID"}),
        _fields(_BASE).rename(columns={"position_id": "PositionID"}),
        {"fov_col": "PositionID"},
    ),
    "index kept": (_BASE.set_axis(list("wxyz")), _fields(_BASE), {}),
}


@pytest.mark.parametrize("name", sorted(_AGREEMENT_CASES))
def test_a20_check_tables_accepts_exactly_what_the_validator_accepts(name):
    objects, fovs, arguments = _AGREEMENT_CASES[name]
    _agree(objects, fovs, **arguments)


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
    assert not accepted and any(column in t for t in _titles(check))


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
