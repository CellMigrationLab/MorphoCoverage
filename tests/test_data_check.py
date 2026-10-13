"""check_tables: every problem at once, in plain words, and in agreement with the validator."""

from __future__ import annotations

import datetime
import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_tables import (
    _BASE,
    CONTINUOUS_FEATURES,
    _agree,
    _cells,
    _checks,
    _continuous_tables,
    _error_titles,
    _fields,
    _found,
    _four_cell_tables,
    _labelled_tables,
    _plate_tables,
    _six_cell_tables,
    _warning_titles,
)
from _helpers_workflows import _flat
from scenarios import FEATURE_COLS, SCENARIOS

import morphocoverage as mc
from morphocoverage import (
    DataCheck,
    FixedFOVStrategy,
    ObjectiveConfig,
    check_tables,
    planes_as_features,
    run_strategies,
    validate_input_tables,
)
from morphocoverage._style import CAUTION, OK, PROBLEM, Status
from morphocoverage.benchmark import permuted_fovs
from morphocoverage.metadata import canonical_spelling, sorted_by_key
from morphocoverage.reservoir import ReservoirSplit
from morphocoverage.retrospective import dataset_overview, inspect_fovs
from morphocoverage.schema import MissingFovTableWarning
from morphocoverage.strategies.rare_incidence import _positive_fovs


def _broken() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One dataset with one of each common mistake."""
    fovs = pd.DataFrame(
        {
            "condition": ["MOCK", "MOCK", "drug", "drug", "drug"],
            "Repeat": ["r1"] * 5,  # misspelt
            "technical_repeat": ["B02", "B02", None, "C02", "C03"],  # one empty
            "image_file": ["f1.tif", "f1.tif", "f2.tif", "f3.tif", "f4.tif"],  # f1 twice
        }
    )
    objects = pd.DataFrame(
        {
            "condition": ["MOCK", "MOCK", "drug", "drug", "drug", "drug"],
            "image_file": ["f1.tif", "f1.tif", "f2.tif", "f3.tif", "f3.tif", "f3.tif"],
            "area": [1.0, 2.0, np.inf, 3.0, 4.0, np.nan],  # one infinite, one missing
            "intensity": ["12,5", "3.1", "4", "n/a", "5", "6"],  # not numeric
        }
    )
    return objects, fovs


def _titles(check: DataCheck, level: str) -> str:
    return "\n".join(p.title for p in check.problems if p.level == level)


def test_clean_tables_pass_and_say_what_was_understood():
    built = SCENARIOS["condition_repeat_groups"]()
    check = check_tables(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), observation_unit="cell"
    )
    assert check.ok and not check.errors and not check.warnings
    text = check.summary()
    assert text.startswith("✅ MorphoCoverage data check — 0 errors, 0 warnings; the tables can be")
    assert "A field is identified by 'image_file' within condition, biological_repeat." in text
    assert "Analysis groups: condition, biological_repeat." in text
    check.raise_if_errors()


def test_every_problem_is_reported_at_once_with_where_and_how_to_fix():
    objects, fovs = _broken()
    check = check_tables(
        objects, fovs, feature_cols=["area", "intensity", "perimeter"], observation_unit="cell"
    )
    errors = _titles(check, "error")
    assert "Column 'Repeat' looks like the MorphoBricks column 'biological_repeat'" in errors
    assert "The group column 'technical_repeat' is empty in some rows" in errors
    assert "The FOV table lists a field more than once." in errors
    assert "1 feature column named in feature_cols is not in the observation table." in errors
    assert "Feature 'area' contains infinite values." in errors
    assert "Feature 'intensity' is not numeric." in errors
    # A value copied from the FOV table is reported once, where it came from.
    assert errors.count("'technical_repeat' is empty") == 1
    warnings = _titles(check, "warning")
    assert "1 field of the FOV table holds no observation." in warnings
    assert "Feature 'area' is missing in 17% of observations." in warnings
    # Where each problem is: in the full form (the short one gives the title and the fix).
    text = " ".join(check.summary(detail="full").split())
    assert (
        "Where: observation table, 2 rows: row 0 (intensity='12,5'); row 3 (intensity='n/a')"
        in (text)
    )
    assert "Where: observation table, 1 row: row 2 (area=inf)." in text
    short = " ".join(check.summary().split())
    assert "Feature 'intensity' is not numeric." in short and "Where:" not in short
    assert "np.float64" not in text and "nan" not in text.lower().replace("nan)", "")
    assert text.rstrip().endswith("is in docs/common/data_format.md.")
    with pytest.raises(ValueError, match="6 errors"):
        check.raise_if_errors()


def test_fields_missing_from_the_fov_table_are_named():
    fovs = pd.DataFrame(
        {
            "condition": ["MOCK", "drug"],
            "technical_repeat": ["B02", "C02"],
            "image_file": ["a", "b"],
        }
    )
    objects = pd.DataFrame(
        {"condition": ["MOCK", "drug"], "image_file": ["a", "z"], "x": [1.0, 2.0]}
    )
    check = check_tables(objects, fovs, feature_cols=["x"], observation_unit="cell")
    problem = next(p for p in check.errors if "not in the FOV table" in p.title)
    assert problem.where == "observation table, 1 row: row 1 (image_file='z')."


def test_warnings_for_a_missing_fov_table_unit_and_control():
    built = SCENARIOS["condition_repeat_groups"]()
    check = check_tables(built.objects, feature_cols=list(FEATURE_COLS))
    assert check.ok
    warnings = _titles(check, "warning")
    assert "No FOV table was given." in warnings and "not declared" in warnings
    control = built.fovs["condition"].iloc[0]
    lacking = built.fovs[
        ~((built.fovs["condition"] == control) & (built.fovs["biological_repeat"] == "R2"))
    ]
    kept = built.objects.merge(lacking[["condition", "biological_repeat", "image_file"]])
    check = check_tables(
        kept,
        lacking,
        feature_cols=list(FEATURE_COLS),
        observation_unit="cell",
        control={"condition": control},
    )
    assert "1 repeat has no control field." in _titles(check, "warning")
    wrong = check_tables(
        kept,
        lacking,
        feature_cols=list(FEATURE_COLS),
        observation_unit="cell",
        control={"condition": "nothing"},
    )
    assert "No field matches the control {'condition': 'nothing'}." in _titles(wrong, "error")


def test_empty_optional_columns_are_explained_not_refused():
    built = SCENARIOS["condition_repeat_groups"]()
    objects = built.objects.assign(position_id="", technical_repeat=None)
    fovs = built.fovs.assign(position_id="", technical_repeat=None)
    check = check_tables(objects, fovs, feature_cols=list(FEATURE_COLS), observation_unit="cell")
    assert check.ok
    assert "'position_id' is empty in every row, so it is read as absent." in check.understood
    assert "'technical_repeat' is empty in every row, so it is read as absent." in check.understood


def test_input_that_is_not_a_table_is_explained():
    check = check_tables([1, 2], None, feature_cols="x")
    assert not check.ok and "The observation table is not a table." in check.summary()
    built = SCENARIOS["homogeneous"]()
    check = check_tables(built.objects, built.fovs, feature_cols=FEATURE_COLS[0])
    assert "feature_cols is a single string" in _titles(check, "error")


def test_the_validator_raises_with_the_full_report_after_its_own_first_line():
    objects, fovs = _broken()
    with pytest.raises(ValueError) as caught:
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    message = str(caught.value)
    first, _, rest = message.partition("\n\n")
    assert "non-canonical spelling of MorphoBricks column 'biological_repeat'" in first
    assert rest.startswith("❌ MorphoCoverage data check — ")
    assert "The FOV table lists a field more than once." in rest
    assert "Where: FOV table" in rest  # the full report, with where each problem is


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_check_tables_and_the_validator_agree_on_every_scenario(name):
    built = SCENARIOS[name]()
    arguments = {"feature_cols": list(FEATURE_COLS), "observation_unit": "cell"}
    assert check_tables(built.objects, built.fovs, **arguments).ok
    validate_input_tables(built.objects, fovs=built.fovs, **arguments)
    # Each broken variant is refused by both.
    variants = {
        "duplicate field": (built.objects, pd.concat([built.fovs, built.fovs.iloc[[0]]])),
        "unlisted field": (
            built.objects,
            built.fovs[built.fovs["image_file"] != built.objects["image_file"].iloc[0]],
        ),
        "missing group value": (built.objects.assign(condition=None), built.fovs),
        "infinite feature": (built.objects.assign(**{FEATURE_COLS[0]: np.inf}), built.fovs),
    }
    for label, (objects, fovs) in variants.items():
        assert not check_tables(objects, fovs, **arguments).ok, label
        with pytest.raises((ValueError, TypeError)):
            validate_input_tables(objects, fovs=fovs, **arguments)


def test_issue_157_labels_are_matched_loosely_by_their_exact_value():
    from morphocoverage._data_rules import _loose

    assert _loose("1") == _loose("01") == _loose("1.0") == _loose(1) == _loose(" 1 ")
    assert _loose("B02") == _loose("b2")
    assert _loose("9007199254740992") != _loose("9007199254740993")
    assert _loose("1e999") != _loose("2e999")
    fovs = pd.DataFrame(
        {
            "condition": ["a", "a"],
            "biological_repeat": ["r", "r"],
            "technical_repeat": ["9007199254740992", "9007199254740993"],
            "image_file": ["f1.tif", "f2.tif"],
        }
    )
    check = check_tables(fovs.assign(x=[1.0, 2.0]), fovs, feature_cols=["x"])
    assert not any("differ only in spaces" in p.title for p in check.warnings)


def test_the_estimator_and_the_inspection_refuse_the_same_labels():
    objects, fovs = _labelled_tables(lambda n: [2] + [0] * (n - 1))
    data = validate_input_tables(
        objects=objects, fovs=fovs, feature_cols=["x"], fov_col="position_id"
    )
    with pytest.raises(ValueError, match="must hold booleans or 0/1"):
        _positive_fovs(data, "state")
    with pytest.raises(ValueError, match="must hold booleans or 0/1"):
        inspect_fovs(objects, fovs, feature_cols=["x"], fov_col="position_id", label_col="state")


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
    assert accepted and not any("copy" in t for t in _error_titles(check, "warning"))
    copied = pd.concat([objects, objects.iloc[[0]]])
    _, check = _agree(copied, _fields(objects))
    assert any("exact copy" in t for t in _error_titles(check, "warning"))


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
    assert _error_titles(check) == [
        "The observation table has more than one column with the same name."
    ]


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
    assert accepted and not any("label_id" in t for t in _error_titles(check, "warning"))
    twice = objects.assign(label_id=[1, 1, 1, 2])
    _, check = _agree(twice, fovs, fov_col="site")
    assert any("label_id" in t for t in _error_titles(check, "warning"))


def test_a8_a_named_group_missing_from_the_fov_table_is_said_so():
    objects = pd.DataFrame(
        {"plate": ["P1", "P1", "P2", "P2"], "position_id": ["p1", "p2"] * 2, "area": [1.0] * 4}
    )
    fovs = pd.DataFrame({"position_id": ["p1", "p2", "p1", "p2"]})
    accepted, check = _agree(objects, fovs, group_cols=["plate"])
    assert not accepted
    assert _error_titles(check) == ["'plate' is in the observation table but not in the FOV table."]


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
    assert accepted and any("plane/channel/frame" in t for t in _error_titles(check, "warning"))
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
    assert "No field matches the control {'condition': 'M'}." in _error_titles(check)
    for control in ({}, ["condition"]):
        check = check_tables(
            objects,
            _fields(objects),
            feature_cols=["area"],
            observation_unit="cell",
            control=control,
        )
        assert "The control does not name a column and its value." in _error_titles(check)


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
    assert "Feature 'area' is missing in 0.02% of observations." in _error_titles(check, "warning")


def test_a14_a_repeated_index_label_shows_each_rows_own_value():
    objects = _cells(area=[1.0, np.inf, 3.0, 4.0]).set_axis([0, 0, 1, 1])
    _, check = _agree(objects, _fields(objects))
    (problem,) = [p for p in check.errors if "infinite" in p.title]
    assert problem.where == "observation table, 1 row: row 0 (area=inf)."


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


def test_a4_an_optional_column_empty_in_one_table_is_read_from_the_other():
    cells, fovs = _six_cell_tables()
    cells["technical_repeat"] = np.nan  # CellPatcher writes it and leaves it empty
    fovs["technical_repeat"] = ["w1", "w1", "w2", "w2"]  # the FOV table knows the wells
    accepted, check = _agree(cells, fovs)
    assert accepted
    assert any("so the FOV table's values are used" in u for u in check.understood)
    data = validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert "technical_repeat" in data.group_cols
    assert data.objects["technical_repeat"].tolist() == ["w1", "w1", "w1", "w2", "w2", "w2"]
    # position_id the same way, through the file both tables name.
    cells, fovs = _six_cell_tables()
    cells["position_id"] = ""
    fovs["position_id"] = ["p1", "p2", "p3", "p4"]
    accepted, _ = _agree(cells, fovs)
    data = validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert accepted and data.fov_col == "position_id"
    assert data.objects["position_id"].tolist() == ["p1", "p1", "p2", "p3", "p3", "p4"]


def test_a5_label_variants_within_a_table_warn():
    cells = pd.DataFrame(
        {"image_file": [f"f{i}" for i in range(6) for _ in range(2)], "area": np.arange(12.0)}
    )
    fovs = pd.DataFrame(
        {
            "condition": ["MOCK", "MOCK", "drug", "drug", "drug ", "Drug"],
            "biological_repeat": "r1",
            "image_file": [f"f{i}" for i in range(6)],
        }
    )
    accepted, check = _agree(cells, fovs)
    assert accepted and check.ok  # a warning, not an error
    [warning] = [p for p in check.warnings if "differ only in spaces" in p.title]
    assert "'drug' against 'drug ' against 'Drug'" in warning.where
    mixed = pd.concat(
        [
            pd.DataFrame(
                {"condition": "A", "biological_repeat": 1, "image_file": ["f1", "f2"], "area": 1.0}
            ),
            pd.DataFrame(
                {
                    "condition": "A",
                    "biological_repeat": "1",
                    "image_file": ["f3", "f4"],
                    "area": 2.0,
                }
            ),
        ],
        ignore_index=True,
    )
    check = check_tables(mixed, mixed.drop(columns="area"), feature_cols=["area"])
    assert [
        "'biological_repeat'" in t for t in _error_titles(check, "warning") if "differ only" in t
    ] == [True]


def test_a7_no_fov_table_warns_once_and_is_recorded():
    cells, _ = _six_cell_tables()
    with pytest.warns(MissingFovTableWarning, match="acquired FOVs without observations"):
        data = validate_input_tables(cells, feature_cols=["area"], observation_unit="cell")
    assert data.fovs_from_observations
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_strategies(
            cells,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["area"],
            objective=ObjectiveConfig(),
            observation_unit="cell",
        )
    missing = [w for w in caught if issubclass(w.category, MissingFovTableWarning)]
    assert len(missing) == 1 and missing[0].filename == __file__  # it names the user's line
    fovs = _six_cell_tables()[1]
    assert not validate_input_tables(cells, fovs=fovs, feature_cols=["area"]).fovs_from_observations
    # check_tables says it once, in its own words.
    check = check_tables(cells, None, feature_cols=["area"], observation_unit="cell")
    assert [t for t in _error_titles(check, "warning") if "FOV table" in t] == [
        "No FOV table was given."
    ]


def test_a11_a_set_of_columns_is_refused():
    cells, fovs = _six_cell_tables()
    cells["intensity"] = cells["area"] * 2
    accepted, check = _agree(cells, fovs, feature_cols={"area", "intensity"})
    assert not accepted and "feature_cols is not a list of column names." in _error_titles(check)
    with pytest.raises(TypeError, match="not a set"):
        validate_input_tables(cells, fovs=fovs, feature_cols={"area"})


def test_a11_a_mapping_of_columns_is_refused_not_read_as_its_keys():
    cells, fovs = _six_cell_tables()
    for arguments in ({"feature_cols": {"area": False}}, {"group_cols": {"condition": 1}}):
        arguments = {"feature_cols": ["area"], **arguments}
        accepted, check = _agree(cells, fovs, **arguments)
        assert not accepted and "is not a list of column names." in _error_titles(check)[0]
        with pytest.raises(TypeError, match="not a mapping"):
            validate_input_tables(cells, fovs=fovs, **arguments)


def test_a12_a_group_column_named_twice_is_refused():
    cells, fovs = _six_cell_tables()
    accepted, check = _agree(cells, fovs, group_cols=["condition", "condition"])
    assert not accepted and "A group column is named more than once." in _error_titles(check)


def test_a13_missing_identifiers_are_named_before_the_plane_rule():
    cells, fovs = _six_cell_tables()
    cells["position_id"] = [None, None, None, "p3", "p3", "p4"]
    fovs["position_id"] = [None, None, "p3", "p4"]
    with pytest.raises(ValueError) as error:
        validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert str(error.value).startswith("objects['position_id'] contains missing FOV identifiers")


def test_a14_a_stage_z_distinct_in_every_field_is_not_a_plane():
    cells, fovs = _six_cell_tables()
    autofocus = fovs.assign(z=[1502.3, 1503.1, 1501.8, 1502.9])  # one height per position
    assert _agree(cells, autofocus)[0]
    planes = fovs.assign(z=[0, 1, 0, 1])  # plane values recur across fields
    accepted, check = _agree(cells, planes)
    assert not accepted and "in the FOV table" in _error_titles(check)[0]


def test_a15_a_header_only_table_is_not_refused_as_text():
    cells = pd.DataFrame(
        columns=["condition", "biological_repeat", "image_file", "area"]
    )  # object dtypes
    fovs = pd.DataFrame({"condition": "A", "biological_repeat": "r1", "image_file": ["f1", "f2"]})
    accepted, check = _agree(cells, fovs)
    assert accepted and "The observation table has no rows." in _error_titles(check, "warning")
    # Numbers stored as objects are still refused, with the conversion to make.
    cells, fovs = _six_cell_tables()
    accepted, check = _agree(cells.assign(area=cells["area"].astype(object)), fovs)
    assert not accepted and "pd.to_numeric" in check.errors[0].fix


def test_a16_check_tables_reports_instead_of_raising():
    cells, fovs = _six_cell_tables()
    for arguments in ({"fov_col": ["image_file"]}, {"fov_col": 3}):
        accepted, check = _agree(cells, fovs, **arguments)
        assert not accepted and "fov_col is not a column name." in _error_titles(check)
    check = check_tables(cells, fovs, feature_cols=["area"], control={"condition": ["A"]})
    assert not check.ok and "several values" in check.errors[0].title
    listed_cells = cells.assign(image_file=[[f] for f in cells["image_file"]])
    listed_fovs = fovs.assign(image_file=[[f] for f in fovs["image_file"]])
    accepted, check = _agree(listed_cells, listed_fovs)
    assert not accepted and "holds values such as lists" in _error_titles(check)[0]


def test_a17_complex_features_are_refused():
    cells, fovs = _six_cell_tables()
    accepted, check = _agree(cells.assign(area=cells["area"] + 2j), fovs)
    assert not accepted and "Feature 'area' holds complex numbers." in _error_titles(check)


def test_a18_a_condition_only_in_the_fov_table_joins_within_its_repeat():
    fovs = pd.DataFrame(
        {
            "condition": ["A", "B", "B", "A"],
            "biological_repeat": ["r1", "r1", "r2", "r2"],
            "image_file": ["f1", "f2", "f1", "f2"],
        }
    )
    cells = pd.DataFrame(
        {
            "biological_repeat": ["r1", "r1", "r2", "r2"],
            "image_file": ["f1", "f2", "f1", "f2"],
            "area": [1.0, 2, 3, 4],
        }
    )
    accepted, _ = _agree(cells, fovs)
    data = validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert accepted and data.objects["condition"].tolist() == ["A", "B", "B", "A"]
    # Still refused when (repeat, field) does not tell the conditions apart either.
    ambiguous = pd.concat([fovs, fovs.assign(condition="C", image_file="f9")], ignore_index=True)
    ambiguous.loc[4, "image_file"] = "f1"
    accepted, _ = _agree(cells, ambiguous)
    assert not accepted


def test_a20_problem_is_exported_and_the_docstrings_name_public_rules():
    assert mc.Problem is mc.data_check.Problem and "Problem" in mc.__all__
    assert "schema._validate_input_tables" not in mc.validate_input_tables.__doc__
    assert "technical_repeat" in mc.run_strategies.__doc__ and "fovs" in mc.run_strategies.__doc__


def test_j8_j16_messages_share_the_wording_and_modules_list_their_names():
    check = check_tables(
        *_six_cell_tables(), feature_cols=["area", "nope", "none"], observation_unit="cell"
    )
    assert (
        "2 feature columns named in feature_cols are not in the observation table."
        in _error_titles(check)
    )
    for name in (
        "metadata",
        "schema",
        "evidence",
        "policies",
        "results",
        "evaluator",
        "objectives",
        "provenance",
    ):
        module = importlib.import_module(f"morphocoverage.{name}")
        assert all(hasattr(module, n) and not n.startswith("_") for n in module.__all__), name


def test_p7_9_an_appended_table_with_a_fresh_index_and_object_numbers_is_reported():
    objects, fovs = _continuous_tables(n_fields=12, size=20)
    objects["ObjectNumber"] = objects.groupby("image_file").cumcount() + 1
    twice = pd.concat([objects, objects], ignore_index=True).reset_index()
    twice = twice.rename(columns={"index": "Unnamed: 0"})
    report = check_tables(twice, fovs, feature_cols=CONTINUOUS_FEATURES, observation_unit="cell")
    assert any("repeat another observation of the same field" in p.title for p in report.warnings)
    result = _checks(twice.sort_values(["image_file", "Unnamed: 0"]), fovs)
    assert set(_found(result)) == {"repeated within a field"}
    detail = result.findings["detail"][0]
    assert "holds its observations 1 to 20 again from observation 21" in detail


def test_p7_10_an_exact_copy_does_not_hide_the_label_warning():
    objects, fovs = _continuous_tables(n_fields=6, size=10)
    objects["label_id"] = 1 + np.arange(len(objects)) % 10
    # Two fields of well W0 exported under one file name: their labels collide.
    objects.loc[objects["image_file"].eq("f00004.tif"), "image_file"] = "f00000.tif"
    fovs = fovs[fovs["image_file"] != "f00004.tif"]
    doubled = pd.concat([objects, objects.iloc[[30]]], ignore_index=True)
    titles = [
        p.title for p in check_tables(doubled, fovs, feature_cols=CONTINUOUS_FEATURES).warnings
    ]
    assert any("exact copy" in t for t in titles)
    assert any("label_ids appear twice in the same field" in t for t in titles)


def test_p7_15_frame_columns_and_failed_segmentations_are_worded_honestly():
    objects, fovs = _continuous_tables(n_fields=6, size=10)
    objects["label_id"] = 1 + np.arange(len(objects)) % 10
    frames = pd.concat(
        [objects.assign(cycle=0), objects.assign(cycle=1, area=objects["area"] * 1.01)],
        ignore_index=True,
    )
    warnings = check_tables(
        frames, fovs, feature_cols=CONTINUOUS_FEATURES, observation_unit="cell"
    ).warnings
    [warning] = [p for p in warnings if "label_id" in p.title]
    assert "a plane or frame column not called z, c or t" in warning.fix
    failed = objects.drop(columns="label_id").iloc[:4].copy()
    failed[CONTINUOUS_FEATURES] = np.nan  # failed segmentations: no value, no label
    rows = pd.concat([objects.drop(columns="label_id"), failed, failed], ignore_index=True)
    titles = [p.title for p in check_tables(rows, fovs, feature_cols=CONTINUOUS_FEATURES).warnings]
    assert not any("exact copy" in t for t in titles)


def test_p5_4_a_merge_with_one_row_per_channel_is_reported():
    objects, fovs = _continuous_tables(n_fields=12, size=15)
    objects["ObjectNumber"] = objects.groupby("image_file").cumcount() + 1
    merged = pd.concat([objects.assign(Channel="DAPI"), objects.assign(Channel="GFP")])
    merged = merged.sort_values(["image_file", "Channel"], kind="stable")
    # A column named like the channel axis is refused as a misspelt 'c' (P5-2) ...
    errors = [p.title for p in check_tables(merged, fovs, feature_cols=CONTINUOUS_FEATURES).errors]
    assert any("'Channel' looks like the within-field coordinate 'c'" in t for t in errors)
    # ... and under any other name the repeated rows are still warned about.
    merged = merged.rename(columns={"Channel": "stain"})
    titles = [
        p.title for p in check_tables(merged, fovs, feature_cols=CONTINUOUS_FEATURES).warnings
    ]
    assert any("repeat another observation of the same field in every feature" in t for t in titles)
    result = _checks(merged, fovs)
    assert set(_found(result)) == {"repeated within a field"} and len(result.findings) == 12
    assert "nothing found" not in result.summary()


def test_p5_1_rows_of_several_planes_in_one_field_are_refused():
    cells, fovs = _plate_tables()
    stack = pd.concat([cells.assign(z=z) for z in range(5)], ignore_index=True)
    accepted, check = _agree(stack, fovs)
    assert not accepted
    message = _warning_titles(check, "error")[0]
    assert "several 'z' values" in message and "planes_as_features" in message
    assert "observation_unit='slice'" in message
    # Undeclared (assumed objects) is refused too; a plane-level unit is what the rows are.
    assert not _agree(stack, fovs, observation_unit=None)[0]
    data = validate_input_tables(stack, fovs=fovs, feature_cols=["area"], observation_unit="slice")
    assert len(data.objects) == 5 * len(cells)
    # CellPatcher's per-slice export: z0/z1 per plane, one image_file.
    patches = pd.concat([cells.assign(z0=z, z1=z + 1) for z in range(3)], ignore_index=True)
    accepted, check = _agree(patches, fovs, observation_unit="patch")
    assert not accepted and "several 'z0' values" in _warning_titles(check, "error")[0]
    # One plane per field is fine, and so is the pivot's one row per object.
    assert _agree(cells.assign(z=0), fovs)[0]
    wide, names = planes_as_features(stack, fovs, features=["area"])
    assert _agree(wide, fovs, feature_cols=names)[0]


def test_p5_2_plane_synonyms_and_plane_tokens_in_file_names():
    for name, meant in (
        ("Metadata_Z", "z"),
        ("Metadata_T", "t"),
        ("plane", "z"),
        ("Slice", "z"),
        ("frame", "t"),
        ("Timepoint", "t"),
        ("channel", "c"),
        ("Metadata_Channel", "c"),
    ):
        assert canonical_spelling(name) == meant, name
    for name in ("Metadata_Well", "Metadata_Plate", "time", "position", "planes_z"):
        assert canonical_spelling(name) is None, name
    cells, fovs = _plate_tables()
    per_plane = pd.concat([cells.assign(Metadata_Z=z) for z in range(3)], ignore_index=True)
    per_plane["image_file"] = (
        per_plane["image_file"].str.replace(".tif", "") + "_z" + per_plane["Metadata_Z"].astype(str)
    ) + ".tif"
    files = per_plane[
        ["condition", "biological_repeat", "technical_repeat", "image_file"]
    ].drop_duplicates()
    accepted, check = _agree(per_plane, files)
    assert not accepted and "'z'" in _warning_titles(check, "error")[0]
    # The plane only in the file name: 3 files per position counted as fields, with a warning.
    only_names = per_plane.drop(columns="Metadata_Z")
    accepted, check = _agree(only_names, files)
    assert accepted and any("plane, channel or frame token" in t for t in _warning_titles(check))
    with pytest.warns(UserWarning, match="plane, channel or frame token"):
        validate_input_tables(only_names, fovs=files, feature_cols=["area"])
    # Well C02 in a file name is not a channel token.
    assert not any(
        "token" in t for t in _warning_titles(check_tables(cells, fovs, feature_cols=["area"]))
    )


def test_p5_8_several_stage_positions_in_one_field_are_refused():
    cells, fovs = _plate_tables()
    for table in (cells, fovs):
        table["series"] = table["image_file"]
        table["image_file"] = table["biological_repeat"] + "_" + table["technical_repeat"] + ".nd2"
    fovs = fovs.drop_duplicates(
        ["condition", "biological_repeat", "technical_repeat", "image_file"]
    )
    fovs = fovs.drop(columns="series")
    accepted, check = _agree(cells, fovs)
    assert not accepted
    assert "several 'series' values" in _warning_titles(check, "error")[0]
    assert any("multi-position files" in t for t in _warning_titles(check))
    # With position_id the files hold their positions apart.
    cells["position_id"] = cells["image_file"] + "#" + cells["series"]
    listed = cells[
        ["condition", "biological_repeat", "technical_repeat", "position_id"]
    ].drop_duplicates()
    assert _agree(cells, listed)[0]


def test_p5_11_wells_without_any_observation_are_warned_about():
    cells, fovs = _plate_tables()
    plate_map = pd.concat(
        [fovs, fovs[fovs.technical_repeat == "C02"].assign(technical_repeat="C09")],
        ignore_index=True,
    )
    plate_map.loc[plate_map.technical_repeat == "C09", "image_file"] = "never_" + plate_map.loc[
        plate_map.technical_repeat == "C09", "image_file"
    ].astype(str)
    accepted, check = _agree(cells, plate_map)
    assert accepted
    assert (
        "2 technical repeats of the FOV table hold no observation in any field."
        in _warning_titles(check)
    )
    # One empty field in an imaged well is only an empty field.
    one_empty = cells[cells.image_file != fovs.image_file.iloc[0]]
    assert not any("technical repeat" in t for t in _warning_titles(_agree(one_empty, fovs)[1]))


def test_p5_12_zero_padding_and_one_well_in_two_conditions():
    cells, fovs = _plate_tables()
    for table in (cells, fovs):
        second = table["biological_repeat"] == "2026_01_02"
        table.loc[second, "technical_repeat"] = table.loc[second, "technical_repeat"].str.replace(
            "0", ""
        )
    accepted, check = _agree(cells, fovs)
    assert accepted and any("zero padding" in t for t in _warning_titles(check))
    cells, fovs = _plate_tables()
    for table in (cells, fovs):
        table.loc[table.technical_repeat == "C02", "technical_repeat"] = "B02"
    accepted, check = _agree(cells, fovs)
    assert accepted
    assert any("plate well is named in several conditions" in t for t in _warning_titles(check))
    # Dishes that restart in each condition are not plate wells.
    cells, fovs = _plate_tables(wells=("d1", "d2"))
    for table in (cells, fovs):
        table["technical_repeat"] = "d" + table["technical_repeat"].str[-1]
    assert set(fovs["technical_repeat"]) == {"d1", "d2"}
    assert not any("plate well" in t for t in _warning_titles(_agree(cells, fovs)[1]))


def test_p2_13_invalid_cellpatcher_rows_are_warned_about():
    cells, fovs = _plate_tables()
    cells["valid"] = True
    cells.loc[:4, "valid"] = False
    cells["reason"] = np.where(cells["valid"], "", "patch outside image")
    with pytest.warns(UserWarning, match="5 observations have valid=False"):
        validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="patch")
    accepted, check = _agree(cells, fovs, observation_unit="patch")
    assert accepted and any("valid=False" in t for t in _warning_titles(check))
    assert any("patch outside image" in t for t in _warning_titles(check))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_input_tables(
            cells.assign(valid=True), fovs=fovs, feature_cols=["area"], observation_unit="patch"
        )


def test_p9_3_category_order_does_not_change_orders_or_reservoirs():
    cells, fovs = _plate_tables(n_repeats=1, fields=8)
    one = fovs[fovs.technical_repeat == "B02"].reset_index(drop=True)
    key = ["condition", "biological_repeat", "technical_repeat", "image_file"]
    group = {"condition": "control", "biological_repeat": "2026_01_01", "technical_repeat": "B02"}
    reordered = one.assign(
        image_file=pd.Categorical(one["image_file"], categories=sorted(one["image_file"])[::-1])
    )
    plain = permuted_fovs(one, key, 0, group, 1)["image_file"].tolist()
    categorical = permuted_fovs(reordered, key, 0, group, 1)["image_file"].astype(str).tolist()
    assert plain == categorical
    assert sorted_by_key(reordered, key)["image_file"].astype(str).tolist() == sorted(
        one["image_file"]
    )
    data = validate_input_tables(
        cells[cells.technical_repeat == "B02"], fovs=one, feature_cols=["area"]
    )
    data_cat = validate_input_tables(
        cells[cells.technical_repeat == "B02"].assign(
            image_file=lambda t: pd.Categorical(
                t["image_file"], categories=reordered["image_file"].cat.categories
            )
        ),
        fovs=reordered,
        feature_cols=["area"],
    )
    split = ReservoirSplit(0.25, seed=3, block_col=None)
    held = split.split(group, data)[1].fovs["image_file"].tolist()
    held_cat = split.split(group, data_cat)[1].fovs["image_file"].astype(str).tolist()
    assert held == held_cat


def test_p9_9_index_levels_named_like_columns_are_accepted():
    cells, fovs = _plate_tables()
    indexed = cells.set_index(["condition", "biological_repeat"], drop=False)
    listed = fovs.set_index(["condition", "biological_repeat"], drop=False)
    data = validate_input_tables(indexed, fovs=listed, feature_cols=["area"])
    # The caller's labels are kept; only the clashing level names are dropped.
    assert data.objects.index.tolist() == indexed.index.tolist()
    summary = dataset_overview(indexed, listed, feature_cols=["area"]).summary()
    assert "2 conditions" in summary
    assert list(indexed.index.names) == [
        "condition",
        "biological_repeat",
    ]  # the input is not changed


def test_issue_160_one_well_in_two_conditions_is_seen_on_a_1536_well_plate():
    for control, drug in (("R", "S"), ("AA", "AB")):
        cells, fovs = _plate_tables()
        for table in (cells, fovs):
            well = table["technical_repeat"].replace({"C02": "B02"})
            rows = well.str[0].map({"B": control, "C": drug})
            table["technical_repeat"] = rows + well.str[1:]
        assert {"R02", "AA02"} & set(fovs["technical_repeat"])
        accepted, check = _agree(cells, fovs)
        assert accepted
        assert any("plate well is named in several conditions" in t for t in _warning_titles(check))


def _problems(result) -> list[tuple[str, str, str, str]]:
    return [(p.level, p.title, p.where, p.fix) for p in result.problems]


def test_check_tables_tables_that_are_not_tables_and_a_feature_named_twice():
    objects, fovs = _four_cell_tables()
    assert _problems(check_tables([1], fovs, feature_cols=["area"])) == [
        (
            "error",
            "The observation table is not a table.",
            "objects is a list.",
            "Pass a pandas DataFrame, e.g. pd.read_csv('cells.csv').",
        )
    ]
    assert _problems(check_tables(objects, [1], feature_cols=["area"])) == [
        (
            "error",
            "The FOV table is not a table.",
            "fovs is a list.",
            "Pass a pandas DataFrame with one row per acquired field.",
        )
    ]
    twice = check_tables(objects, fovs, feature_cols=["area", "area"], observation_unit="cell")
    assert _problems(twice) == [
        (
            "error",
            "A feature column is named twice.",
            "feature_cols: 'area'.",
            "Name each feature once.",
        )
    ]


def test_check_tables_warns_again_what_the_validator_warns_that_is_not_a_user_warning(
    monkeypatch,
):
    # The module whose name check_tables calls the validator by (data_check before the
    # refactor, _data_check_steps after it).
    try:
        holder = importlib.import_module("morphocoverage._data_check_steps")
    except ImportError:
        holder = importlib.import_module("morphocoverage.data_check")
    real = holder._validate_input_tables

    def validator(*args, **kwargs):
        warnings.warn("a future change", FutureWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(holder, "_validate_input_tables", validator)
    objects, fovs = _four_cell_tables()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_tables(objects, fovs, feature_cols=["area"], observation_unit="cell")
    assert result.ok and not result.problems
    assert [(w.category, str(w.message)) for w in caught] == [(FutureWarning, "a future change")]
    # The target frame: the caller of check_tables, its one_warning wrapper.
    assert caught[0].filename.endswith("_former_repeat.py")


def test_data_check_marker_and_forms():
    built = SCENARIOS["condition_repeat_groups"]()
    clean = check_tables(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), observation_unit="cell"
    )
    assert clean.status is Status.OK and clean.summary().startswith(f"{OK} MorphoCoverage")
    warned = check_tables(built.objects, built.fovs, feature_cols=list(FEATURE_COLS))
    assert warned.status is Status.CAUTION
    short, full = warned.summary(), warned.summary(detail="full")
    assert short.startswith(f"{CAUTION} MorphoCoverage data check")
    for problem in warned.problems:
        assert _flat(problem.title) in _flat(short) and _flat(problem.fix) in _flat(short)
        assert f"Where: {_flat(problem.where)}" in _flat(full)
    assert "Where:" not in short and "summary(detail='full')" in short
    broken = check_tables(built.objects, built.fovs, feature_cols=["nope"])
    assert broken.status is Status.PROBLEM and broken.summary().startswith(PROBLEM)
    with pytest.raises(ValueError, match="Where: missing"):
        broken.raise_if_errors()  # a refusal carries the full form
