"""Regressions for the third audit, group A (validation, check_tables, results, the plane pivot).

One test per finding, ``test_a<n>_...`` (and ``test_j<n>_...`` for the structure
findings in these modules). Where a finding is about what the tables accept,
the test also checks that ``check_tables`` and ``validate_input_tables`` agree
(``_agree``).
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import morphocoverage as mc
from morphocoverage import (
    Estimator,
    Evidence,
    FixedFOVStrategy,
    ObjectiveConfig,
    Quantity,
    SamplingStrategy,
    ThresholdPolicy,
    check_tables,
    planes_as_features,
    run_strategies,
    validate_input_tables,
)
from morphocoverage._seeds import group_identity
from morphocoverage._wording import percent
from morphocoverage.metadata import canonical_spelling
from morphocoverage.schema import MissingFovTableWarning
from morphocoverage.workflows import Setup


def _tables():
    """Six cells in four fields of two conditions, and the FOV table."""
    fovs = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "biological_repeat": ["r1"] * 4,
            "image_file": ["f1", "f2", "f3", "f4"],
        }
    )
    cells = pd.DataFrame(
        {
            "condition": ["A", "A", "A", "B", "B", "B"],
            "biological_repeat": ["r1"] * 6,
            "image_file": ["f1", "f1", "f2", "f3", "f3", "f4"],
            "area": [1.0, 2, 3, 4, 5, 6],
        }
    )
    return cells, fovs


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


def _planes(n_positions=2, n_planes=3, n_cells=4, **columns):
    """Cells measured at every z-plane of each position, one file per plane."""
    rows = [
        {
            "condition": "A",
            "biological_repeat": "r1",
            "position_id": f"p{p}",
            "image_file": f"p{p}_z{z}.tif",
            "z": z,
            "label_id": k,
            "area": float(10 * p + z + k),
        }
        for p in range(n_positions)
        for z in range(n_planes)
        for k in range(1, n_cells + 1)
    ]
    return pd.DataFrame(rows).assign(**columns)


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
    assert not accepted and "within-field coordinate 'z'" in _titles(check)[0]


def test_a2_dots_and_pandas_copies_are_near_misses():
    assert canonical_spelling("technical.repeat") == "technical_repeat"
    assert (
        canonical_spelling("Repeat.1") == "biological_repeat"
        and canonical_spelling("repeat.1") == "biological_repeat"
    )
    cells, fovs = _tables()
    cells["technical.repeat"] = ["w1", "w1", "w2", "w3", "w3", "w4"]
    fovs["technical.repeat"] = ["w1", "w2", "w3", "w4"]
    accepted, check = _agree(cells, fovs)
    assert not accepted and "'technical_repeat'" in _titles(check)[0]
    # Named explicitly, it is used as it is.
    accepted, _ = _agree(
        cells, fovs, group_cols=["condition", "biological_repeat", "technical.repeat"]
    )
    assert accepted


def test_a3_plane_coordinates_are_never_features_and_the_pivot_splits_them():
    cells = _planes()
    fovs = cells[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    for plane in ("z", "c", "t", "z0", "z1"):
        table = cells.assign(**{plane: cells["z"]})
        accepted, check = _agree(table, fovs, feature_cols=["area", plane])
        assert not accepted and "within-field coordinate" in _titles(check)[0]
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


def test_a4_an_optional_column_empty_in_one_table_is_read_from_the_other():
    cells, fovs = _tables()
    cells["technical_repeat"] = np.nan  # CellPatcher writes it and leaves it empty
    fovs["technical_repeat"] = ["w1", "w1", "w2", "w2"]  # the FOV table knows the wells
    accepted, check = _agree(cells, fovs)
    assert accepted
    assert any("so the FOV table's values are used" in u for u in check.understood)
    data = validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert "technical_repeat" in data.group_cols
    assert data.objects["technical_repeat"].tolist() == ["w1", "w1", "w1", "w2", "w2", "w2"]
    # position_id the same way, through the file both tables name.
    cells, fovs = _tables()
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
        "'biological_repeat'" in t for t in _titles(check, "warning") if "differ only" in t
    ] == [True]


def test_a6_percent_never_rounds_to_0_or_100_between():
    assert percent(0.996) == "99.6%" and percent(0.004) == "0.4%"
    cells = pd.DataFrame(
        {"image_file": [f"f{i % 10}" for i in range(1000)], "area": [np.nan] * 999 + [1.0]}
    )
    fovs = pd.DataFrame({"image_file": [f"f{i}" for i in range(10)]})
    check = check_tables(cells, fovs, feature_cols=["area"], observation_unit="cell")
    assert "Feature 'area' is missing in 99.9% of observations." in _titles(check, "warning")


def test_a7_no_fov_table_warns_once_and_is_recorded():
    cells, _ = _tables()
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
    fovs = _tables()[1]
    assert not validate_input_tables(cells, fovs=fovs, feature_cols=["area"]).fovs_from_observations
    # check_tables says it once, in its own words.
    check = check_tables(cells, None, feature_cols=["area"], observation_unit="cell")
    assert [t for t in _titles(check, "warning") if "FOV table" in t] == ["No FOV table was given."]


class _Constant(Estimator):
    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = value

    @property
    def parameters(self):
        return {"value": self.value}

    def estimate(self, data, *, objective):
        return Evidence(
            {
                "q": Quantity.available(
                    self.value,
                    direction="lower_is_better",
                    units="fraction",
                    component="population",
                )
            }
        )


def test_a8_explain_prints_the_digits_that_separate_value_and_threshold():
    cells, fovs = _tables()
    strategy = SamplingStrategy([_Constant(0.2504)], [ThresholdPolicy("q", 0.25)], name="c")
    [result] = run_strategies(
        cells,
        fovs=fovs,
        strategies=[strategy],
        feature_cols=["area"],
        objective=ObjectiveConfig(),
        group_cols=[],
        observation_unit="cell",
    )
    assert result.explain().startswith("continue: q = 0.2504 (stop at <= 0.25, population)")


def test_a9_counts_are_singular_for_one():
    cells, fovs = _tables()
    check = check_tables(
        cells.iloc[:1], fovs.iloc[:1], feature_cols=["area"], observation_unit="Cells"
    )
    assert check.understood[0] == "1 observation (cell); 1 row in the FOV table."
    strategy = SamplingStrategy([_Constant(0.1)], [ThresholdPolicy("q", 0.25)], name="c")
    [result] = run_strategies(
        cells.iloc[:1],
        fovs=fovs.iloc[:1],
        strategies=[strategy],
        feature_cols=["area"],
        objective=ObjectiveConfig(),
        group_cols=[],
        observation_unit="fov_summary",
    )
    # A stop then says what it rests on (audit 4, Q2).
    assert "after 1 FOV, 1 field summary; treats the fields as a random sample" in result.explain()


def test_a10_group_identity_is_the_same_for_numpy_and_python_values():
    assert group_identity({"biological_repeat": np.int64(1)}) == group_identity(
        {"biological_repeat": 1}
    )
    assert group_identity({"flag": np.True_}) == group_identity({"flag": True})


def test_a11_a_set_of_columns_is_refused():
    cells, fovs = _tables()
    cells["intensity"] = cells["area"] * 2
    accepted, check = _agree(cells, fovs, feature_cols={"area", "intensity"})
    assert not accepted and "feature_cols is not a list of column names." in _titles(check)
    with pytest.raises(TypeError, match="not a set"):
        validate_input_tables(cells, fovs=fovs, feature_cols={"area"})


def test_a12_a_group_column_named_twice_is_refused():
    cells, fovs = _tables()
    accepted, check = _agree(cells, fovs, group_cols=["condition", "condition"])
    assert not accepted and "A group column is named more than once." in _titles(check)


def test_a13_missing_identifiers_are_named_before_the_plane_rule():
    cells, fovs = _tables()
    cells["position_id"] = [None, None, None, "p3", "p3", "p4"]
    fovs["position_id"] = [None, None, "p3", "p4"]
    with pytest.raises(ValueError) as error:
        validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="cell")
    assert str(error.value).startswith("objects['position_id'] contains missing FOV identifiers")


def test_a14_a_stage_z_distinct_in_every_field_is_not_a_plane():
    cells, fovs = _tables()
    autofocus = fovs.assign(z=[1502.3, 1503.1, 1501.8, 1502.9])  # one height per position
    assert _agree(cells, autofocus)[0]
    planes = fovs.assign(z=[0, 1, 0, 1])  # plane values recur across fields
    accepted, check = _agree(cells, planes)
    assert not accepted and "in the FOV table" in _titles(check)[0]


def test_a15_a_header_only_table_is_not_refused_as_text():
    cells = pd.DataFrame(
        columns=["condition", "biological_repeat", "image_file", "area"]
    )  # object dtypes
    fovs = pd.DataFrame({"condition": "A", "biological_repeat": "r1", "image_file": ["f1", "f2"]})
    accepted, check = _agree(cells, fovs)
    assert accepted and "The observation table has no rows." in _titles(check, "warning")
    # Numbers stored as objects are still refused, with the conversion to make.
    cells, fovs = _tables()
    accepted, check = _agree(cells.assign(area=cells["area"].astype(object)), fovs)
    assert not accepted and "pd.to_numeric" in check.errors[0].fix


def test_a16_check_tables_reports_instead_of_raising():
    cells, fovs = _tables()
    for arguments in ({"fov_col": ["image_file"]}, {"fov_col": 3}):
        accepted, check = _agree(cells, fovs, **arguments)
        assert not accepted and "fov_col is not a column name." in _titles(check)
    check = check_tables(cells, fovs, feature_cols=["area"], control={"condition": ["A"]})
    assert not check.ok and "several values" in check.errors[0].title
    listed_cells = cells.assign(image_file=[[f] for f in cells["image_file"]])
    listed_fovs = fovs.assign(image_file=[[f] for f in fovs["image_file"]])
    accepted, check = _agree(listed_cells, listed_fovs)
    assert not accepted and "holds values such as lists" in _titles(check)[0]


def test_a17_complex_features_are_refused():
    cells, fovs = _tables()
    accepted, check = _agree(cells.assign(area=cells["area"] + 2j), fovs)
    assert not accepted and "Feature 'area' holds complex numbers." in _titles(check)


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


def test_a19_one_combination_of_decisions():
    from morphocoverage import policies

    assert not hasattr(policies, "_combine")
    assert "redecide" in policies.__doc__ and "strategy_config_id" in policies.__doc__


def test_a20_problem_is_exported_and_the_docstrings_name_public_rules():
    assert mc.Problem is mc.data_check.Problem and "Problem" in mc.__all__
    assert "schema._validate_input_tables" not in mc.validate_input_tables.__doc__
    assert "technical_repeat" in mc.run_strategies.__doc__ and "fovs" in mc.run_strategies.__doc__


def test_j8_j16_messages_share_the_wording_and_modules_list_their_names():
    check = check_tables(*_tables(), feature_cols=["area", "nope", "none"], observation_unit="cell")
    assert "2 feature columns named in feature_cols are not in the observation table." in _titles(
        check
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
