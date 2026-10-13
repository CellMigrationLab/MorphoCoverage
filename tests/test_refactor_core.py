"""Characterisation tests of the refactored core functions (refactor R1, 2026-10-12).

Each test pins, to the character, a branch that neither a golden job nor another test
executed: the exact exception or text of the unchanged code, so the refactor into named
steps is shown to keep it.
"""

from __future__ import annotations

import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from morphocoverage import check_tables
from morphocoverage._caveats import is_assumption
from morphocoverage._control import check_contrast_table
from morphocoverage._plots_text import title
from morphocoverage._style import rule_label
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    run_benchmark,
)
from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.planes import planes_as_features
from morphocoverage.plots import plots_of_data
from morphocoverage.plots_pairwise import PAIR_COLUMNS, pairwise_contrasts
from morphocoverage.schema import _validate_input_tables
from morphocoverage.strategies import FixedFOVStrategy
from morphocoverage.strategies._fov_bootstrap_checks import _check
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.sufficiency_short import _novelty_lines
from morphocoverage.sufficiency_text import _next_fov_answer

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    objects = pd.DataFrame(
        {
            "position_id": ["a", "a", "b", "b"],
            "condition": ["A", "A", "B", "B"],
            "biological_repeat": ["r1"] * 4,
            "label_id": [1, 2, 1, 2],
            "area": [1.0, 2.0, 3.0, 4.5],
            "z": [0, 0, 0, 0],
        }
    )
    fovs = pd.DataFrame(
        {"position_id": ["a", "b"], "condition": ["A", "B"], "biological_repeat": ["r1", "r1"]}
    )
    return objects, fovs


def _problems(result) -> list[tuple[str, str, str, str]]:
    return [(p.level, p.title, p.where, p.fix) for p in result.problems]


def test_check_contrast_table_refusals():
    with pytest.raises(TypeError) as error:
        check_contrast_table([1])
    assert str(error.value) == "the contrast table must be a pandas DataFrame (CONTRAST_COLUMNS)"
    with pytest.raises(ValueError) as error:
        check_contrast_table(pd.DataFrame({"kind": []}))
    assert str(error.value) == "the contrast table needs the columns ['condition', 'reference']"
    with pytest.raises(ValueError) as error:
        check_contrast_table(pd.DataFrame(columns=["kind", "condition", "reference"]))
    assert str(error.value) == "the contrast table holds no row"
    with pytest.raises(ValueError) as error:
        check_contrast_table(pd.DataFrame({"kind": ["x"], "condition": ["A"], "reference": ["B"]}))
    assert str(error.value) == (
        "contrast kind 'x' is not one of ['own_control', 'vs_master_raw', 'across_normalised']"
    )
    with pytest.raises(ValueError) as error:
        check_contrast_table(
            pd.DataFrame({"kind": ["own_control"], "condition": ["A"], "reference": ["A"]})
        )
    assert str(error.value) == "a contrast compares 'A' with itself"
    twice = pd.DataFrame(
        {
            "kind": ["own_control", "vs_master_raw"],
            "condition": ["A", "B"],
            "reference": ["C", "C"],
            "label": ["same", "same"],
        }
    )
    with pytest.raises(ValueError) as error:
        check_contrast_table(twice)
    assert str(error.value) == "the contrast table holds 'same' twice"


def test_check_tables_tables_that_are_not_tables_and_a_feature_named_twice():
    objects, fovs = _tables()
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
    objects, fovs = _tables()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_tables(objects, fovs, feature_cols=["area"], observation_unit="cell")
    assert result.ok and not result.problems
    assert [(w.category, str(w.message)) for w in caught] == [(FutureWarning, "a future change")]
    # The target frame: the caller of check_tables, its one_warning wrapper.
    assert caught[0].filename.endswith("_former_repeat.py")


def test_title_with_technical_repeats_and_no_cluster():
    layout = SimpleNamespace(
        repeat_col=None, technical_col="well", pooled_repeats=False, pooled_cols=(), max_repeats=8
    )
    assert title("area", "", layout, "cell", "mean", 0.95) == (
        "area\nevery cell a point in its condition's colour, its shape the well\nmean with its "
        "95% interval from resampling whole wells with their fields, or the fields where a "
        "condition has one well"
    )


def test_rule_label_keeps_the_name_of_an_unknown_variant():
    assert rule_label({"criterion": "scalar_precision_x", "settings": ""}, "cell") == (
        "scalar_precision_x"
    )
    assert rule_label({"criterion": "descriptor_spread_x", "settings": ""}, "cell") == (
        "descriptor_spread_x"
    )
    row = {"criterion": "support_novelty", "settings": "", "parameters": {"quantity": "other"}}
    assert rule_label(row, "cell") == "support_novelty"


def test_run_benchmark_refusals():
    objects, fovs = _tables()

    def load():
        return objects.assign(order=range(4)), fovs.assign(order=range(2))

    rule = [FixedFOVStrategy(2)]
    registry = BenchmarkRegistry(
        [BenchmarkDataset("d", "development", load, ("area",), "order", area_col="missing")]
    )
    with pytest.raises(ValueError) as error:
        run_benchmark(registry, "d", strategies=rule, metrics=[], objective=POPULATION, stop_on="x")
    assert str(error.value) == "stop_on must be 'decision' or 'rule', not 'x'"
    with pytest.raises(ValueError) as error:
        run_benchmark(registry, "d", strategies=rule, metrics=[], objective=POPULATION)
    assert str(error.value) == "area_col 'missing' is not a column of the FOV table"
    registry = BenchmarkRegistry([BenchmarkDataset("d", "development", load, ("area",), "order")])
    with pytest.raises(TypeError) as error:
        run_benchmark(
            registry, "d", strategies=rule, metrics=[], objective=POPULATION, reservoir="x"
        )
    assert str(error.value) == "reservoir must be a ReservoirSplit"


def test_attach_feature_tables_refusals():
    objects, fovs = _tables()
    with pytest.raises(TypeError) as error:
        attach_feature_tables([1], fovs, {"t": objects})
    assert str(error.value) == "objects must be a pandas DataFrame"
    with pytest.raises(TypeError) as error:
        attach_feature_tables(objects, fovs, {})
    assert str(error.value) == (
        "feature_tables must map each table's name to its DataFrame, or to (DataFrame, "
        "'explainable' or 'non-explainable')"
    )
    with pytest.raises(ValueError) as error:
        attach_feature_tables(objects, fovs, {"t": [1]})
    assert str(error.value) == (
        "the other feature tables cannot be used: the table 't' must be a pandas DataFrame"
    )


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
    objects, fovs = _tables()
    with pytest.raises(kind) as error:
        planes_as_features(objects, fovs, **arguments)
    assert str(error.value) == message


def test_planes_as_features_refuses_a_missing_plane():
    objects, fovs = _tables()
    with pytest.raises(ValueError) as error:
        planes_as_features(objects.assign(z=[0, np.nan, 0, 0]), fovs, features=["area"])
    assert str(error.value) == "objects['z'] is missing in some rows, so their plane is unknown"


def test_plots_of_data_refuses_no_feature():
    objects, fovs = _tables()
    with pytest.raises(ValueError) as error:
        plots_of_data(objects, fovs, features=[])
    assert str(error.value) == "features must name at least one column"


def test_pairwise_contrasts_without_conditions_and_with_a_condition_not_imaged():
    objects, fovs = _tables()
    empty = pairwise_contrasts(
        objects.drop(columns="condition"), fovs.drop(columns="condition"), feature="area"
    )
    assert empty.empty and list(empty.columns) == PAIR_COLUMNS
    table = pairwise_contrasts(objects, fovs, feature="area", conditions=["A", "C", "B"])
    assert table.iloc[:, :6].to_dict("list") == {
        "feature": ["area"],
        "first": ["A"],
        "second": ["B"],
        "d": [-2.4961508830135313],
        "observations": [2],
        "second_observations": [2],
    }


def test_the_validator_refuses_tables_that_are_not_tables():
    objects, _ = _tables()
    with pytest.raises(TypeError) as error:
        _validate_input_tables([1], feature_cols=["area"])
    assert str(error.value) == "objects must be a pandas DataFrame"
    with pytest.raises(TypeError) as error:
        _validate_input_tables(objects, feature_cols=["area"], fovs=[1])
    assert str(error.value) == "fovs must be a pandas DataFrame when provided"


def test_fov_bootstrap_refuses_a_count_that_is_not_an_integer():
    with pytest.raises(TypeError) as error:
        _check("centroid", 2.5, 0, 2, None, None, None)
    assert str(error.value) == "n_resamples must be an integer"


def test_sufficiency_report_refusals():
    objects, fovs = _tables()
    asked = {
        "feature_cols": ["area"],
        "strategies": [FixedFOVStrategy(2)],
        "objective": POPULATION,
        "orders": [AcquisitionOrder.permuted(0)],
    }
    with pytest.raises(TypeError) as error:
        sufficiency_report(objects, fovs, **asked, novelty_cols=[])
    assert str(error.value) == "novelty_cols must be a non-empty sequence of column names"
    with pytest.raises(ValueError) as error:
        sufficiency_report(objects, fovs, **asked, few_fovs=3)
    assert str(error.value) == "few_fovs needs a novelty definition"
    with pytest.raises(ValueError) as error:
        sufficiency_report(objects, fovs.assign(_report_row_order=0), **asked)
    assert str(error.value) == "the FOV table already has a column '_report_row_order'"


def test_short_novelty_lines_with_pooled_technical_repeats():
    frame = pd.DataFrame(
        {
            "chance_new": [0.1, 0.2],
            "reason": [None, None],
            "confidence": [0.95, 0.95],
            "chance_new_at_most": [0.3, 0.4],
            "fovs_with_new": [0, 0],
            "unseen_state_max_frequency": [0.2, 0.25],
            "unseen_state_max_technical_share": [0.5, np.nan],
            "min_novel_observations": [1, 1],
            "fovs_without_observations": [0, 0],
            "condition": ["A", "B"],
        }
    )
    report = SimpleNamespace(
        next_fov=frame,
        group_cols=("condition",),
        observation_unit="cell",
        observation_unit_declared=True,
    )
    lines, cautions = _novelty_lines(report, None)
    assert lines == [
        (
            "One more field: no field showed a state counted as new, in both groups; nominal 95% "
            "bound for the next field 30 to 40%."
        ),
        (
            "Not yet seen: a state no field has shown is in at most 20 to 25% of fields (95% "
            "confidence); rarer states can remain."
        ),
        (
            "Pooled technical repeats (1 of 2 groups): a state none of them showed is in at most "
            "50% of technical repeats (95% confidence)."
        ),
    ]
    assert cautions == [
        (
            "the next-field bound is approximate: the leave-one-field-out outcomes share their "
            "reference fields"
        ),
        (
            "the novelty bounds treat fields as independent, representative draws: neighbouring "
            "fields acquired in a raster over patchy states make them too low"
        ),
        (
            "fields of one technical repeat are not independent draws: where a group pools "
            "technical repeats, the field bound is optimistic"
        ),
    ]
    assert [is_assumption(c) for c in cautions] == [True, True, False]


def test_next_fov_answer_when_states_in_few_fields_are_not_estimable():
    row = {
        "chance_new": 0.1,
        "reason": None,
        "confidence": 0.95,
        "chance_new_at_most": 0.3,
        "fovs_with_new": 1,
        "fovs_imaged": 10,
        "fovs_without_observations": 0,
        "min_novel_observations": 1,
        "unseen_state_max_frequency": 0.26,
        "few_fovs": 3,
        "fovs_with_state_in_few": np.nan,
        "few_reason": "too few fields",
        "technical_repeats": None,
    }
    assert _next_fov_answer(row, None, None) == (
        "Chance: 10% (nominal 95% upper bound 30%, approximate: the leave-one-FOV-out outcomes "
        "share their reference FOVs, and it treats FOVs as independent, representative draws, "
        "so neighbouring fields acquired in a raster over patchy states make it too low); 1 of "
        "10 FOVs showed a state that no other FOV shows.\nNot yet seen: a state this definition "
        "detects but no FOV has shown is in at most 26% of FOVs (95% confidence, about 3 / 10), "
        "if FOVs are independent, representative draws. Rarer states can remain.\nSeen in few "
        "FOVs: not estimable (too few fields)."
    )
