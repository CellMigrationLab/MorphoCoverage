"""Regression tests for architecture defects found in adversarial review of PR #1.

Each test encodes a scientific/data contract from AGENTS.md that the original
modular framework violated without raising. They constrain behaviour, not a particular
implementation.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from morphocoverage import (
    Component,
    DecisionPolicy,
    Estimator,
    Evidence,
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    Quantity,
    SamplingObjective,
    SamplingStrategy,
    evaluate_acquisition_prefixes,
    run_strategies,
    validate_input_tables,
)

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _run_prefixes(objects, fovs, strategy, **kwargs):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=POPULATION,
        **kwargs,
    )


# --- Grouping and FOV identity -------------------------------------------------


def test_condition_in_objects_but_not_fov_table_is_not_silently_pooled():
    # Common real case: the acquisition table only lists image_file while the
    # object table carries condition. Defaults used to pool control and drug.
    objects = pd.DataFrame(
        {
            "condition": ["control", "control", "drug", "drug"],
            "image_file": ["a.tif", "b.tif", "c.tif", "d.tif"],
            "area": [1.0, 2.0, 30.0, 40.0],
        }
    )
    fovs = pd.DataFrame({"image_file": ["a.tif", "b.tif", "c.tif", "d.tif"]})
    with pytest.raises(ValueError, match="'condition' is present in objects but absent"):
        run_strategies(
            objects,
            fovs=fovs,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["area"],
            objective=POPULATION,
        )


def test_canonical_column_missing_from_fov_table_raises_even_when_pooling():
    objects = pd.DataFrame(
        {"biological_repeat": ["R1", "R2"], "image_file": ["a.tif", "a.tif"], "area": [1.0, 2.0]}
    )
    fovs = pd.DataFrame({"image_file": ["a.tif"]})
    with pytest.raises(ValueError, match="'biological_repeat' is present in objects but absent"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"], group_cols=[])


@pytest.mark.parametrize("variant", ["Repeat", "REPEAT", "Condition"])
def test_non_canonical_spelling_of_group_column_is_rejected(variant):
    # fiberfeatures emits both 'Repeat' and 'biological_repeat'; 'Repeat' would otherwise
    # bypass canonical handling and pool biological repeats.
    objects = pd.DataFrame(
        {variant: ["R1", "R2"], "image_file": ["a.tif", "b.tif"], "area": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="non-canonical spelling"):
        validate_input_tables(objects, feature_cols=["area"])


def test_explicitly_named_non_canonical_group_column_is_accepted():
    objects = pd.DataFrame(
        {"Repeat": ["R1", "R2"], "image_file": ["a.tif", "b.tif"], "area": [1.0, 2.0]}
    )
    data = validate_input_tables(objects, feature_cols=["area"], group_cols=["Repeat"])
    assert data.group_cols == ("Repeat",)


def test_explicit_pooling_does_not_merge_fovs_with_reused_ids():
    # Microscope software and CellPatcher basenames repeat per well/repeat
    # folder. Pooling repeats must not turn R1/F1 and R2/F1 into one FOV.
    objects = pd.DataFrame(
        {
            "biological_repeat": ["R1", "R1", "R2", "R2"],
            "fov_id": ["F1", "F2", "F1", "F2"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    data = validate_input_tables(objects, feature_cols=["area"], group_cols=[])
    assert data.group_cols == ()
    assert data.fov_key == ("biological_repeat", "fov_id")
    assert len(data.fovs) == 4


def test_pooled_run_counts_every_fov_and_keeps_zero_object_fovs():
    objects = pd.DataFrame(
        {
            "biological_repeat": ["R1", "R2"],
            "image_file": ["img_001.tif", "img_001.tif"],
            "area": [1.0, 2.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "biological_repeat": ["R1", "R2", "R2"],
            "image_file": ["img_001.tif", "img_001.tif", "img_002.tif"],
        }
    )
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedFOVStrategy(3)],
        feature_cols=["area"],
        objective=POPULATION,
        group_cols=[],
    )
    assert (result.n_fovs, result.n_observations) == (3, 2)


def test_partial_grouping_keeps_other_canonical_columns_in_fov_identity():
    objects = pd.DataFrame(
        {
            "condition": ["control"] * 2,
            "biological_repeat": ["R1", "R2"],
            "image_file": ["img_001.tif", "img_001.tif"],
            "area": [1.0, 2.0],
        }
    )
    data = validate_input_tables(objects, feature_cols=["area"], group_cols=["condition"])
    assert data.fov_key == ("condition", "biological_repeat", "image_file")
    assert len(data.fovs) == 2


def test_image_file_defines_the_fov_even_for_multidimensional_stacks():
    # One image file is one FOV: objects measured on different z/t/c planes of
    # the same stack (distinct plane ids) must not be split into several FOVs.
    objects = pd.DataFrame(
        {
            "image_file": ["stack.tif"] * 3 + ["stack2.tif"],
            "fov_id": ["z0", "z1", "z2", "z0"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    data = validate_input_tables(objects, feature_cols=["area"])
    assert data.fov_col == "image_file"
    assert len(data.fovs) == 2


def test_fov_identifier_resolution_does_not_silently_fall_back():
    # The object table has image_file, the FOV table does not; resolution used
    # to fall back to fov_id without telling the caller.
    objects = pd.DataFrame(
        {"image_file": ["a.tif", "b.tif"], "fov_id": ["F1", "F2"], "area": [1.0, 2.0]}
    )
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"]})
    with pytest.raises(ValueError, match="'image_file' is present in objects but absent"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"], fov_col="fov_id")
    assert len(data.fovs) == 3


@pytest.mark.parametrize("argument", ["feature_cols", "group_cols"])
def test_bare_string_column_lists_are_rejected(argument):
    kwargs = {"feature_cols": ["area"], argument: "area" if argument == "feature_cols" else "rep"}
    with pytest.raises(TypeError, match="not a string"):
        validate_input_tables(pd.DataFrame({"fov_id": ["F1"], "area": [1.0]}), **kwargs)


# --- Sequential acquisition replay ---------------------------------------------


def test_sequential_runner_rejects_objects_from_unlisted_fovs():
    # run_strategies raised on this input; the sequential runner silently dropped it.
    objects = pd.DataFrame({"fov_id": ["F1", "F2", "GHOST"], "area": [1.0, 2.0, 999.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2"], "order": [1, 2]})
    with pytest.raises(ValueError, match="absent from the supplied fovs table"):
        _run_prefixes(objects, fovs, FixedFOVStrategy(1))


def test_sequential_runner_rejects_tied_acquisition_order():
    # With ties, which FOV is "first" depends on table row order.
    objects = pd.DataFrame({"fov_id": ["F1", "F2"], "area": [1.0, 2.0]})
    fovs = pd.DataFrame({"fov_id": ["F2", "F1"], "order": [1, 1]})
    with pytest.raises(ValueError, match="ties"):
        _run_prefixes(objects, fovs, FixedFOVStrategy(1))


def test_pooled_replay_requires_order_defined_across_pooled_fovs():
    # Per-repeat order (1, 2 in each) is not an acquisition history of the pool.
    objects = pd.DataFrame(
        {
            "biological_repeat": ["R1", "R1", "R2", "R2"],
            "fov_id": ["F1", "F2", "F1", "F2"],
            "area": [1.0] * 4,
        }
    )
    fovs = objects.loc[:, ["biological_repeat", "fov_id"]].assign(order=[1, 2, 1, 2])
    with pytest.raises(ValueError, match="ties"):
        _run_prefixes(objects, fovs, FixedFOVStrategy(1), group_cols=[])

    fovs["order"] = [1, 2, 3, 4]
    results = _run_prefixes(objects, fovs, FixedFOVStrategy(1), group_cols=[])
    assert results[1].acquired_fovs == (("R1", "F1"), ("R1", "F2"))
    assert results[2].result.n_observations == 3


def test_acquisition_order_cannot_be_a_feature():
    objects = pd.DataFrame({"fov_id": ["F1"], "area": [1.0]})
    fovs = pd.DataFrame({"fov_id": ["F1"], "order": [1]})
    with pytest.raises(ValueError, match="acquisition order"):
        evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[FixedFOVStrategy(1)],
            feature_cols=["area", "order"],
            acquisition_order_col="order",
            objective=POPULATION,
        )


# --- Strategy isolation and leakage --------------------------------------------


def _strategy(estimator):
    return SamplingStrategy([estimator])


class _CallCounter(Estimator):
    """Stateful estimator: it reports the number of calls this instance has seen."""

    name = "call_counter"

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, data, *, objective):
        self.calls += 1
        return Evidence(diagnostics={"calls": self.calls})


def test_strategy_state_does_not_flow_between_analysis_groups():
    # Any fitted state (scaler, PCA, reference sample) held by a shared instance
    # would carry control-group information into the drug group.
    objects = pd.DataFrame(
        {"condition": ["control", "drug"], "image_file": ["a.tif", "b.tif"], "area": [1.0, 99.0]}
    )
    estimator = _CallCounter()
    results = run_strategies(
        objects, strategies=[_strategy(estimator)], feature_cols=["area"], objective=POPULATION
    )
    assert [r.diagnostics["calls"] for r in results] == [1, 1]
    assert estimator.calls == 0


def test_strategy_state_does_not_flow_between_prefixes():
    objects = pd.DataFrame({"fov_id": ["F1", "F2", "F3"], "area": [1.0, 2.0, 3.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"], "order": [1, 2, 3]})
    results = _run_prefixes(objects, fovs, _strategy(_CallCounter()))
    assert [item.result.diagnostics["calls"] for item in results] == [1, 1, 1]


class _VisibleFovIds(Estimator):
    """Reports every FOV identifier and row label reachable from its input."""

    name = "visible_fov_ids"

    def estimate(self, data, *, objective):
        visible = set(data.fovs[data.fov_col]) | set(data.objects[data.fov_col])
        for column in (data.fovs[data.fov_col], data.objects[data.fov_col]):
            if isinstance(column.dtype, pd.CategoricalDtype):
                visible |= set(column.cat.categories)
        return Evidence(diagnostics={"ids": visible, "object_index": list(data.objects.index)})


def test_prefix_does_not_expose_future_fov_ids_via_categorical_levels():
    ids = pd.Categorical(["F1", "F2", "F3"])
    objects = pd.DataFrame({"fov_id": ids, "area": [1.0, 2.0, 3.0]})
    fovs = pd.DataFrame({"fov_id": ids, "order": [1, 2, 3]})
    results = _run_prefixes(objects, fovs, _strategy(_VisibleFovIds()))
    assert results[0].result.diagnostics["ids"] == {"F1"}


def test_group_does_not_expose_other_groups_fov_ids_via_categorical_levels():
    objects = pd.DataFrame(
        {
            "condition": ["control", "drug"],
            "fov_id": pd.Categorical(["C1", "D1"]),
            "area": [1.0, 2.0],
        }
    )
    results = run_strategies(
        objects,
        strategies=[_strategy(_VisibleFovIds())],
        feature_cols=["area"],
        objective=POPULATION,
    )
    assert [r.diagnostics["ids"] for r in results] == [{"C1"}, {"D1"}]


def test_prefix_row_labels_do_not_reveal_withheld_objects():
    # Original index gaps would reveal how many hidden objects lie between
    # visible ones when the object table is not sorted by acquisition order.
    objects = pd.DataFrame({"fov_id": ["F1", "F3", "F3", "F2"], "area": [1.0, 2.0, 3.0, 4.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"], "order": [1, 2, 3]})
    results = _run_prefixes(objects, fovs, _strategy(_VisibleFovIds()))
    assert results[1].result.diagnostics["object_index"] == [0, 1]


# --- Framework-owned result fields ---------------------------------------------


class _NaiveCounter(Estimator):
    """Counts FOVs from the object table, as a naive estimator might (drops empty FOVs)."""

    name = "naive_counter"

    def __init__(self, as_quantity: bool) -> None:
        self.as_quantity = as_quantity

    def estimate(self, data, *, objective):
        n_fovs = data.objects[data.fov_col].nunique()
        if self.as_quantity:
            return Evidence(
                {
                    "n_fovs": Quantity.available(
                        n_fovs, direction="higher_is_better", units="FOVs", component="population"
                    )
                }
            )
        return Evidence(diagnostics={"object_fovs": n_fovs})


def _zero_object_fov_tables():
    return (
        pd.DataFrame({"fov_id": ["F1"], "area": [1.0]}),
        pd.DataFrame({"fov_id": ["F1", "F2", "F3"]}),
    )


def test_framework_fills_fov_counts_including_zero_object_fovs():
    objects, fovs = _zero_object_fov_tables()
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[_strategy(_NaiveCounter(as_quantity=False))],
        feature_cols=["area"],
        objective=POPULATION,
    )
    assert (result.n_fovs, result.n_observations) == (3, 1)


def test_estimators_cannot_report_framework_owned_effort():
    objects, fovs = _zero_object_fov_tables()
    with pytest.raises(ValueError, match="framework-owned"):
        run_strategies(
            objects,
            fovs=fovs,
            strategies=[_strategy(_NaiveCounter(as_quantity=True))],
            feature_cols=["area"],
            objective=POPULATION,
        )


class _FreeFormPolicy(DecisionPolicy):
    name = "free_form"

    def decide(self, evidence, effort):
        return Component.POPULATION, "probably fine"


class _NotEvidence(Estimator):
    name = "not_evidence"

    def estimate(self, data, *, objective):
        return 0.5


@pytest.mark.parametrize(
    ("strategy", "error", "match"),
    [
        (SamplingStrategy(policies=[_FreeFormPolicy()]), ValueError, "decision"),
        (SamplingStrategy([_NotEvidence()]), TypeError, "must return Evidence"),
    ],
)
def test_decision_and_evidence_vocabulary_is_validated(strategy, error, match):
    # Benchmarks aggregate stop/continue across strategies; an unknown label
    # would silently count as neither.
    with pytest.raises(error, match=match):
        run_strategies(
            pd.DataFrame({"fov_id": ["F1"], "area": [1.0]}),
            strategies=[strategy],
            feature_cols=["area"],
            objective=POPULATION,
        )


def test_feature_matrix_exposes_only_explicit_features():
    data = validate_input_tables(
        pd.DataFrame({"fov_id": ["F1"], "area": [1.0], "stage_x": [5.0], "label_id": [7]}),
        feature_cols=["area"],
    )
    assert data.feature_matrix().tolist() == [[1.0]]


# --- Second audit (ChatGPT review of main at da3953c) ------------------------------


class _Mutator(Estimator):
    """Edits everything it is given in place, as careless preprocessing might."""

    name = "mutator"

    def estimate(self, data, *, objective):
        matrix = data.feature_matrix()
        matrix -= 100.0  # must not reach the tables
        data.objects[list(data.feature_cols)] = -1.0
        data.fovs.drop(data.fovs.index[-1], inplace=True)
        return Evidence()


class _Reader(Estimator):
    name = "reader"

    def estimate(self, data, *, objective):
        return Evidence(
            diagnostics={
                "sum": float(data.feature_values("area").sum()),
                "fov_rows": len(data.fovs),
            }
        )


def test_in_place_edits_by_one_strategy_do_not_reach_the_next():
    objects = pd.DataFrame({"fov_id": ["F1", "F1", "F2"], "area": [1.0, 2.0, 3.0]})
    mutated, read = run_strategies(
        objects,
        strategies=[_strategy(_Mutator()), _strategy(_Reader())],
        feature_cols=["area"],
        objective=POPULATION,
    )
    assert read.diagnostics["sum"] == 6.0 and read.diagnostics["fov_rows"] == 2
    # Framework-owned counts come from the original input, not the mutated copy.
    assert (mutated.n_fovs, read.n_fovs) == (2, 2)
    assert objects["area"].tolist() == [1.0, 2.0, 3.0]


def test_in_place_edits_do_not_reach_later_prefixes():
    objects = pd.DataFrame({"fov_id": ["F1", "F2", "F3"], "area": [1.0, 2.0, 3.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"], "order": [1, 2, 3]})
    results = _run_prefixes(objects, fovs, SamplingStrategy([_Mutator(), _Reader()]))
    results += _run_prefixes(objects, fovs, _strategy(_Reader()))
    # Also within one strategy: the second estimator sees the original tables.
    reads = [r.result.diagnostics["sum"] for r in results]
    assert reads == [1.0, 3.0, 6.0] * 2


def test_feature_arrays_are_owned_copies():
    data = validate_input_tables(
        pd.DataFrame({"fov_id": ["F1", "F2"], "area": [1.0, 2.0]}), feature_cols=["area"]
    )
    for array in (data.feature_matrix(), data.feature_values("area")):
        assert array.flags.writeable
        assert not np.shares_memory(array, data.objects["area"].to_numpy())


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown sampling objective"):
        ObjectiveConfig("something_else")
    assert ObjectiveConfig("population").objective is SamplingObjective.POPULATION


@pytest.mark.parametrize("value", [2.5, True, 0])
@pytest.mark.parametrize("strategy", [FixedFOVStrategy, FixedObjectStrategy])
def test_fixed_targets_must_be_positive_integers(strategy, value):
    with pytest.raises((TypeError, ValueError)):
        strategy(value)


def test_min_fovs_larger_than_a_group_raises_instead_of_dropping_it():
    objects = pd.DataFrame({"fov_id": ["F1", "F2"], "area": [1.0, 2.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2"], "order": [1, 2]})
    with pytest.raises(ValueError, match="min_fovs=5"):
        _run_prefixes(objects, fovs, FixedFOVStrategy(1), min_fovs=5)


def test_estimators_cannot_report_framework_effort_under_any_name():
    from morphocoverage.evidence import Evidence, Quantity

    for name in ("n_fovs", "n_observations", "n_objects"):
        with pytest.raises(ValueError):
            Evidence(
                {
                    name: Quantity.available(
                        5, direction="higher_is_better", units="count", component="population"
                    )
                }
            )


def test_j11_j19_sampling_unit_and_schema_import_without_cycles():
    code = (
        "import sys, morphocoverage.sampling_unit, morphocoverage.schema\n"
        "assert not [m for m in sys.modules if m.startswith('morphocoverage.retrospective')]\n"
        "import morphocoverage.schema as s\n"
        "from morphocoverage.data_check import validate_input_tables\n"
        "assert s.validate_input_tables is validate_input_tables\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
