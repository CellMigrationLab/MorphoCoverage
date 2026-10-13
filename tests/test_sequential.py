import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_benchmark import MEAN_F, POPULATION
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Estimator,
    Evidence,
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    SamplingStrategy,
    evaluate_acquisition_prefixes,
)
from morphocoverage import sequential as sequential_module
from morphocoverage.retrospective import replay_acquisitions
from morphocoverage.strategies import EstimatorState


def test_sequential_runner_exposes_only_acquired_fovs():
    objects = pd.DataFrame(
        {
            "fov_id": ["F1", "F2", "F2", "F3"],
            "area": [1.0, 2.0, 2.5, 3.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "fov_id": ["F2", "F1", "F3"],
            "order": [2, 1, 3],
        }
    )
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        acquisition_order_col="order",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )

    assert [item.step for item in results] == [1, 2, 3]
    assert results[0].fov_key == ("fov_id",)
    assert results[0].acquired_fovs == (("F1",),)
    assert results[1].acquired_fovs == (("F1",), ("F2",))
    assert results[0].result.n_objects == 1
    assert results[1].result.n_objects == 3
    assert [item.result.decision for item in results] == [
        "continue",
        "stop",
        "stop",
    ]


def test_sequential_runner_preserves_zero_object_fov():
    objects = pd.DataFrame({"fov_id": ["F1"], "area": [1.0]})
    fovs = pd.DataFrame(
        {
            "fov_id": ["F1", "F2"],
            "order": [1, 2],
        }
    )
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        acquisition_order_col="order",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )

    assert results[-1].result.n_fovs == 2
    assert results[-1].result.n_objects == 1


def test_sequential_runner_uses_morphobricks_defaults():
    objects = pd.DataFrame(
        {
            "condition": ["control", "control", "drug"],
            "biological_repeat": ["R1", "R1", "R1"],
            "image_file": ["c1.tif", "c2.tif", "d1.tif"],
            "area": [1.0, 2.0, 3.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control", "drug"],
            "biological_repeat": ["R1", "R1", "R1"],
            "image_file": ["c1.tif", "c2.tif", "d1.tif"],
            "order": [1, 2, 1],
        }
    )

    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )

    assert len(results) == 3
    assert results[0].result.group == {"condition": "control", "biological_repeat": "R1"}
    assert results[-1].result.group == {"condition": "drug", "biological_repeat": "R1"}


# --- Incremental replay ---------------------------------------------------------


class _SeenRows(Estimator):
    """Incremental probe: records exactly which rows each update receives."""

    name = "seen_rows"

    def estimate(self, data, *, objective):
        return Evidence(diagnostics={"seen": sorted(data.fovs["fov_id"])})

    def start(self, template, *, objective):
        assert len(template.fovs) == 0 and len(template.objects) == 0
        return _SeenRowsState()


class _SeenRowsState(EstimatorState):
    def __init__(self):
        self.seen = []
        self.updates = []

    def update(self, new):
        self.updates.append(sorted(new.fovs["fov_id"]))
        self.seen.extend(new.fovs["fov_id"])
        new.fovs.drop(new.fovs.index, inplace=True)  # edits must not reach the framework
        return Evidence(diagnostics={"seen": sorted(self.seen), "updates": list(self.updates)})


def _incremental_tables():
    objects = pd.DataFrame(
        {
            "condition": ["a", "a", "a", "b", "b"],
            "fov_id": ["F1", "F2", "F3", "F1", "F2"],
            "area": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    fovs = objects[["condition", "fov_id"]].assign(order=[3, 1, 2, 2, 1])
    return objects, fovs


def _run_incremental(strategies, **kwargs):
    objects, fovs = _incremental_tables()
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        incremental=True,
        **kwargs,
    )


def test_incremental_updates_receive_only_the_new_fov_in_order():
    results = _run_incremental([SamplingStrategy([_SeenRows()])])
    by_group = {}
    for item in results:
        by_group.setdefault(item.result.group["condition"], []).append(item.result.diagnostics)
    # Each group starts from a fresh state and sees its own FOVs one at a time.
    assert by_group["a"][-1]["updates"] == [["F2"], ["F3"], ["F1"]]
    assert by_group["b"][-1]["updates"] == [["F2"], ["F1"]]
    assert [d["seen"] for d in by_group["a"]] == [["F2"], ["F2", "F3"], ["F1", "F2", "F3"]]
    assert [item.result.n_fovs for item in results] == [1, 2, 3, 1, 2]
    assert [item.result.n_objects for item in results] == [1, 2, 3, 1, 2]


def test_incremental_warm_up_feeds_every_fov_before_min_fovs():
    results = _run_incremental([SamplingStrategy([_SeenRows()])], min_fovs=2)
    first = results[0].result
    assert (results[0].step, first.diagnostics["updates"]) == (2, [["F2"], ["F3"]])
    assert first.timing.initial_fit_seconds > 0


def test_start_must_return_an_estimator_state():
    class _BadStart(_SeenRows):
        def start(self, template, *, objective):
            return {"not": "a state"}

    with pytest.raises(TypeError, match="EstimatorState"):
        _run_incremental([SamplingStrategy([_BadStart()])])


def test_slice_by_slice_stack_is_revealed_as_one_position():
    objects = pd.DataFrame(
        {
            "condition": ["control"] * 6,
            "biological_repeat": ["R1"] * 6,
            "position_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "image_file": [
                "P1_z000.tif",
                "P1_z001.tif",
                "P1_z002.tif",
                "P2_z000.tif",
                "P2_z001.tif",
                "P2_z002.tif",
            ],
            "z": [0, 1, 2, 0, 1, 2],
            "area": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control"],
            "biological_repeat": ["R1", "R1"],
            "position_id": ["P1", "P2"],
            "order": [1, 2],
        }
    )

    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        observation_unit="slice",  # one row per plane (audit 4, D18)
    )

    assert [item.result.n_fovs for item in results] == [1, 2]
    assert [item.result.n_objects for item in results] == [3, 6]
    assert results[0].acquired_fovs == (("control", "R1", "P1"),)
    assert results[0].result.decision == "continue"
    assert results[1].result.decision == "stop"


def test_an_order_column_named_twice_is_refused_by_the_validator():
    objects = pd.DataFrame({"fov_id": ["F1", "F2"], "area": [1.0, 2.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2"], "order": [1, 2]})
    fovs = pd.concat([fovs, fovs[["order"]]], axis=1)
    with pytest.raises(ValueError) as error:
        evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[FixedFOVStrategy(2)],
            feature_cols=["area"],
            fov_col="fov_id",
            acquisition_order_col="order",
            objective=ObjectiveConfig(SamplingObjective.POPULATION),
        )
    assert "truth value" not in str(error.value) and "order" in str(error.value).split("\n")[0]


SRC = Path(__file__).resolve().parents[1] / "src" / "morphocoverage"


def test_b3_text_acquisition_order_is_refused():
    fovs = pd.DataFrame(
        {"position_id": [f"p{i}" for i in range(1, 13)], "order": [str(i) for i in range(1, 13)]}
    )
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 2), "f": np.arange(24.0)})
    with pytest.raises(ValueError, match="not numbers or date-times"):
        evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["f"],
            acquisition_order_col="order",
            objective=POPULATION,
        )
    with pytest.raises(ValueError, match="not numbers or date-times"):
        replay_acquisitions(
            objects,
            fovs,
            feature_cols=["f"],
            strategies=[FixedFOVStrategy(3)],
            criteria=[MEAN_F],
            objective=POPULATION,
            reservoir_fraction=0.3,
            acquisition_order_col="order",
        )
    # Date-times are an order.
    timed = fovs.assign(order=pd.date_range("2026-01-01", periods=12, freq="min"))
    results = evaluate_acquisition_prefixes(
        objects,
        timed,
        strategies=[FixedFOVStrategy(3)],
        feature_cols=["f"],
        acquisition_order_col="order",
        objective=POPULATION,
    )
    assert [k[-1] for k in results[2].acquired_fovs] == ["p1", "p2", "p3"]


def test_a_categorical_or_incomplete_acquisition_order_is_refused_naming_the_replay():
    # Owner, 2026-10-12: the replay checks its order as a pilot does (check_order_col), its
    # messages naming the replay's argument.
    fovs = pd.DataFrame({"position_id": [f"p{i}" for i in range(1, 13)], "order": range(12)})
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 2), "f": np.arange(24.0)})
    categorical = fovs.assign(order=pd.Categorical(range(12), ordered=True))
    replay = "the replay of the acquisition order"
    named = "acquisition_order_col 'order'"  # the replay's own argument, not order_col
    with pytest.raises(TypeError, match=f"{named} is categorical.*give {replay} its order"):
        evaluate_acquisition_prefixes(
            objects,
            categorical,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["f"],
            acquisition_order_col="order",
            objective=POPULATION,
        )
    with pytest.raises(TypeError, match=f"{named} is categorical.*give {replay} its order"):
        replay_acquisitions(
            objects,
            categorical,
            feature_cols=["f"],
            strategies=[FixedFOVStrategy(3)],
            criteria=[MEAN_F],
            objective=POPULATION,
            reservoir_fraction=0.3,
            acquisition_order_col="order",
        )
    gap = fovs.assign(order=[np.nan, *range(1, 12)])
    with pytest.raises(ValueError, match=f"{named} has missing values; {replay} needs an"):
        evaluate_acquisition_prefixes(
            objects,
            gap,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["f"],
            acquisition_order_col="order",
            objective=POPULATION,
        )


def test_b17_shared_private_sequential_helpers_are_documented():
    shared = set()
    for path in (SRC / "benchmark.py", SRC / "retrospective" / "replay.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("sequential"):
                shared |= {alias.name for alias in node.names if alias.name.startswith("_")}
    assert shared
    doc = sequential_module.__doc__
    assert "Shared-private" in doc
    assert all(f"``{name}``" in doc for name in shared), shared


def test_b6_replay_reads_generator_feature_columns_once():
    built = SCENARIOS["homogeneous"]()
    kwargs = {
        "strategies": [FixedFOVStrategy(3)],
        "acquisition_order_col": ORDER_COL,
        "objective": POPULATION,
    }
    from_list = evaluate_acquisition_prefixes(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), **kwargs
    )
    from_generator = evaluate_acquisition_prefixes(
        built.objects, built.fovs, feature_cols=(c for c in FEATURE_COLS), **kwargs
    )
    assert [r.result.without_timing() for r in from_generator] == [
        r.result.without_timing() for r in from_list
    ]
    with pytest.raises(ValueError, match="acquisition order cannot also be a phenotype feature"):
        evaluate_acquisition_prefixes(
            built.objects, built.fovs, feature_cols=iter([*FEATURE_COLS, ORDER_COL]), **kwargs
        )


@pytest.mark.parametrize("incremental", [False, True])
def test_b7_min_fovs_must_be_an_integer(incremental):
    built = SCENARIOS["homogeneous"]()
    kwargs = {
        "strategies": [FixedFOVStrategy(3)],
        "feature_cols": list(FEATURE_COLS),
        "acquisition_order_col": ORDER_COL,
        "objective": POPULATION,
        "incremental": incremental,
    }
    for bad in (2.5, 2.0, 0, True):
        with pytest.raises(ValueError, match="min_fovs must be an integer of at least 1"):
            evaluate_acquisition_prefixes(built.objects, built.fovs, min_fovs=bad, **kwargs)
    steps = evaluate_acquisition_prefixes(built.objects, built.fovs, min_fovs=np.int64(2), **kwargs)
    assert min(r.step for r in steps) == 2
