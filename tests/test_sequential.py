import pandas as pd
import pytest

from morphocoverage import (
    Estimator,
    Evidence,
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    SamplingStrategy,
    evaluate_acquisition_prefixes,
)
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
            "repeat": ["R1", "R1", "R1"],
            "image_file": ["c1.tif", "c2.tif", "d1.tif"],
            "area": [1.0, 2.0, 3.0],
        }
    )
    fovs = pd.DataFrame(
        {
            "condition": ["control", "control", "drug"],
            "repeat": ["R1", "R1", "R1"],
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
    assert results[0].result.group == {"condition": "control", "repeat": "R1"}
    assert results[-1].result.group == {"condition": "drug", "repeat": "R1"}


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
            "repeat": ["R1"] * 6,
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
            "repeat": ["R1", "R1"],
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
    )

    assert [item.result.n_fovs for item in results] == [1, 2]
    assert [item.result.n_objects for item in results] == [3, 6]
    assert results[0].acquired_fovs == (("control", "R1", "P1"),)
    assert results[0].result.decision == "continue"
    assert results[1].result.decision == "stop"
