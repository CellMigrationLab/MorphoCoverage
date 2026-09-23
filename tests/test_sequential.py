import pandas as pd

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
)


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
