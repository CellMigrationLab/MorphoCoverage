import pandas as pd

from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    SamplingObjective,
    run_strategies,
)


def _tables():
    objects = pd.DataFrame(
        {
            "fov_id": ["F1", "F1", "F2"],
            "area": [1.0, 2.0, 3.0],
        }
    )
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"]})
    return objects, fovs


def test_strategies_are_swappable_on_same_validated_tables():
    objects, fovs = _tables()
    results = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedFOVStrategy(3), FixedObjectStrategy(4)],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )
    assert [result.strategy for result in results] == ["fixed_fov", "fixed_objects"]
    assert [result.decision for result in results] == ["stop", "continue"]
    assert all(result.runtime_seconds is not None for result in results)
    assert all(result.runtime_seconds >= 0 for result in results)


def test_strategy_result_preserves_objective_and_parameters():
    objects, fovs = _tables()
    objective = ObjectiveConfig(
        SamplingObjective.RARE_DISCOVERY,
        rare_prevalence=0.01,
    )
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedFOVStrategy(5)],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=objective,
    )
    assert result.objective == objective
    assert result.parameters == {"target_fovs": 5}
