import pandas as pd
import pytest

from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    SamplingObjective,
    SamplingStrategy,
    StrategyResult,
    evaluate_acquisition_prefixes,
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


class _ExternalMeanStrategy(SamplingStrategy):
    """A strategy defined outside the package, as a user or new family would write it."""

    name = "external_mean"

    def __init__(self, feature):
        self.feature = feature

    @property
    def parameters(self):
        return {"feature": self.feature}

    def evaluate(self, data, *, objective):
        values = data.feature_values(self.feature)
        return StrategyResult(
            self.name,
            objective,
            score=float(values.mean()) if len(values) else None,
            parameters=self.parameters,
        )


def test_external_strategy_plugs_in_without_framework_changes():
    objects, fovs = _tables()
    fovs = fovs.assign(order=[1, 2, 3])
    strategy = _ExternalMeanStrategy("area")
    [single] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[strategy],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )
    replay = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy, FixedFOVStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        acquisition_order_col="order",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )
    assert single.score == 2.0
    # Framework-owned counts and provenance are filled for any strategy.
    assert (single.n_fovs, single.n_objects) == (3, 3)
    assert single.feature_cols == ("area",)
    assert single.fov_key == ("fov_id",)
    assert single.parameters == {"feature": "area"}
    assert [r.result.strategy for r in replay] == ["external_mean", "fixed_fov"] * 3


def test_strategy_cannot_read_metadata_as_a_feature():
    objects, fovs = _tables()
    objects["stage_x"] = 1.0
    with pytest.raises(ValueError, match="explicitly selected"):
        run_strategies(
            objects,
            fovs=fovs,
            strategies=[_ExternalMeanStrategy("stage_x")],
            feature_cols=["area"],
            fov_col="fov_id",
            objective=ObjectiveConfig(SamplingObjective.POPULATION),
        )
