import pandas as pd
import pytest

from morphocoverage import (
    Component,
    Direction,
    Estimator,
    Evidence,
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
    ObjectiveConfig,
    Quantity,
    RareTarget,
    SamplingObjective,
    SamplingStrategy,
    ThresholdPolicy,
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
    # The fixed-FOV rule needs no estimator, so there is no estimation time.
    assert results[0].compute_seconds == 0
    assert all(result.compute_seconds >= 0 for result in results)
    assert all(result.runtime_seconds == result.compute_seconds for result in results)


def test_strategy_result_preserves_objective_and_parameters():
    objects, fovs = _tables()
    objective = ObjectiveConfig(
        SamplingObjective.RARE_DISCOVERY,
        rare=RareTarget(
            object_prevalence=0.01,
            detection_probability=0.9,
            min_fov_occurrences=1,
            phenotype="area above 10",
        ),
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
    # A fixed FOV count is a population rule: it does not claim to answer rare discovery.
    assert result.decision == "unsupported"
    assert result.component_decisions == {"population": "continue", "rare": "unsupported"}


class _ExternalMean(Estimator):
    """An estimator defined outside the package, as a user or new family would write it."""

    name = "external_mean"

    def __init__(self, feature):
        self.feature = feature

    @property
    def parameters(self):
        return {"feature": self.feature}

    def estimate(self, data, *, objective):
        values = data.feature_values(self.feature)
        meta = {
            "direction": Direction.HIGHER_IS_BETTER,
            "units": "area units",
            "component": Component.POPULATION,
        }
        if not len(values):
            return Evidence({"mean": Quantity.insufficient("no objects", **meta)})
        return Evidence({"mean": Quantity.available(float(values.mean()), **meta)})


def _external_mean_strategy(feature):
    return SamplingStrategy([_ExternalMean(feature)], [ThresholdPolicy("mean", 2.0)])


def test_external_strategy_plugs_in_without_framework_changes():
    objects, fovs = _tables()
    fovs = fovs.assign(order=[1, 2, 3])
    strategy = _external_mean_strategy("area")
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
    assert single.evidence.value("mean") == 2.0
    assert single.decision == "stop"  # higher is better, and 2.0 reaches the threshold
    # Framework-owned counts and provenance are filled for any strategy.
    assert (single.n_fovs, single.n_objects) == (3, 3)
    assert single.feature_cols == ("area",)
    assert single.fov_key == ("fov_id",)
    assert single.configuration["estimators"][0]["parameters"] == {"feature": "area"}
    assert single.strategy == "external_mean+threshold:mean"
    assert [r.result.strategy for r in replay] == ["external_mean+threshold:mean", "fixed_fov"] * 3


def test_strategy_cannot_read_metadata_as_a_feature():
    objects, fovs = _tables()
    objects["stage_x"] = 1.0
    with pytest.raises(ValueError, match="explicitly selected"):
        run_strategies(
            objects,
            fovs=fovs,
            strategies=[_external_mean_strategy("stage_x")],
            feature_cols=["area"],
            fov_col="fov_id",
            objective=ObjectiveConfig(SamplingObjective.POPULATION),
        )


def test_fixed_object_strategy_is_undefined_for_generic_slice_rows():
    # A slice count is not an object count: no decision, but no crash either,
    # so a benchmark or live session with other strategies keeps running.
    objects, fovs = _tables()
    by_objects, by_observations = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedObjectStrategy(2), FixedObservationStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        observation_unit="slice",
    )
    assert by_objects.decision is None
    assert by_observations.decision == "stop"


@pytest.mark.parametrize("unit", ["cells", "Cell", "nuclei", " Objects "])
def test_fixed_object_strategy_accepts_spelling_variants_of_object_units(unit):
    objects, fovs = _tables()
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedObjectStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        observation_unit=unit,
    )
    assert result.decision == "stop"


def test_undefined_object_count_says_why():
    objects, fovs = _tables()
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedObjectStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        observation_unit="patch",
    )
    assert result.decision is None
    assert "(patch)" in result.diagnostics["undefined_reason"]


@pytest.mark.parametrize(("unit", "declared"), [(None, False), ("object", True), ("cells", True)])
def test_results_record_whether_the_observation_unit_was_declared(unit, declared):
    # Option (b): rows default to objects, but an assumed unit is never silent.
    objects, fovs = _tables()
    kwargs = {} if unit is None else {"observation_unit": unit}
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        fov_col="fov_id",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        **kwargs,
    )
    assert result.observation_unit_declared is declared
    assert result.observation_unit == ("object" if unit in (None, "object") else "cell")
