from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_benchmark import POPULATION
from _helpers_strategies import _report_both_orders
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Component,
    Direction,
    Estimator,
    Evidence,
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
    ObjectiveConfig,
    OnlineSession,
    Quantity,
    RareTarget,
    SamplingObjective,
    SamplingStrategy,
    ThresholdPolicy,
    evaluate_acquisition_prefixes,
    run_strategies,
)
from morphocoverage.policies import EffortPolicy
from morphocoverage.strategies.base import EstimatorState
from morphocoverage.workflows import LiveFolder, Setup, live_session, write_fov


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
    assert (single.n_fovs, single.n_observations) == (3, 3)
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


_SEEN: list = []


class _Spy(Estimator):
    """Records what the framework hands an estimator."""

    name = "spy"

    @property
    def parameters(self):
        return {}

    def estimate(self, data, *, objective):
        column = data.objects["image_file"]
        _SEEN.append(("estimate", data.objects.index.tolist(), list(column.cat.categories)))
        return Evidence({}, {})


class _SpyState(EstimatorState):
    def update(self, new):
        column = new.objects["image_file"]
        _SEEN.append(("update", new.objects.index.tolist(), list(column.cat.categories)))
        return Evidence({}, {})


class _IncrementalSpy(_Spy):
    name = "incremental_spy"

    def start(self, template, *, objective):
        return _SpyState()


@pytest.mark.parametrize("estimator", [_Spy(), _IncrementalSpy()], ids=["prefix", "update"])
def test_b3_live_estimator_inputs_expose_only_visible_rows(estimator):
    built = SCENARIOS["homogeneous"]()
    objects = built.objects.assign(image_file=pd.Categorical(built.objects.image_file))
    fovs = built.fovs.assign(image_file=pd.Categorical(built.fovs.image_file))
    strategy = SamplingStrategy([estimator], [], name="spy")
    kwargs = {"strategies": [strategy], "feature_cols": list(FEATURE_COLS), "objective": POPULATION}
    _SEEN.clear()
    evaluate_acquisition_prefixes(
        objects, fovs, acquisition_order_col=ORDER_COL, incremental=True, **kwargs
    )
    replayed = _SEEN[:3]
    _SEEN.clear()
    session = OnlineSession(**kwargs)
    for _, meta in fovs.sort_values(ORDER_COL).head(3).iterrows():
        session.add_fov(objects[objects.image_file == meta.image_file], meta.to_dict())
    assert _SEEN == replayed
    step, index, categories = _SEEN[-1]
    assert index[0] == 0 and len(categories) == (3 if step == "estimate" else 1)


def test_e8_one_fov_is_singular():
    answer = _report_both_orders("homogeneous", [FixedFOVStrategy(1)]).enough["answer"].iloc[0]
    assert "first met after 1 FOV;" in answer and "1 FOVs" not in answer


SETUP = Setup(readout="x", observation_unit="cell")


def _fields(folder, n: int = 6, condition: str = "MOCK") -> None:
    """``n`` fields of one well, deterministic."""
    for index in range(n):
        values = pd.DataFrame({"x": 100.0 + np.arange(5, dtype=float) + index})
        metadata = {
            "condition": condition,
            "biological_repeat": "R1",
            "technical_repeat": "A01",
            "image_file": f"A01_f{index:03d}.tif",
        }
        write_fov(folder, values, metadata)


def _fingerprint_rule(fovs: int) -> SamplingStrategy:
    """A rule that decides the fingerprint only, after ``fovs`` fields."""
    policy = EffortPolicy("n_fovs", fovs, Component.DISTRIBUTION)
    return SamplingStrategy([], [policy], name=f"fingerprint_{fovs}")


def _watch(folder, rules, objective: str = "population", **kwargs) -> int:
    session = live_session(SETUP, rules=rules, objective=ObjectiveConfig(objective))
    seen = LiveFolder(folder, session).watch(
        interval=0.01, timeout=2, until_stop=True, on_update=None, **kwargs
    )
    return len(seen)


def test_85_one_target_one_rule_stops_as_before(tmp_path):
    _fields(tmp_path)
    assert _watch(tmp_path, (FixedFOVStrategy(2),)) == 2
    assert _watch(tmp_path, (FixedFOVStrategy(2),), stop_on="rule") == 2


def test_85_two_targets_stop_when_both_stop(tmp_path):
    # Each rule alone is unsupported for the phenotype objective; together they decide it.
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(3))
    assert _watch(tmp_path, rules, "phenotype") == 3


def test_85_a_target_no_rule_decides_never_stops_the_watch(tmp_path):
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2),)
    assert _watch(tmp_path, rules, "phenotype") == 6  # every field, no stop
    assert _watch(tmp_path, rules, "phenotype", stop_on="rule") == 2


def test_85_one_target_still_continuing_keeps_the_watch_going(tmp_path):
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(5))
    assert _watch(tmp_path, rules, "phenotype") == 5
    three = (FixedFOVStrategy(2), FixedFOVStrategy(3), FixedFOVStrategy(4))
    assert _watch(tmp_path, three) == 4


def test_85_a_rule_on_a_target_not_required_does_not_hold_the_stop(tmp_path):
    # The live view's overall ignores a fingerprint rule under the population objective.
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(5))
    assert _watch(tmp_path, rules) == 2
    assert _watch(tmp_path, rules, stop_on="rule") == 5


def test_85_the_condition_label_does_not_change_the_stop(tmp_path):
    for condition in ("MOCK", "drug"):
        folder = tmp_path / condition
        _fields(folder, condition=condition)
        assert _watch(folder, (FixedFOVStrategy(2), _fingerprint_rule(3)), "phenotype") == 3


def test_85_stop_on_is_checked(tmp_path):
    _fields(tmp_path, 1)
    with pytest.raises(ValueError, match="stop_on"):
        _watch(tmp_path, (FixedFOVStrategy(2),), stop_on="every_rule")
