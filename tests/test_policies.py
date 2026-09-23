"""Tests for evidence, decision policies, their combination per objective, and config ids."""

import math

import pandas as pd
import pytest

from morphocoverage import (
    Availability,
    Component,
    Decision,
    DescriptorSpreadStrategy,
    Direction,
    Effort,
    EffortPolicy,
    Estimator,
    Evidence,
    FixedFOVStrategy,
    ObjectiveConfig,
    Quantity,
    RareTarget,
    SamplingObjective,
    SamplingStrategy,
    ScalarPrecisionEstimator,
    ScalarPrecisionStrategy,
    ThresholdPolicy,
    evaluate_acquisition_prefixes,
    redecide,
    run_strategies,
)
from morphocoverage.policies import decide

_RARE = RareTarget(
    fov_incidence=0.1, detection_probability=0.9, min_fov_occurrences=2, phenotype="label"
)
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
RARE = ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=_RARE)
BALANCED = ObjectiveConfig(SamplingObjective.BALANCED, rare=_RARE)
EFFORT = Effort(n_fovs=10, n_objects=100)


def _quantity(value, component=Component.POPULATION, direction=Direction.LOWER_IS_BETTER):
    return Quantity.available(value, direction=direction, units="u", component=component)


def _evidence(**values):
    return Evidence({name: quantity for name, quantity in values.items()})


# --- Quantity and Evidence ------------------------------------------------------


def test_available_quantity_needs_a_number_and_rejects_nan():
    assert _quantity(1).value == 1.0
    assert _quantity(math.inf).value == math.inf
    with pytest.raises(ValueError, match="NaN"):
        _quantity(math.nan)
    with pytest.raises(TypeError):
        _quantity(None)
    with pytest.raises(TypeError):
        _quantity(True)


def test_numpy_scalars_are_accepted_as_values():
    import numpy as np

    assert _quantity(np.int64(3)).value == 3.0
    assert type(_quantity(np.float32(0.5)).value) is float


@pytest.mark.parametrize("threshold", [math.nan, math.inf])
def test_threshold_must_be_finite(threshold):
    with pytest.raises(ValueError, match="finite"):
        ThresholdPolicy("q", threshold)


def test_unavailable_quantity_needs_a_reason_and_no_value():
    meta = {"direction": "lower_is_better", "units": "u", "component": "rare"}
    quantity = Quantity.insufficient("too few FOVs", **meta)
    assert (quantity.value, quantity.availability) == (None, Availability.INSUFFICIENT_DATA)
    assert quantity.component is Component.RARE
    with pytest.raises(ValueError, match="reason"):
        Quantity.not_applicable("", **meta)
    with pytest.raises(ValueError, match="value=None"):
        Quantity(1.0, "lower_is_better", "u", "rare", "not_applicable", "why")


@pytest.mark.parametrize(
    ("field", "value"),
    [("direction", "smaller"), ("component", "everything"), ("availability", "maybe")],
)
def test_quantity_vocabulary_is_validated(field, value):
    config = {
        "value": None,
        "direction": "lower_is_better",
        "units": "u",
        "component": "population",
        "availability": "insufficient_data",
        "reason": "r",
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        Quantity(**config)


def test_quantity_needs_units():
    with pytest.raises(ValueError, match="units"):
        Quantity.available(1.0, direction="lower_is_better", units="", component="population")


# --- Policies -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "value", "expected"),
    [
        (Direction.LOWER_IS_BETTER, 0.05, Decision.STOP),
        (Direction.LOWER_IS_BETTER, 0.06, Decision.CONTINUE),
        (Direction.HIGHER_IS_BETTER, 0.05, Decision.STOP),
        (Direction.HIGHER_IS_BETTER, 0.04, Decision.CONTINUE),
    ],
)
def test_threshold_policy_follows_the_quantity_direction(direction, value, expected):
    evidence = _evidence(q=_quantity(value, direction=direction))
    assert ThresholdPolicy("q", 0.05).decide(evidence, EFFORT) == (Component.POPULATION, expected)


def test_threshold_policy_continues_on_insufficient_data_and_abstains_when_undefined():
    meta = {"direction": "lower_is_better", "units": "u", "component": "rare"}
    policy = ThresholdPolicy("q", 1.0)
    insufficient = _evidence(q=Quantity.insufficient("few", **meta))
    undefined = _evidence(q=Quantity.not_applicable("negative values", **meta))
    assert policy.decide(insufficient, EFFORT) == (Component.RARE, Decision.CONTINUE)
    assert policy.decide(undefined, EFFORT) == (Component.RARE, None)


def test_threshold_policy_on_a_missing_quantity_raises():
    with pytest.raises(ValueError, match="needs quantity 'missing'"):
        ThresholdPolicy("missing", 1.0).decide(_evidence(q=_quantity(1.0)), EFFORT)


def test_effort_policy_uses_framework_effort():
    assert EffortPolicy("n_fovs", 10).decide(Evidence(), EFFORT)[1] is Decision.STOP
    assert EffortPolicy("n_objects", 101).decide(Evidence(), EFFORT)[1] is Decision.CONTINUE
    with pytest.raises(ValueError):
        EffortPolicy("area", 10)
    with pytest.raises(TypeError):
        EffortPolicy("n_fovs", 2.5)


# --- Combination per objective --------------------------------------------------


def _combined(objective, **values):
    evidence = _evidence(
        **{
            name: _quantity(value, component=name.split("_")[0])
            for name, value in values.items()
        }
    )
    policies = [ThresholdPolicy(name, 1.0) for name in values]
    return decide(evidence, EFFORT, policies, objective)


def test_population_evidence_cannot_answer_rare_discovery():
    decision, components = _combined(RARE, population_cv=0.5)
    assert decision is Decision.UNSUPPORTED
    assert components == {"population": Decision.STOP, "rare": Decision.UNSUPPORTED}


@pytest.mark.parametrize(
    ("population", "rare", "expected"),
    [
        (0.5, 0.5, Decision.STOP),
        (0.5, 2.0, Decision.CONTINUE),
        (2.0, 0.5, Decision.CONTINUE),
    ],
)
def test_balanced_stops_only_when_both_components_stop(population, rare, expected):
    decision, _ = _combined(BALANCED, population_cv=population, rare_miss=rare)
    assert decision is expected


def test_balanced_with_one_component_is_unsupported():
    assert _combined(BALANCED, population_cv=0.5)[0] is Decision.UNSUPPORTED


def test_non_required_components_are_reported_but_do_not_decide():
    decision, components = _combined(POPULATION, population_cv=2.0, rare_miss=0.5)
    assert decision is Decision.CONTINUE
    assert components == {"population": Decision.CONTINUE, "rare": Decision.STOP}


def test_all_policies_of_a_component_must_stop_and_undefined_wins():
    evidence = _evidence(a=_quantity(0.5), b=_quantity(2.0))
    both = [ThresholdPolicy("a", 1.0), ThresholdPolicy("b", 1.0)]
    assert decide(evidence, EFFORT, both, POPULATION)[0] is Decision.CONTINUE
    undefined = _evidence(
        a=_quantity(0.5),
        b=Quantity.not_applicable(
            "x", direction="lower_is_better", units="u", component="population"
        ),
    )
    assert decide(undefined, EFFORT, both, POPULATION)[0] is None


def test_no_policy_is_unsupported_not_stop():
    assert decide(_evidence(q=_quantity(0.0)), EFFORT, [], POPULATION)[0] is Decision.UNSUPPORTED


# --- Strategies built from parts, and redecide ----------------------------------


def _tables():
    objects = pd.DataFrame(
        {"fov_id": [f"F{j}" for j in range(8) for _ in range(5)], "area": range(1, 41)}
    )
    fovs = pd.DataFrame({"fov_id": [f"F{j}" for j in range(8)], "order": range(8)})
    return objects, fovs


def _replay(strategies, objective=POPULATION):
    objects, fovs = _tables()
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=["area"],
        fov_col="fov_id",
        acquisition_order_col="order",
        objective=objective,
    )


def _without_time(result):
    return result.without_timing()


def test_redecide_equals_rerunning_with_the_new_policies():
    estimator = ScalarPrecisionEstimator("area", "fov", "point")
    original = SamplingStrategy([estimator], [ThresholdPolicy("averaging_cv", 0.01)], name="cv")
    for threshold in (0.05, 0.1, 0.2):
        policies = [ThresholdPolicy("averaging_cv", threshold)]
        rerun = _replay([SamplingStrategy([estimator], policies, name="sweep")])
        swept = [redecide(item.result, policies, name="sweep") for item in _replay([original])]
        assert [_without_time(r) for r in swept] == [_without_time(i.result) for i in rerun]
    decisions = {
        threshold: [
            redecide(i.result, [ThresholdPolicy("averaging_cv", threshold)], name="s").decision
            for i in _replay([original])
        ]
        for threshold in (0.01, 0.5)
    }
    assert decisions[0.01] != decisions[0.5]


def test_redecide_can_add_a_component_the_objective_requires():
    [result] = run_strategies(
        *_tables()[:1],
        strategies=[FixedFOVStrategy(3)],
        feature_cols=["area"],
        objective=BALANCED,
    )
    assert result.decision is Decision.UNSUPPORTED
    both = [EffortPolicy("n_fovs", 3), EffortPolicy("n_objects", 10, component="rare")]
    assert redecide(result, both, name="both").decision is Decision.STOP


def test_strategy_name_defaults_to_its_parts():
    strategy = SamplingStrategy(
        [ScalarPrecisionEstimator("area", "object", "point")],
        [ThresholdPolicy("averaging_cv", 0.05)],
    )
    assert strategy.name == "averaging_cv_object+threshold:averaging_cv"
    with pytest.raises(ValueError, match="at least one"):
        SamplingStrategy()
    with pytest.raises(TypeError, match="not an Estimator"):
        SamplingStrategy([ThresholdPolicy("q", 1.0)])


class _Named(Estimator):
    def __init__(self, name, quantity):
        self.name, self.quantity = name, quantity

    def estimate(self, data, *, objective):
        return Evidence({self.quantity: _quantity(1.0)}, {"source": self.name})


def test_estimators_with_clashing_names_raise_instead_of_overwriting():
    with pytest.raises(ValueError, match="both report quantity"):
        _replay([SamplingStrategy([_Named("a", "q"), _Named("b", "q")])])
    with pytest.raises(ValueError, match="both report diagnostic"):
        _replay([SamplingStrategy([_Named("a", "q"), _Named("b", "r")])])


# --- Configuration identity -----------------------------------------------------


def test_config_id_distinguishes_parameters_that_share_a_name():
    # The same human-readable name at two targets are two configurations.
    loose = ScalarPrecisionStrategy("area", 0.05, "object", "point")
    tight = ScalarPrecisionStrategy("area", 0.025, "object", "point")
    assert loose.name == tight.name
    assert loose.config_id != tight.config_id
    assert loose.estimators[0].config_id == tight.estimators[0].config_id


def test_config_id_is_stable_and_ignores_names():
    first = ScalarPrecisionStrategy("area", 0.05, "object", "point")
    assert first.config_id == ScalarPrecisionStrategy("area", 0.05, "object", "point").config_id
    assert len(first.config_id) == 16
    composed = SamplingStrategy(first.estimators, first.policies, name="renamed")
    assert composed.config_id == first.config_id
    # Integer and float thresholds are the same threshold.
    assert ThresholdPolicy("q", 1).config_id == ThresholdPolicy("q", 1.0).config_id


def test_config_id_depends_on_the_seed_and_the_class():
    a = DescriptorSpreadStrategy("area", 0.1, "object", 10, 0)
    b = DescriptorSpreadStrategy("area", 0.1, "object", 10, 1)
    assert a.config_id != b.config_id
    assert EffortPolicy("n_fovs", 3).config_id != EffortPolicy("n_objects", 3).config_id


def test_results_carry_config_id_and_configuration():
    strategy = ScalarPrecisionStrategy("area", 0.05, "fov", "point")
    [item, *_] = _replay([strategy])
    assert item.result.config_id == strategy.config_id
    [estimator] = item.result.configuration["estimators"]
    assert estimator["config_id"] == strategy.estimators[0].config_id
    assert estimator["type"].endswith("ScalarPrecisionEstimator")
    [policy] = item.result.configuration["policies"]
    assert policy["parameters"] == {"quantity": "averaging_cv", "threshold": 0.05}


def test_non_serializable_parameters_are_rejected_for_provenance():
    class _Opaque(Estimator):
        name = "opaque"

        @property
        def parameters(self):
            return {"transform": object()}

        def estimate(self, data, *, objective):
            return Evidence()

    with pytest.raises(TypeError, match="JSON-serializable"):
        _ = _Opaque().config_id
