"""Tests for evidence, decision policies, their combination per objective, and config ids."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_results import _available_quantity, _strategy_result
from _helpers_tables import _six_cell_tables
from _helpers_workflows import _scenario_report

import morphocoverage as mc
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
    check_tables,
    evaluate_acquisition_prefixes,
    redecide,
    run_strategies,
)
from morphocoverage._frozen import own
from morphocoverage.policies import decide
from morphocoverage.provenance import config_hash
from morphocoverage.workflows import Setup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios

_RARE = RareTarget(
    fov_incidence=0.1, detection_probability=0.9, min_fov_occurrences=2, phenotype="label"
)
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
RARE = ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=_RARE)
BALANCED = ObjectiveConfig(SamplingObjective.BALANCED, rare=_RARE)
EFFORT = Effort(n_fovs=10, n_observations=100)


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
    assert EffortPolicy("n_observations", 101).decide(Evidence(), EFFORT)[1] is Decision.CONTINUE
    with pytest.raises(ValueError):
        EffortPolicy("area", 10)
    with pytest.raises(TypeError):
        EffortPolicy("n_fovs", 2.5)


# --- Combination per objective --------------------------------------------------


def _combined(objective, **values):
    evidence = _evidence(
        **{name: _quantity(value, component=name.split("_")[0]) for name, value in values.items()}
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


def test_distribution_evidence_cannot_answer_the_population_objective():
    decision, components = _combined(POPULATION, distribution_se=0.5)
    assert decision is Decision.UNSUPPORTED
    assert components == {"distribution": Decision.STOP, "population": Decision.UNSUPPORTED}
    phenotype = ObjectiveConfig(SamplingObjective.PHENOTYPE)
    assert _combined(phenotype, population_cv=0.5, distribution_se=0.5)[0] is Decision.STOP
    assert _combined(phenotype, population_cv=0.5, distribution_se=2.0)[0] is Decision.CONTINUE
    assert _combined(phenotype, population_cv=0.5)[0] is Decision.UNSUPPORTED


def _result(objective, **values):
    from morphocoverage.results import StrategyResult

    decision, components = _combined(objective, **values)
    return StrategyResult(
        strategy="rule",
        config_id="x",
        objective=objective,
        evidence=Evidence(),
        decision=decision,
        component_decisions=components,
    )


def test_the_rule_verdict_is_what_the_rule_said_whatever_the_objective():
    # A novelty rule under the population objective: unsupported for the objective,
    # but the rule itself said stop, which is what a self-assessment or replay reads.
    novelty = _result(POPULATION, distribution_se=0.5)
    assert novelty.decision is Decision.UNSUPPORTED
    assert novelty.rule_decision is Decision.STOP
    mean = _result(RARE, population_cv=2.0)
    assert (mean.decision, mean.rule_decision) == (Decision.UNSUPPORTED, Decision.CONTINUE)
    both = _result(POPULATION, population_cv=0.5, distribution_se=2.0)
    assert (both.decision, both.rule_decision) == (Decision.STOP, Decision.CONTINUE)


def test_decision_view_shows_every_target_and_the_overall():
    from morphocoverage.results import decision_view

    phenotype = ObjectiveConfig(SamplingObjective.PHENOTYPE)
    mean = _result(phenotype, population_cv=0.5)
    fingerprint = _result(phenotype, distribution_se=2.0)
    view = decision_view([mean, fingerprint])
    assert view == (
        "Population summary: stop (1 rule); Phenotypic fingerprint: continue (1 rule); "
        "Overall (phenotype objective): continue"
    )
    # A required target no rule decides is said so, and the overall is unsupported.
    alone = decision_view([mean])
    assert "Phenotypic fingerprint: no rule decides it" in alone
    assert alone.endswith(
        "Overall (phenotype objective): unsupported (no rule decides the phenotypic fingerprint)"
    )
    # Under the population objective a fingerprint rule is shown but does not decide.
    view = decision_view(
        [_result(POPULATION, population_cv=0.5), _result(POPULATION, distribution_se=0.5)]
    )
    # An overall stop says what it rests on (audit 4, Q2): without an order check,
    # the fields as a random sample of the well's positions.
    assert view.endswith(
        "Overall (population objective): stop; treats the fields as a random sample of the "
        "well's positions; a raster over a spatial trend can stop early"
    )
    with pytest.raises(ValueError, match="one objective"):
        decision_view([mean, _result(POPULATION, population_cv=0.5)])


def test_a_rare_rule_beside_another_objective_adds_no_rare_caveat_to_its_stop():
    from morphocoverage.results import decision_view

    view = decision_view(
        [_result(POPULATION, population_cv=0.5), _result(POPULATION, rare_miss=0.5)]
    )
    assert "Rare phenotype: stop (1 rule)" in view
    assert "exchangeable" not in view.split("Overall")[1]
    balanced = decision_view([_result(BALANCED, population_cv=0.5, rare_miss=0.5)])
    assert "exchangeable" in balanced.split("Overall")[1]
    assert "exchangeable" in decision_view([_result(RARE, rare_miss=0.5)]).split("Overall")[1]


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
    both = [EffortPolicy("n_fovs", 3), EffortPolicy("n_observations", 10, component="rare")]
    assert redecide(result, both, name="both").decision is Decision.STOP


def test_strategy_name_defaults_to_its_parts():
    strategy = SamplingStrategy(
        [ScalarPrecisionEstimator("area", "object", "point")],
        [ThresholdPolicy("averaging_cv", 0.05)],
    )
    assert strategy.name == "averaging_cv_object+threshold:averaging_cv"
    with pytest.raises(ValueError, match="at least one"):
        SamplingStrategy()
    with pytest.raises(ValueError, match="at least one"):  # a name is no component (#323)
        SamplingStrategy([], [], name="empty")
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
    assert EffortPolicy("n_fovs", 3).config_id != EffortPolicy("n_observations", 3).config_id


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


def test_redecide_refuses_a_result_without_recorded_configuration():
    [item, *_] = _replay([ScalarPrecisionStrategy("area", 0.05, "fov", "point")])
    broken = replace(item.result, configuration={})
    with pytest.raises(ValueError, match="no recorded estimator configuration"):
        redecide(broken, [ThresholdPolicy("averaging_cv", 0.1)], name="x")


def test_the_former_n_objects_effort_is_refused():
    with pytest.raises(ValueError, match="effort must be one of"):
        EffortPolicy("n_objects", 5)


@pytest.mark.parametrize(
    "name", ["a10_embed2_several_norm_cells_none_tol", "a21_embed1_one_norm_cells_none_tol"]
)
def test_a_setup_with_no_target_has_no_component_and_nothing_decides_a_stop(name: str) -> None:
    # C12: "a stop needs every declared target (phenotypic fingerprint)" with none declared.
    setup = Setup(**make_scenarios.load(name).setup)
    objective = setup.objective()
    assert objective.objective is SamplingObjective.NO_TARGET
    assert objective.required_components == frozenset()
    said = [line for line in setup.declaration() if line.startswith("Live decision")]
    assert said == ["Live decision: no target: nothing decides a stop."]
    for detail in ("short", "full"):
        text = " ".join(_scenario_report(name).summary(detail).split())
        assert "a stop needs every declared target" not in text
    assert "Live decision: no target: nothing decides a stop." in text
    # Whatever a rule says, no target means no stop.
    stop = EffortPolicy("n_fovs", 1, Component.DISTRIBUTION)
    overall, per_component = decide(
        Evidence(), Effort(n_fovs=10, n_observations=100), [stop], objective
    )
    assert per_component == {"distribution": Decision.STOP}
    assert overall is Decision.UNSUPPORTED


def test_evidence_owns_its_quantities_and_diagnostics():
    quantities = {"score": _available_quantity()}
    diagnostics = {"note": ["a"], "inner": {"k": 1}}
    evidence = Evidence(quantities, diagnostics)
    quantities["n_fovs"] = "not a Quantity"  # a reserved name and a wrong type, added later
    quantities["other"] = _available_quantity()
    diagnostics["note"].append("b")
    assert set(evidence.quantities) == {"score"}
    assert evidence.diagnostics["note"] == ["a"]
    with pytest.raises(TypeError, match="read-only"):
        evidence.quantities["n_fovs"] = "x"
    with pytest.raises(TypeError, match="read-only"):
        evidence.diagnostics["inner"]["k"] = 2
    assert evidence.value("score") == 0.1  # ordinary reads are unchanged


def test_evidence_still_validates_what_it_is_given():
    with pytest.raises(ValueError, match="framework-owned"):
        Evidence({"n_fovs": _available_quantity()})
    with pytest.raises(TypeError, match="must be a Quantity"):
        Evidence({"x": 1.0})


def test_redecide_builds_an_independent_snapshot():
    result = _strategy_result()
    again = redecide(result, [ThresholdPolicy("score", 0.2)], name="t")
    assert again.configuration["estimators"] == result.configuration["estimators"]
    assert again.configuration is not result.configuration
    assert again.parameters is not again.configuration
    assert own({}) == {}


def test_a16_numpy_numbers_configure_targets_and_policies_like_python_ones():
    rare = mc.RareTarget(
        detection_probability=np.float64(0.95),
        min_fov_occurrences=np.int64(2),
        phenotype="label",
        object_prevalence=np.float32(0.5),
    )
    assert type(rare.min_fov_occurrences) is int and type(rare.object_prevalence) is float
    assert rare == mc.RareTarget(
        detection_probability=0.95, min_fov_occurrences=2, phenotype="label", object_prevalence=0.5
    )
    policy = mc.EffortPolicy("n_fovs", np.int64(10))
    assert type(policy.target) is int
    assert policy.config_id == mc.EffortPolicy("n_fovs", 10).config_id
    strategy = mc.FixedFOVStrategy(np.int32(4))
    assert type(strategy.target_fovs) is int
    assert strategy.config_id == mc.FixedFOVStrategy(4).config_id
    with pytest.raises(TypeError):
        mc.FixedFOVStrategy(np.float64(4.0))
    with pytest.raises(TypeError):
        mc.EffortPolicy("n_fovs", True)


def test_a17_config_ids_tell_tuples_from_lists_and_number_keys_from_text():
    assert config_hash("x", {1: "a"}) != config_hash("x", {"1": "a"})
    assert config_hash("x", {"k": (1, 2)}) != config_hash("x", {"k": [1, 2]})
    assert config_hash("x", {1: "a", "b": 2}) == config_hash("x", {"b": 2, 1: "a"})
    # Parameters JSON already keeps apart keep the id earlier versions gave them.
    parameters = {"target": 3, "names": ["a", "b"], "nested": {"z": None, "y": 1.5}}
    payload = json.dumps(
        {"identity": "x", "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert config_hash("x", parameters) == hashlib.sha256(payload.encode()).hexdigest()[:16]
    with pytest.raises(TypeError):
        config_hash("x", {"__tuple__": 1})


class _Constant(Estimator):
    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = value

    @property
    def parameters(self):
        return {"value": self.value}

    def estimate(self, data, *, objective):
        return Evidence(
            {
                "q": Quantity.available(
                    self.value,
                    direction="lower_is_better",
                    units="fraction",
                    component="population",
                )
            }
        )


def test_a8_explain_prints_the_digits_that_separate_value_and_threshold():
    cells, fovs = _six_cell_tables()
    strategy = SamplingStrategy([_Constant(0.2504)], [ThresholdPolicy("q", 0.25)], name="c")
    [result] = run_strategies(
        cells,
        fovs=fovs,
        strategies=[strategy],
        feature_cols=["area"],
        objective=ObjectiveConfig(),
        group_cols=[],
        observation_unit="cell",
    )
    assert result.explain().startswith("continue: q = 0.2504 (stop at <= 0.25, population)")


def test_a9_counts_are_singular_for_one():
    cells, fovs = _six_cell_tables()
    check = check_tables(
        cells.iloc[:1], fovs.iloc[:1], feature_cols=["area"], observation_unit="Cells"
    )
    assert check.understood[0] == "1 observation (cell); 1 row in the FOV table."
    strategy = SamplingStrategy([_Constant(0.1)], [ThresholdPolicy("q", 0.25)], name="c")
    [result] = run_strategies(
        cells.iloc[:1],
        fovs=fovs.iloc[:1],
        strategies=[strategy],
        feature_cols=["area"],
        objective=ObjectiveConfig(),
        group_cols=[],
        observation_unit="fov_summary",
    )
    # A stop then says what it rests on (audit 4, Q2).
    assert "after 1 FOV, 1 field summary; treats the fields as a random sample" in result.explain()


def test_a19_one_combination_of_decisions():
    from morphocoverage import policies

    assert not hasattr(policies, "_combine")
    assert "redecide" in policies.__doc__ and "strategy_config_id" in policies.__doc__
