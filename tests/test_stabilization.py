"""Tests for the Marcotti et al.-style descriptor-stabilization baselines."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    run_strategies,
)
from morphocoverage.strategies.stabilization import describe

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _tables(values_per_fov):
    rows = [
        {"image_file": f"f{j:03d}.tif", "area": value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["image_file", "area"])
    fovs = pd.DataFrame(
        {
            "image_file": [f"f{j:03d}.tif" for j in range(len(values_per_fov))],
            "order": range(1, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _evaluate(objects, fovs, strategy, feature_cols=("area",)):
    [result] = run_strategies(
        objects, fovs=fovs, strategies=[strategy], feature_cols=list(feature_cols),
        objective=POPULATION,
    )
    return result


def _replay(objects, fovs, strategies, order="order", feature_cols=("area",)):
    return evaluate_acquisition_prefixes(
        objects, fovs, strategies=strategies, feature_cols=list(feature_cols),
        acquisition_order_col=order, objective=POPULATION,
    )


def _score(result):
    """The strategy's single evidence quantity."""
    [quantity] = result.evidence.quantities.values()
    return quantity.value


def _spread(tolerance, unit, n_resamples=50, seed=0):
    return DescriptorSpreadStrategy("area", tolerance, unit, n_resamples, seed)


# --- Descriptors -----------------------------------------------------------------


def test_descriptors_match_hand_calculation():
    result = describe(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert result == pytest.approx(
        {"mean": 3.0, "median": 3.0, "sd": np.std([1, 2, 3, 4, 5], ddof=1), "iqr": 2.0}
    )


def test_descriptors_are_undefined_for_fewer_than_two_values():
    assert all(np.isnan(v) for v in describe(np.array([1.0])).values())


# --- Cumulative plateau (Fig. 2B) --------------------------------------------------


def _plateau_oracle(objects, fovs, tolerance, window):
    """First replay step at which the plateau rule stops, computed independently."""
    ordered = fovs.sort_values("order")["image_file"].tolist()
    states = []
    for step in range(1, len(ordered) + 1):
        visible = objects[objects["image_file"].isin(ordered[:step])]["area"].to_numpy()
        if objects["image_file"].eq(ordered[step - 1]).any():
            states.append(describe(visible))
        recent = states[-(window + 1):]
        if len(recent) == window + 1:
            ranges = [np.ptp([s[name] for s in recent]) for name in recent[0]]
            if not np.isnan(ranges).any() and max(ranges) <= tolerance:
                return step
    return None


@pytest.mark.parametrize("tolerance, window", [(0.3, 2), (0.15, 3), (0.05, 5)])
def test_plateau_matches_an_independent_oracle(tolerance, window):
    scenario = SCENARIOS["neighbour_correlation"]()
    objects = scenario.objects.rename(columns={"feature_a": "area"})
    fovs = scenario.fovs.rename(columns={ORDER_COL: "order"})
    strategy = DescriptorPlateauStrategy("area", tolerance, window, "order")
    results = _replay(objects, fovs, [strategy])
    stop = next((r.step for r in results if r.result.decision == "stop"), None)
    assert stop == _plateau_oracle(objects, fovs, tolerance, window)


def test_plateau_waits_for_window_plus_one_fovs_with_objects():
    objects, fovs = _tables([[1.0, 3.0]] * 4)
    results = _replay(objects, fovs, [DescriptorPlateauStrategy("area", 10.0, 3, "order")])
    assert [r.result.decision for r in results] == ["continue"] * 3 + ["stop"]


def test_empty_fovs_do_not_create_a_plateau():
    # Three empty FOVs after two FOVs of cells: no new cells, so no new states.
    objects, fovs = _tables([[1.0, 3.0], [2.0, 6.0], [], [], []])
    results = _replay(objects, fovs, [DescriptorPlateauStrategy("area", 10.0, 2, "order")])
    assert all(r.result.decision == "continue" for r in results)


def test_plateau_uses_acquisition_order_not_table_order():
    scenario = SCENARIOS["neighbour_correlation"]()
    strategy = DescriptorPlateauStrategy("feature_a", 0.2, 2, ORDER_COL)
    shuffled = scenario.fovs.sample(frac=1.0, random_state=0)
    first = _evaluate(scenario.objects, scenario.fovs, strategy, FEATURE_COLS)
    second = _evaluate(scenario.objects, shuffled, strategy, FEATURE_COLS)
    assert _score(first) == _score(second)


# --- Resampling spread (Fig. 2A) ---------------------------------------------------


@pytest.mark.parametrize("unit", ["object", "fov"])
def test_identical_values_have_zero_spread(unit):
    objects, fovs = _tables([[2.0, 2.0]] * 5)
    result = _evaluate(objects, fovs, _spread(0.01, unit))
    assert _score(result) == 0.0 and result.decision == "stop"


def test_spread_shrinks_with_more_cells():
    rng = np.random.default_rng(1)
    small = _tables([rng.normal(10, 1, 40)])
    large = _tables([rng.normal(10, 1, 4000)])
    assert _score(_evaluate(*large, _spread(1.0, "object"))) < _score(
        _evaluate(*small, _spread(1.0, "object"))
    ) / 5


def test_spatial_clustering_inflates_only_the_fov_variant():
    # Same cells, randomly mixed across FOVs or sorted so each FOV is homogeneous.
    values = np.random.default_rng(3).normal(10, 1, 400)
    mixed = _tables(values.reshape(20, 20))
    clustered = _tables(np.sort(values).reshape(20, 20))
    cells_mixed = _score(_evaluate(*mixed, _spread(1.0, "object")))
    cells_clustered = _score(_evaluate(*clustered, _spread(1.0, "object")))
    fovs_mixed = _score(_evaluate(*mixed, _spread(1.0, "fov")))
    fovs_clustered = _score(_evaluate(*clustered, _spread(1.0, "fov")))
    # Same cells: the object variant differs only through which cells the draws hit.
    assert cells_clustered == pytest.approx(cells_mixed, rel=0.5)
    assert fovs_mixed < 2 * cells_mixed
    assert fovs_clustered > 3 * cells_clustered


def test_spread_is_reproducible_and_seed_dependent():
    rng = np.random.default_rng(2)
    tables = _tables(rng.normal(10, 1, (10, 20)))
    for unit in ("object", "fov"):
        same = [_score(_evaluate(*tables, _spread(1.0, unit, seed=7))) for _ in range(2)]
        other = _score(_evaluate(*tables, _spread(1.0, unit, seed=8)))
        assert same[0] == same[1] != other


def test_fov_spread_counts_empty_fovs_as_sampled_units():
    # 2 FOVs with cells and 8 empty ones: some FOV resamples contain fewer than 2
    # cells, so descriptors are undefined and the spread is unbounded. If empty
    # FOVs were dropped, only the 2 non-empty FOVs would be resampled.
    objects, fovs = _tables([[1.0, 1.0], [3.0, 3.0]] + [[]] * 8)
    result = _evaluate(objects, fovs, _spread(1.0, "fov"))
    assert _score(result) == float("inf") and result.decision == "continue"


# --- Input rules -------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DescriptorPlateauStrategy("area", 0.0, 3, "order"),
        lambda: DescriptorPlateauStrategy("area", 0.1, 0, "order"),
        lambda: DescriptorPlateauStrategy("area", 0.1, 3, "order", descriptors=()),
        lambda: DescriptorPlateauStrategy("area", 0.1, 3, "order", descriptors=("mode",)),
        lambda: DescriptorSpreadStrategy("area", 0.1, "cell", 10, 0),
        lambda: DescriptorSpreadStrategy("area", 0.1, "object", 1, 0),
        lambda: DescriptorSpreadStrategy("area", -1.0, "object", 10, 0),
        lambda: DescriptorSpreadStrategy("area", float("inf"), "object", 10, 0),
    ],
)
def test_invalid_configuration_is_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_feature_must_be_explicit():
    objects, fovs = _tables([[1.0, 2.0]])
    objects["stage_x"] = 1.0
    with pytest.raises(ValueError, match="explicitly selected"):
        _evaluate(objects, fovs, DescriptorSpreadStrategy("stage_x", 0.1, "object", 10, 0))


def test_evidence_states_its_meaning():
    objects, fovs = _tables([[1.0, 2.0, 4.0]] * 4)
    plateau = _evaluate(objects, fovs, DescriptorPlateauStrategy("area", 0.1, 2, "order"))
    spread = _evaluate(objects, fovs, _spread(0.1, "object"))
    for result, name in ((plateau, "max_descriptor_range"), (spread, "max_descriptor_spread")):
        quantity = result.evidence.quantities[name]
        assert (quantity.direction, quantity.component) == ("lower_is_better", "population")
        assert quantity.units == "units of 'area'"
    ranges = plateau.diagnostics["window_ranges"]
    assert plateau.evidence.value("max_descriptor_range") == max(ranges.values())
    # Too few FOVs with objects: insufficient data, so the rule continues.
    early = _evaluate(objects, fovs, DescriptorPlateauStrategy("area", 0.1, 5, "order"))
    assert early.evidence.quantities["max_descriptor_range"].availability == "insufficient_data"
    assert early.decision == "continue"


def test_names_and_parameters_record_the_variant():
    assert DescriptorPlateauStrategy("area", 0.1, 3, "order").name == "descriptor_plateau"
    spread = DescriptorSpreadStrategy("area", 0.1, "fov", 100, 3, descriptors=("iqr",))
    assert spread.name == "descriptor_spread_fov"
    assert spread.parameters == {
        "feature": "area", "tolerance": 0.1, "resampling_unit": "fov",
        "n_resamples": 100, "random_state": 3, "descriptors": ["iqr"],
    }


# --- Counterexample --------------------------------------------------------------


def test_current_practice_rules_stop_before_a_late_clustered_rare_state():
    # Known failure mode, by construction, not evidence about real data: 1% of
    # cells form a distinct state, all in FOVs 31-32. Every cell-number rule
    # below stabilizes on the common population and stops long before them.
    scenario = SCENARIOS["rare_clustered"]()
    strategies = [
        DescriptorPlateauStrategy("feature_a", 0.1, 3, ORDER_COL),
        DescriptorSpreadStrategy("feature_a", 0.3, "object", 100, 0),
        ScalarPrecisionStrategy("feature_a", 0.01, "object", "upper_95"),
    ]
    results = _replay(scenario.objects, scenario.fovs, strategies, ORDER_COL, FEATURE_COLS)
    for strategy in strategies:
        stop = next(
            r for r in results if r.result.strategy == strategy.name and r.result.decision == "stop"
        )
        assert stop.step < scenario.truth["first_rare_step"]
        seen = {fov[-1] for fov in stop.acquired_fovs}
        rare_fovs = set(scenario.objects.loc[scenario.objects["true_state"].eq(1), "image_file"])
        assert not seen & rare_fovs


def test_current_practice_rules_are_unsupported_for_rare_discovery():
    # The same rules asked about the rare state do not claim to answer it:
    # they report population evidence only, so the decision is UNSUPPORTED.
    scenario = SCENARIOS["rare_clustered"]()
    rare = ObjectiveConfig(
        SamplingObjective.RARE_DISCOVERY,
        rare=RareTarget(
            object_prevalence=0.01,
            fov_incidence=0.05,
            detection_probability=0.95,
            min_fov_occurrences=2,
            phenotype="true_state column",
        ),
    )
    strategies = [
        DescriptorPlateauStrategy("feature_a", 0.1, 3, ORDER_COL),
        DescriptorSpreadStrategy("feature_a", 0.3, "object", 100, 0),
        ScalarPrecisionStrategy("feature_a", 0.01, "object", "upper_95"),
    ]
    results = evaluate_acquisition_prefixes(
        scenario.objects, scenario.fovs, strategies=strategies, feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL, objective=rare,
    )
    assert {r.result.decision for r in results} == {"unsupported"}
    # The population verdict is still reported, for comparison.
    assert "stop" in {r.result.component_decisions["population"] for r in results}


@pytest.mark.parametrize(
    "strategy",
    [
        DescriptorPlateauStrategy("area", 0.1, 2, "order"),
        DescriptorSpreadStrategy("area", 0.1, "object", 20, 0),
    ],
)
def test_descriptor_rules_do_not_silently_drop_missing_values(strategy):
    objects, fovs = _tables([[1.0, float("nan")], [2.0, 3.0], [4.0, 5.0]])
    result = _evaluate(objects, fovs, strategy)
    [quantity] = result.evidence.quantities.values()
    assert quantity.availability == "not_applicable"
    assert result.decision is None
    assert "missing values" in quantity.reason


@pytest.mark.parametrize(
    "strategy",
    [
        DescriptorPlateauStrategy("area", 0.1, 2, "order"),
        DescriptorSpreadStrategy("area", 0.1, "object", 20, 0),
    ],
)
def test_cell_stabilization_rules_are_not_applicable_to_slice_observations(strategy):
    objects, fovs = _tables([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[strategy],
        feature_cols=["area"],
        objective=POPULATION,
        observation_unit="slice",
    )
    [quantity] = result.evidence.quantities.values()
    assert quantity.availability == "not_applicable"
    assert "'slice'" in quantity.reason
    assert result.decision is None


def test_fov_spread_variant_applies_to_any_observation_unit():
    # Resampling whole FOVs does not assume the rows are cells.
    objects, fovs = _tables([[2.0, 2.0]] * 5)
    results = [
        run_strategies(
            objects,
            fovs=fovs,
            strategies=[_spread(0.01, "fov")],
            feature_cols=["area"],
            objective=POPULATION,
            observation_unit=unit,
        )[0]
        for unit in ("slice", "cell")
    ]
    assert [r.decision for r in results] == ["stop", "stop"]
    assert results[0].evidence == results[1].evidence
