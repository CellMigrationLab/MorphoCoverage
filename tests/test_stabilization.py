"""Tests for the Marcotti et al.-style descriptor-stabilization baselines."""

import json
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from _helpers_strategies import (
    DISTRIBUTION,
    _acquisition,
    _area_run,
    _area_tables,
    _joined_words,
    _ordered_acquisition,
    _quantity,
    _run,
    _summary,
    _well,
    _words,
    _x_replay,
    _x_run,
)
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    ObjectiveConfig,
    OnlineSession,
    RareTarget,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    run_strategies,
    validate_input_tables,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.strategies import (
    DescriptorSpreadEstimator,
    FOVBootstrapStrategy,
    _descriptor_resamples,
    stabilization,
)
from morphocoverage.strategies.stabilization import describe
from morphocoverage.sufficiency import sufficiency_report

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
        objects,
        fovs=fovs,
        strategies=[strategy],
        feature_cols=list(feature_cols),
        objective=POPULATION,
    )
    return result


def _replay(objects, fovs, strategies, order="order", feature_cols=("area",)):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=list(feature_cols),
        acquisition_order_col=order,
        objective=POPULATION,
    )


def _score(result):
    """The largest range or spread over every descriptor (``max_descriptor_*``)."""
    [name] = [n for n in result.evidence.quantities if n.startswith("max_descriptor_")]
    return result.evidence.quantities[name].value


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
        recent = states[-(window + 1) :]
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
    stop = next((r.step for r in results if r.result.rule_decision == "stop"), None)
    assert stop == _plateau_oracle(objects, fovs, tolerance, window)


def test_plateau_waits_for_window_plus_one_fovs_with_objects():
    objects, fovs = _tables([[1.0, 3.0]] * 4)
    results = _replay(objects, fovs, [DescriptorPlateauStrategy("area", 10.0, 3, "order")])
    assert [r.result.rule_decision for r in results] == ["continue"] * 3 + ["stop"]


def test_empty_fovs_do_not_create_a_plateau():
    # Three empty FOVs after two FOVs of cells: no new cells, so no new states.
    objects, fovs = _tables([[1.0, 3.0], [2.0, 6.0], [], [], []])
    results = _replay(objects, fovs, [DescriptorPlateauStrategy("area", 10.0, 2, "order")])
    assert all(r.result.rule_decision == "continue" for r in results)


def test_plateau_uses_acquisition_order_not_table_order():
    scenario = SCENARIOS["neighbour_correlation"]()
    strategy = DescriptorPlateauStrategy("feature_a", 0.2, 2, ORDER_COL)
    shuffled = scenario.fovs.sample(frac=1.0, random_state=0)
    first = _evaluate(scenario.objects, scenario.fovs, strategy, FEATURE_COLS)
    second = _evaluate(scenario.objects, shuffled, strategy, FEATURE_COLS)
    assert _score(first) == _score(second)


# --- Resampling spread (Fig. 2A) ---------------------------------------------------


@pytest.mark.parametrize("unit", ["object", "fov"])
def test_identical_values_are_not_yet_enough_data(unit):
    # A spread of 0 from values that are all alike says nothing (audit 4, P4-1): not a stop.
    objects, fovs = _tables([[2.0, 2.0]] * 5)
    result = _evaluate(objects, fovs, _spread(0.01, unit))
    assert _score(result) is None and result.rule_decision == "continue"
    assert result.evidence.quantities["max_descriptor_spread"].reason == (
        "no spread among the objects so far"
    )
    # FOVs that are all alike, each with a spread, have a spread of 0 across resamples.
    objects, fovs = _tables([[2.0, 3.0]] * 5)
    result = _evaluate(objects, fovs, _spread(0.01, unit))
    assert (_score(result) == 0.0) == (unit == "fov")


def test_spread_shrinks_with_more_cells():
    rng = np.random.default_rng(1)
    small = _tables([rng.normal(10, 1, 40)])
    large = _tables([rng.normal(10, 1, 4000)])
    assert (
        _score(_evaluate(*large, _spread(1.0, "object")))
        < _score(_evaluate(*small, _spread(1.0, "object"))) / 5
    )


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
    # 5 FOVs with one cell and 45 empty ones: some FOV resamples contain fewer than 2
    # cells, so descriptors are undefined and the spread is unbounded. If empty
    # FOVs were dropped, only the 5 non-empty FOVs would be resampled, always 5 cells.
    # (The per-FOV spread needs 5 FOVs with objects, audit 4, P4-6.)
    objects, fovs = _tables([[1.0], [2.0], [3.0], [4.0], [5.0]] + [[]] * 45)
    result = _evaluate(objects, fovs, _spread(1.0, "fov"))
    assert _score(result) == float("inf") and result.rule_decision == "continue"


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
        # The paper's four descriptors include SD and IQR: the distribution of phenotypes.
        assert (quantity.direction, quantity.component) == ("lower_is_better", "distribution")
        assert quantity.units == "units of 'area'"
    location_only = _evaluate(
        objects, fovs, DescriptorSpreadStrategy("area", 0.1, "object", 10, 0, ("mean", "median"))
    )
    assert location_only.evidence.quantities["max_descriptor_spread"].component == "population"
    ranges = plateau.diagnostics["window_ranges"]
    assert plateau.evidence.value("max_descriptor_range") == max(ranges.values())
    # Too few FOVs with objects: insufficient data, so the rule continues.
    early = _evaluate(objects, fovs, DescriptorPlateauStrategy("area", 0.1, 5, "order"))
    assert early.evidence.quantities["max_descriptor_range"].availability == "insufficient_data"
    assert early.rule_decision == "continue"


def test_names_and_parameters_record_the_variant():
    assert DescriptorPlateauStrategy("area", 0.1, 3, "order").name == "descriptor_plateau"
    spread = DescriptorSpreadStrategy("area", 0.1, "fov", 100, 3, descriptors=("iqr",))
    assert spread.name == "descriptor_spread_fov"
    assert spread.parameters == {
        "feature": "area",
        "tolerance": 0.1,
        "resampling_unit": "fov",
        "n_resamples": 100,
        "random_state": 3,
        "descriptors": ["iqr"],
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
            r
            for r in results
            if r.result.strategy == strategy.name and r.result.rule_decision == "stop"
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
        scenario.objects,
        scenario.fovs,
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=rare,
    )
    assert {r.result.decision for r in results} == {"unsupported"}
    # Each rule's own verdict is still reported, for comparison.
    assert "stop" in {r.result.rule_decision for r in results}


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
    for quantity in result.evidence.quantities.values():
        assert quantity.availability == "not_applicable"
        assert "missing values" in quantity.reason
    assert result.rule_decision is None


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
    for quantity in result.evidence.quantities.values():
        assert quantity.availability == "not_applicable"
        assert "(slice)" in quantity.reason
    assert result.rule_decision is None


def test_fov_spread_variant_applies_to_any_observation_unit():
    # Resampling whole FOVs does not assume the rows are cells (alike FOVs, each with a
    # spread: values all alike are not yet enough data, audit 4, P4-1).
    objects, fovs = _tables([[2.0, 3.0]] * 5)
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
    assert [r.rule_decision for r in results] == ["stop", "stop"]
    assert results[0].evidence == results[1].evidence


def test_b2_replay_prefixes_equal_the_live_arrival_order():
    # Object-drawing estimators see the same rows in a replay (full
    # recomputation) and live, also when the table is not sorted by FOV.
    strategy = DescriptorSpreadStrategy("feature_a", 0.3, "object", 20, 0)
    built = SCENARIOS["duplicated_objects"]()
    assert not built.objects.image_file.is_monotonic_increasing
    replayed = evaluate_acquisition_prefixes(
        built.objects,
        built.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )
    by_step = {(tuple(i.result.group.items()), i.step): i.result for i in replayed}
    session = OnlineSession(
        strategies=[strategy], feature_cols=list(FEATURE_COLS), objective=POPULATION
    )
    key = ["condition", "biological_repeat", "image_file"]
    for _, meta in built.fovs.sort_values(ORDER_COL).iterrows():
        mask = (built.objects[key] == meta[key]).all(axis=1)
        (live,) = session.add_fov(built.objects[mask], meta.to_dict())
        reference = by_step[(tuple(live.group.items()), live.n_fovs)]
        assert live.without_timing().evidence == reference.without_timing().evidence
        assert live.rule_decision == reference.rule_decision


def test_c5_descriptor_plateau_refuses_a_missing_order_value():
    objects, fovs = _area_tables([[1.0, 2.0], [10.0, 11.0], [3.0, 2.5], [2.0, 1.5]])
    fovs["acq"] = [1.0, np.nan, 2.0, 3.0]
    strategy = DescriptorPlateauStrategy("area", 100.0, 1, "acq", descriptors=("mean",))
    with pytest.raises(ValueError, match="'acq' has missing values; the descriptor-plateau"):
        _run(objects, fovs, strategy)


def test_c17_object_resampling_is_chunked_and_gives_the_same_spreads(monkeypatch):
    rng = np.random.default_rng(0)
    objects = pd.DataFrame(
        {
            "image_file": rng.integers(0, 50, 5000).astype(str),
            "area": rng.gamma(4, 50, 5000),
        }
    )
    data = validate_input_tables(
        objects, fovs=pd.DataFrame({"image_file": np.arange(50).astype(str)}), feature_cols=["area"]
    )
    estimator = DescriptorSpreadEstimator("area", "object", 100, 0)
    whole = estimator.estimate(data, objective=POPULATION)
    monkeypatch.setattr(_descriptor_resamples, "_MAX_RESAMPLED_VALUES", 12_000)
    tracemalloc.start()
    chunked = estimator.estimate(data, objective=POPULATION)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert chunked.diagnostics["resample_spreads"] == whole.diagnostics["resample_spreads"]
    # 100 resamples of 5000 values would hold 4 MB of values and 4 MB of indices at once.
    assert peak < 3_000_000


PHENOTYPE = ObjectiveConfig(SamplingObjective.PHENOTYPE)


def test_c3_the_object_unit_jcs_rule_states_its_random_object_assumption():
    rng = np.random.default_rng(1)
    objects, fovs = _acquisition([list(rng.normal(10, 2, 15)) for _ in range(8)])
    text = _summary(
        objects,
        fovs,
        [
            DescriptorSpreadStrategy("area", 5.0, "object", 50, 0),
            DescriptorSpreadStrategy("area", 5.0, "fov", 50, 0),
        ],
        objective=DISTRIBUTION,
    )
    lines = {line.split(" (")[0].strip(" -"): line for line in text.splitlines() if "(" in line}
    clause = "assumes a spatially uniform sample"
    assert clause in lines["JCS spread, per object, ≤5"]
    assert clause not in lines["JCS spread, per field, ≤5"]
    # Stated once in the short summary, for the object-unit rule only.
    assert "JCS spread, per object, ≤5 treats objects as independent draws: it assumes" in text
    assert "JCS spread, per field, ≤5 treats" not in text


def test_c10_numpy_integers_are_accepted_and_stored_as_python_ints():
    strategies = [
        DescriptorPlateauStrategy("area", 0.1, np.int64(3), "acq"),
        DescriptorSpreadStrategy("area", 0.1, "fov", np.int64(100), np.int64(0)),
        FOVBootstrapStrategy("centroid", 0.1, None, np.int32(100), np.int64(0), np.int64(3)),
        FOVBootstrapStrategy("subspace", 0.1, None, 100, 0, 3, None, np.int64(1)),
    ]
    for strategy in strategies:
        text = json.dumps(strategy.parameters)  # provenance is JSON: no numpy integer left
        assert "int64" not in text
        for estimator in strategy.estimators:
            json.dumps(estimator.parameters)
    assert type(strategies[0].window) is int and type(strategies[2].min_fovs) is int
    with pytest.raises(ValueError, match="random_state"):
        DescriptorSpreadStrategy("area", 0.1, "fov", 100, -1)
    with pytest.raises(TypeError, match="n_resamples"):
        DescriptorSpreadStrategy("area", 0.1, "fov", True, 0)


def test_c14_marcotti_rules_answer_population_and_distribution_with_two_quantities():
    rng = np.random.default_rng(1)
    objects, fovs = _acquisition([list(np.abs(rng.normal(10, 2, 15))) for _ in range(12)])
    rules = [
        DescriptorPlateauStrategy("area", 5.0, 2, "acq"),
        DescriptorSpreadStrategy("area", 5.0, "fov", 50, 0),
        DescriptorSpreadStrategy("area", 5.0, "object", 50, 0),
    ]
    for objective in (POPULATION, DISTRIBUTION, PHENOTYPE):
        for result in _area_run(objects, fovs, rules, objective=objective):
            assert result.decision in ("stop", "continue"), (objective, result.strategy)
    [plateau, spread, _] = _area_run(objects, fovs, rules)
    for result, measure in ((plateau, "range"), (spread, "spread")):
        quantities = result.evidence.quantities
        assert quantities[f"max_location_{measure}"].component == "population"
        assert quantities[f"max_scale_{measure}"].component == "distribution"
        # The earlier quantity keeps its meaning: the largest over every descriptor.
        combined = quantities[f"max_descriptor_{measure}"]
        assert combined.component == "distribution"
        assert combined.value == max(
            quantities[f"max_location_{measure}"].value, quantities[f"max_scale_{measure}"].value
        )
    # The rule's own verdict is unchanged: stop when every descriptor is within tolerance.
    for tolerance in (0.01, 0.2, 0.5, 1.0, 5.0):
        rule = DescriptorSpreadStrategy("area", tolerance, "fov", 50, 0)
        [result] = _area_run(objects, fovs, [rule])
        expected = result.evidence.value("max_descriptor_spread") <= tolerance
        assert (result.rule_decision == "stop") == expected
    # Location descriptors alone answer the population summary only, as before.
    location = DescriptorSpreadStrategy("area", 5.0, "fov", 50, 0, ("mean", "median"))
    [result] = _area_run(objects, fovs, [location], objective=DISTRIBUTION)
    assert result.decision == "unsupported"
    assert set(result.evidence.quantities) == {"max_descriptor_spread", "max_location_spread"}


def test_c16_the_plateau_window_counts_fovs_with_objects_and_missing_values_are_stated():
    rng = np.random.default_rng(1)
    values = [list(rng.normal(10, 2, 8)) if j % 2 == 0 else [] for j in range(10)]
    objects, fovs = _acquisition(values)
    [result] = _area_run(objects, fovs, [DescriptorPlateauStrategy("area", 100.0, 2, "acq")])
    assert result.evidence.diagnostics["fovs_in_window"] == [5, 7, 9]
    assert "``window`` counts FOVs that added objects" in _words(
        stabilization.DescriptorPlateauStrategy.__doc__
    )
    objects.loc[objects.index[3], "area"] = np.nan
    rules = [
        DescriptorPlateauStrategy("area", 1.0, 2, "acq"),
        DescriptorSpreadStrategy("area", 1.0, "fov", 20, 0),
    ]
    for result, estimator in zip(
        _area_run(objects, fovs, rules), (r.estimators[0] for r in rules), strict=True
    ):
        assert all(q.availability == "not_applicable" for q in result.evidence.quantities.values())
        assert "Missing values in the feature" in _words(type(estimator).__doc__)


def test_p4_1_a_readout_constant_so_far_is_not_enough_data_for_the_jcs_rules_and_bootstrap():
    rng = np.random.default_rng(1)
    saturated = [[4095.0] * 12 for _ in range(5)]
    varied = [list(np.minimum(4095.0, rng.normal(3000, 600, 12))) for _ in range(3)]
    objects, fovs = _ordered_acquisition(saturated + varied)
    rules = [
        DescriptorSpreadStrategy("x", 50.0, "object", 50, 0),
        DescriptorSpreadStrategy("x", 50.0, "fov", 50, 0),
        DescriptorPlateauStrategy("x", 50.0, 2, "acq"),
        FOVBootstrapStrategy("centroid", 50.0, None, 100, 0, 2, features=("x",)),
        FOVBootstrapStrategy("quantiles", 50.0, None, 100, 0, 2, (0.5,), features=("x",)),
    ]
    results = _x_replay(objects, fovs, rules)
    for item in results:
        if item.step <= 5:
            assert item.result.rule_decision == "continue", (item.step, item.result.strategy)
            for quantity in item.result.evidence.quantities.values():
                assert quantity.availability == "insufficient_data"
    constant = {
        r.result.strategy: _quantity(r.result, next(iter(r.result.evidence.quantities))).reason
        for r in results
        if r.step == 5
    }
    assert constant == {
        "descriptor_spread_object": "no spread among the objects so far",
        "descriptor_spread_fov": "no spread among the objects so far",
        "descriptor_plateau": "no spread among the objects so far",
        "fov_bootstrap_centroid": "no spread between the FOV means of x so far",
        "fov_bootstrap_quantiles": "no spread among the values of x so far",
    }
    # Once a field differs every rule has a value again.
    last = [r for r in results if r.step == len(fovs)]
    assert all(
        q.availability == "available" for r in last for q in r.result.evidence.quantities.values()
    )


def test_p4_5_a_plateau_stop_is_worded_as_change_over_the_window():
    rng = np.random.default_rng(3)
    objects, fovs = _ordered_acquisition([list(rng.normal(10, 1, 30)) for _ in range(6)])
    rule = DescriptorPlateauStrategy("x", 5.0, 2, "acq")
    [result] = _x_run(objects, fovs, [rule])
    assert result.rule_decision == "stop"
    caution = result.evidence.diagnostics["warning"]
    assert caution.startswith(
        "the tolerance bounds how much the descriptors changed over the last 2 FOVs with "
        "objects, not their error against the well"
    )
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["x"],
        strategies=[rule],
        objective=POPULATION,
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="acq",
        fov_col="position_id",
    )
    answer = report.enough["answer"].iloc[0]
    assert "yes, its threshold was met by the 6 acquired FOVs; " in answer
    # A standing assumption (#139): stated in the answer, never as a caution.
    assert f"{caution[0].upper()}{caution[1:]}" in answer and "Caution" not in answer
    # Stated once in both summaries (their lines are wrapped, so compare the words).
    for detail in ("short", "full"):
        assert caution in " ".join(report.summary(detail=detail).split())
    assert "not that they are within ``tolerance`` of the well's" in _joined_words(
        DescriptorPlateauStrategy.__doc__
    )
    assert "40% of stops in homogeneous wells" in _joined_words(stabilization.__doc__)


def test_p4_6_the_per_field_spread_needs_five_fields_with_objects():
    rng = np.random.default_rng(4)
    objects, fovs = _ordered_acquisition([list(rng.normal(10, 1, 10)) for _ in range(6)] + [[]])
    results = _x_replay(objects, fovs, [DescriptorSpreadStrategy("x", 1.0, "fov", 50, 0)])
    reasons = {r.step: _quantity(r.result, "max_location_spread").reason for r in results}
    for step in (2, 3, 4):
        assert reasons[step].startswith("fewer than 5 acquired FOVs with objects")
    assert all(reasons[step] is None for step in (5, 6, 7))
    assert stabilization.MIN_SPREAD_FOVS == 5


def test_p9_4_object_row_order_within_fields_does_not_change_the_live_stop():
    objects, fovs = _well(6, n_fovs=10, per_fov=15)
    rule = DescriptorSpreadStrategy("x", 0.6, "object", 100, 0)
    base = _x_replay(objects, fovs, [rule])
    reordered = (
        objects.assign(_key=np.random.default_rng(0).random(len(objects)))
        .sort_values(["position_id", "_key"])
        .drop(columns="_key")
    )
    again = _x_replay(reordered, fovs, [rule])
    assert [r.result.evidence.quantities for r in base] == [
        r.result.evidence.quantities for r in again
    ]
    assert [r.result.rule_decision for r in base] == [r.result.rule_decision for r in again]


def test_the_resample_helpers_refuse_inputs_they_cannot_resample_532():
    # Their callers guard these cases; the helpers now say so themselves instead of a
    # ZeroDivisionError, draws over no FOV or an index beyond the FOVs.
    rng = np.random.default_rng(0)
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="no value"):
        _descriptor_resamples._object_resamples(np.array([]), 5, rng, ("mean",))
    with pytest.raises(ValueError, match="n_resamples"):
        _descriptor_resamples._object_resamples(values, 0, rng, ("mean",))
    fov = _descriptor_resamples._fov_resamples
    for positions, k, n, match in (
        (np.array([0, 0, 1]), 0, 5, "k"),
        (np.array([0, 1, 2]), 2, 5, "positions"),
        (np.array([0, -1, 1]), 2, 5, "positions"),
        (np.array([0, 1]), 2, 5, "positions"),
        (np.array([0, 1, 1]), 2, 0, "n_resamples"),
    ):
        with pytest.raises(ValueError, match=match):
            fov(values, positions, k, n, rng, ("mean",))
    assert fov(np.array([]), np.array([], dtype=int), 3, 4, rng, ("mean",))["mean"].shape == (4,)
