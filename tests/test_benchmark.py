"""Tests for the benchmark protocol (roles, orders, records) and the outcome metrics."""

import math

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    ScalarPrecisionStrategy,
    validate_input_tables,
)
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    DatasetRole,
    leave_one_dataset_out,
    records_to_frame,
    run_benchmark,
)
from morphocoverage.evaluator import objects_in_fovs, restrict_input
from morphocoverage.outcomes import (
    RandomCellRelativeError,
    RareStateDetected,
    RelativeMeanError,
    srs_relative_rmse,
)

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _dataset(name, role="development", scenario="homogeneous", **kwargs):
    built = SCENARIOS[scenario]()
    return BenchmarkDataset(
        name=name,
        role=role,
        load=lambda: (built.objects, built.fovs),
        feature_cols=FEATURE_COLS,
        acquisition_order_col=ORDER_COL,
        **kwargs,
    )


def _registry(*datasets):
    return BenchmarkRegistry(datasets or [_dataset("dev"), _dataset("test", "held_out")])


def _run(registry, name="dev", strategies=None, metrics=(), **kwargs):
    return run_benchmark(
        registry,
        name,
        strategies=list(strategies or [FixedFOVStrategy(5)]),
        metrics=list(metrics),
        objective=kwargs.pop("objective", POPULATION),
        **kwargs,
    )


# --- Dataset roles --------------------------------------------------------------


def test_registry_keeps_roles_and_rejects_duplicates():
    registry = _registry()
    assert registry.names() == ["dev", "test"]
    assert registry.names("held_out") == ["test"]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_dataset("dev"))
    with pytest.raises(KeyError, match="not registered"):
        registry["missing"]
    with pytest.raises(ValueError):
        _dataset("x", role="training")


def test_tuning_on_held_out_data_is_rejected():
    registry = _registry()
    with pytest.raises(ValueError, match="held out"):
        _run(registry, "test", tuned_on=["test"])
    with pytest.raises(ValueError, match="held out"):
        _run(registry, "dev", tuned_on=["test"])
    with pytest.raises(TypeError, match="not a string"):
        _run(registry, "dev", tuned_on="dev")


def test_runs_record_whether_they_are_in_sample():
    registry = _registry()
    [in_sample] = _run(registry, "dev", tuned_on=["dev"])
    [out_of_sample] = _run(registry, "test", tuned_on=["dev"])
    [untuned] = _run(registry, "test")
    assert (in_sample.in_sample, out_of_sample.in_sample, untuned.in_sample) == (
        True,
        False,
        False,
    )
    assert (out_of_sample.role, out_of_sample.tuned_on) == (DatasetRole.HELD_OUT, ("dev",))


def test_leave_one_dataset_out_uses_development_data_only():
    registry = _registry(_dataset("a"), _dataset("b"), _dataset("c"), _dataset("h", "held_out"))
    assert list(leave_one_dataset_out(registry)) == [
        ("a", ("b", "c")),
        ("b", ("a", "c")),
        ("c", ("a", "b")),
    ]


# --- Stops, effort and orders ---------------------------------------------------


def test_stop_effort_matches_the_first_stop_of_each_group():
    registry = _registry(_dataset("groups", scenario="condition_repeat_groups"))
    records = _run(registry, "groups", strategies=[FixedFOVStrategy(3)])
    scenario = SCENARIOS["condition_repeat_groups"]()
    assert len(records) == scenario.truth["n_groups"]
    for record in records:
        assert (record.stopped, record.stop_step, record.n_fovs) == (True, 3, 3)
        assert record.final_decision == "stop"
        assert record.total_fovs == scenario.truth["fovs_per_group"]
        group_fovs = scenario.fovs[
            scenario.fovs["condition"].eq(record.group["condition"])
            & scenario.fovs["repeat"].eq(record.group["repeat"])
        ]
        first = set(group_fovs.nsmallest(3, ORDER_COL)["image_file"])
        objects = scenario.objects
        expected = objects["condition"].eq(record.group["condition"]) & objects["repeat"].eq(
            record.group["repeat"]
        ) & objects["image_file"].isin(first)
        assert record.n_objects == expected.sum()


def test_zero_object_fovs_count_as_effort_at_the_stop():
    registry = _registry(_dataset("empty", scenario="zero_object_fovs"))
    [record] = _run(registry, "empty", strategies=[FixedObjectStrategy(25)])
    # FOV 1 is empty, FOV 2 has 25 objects.
    assert (record.n_fovs, record.n_objects) == (2, 25)


def test_imaged_area_at_the_stop_is_recorded():
    registry = _registry(_dataset("area", scenario="unequal_fov_area", area_col="area_um2"))
    [record] = _run(registry, "area", strategies=[FixedFOVStrategy(3)])
    assert record.area == 10_000 + 40_000 + 10_000
    assert record.total_area == 10 * 10_000 + 10 * 40_000


def test_area_column_must_not_be_a_feature():
    dataset = BenchmarkDataset(
        "area",
        "development",
        lambda: (SCENARIOS["homogeneous"]().objects, SCENARIOS["homogeneous"]().fovs),
        feature_cols=FEATURE_COLS,
        acquisition_order_col=ORDER_COL,
        area_col="feature_a",
    )
    with pytest.raises(ValueError, match="not a phenotype feature"):
        _run(BenchmarkRegistry([dataset]), "area")


def test_a_run_that_never_stops_is_recorded_at_the_full_acquisition():
    registry = _registry()
    [record] = _run(
        registry, strategies=[FixedFOVStrategy(10_000)], metrics=[RelativeMeanError("feature_a")]
    )
    assert (record.stopped, record.stop_step, record.final_decision) == (False, None, "continue")
    assert (record.n_fovs, record.n_objects) == (record.total_fovs, record.total_objects)
    assert record.outcomes["relative_mean_error:feature_a"] == 0.0


def test_unsupported_objectives_never_count_as_stops():
    rare = ObjectiveConfig(
        SamplingObjective.RARE_DISCOVERY,
        rare=RareTarget(
            object_prevalence=0.01,
            detection_probability=0.9,
            min_fov_occurrences=2,
            phenotype="true_state",
        ),
    )
    [record] = _run(_registry(), strategies=[FixedFOVStrategy(2)], objective=rare)
    assert (record.stopped, record.final_decision) == (False, "unsupported")


def test_permuted_orders_are_reproducible_and_differ_from_the_recorded_order():
    registry = _registry(_dataset("rare", scenario="rare_clustered"))
    metric = RareStateDetected("true_state", 1)
    strategy = FixedFOVStrategy(10)

    def detections(order):
        [record] = _run(registry, "rare", [strategy], [metric], orders=[order])
        return record.outcomes[metric.name]

    # Recorded order puts the rare FOVs at positions 31-32: 10 FOVs never see them.
    assert detections(AcquisitionOrder.recorded()) == 0.0
    seeds = range(20)
    permuted = [detections(AcquisitionOrder.permuted(seed)) for seed in seeds]
    assert permuted == [detections(AcquisitionOrder.permuted(seed)) for seed in seeds]
    # P(at least one of 2 rare FOVs among 10 of 40) = 1 - C(38,10)/C(40,10) = 0.44.
    assert 0 < sum(permuted) < len(permuted)


def test_permutation_is_by_whole_fov_and_within_groups():
    scenario = SCENARIOS["condition_repeat_groups"]()
    ordered, column = AcquisitionOrder.permuted(3).apply(scenario.fovs, ORDER_COL)
    assert sorted(ordered[column]) == list(range(len(scenario.fovs)))
    assert column != ORDER_COL and ORDER_COL in ordered
    with pytest.raises(ValueError, match="already has"):
        AcquisitionOrder.permuted(3).apply(ordered, ORDER_COL)
    with pytest.raises(TypeError):
        AcquisitionOrder.permuted(None)
    with pytest.raises(ValueError):
        AcquisitionOrder("recorded", 1)


def test_records_carry_configuration_identity_and_flatten_to_rows():
    strategies = [
        ScalarPrecisionStrategy("feature_a", 0.05, "object", "point"),
        ScalarPrecisionStrategy("feature_a", 0.02, "object", "point"),
    ]
    metrics = [RelativeMeanError("feature_a"), RandomCellRelativeError("feature_a")]
    records = _run(_registry(), strategies=strategies, metrics=metrics)
    assert [r.config_id for r in records] == [s.config_id for s in strategies]
    assert records[0].strategy == records[1].strategy  # same name, distinct ids
    assert records[0].estimator_ids == records[1].estimator_ids
    assert records[0].policy_ids != records[1].policy_ids
    assert records[0].outcome_ids == {m.name: m.config_id for m in metrics}
    frame = records_to_frame(records)
    assert len(frame) == 2
    assert {"group.condition", "group.repeat", "outcome.relative_mean_error:feature_a"} <= set(
        frame.columns
    )
    assert (frame["compute_seconds_to_stop"] >= 0).all()


def test_metric_names_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        _run(_registry(), metrics=[RelativeMeanError("feature_a")] * 2)


# --- Outcome metrics ------------------------------------------------------------


def _prefix(scenario, n_fovs):
    data = validate_input_tables(
        scenario.objects, fovs=scenario.fovs, feature_cols=list(FEATURE_COLS)
    )
    fovs = data.fovs.nsmallest(n_fovs, ORDER_COL)
    return data, restrict_input(data, objects_in_fovs(data, fovs), fovs)


def test_relative_mean_error_matches_hand_calculation():
    scenario = SCENARIOS["neighbour_correlation"]()
    full, stopped = _prefix(scenario, 5)
    in_first = scenario.objects["image_file"].isin(
        scenario.fovs.nsmallest(5, ORDER_COL)["image_file"]
    )
    reference = scenario.objects["feature_a"].mean()
    expected = abs(scenario.objects.loc[in_first, "feature_a"].mean() - reference) / reference
    assert RelativeMeanError("feature_a").evaluate(full, stopped) == pytest.approx(expected)


def test_relative_mean_error_is_undefined_without_objects_at_the_stop():
    full, stopped = _prefix(SCENARIOS["zero_object_fovs"](), 1)  # first FOV is empty
    assert RelativeMeanError("feature_a").evaluate(full, stopped) is None
    assert RandomCellRelativeError("feature_a").evaluate(full, stopped) is None


def test_random_cell_error_is_the_finite_population_srs_error():
    scenario = SCENARIOS["homogeneous"]()
    full, stopped = _prefix(scenario, 4)
    values = scenario.objects["feature_a"]
    n, total = len(stopped.objects), len(values)
    expected = math.sqrt(values.var() / n * (1 - n / total)) / values.mean()
    assert RandomCellRelativeError("feature_a").evaluate(full, stopped) == pytest.approx(expected)
    assert srs_relative_rmse(values.var(), values.mean(), total, total) == 0.0


def test_srs_error_matches_a_simulation():
    rng = np.random.default_rng(0)
    population = rng.gamma(4.0, 1.0, 500)
    draws = [rng.choice(population, 50, replace=False).mean() for _ in range(20_000)]
    simulated = np.sqrt(np.mean((np.asarray(draws) - population.mean()) ** 2)) / population.mean()
    expected = srs_relative_rmse(population.var(ddof=1), population.mean(), 50, 500)
    assert simulated == pytest.approx(expected, rel=0.03)


def test_rare_state_detection_needs_the_required_number_of_fovs():
    scenario = SCENARIOS["rare_clustered"]()  # rare state in FOVs 31 and 32 only
    full, early = _prefix(scenario, 31)
    _, late = _prefix(scenario, 32)
    assert RareStateDetected("true_state", 1).evaluate(full, early) == 1.0
    assert RareStateDetected("true_state", 2).evaluate(full, early) == 0.0
    assert RareStateDetected("true_state", 2).evaluate(full, late) == 1.0
    # The full acquisition never reaches 3 FOVs: no stop could detect it.
    assert RareStateDetected("true_state", 3).evaluate(full, late) is None


def test_rare_state_label_must_be_boolean_or_binary():
    scenario = SCENARIOS["rare_clustered"]()
    objects = scenario.objects.assign(true_state=scenario.objects["true_state"] * 2)
    data = validate_input_tables(objects, fovs=scenario.fovs, feature_cols=list(FEATURE_COLS))
    with pytest.raises(ValueError, match="booleans or 0/1"):
        RareStateDetected("true_state", 1).evaluate(data, data)
    with pytest.raises(ValueError, match="not a column"):
        RareStateDetected("missing", 1).evaluate(data, data)


def test_outcome_features_must_be_explicit():
    full, stopped = _prefix(SCENARIOS["homogeneous"](), 3)
    with pytest.raises(ValueError, match="explicitly selected"):
        RelativeMeanError("stage_x").evaluate(full, stopped)


def test_metric_config_ids_distinguish_parameters():
    assert RelativeMeanError("a").config_id != RelativeMeanError("b").config_id
    assert RelativeMeanError("a").config_id != RandomCellRelativeError("a").config_id
    assert (
        RareStateDetected("x", 1).config_id != RareStateDetected("x", 2).config_id
    )


def test_benchmark_input_tables_are_not_modified():
    scenario = SCENARIOS["homogeneous"]()
    objects, fovs = scenario.objects.copy(), scenario.fovs.copy()
    dataset = BenchmarkDataset(
        "d", "development", lambda: (objects, fovs), FEATURE_COLS, ORDER_COL
    )
    _run(
        BenchmarkRegistry([dataset]),
        "d",
        metrics=[RelativeMeanError("feature_a")],
        orders=[AcquisitionOrder.recorded(), AcquisitionOrder.permuted(0)],
    )
    pd.testing.assert_frame_equal(objects, scenario.objects)
    pd.testing.assert_frame_equal(fovs, scenario.fovs)
