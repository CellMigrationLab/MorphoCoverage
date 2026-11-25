"""Tests for the benchmark protocol (roles, orders, records) and the outcome metrics."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _helpers_benchmark import MEAN_F, _two_groups
from _helpers_retrospective import RARE, _f_reliability, _f_replay, _tables
from _helpers_sampling_unit import _orders
from _helpers_strategies import _report_both_orders
from _helpers_tables import _four_cell_tables
from _helpers_workflows import READOUT, _acquisition
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    ScalarPrecisionStrategy,
    benchmark,
    calibration,
    online,
    reservoir,
    sequential,
    validate_input_tables,
)
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    DatasetRole,
    _area,
    _record_run,
    leave_one_dataset_out,
    permuted_fovs,
    records_to_frame,
    run_benchmark,
)
from morphocoverage.evaluator import iter_analysis_groups, objects_in_fovs, restrict_input
from morphocoverage.outcomes import (
    RandomObjectRelativeError,
    RareStateDetected,
    RelativeMeanError,
    srs_relative_rmse,
)
from morphocoverage.reservoir import ReservoirSplit
from morphocoverage.results import Timing
from morphocoverage.retrospective import (
    OutcomeCriterion,
    adequacy_points,
    order_efficiency,
    replay_acquisitions,
)
from morphocoverage.sequential import evaluate_acquisition_prefixes
from morphocoverage.strategies import FOVBootstrapEstimator, ImagedAreaEstimator
from morphocoverage.workflows import default_families

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
            & scenario.fovs["biological_repeat"].eq(record.group["biological_repeat"])
        ]
        first = set(group_fovs.nsmallest(3, ORDER_COL)["image_file"])
        objects = scenario.objects
        expected = (
            objects["condition"].eq(record.group["condition"])
            & objects["biological_repeat"].eq(record.group["biological_repeat"])
            & objects["image_file"].isin(first)
        )
        assert record.n_observations == expected.sum()


def test_zero_object_fovs_count_as_effort_at_the_stop():
    registry = _registry(_dataset("empty", scenario="zero_object_fovs"))
    [record] = _run(registry, "empty", strategies=[FixedObjectStrategy(25)])
    # FOV 1 is empty, FOV 2 has 25 objects.
    assert (record.n_fovs, record.n_observations) == (2, 25)


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
    assert (record.n_fovs, record.n_observations) == (record.total_fovs, record.total_observations)
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
    data = validate_input_tables(
        scenario.objects, fovs=scenario.fovs, feature_cols=list(FEATURE_COLS)
    )
    groups = list(iter_analysis_groups(data))
    ordered, column = AcquisitionOrder.permuted(3).apply(groups, ORDER_COL)
    assert len(ordered) == len(scenario.fovs)
    # Ranks are drawn within each group: every group holds ranks 0 .. n - 1.
    for group, group_data in groups:
        ranks = ordered.loc[(ordered[list(group)] == pd.Series(group)).all(axis=1), column]
        assert sorted(ranks) == list(range(len(group_data.fovs)))
    assert column != ORDER_COL and ORDER_COL in ordered
    marked = [(group, replace(g, fovs=g.fovs.assign(**{column: 0}))) for group, g in groups]
    with pytest.raises(ValueError, match="already has"):
        AcquisitionOrder.permuted(3).apply(marked, ORDER_COL)
    with pytest.raises(TypeError):
        AcquisitionOrder.permuted(None)
    with pytest.raises(ValueError):
        AcquisitionOrder("recorded", 1)


def test_records_carry_configuration_identity_and_flatten_to_rows():
    strategies = [
        ScalarPrecisionStrategy("feature_a", 0.05, "object", "point"),
        ScalarPrecisionStrategy("feature_a", 0.02, "object", "point"),
    ]
    metrics = [RelativeMeanError("feature_a"), RandomObjectRelativeError("feature_a")]
    records = _run(_registry(), strategies=strategies, metrics=metrics)
    assert [r.config_id for r in records] == [s.config_id for s in strategies]
    assert records[0].strategy == records[1].strategy  # same name, distinct ids
    assert records[0].estimator_ids == records[1].estimator_ids
    assert records[0].policy_ids != records[1].policy_ids
    assert records[0].outcome_ids == {m.name: m.config_id for m in metrics}
    frame = records_to_frame(records)
    assert len(frame) == 2
    assert {
        "group.condition",
        "group.biological_repeat",
        "outcome.relative_mean_error:feature_a",
    } <= set(frame.columns)
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
    assert RandomObjectRelativeError("feature_a").evaluate(full, stopped) is None


def test_random_cell_error_is_the_finite_population_srs_error():
    scenario = SCENARIOS["homogeneous"]()
    full, stopped = _prefix(scenario, 4)
    values = scenario.objects["feature_a"]
    n, total = len(stopped.objects), len(values)
    expected = math.sqrt(values.var() / n * (1 - n / total)) / values.mean()
    assert RandomObjectRelativeError("feature_a").evaluate(full, stopped) == pytest.approx(expected)
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
    assert RelativeMeanError("a").config_id != RandomObjectRelativeError("a").config_id
    assert RareStateDetected("x", 1).config_id != RareStateDetected("x", 2).config_id


def test_benchmark_input_tables_are_not_modified():
    scenario = SCENARIOS["homogeneous"]()
    objects, fovs = scenario.objects.copy(), scenario.fovs.copy()
    dataset = BenchmarkDataset("d", "development", lambda: (objects, fovs), FEATURE_COLS, ORDER_COL)
    _run(
        BenchmarkRegistry([dataset]),
        "d",
        metrics=[RelativeMeanError("feature_a")],
        orders=[AcquisitionOrder.recorded(), AcquisitionOrder.permuted(0)],
    )
    pd.testing.assert_frame_equal(objects, scenario.objects)
    pd.testing.assert_frame_equal(fovs, scenario.fovs)


def test_incremental_benchmark_records_the_same_stops_and_latencies():
    strategies = [ScalarPrecisionStrategy("feature_a", 0.01, "fov", "point"), FixedFOVStrategy(4)]
    metrics = [RelativeMeanError("feature_a")]
    full = _run(_registry(), strategies=strategies, metrics=metrics)
    incremental = _run(_registry(), strategies=strategies, metrics=metrics, incremental=True)
    for before, after in zip(full, incremental, strict=True):
        assert (before.stop_step, before.n_observations) == (after.stop_step, after.n_observations)
        assert before.outcomes == after.outcomes
        assert (before.incremental, after.incremental) == (False, True)
        assert 0 <= after.median_decision_seconds <= after.max_decision_seconds
        assert after.decision_seconds_to_stop >= after.compute_seconds_to_stop


def test_latency_summaries_stop_at_the_first_stop():
    scenario = SCENARIOS["homogeneous"]()
    dataset = _dataset("timing", scenario="homogeneous")
    history = evaluate_acquisition_prefixes(
        scenario.objects,
        scenario.fovs,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
    )
    timed = []
    for index, item in enumerate(history, start=1):
        seconds = float(index)
        result = replace(
            item.result,
            compute_seconds=seconds,
            timing=Timing(seconds, 0.0, 0.0, 0.0),
        )
        timed.append(replace(item, result=result))

    data = validate_input_tables(
        scenario.objects,
        fovs=scenario.fovs,
        feature_cols=list(FEATURE_COLS),
    )
    _, full = next(iter(iter_analysis_groups(data)))
    record = _record_run(
        timed,
        full,
        full,
        [],
        dataset,
        in_sample=False,
        tuned_on=(),
        order=AcquisitionOrder.recorded(),
        incremental=False,
    )

    assert record.stop_step == 2
    assert record.compute_seconds_to_stop == 3.0
    assert record.max_compute_seconds == 2.0
    assert record.decision_seconds_to_stop == 3.0
    assert record.median_decision_seconds == 1.5
    assert record.max_decision_seconds == 2.0


def test_benchmark_records_generic_observation_effort():
    dataset = _dataset("slices", observation_unit="slice")
    [record] = _run(BenchmarkRegistry([dataset]), "slices", strategies=[FixedFOVStrategy(2)])
    assert record.observation_unit == "slice"
    row = record.as_row()
    assert row["n_observations"] == record.n_observations
    assert row["total_observations"] == record.total_observations
    # The owner removed the duplicate columns (2026-11-10).
    assert not {"n_objects", "total_objects"} & row.keys()


def test_outcome_metrics_do_not_silently_drop_missing_feature_values():
    scenario = SCENARIOS["homogeneous"]()
    objects = scenario.objects.copy()
    objects.loc[objects.index[0], "feature_a"] = float("nan")
    data = validate_input_tables(
        objects,
        fovs=scenario.fovs,
        feature_cols=list(FEATURE_COLS),
    )
    fovs = data.fovs.nsmallest(3, ORDER_COL)
    stopped = restrict_input(data, objects_in_fovs(data, fovs), fovs)

    assert RelativeMeanError("feature_a").evaluate(data, stopped) is None
    assert RandomObjectRelativeError("feature_a").evaluate(data, stopped) is None


def test_benchmark_completes_when_a_rule_does_not_fit_the_rows():
    dataset = _dataset("slices", observation_unit="Slices")
    assert dataset.observation_unit == "slice"
    records = _run(
        BenchmarkRegistry([dataset]),
        "slices",
        strategies=[
            ScalarPrecisionStrategy("feature_a", 0.05, "object", "point"),
            FixedObjectStrategy(10),
            FixedFOVStrategy(2),
        ],
    )
    by_name = {record.strategy: record for record in records}
    for name in ("scalar_precision_object", "fixed_objects"):
        assert (by_name[name].stopped, by_name[name].final_decision) == (False, None)
    assert by_name["fixed_fov"].stop_step == 2


def test_benchmark_records_whether_the_observation_unit_was_declared():
    assumed = _run(_registry(_dataset("d")), "d")[0]
    declared = _run(_registry(_dataset("d", observation_unit="cell")), "d")[0]
    assert (assumed.observation_unit, assumed.observation_unit_declared) == ("object", False)
    assert (declared.observation_unit, declared.observation_unit_declared) == ("cell", True)
    assert records_to_frame([assumed])["observation_unit_declared"].tolist() == [False]


def test_worker_processes_give_identical_records_apart_from_latencies():
    registry = _registry(_dataset("dev", scenario="condition_repeat_groups"))
    settings = {
        "strategies": [FixedFOVStrategy(3), FixedFOVStrategy(6)],
        "metrics": [RelativeMeanError("feature_a")],
    }

    def rows(n_jobs):
        table = records_to_frame(_run(registry, n_jobs=n_jobs, **settings))
        return table.drop(columns=[c for c in table.columns if "seconds" in c])

    pd.testing.assert_frame_equal(rows(1), rows(2))


def test_a_missing_field_area_is_refused_not_counted_as_zero():
    scenario = SCENARIOS["unequal_fov_area"]()
    fovs = scenario.fovs.copy()
    fovs.loc[fovs.index[1], "area_um2"] = float("nan")
    dataset = BenchmarkDataset(
        "area",
        "development",
        lambda: (scenario.objects, fovs),
        feature_cols=FEATURE_COLS,
        acquisition_order_col=ORDER_COL,
        area_col="area_um2",
    )
    with pytest.raises(ValueError, match="missing for some fields"):
        _run(BenchmarkRegistry([dataset]), "area", strategies=[FixedFOVStrategy(3)])


def _one_dataset_registry(objects, fovs, **dataset):
    return BenchmarkRegistry(
        [BenchmarkDataset("d", "development", lambda: (objects, fovs), ("f",), "order", **dataset)]
    )


def _permuted_error(objects, fovs, seed=5):
    records = run_benchmark(
        _one_dataset_registry(objects, fovs, fov_col="position_id", group_cols=("condition",)),
        "d",
        strategies=[FixedFOVStrategy(3)],
        metrics=[RelativeMeanError("f")],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(seed)],
    )
    return {r.group["condition"]: r.outcomes["relative_mean_error:f"] for r in records}


def test_b4_permuted_orders_are_per_group_and_ignore_row_order():
    objects, fovs = _two_groups()
    both = _permuted_error(objects, fovs)["a"]
    alone = _permuted_error(objects[objects.condition == "a"], fovs[fovs.condition == "a"])["a"]
    shuffled = _permuted_error(objects, fovs.sample(frac=1, random_state=3))["a"]
    assert both == alone == shuffled
    # The replay draws the same permutation function, seeded from random_state.
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    group, group_data = next(iter(iter_analysis_groups(data)))
    order = permuted_fovs(group_data.fovs, group_data.fov_key, 0, group, 1)
    reversed_rows = group_data.fovs.iloc[::-1]
    again = permuted_fovs(reversed_rows, group_data.fov_key, 0, group, 1)
    assert order.position_id.tolist() == again.position_id.tolist()
    replays = {
        state: replay_acquisitions(
            objects,
            fovs,
            feature_cols=["f"],
            strategies=[FixedFOVStrategy(3)],
            criteria=[MEAN_F],
            objective=POPULATION,
            reservoir_fraction=0.3,
            n_splits=1,
            orders=[AcquisitionOrder.permuted(0)],
            random_state=state,
        ).outcomes
        for state in (0, 1)
    }
    assert not replays[0]["value"].equals(replays[1]["value"])


@pytest.mark.parametrize(
    "areas",
    [
        [100.0, -100.0, 100.0, 100.0],
        [100.0, 0.0, 100.0, 100.0],
        [100.0, np.inf, 100.0, 100.0],
        [1e308] * 4,  # each finite, their total overflows (#544)
        ["100"] * 4,
    ],
)
def test_b7_benchmark_refuses_unusable_areas(areas):
    fovs = pd.DataFrame({"position_id": [f"p{i}" for i in range(4)], "order": range(4)})
    fovs["area"] = pd.Series(areas, dtype=object if isinstance(areas[0], str) else float)
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 3), "f": np.arange(12.0)})
    with pytest.raises(ValueError, match="area_col"):
        run_benchmark(
            _one_dataset_registry(objects, fovs, area_col="area"),
            "d",
            strategies=[FixedFOVStrategy(2)],
            metrics=[RelativeMeanError("f")],
            objective=POPULATION,
        )


def test_b9_records_keep_stop_on_and_a_consistent_final_decision():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"position_id": [f"p{i}" for i in range(8)], "order": range(8)})
    objects = pd.DataFrame(
        {"position_id": np.repeat(fovs.position_id, 50), "f": rng.uniform(9, 11, 400)}
    )
    phenotype = ObjectiveConfig(SamplingObjective.PHENOTYPE)
    for stop_on in ("decision", "rule"):
        (record,) = run_benchmark(
            _one_dataset_registry(objects, fovs),
            "d",
            strategies=[ScalarPrecisionStrategy("f", 0.05, "object", "point")],
            metrics=[RelativeMeanError("f")],
            objective=phenotype,
            stop_on=stop_on,
        )
        assert record.stop_on == stop_on == record.as_row()["stop_on"]
        assert record.stopped == (record.final_decision.value == "stop")
    assert record.stopped  # the rule's own verdict stops


def test_b11_numpy_group_values_seed_like_python_ones():
    fovs = pd.DataFrame({"biological_repeat": 1, "position_id": [f"p{i:02d}" for i in range(20)]})
    objects = pd.DataFrame(
        {"biological_repeat": 1, "position_id": fovs.position_id, "f": np.arange(20.0)}
    )
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    group_from_table = {"biological_repeat": data.fovs["biological_repeat"].iloc[0]}
    assert isinstance(group_from_table["biological_repeat"], np.integer)
    split = ReservoirSplit(0.3, 0, None)
    held = split.split(group_from_table, data)[1].fovs["position_id"].tolist()
    assert held == split.split({"biological_repeat": 1}, data)[1].fovs["position_id"].tolist()
    order = permuted_fovs(data.fovs, data.fov_key, 0, group_from_table)
    assert order.equals(permuted_fovs(data.fovs, data.fov_key, 0, {"biological_repeat": 1}))


def test_b12_benchmark_and_imaged_area_share_one_area_check():
    built = SCENARIOS["homogeneous"]()
    estimator = ImagedAreaEstimator("area_um2")
    for bad in (np.nan, -1.0, 0.0, np.inf):
        fovs = built.fovs.copy()
        fovs.loc[0, "area_um2"] = bad
        data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
        quantity = estimator.estimate(data, objective=POPULATION).quantities["imaged_area"]
        assert quantity.availability.value == "not_applicable"
        with pytest.raises(ValueError) as refused:
            _area(data, "area_um2")
        assert str(refused.value) == quantity.reason
    fovs = built.fovs.assign(area_um2=built.fovs["area_um2"].astype(str))
    data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    for check in (
        lambda: _area(data, "area_um2"),
        lambda: estimator.estimate(data, objective=POPULATION),
    ):
        with pytest.raises(ValueError, match="not numbers"):
            check()


def test_j16_modules_list_their_public_names():
    assert online.__all__ == ["OnlineSession"]
    assert sequential.__all__ == ["SequentialResult", "evaluate_acquisition_prefixes"]
    assert "run_benchmark" in benchmark.__all__ and "permuted_fovs" not in benchmark.__all__
    for module in (online, sequential, benchmark, calibration, reservoir):
        assert all(not name.startswith("_") and hasattr(module, name) for name in module.__all__)


def test_e4_recorded_and_random_orders_are_summarised_apart():
    report = _report_both_orders("neighbour_correlation", [FixedFOVStrategy(4)])
    stops = report.stops
    assert list(stops["order_kind"]) == ["recorded", "permuted"]
    assert list(stops["orders"]) == [1, 9]
    answer = report.enough["answer"].iloc[0]
    assert "in the recorded order, first met after 4 FOVs" in answer
    assert "in all 9 random orders, first met after 4 FOVs" in answer
    row = report.enough.iloc[0]
    assert row["order_kind"] == "permuted" and row["orders"] == 9
    recorded_only = _report_both_orders(
        "neighbour_correlation", [FixedFOVStrategy(4)], orders=[AcquisitionOrder.recorded()]
    )
    assert recorded_only.enough["order_kind"].iloc[0] == "recorded"


def test_f6_neighbouring_seeds_share_no_reservoir_and_both_analyses_hold_the_same_one():
    objects, fovs = _tables(n_fovs=20)
    kwargs = {"orders": [AcquisitionOrder.recorded()], "acquisition_order_col": "order"}
    first = _f_replay(objects, fovs, random_state=0, n_splits=5, **kwargs).outcomes
    second = _f_replay(objects, fovs, random_state=1, n_splits=5, **kwargs).outcomes
    for split in range(4):
        shifted = first[first["split"] == split + 1]["value"].round(12).tolist()
        assert second[second["split"] == split]["value"].round(12).tolist() != shifted
    # The same random_state and split hold back the same fields in reliability and replay:
    # the whole stream scores the same against the reservoir in both.
    replay = _f_replay(objects, fovs, random_state=4, n_splits=3, **kwargs)
    stream = int(replay.streams["stream_fovs"].iloc[0])
    whole = replay.outcomes[replay.outcomes["fovs"] == stream].set_index("split")["value"]
    draws = _f_reliability(
        objects, fovs, fov_counts=[stream], reservoir_fraction=0.3, random_state=4
    ).draws
    drawn = draws.groupby("split")["value"].first()
    assert np.allclose(drawn.sort_index().to_numpy(), whole.sort_index().to_numpy())


def test_f7_order_efficiency_leaves_unjudged_runs_out():
    objects, fovs = _tables(n_fovs=16, rare_fovs=(3, 9))
    replay = _f_replay(
        objects,
        fovs,
        criteria=[RARE],
        orders=[AcquisitionOrder.permuted(s) for s in range(4)],
        n_splits=6,
        stage_cols=("x", "y"),
    )
    realized = adequacy_points(replay).realized
    unjudged = int((realized[realized["criterion"] == "all criteria"]["judged_fovs"] == 0).sum())
    assert unjudged > 0
    result = order_efficiency(replay)
    row = result.table.iloc[0]
    assert row["unjudged"] == unjudged
    judged = result.runs[result.runs["judged"]]
    assert row["reached"] == pytest.approx(judged["sustained_realized"].notna().mean())
    assert f"{unjudged} of {int(row['runs'])} runs could not be judged" in result.summary()


def test_f13_dispersed_seeds_beyond_the_stream_are_replayed_once():
    objects, fovs = _tables(n_fovs=5)
    replay = _f_replay(
        objects,
        fovs,
        reservoir_fraction=0.4,
        orders=[AcquisitionOrder.dispersed(s) for s in range(5)],
        n_splits=1,
        stage_cols=("x", "y"),
    )
    stream = int(replay.streams["stream_fovs"].iloc[0])
    assert stream == 3 and len(replay.streams) == stream
    firsts = replay.paths[replay.paths["fovs"] == 2].groupby("order_id")["travel"].first()
    assert len(firsts) == stream


def test_h3_the_benchmark_bootstrap_of_the_mean_reads_the_readout_only():
    family = default_families(READOUT, n_resamples=20)[-1]
    assert family.strategy.estimators[0].features == (READOUT,)
    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    everything = validate_input_tables(objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    alone = validate_input_tables(objects, fovs=fovs, feature_cols=[READOUT])
    selected = FOVBootstrapEstimator("centroid", None, 50, 0, 2, features=(READOUT,))
    plain = FOVBootstrapEstimator("centroid", None, 50, 0, 2)
    name = "fov_bootstrap_centroid_max_se"
    assert selected.estimate(everything, objective=POPULATION).value(name) == pytest.approx(
        plain.estimate(alone, objective=POPULATION).value(name)
    )
    # Without features the configuration is the one it always was.
    assert "features" not in plain.parameters


def test_orders_that_agree_within_one_field_are_called_a_null_result():
    built = SCENARIOS["homogeneous"]()
    replay = replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(2)],
        criteria=[
            OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")
        ],
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=[
            AcquisitionOrder.recorded(),
            AcquisitionOrder.permuted(0),
            AcquisitionOrder.dispersed(0),
        ],
        acquisition_order_col=ORDER_COL,
        stage_cols=("stage_x", "stage_y"),
        n_splits=2,
    )
    result = order_efficiency(replay)
    spread = result.table["median_fovs"].max() - result.table["median_fovs"].min()
    text = result.summary()
    assert ("null result" in text) == (spread <= 1 and (result.table["reached"] == 1).all())


def test_run_benchmark_refusals():
    objects, fovs = _four_cell_tables()

    def load():
        return objects.assign(order=range(4)), fovs.assign(order=range(2))

    rule = [FixedFOVStrategy(2)]
    registry = BenchmarkRegistry(
        [BenchmarkDataset("d", "development", load, ("area",), "order", area_col="missing")]
    )
    with pytest.raises(ValueError) as error:
        run_benchmark(registry, "d", strategies=rule, metrics=[], objective=POPULATION, stop_on="x")
    assert str(error.value) == "stop_on must be 'decision' or 'rule', not 'x'"
    with pytest.raises(ValueError) as error:
        run_benchmark(registry, "d", strategies=rule, metrics=[], objective=POPULATION)
    assert str(error.value) == "area_col 'missing' is not a column of the FOV table"
    registry = BenchmarkRegistry([BenchmarkDataset("d", "development", load, ("area",), "order")])
    with pytest.raises(TypeError) as error:
        run_benchmark(
            registry, "d", strategies=rule, metrics=[], objective=POPULATION, reservoir="x"
        )
    assert str(error.value) == "reservoir must be a ReservoirSplit"


def test_the_order_comparison_is_one_table_per_order_kind_and_every_median_in_full():
    result = _orders()
    short = result.summary()
    assert re.search(r"order\s+fields\s+stage travel\s+time\s+never enough", short)
    for name in ("recorded", "random", "spread out"):
        assert re.search(rf"^  {name}\s", short, re.MULTILINE)
    assert short.count("straight-line path") == 1
    full = result.summary(detail="full")
    table = result.table
    for row in table.to_dict("records"):
        if row["median_fovs"] is not None and not np.isnan(row["median_fovs"]):
            fields = f"{row['median_fovs']:g} field"
            assert fields in full
    assert "Analysis groups: condition · biological repeat." in full
    assert "min" in full  # the timing was given
    with pytest.raises(ValueError, match="short"):
        result.summary(detail="brief")
