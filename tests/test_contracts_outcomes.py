"""Outcome-metric contracts: finite, deterministic, input-preserving and strategy-independent.
"""


import math

import pandas as pd
import pytest
from contract_registry import (
    METRIC_IDS,
    METRICS_UNDER_CONTRACT,
    POPULATION,
    SCENARIO_IDS,
)
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    outcomes,
    validate_input_tables,
)
from morphocoverage.benchmark import BenchmarkDataset, BenchmarkRegistry, run_benchmark
from morphocoverage.evaluator import iter_analysis_groups, objects_in_fovs, restrict_input

# --- 4. Outcome metric contracts -----------------------------------------------


def test_every_package_outcome_metric_is_under_contract():
    exported = {
        obj
        for obj in vars(outcomes).values()
        if isinstance(obj, type)
        and issubclass(obj, outcomes.OutcomeMetric)
        and obj is not outcomes.OutcomeMetric
    }
    covered = {type(metric) for metric in METRICS_UNDER_CONTRACT}
    missing = sorted(cls.__name__ for cls in exported - covered)
    assert not missing, f"outcome metrics missing from METRICS_UNDER_CONTRACT: {missing}"


def _group_prefixes(scenario):
    data = validate_input_tables(
        scenario.objects, fovs=scenario.fovs, feature_cols=list(FEATURE_COLS)
    )
    for _, full in iter_analysis_groups(data):
        ordered = full.fovs.sort_values(ORDER_COL)
        for k in sorted({1, 2, len(ordered) // 2, len(ordered)}):
            fovs = ordered.iloc[:k]
            yield full, restrict_input(full, objects_in_fovs(full, fovs), fovs)


@pytest.mark.parametrize("metric", METRICS_UNDER_CONTRACT, ids=METRIC_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_outcome_metrics_are_finite_deterministic_and_leave_inputs_alone(name, metric):
    # Outcomes are compared across strategies, so a NaN or an input edited in
    # place would corrupt every later comparison.
    for full, stopped in _group_prefixes(SCENARIOS[name]()):
        before = (full.objects.copy(), stopped.objects.copy())
        value = metric.evaluate(full, stopped)
        assert value is None or (isinstance(value, float) and math.isfinite(value))
        assert metric.evaluate(full, stopped) == value
        pd.testing.assert_frame_equal(full.objects, before[0])
        pd.testing.assert_frame_equal(stopped.objects, before[1])


def test_outcomes_depend_on_the_stop_not_on_the_strategy():
    # Two different rules that stop at the same prefix get identical outcomes.
    scenario = SCENARIOS["homogeneous"]()
    dataset = BenchmarkDataset(
        "homogeneous",
        "development",
        lambda: (scenario.objects, scenario.fovs),
        FEATURE_COLS,
        ORDER_COL,
    )
    per_fov = scenario.objects.groupby("image_file").size().iloc[0]
    by_fovs, by_objects = run_benchmark(
        BenchmarkRegistry([dataset]),
        "homogeneous",
        strategies=[FixedFOVStrategy(4), FixedObjectStrategy(int(4 * per_fov))],
        metrics=METRICS_UNDER_CONTRACT[:2],
        objective=POPULATION,
    )
    assert by_fovs.stop_step == by_objects.stop_step == 4
    assert by_fovs.config_id != by_objects.config_id
    assert by_fovs.outcomes == by_objects.outcomes


