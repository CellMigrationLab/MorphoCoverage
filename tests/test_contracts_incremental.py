"""Incremental-update and timing contracts: incremental replay equals recomputation."""

import pytest
from contract_registry import (
    POPULATION,
    SCENARIO_IDS,
    STRATEGIES_UNDER_CONTRACT,
    STRATEGY_IDS,
    replay,
)
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    DescriptorPlateauStrategy,
    FixedFOVStrategy,
    SamplingStrategy,
    ScalarPrecisionStrategy,
    validate_input_tables,
)

# --- 6. Incremental updates -----------------------------------------------------


def _assert_close(left, right, path="result"):
    """Equal, except floats compared to a relative 1e-9 (merge order changes rounding)."""
    if isinstance(left, float) or isinstance(right, float):
        assert left == pytest.approx(right, rel=1e-9, abs=1e-12), path
    elif hasattr(left, "__dataclass_fields__") and not isinstance(left, type):
        assert type(left) is type(right), path
        for name in left.__dataclass_fields__:
            _assert_close(getattr(left, name), getattr(right, name), f"{path}.{name}")
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_close(left[key], right[key], f"{path}[{key!r}]")
    elif isinstance(left, list | tuple):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_close(a, b, f"{path}[{i}]")
    else:
        assert left == right, path


@pytest.mark.parametrize("strategy", STRATEGIES_UNDER_CONTRACT, ids=STRATEGY_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_incremental_replay_equals_full_recomputation(name, strategy):
    # Proposal item 6: incremental evidence at step k must equal full
    # recomputation on prefix k. Strategies without incremental support are
    # recomputed in both runs, so they must match exactly as well.
    scenario = SCENARIOS[name]()
    full = replay(scenario, [strategy])
    incremental = replay(scenario, [strategy], incremental=True)
    assert [i.step for i in full] == [i.step for i in incremental]
    for before, after in zip(full, incremental, strict=True):
        assert before.acquired_fovs == after.acquired_fovs
        _assert_close(before.result.without_timing(), after.result.without_timing())


def test_packaged_incremental_estimators_are_exercised():
    # Guards the test above against silently comparing recomputation with itself.
    from morphocoverage.evaluator import IncrementalReplay

    scenario = SCENARIOS["homogeneous"]()
    data = validate_input_tables(
        scenario.objects, fovs=scenario.fovs, feature_cols=list(FEATURE_COLS)
    )
    incremental = {
        type(estimator).__name__
        for strategy in STRATEGIES_UNDER_CONTRACT
        for estimator in strategy.estimators
        if IncrementalReplay(
            SamplingStrategy([estimator]), data, objective=POPULATION, group={}
        ).incremental
    }
    assert incremental == {
        "ScalarPrecisionEstimator",
        "_ObjectRowsCheck",
        "FOVIncidencePlanEstimator",
    }


@pytest.mark.parametrize("incremental", [False, True])
def test_timing_accounts_for_the_whole_decision(incremental):
    scenario = SCENARIOS["unequal_objects_per_fov"]()
    strategies = [
        ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
        DescriptorPlateauStrategy("feature_a", 0.1, 3, ORDER_COL),
        FixedFOVStrategy(5),
    ]
    results = replay(scenario, strategies, incremental=incremental, min_fovs=3)
    for item in results:
        timing = item.result.timing
        assert item.result.compute_seconds == timing.compute_seconds
        parts = (
            timing.recompute_seconds,
            timing.update_seconds,
            timing.initial_fit_seconds,
            timing.framework_overhead_seconds,
        )
        assert all(part >= 0 for part in parts)
        assert timing.total_decision_seconds == pytest.approx(sum(parts))
        if item.step > 3:
            assert timing.initial_fit_seconds == 0.0
        precision = item.result.strategy.startswith("scalar_precision")
        if incremental and precision:
            assert timing.recompute_seconds == 0.0 and timing.update_seconds > 0
        else:
            assert timing.update_seconds == 0.0
