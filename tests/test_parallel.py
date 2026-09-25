"""Worker processes change where work runs, never what it returns.

Every entry point with ``n_jobs`` must give the same results, in the same
order, as the serial reference, apart from time measurements. Strategies are
run apart across processes and put back in the order of one joint replay.
"""

import pandas as pd
import pytest
from contract_registry import (
    POPULATION,
    STRATEGIES_UNDER_CONTRACT,
    STRATEGY_IDS,
    replay,
    without_timing,
)
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import FixedFOVStrategy, ScalarPrecisionStrategy, run_strategies
from morphocoverage._parallel import check_n_jobs, map_units
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import OutcomeCriterion, replay_acquisitions
from morphocoverage.sequential import _strategy_units


def _square(x, y):
    return x * x + y


def test_map_units_keeps_input_order_and_checks_n_jobs():
    jobs = [(i, 1) for i in range(23)]
    expected = [i * i + 1 for i in range(23)]
    assert map_units(_square, jobs, 1) == expected
    assert map_units(_square, jobs, 3) == expected
    assert map_units(_square, [], 3) == []
    assert check_n_jobs(-1) >= 1
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="n_jobs"):
            check_n_jobs(bad)


def test_every_strategy_gives_identical_results_in_worker_processes():
    scenario = SCENARIOS["condition_repeat_groups"]()
    strategies = list(STRATEGIES_UNDER_CONTRACT)

    def run(n_jobs):
        results = run_strategies(
            scenario.objects,
            fovs=scenario.fovs,
            strategies=strategies,
            feature_cols=list(FEATURE_COLS),
            objective=POPULATION,
            n_jobs=n_jobs,
        )
        return [result.without_timing() for result in results]

    serial = run(1)
    assert len(serial) == 4 * len(strategies)
    assert run(3) == serial


def test_strategies_go_apart_only_when_runs_cannot_occupy_every_worker():
    strategies = [FixedFOVStrategy(3), FixedFOVStrategy(6)]
    assert _strategy_units(strategies, 1, 1) == [strategies]
    assert _strategy_units(strategies, 4, 4) == [strategies]
    assert _strategy_units(strategies, 4, 3) == [[s] for s in strategies]


# condition_repeat_groups has 4 analysis groups: 2 workers replay whole groups,
# 6 workers replay every strategy apart.
@pytest.mark.parametrize("n_jobs", [2, 6], ids=["strategies_together", "strategies_apart"])
@pytest.mark.parametrize("incremental", [False, True])
def test_prefix_replay_is_identical_in_worker_processes(incremental, n_jobs):
    scenario = SCENARIOS["condition_repeat_groups"]()
    strategies = STRATEGIES_UNDER_CONTRACT[::3]
    serial = without_timing(replay(scenario, strategies, incremental=incremental, min_fovs=2))
    parallel = without_timing(
        replay(scenario, strategies, incremental=incremental, min_fovs=2, n_jobs=n_jobs)
    )
    assert parallel == serial
    # Strategy order within each step is that of the serial joint replay.
    names = [STRATEGY_IDS[i] for i in range(0, len(STRATEGY_IDS), 3)]
    assert [item.result.strategy for item in serial[: len(names)]] == names


# 4 groups x 2 splits x 1 order = 8 runs: 3 workers replay whole groups, 6
# workers single runs, 9 workers every strategy of every run apart.
@pytest.mark.parametrize("n_jobs", [3, 6, 9], ids=["groups", "runs", "strategies_apart"])
def test_acquisition_replay_tables_are_identical_in_worker_processes(n_jobs):
    scenario = SCENARIOS["condition_repeat_groups"]()
    criterion = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better")
    settings = {
        "feature_cols": list(FEATURE_COLS),
        "strategies": [
            FixedFOVStrategy(3),
            ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
            FixedFOVStrategy(6),
        ],
        "criteria": [criterion],
        "objective": POPULATION,
        "reservoir_fraction": 0.3,
        "acquisition_order_col": ORDER_COL,
        "n_splits": 2,
        "min_fovs": 2,
        "incremental": True,
        "orders": [AcquisitionOrder.recorded()],
    }
    one = replay_acquisitions(scenario.objects, scenario.fovs, **settings)
    two = replay_acquisitions(scenario.objects, scenario.fovs, **settings, n_jobs=n_jobs)
    for name in ("decisions", "outcomes", "streams", "paths"):
        pd.testing.assert_frame_equal(getattr(one, name), getattr(two, name))
