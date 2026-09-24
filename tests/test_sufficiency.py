"""Tests for the retrospective sufficiency report."""

import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
    run_strategies,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.sufficiency import sufficiency_report

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
PERMUTED = [AcquisitionOrder.permuted(seed) for seed in range(5)]


def _report(scenario, strategies, orders=PERMUTED, **kwargs):
    built = SCENARIOS[scenario]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=orders,
        **kwargs,
    )


def test_verdicts_match_the_strategies_on_the_full_acquisition():
    built = SCENARIOS["homogeneous"]()
    strategies = [FixedFOVStrategy(5), ScalarPrecisionStrategy("feature_a", 0.001, "fov", "point")]
    report = _report("homogeneous", strategies)
    direct = run_strategies(
        built.objects,
        fovs=built.fovs,
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    assert list(report.final["verdict"]) == [r.decision.value for r in direct]
    assert list(report.final["n_fovs"]) == [20, 20]
    assert report.final["q.averaging_cv"].iloc[1] == pytest.approx(
        direct[1].evidence.value("averaging_cv")
    )


def test_stop_points_and_spare_fovs_follow_the_replay():
    stops = _report("homogeneous", [FixedFOVStrategy(6)]).stops.iloc[0]
    assert (stops["acquired_fovs"], stops["orders"], stops["never_stopped"]) == (20, 5, 0)
    assert stops["stop_fov_median"] == stops["stop_fov_p10"] == stops["stop_fov_p90"] == 6
    assert stops["spare_fovs_median"] == 14


def test_criteria_never_met_are_counted_and_have_no_stop_point():
    stops = _report("homogeneous", [FixedFOVStrategy(10**6)]).stops.iloc[0]
    assert stops["never_stopped"] == 5
    assert stops["stop_fov_median"] is None and stops["spare_fovs_median"] is None
    final = _report("homogeneous", [FixedFOVStrategy(10**6)]).final.iloc[0]
    assert final["verdict"] == "continue"


def test_variants_of_one_strategy_stay_apart():
    report = _report("homogeneous", [FixedFOVStrategy(3), FixedFOVStrategy(9)])
    assert report.stops["config_id"].nunique() == 2
    assert sorted(report.stops["stop_fov_median"]) == [3, 9]
    assert set(report.final["strategy"]) == {"fixed_fov"}


def test_analysis_groups_are_reported_separately():
    report = _report("condition_repeat_groups", [FixedFOVStrategy(2)])
    assert len(report.final) == len(report.stops) == 4
    assert {"condition", "repeat"} <= set(report.stops.columns)
    assert set(report.stops["acquired_fovs"]) == {8}


def test_the_recorded_order_needs_an_order_column():
    with pytest.raises(ValueError, match="acquisition_order_col"):
        _report("homogeneous", [FixedFOVStrategy(3)], orders=[AcquisitionOrder.recorded()])
    recorded = _report(
        "homogeneous",
        [FixedFOVStrategy(3)],
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col=ORDER_COL,
    )
    assert recorded.stops["orders"].iloc[0] == 1


@pytest.mark.parametrize("strategies, orders", [([], PERMUTED), ([FixedFOVStrategy(3)], [])])
def test_empty_inputs_are_refused(strategies, orders):
    with pytest.raises(ValueError):
        _report("homogeneous", strategies, orders=orders)
