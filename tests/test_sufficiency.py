"""Tests for the retrospective sufficiency report."""

import numpy as np
import pandas as pd
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
from morphocoverage.bounds import upper_bound
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.sufficiency import CalibrationSettings, sufficiency_report

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


def _novel_acquisition(n_fovs=12, state_fovs=(3, 7), outlier_fovs=(5,)):
    """Common cells everywhere; its own 3-cell state in ``state_fovs``, one odd cell elsewhere."""
    rng = np.random.default_rng(0)
    rows = []
    for j in range(n_fovs):
        cells = rng.normal(10, 0.1, size=(20, 2))
        if j in state_fovs:
            cells = np.vstack([cells, rng.normal(20 + 5 * j, 0.05, size=(3, 2))])
        if j in outlier_fovs:
            cells = np.vstack([cells, [[50.0, 50.0]]])
        rows += [
            {"image_file": f"f{j}", "label_id": i, "a": a, "b": b} for i, (a, b) in enumerate(cells)
        ]
    fovs = pd.DataFrame({"image_file": [f"f{j}" for j in range(n_fovs + 1)]})  # one empty
    return pd.DataFrame(rows), fovs


NOVELTY = SupportNoveltyEstimator(0.5, 1, None, 2, 3)


def _novelty_report(strategies=(), **kwargs):
    objects, fovs = _novel_acquisition()
    return sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=list(strategies),
        objective=POPULATION,
        orders=PERMUTED,
        novelty=NOVELTY,
        **kwargs,
    )


def test_next_fov_chance_counts_fovs_with_enough_new_cells():
    report = _novelty_report()
    row = report.next_fov.iloc[0]
    # Two of 13 acquired FOVs (one empty) hold a 3-cell state no other FOV shows;
    # the single odd cell does not count.
    assert (row["fovs_imaged"], row["fovs_with_new"]) == (13, 2)
    assert row["chance_new"] == pytest.approx(2 / 13)
    assert row["chance_new_at_most"] == pytest.approx(upper_bound(2, 13, 0.95))
    assert pd.isna(row["extra_fovs_if_nothing_new"])
    assert report.enough.empty and report.calibration.empty


def test_new_observations_lists_only_cells_of_new_fovs_without_features():
    new = _novelty_report().new_observations
    assert sorted(new["image_file"].unique()) == ["f3", "f7"] and len(new) == 6
    assert not {"a", "b"} & set(new.columns) and "label_id" in new.columns


def test_a_target_gives_fovs_still_needed_and_a_criterion():
    report = _novelty_report(novelty_target=0.2)
    row = report.next_fov.iloc[0]
    needed = int(row["extra_fovs_if_nothing_new"])
    assert upper_bound(2, 13 + needed, 0.95) <= 0.2 < upper_bound(2, 12 + needed, 0.95)
    # 2 of 13 is below 20%, so the bound falls below it at the current rate too, later.
    assert row["extra_fovs_at_current_rate"] >= needed
    criterion = report.enough.iloc[0]
    assert criterion["criterion"] == "support_novelty"
    assert criterion["settings"] == "lofo_novel_fov_upper <= 0.2"
    assert criterion["sampled_enough"] == "no"


def test_summary_answers_both_questions_in_words():
    report = _novelty_report(
        [FixedFOVStrategy(5)],
        novelty_target=0.2,
        calibration=CalibrationSettings((6,), 10, 0),
    )
    text = report.summary()
    assert "Did we sample enough?" in text and "Would one more FOV show something new?" in text
    assert "fixed_fov" in text and "yes, met by the 13 acquired FOVs" in text
    assert "Chance: 15% (at most" in text and "2 of 13 FOVs" in text
    assert "Check on this dataset: predicting from 6 random FOVs" in text
    assert "'New' means at least 3 observations with fewer than 1 other FOV" in text
    assert list(report.calibration["fovs_imaged"]) == [6]


def test_readable_criteria_state_the_threshold_direction():
    report = _report("homogeneous", [ScalarPrecisionStrategy("feature_a", 0.001, "fov", "point")])
    assert report.enough["settings"].iloc[0] == "averaging_cv <= 0.001"
    assert report.enough["sampled_enough"].iloc[0] in {"yes", "no"}
    assert report.next_fov.empty and report.new_observations.empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategies": [], "novelty": None},
        {"novelty": None, "novelty_target": 0.1},
        {"novelty_target": 1.0},
        {"novelty": None, "calibration": CalibrationSettings((5,), 5, 0)},
        {"novelty": "support_novelty"},
    ],
)
def test_inconsistent_questions_are_refused(kwargs):
    objects, fovs = _novel_acquisition()
    settings = {"strategies": [FixedFOVStrategy(3)], "novelty": NOVELTY, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        sufficiency_report(
            objects,
            fovs,
            feature_cols=["a", "b"],
            objective=POPULATION,
            orders=PERMUTED,
            **settings,
        )
