"""Tests for acquisition-order efficiency (P8 of docs/optimise/optimise.md)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    FIXED,
    MEAN_OF_FEATURE_A,
    POPULATION,
    STAGE,
    _efficiency,
    _f_reliability,
    _f_replay,
    _gradient,
    _homogeneous_replay,
    _homogeneous_tables,
    _make,
    _tables,
)
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    adequacy_points,
    dispersed_order,
    order_efficiency,
    order_text,
    redundancy_distance,
    replay_acquisitions,
    spatial,
    stage_travel,
    stopping_calibration,
)
from morphocoverage.retrospective._text import order_name
from morphocoverage.retrospective.adequacy import _median_point
from morphocoverage.retrospective.frontier import family, frontier
from morphocoverage.retrospective.spatial import EfficiencyResult, efficiency_summary


def test_dispersed_order_is_farthest_point_first():
    line = np.column_stack([np.arange(10.0), np.zeros(10)])
    order = dispersed_order(line, 0)
    assert order[:3].tolist() == [0, 9, 4]
    assert sorted(order.tolist()) == list(range(10))
    assert dispersed_order(line, 5)[1] == 0  # both ends are 5 away; the lower index wins
    with pytest.raises(ValueError):
        dispersed_order(line, 10)


def test_stage_travel_is_the_cumulative_path():
    path = stage_travel(np.array([[0, 0], [3, 4], [3, 0]]))
    assert path.tolist() == [0.0, 5.0, 9.0]
    assert stage_travel(np.empty((0, 2))).size == 0


def test_a_gradient_is_covered_sooner_by_dispersed_orders_with_more_travel_per_field():
    orders = [AcquisitionOrder.recorded()] + [AcquisitionOrder.dispersed(s) for s in range(5)]
    result = _efficiency(*_gradient(), orders)
    table = result.table.set_index("order")
    assert table.loc["dispersed", "median_fovs"] < table.loc["recorded", "median_fovs"]
    per_field = table["median_travel"] / table["median_fovs"]
    assert per_field["dispersed"] > per_field["recorded"]
    text = result.summary()
    assert re.search(r"^  spread out ", text, re.MULTILINE)
    assert re.search(r"^  recorded ", text, re.MULTILINE)
    assert " min" in text  # timing given
    assert "about" in result.summary(detail="full") and " min" in result.summary(detail="full")
    assert (result.runs["travel_total"] >= result.runs["travel_to_sustained"].fillna(0)).all()


def test_in_a_uniform_sample_orders_do_not_differ_beyond_noise():
    built = SCENARIOS["homogeneous"]()
    orders = [AcquisitionOrder.permuted(s) for s in range(8)] + [
        AcquisitionOrder.dispersed(s) for s in range(8)
    ]
    table = _efficiency(built.objects, built.fovs, orders).table.set_index("order")
    assert abs(table.loc["dispersed", "median_fovs"] - table.loc["permuted", "median_fovs"]) <= 3


def test_dispersed_orders_need_stage_coordinates():
    built = SCENARIOS["homogeneous"]()
    with pytest.raises(ValueError, match="needs stage_cols"):
        replay_acquisitions(
            built.objects,
            built.fovs,
            feature_cols=list(FEATURE_COLS),
            strategies=[FixedFOVStrategy(5)],
            criteria=[OutcomeCriterion(RelativeMeanError("feature_a"), 0.02, "lower_is_better")],
            objective=POPULATION,
            reservoir_fraction=0.3,
            orders=[AcquisitionOrder.dispersed(0)],
        )
    with pytest.raises(ValueError, match="stage coordinates"):
        AcquisitionOrder.dispersed(0).apply(built.fovs, "acquisition_order")


def test_redundancy_distance_is_found_where_neighbours_are_alike():
    objects, fovs = _gradient()
    table = redundancy_distance(
        objects,
        fovs,
        feature_cols=["feature_a"],
        stage_cols=STAGE,
        bin_edges=[0, 250, 450, 650, 850, 1100, 2500],
    ).table
    row = table.iloc[0]
    assert row["redundancy_distance"] > row["smallest_lag"]
    flat = SCENARIOS["homogeneous"]()
    uniform = redundancy_distance(
        flat.objects,
        flat.fovs,
        feature_cols=["feature_a"],
        stage_cols=STAGE,
        bin_edges=[0, 150, 300, 600, 1200, 2500],
    ).table.iloc[0]
    assert uniform["redundancy_distance"] == pytest.approx(uniform["smallest_lag"])


def test_pairs_closer_counts_every_pair_closer_than_the_distance():
    # The distance is the mean lag of its bin, so that bin also holds closer pairs (#239).
    objects, fovs = _gradient()
    edges = [0, 250, 450, 650, 850, 1100, 2500]
    row = redundancy_distance(
        objects, fovs, feature_cols=["feature_a"], stage_cols=STAGE, bin_edges=edges
    ).table.iloc[0]
    xy = fovs[list(STAGE)].to_numpy(dtype=float)
    lags = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))[np.triu_indices(60, 1)]
    closer = int(np.sum(lags < row["redundancy_distance"]))
    assert row["pairs_closer"] == closer
    assert np.any((lags < row["redundancy_distance"]) & (lags >= 450))


def test_one_stage_axis_given_twice_is_refused():
    # ("stage_x", "stage_x") would run as a diagonal line: fake 2-D geometry (#232).
    objects, fovs = _gradient()
    twice = ("stage_x", "stage_x")
    with pytest.raises(ValueError, match="not one axis twice"):
        redundancy_distance(
            objects, fovs, feature_cols=["feature_a"], stage_cols=twice, bin_edges=[0, 2500]
        )
    with pytest.raises(ValueError, match="not one axis twice"):
        replay_acquisitions(
            objects,
            fovs,
            feature_cols=["feature_a"],
            strategies=[FixedFOVStrategy(4)],
            criteria=[OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better")],
            objective=POPULATION,
            reservoir_fraction=0.3,
            stage_cols=twice,
        )


def test_f3_a_group_without_two_fields_gets_a_row_with_a_reason():
    good = _make(n_fovs=20)
    bad = _make(n_fovs=3, empty=(1, 2), groups=(("B", "r1"),), seed=1)
    objects = pd.concat([good[0], bad[0]], ignore_index=True)
    fovs = pd.concat([good[1], bad[1]], ignore_index=True)
    table = redundancy_distance(
        objects, fovs, feature_cols=["f"], stage_cols=("x", "y"), bin_edges=[0, 150, 300, 600]
    ).table.set_index("condition")
    assert pd.isna(table.loc["A", "reason"]) and table.loc["A", "smallest_lag"] > 0
    assert table.loc["B", "reason"].startswith("only 1 field with observations")
    assert pd.isna(table.loc["B", "redundancy_distance"])
    far = redundancy_distance(
        *good, feature_cols=["f"], stage_cols=("x", "y"), bin_edges=[5000, 6000]
    ).table
    assert "no pair of fields is between 5000 and 6000" in far["reason"].iloc[0]


def _replay(n_fovs=16, seed=7, threshold=0.03, random_state=0, **kwargs):
    objects, fovs = _make(n_fovs=n_fovs, seed=seed)
    criterion = OutcomeCriterion(RelativeMeanError("f"), threshold, "lower_is_better")
    return replay_acquisitions(
        objects,
        fovs,
        feature_cols=["f"],
        strategies=[FixedFOVStrategy(5)],
        criteria=[criterion],
        objective=ObjectiveConfig(),
        reservoir_fraction=0.3,
        n_splits=2,
        random_state=random_state,
        stage_cols=("x", "y"),
        **kwargs,
    )


def test_f4_unreached_runs_count_as_beyond_their_own_stream_in_the_median():
    replay = _replay(orders=[AcquisitionOrder.permuted(s) for s in range(2)])
    result = order_efficiency(replay)
    runs = result.runs["sustained_realized"]
    assert runs.isna().sum() == len(runs) / 2  # half the runs never reached it
    row = result.table.iloc[0]
    assert row["median_fovs"] is None and row["median_travel"] is None
    full = result.summary(detail="full")
    assert "not in half or more of runs" in full and "50% of runs, within 11 fields" in full
    assert re.search(r"random +never +– +50%", result.summary())
    # A median over reached runs and runs beyond their own stream, never a fill value.
    assert order_efficiency(_replay(threshold=0.2)).table["median_fovs"].iloc[0] >= 1


def test_f5_adequacy_from_another_replay_is_refused():
    replay, other = _replay(), _replay(random_state=7)
    with pytest.raises(ValueError, match="another replay .*random_state"):
        order_efficiency(replay, adequacy_points(other))
    assert len(order_efficiency(replay, adequacy_points(replay)).table) == 1


def test_f14_the_order_comparison_names_every_criterion_and_short_times():
    table = pd.DataFrame(
        [
            {
                "order": order,
                "runs": 2,
                "reached": 1.0,
                "median_fovs": 3.0,
                "median_travel": 100.0,
                "median_travel_total": 400.0,
                "median_time": 20.0,
                "stream_fovs": 10,
                "longest_stream_fovs": 10,
            }
            for order in ("permuted", "dispersed")
        ]
    )
    two = {"labels": {"a": "a", "b": "b"}}
    text = efficiency_summary(EfficiencyResult(pd.DataFrame(), table, (), two))
    assert "for all the criteria together" in text and "20 s" in text
    full = efficiency_summary(EfficiencyResult(pd.DataFrame(), table, (), two), "full")
    assert "about 20 s" in full
    one = efficiency_summary(EfficiencyResult(pd.DataFrame(), table, (), {"labels": {"a": "a"}}))
    assert "for this criterion" in one


def test_f10_groups_the_replay_skipped_are_named_in_every_summary():
    big, big_fovs = _tables(n_fovs=8, wells=("A1",))
    one, one_fovs = _tables(n_fovs=1, wells=("A2",))
    replay = _f_replay(
        pd.concat([big, one]), pd.concat([big_fovs, one_fovs]), stage_cols=("x", "y")
    )
    assert replay.skipped["technical_repeat"].tolist() == ["A2"]
    for result in (adequacy_points(replay), stopping_calibration(replay)):
        assert result.skipped["technical_repeat"].tolist() == ["A2"]
        # Every summary names the skipped group (by its values, or as column=value).
        assert "A2: skipped, only 1 field" in result.summary()
        assert "A2: skipped, only 1 field" in result.summary(detail="full")
    efficiency = order_efficiency(replay)
    assert efficiency.skipped["technical_repeat"].tolist() == ["A2"]
    # The order comparison names the group by its values (condition · biological_repeat · well).
    for detail in ("short", "full"):
        assert "R1 · A2: skipped, only 1 field" in efficiency.summary(detail=detail)


def test_f14_order_efficiency_refuses_adequacy_of_other_data():
    kwargs = {"orders": [AcquisitionOrder.permuted(s) for s in range(2)], "stage_cols": ("x", "y")}
    first = _f_replay(*_tables(n_fovs=16, seed=1, scale=0.2), **kwargs)
    other = _f_replay(*_tables(n_fovs=16, seed=2, scale=3.0), **kwargs)
    assert first.settings == other.settings
    with pytest.raises(ValueError, match="other realized points"):
        order_efficiency(first, adequacy_points(other))
    assert len(order_efficiency(first, adequacy_points(first)).table)


def _redundancy(objects, fovs):
    return redundancy_distance(
        objects,
        fovs,
        feature_cols=["f"],
        stage_cols=("x", "y"),
        bin_edges=[0, 150, 300, 500],
        fov_col="position_id",
    ).table.set_index("biological_repeat")


def test_f22_a_constant_readout_has_no_redundancy_distance():
    objects, fovs = _tables(n_fovs=16, repeats=("R1", "R2"))
    table = _redundancy(objects.assign(f=5.0), fovs)
    assert table["redundancy_distance"].isna().all()
    assert table["reason"].str.contains("do not vary").all()


def test_f23_a_missing_value_in_one_group_leaves_the_others_measured():
    objects, fovs = _tables(n_fovs=16, repeats=("R1", "R2"))
    objects.loc[objects.index[objects["biological_repeat"] == "R2"][0], "f"] = np.nan
    table = _redundancy(objects, fovs)
    assert pd.notna(table.loc["R1", "smallest_lag"]) and pd.isna(table.loc["R1", "reason"])
    assert "missing values in f (1 observation)" in table.loc["R2", "reason"]


def test_f24_one_order_phrase_one_median_rule_one_skip_rule():
    assert not hasattr(spatial, "_ORDER_WORDS") and not hasattr(spatial, "_median_beyond")
    point = _median_point(pd.Series([3, None, None], dtype="Int64"), pd.Series([12, 12, 12]))
    assert point.startswith("not within the fields available (up to 12)")
    assert _median_point(pd.Series([3, 5, None], dtype="Int64"), pd.Series([12] * 3)) == "field 5"
    objects, fovs = _tables(n_fovs=8, wells=("A1",))
    one, one_fovs = _tables(n_fovs=1, wells=("A2",))
    tables = (pd.concat([objects, one]), pd.concat([fovs, one_fovs]))
    skipped_replay = _f_replay(*tables).skipped["reason"].iloc[0]
    skipped_reliability = _f_reliability(*tables).skipped["reason"].iloc[0]
    assert skipped_replay == skipped_reliability
    assert not hasattr(order_text, "_ORDER_NAMES")  # the words of an order kind live in _text
    assert (
        f"{order_name('recorded')}  "
        in order_efficiency(
            _f_replay(
                objects,
                fovs,
                orders=[AcquisitionOrder.recorded()],
                acquisition_order_col="order",
                stage_cols=("x", "y"),
            )
        ).summary()
    )


def test_order_efficiency_states_the_runs_that_never_reached_adequacy(gradient_replay):
    result = order_efficiency(gradient_replay)
    table = result.table.set_index("order")
    text = result.summary()
    for order, row in table.iterrows():
        if pd.isna(row["median_fovs"]) or row["reached"] == 1:
            continue
        share = f"{100 * (1 - row['reached']):.0f}% of runs"
        assert f"never within the {row['stream_fovs']} fields in {share}" in text, order
    if (table["reached"] == 1).all():
        pytest.skip("every replayed order reached adequacy; nothing to state")


def test_order_efficiency_frontier_and_redundancy_refuse_what_they_cannot_read():
    objects, fovs = _homogeneous_tables()
    with pytest.raises(TypeError, match="^replay must be a ReplayResult$"):
        order_efficiency(None)
    orders = [AcquisitionOrder.permuted(0)]
    plain = _homogeneous_replay(reservoir_fraction=0.3, n_splits=1, orders=orders)
    with pytest.raises(ValueError) as raised:
        order_efficiency(plain)
    expected = "the replay has no stage paths; run replay_acquisitions with stage_cols"
    assert str(raised.value) == expected
    staged = _homogeneous_replay(
        reservoir_fraction=0.3, n_splits=1, orders=orders, stage_cols=("stage_x", "stage_y")
    )
    with pytest.raises(ValueError) as raised:
        order_efficiency(staged, stage_speed=0)
    assert str(raised.value) == "stage_speed must be a positive, finite number"
    one = family(staged, "fixed", {FIXED[0].config_id: 3})
    two = family(staged, "fixed", {FIXED[1].config_id: 5})
    with pytest.raises(ValueError) as raised:
        frontier(staged, [one, two], criterion=MEAN_OF_FEATURE_A.name, alphas=[0.2], unit_cols=[])
    assert str(raised.value) == "family names used twice: ['fixed']"
    with pytest.raises(ValueError) as raised:
        redundancy_distance(
            objects,
            fovs,
            feature_cols=["feature_a"],
            stage_cols=("x", "y"),
            bin_edges=[0, 1],
            share=0,
        )
    assert str(raised.value) == "share must be in (0, 1]"
