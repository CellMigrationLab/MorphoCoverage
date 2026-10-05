"""Tests for acquisition-order efficiency (P8 of docs/questions.md)."""

import re

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    dispersed_order,
    order_efficiency,
    redundancy_distance,
    replay_acquisitions,
    stage_travel,
)

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
STAGE = ("stage_x", "stage_y")


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


def _gradient(seed=0):
    """A 10 x 6 raster; the phenotype changes along y, the raster runs along x first."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for i in range(60):
        x, y = i % 10, i // 10
        name = f"f{i:02d}"
        fov_rows.append(
            {
                "condition": "control",
                "biological_repeat": "R1",
                "image_file": name,
                "acquisition_order": i,
                "stage_x": x * 200.0,
                "stage_y": y * 200.0,
            }
        )
        values = rng.normal(10 + y, 0.5, size=(15, 2))
        object_rows.append(
            pd.DataFrame(values, columns=list(FEATURE_COLS)).assign(
                condition="control", biological_repeat="R1", image_file=name
            )
        )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def _efficiency(objects, fovs, orders, tolerance=0.06):
    criterion = OutcomeCriterion(RelativeMeanError("feature_a"), tolerance, "lower_is_better")
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(5)],
        criteria=[criterion],
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=orders,
        acquisition_order_col="acquisition_order",
        stage_cols=STAGE,
        n_splits=3,
    )
    return order_efficiency(replay, per_fov_seconds=2.0, stage_speed=1000.0)


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
    )
    row = table.iloc[0]
    assert row["redundancy_distance"] > row["smallest_lag"]
    flat = SCENARIOS["homogeneous"]()
    uniform = redundancy_distance(
        flat.objects,
        flat.fovs,
        feature_cols=["feature_a"],
        stage_cols=STAGE,
        bin_edges=[0, 150, 300, 600, 1200, 2500],
    ).iloc[0]
    assert uniform["redundancy_distance"] == pytest.approx(uniform["smallest_lag"])


def test_pairs_closer_counts_every_pair_closer_than_the_distance():
    # The distance is the mean lag of its bin, so that bin also holds closer pairs (#239).
    objects, fovs = _gradient()
    edges = [0, 250, 450, 650, 850, 1100, 2500]
    row = redundancy_distance(
        objects, fovs, feature_cols=["feature_a"], stage_cols=STAGE, bin_edges=edges
    ).iloc[0]
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
