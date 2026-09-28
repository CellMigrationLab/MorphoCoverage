"""Regressions for the second audit, group B (live session, replay, benchmark, reservoir,
calibration), one test per finding."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    DescriptorSpreadStrategy,
    FixedFOVStrategy,
    ObjectiveConfig,
    OnlineSession,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    validate_input_tables,
)
from morphocoverage import sequential as sequential_module
from morphocoverage._parallel import check_n_jobs
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    permuted_fovs,
    run_benchmark,
)
from morphocoverage.calibration import calibrate_next_fov_novelty
from morphocoverage.evaluator import FovSubsets, iter_analysis_groups
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.reservoir import ReservoirSplit, reference_error_floor, spatial_blocks
from morphocoverage.retrospective import OutcomeCriterion, replay_acquisitions
from morphocoverage.strategies.support_novelty import SupportNoveltyEstimator

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
SRC = Path(__file__).resolve().parents[1] / "src" / "morphocoverage"
MEAN_F = OutcomeCriterion(RelativeMeanError("f"), 0.05, "lower_is_better")


def _two_groups(n_a=12, n_b=12, seed=0):
    """Two conditions of ``n_a`` and ``n_b`` positions, 10 objects each, with an order."""
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        {
            "condition": ["a"] * n_a + ["b"] * n_b,
            "repeat": "r1",
            "position_id": [f"p{i:02d}" for i in range(n_a + n_b)],
            "order": list(range(n_a)) + list(range(n_b)),
        }
    )
    objects = fovs.loc[fovs.index.repeat(10), ["condition", "repeat", "position_id"]]
    objects = objects.reset_index(drop=True)
    objects["f"] = rng.normal(10, 1, len(objects)) + np.repeat(rng.normal(0, 1, len(fovs)), 10)
    return objects, fovs


def _registry(objects, fovs, **dataset):
    return BenchmarkRegistry(
        [BenchmarkDataset("d", "development", lambda: (objects, fovs), ("f",), "order", **dataset)]
    )


def test_b1_live_session_refuses_z_planes_counted_as_fovs():
    # 3 positions x 4 z-planes, one file per plane and no position_id: the
    # validator refuses the table, so the live session must refuse it too.
    rows, fov_rows = [], []
    for position in range(3):
        for z in range(4):
            name = f"pos{position}_z{z}.tif"
            fov_rows.append({"condition": "a", "repeat": "r1", "image_file": name})
            rows += [
                {"condition": "a", "repeat": "r1", "image_file": name, "z": z, "f": v}
                for v in np.linspace(1, 2, 5)
            ]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fov_rows)
    with pytest.raises(ValueError, match="plane/channel/frame"):
        validate_input_tables(objects, feature_cols=["f"], fovs=fovs)
    session = OnlineSession(
        strategies=[FixedFOVStrategy(5)], feature_cols=["f"], objective=POPULATION
    )
    first = fovs.iloc[0]
    session.add_fov(objects[objects.image_file == first.image_file], first.to_dict())
    second = fovs.iloc[1]
    with pytest.raises(ValueError, match="plane/channel/frame"):
        session.add_fov(objects[objects.image_file == second.image_file], second.to_dict())
    # With fov_col named explicitly, each file is accepted as one position.
    named = OnlineSession(
        strategies=[FixedFOVStrategy(5)],
        feature_cols=["f"],
        objective=POPULATION,
        fov_col="image_file",
    )
    for _, meta in fovs.iterrows():
        (result,) = named.add_fov(objects[objects.image_file == meta.image_file], meta.to_dict())
    assert result.n_fovs == 12


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
    key = ["condition", "repeat", "image_file"]
    for _, meta in built.fovs.sort_values(ORDER_COL).iterrows():
        mask = (built.objects[key] == meta[key]).all(axis=1)
        (live,) = session.add_fov(built.objects[mask], meta.to_dict())
        reference = by_step[(tuple(live.group.items()), live.n_fovs)]
        assert live.without_timing().evidence == reference.without_timing().evidence
        assert live.rule_decision == reference.rule_decision


def test_b3_text_acquisition_order_is_refused():
    fovs = pd.DataFrame(
        {"position_id": [f"p{i}" for i in range(1, 13)], "order": [str(i) for i in range(1, 13)]}
    )
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 2), "f": np.arange(24.0)})
    with pytest.raises(ValueError, match="not numbers or date-times"):
        evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["f"],
            acquisition_order_col="order",
            objective=POPULATION,
        )
    with pytest.raises(ValueError, match="not numbers or date-times"):
        replay_acquisitions(
            objects,
            fovs,
            feature_cols=["f"],
            strategies=[FixedFOVStrategy(3)],
            criteria=[MEAN_F],
            objective=POPULATION,
            reservoir_fraction=0.3,
            acquisition_order_col="order",
        )
    # Date-times are an order.
    timed = fovs.assign(order=pd.date_range("2026-01-01", periods=12, freq="min"))
    results = evaluate_acquisition_prefixes(
        objects,
        timed,
        strategies=[FixedFOVStrategy(3)],
        feature_cols=["f"],
        acquisition_order_col="order",
        objective=POPULATION,
    )
    assert [k[-1] for k in results[2].acquired_fovs] == ["p1", "p2", "p3"]


def _permuted_error(objects, fovs, seed=5):
    records = run_benchmark(
        _registry(objects, fovs, fov_col="position_id", group_cols=("condition",)),
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


def test_b5_calibration_row_of_a_group_ignores_other_groups_and_counts():
    objects, fovs = _two_groups()
    novelty = SupportNoveltyEstimator(0.5, 1, None, 3)

    def row_b(o, f, counts):
        table = calibrate_next_fov_novelty(
            o,
            f,
            feature_cols=["f"],
            novelty=novelty,
            fov_counts=counts,
            draws=15,
            random_state=0,
        )
        return (
            table[(table.condition == "b") & (table.fovs_imaged == 4)]
            .drop(columns="condition")
            .reset_index(drop=True)
        )

    with_a = row_b(objects, fovs, [4])
    pd.testing.assert_frame_equal(
        with_a, row_b(objects[objects.condition == "b"], fovs[fovs.condition == "b"], [4])
    )
    pd.testing.assert_frame_equal(with_a, row_b(objects, fovs, [3, 4]))


def test_b6_add_fov_is_atomic_when_an_estimator_refuses():
    def observations(position):
        return pd.DataFrame({"position_id": position, "f": np.linspace(1, 2, 5)})

    pilot = ScalarPrecisionStrategy("f", 0.05, "object", "point", pilot_fovs=2, order_col="order")
    session = OnlineSession(
        strategies=[FixedFOVStrategy(10), pilot, FixedFOVStrategy(20)],
        feature_cols=["f"],
        objective=POPULATION,
    )
    session.add_fov(observations("p1"), {"position_id": "p1", "order": 1})
    with pytest.raises(ValueError, match="missing values"):
        session.add_fov(observations("p2"), {"position_id": "p2", "order": None})
    session.add_fov(observations("p2"), {"position_id": "p2", "order": 2})  # the retry
    results = session.add_fov(observations("p3"), {"position_id": "p3", "order": 3})
    assert [(r.n_fovs, r.n_observations) for r in results] == [(3, 15)] * 3
    # A refused first FOV does not fix the session's columns.
    fresh = OnlineSession(
        strategies=[FixedFOVStrategy(3)], feature_cols=["f"], objective=POPULATION
    )
    with pytest.raises(ValueError):
        fresh.add_fov(pd.DataFrame({"f": [np.inf]}), {"position_id": "p1"})
    assert fresh.fov_col is None


@pytest.mark.parametrize(
    "areas",
    [
        [100.0, -100.0, 100.0, 100.0],
        [100.0, 0.0, 100.0, 100.0],
        [100.0, np.inf, 100.0, 100.0],
        ["100"] * 4,
    ],
)
def test_b7_benchmark_refuses_unusable_areas(areas):
    fovs = pd.DataFrame({"position_id": [f"p{i}" for i in range(4)], "order": range(4)})
    fovs["area"] = pd.Series(areas, dtype=object if isinstance(areas[0], str) else float)
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 3), "f": np.arange(12.0)})
    with pytest.raises(ValueError, match="area_col"):
        run_benchmark(
            _registry(objects, fovs, area_col="area"),
            "d",
            strategies=[FixedFOVStrategy(2)],
            metrics=[RelativeMeanError("f")],
            objective=POPULATION,
        )


def test_b8_reservoir_split_and_floor_refuse_several_groups():
    objects, fovs = _two_groups(10, 10)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    with pytest.raises(ValueError, match="2 analysis groups"):
        ReservoirSplit(0.5, 0, None).split({}, data)
    with pytest.raises(ValueError, match="2 analysis groups"):
        reference_error_floor(RelativeMeanError("f"), data, n_splits=2, seed=0, block_col=None)
    for group, group_data in iter_analysis_groups(data):
        stream, held = ReservoirSplit(0.5, 0, None).split(group, group_data)
        assert set(stream.fovs.condition) == set(held.fovs.condition) == {group["condition"]}


def test_b9_records_keep_stop_on_and_a_consistent_final_decision():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"position_id": [f"p{i}" for i in range(8)], "order": range(8)})
    objects = pd.DataFrame(
        {"position_id": np.repeat(fovs.position_id, 50), "f": rng.uniform(9, 11, 400)}
    )
    phenotype = ObjectiveConfig(SamplingObjective.PHENOTYPE)
    for stop_on in ("decision", "rule"):
        (record,) = run_benchmark(
            _registry(objects, fovs),
            "d",
            strategies=[ScalarPrecisionStrategy("f", 0.05, "object", "point")],
            metrics=[RelativeMeanError("f")],
            objective=phenotype,
            stop_on=stop_on,
        )
        assert record.stop_on == stop_on == record.as_row()["stop_on"]
        assert record.stopped == (record.final_decision.value == "stop")
    assert record.stopped  # the rule's own verdict stops


def test_b11_reservoir_size_is_rounded_before_the_ceiling():
    fovs = pd.DataFrame({"position_id": [f"p{i:03d}" for i in range(100)]})
    objects = pd.DataFrame({"position_id": fovs.position_id, "f": np.arange(100.0)})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    assert 0.07 * 100 > 7  # floating point: 7.000000000000001
    _, held = ReservoirSplit(0.07, 0, None).split({}, data)
    assert len(held.fovs) == 7


def test_b12_calibration_reports_a_group_too_small_for_every_count():
    objects, fovs = _two_groups(12, 5)
    table = calibrate_next_fov_novelty(
        objects,
        fovs,
        feature_cols=["f"],
        novelty=SupportNoveltyEstimator(0.5, 1, None, 3),
        fov_counts=[6],
        draws=5,
        random_state=0,
    )
    small = table[table.condition == "b"]
    assert len(small) == 1 and small.draws.iloc[0] == 0
    assert "has 5 FOVs" in small.reason.iloc[0]
    assert table[table.condition == "a"].reason.isna().all()


def test_b13_spatial_blocks_refuse_infinite_coordinates():
    xy = pd.DataFrame({"x": [0.0, 150.0, np.inf], "y": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="infinite"):
        spatial_blocks(xy, "x", "y", 100.0)


def test_b14_numpy_integers_are_accepted():
    assert check_n_jobs(np.int64(2)) == 2 and type(check_n_jobs(np.int64(2))) is int
    assert ReservoirSplit(0.3, np.int64(1), None).seed == 1
    assert AcquisitionOrder.permuted(np.int64(3)) == AcquisitionOrder.permuted(3)
    objects, fovs = _two_groups(8, 8)
    table = calibrate_next_fov_novelty(
        objects,
        fovs,
        feature_cols=["f"],
        novelty=SupportNoveltyEstimator(0.5, 1, None, 3),
        fov_counts=[np.int64(4)],
        draws=np.int64(3),
        random_state=np.int64(0),
    )
    assert table.fovs_imaged.tolist() == [4, 4]
    data = validate_input_tables(
        objects[objects.condition == "a"], fovs=fovs[fovs.condition == "a"], feature_cols=["f"]
    )
    floors = reference_error_floor(
        RelativeMeanError("f"), data, n_splits=np.int64(2), seed=np.int64(0), block_col=None
    )
    assert len(floors) == 2


def test_b15_reference_error_floor_with_odd_blocks_says_which_side_is_larger():
    doc = inspect.getdoc(reference_error_floor)
    assert "odd number of blocks" in doc
    fovs = pd.DataFrame({"position_id": ["p0", "p1", "p2"]})
    objects = pd.DataFrame({"position_id": fovs.position_id, "f": [1.0, 2.0, 3.0]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    scored, reference = ReservoirSplit(0.5, 0, None).split({}, data)
    assert (len(scored.fovs), len(reference.fovs)) == (1, 2)


def test_b17_shared_private_sequential_helpers_are_documented():
    shared = set()
    for path in (SRC / "benchmark.py", SRC / "retrospective" / "replay.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("sequential"):
                shared |= {alias.name for alias in node.names if alias.name.startswith("_")}
    assert shared
    doc = sequential_module.__doc__
    assert "Shared-private" in doc
    assert all(f"``{name}``" in doc for name in shared), shared


def test_b18_fovsubsets_take_refuses_repeated_rows():
    fovs = pd.DataFrame({"position_id": ["p0", "p1", "p2"]})
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 4), "f": np.arange(12.0)})
    subsets = FovSubsets(validate_input_tables(objects, feature_cols=["f"], fovs=fovs))
    with pytest.raises(ValueError, match="distinct FOV rows"):
        subsets.take(np.array([0, 0, 1]))
    assert len(subsets.take(np.array([0, 2])).objects) == 8


def test_f15a_replay_skips_a_single_field_group_with_its_reason():
    objects, fovs = _two_groups(12, 1)
    result = replay_acquisitions(
        objects,
        fovs,
        feature_cols=["f"],
        strategies=[FixedFOVStrategy(3)],
        criteria=[MEAN_F],
        objective=POPULATION,
        reservoir_fraction=0.3,
        n_splits=2,
        acquisition_order_col="order",
    )
    assert set(result.decisions.condition) == {"a"}
    assert result.skipped.condition.tolist() == ["b"]
    assert "only 1 field" in result.skipped.reason.iloc[0]
    with pytest.raises(ValueError, match="no analysis group"):
        replay_acquisitions(
            objects[objects.condition == "b"],
            fovs[fovs.condition == "b"],
            feature_cols=["f"],
            strategies=[FixedFOVStrategy(3)],
            criteria=[MEAN_F],
            objective=POPULATION,
            reservoir_fraction=0.3,
        )
