"""Tests for fixed evaluation reservoirs and their use in the benchmark."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    EffortPolicy,
    Estimator,
    Evidence,
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    SamplingStrategy,
    validate_input_tables,
)
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    records_to_frame,
    run_benchmark,
)
from morphocoverage.evaluator import iter_analysis_groups
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.reservoir import ReservoirSplit, reference_error_floor, spatial_blocks

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
NEVER_STOPS = FixedFOVStrategy(10**6)


def _group(scenario="homogeneous", **fov_columns):
    built = SCENARIOS[scenario]()
    fovs = built.fovs.assign(**fov_columns)
    data = validate_input_tables(built.objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    return next(iter(iter_analysis_groups(data)))


def _keys(data):
    return set(data.fovs.loc[:, list(data.fov_key)].itertuples(index=False, name=None))


def test_split_is_a_seeded_partition_of_the_group():
    group, data = _group()
    stream, held = ReservoirSplit(0.3, 1, None).split(group, data)
    assert _keys(stream) | _keys(held) == _keys(data)
    assert not _keys(stream) & _keys(held)
    assert len(held.fovs) == 6  # ceil(0.3 * 20) single-FOV blocks
    assert len(stream.objects) + len(held.objects) == len(data.objects)
    again = ReservoirSplit(0.3, 1, None).split(group, data)[1]
    assert _keys(again) == _keys(held)
    assert _keys(ReservoirSplit(0.3, 2, None).split(group, data)[1]) != _keys(held)


def test_split_keeps_whole_blocks_on_one_side():
    blocks = np.repeat(np.arange(5), 4)  # 20 FOVs in blocks of 4
    group, data = _group(block=blocks)
    stream, held = ReservoirSplit(0.25, 0, "block").split(group, data)
    assert not set(stream.fovs["block"]) & set(held.fovs["block"])
    assert len(held.fovs) % 4 == 0


def test_groups_get_their_own_split_from_the_same_seed():
    built = SCENARIOS["condition_repeat_groups"]()
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    split = ReservoirSplit(0.25, 0, None)
    positions = []
    for group, group_data in iter_analysis_groups(data):
        held = split.split(group, group_data)[1]
        assert set(held.fovs["condition"]) == {group["condition"]}
        positions.append(tuple(sorted(held.fovs["image_file"])))
    # Same image names in every group, but not the same held-out positions.
    assert len(set(positions)) > 1


def test_blocks_are_held_out_with_equal_probability_whatever_their_size():
    # Regression (external audit): skipping blocks that would overshoot a
    # FOV-count cap never held out an 8-FOV block next to a 2-FOV block, so the
    # "reference" was always the small block, here also the one with a
    # different phenotype. Every block must have the same inclusion chance.
    fovs = pd.DataFrame(
        {"image_file": [f"f{i}" for i in range(10)], "block": ["big"] * 8 + ["small"] * 2}
    )
    objects = pd.DataFrame({"image_file": fovs["image_file"], "x": [0.0] * 8 + [10.0] * 2})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    held_big = [
        set(ReservoirSplit(0.5, seed, "block").split({}, data)[1].fovs["block"]) == {"big"}
        for seed in range(200)
    ]
    assert 0.4 < np.mean(held_big) < 0.6


def test_every_fov_has_the_same_inclusion_probability_with_unequal_blocks():
    # Blocks of 1 to 6 FOVs whose phenotype grows with block size: a
    # size-dependent rule would bias the reservoir's mean.
    sizes = [1, 1, 2, 2, 3, 4, 5, 6]
    block = np.repeat(np.arange(len(sizes)), sizes)
    fovs = pd.DataFrame({"image_file": [f"f{i}" for i in range(len(block))], "block": block})
    objects = pd.DataFrame(
        {"image_file": fovs["image_file"], "x": np.repeat(np.array(sizes, float), sizes)}
    )
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    counts = pd.Series(0, index=fovs["image_file"])
    runs = 2000
    for seed in range(runs):
        held = ReservoirSplit(0.25, seed, "block").split({}, data)[1]
        assert held.fovs["block"].nunique() == 2  # ceil(0.25 * 8) blocks
        counts[held.fovs["image_file"]] += 1
    assert np.allclose(counts / runs, 2 / 8, atol=0.04)


def test_groups_too_small_to_split_are_refused():
    group, data = _group(block=0)
    with pytest.raises(ValueError, match="too few FOVs or blocks"):
        ReservoirSplit(0.3, 0, "block").split(group, data)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ReservoirSplit(0.0, 0, None),
        lambda: ReservoirSplit(1.0, 0, None),
        lambda: ReservoirSplit(0.5, 0.5, None),
        lambda: ReservoirSplit(0.5, 0, 3),
    ],
)
def test_invalid_splits_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()


def test_block_column_must_exist_and_be_complete():
    group, data = _group()
    with pytest.raises(ValueError, match="not a column"):
        ReservoirSplit(0.3, 0, "block").split(group, data)
    group, data = _group(block=[np.nan] + [1.0] * 19)
    with pytest.raises(ValueError, match="missing values"):
        ReservoirSplit(0.3, 0, "block").split(group, data)


def test_spatial_blocks_are_a_square_grid():
    fovs = pd.DataFrame({"x": [0.0, 99.0, 100.0, -1.0], "y": [0.0, 50.0, 250.0, 0.0]})
    assert list(spatial_blocks(fovs, "x", "y", 100.0)) == ["0_0", "0_0", "1_2", "-1_0"]
    with pytest.raises(ValueError):
        spatial_blocks(fovs, "x", "y", 0)
    with pytest.raises(ValueError):
        spatial_blocks(fovs.assign(x=[np.nan, 1, 2, 3]), "x", "y", 10.0)


def _benchmark(scenario, reservoir, strategies=(NEVER_STOPS,), **kwargs):
    built = SCENARIOS[scenario]()
    dataset = BenchmarkDataset(
        scenario,
        "development",
        lambda: (built.objects, built.fovs),
        FEATURE_COLS,
        ORDER_COL,
    )
    return run_benchmark(
        BenchmarkRegistry([dataset]),
        scenario,
        strategies=list(strategies),
        metrics=[RelativeMeanError("feature_a")],
        objective=POPULATION,
        reservoir=reservoir,
        **kwargs,
    )


def test_full_reference_errors_vanish_at_the_end_but_reservoir_errors_do_not():
    # The motivating defect: against the full acquisition, a run that never
    # stops has zero error by construction.
    [full] = _benchmark("multimodal", None)
    [held] = _benchmark("multimodal", ReservoirSplit(0.3, 0, None))
    assert full.outcomes["relative_mean_error:feature_a"] == 0
    assert held.outcomes["relative_mean_error:feature_a"] > 0
    assert (full.reference, held.reference) == ("full", "reservoir")
    assert held.total_fovs + held.reservoir_fovs == full.total_fovs
    assert held.reservoir == {"fraction": 0.3, "seed": 0, "block_col": None}


class _NoReservoirProbe(Estimator):
    """Fails the replay if a strategy is ever shown a reservoir FOV."""

    name = "no_reservoir_probe"

    def __init__(self, held):
        self.held = held

    def estimate(self, data, *, objective):
        assert not _keys(data) & self.held, "a reservoir FOV reached a strategy"
        return Evidence({})


def test_replay_never_reveals_reservoir_fovs():
    built = SCENARIOS["homogeneous"]()
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    group, group_data = next(iter(iter_analysis_groups(data)))
    split = ReservoirSplit(0.4, 3, None)
    held = _keys(split.split(group, group_data)[1])
    probe = SamplingStrategy([_NoReservoirProbe(held)], [EffortPolicy("n_fovs", 10**6)])
    for order in (AcquisitionOrder.recorded(), AcquisitionOrder.permuted(5)):
        [record] = _benchmark("homogeneous", split, strategies=[probe], orders=[order])
        assert not record.stopped
        assert record.total_fovs == record.n_fovs == len(group_data.fovs) - len(held)


def test_every_group_is_scored_against_its_own_reservoir():
    records = _benchmark("condition_repeat_groups", ReservoirSplit(0.25, 0, None))
    assert len(records) == 4
    assert all(r.reservoir_fovs == 2 and r.total_fovs == 6 for r in records)
    row = records_to_frame(records).iloc[0]
    assert row["reference"] == "reservoir" and row["reservoir.fraction"] == 0.25


def test_reference_error_floor_is_seeded_and_positive():
    group, data = _group("multimodal")
    held = ReservoirSplit(0.5, 0, None).split(group, data)[1]
    metric = RelativeMeanError("feature_a")
    floor = reference_error_floor(metric, held, n_splits=5, seed=1, block_col=None)
    assert len(floor) == 5 and all(value > 0 for value in floor)
    assert floor == reference_error_floor(metric, held, n_splits=5, seed=1, block_col=None)
    with pytest.raises(ValueError):
        reference_error_floor(metric, held, n_splits=0, seed=1, block_col=None)


def _raster(n=100):
    """A raster scan of ``n`` FOVs along x whose phenotype drifts smoothly with position."""
    fovs = pd.DataFrame(
        {
            "image_file": [f"f{i}" for i in range(n)],
            "stage_x": np.arange(n, dtype=float),
            "block": np.arange(n) // 10,
        }
    )
    drift = np.cumsum(np.random.default_rng(0).normal(size=n))
    objects = pd.DataFrame({"image_file": fovs["image_file"], "x": drift})
    return validate_input_tables(objects, fovs=fovs, feature_cols=["x"])


def _stream_steps(stream):
    """Distance and phenotype change between successive stream FOVs in acquisition order."""
    ordered = stream.fovs.sort_values("stage_x")
    values = stream.objects.set_index("image_file").loc[ordered["image_file"], "x"].to_numpy()
    return np.diff(ordered["stage_x"].to_numpy()), np.abs(np.diff(values))


def test_the_reservoir_stream_is_a_thinned_acquisition():
    # External re-audit: holding FOVs out also removes them from the recorded
    # order. Successive stream FOVs of a raster scan are about twice as far
    # apart, and differ more in a drifting phenotype, than successive acquired
    # FOVs; contiguous blocks keep most neighbours together.
    data = _raster()
    full_step, full_change = _stream_steps(data)
    singles, blocked = [], []
    for seed in range(20):
        singles.append(_stream_steps(ReservoirSplit(0.5, seed, None).split({}, data)[0]))
        blocked.append(_stream_steps(ReservoirSplit(0.5, seed, "block").split({}, data)[0]))
    single_step = np.mean([step.mean() for step, _ in singles])
    single_change = np.mean([change.mean() for _, change in singles])
    assert full_step.mean() == 1.0
    assert 1.7 < single_step < 2.3
    assert single_change > 1.2 * full_change.mean()
    # Blocks of 10 contiguous FOVs: most successive stream FOVs are still neighbours.
    assert np.mean([np.median(step) for step, _ in blocked]) == 1.0
