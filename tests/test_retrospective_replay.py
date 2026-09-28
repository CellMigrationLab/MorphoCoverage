"""Tests for replayed acquisitions: stopping calibration (P6) and adequacy points (R2, P4)."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import OutcomeMetric, RelativeMeanError
from morphocoverage.retrospective import (
    ALL_CRITERIA,
    OutcomeCriterion,
    adequacy_points,
    first_apparent_adequacy,
    first_realized_adequacy,
    replay_acquisitions,
    sampling_reliability,
    stopping_calibration,
    sustained_apparent_adequacy,
    sustained_realized_adequacy,
)
from morphocoverage.schema import validate_input_tables

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
MEAN_A = OutcomeCriterion(RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean")


def _replay(scenario, strategies, criteria=(MEAN_A,), **kwargs):
    built = SCENARIOS[scenario]()
    settings = {"reservoir_fraction": 0.3, "n_splits": 2}
    settings.update(kwargs)
    return replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=list(strategies),
        criteria=list(criteria),
        objective=POPULATION,
        **settings,
    )


# --- The adequacy points (the examples of docs/benchmarking.md) --------------------


def test_apparent_points_follow_the_documented_example():
    steps = list(range(10, 31))
    decisions = ["continue" if s < 16 or s == 24 else "stop" for s in steps]
    assert first_apparent_adequacy(steps, decisions) == 16
    assert sustained_apparent_adequacy(steps, decisions) == 25
    # An undefined decision at 30 breaks the sustained run; nothing after it.
    undefined = decisions[:-1] + [None]
    assert sustained_apparent_adequacy(steps, undefined) is None
    unsupported = [*decisions[:-1], "unsupported", "stop"]
    assert sustained_apparent_adequacy([*steps, 31], unsupported) == 31
    assert first_apparent_adequacy(steps, ["continue"] * len(steps)) is None


def test_realized_points_follow_the_documented_example():
    steps = list(range(8, 20))
    statuses = ["preserved" if s in (10, 11) or s >= 13 else "not_preserved" for s in steps]
    assert first_realized_adequacy(steps, statuses) == 10
    assert sustained_realized_adequacy(steps, statuses) == 13
    # An undefined step neither confirms nor breaks a sustained run.
    with_gap = statuses[:]
    with_gap[steps.index(15)] = "undefined"
    assert sustained_realized_adequacy(steps, with_gap) == 13
    assert sustained_realized_adequacy(steps, [*statuses[:-1], "not_preserved"]) is None
    with pytest.raises(ValueError):
        sustained_realized_adequacy([1, 2], ["preserved", "maybe"])
    with pytest.raises(ValueError, match="increasing"):
        first_apparent_adequacy([2, 1], ["stop", "stop"])


# --- The replay --------------------------------------------------------------------


class _Probe(OutcomeMetric):
    """Records the reservoir and prefix of every assessment."""

    name = "probe"

    def __init__(self, key):
        self.key = key
        self.calls = []

    def evaluate(self, full, stopped):
        held = frozenset(full.fovs[self.key].itertuples(index=False, name=None))
        seen = frozenset(stopped.fovs[self.key].itertuples(index=False, name=None))
        self.calls.append((held, seen))
        return 0.0


def test_the_reservoir_is_that_of_sampling_reliability_and_never_replayed():
    built = SCENARIOS["zero_object_fovs"]()
    key = list(
        validate_input_tables(
            built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS)
        ).fov_key
    )
    replayed, drawn = _Probe(key), _Probe(key)
    _replay(
        "zero_object_fovs",
        [FixedFOVStrategy(3)],
        [OutcomeCriterion(replayed, 0.0, "lower_is_better")],
        n_splits=3,
    )
    sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[OutcomeCriterion(drawn, 0.0, "lower_is_better")],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=3,
        draws=1,
    )
    assert {held for held, _ in replayed.calls} == {held for held, _ in drawn.calls}
    assert all(not held & seen for held, seen in replayed.calls)
    empty = set(built.truth["empty_fovs"])
    assert any(any(row[-1] in empty for row in seen) for _, seen in replayed.calls)


def test_outcomes_are_assessed_on_every_prefix_whatever_the_strategies():
    one = _replay("homogeneous", [FixedFOVStrategy(3)])
    other = _replay("homogeneous", [FixedFOVStrategy(9), FixedFOVStrategy(2)])
    pd.testing.assert_frame_equal(one.outcomes, other.outcomes)
    lengths = one.outcomes.groupby(["split", "order_id"])["fovs"].agg(["min", "max", "count"])
    assert (lengths["min"] == 1).all()
    assert (lengths["max"] == lengths["count"]).all()
    # Realized adequacy is a property of the order, never of a strategy.
    pd.testing.assert_frame_equal(adequacy_points(one).realized, adequacy_points(other).realized)


def test_recorded_order_follows_the_acquisition_order_column():
    built = SCENARIOS["homogeneous"]()
    fovs = built.fovs.assign(acquisition_order=built.fovs["acquisition_order"].to_numpy()[::-1])
    probe = _Probe(["condition", "repeat", "image_file"])
    result = replay_acquisitions(
        built.objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(4)],
        criteria=[OutcomeCriterion(probe, 0.0, "lower_is_better")],
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="acquisition_order",
        n_splits=1,
    )
    assert set(result.decisions["order"]) == {"recorded"}
    rank = dict(zip(fovs["image_file"], fovs["acquisition_order"], strict=True))
    revealed = []
    for _, seen in probe.calls:
        new = {row[-1] for row in seen} - set(revealed)
        assert len(new) == 1  # one whole FOV per step
        revealed += new
    assert [rank[f] for f in revealed] == sorted(rank[f] for f in revealed)
    with pytest.raises(ValueError, match="recorded order needs"):
        _replay("homogeneous", [FixedFOVStrategy(4)], orders=[AcquisitionOrder.recorded()])


def test_default_orders_and_determinism():
    first = _replay("homogeneous", [FixedFOVStrategy(4)])
    assert set(first.decisions["order"]) == {"permuted"}
    assert first.decisions["order_id"].nunique() == 10
    again = _replay("homogeneous", [FixedFOVStrategy(4)])
    pd.testing.assert_frame_equal(first.outcomes, again.outcomes)
    moved = _replay("homogeneous", [FixedFOVStrategy(4)], random_state=7)
    assert not moved.outcomes.equals(first.outcomes)


def test_incremental_replay_gives_the_same_decisions():
    strategies = [ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"), FixedFOVStrategy(5)]
    full = _replay("unequal_objects_per_fov", strategies, min_fovs=2)
    fast = _replay("unequal_objects_per_fov", strategies, min_fovs=2, incremental=True)
    pd.testing.assert_frame_equal(full.decisions, fast.decisions)


def test_groups_are_independent():
    both = _replay("condition_repeat_groups", [FixedFOVStrategy(3)])
    built = SCENARIOS["condition_repeat_groups"]()
    first = built.fovs[["condition", "repeat"]].drop_duplicates().iloc[0].to_dict()
    mask = np.ones(len(built.fovs), dtype=bool)
    for column, value in first.items():
        mask &= built.fovs[column].to_numpy() == value
    fovs = built.fovs[mask]
    objects = built.objects.merge(fovs[["condition", "repeat"]].drop_duplicates())
    alone = replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(3)],
        criteria=[MEAN_A],
        objective=POPULATION,
        reservoir_fraction=0.3,
        n_splits=2,
    )
    selected = both.outcomes
    for column, value in first.items():
        selected = selected[selected[column] == value]
    pd.testing.assert_frame_equal(
        selected.reset_index(drop=True), alone.outcomes.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategies": []},
        {"strategies": [FixedFOVStrategy(3), FixedFOVStrategy(3)]},
        {"criteria": []},
        {"orders": []},
        {"n_splits": 0},
        {"min_fovs": 0},
        {"random_state": 1.5},
    ],
)
def test_invalid_replays_are_refused(kwargs):
    strategies = kwargs.pop("strategies", [FixedFOVStrategy(3)])
    criteria = kwargs.pop("criteria", [MEAN_A])
    with pytest.raises((TypeError, ValueError)):
        _replay("homogeneous", strategies, criteria, **kwargs)


# --- Stopping calibration ------------------------------------------------------------


def _row(calibration, config_id, criterion=MEAN_A.name):
    table = calibration.table
    return table[(table["config_id"] == config_id) & (table["criterion"] == criterion)].iloc[0]


def test_a_rule_that_stops_at_once_and_one_that_never_stops():
    at_once, never = FixedFOVStrategy(1), FixedFOVStrategy(1000)
    # 4 splits: the 2 hold-outs of the label-independent seeds (audit 4, P9-2) both
    # happened to match one field's mean.
    calibration = stopping_calibration(_replay("multimodal", [at_once, never], n_splits=4))
    first = _row(calibration, at_once.config_id)
    assert first["stopped"] == 1 and first["median_stop_fovs"] == 1
    # One FOV of a multimodal sample does not always match (audit 4, P2-9: the stops not
    # preserved are premature or not preserved by the stream, and all four kinds add up).
    assert first["not_preserved_at_stop"] > 0
    kinds = ["correct", "premature", "not_preserved_by_stream", "undefined_at_stop"]
    assert np.isclose(sum(first[k] for k in kinds), 1)
    last = _row(calibration, never.config_id)
    assert last["never_stopped"] == 1 and last["stopped"] == 0
    assert last["median_stop_fovs"] is None or pd.isna(last["median_stop_fovs"])
    assert last["never_stopped_preserved_at_end"] is not None


def test_a_fixed_fov_rule_in_random_orders_matches_sampling_reliability():
    built = SCENARIOS["homogeneous"]()
    rule = FixedFOVStrategy(4)
    orders = [AcquisitionOrder.permuted(seed) for seed in range(60)]
    calibration = stopping_calibration(_replay("homogeneous", [rule], n_splits=3, orders=orders))
    reliability = sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[MEAN_A],
        fov_counts=[4],
        reservoir_fraction=0.3,
        n_splits=3,
        draws=60,
    )
    correct = _row(calibration, rule.config_id)["correct"]
    preserved = reliability.curve["preserved"].iloc[0]
    assert abs(correct - preserved) < 0.1


def test_all_criteria_together_is_never_easier_than_either():
    other = OutcomeCriterion(RelativeMeanError("feature_b"), 0.02, "lower_is_better")
    calibration = stopping_calibration(
        _replay("homogeneous", [FixedFOVStrategy(3)], criteria=[MEAN_A, other])
    )
    table = calibration.table.set_index("criterion")
    assert table.loc[ALL_CRITERIA, "correct"] <= min(
        table.loc[MEAN_A.name, "correct"], table.loc[other.name, "correct"]
    )


def test_calibration_and_adequacy_speak_plainly():
    replay = _replay("homogeneous", [FixedFOVStrategy(3), FixedFOVStrategy(1000)])
    calibration = stopping_calibration(replay).summary(detail="full")
    assert "fixed_fov (target_fovs=3)" in calibration
    assert "not preserved" in calibration and "never stopped" in calibration
    short = stopping_calibration(replay).summary()
    assert "fixed 3 fields" in short and "not preserved" in short and "never stopped" in short
    adequacy = adequacy_points(replay).summary()
    assert "reliably" in adequacy and "would have first said stop" in adequacy
    assert "fixed 3 fields" in adequacy and "never" in adequacy
    full = adequacy_points(replay).summary(detail="full")
    assert "would have first said stop" in full and "first preserved at" in full
    for text in (calibration, short, adequacy):
        for word in ("reservoir", "criterion", "prefix", "config_id"):
            assert word not in text


def test_adequacy_points_are_counts_within_the_stream():
    result = adequacy_points(_replay("homogeneous", [FixedFOVStrategy(5)]))
    apparent = result.apparent
    assert (apparent["first_apparent"] == 5).all()
    assert (apparent["sustained_apparent"] == 5).all()
    realized = result.realized.dropna(subset=["sustained_realized"])
    assert (realized["first_realized"] <= realized["sustained_realized"]).all()
    assert (realized["sustained_realized"] <= realized["stream_fovs"]).all()
    assert set(result.realized["criterion"]) == {ALL_CRITERIA, MEAN_A.name}


def test_combined_status_lets_a_failure_beat_an_undefined_outcome():
    from morphocoverage.retrospective.adequacy import combined_status

    assert combined_status(["preserved", "preserved"]) == "preserved"
    assert combined_status(["preserved", "undefined"]) == "undefined"
    assert combined_status(["undefined", "not_preserved"]) == "not_preserved"
    with pytest.raises(ValueError):
        combined_status([])


def test_strategy_labels_name_the_scalar_settings():
    from morphocoverage.retrospective.replay import strategy_label

    assert strategy_label(FixedFOVStrategy(5)) == "fixed_fov (target_fovs=5)"
    label = strategy_label(ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"))
    assert "target_cv=0.05" in label and "resampling_unit=fov" in label


def test_distinct_dispersed_seeds_give_distinct_orders():
    orders = [AcquisitionOrder.dispersed(seed) for seed in range(5)]
    result = _replay(
        "homogeneous",
        [FixedFOVStrategy(3)],
        orders=orders,
        stage_cols=("stage_x", "stage_y"),
        n_splits=1,
    )
    firsts = result.paths[result.paths["fovs"] == 2].groupby("order_id")["travel"].first()
    sequences = {
        order: tuple(rows.sort_values("fovs")["travel"])
        for order, rows in result.paths.groupby("order_id")
    }
    assert len(set(sequences.values())) == len(orders)
    assert len(firsts) == len(orders)


def test_worker_processes_give_identical_replays():
    one = _replay("condition_repeat_groups", [FixedFOVStrategy(3)], n_splits=2)
    two = _replay("condition_repeat_groups", [FixedFOVStrategy(3)], n_splits=2, n_jobs=2)
    for name in ("decisions", "outcomes", "streams", "paths"):
        pd.testing.assert_frame_equal(getattr(one, name), getattr(two, name))
    with pytest.raises(ValueError, match="n_jobs"):
        _replay("homogeneous", [FixedFOVStrategy(3)], n_jobs=0)


def test_outcome_prefixes_are_in_acquisition_order():
    # Outcome metrics and estimators that subsample rows by position (energy,
    # MMD) see the prefix rows in acquisition order, as a live session joins
    # the FOVs one by one (second audit, B-2).
    from morphocoverage.sequential import _Acquisition

    built = SCENARIOS["unequal_objects_per_fov"]()
    data = validate_input_tables(
        built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS), group_cols=[]
    )
    ordered = data.fovs.iloc[np.random.default_rng(4).permutation(len(data.fovs))]
    acquisition = _Acquisition(data, ordered)
    for step in range(1, acquisition.n_fovs + 1):
        arrived = pd.concat(
            [acquisition.new_fov(i).objects for i in range(1, step + 1)], ignore_index=True
        )
        pd.testing.assert_frame_equal(acquisition.prefix(step).objects, arrived)
