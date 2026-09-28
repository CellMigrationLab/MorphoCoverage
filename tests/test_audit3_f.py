"""Regressions for the third audit, group F: retrospective criteria, reliability, replay,
protocol, adequacy, calibration and acquisition-order efficiency.

One test per finding, named after it; each fails on the code before the fix.
"""

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy, ObjectiveConfig
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.bounds import interval
from morphocoverage.data_check import validate_input_tables
from morphocoverage.multivariate_outcomes import CovarianceError, EnergyError
from morphocoverage.outcomes import RareStateDetected, RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    adequacy,
    adequacy_points,
    order_efficiency,
    protocol,
    protocol_reliability,
    redundancy_distance,
    replay_acquisitions,
    sampling_reliability,
    spatial,
    stopping_calibration,
)
from morphocoverage.retrospective.adequacy import (
    _median_point,
    last_judged_step,
    sustained_realized_adequacy,
)
from morphocoverage.retrospective.criteria import Preservation
from morphocoverage.retrospective.reliability import _curve_row

OBJECTIVE = ObjectiveConfig()
MEAN = OutcomeCriterion(RelativeMeanError("f"), 0.02, "lower_is_better", "the mean within 2%")
RARE = OutcomeCriterion(RareStateDetected("rare", 1), 1.0, "higher_is_better", "the rare state")
FIX3 = FixedFOVStrategy(target_fovs=3)
GROUPS = ("condition", "repeat", "technical_repeat")
TECHNICAL = SamplingHierarchy("position_id", "technical_repeat", "repeat")
BY_WELL = SamplingHierarchy("image_file", "well", "repeat")


def _tables(
    n_fovs=12,
    objs=5,
    seed=0,
    wells=("A1",),
    repeats=("R1",),
    rare_fovs=(),
    scale=1.0,
):
    """Objects and FOVs of one condition: repeats x wells x fields on a 4-wide grid."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for repeat in repeats:
        for well in wells:
            for i in range(n_fovs):
                fid = f"ctrl_{repeat}_{well}_{i:02d}"
                base = {"condition": "ctrl", "repeat": repeat, "technical_repeat": well}
                fov_rows.append(
                    {
                        **base,
                        "position_id": fid,
                        "order": i + 1,
                        "x": (i % 4) * 100.0,
                        "y": (i // 4) * 100.0,
                    }
                )
                object_rows += [
                    {
                        **base,
                        "position_id": fid,
                        "f": float(rng.normal(10.0, scale)),
                        "rare": bool(i in rare_fovs and j == 0),
                    }
                    for j in range(objs)
                ]
    return pd.DataFrame(object_rows), pd.DataFrame(fov_rows)


def _replay(objects, fovs, **kwargs):
    settings = {
        "feature_cols": ["f"],
        "strategies": [FIX3],
        "criteria": [MEAN],
        "objective": OBJECTIVE,
        "reservoir_fraction": 0.3,
        "orders": [AcquisitionOrder.permuted(0)],
        "n_splits": 2,
        "fov_col": "position_id",
    }
    settings.update(kwargs)
    return replay_acquisitions(objects, fovs, **settings)


def _reliability(objects, fovs, **kwargs):
    settings = {
        "feature_cols": ["f"],
        "criteria": [MEAN],
        "fov_counts": [2],
        "reservoir_fraction": 0.5,
        "n_splits": 3,
        "draws": 4,
        "fov_col": "position_id",
    }
    settings.update(kwargs)
    return sampling_reliability(objects, fovs, **settings)


def _curve(preserved, **extra):
    """A reliability result from {(repeat, well): {k: preserved}}, streams of 12 fields."""
    rows = [
        {
            "condition": "c",
            "repeat": repeat,
            "well": well,
            "criterion": "m",
            "fovs": k,
            "preserved": value,
            "stream_fovs": 12,
            **extra,
        }
        for (repeat, well), by_k in preserved.items()
        for k, value in by_k.items()
    ]
    return ReliabilityResult(
        pd.DataFrame(rows), pd.DataFrame(), ("condition", "repeat", "well"), {"labels": {"m": "m"}}
    )


def _ninety_percent_result():
    """Two wells, each preserved in exactly 9 of 10 draws in each of 9 splits."""
    draws = pd.DataFrame(
        [
            {
                "condition": "c",
                "repeat": "R1",
                "technical_repeat": well,
                "split": split,
                "fovs": 4,
                "draw": draw,
                "criterion": "m",
                "value": 0.0,
                "status": "preserved" if draw < 9 else "not_preserved",
                "reservoir_fovs": 4,
                "stream_fovs": 8,
            }
            for well in ("A1", "A2")
            for split in range(9)
            for draw in range(10)
        ]
    )
    curve = pd.DataFrame(
        [
            {**dict(zip(GROUPS, key, strict=True)), "criterion": "m", "fovs": 4, **_curve_row(runs)}
            for key, runs in draws.groupby(list(GROUPS))
        ]
    )
    return ReliabilityResult(curve, draws, GROUPS, {"labels": {"m": "the mean"}, "n_splits": 9})


def test_f1_a_reliability_of_exactly_the_threshold_is_adequate():
    result = _ninety_percent_result()
    assert result.curve["preserved"].tolist() == [0.9, 0.9]
    table = protocol_reliability(result, TECHNICAL, threshold=0.9, strata=["condition"]).table
    assert table.loc[0, ["eligible", "adequate"]].tolist() == [2, 2]


def test_f2_protocol_summary_runs_when_no_sample_is_judged_anywhere():
    objects, fovs = _tables(wells=("A1", "A2"), repeats=("R1", "R2"))  # no rare state
    result = protocol_reliability(
        _reliability(objects, fovs, criteria=[RARE], fov_counts=[2, 4]),
        TECHNICAL,
        threshold=0.9,
        strata=["condition"],
    )
    assert result.per_unit.empty and "unit" in result.per_unit.columns
    assert "Could not be judged at any depth" in result.summary()


def test_f3_an_undefined_criterion_is_not_called_too_few_fields():
    objects, fovs = _tables(wells=("A1", "A2"), repeats=("R1", "R2"))
    result = protocol_reliability(
        _reliability(objects, fovs, criteria=[RARE], fov_counts=[2]),
        TECHNICAL,
        threshold=0.9,
        strata=["condition"],
    )
    row = result.table.iloc[0]
    assert (row["undefined"], row["beyond_stream"]) == (4, 0)
    text = result.summary()
    assert "no technical repeat could be judged at this depth" in text
    assert "had that many fields" not in text
    # Beyond the stream, it still says so.
    short = protocol_reliability(
        _reliability(objects, fovs, fov_counts=[20]), TECHNICAL, threshold=0.9, strata=[]
    )
    assert short.table.iloc[0]["beyond_stream"] == 4
    assert "no technical repeat had that many fields to judge" in short.summary()


def test_f4_fields_needed_median_counts_samples_that_never_got_there():
    reached = {2: 0.95, 4: 0.97, 8: 0.99}
    never = {2: 0.5, 4: 0.6, 8: 0.7}
    curve = {("R1", "A1"): reached, ("R1", "A2"): never, ("R2", "A1"): never}
    text = protocol_reliability(
        _curve(curve), BY_WELL, threshold=0.9, strata=["condition"]
    ).summary()
    assert "median not reached within the listed counts (only 1 of 3 samples got there" in text
    curve[("R3", "A1")] = {2: 0.5, 4: 0.95, 8: 0.95}
    curve[("R3", "A2")] = reached
    text = protocol_reliability(
        _curve(curve), BY_WELL, threshold=0.9, strata=["condition"]
    ).summary()
    # Reached at 2, 2 and 4, and 2 never: the median of 2, 2, 4, beyond, beyond is 4.
    assert "Fields needed per sample: median 4, range 2–4; 2 needing more counted as more" in text


def test_f5_protocol_interval_is_never_narrower_than_the_exact_one():
    def interval_of(n_fail):
        curve = {
            (f"R{u}", f"W{w}"): {4: 0.5 if u * 4 + w < n_fail else 0.95}
            for u in range(5)
            for w in range(4)
        }
        row = protocol_reliability(
            _curve(curve), BY_WELL, threshold=0.9, strata=["condition"]
        ).table.iloc[0]
        return row["share_low"], row["share_high"]

    all_low, all_high = interval_of(0)
    assert (all_low, all_high) == pytest.approx(interval(5, 5, 0.95))
    for n_fail in (1, 2, 4):
        low, high = interval_of(n_fail)
        assert 0 <= low <= all_low and high <= 1
    # At whole counts the exact interval is Clopper-Pearson's.
    assert protocol._exact(0.6, 5) == pytest.approx(interval(3, 5, 0.95))


def test_f6_neighbouring_seeds_share_no_reservoir_and_both_analyses_hold_the_same_one():
    objects, fovs = _tables(n_fovs=20)
    kwargs = {"orders": [AcquisitionOrder.recorded()], "acquisition_order_col": "order"}
    first = _replay(objects, fovs, random_state=0, n_splits=5, **kwargs).outcomes
    second = _replay(objects, fovs, random_state=1, n_splits=5, **kwargs).outcomes
    for split in range(4):
        shifted = first[first["split"] == split + 1]["value"].round(12).tolist()
        assert second[second["split"] == split]["value"].round(12).tolist() != shifted
    # The same random_state and split hold back the same fields in reliability and replay:
    # the whole stream scores the same against the reservoir in both.
    replay = _replay(objects, fovs, random_state=4, n_splits=3, **kwargs)
    stream = int(replay.streams["stream_fovs"].iloc[0])
    whole = replay.outcomes[replay.outcomes["fovs"] == stream].set_index("split")["value"]
    draws = _reliability(
        objects, fovs, fov_counts=[stream], reservoir_fraction=0.3, random_state=4
    ).draws
    drawn = draws.groupby("split")["value"].first()
    assert np.allclose(drawn.sort_index().to_numpy(), whole.sort_index().to_numpy())


def test_f7_order_efficiency_leaves_unjudged_runs_out():
    objects, fovs = _tables(n_fovs=16, rare_fovs=(3, 9))
    replay = _replay(
        objects,
        fovs,
        criteria=[RARE],
        orders=[AcquisitionOrder.permuted(s) for s in range(4)],
        n_splits=6,
        stage_cols=("x", "y"),
    )
    realized = adequacy_points(replay).realized
    unjudged = int((realized[realized["criterion"] == "all criteria"]["judged_fovs"] == 0).sum())
    assert unjudged > 0
    result = order_efficiency(replay)
    row = result.table.iloc[0]
    assert row["unjudged"] == unjudged
    judged = result.runs[result.runs["judged"]]
    assert row["reached"] == pytest.approx(judged["sustained_realized"].notna().mean())
    assert f"{unjudged} of {int(row['runs'])} runs could not be judged" in result.summary()


def _calibration():
    objects, fovs = _tables(n_fovs=16, scale=3.0)
    tight = OutcomeCriterion(RelativeMeanError("f"), 0.0005, "lower_is_better", "the mean")
    replay = _replay(
        objects,
        fovs,
        strategies=[FixedFOVStrategy(target_fovs=3), FixedFOVStrategy(target_fovs=50)],
        criteria=[tight],
        orders=[AcquisitionOrder.permuted(s) for s in range(4)],
        n_splits=5,
    )
    return stopping_calibration(replay)


def test_f8_a_stop_the_whole_stream_could_not_fix_is_not_too_early():
    result = _calibration()
    runs = result.runs[result.runs["strategy"] == "fixed_fov"]
    assert (runs["status_at_end"] == "not_preserved").all()
    stops = result.table[result.table["stopped"] == 1.0].iloc[0]
    assert stops["premature"] == 0 and stops["not_preserved_by_stream"] == 1.0
    total = stops[["correct", "premature", "not_preserved_by_stream", "undefined_at_stop"]]
    assert total.sum() == pytest.approx(stops["stopped"])
    text = result.summary()
    assert "0% of acquisitions stopped too early" in text
    assert "100% stopped where not even every available field together preserved it" in text


def test_f9_a_rule_that_never_stopped_says_whether_the_last_field_had_it():
    text = _calibration().summary()
    never = text.split("target_fovs=50")[1]
    assert "it never stopped within" in never
    assert "for the mean: 0% of those acquisitions had it by the last field" in never


def test_f10_groups_the_replay_skipped_are_named_in_every_summary():
    big, big_fovs = _tables(n_fovs=8, wells=("A1",))
    one, one_fovs = _tables(n_fovs=1, wells=("A2",))
    replay = _replay(pd.concat([big, one]), pd.concat([big_fovs, one_fovs]), stage_cols=("x", "y"))
    assert replay.skipped["technical_repeat"].tolist() == ["A2"]
    for result in (adequacy_points(replay), stopping_calibration(replay), order_efficiency(replay)):
        assert result.skipped["technical_repeat"].tolist() == ["A2"]
        assert "technical_repeat=A2: skipped, only 1 field" in result.summary()


def test_f11_too_few_objects_for_the_metric_is_not_preserved():
    objects = pd.DataFrame(
        {
            "condition": "c",
            "repeat": "r",
            "position_id": ["p0", "p1", "p1", "p2"],
            "a": [1.0, 2.0, 4.0, 3.0],
            "b": [5.0, 4.0, 7.0, 1.0],
        }
    )
    fovs = pd.DataFrame({"condition": "c", "repeat": "r", "position_id": ["p0", "p1", "p2"]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["a", "b"], fov_col="position_id")
    one = validate_input_tables(
        objects.iloc[:1], fovs=fovs.iloc[:1], feature_cols=["a", "b"], fov_col="position_id"
    )
    metric = CovarianceError(("a", "b"), standardize=False)
    assert metric.min_observations == 2 and RelativeMeanError("a").min_observations == 1
    assert EnergyError(("a",), False, None, 0).min_observations == 2
    criterion = OutcomeCriterion(metric, 0.5, "lower_is_better")
    assert criterion.assess(data, one) == (Preservation.NOT_PRESERVED, None)
    # A reference that leaves the metric undefined still gives undefined.
    flat = validate_input_tables(
        objects.assign(a=1.0, b=2.0), fovs=fovs, feature_cols=["a", "b"], fov_col="position_id"
    )
    assert criterion.assess(flat, one)[0] == Preservation.UNDEFINED


def test_f12_an_undefined_tail_leaves_sustained_realized_unreached():
    statuses = ["preserved", "not_preserved", "preserved", "preserved", "undefined", "undefined"]
    steps = list(range(1, 7))
    assert sustained_realized_adequacy(steps, statuses) is None
    assert last_judged_step(steps, statuses) == 4
    # Undefined steps in between are still skipped.
    assert sustained_realized_adequacy([1, 2, 3], ["preserved", "undefined", "preserved"]) == 1
    objects, fovs = _tables(n_fovs=16, scale=0.2)
    objects.loc[objects.index[objects["position_id"].str.endswith("_05")][0], "f"] = np.nan
    loose = OutcomeCriterion(RelativeMeanError("f"), 0.01, "lower_is_better", "the mean")
    replay = _replay(
        objects,
        fovs,
        criteria=[loose],
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="order",
        reservoir_fraction=0.25,
        n_splits=3,
        random_state=3,
    )
    result = adequacy_points(replay)
    realized = result.realized[result.realized["criterion"] == "all criteria"]
    assert realized["sustained_realized"].isna().all()
    assert (realized["last_judged"].dropna() < realized["stream_fovs"].min()).all()
    assert "the last fields could not be judged" in result.summary()


def test_f13_dispersed_seeds_beyond_the_stream_are_replayed_once():
    objects, fovs = _tables(n_fovs=5)
    replay = _replay(
        objects,
        fovs,
        reservoir_fraction=0.4,
        orders=[AcquisitionOrder.dispersed(s) for s in range(5)],
        n_splits=1,
        stage_cols=("x", "y"),
    )
    stream = int(replay.streams["stream_fovs"].iloc[0])
    assert stream == 3 and len(replay.streams) == stream
    firsts = replay.paths[replay.paths["fovs"] == 2].groupby("order_id")["travel"].first()
    assert len(firsts) == stream


def test_f14_order_efficiency_refuses_adequacy_of_other_data():
    kwargs = {"orders": [AcquisitionOrder.permuted(s) for s in range(2)], "stage_cols": ("x", "y")}
    first = _replay(*_tables(n_fovs=16, seed=1, scale=0.2), **kwargs)
    other = _replay(*_tables(n_fovs=16, seed=2, scale=3.0), **kwargs)
    assert first.settings == other.settings
    with pytest.raises(ValueError, match="other realized points"):
        order_efficiency(first, adequacy_points(other))
    assert len(order_efficiency(first, adequacy_points(first)).table)


def test_f15_protocol_wording_says_batches_and_at_least():
    curve = _curve({("B1", "A1"): {4: 0.95}, ("B1", "A2"): {4: 0.5}})
    curve = ReliabilityResult(
        curve.curve.rename(columns={"repeat": "batch"}),
        pd.DataFrame(),
        ("condition", "batch", "well"),
        curve.settings,
    )
    text = protocol_reliability(
        curve, SamplingHierarchy("image_file", "well", "batch"), threshold=0.9, strata=["condition"]
    ).summary()
    assert "no number can be chosen on other batches" in text and "batchs" not in text
    assert "counts when at least 90% of random acquisitions were enough" in text


def test_f16_reliability_says_what_it_rests_on_at_the_edges():
    objects, fovs = _tables()
    beyond = _reliability(objects, fovs, fov_counts=[20]).summary()
    assert "Judged on 0 or more" not in beyond
    assert "Judged on 6 or more held-back fields" in beyond
    nothing = _reliability(objects, fovs, criteria=[RARE]).summary()
    assert "Judged on" not in nothing
    assert "the criterion could not be judged against any of them" in nothing


def test_f17_identical_split_shares_print_no_rounding_noise():
    text = _ninety_percent_result().summary()
    assert "enough in 90% of random acquisitions (±0% across hold-outs)" in text
    assert "e-" not in text


def test_f18_repeated_hold_outs_are_counted_as_splits():
    objects, fovs = _tables(scale=2.0)
    fovs["block"] = ["b0", "b1", "b2"] * 4
    result = _reliability(
        objects,
        fovs,
        fov_counts=[2, 8],
        reservoir_fraction=0.3,
        n_splits=10,
        draws=5,
        block_col="block",
    )
    assert result.ceiling["distinct_holdouts"].iloc[0] == 3
    text = result.summary()
    assert "across splits" in text and "one acquisition per split" in text
    assert "were enough in 40% of splits" in text or "% of splits:" in text
    assert "across hold-outs" not in text and "% of hold-outs" not in text


def test_f19_unequal_streams_state_the_range_of_available_fields():
    objects, fovs = _tables(scale=2.0)
    fovs["block"] = ["big"] * 6 + ["b1", "b1", "b2", "b2", "b3", "b3"]
    result = _reliability(
        objects,
        fovs,
        reservoir_fraction=0.25,
        n_splits=4,
        draws=3,
        block_col="block",
        random_state=1,
    )
    ceiling = result.ceiling.iloc[0]
    assert (ceiling["shortest_stream_fovs"], ceiling["stream_fovs"]) == (6, 10)
    text = result.summary()
    assert "All 6 to 10 available fields together" in text
    assert "nothing is said beyond 10 fields (6 in some splits)" in text


def test_f20_numpy_integers_are_accepted_as_counts_and_seeds():
    objects, fovs = _tables(n_fovs=8)
    result = _reliability(
        objects, fovs, n_splits=np.int64(2), draws=np.int64(3), random_state=np.int64(0)
    )
    assert result.settings["n_splits"] == 2 and type(result.settings["random_state"]) is int
    assert _reliability(objects, fovs, n_splits=2, draws=3).curve.equals(result.curve)
    replay = _replay(objects, fovs, n_splits=np.int64(2), random_state=np.int64(0))
    assert replay.settings["n_splits"] == 2 and type(replay.settings["random_state"]) is int
    table = protocol_reliability(
        _curve({("R1", "A"): {2: 1.0}, ("R2", "A"): {2: 0.5}}),
        BY_WELL,
        threshold=0.9,
        strata=[],
        n_boot=np.int64(100),
        random_state=np.int64(0),
    ).table
    assert len(table) == 1


def test_f21_replay_refuses_a_missing_objective_before_running():
    objects, fovs = _tables(n_fovs=8)
    with pytest.raises(TypeError, match="objective must be an ObjectiveConfig"):
        _replay(objects, fovs, objective=None)


def _redundancy(objects, fovs):
    return redundancy_distance(
        objects,
        fovs,
        feature_cols=["f"],
        stage_cols=("x", "y"),
        bin_edges=[0, 150, 300, 500],
        fov_col="position_id",
    ).set_index("repeat")


def test_f22_a_constant_readout_has_no_redundancy_distance():
    objects, fovs = _tables(n_fovs=16, repeats=("R1", "R2"))
    table = _redundancy(objects.assign(f=5.0), fovs)
    assert table["redundancy_distance"].isna().all()
    assert table["reason"].str.contains("do not vary").all()


def test_f23_a_missing_value_in_one_group_leaves_the_others_measured():
    objects, fovs = _tables(n_fovs=16, repeats=("R1", "R2"))
    objects.loc[objects.index[objects["repeat"] == "R2"][0], "f"] = np.nan
    table = _redundancy(objects, fovs)
    assert pd.notna(table.loc["R1", "smallest_lag"]) and pd.isna(table.loc["R1", "reason"])
    assert "missing feature values" in table.loc["R2", "reason"]


def test_f24_one_order_phrase_one_median_rule_one_skip_rule():
    assert not hasattr(spatial, "_ORDER_WORDS") and not hasattr(spatial, "_median_beyond")
    point = _median_point(pd.Series([3, None, None], dtype="Int64"), pd.Series([12, 12, 12]))
    assert point.startswith("not within the fields available (up to 12)")
    assert _median_point(pd.Series([3, 5, None], dtype="Int64"), pd.Series([12] * 3)) == "field 5"
    objects, fovs = _tables(n_fovs=8, wells=("A1",))
    one, one_fovs = _tables(n_fovs=1, wells=("A2",))
    tables = (pd.concat([objects, one]), pd.concat([fovs, one_fovs]))
    skipped_replay = _replay(*tables).skipped["reason"].iloc[0]
    skipped_reliability = _reliability(*tables).skipped["reason"].iloc[0]
    assert skipped_replay == skipped_reliability
    assert (
        "In the recorded order"
        in order_efficiency(
            _replay(
                objects,
                fovs,
                orders=[AcquisitionOrder.recorded()],
                acquisition_order_col="order",
                stage_cols=("x", "y"),
            )
        ).summary()
    )


def test_f25_docstrings_state_the_interval_and_the_seeding():
    assert "never narrower" in protocol.__doc__ and "percentile interval" not in (
        protocol.ProtocolResult.__doc__
    )
    from morphocoverage.retrospective import reliability, replay

    assert "split_seed" in reliability.__doc__ and "split_seed" in replay.__doc__
    assert "last prefix must be evaluable" in adequacy.__doc__
