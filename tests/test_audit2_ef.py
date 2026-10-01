"""Regressions for the second audit, groups E and F: normalisation, outcomes, sufficiency,
spatial diagnostics and the retrospective criteria, reliability, protocol and order checks.

One test per finding, named after it; each fails on the code before the fix.
"""

import re

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.data_check import validate_input_tables
from morphocoverage.normalize import control_means, normalize_to_control
from morphocoverage.outcomes import RandomObjectRelativeError, RareStateDetected, RelativeMeanError
from morphocoverage.plots import plots_of_data
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    adequacy_points,
    dispersed_order,
    order_efficiency,
    protocol_reliability,
    redundancy_distance,
    replay_acquisitions,
    sampling_reliability,
    stage_travel,
)
from morphocoverage.retrospective._text import order_phrase, plural
from morphocoverage.retrospective.criteria import Preservation
from morphocoverage.retrospective.protocol import _bootstrap, _exact
from morphocoverage.retrospective.spatial import EfficiencyResult, efficiency_summary
from morphocoverage.spatial import empirical_semivariogram
from morphocoverage.statistics import bootstrap_fields
from morphocoverage.strategies import SupportNoveltyEstimator, SupportNoveltyStrategy
from morphocoverage.strategies.support_novelty import UPPER_QUANTITY
from morphocoverage.sufficiency import sufficiency_report

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)
MEAN_F = OutcomeCriterion(RelativeMeanError("f"), 0.05, "lower_is_better")


def _make(n_fovs=20, per_fov=8, empty=(), seed=0, groups=(("A", "r1"),)):
    """Objects and FOVs, one analysis group per (condition, repeat), FOVs on a 5-wide grid."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat in groups:
        for i in range(n_fovs):
            position = f"p{i:02d}"
            fovs.append(
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": position,
                    "x": float(i % 5) * 100.0,
                    "y": float(i // 5) * 100.0,
                }
            )
            if i in empty:
                continue
            shift = rng.normal(0, 1)
            objects += [
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": position,
                    "f": 10 + shift + rng.normal(0, 1),
                }
                for _ in range(per_fov)
            ]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _control_tables(control_level, treated_level):
    """Two repeats, a control and a treated condition, three fields of four objects each."""
    rows = []
    for repeat in ("r1", "r2"):
        for condition, level in (("ctrl", control_level), ("drug", treated_level)):
            for field_index in range(3):
                rows += [
                    {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "position_id": f"{repeat}_{condition}_{field_index}",
                        "x": value,
                    }
                    for value in level(repeat)
                ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


# --- E: normalisation, outcomes, sufficiency, spatial --------------------------------------


def test_e1_a_negative_control_mean_is_refused_naming_feature_and_repeat():
    objects, fovs = _control_tables(
        lambda r: [-2.0, -2.1, -1.9, -2.0], lambda r: [-1.0, -1.1, -0.9, -1.0]
    )
    with pytest.raises(ValueError, match=r"repeat=r1: 'x' is negative .*flips the sign"):
        normalize_to_control(objects, fovs, feature_cols=["x"], control={"condition": "ctrl"})


def test_e2_an_explicit_repeat_column_that_is_absent_raises_instead_of_pooling():
    objects, fovs = _control_tables(lambda r: [1.0, 2.0, 1.5, 1.5], lambda r: [3.0] * 4)
    objects, fovs = (
        objects.rename(columns={"biological_repeat": "plate"}),
        fovs.rename(columns={"biological_repeat": "plate"}),
    )
    with pytest.raises(ValueError, match="repeat_col 'Plate' is not a column"):
        control_means(
            objects, fovs, feature_cols=["x"], control={"condition": "ctrl"}, repeat_col="Plate"
        )
    # The default name may be absent (one implicit repeat), and None asks for that.
    for repeat_col in ({}, {"repeat_col": None}):
        means = control_means(
            objects, fovs, feature_cols=["x"], control={"condition": "ctrl"}, **repeat_col
        )
        assert means["biological_repeat"].tolist() == [None]
    assert control_means(
        objects, fovs, feature_cols=["x"], control={"condition": "ctrl"}, repeat_col="plate"
    )["biological_repeat"].tolist() == ["r1", "r2"]


def test_e3_a_mean_that_is_rounding_noise_counts_as_zero():
    objects, fovs = _control_tables(lambda r: [0.1, 0.2, -0.3, 0.0], lambda r: [1.0] * 4)
    assert 0 < abs(objects.loc[objects["condition"] == "ctrl", "x"].mean()) < 1e-15
    with pytest.raises(ValueError, match="'x' is zero"):
        normalize_to_control(objects, fovs, feature_cols=["x"], control={"condition": "ctrl"})
    table = pd.DataFrame({"position_id": ["p0", "p0", "p1", "p1"], "x": [0.1, 0.2, -0.3, 0.0]})
    positions = pd.DataFrame({"position_id": ["p0", "p1"]})
    full = validate_input_tables(table, fovs=positions, feature_cols=["x"])
    stop = validate_input_tables(
        table[table["position_id"] == "p0"], fovs=positions.iloc[:1], feature_cols=["x"]
    )
    assert RelativeMeanError("x").evaluate(full, stop) is None
    assert RandomObjectRelativeError("x").evaluate(full, stop) is None


def _few_acquisition(shared):
    """30 common fields and two with a far-away state: in one field each, or both in one pair."""
    rng = np.random.default_rng(0)
    rows = []
    for j in range(32):
        centre = 0.0 if j < 30 else (100.0 if shared else 100.0 * (j - 29))
        rows += [{"image_file": f"f{j:02d}", "a": v} for v in rng.normal(centre, 0.1, 10)]
    return pd.DataFrame(rows), pd.DataFrame({"image_file": [f"f{j:02d}" for j in range(32)]})


def test_e4_seen_in_few_counts_recurrent_states_only_not_the_new_ones():
    novelty = SupportNoveltyEstimator(1.0, 1, None, 2, 1)
    counts = {}
    for shared in (False, True):
        objects, fovs = _few_acquisition(shared)
        report = sufficiency_report(
            objects,
            fovs,
            feature_cols=["a"],
            strategies=[],
            objective=DISTRIBUTION,
            orders=[AcquisitionOrder.permuted(0)],
            novelty=novelty,
            few_fovs=3,
        )
        row = report.next_fov.iloc[0]
        counts[shared] = (int(row["fovs_with_new"]), int(row["fovs_with_state_in_few"]))
    # Two one-field states are new, and not "seen in few" as well.
    assert counts[False] == (2, 0)
    # One state shared by two fields is not new but seen in few.
    assert counts[True] == (0, 2)
    assert "FOVs counted as new above are not included" in report.next_fov["answer"].iloc[0]
    assert "fields counted as new are not among them" in report.summary(detail="full")


def test_e5_plot_interval_says_how_many_technical_repeats_and_warns_with_two_or_three():
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in ("c0", "c1"):
        for field_index in range(6):
            name = f"{condition}_{field_index}"
            well = f"w{field_index % 2}"
            fovs.append({"condition": condition, "technical_repeat": well, "image_file": name})
            rows += [
                {"condition": condition, "technical_repeat": well, "image_file": name, "x": v}
                for v in rng.normal(size=10)
            ]
    figures = plots_of_data(pd.DataFrame(rows), pd.DataFrame(fovs), features=["x"], n_boot=20)
    title = figures["x"][0].axes[0].get_title(loc="left")
    assert "whole technical repeats with their fields as acquired (2 per cluster" in title
    assert "with only 2-3 technical repeats a cluster's interval is not reliable" in title


def test_e6_no_fields_beyond_enough_when_the_acquired_fields_are_not_enough():
    rng = np.random.default_rng(0)
    rows = []
    for j in range(55):
        centre = 0.0 if j < 40 else 100.0 * (j - 39)  # the last 15 fields each hold a new state
        values = rng.normal(centre, 0.1, 10)
        rows += [{"image_file": f"f{j:02d}", "order": j, "a": v} for v in values]
    objects = pd.DataFrame(rows)
    fovs = objects[["image_file", "order"]].drop_duplicates().reset_index(drop=True)
    rule = SupportNoveltyStrategy(1.0, 0.25, 1, None, 2, 1, UPPER_QUANTITY, 0.95)
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a"],
        strategies=[rule],
        objective=DISTRIBUTION,
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="order",
    )
    row = report.enough.iloc[0]
    assert row["sampled_enough"] == "no" and row["enough_after_fovs"] == 11
    assert pd.isna(row["fovs_beyond_enough"])


def _two_groups():
    rng = np.random.default_rng(0)
    rows = [
        {"condition": c, "biological_repeat": r, "image_file": f"{c}{r}_{j}", "a": v}
        for c in ("A", "B")
        for r in (1, 2)
        for j in range(12)
        for v in rng.normal(10, 1, 10)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_e7_a_criterion_passed_twice_is_refused_clearly():
    objects, fovs = _two_groups()
    orders = [AcquisitionOrder.permuted(0)]
    with pytest.raises(ValueError, match="passed twice in strategies"):
        sufficiency_report(
            objects,
            fovs,
            feature_cols=["a"],
            strategies=[FixedFOVStrategy(3), FixedFOVStrategy(3)],
            objective=POPULATION,
            orders=orders,
        )
    with pytest.raises(ValueError, match="the one novelty_target adds"):
        sufficiency_report(
            objects,
            fovs,
            feature_cols=["a"],
            strategies=[SupportNoveltyStrategy(0.5, 0.25, 1, None, 2, 1, UPPER_QUANTITY, 0.95)],
            objective=POPULATION,
            orders=orders,
            novelty=SupportNoveltyEstimator(0.5, 1, None, 2, 1),
            novelty_target=0.25,
        )


def test_e9_one_replayed_order_is_singular():
    objects, fovs = _two_groups()
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a"],
        strategies=[FixedFOVStrategy(3)],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    answers = " ".join(report.enough["answer"])
    assert "in the random order, first met after" in answers and "1 orders" not in answers
    for detail in ("short", "full"):
        assert "1 orders" not in report.summary(detail=detail)
    assert "1 random order: median" in report.summary(detail="full")


def test_e10_the_semivariogram_refuses_infinite_coordinates():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [np.inf, 0.0]])
    with pytest.raises(ValueError, match="must be finite"):
        empirical_semivariogram(np.array([1.0, 2.0, 3.0]), coords, [0, 2, 4])


# --- F: retrospective criteria, reliability, protocol and order checks ---------------------


def test_f1_an_empty_subset_is_undefined_when_the_reference_itself_is_undefined():
    objects, fovs = _make(n_fovs=4, empty=(0,))
    empty = validate_input_tables(
        objects.iloc[:0], fovs=fovs.iloc[:1], feature_cols=["f"], group_cols=[]
    )
    reference = validate_input_tables(
        objects[objects["position_id"] != "p00"], fovs=fovs.iloc[1:], feature_cols=["f"]
    )
    assert MEAN_F.assess(reference, empty) == (Preservation.NOT_PRESERVED, None)
    missing = objects[objects["position_id"] != "p00"].copy()
    missing.iloc[0, missing.columns.get_loc("f")] = np.nan
    undefined = validate_input_tables(missing, fovs=fovs.iloc[1:], feature_cols=["f"])
    assert MEAN_F.assess(undefined, empty) == (Preservation.UNDEFINED, None)
    # A rare state the reference does not show cannot be judged either.
    objects["rare"] = 0
    rare = OutcomeCriterion(RareStateDetected("rare", 1), 1, "higher_is_better")
    no_state = validate_input_tables(objects, fovs=fovs, feature_cols=["f"])
    assert rare.assess(no_state, empty)[0] is Preservation.UNDEFINED


def test_f2_repeated_hold_outs_are_counted_as_distinct_and_warned_about():
    objects, fovs = _make(n_fovs=20)
    fovs["block"] = ["b0"] * 7 + ["b1"] * 7 + ["b2"] * 6
    result = sampling_reliability(
        objects,
        fovs,
        feature_cols=["f"],
        criteria=[MEAN_F],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=10,
        draws=5,
        block_col="block",
    )
    assert result.ceiling["distinct_holdouts"].tolist() == [3]
    text = " ".join(result.summary(detail="full").split())
    assert "in 10 splits over 3 distinct hold-outs" in text
    assert "Warning: the 10 splits repeat hold-outs" in text
    short = " ".join(result.summary().split())
    assert "in 10 splits over 3 distinct hold-outs" in short
    assert "10 splits repeat hold-outs" in short


def test_f3_a_group_without_two_fields_gets_a_row_with_a_reason():
    good = _make(n_fovs=20)
    bad = _make(n_fovs=3, empty=(1, 2), groups=(("B", "r1"),), seed=1)
    objects = pd.concat([good[0], bad[0]], ignore_index=True)
    fovs = pd.concat([good[1], bad[1]], ignore_index=True)
    table = redundancy_distance(
        objects, fovs, feature_cols=["f"], stage_cols=("x", "y"), bin_edges=[0, 150, 300, 600]
    ).set_index("condition")
    assert pd.isna(table.loc["A", "reason"]) and table.loc["A", "smallest_lag"] > 0
    assert table.loc["B", "reason"].startswith("only 1 field with observations")
    assert pd.isna(table.loc["B", "redundancy_distance"])
    far = redundancy_distance(
        *good, feature_cols=["f"], stage_cols=("x", "y"), bin_edges=[5000, 6000]
    )
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


def test_f6_a_criterion_never_judged_is_not_called_not_reached():
    objects, fovs = _make(n_fovs=16)
    objects.loc[objects.groupby("position_id").head(1).index, "f"] = np.nan
    replay = replay_acquisitions(
        objects,
        fovs,
        feature_cols=["f"],
        strategies=[FixedFOVStrategy(5)],
        criteria=[MEAN_F],
        objective=ObjectiveConfig(),
        reservoir_fraction=0.3,
        n_splits=2,
        orders=[AcquisitionOrder.permuted(0)],
    )
    adequacy = adequacy_points(replay)
    assert (adequacy.realized["judged_fovs"] == 0).all()
    text = adequacy.summary()
    assert "could not be judged at any field count" in text
    assert "first preserved not within" not in text


def _curve(preserved, stream=12, unit="plate"):
    """A reliability result from {(plate, well): {k: preserved}}, one criterion."""
    rows = [
        {
            "condition": "ctrl",
            unit: plate,
            "well": well,
            "criterion": "mean",
            "fovs": k,
            "preserved": value,
            "stream_fovs": stream,
        }
        for (plate, well), by_k in preserved.items()
        for k, value in by_k.items()
    ]
    return ReliabilityResult(pd.DataFrame(rows), pd.DataFrame(), ("condition", unit, "well"))


def test_f7_a_boundary_share_gets_an_exact_interval_over_units():
    every = {(f"P{i}", "A1"): {5: 0.99} for i in range(5)}
    table = protocol_reliability(
        _curve(every), SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    ).table
    assert table["share"].iloc[0] == 1.0
    assert table["share_low"].iloc[0] == pytest.approx(0.478, abs=1e-3)
    assert table["share_high"].iloc[0] == 1.0
    block = pd.DataFrame({"unit": list("aabbbcd"), "adequate": [1, 0, 1, 1, 0, 1, 0]})
    units = [block.loc[block["unit"] == u, "adequate"].to_numpy(float) for u in "abcd"]
    shared = bootstrap_fields(units, np.mean, n_boot=500, rng=np.random.default_rng(1))
    # Audit 3 (F-5): never narrower than the exact interval over the units, within 0-1.
    exact = _exact(4 / 7, 4)
    expected = (max(0.0, min(shared[1], exact[0])), min(1.0, max(shared[2], exact[1])))
    assert _bootstrap(block, 500, np.random.default_rng(1)) == pytest.approx(expected)


def test_f8_a_low_ceiling_is_not_called_unjudgeable():
    objects, fovs = [], []
    for plate in range(2):
        for well in range(2):
            o, f = _make(n_fovs=8, seed=10 * plate + well)
            for table in (o, f):
                table["plate"], table["well"] = f"b{plate}", f"w{well}"
            objects.append(o)
            fovs.append(f)
    criterion = OutcomeCriterion(RelativeMeanError("f"), 0.01, "lower_is_better")
    reliability = sampling_reliability(
        pd.concat(objects, ignore_index=True),
        pd.concat(fovs, ignore_index=True),
        feature_cols=["f"],
        criteria=[criterion],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=4,
        draws=5,
        group_cols=["condition", "biological_repeat", "plate", "well"],
    )
    result = protocol_reliability(
        reliability,
        SamplingHierarchy("position_id", "well", "plate"),
        threshold=0.9,
        strata=["condition", "biological_repeat"],
    )
    assert result.difficulty["ceiling"].notna().all()
    text = " ".join(result.summary(detail="full").split())
    short = " ".join(result.summary().split())
    assert "could not judge" not in text and "could not judge" not in short
    assert "even every available field together was enough in fewer than 90%" in text
    assert "even every available field together was enough in fewer than 90%" in short


def test_f9_a_left_out_unit_with_too_few_fields_is_named_not_counted_as_zero_of_zero():
    curve = {
        ("P1", "A1"): {5: 0.5, 10: 0.95},
        ("P2", "A1"): {5: 0.5, 10: 0.95},
        ("P3", "A1"): {5: 0.5, 10: None},  # too few fields to be judged at 10
    }
    result = protocol_reliability(
        _curve(curve), SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    )
    left_out = result.validation.set_index("left_out")
    assert left_out.loc["P3", "chosen_fovs"] == 10 and left_out.loc["P3", "eligible"] == 0
    text = " ".join(result.summary(detail="full").split())
    assert "Not checked on 1 plate whose samples had fewer fields than" in text
    assert "P3 (chosen 10)" in text
    short = " ".join(result.summary().split())
    assert "not checked on 1 plate with fewer fields" in short and "P3 (chosen 10)" in short


def test_f10_infinite_stage_coordinates_are_refused():
    xy = np.array([[0.0, 0.0], [np.inf, 0.0], [10.0, 0.0]])
    for call in (lambda: dispersed_order(xy, 0), lambda: stage_travel(xy)):
        with pytest.raises(ValueError, match="must be finite"):
            call()
    objects, fovs = _make(n_fovs=12)
    fovs.loc[3, "x"] = np.inf
    with pytest.raises(ValueError, match="stage coordinates must be finite"):
        replay_acquisitions(
            objects,
            fovs,
            feature_cols=["f"],
            strategies=[FixedFOVStrategy(3)],
            criteria=[MEAN_F],
            objective=ObjectiveConfig(),
            reservoir_fraction=0.3,
            n_splits=2,
            orders=[AcquisitionOrder.dispersed(0)],
            stage_cols=("x", "y"),
        )


def test_f11_numpy_integer_fov_counts_are_accepted():
    objects, fovs = _make(n_fovs=12)
    settings = {"feature_cols": ["f"], "criteria": [MEAN_F], "reservoir_fraction": 0.3}
    settings["n_splits"] = 2
    plain = sampling_reliability(objects, fovs, fov_counts=[1, 2, 3], draws=3, **settings)
    numpy = sampling_reliability(objects, fovs, fov_counts=np.arange(1, 4), draws=3, **settings)
    pd.testing.assert_frame_equal(plain.curve, numpy.curve)
    assert numpy.settings["fov_counts"] == [1, 2, 3]
    assert all(type(k) is int for k in numpy.settings["fov_counts"])


def test_f12_a_bare_string_is_not_split_into_letters():
    with pytest.raises(TypeError, match="technical_levels must be a sequence"):
        SamplingHierarchy("position_id", "well", "biological_repeat", technical_levels="plate")
    with pytest.raises(TypeError, match="strata must be a sequence"):
        protocol_reliability(
            _curve({("P1", "A1"): {5: 0.9}}),
            SamplingHierarchy("image_file", "well", "plate"),
            threshold=0.9,
            strata="condition",
        )


def test_f13_unit_names_take_their_english_plural():
    assert [plural(w) for w in ("batch", "well", "repeat", "technical repeat", "assay")] == [
        "batches",
        "wells",
        "repeats",
        "technical repeats",
        "assays",
    ]
    assert plural("colony") == "colonies"
    curve = {(f"b{i}", "A1"): {5: 0.95} for i in range(2)}
    text = protocol_reliability(
        _curve(curve, unit="batch"),
        SamplingHierarchy("image_file", "well", "batch"),
        threshold=0.9,
        strata=[],
    ).summary()
    assert "batches" in text and "batchs" not in text


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


def test_f15_a_single_field_group_is_skipped_with_a_reason_and_one_split_is_singular():
    a = _make(n_fovs=12)
    b = _make(n_fovs=1, groups=(("B", "r1"),), seed=2)
    result = sampling_reliability(
        pd.concat([a[0], b[0]]),
        pd.concat([a[1], b[1]]),
        feature_cols=["f"],
        criteria=[MEAN_F],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=1,
        draws=3,
    )
    assert set(result.curve["condition"]) == {"A"}
    assert result.skipped[["condition", "fovs"]].values.tolist() == [["B", 1]]
    for detail in ("short", "full"):
        text = " ".join(result.summary(detail=detail).split())
        assert "B · r1: skipped, only 1 field: at least two are needed" in text
        assert "in 1 split over 1 distinct hold-out;" in text and "1 hold-outs" not in text


def test_f16_order_phrases_are_defined_once():
    from morphocoverage.retrospective import adequacy, calibration

    assert not hasattr(adequacy, "_ORDER_WORDS") and not hasattr(calibration, "_ORDER_WORDS")
    assert order_phrase("permuted") == "in random orders"
    assert order_phrase("recorded", capital=True) == "In the recorded order"
    assert order_phrase("custom") == "custom"
