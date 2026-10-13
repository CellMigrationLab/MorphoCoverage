"""Tests of ``retrospective.reliability``: sampling reliability and its words."""

from __future__ import annotations

import numpy as np
import pandas as pd
from _helpers_benchmark import MEAN_F
from _helpers_retrospective import (
    MEAN,
    _curve_with_ceiling,
    _f_reliability,
    _full,
    _make,
    _ninety_percent_result,
    _repeat_curve,
    _scenario_reliability,
    _tables,
)
from _helpers_strategies import _flat
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage._wording import percent
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    protocol_reliability,
    sampling_reliability,
)


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


def test_f11_numpy_integer_fov_counts_are_accepted():
    objects, fovs = _make(n_fovs=12)
    settings = {"feature_cols": ["f"], "criteria": [MEAN_F], "reservoir_fraction": 0.3}
    settings["n_splits"] = 2
    plain = sampling_reliability(objects, fovs, fov_counts=[1, 2, 3], draws=3, **settings)
    numpy = sampling_reliability(objects, fovs, fov_counts=np.arange(1, 4), draws=3, **settings)
    pd.testing.assert_frame_equal(plain.curve, numpy.curve)
    assert numpy.settings["fov_counts"] == [1, 2, 3]
    assert all(type(k) is int for k in numpy.settings["fov_counts"])


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
        assert "B · repeat r1: skipped, only 1 field: at least two are needed" in text
        assert "in 1 split over 1 distinct hold-out;" in text and "1 hold-outs" not in text


def test_f15_protocol_wording_says_batches_and_at_least():
    curve = _repeat_curve({("B1", "A1"): {4: 0.95}, ("B1", "A2"): {4: 0.5}})
    curve = ReliabilityResult(
        curve.curve.rename(columns={"biological_repeat": "batch"}),
        pd.DataFrame(),
        ("condition", "batch", "well"),
        curve.settings,
    )
    text = protocol_reliability(
        curve, SamplingHierarchy("image_file", "well", "batch"), threshold=0.9, strata=["condition"]
    ).summary()
    assert "no number can be chosen on other batches" in text and "batchs" not in text
    assert "counts when at least 90% of random acquisitions were enough" in " ".join(text.split())


def test_f17_identical_split_shares_print_no_rounding_noise():
    result = _ninety_percent_result()
    text = result.summary(detail="full")
    # With the Monte Carlo error of the share (audit 4, P9-2), also without noise.
    assert "Monte Carlo SE" in text and "± across hold-outs" in text
    rows = [line.split()[1:4] for line in text.splitlines() if line.split()[:1] == ["4"]]
    assert rows and all(row == ["90%", "0%", "0%"] for row in rows)
    assert "e-" not in text and "e-" not in result.summary() and "90%" in result.summary()


def test_f18_repeated_hold_outs_are_counted_as_splits():
    objects, fovs = _tables(scale=2.0)
    fovs["block"] = ["b0", "b1", "b2"] * 4
    result = _f_reliability(
        objects,
        fovs,
        fov_counts=[2, 8],
        reservoir_fraction=0.3,
        n_splits=10,
        draws=5,
        block_col="block",
    )
    assert result.ceiling["distinct_holdouts"].iloc[0] == 3
    text = " ".join(result.summary(detail="full").split())
    assert "across splits" in text and "one acquisition per split" in text
    assert "were enough in 40% of splits" in text or "% of splits:" in text
    assert "across hold-outs" not in text and "% of hold-outs" not in text
    short = " ".join(result.summary().split())
    assert "10 splits repeat hold-outs" in short and "across hold-outs" not in short


def test_f19_unequal_streams_state_the_range_of_available_fields():
    objects, fovs = _tables(scale=2.0)
    fovs["block"] = ["big"] * 6 + ["b1", "b1", "b2", "b2", "b3", "b3"]
    result = _f_reliability(
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
    text = " ".join(result.summary(detail="full").split())
    assert "All 6 to 10 available fields together" in text
    assert "nothing is said beyond 10 fields (6 in some splits)" in text
    short = " ".join(result.summary().split())
    assert "all 6 to 10" in short and "nothing is said beyond 10 fields (6 in some splits)" in short


def test_p3_12_reliability_says_its_curve_is_conservative_by_construction():
    built = SCENARIOS["homogeneous"]()
    result = sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[MEAN],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=3,
        draws=5,
    )
    assert "conservative by construction" in result.summary()


def test_f16_a_technical_repeat_is_named_as_such_in_the_protocol():
    from morphocoverage.retrospective import ReliabilityResult

    rows = [
        {
            "condition": "ctrl",
            "biological_repeat": repeat,
            "technical_repeat": well,
            "criterion": "mean",
            "fovs": 5,
            "preserved": 0.95,
            "stream_fovs": 12,
        }
        for repeat, well in (("R1", "A1"), ("R1", "A2"), ("R2", "A1"))
    ]
    curve = ReliabilityResult(
        pd.DataFrame(rows),
        pd.DataFrame(),
        ("condition", "biological_repeat", "technical_repeat"),
        {"labels": {"mean": "the mean"}},
    )
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    result = protocol_reliability(curve, hierarchy, threshold=0.9, strata=["condition"])
    for text in (_full(result), " ".join(result.summary().split())):
        assert (
            "ields per technical repeat" in text
            and "3 technical repeats in 2 biological repeats" in text
        )
        assert "technical_repeat" not in text


MEAN_A = OutcomeCriterion(
    RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean of feature_a within 2%"
)


def _reliability(name="homogeneous", counts=(2, 5)):
    built = SCENARIOS[name]()
    return sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[MEAN_A],
        fov_counts=list(counts),
        reservoir_fraction=0.3,
        n_splits=3,
        draws=5,
    )


def test_a_curve_rebuilt_without_its_ceiling_says_so():
    result = _reliability()
    assert "the curve cannot be read above this" in result.summary()
    rebuilt = ReliabilityResult(result.curve, pd.DataFrame(), result.group_cols, result.settings)
    text = rebuilt.summary()
    assert "the curve cannot be read above this" not in text
    assert "was not computed" in text and "capped by them rather than by the fields" in text


def test_protocol_without_a_ceiling_table_keeps_the_old_wording():
    per_sample = _curve_with_ceiling({"A1": {5: 0.3, 18: 0.3}}, {"A1": 0.3})
    bare = ReliabilityResult(
        per_sample.curve, pd.DataFrame(), per_sample.group_cols, per_sample.settings
    )
    result = protocol_reliability(
        bare, SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    )
    assert result.difficulty["ceiling_below_threshold"].isna().all()
    full = " ".join(result.summary(detail="full").split())
    assert "Needs more fields than could be judged: A1 · P1 (more than 18)." in full
    assert "needed more fields than could be judged (A1 · P1 (more than 18))" in " ".join(
        result.summary().split()
    )


def test_the_whole_stream_point_is_labelled_as_one_acquisition():
    result = _reliability(counts=(2, 14))
    stream = int(result.curve["stream_fovs"].max())
    text = result.summary(detail="full")
    cells = [line.split() for line in text.splitlines()]
    rows = {c[0]: " ".join(c) for c in cells if len(c) > 1 and c[1].endswith("%")}
    if stream == 14:
        assert "all fields: one acquisition per hold-out" in rows["14"]
        assert "all 14" in result.summary()  # the short form's last row: the whole stream
    # The 2-field row carries its share, Monte Carlo SE and spread across hold-outs.
    assert "± across hold-outs" in text and len(rows["2"].split()) >= 4


def test_partly_judged_samples_are_named_in_the_protocol():
    curve = pd.DataFrame(
        [
            {
                "condition": "c",
                "plate": "P1",
                "well": "A",
                "criterion": "rare",
                "fovs": 5,
                "preserved": 0.95,
                "stream_fovs": 18,
                "splits": 10,
                "defined_splits": 3,
            },
            {
                "condition": "c",
                "plate": "P1",
                "well": "B",
                "criterion": "rare",
                "fovs": 5,
                "preserved": 0.95,
                "stream_fovs": 18,
                "splits": 10,
                "defined_splits": 10,
            },
        ]
    )
    per_sample = ReliabilityResult(curve, pd.DataFrame(), ("condition", "plate", "well"), {})
    result = protocol_reliability(
        per_sample, SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    )
    assert result.table["partly_judged"].tolist() == [1]
    assert "1 judged on only some of the hold-outs" in result.summary(detail="full")
    assert "1 judged on only some of the hold-outs" in " ".join(result.summary().split())


def test_reliability_short_is_one_table_across_groups_and_full_keeps_every_share():
    result = _scenario_reliability()
    short, full = result.summary(), result.summary(detail="full")
    groups = result.curve[list(result.group_cols)].drop_duplicates()
    assert f"the range over {len(groups)} groups" in _flat(short)
    assert "condition=" not in short and "condition=" not in full
    # Every caveat once in the short form, and still in the full one.
    for caveat in ("random draws", "conservative by construction", "cannot be read above"):
        assert _flat(short).count(caveat) == 1
        assert caveat in _flat(full)
    # They hold for every curve: standing assumptions, listed once, never marked.
    standing = _flat(short.split("Assumptions:")[1])
    for caveat in ("random draws", "conservative by construction", "cannot be read above"):
        assert caveat in standing
    assert short.count("Assumptions:") == 1 and full.count("Assumptions:") == 1
    # Every share of the curve, and every Monte Carlo SE, is printed in the full form.
    for row in result.curve.itertuples(index=False):
        assert percent(row.preserved) in full
    assert "Monte Carlo SE" in full and "Judged on" in full
