"""Tests for protocol-level reliability across samples (roadmap Step 4)."""

import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage.bounds import interval
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    protocol_reliability,
    sampling_reliability,
)

HIERARCHY = SamplingHierarchy("image_file", "well", "plate")


def _curve(preserved, stream=12, criterion="mean"):
    """A reliability result from {(plate, well): {k: preserved}}."""
    rows = []
    for (plate, well), by_k in preserved.items():
        for k, value in by_k.items():
            rows.append(
                {
                    "condition": "ctrl",
                    "plate": plate,
                    "well": well,
                    "criterion": criterion,
                    "fovs": k,
                    "preserved": value,
                    "stream_fovs": stream,
                }
            )
    curve = pd.DataFrame(rows)
    return ReliabilityResult(
        curve, pd.DataFrame(), ("condition", "plate", "well"), {"labels": {criterion: "the mean"}}
    )


PLATES = {
    ("P1", "A1"): {5: 0.95, 10: 0.99},
    ("P1", "A2"): {5: 0.80, 10: 0.95},
    ("P2", "A1"): {5: 0.92, 10: 0.97},
    ("P2", "B3"): {5: 0.50, 10: None},  # stream too short to judge 10 fields
}


def _protocol(preserved=PLATES, **kwargs):
    settings = {"threshold": 0.9, "strata": ["condition"]}
    settings.update(kwargs)
    return protocol_reliability(_curve(preserved), HIERARCHY, **settings)


def test_shares_have_explicit_denominators():
    table = _protocol().table.set_index("fovs")
    assert table.loc[5, ["eligible", "adequate", "not_evaluated"]].tolist() == [4, 2, 0]
    assert table.loc[10, ["eligible", "adequate", "not_evaluated"]].tolist() == [3, 3, 1]
    assert table.loc[10, "share"] == 1.0
    assert (table["units"] == 2).all()


def test_wells_reused_across_plates_are_different_samples():
    difficulty = _protocol().difficulty
    assert len(difficulty) == 4
    needed = difficulty.set_index(["plate", "well"])["fields_needed"]
    assert needed[("P1", "A1")] == 5 and needed[("P1", "A2")] == 10
    assert pd.isna(needed[("P2", "B3")])


def test_per_unit_counts_and_the_named_difficult_sample():
    result = _protocol()
    per = result.per_unit[result.per_unit["fovs"] == 5].set_index("unit")
    assert per.loc["P1", ["eligible", "adequate"]].tolist() == [2, 1]
    text = result.summary()
    assert "enough for 2 of 4 samples" in text
    assert "B3, P2 (more than 5)" in text
    assert "could not be judged at this depth" in text
    assert "Uncertainty is large with only 2 plates" in text
    for word in ("reservoir", "criterion", "bootstrap"):
        assert word not in text


def test_interval_counts_units_not_draws():
    # Duplicating a unit's samples adds samples, not units: every sample adequate in
    # 3 units gives the exact Clopper-Pearson interval of 3 of 3, not of 6 of 6
    # (and not the zero-width 100%-100% a resample of units would give).
    same = {(p, w): {5: 0.95} for p in ("P1", "P2", "P3") for w in ("A1", "A2")}
    row = _protocol(same).table.iloc[0]
    assert (row["share_low"], row["share_high"]) == pytest.approx(interval(3, 3, 0.95))
    # One unit gives no interval at all, rather than a zero-width one that looks exact.
    one_unit = _protocol({k: v for k, v in PLATES.items() if k[0] == "P1"})
    assert one_unit.table["share_low"].isna().all()
    assert "plausible range" not in one_unit.summary()


def test_short_and_never_judged_samples_are_counted_not_dropped():
    wells = dict(PLATES)
    wells[("P2", "C9")] = {5: None, 10: None}  # its stream was shorter than every count
    result = _protocol(wells)
    table = result.table.set_index("fovs")
    assert table.loc[5, "not_evaluated"] == 1 and table.loc[10, "not_evaluated"] == 2
    text = result.summary()
    assert "5 samples in 2 plates" in text
    assert "Could not be judged at any depth: C9, P2." in text
    assert "more than nan" not in text


def test_adding_a_stratum_does_not_change_another_strata_interval():
    alone = _protocol(PLATES).table
    curve = _curve(PLATES).curve
    extra = curve.assign(condition="drug", preserved=curve["preserved"].fillna(0.5) * 0.9)
    both = protocol_reliability(
        ReliabilityResult(pd.concat([curve, extra]), pd.DataFrame(), _curve(PLATES).group_cols),
        HIERARCHY,
        threshold=0.9,
        strata=["condition"],
    ).table
    ctrl = both[both["condition"] == "ctrl"].reset_index(drop=True)
    pd.testing.assert_frame_equal(ctrl, alone.reset_index(drop=True))


def test_the_protocol_number_must_hold_from_there_on():
    dip = {
        ("P1", "A1"): {5: 0.95, 10: 0.5, 20: 0.95},
        ("P1", "A2"): {5: 0.95, 10: 0.5, 20: 0.95},
        ("P2", "A1"): {5: 0.95, 10: 0.95, 20: 0.95},
    }
    check = _protocol(dip).validation.set_index("left_out")
    assert check.loc["P2", "chosen_fovs"] == 20


def test_the_interval_does_not_narrow_with_more_draws_or_splits():
    built = SCENARIOS["condition_repeat_groups"]()
    criterion = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better")
    hierarchy = SamplingHierarchy("image_file", "repeat")

    def run(draws, splits):
        per_sample = sampling_reliability(
            built.objects,
            built.fovs,
            feature_cols=list(FEATURE_COLS),
            criteria=[criterion],
            fov_counts=[1, 3, 5],
            reservoir_fraction=0.3,
            n_splits=splits,
            draws=draws,
        )
        result = protocol_reliability(per_sample, hierarchy, threshold=0.9, strata=["condition"])
        flags = result.difficulty.set_index(["condition", "repeat"])
        return result.table.set_index(["condition", "fovs"]), flags

    few, few_flags = run(4, 2)
    many, many_flags = run(30, 6)
    for key in few.index:
        if few.loc[key, "adequate"] == many.loc[key, "adequate"]:
            assert few.loc[key, "share_low"] == many.loc[key, "share_low"]
            assert few.loc[key, "share_high"] == many.loc[key, "share_high"]
    assert set(few_flags.index) == set(many_flags.index)


def test_leave_one_unit_out_chooses_on_the_others():
    hard_plate = {
        ("P1", "A1"): {5: 0.95, 10: 0.99},
        ("P1", "A2"): {5: 0.95, 10: 0.99},
        ("P2", "A1"): {5: 0.40, 10: 0.95},
        ("P2", "A2"): {5: 0.40, 10: 0.95},
    }
    check = _protocol(hard_plate).validation.set_index("left_out")
    # Chosen on P1 alone, 5 fields; P2 then fails at 5.
    assert check.loc["P2", "chosen_fovs"] == 5 and check.loc["P2", "adequate"] == 0
    # Chosen on P2 alone, 10 fields; P1 is served.
    assert check.loc["P1", "chosen_fovs"] == 10 and check.loc["P1", "adequate"] == 2


def test_pooling_and_sample_identity_must_be_declared():
    with pytest.raises(ValueError, match="not analysis-group columns"):
        protocol_reliability(
            _curve(PLATES), SamplingHierarchy("image_file", "site"), threshold=0.9, strata=[]
        )
    with pytest.raises(ValueError, match="above the sample"):
        _protocol(strata=["well"])
    with pytest.raises(ValueError, match="share"):
        _protocol(threshold=0)
    with pytest.raises(TypeError):
        protocol_reliability(PLATES, HIERARCHY, threshold=0.9, strata=[])
    # Without condition as a stratum or sample level, conditions would be pooled
    # into one sample: refused.
    two = _curve(PLATES)
    doubled = pd.concat([two.curve, two.curve.assign(condition="drug")])
    with pytest.raises(ValueError, match="several analysis groups map to one sample"):
        protocol_reliability(
            ReliabilityResult(doubled, pd.DataFrame(), two.group_cols),
            HIERARCHY,
            threshold=0.9,
            strata=[],
        )
    pooled = protocol_reliability(two, HIERARCHY, threshold=0.9, strata=[])
    assert "all samples" in pooled.summary()
