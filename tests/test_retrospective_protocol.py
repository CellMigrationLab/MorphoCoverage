"""Tests for protocol-level reliability across samples (P2 of docs/optimise/optimise.md)."""

import pandas as pd
import pytest
from _helpers_retrospective import HIERARCHY, _curve
from scenarios import FEATURE_COLS, SCENARIOS
from scipy import stats

from morphocoverage.bounds import interval
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    protocol,
    protocol_reliability,
    sampling_reliability,
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
    text = result.summary(detail="full")
    assert "2 of 4 samples" in text  # at 5 fields, in the full form's table
    assert "B3 · P2 (more than 5)" in text
    assert "could not be judged at this depth" in text
    assert "With 2 plates the field count is a guide" in text
    short = result.summary()
    assert "B3 · P2 (more than 5)" in short and "With 2 plates the field count is a guide" in short
    for word in ("reservoir", "criterion", "bootstrap"):
        assert word not in text and word not in short


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
    text = " ".join(result.summary(detail="full").split())
    assert "5 samples in 2 plates" in text
    assert "Could not be judged at any depth: C9 · P2." in text
    short = " ".join(result.summary().split())
    assert "5 samples in 2 plates" in short and "could not be judged at any depth: C9 · P2" in short
    assert "more than nan" not in text and "more than nan" not in short


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
    hierarchy = SamplingHierarchy("image_file", "biological_repeat")

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
        flags = result.difficulty.set_index(["condition", "biological_repeat"])
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
    # #250: the counts chosen differ by the plate left out, and the text says them.
    full = " ".join(_protocol(hard_plate).summary("full").split())
    assert "(5 to 10 fields, by the plate left out)" in full
    assert "2 of 4 (at 5 to 10 fields)" in " ".join(_protocol(hard_plate).summary().split())
    same = {key: {5: 0.95, 10: 0.99} for key in hard_plate}
    assert "on the other plates (5 fields) and" in " ".join(_protocol(same).summary("full").split())


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


def test_a_protocol_with_no_sample_curve_says_it_is_not_assessed():
    """#248: every sample skipped upstream gives no row, and an explicit answer, not ''."""
    curve = _curve(PLATES)
    empty = ReliabilityResult(curve.curve.iloc[:0], pd.DataFrame(), curve.group_cols, {})
    result = protocol_reliability(empty, HIERARCHY, threshold=0.9, strata=["condition"])
    assert result.table.empty
    for detail in ("short", "full"):
        assert result.summary(detail) == "Not assessed: no sample had a reliability curve to count."
    assert result.summary() != _protocol().summary()


def _units_curve(adequate_per_unit):
    """A one-condition curve from [(samples, adequate samples), ...], one entry per plate."""
    preserved = {}
    for unit, (samples, adequate) in enumerate(adequate_per_unit):
        for sample in range(samples):
            preserved[(f"P{unit}", f"W{sample}")] = {5: 0.95 if sample < adequate else 0.5}
    return preserved


def _exact_reference(successes, trials):
    ci = stats.binomtest(successes, trials).proportion_ci(confidence_level=0.95, method="exact")
    return ci.low, ci.high


@pytest.mark.parametrize("n_units", [2, 3, 5, 10, 12])
@pytest.mark.parametrize("sizes", ["equal", "unequal"])
def test_issue_95_the_exact_floor_counts_whole_units(n_units, sizes, monkeypatch):
    seen = []
    exact = protocol._exact
    monkeypatch.setattr(
        protocol, "_exact", lambda k, n: seen.append((k, n)) or exact(k, n), raising=True
    )
    samples = [3] * n_units if sizes == "equal" else [1 + 7 * (u % 3) for u in range(n_units)]
    # None, every and, in between, one unit whole and all but one sample of the others.
    for adequate in ([0] * n_units, list(samples), [samples[0]] + [s - 1 for s in samples[1:]]):
        # A unit is served when at least protocol_share (0.9) of its samples are adequate.
        served = sum(a / s >= 0.9 for a, s in zip(adequate, samples, strict=True))
        row = _protocol(_units_curve(list(zip(samples, adequate, strict=True)))).table.iloc[0]
        low, high = _exact_reference(served, n_units)
        if row["share"] in (0.0, 1.0):
            assert (row["share_low"], row["share_high"]) == pytest.approx((low, high))
        else:
            assert row["share_low"] <= low + 1e-12 and row["share_high"] >= high - 1e-12
        assert row["share_low"] <= row["share"] <= row["share_high"]
    # Never a fractional pseudo-count: whole successes of whole units only.
    assert seen and all(isinstance(k, int) and isinstance(n, int) for k, n in seen)
    assert all(n == n_units for _, n in seen)


def test_issue_95_more_samples_in_a_unit_do_not_narrow_the_interval():
    for adequate in (lambda s: s, lambda s: 0):
        few = _protocol(_units_curve([(1, adequate(1))] * 3)).table.iloc[0]
        many = _protocol(_units_curve([(20, adequate(20))] * 3)).table.iloc[0]
        assert (few["share_low"], few["share_high"]) == pytest.approx(
            (many["share_low"], many["share_high"])
        )
    # A unit with many samples, 85% adequate, is one unit not served, not 17 successes.
    row = _protocol(_units_curve([(20, 17), (1, 1), (1, 1)])).table.iloc[0]
    assert row["share"] == pytest.approx(19 / 22)
    assert row["share_low"] <= _exact_reference(2, 3)[0] + 1e-12 < _exact_reference(3, 3)[0]
    # Width is driven by the units: all adequate, more units narrow it, more samples do not.
    lows = [_protocol(_units_curve([(4, 4)] * n)).table.iloc[0]["share_low"] for n in (2, 3, 5, 10)]
    assert lows == sorted(lows) and lows[0] < lows[-1]
    assert lows == pytest.approx([_exact_reference(n, n)[0] for n in (2, 3, 5, 10)])
