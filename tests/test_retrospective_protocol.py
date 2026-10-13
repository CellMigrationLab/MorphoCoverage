"""Tests for protocol-level reliability across samples (P2 of docs/optimise/optimise.md)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    BY_WELL,
    HIERARCHY,
    RARE,
    _curve,
    _curve_with_ceiling,
    _f_reliability,
    _f_replay,
    _full,
    _ninety_percent_result,
    _plate_curve,
    _plate_protocol,
    _repeat_curve,
    _tables,
    _well_values,
)
from _helpers_strategies import _flat
from _helpers_workflows import QUICK, _setup, _well_acquisition
from scenarios import FEATURE_COLS, SCENARIOS
from scipy import stats

import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage.bounds import interval
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    adequacy,
    protocol,
    protocol_reliability,
    protocol_text,
    sampling_reliability,
)
from morphocoverage.retrospective.protocol import _bootstrap, _exact
from morphocoverage.statistics import bootstrap_fields
from morphocoverage.workflows import Setup, optimise

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


def test_f7_a_boundary_share_gets_an_exact_interval_over_units():
    every = {(f"P{i}", "A1"): {5: 0.99} for i in range(5)}
    table = protocol_reliability(
        _plate_curve(every),
        SamplingHierarchy("image_file", "well", "plate"),
        threshold=0.9,
        strata=[],
    ).table
    assert table["share"].iloc[0] == 1.0
    assert table["share_low"].iloc[0] == pytest.approx(0.478, abs=1e-3)
    assert table["share_high"].iloc[0] == 1.0
    block = pd.DataFrame({"unit": list("aabbbcd"), "adequate": [1, 0, 1, 1, 0, 1, 0]})
    units = [block.loc[block["unit"] == u, "adequate"].to_numpy(float) for u in "abcd"]
    shared = bootstrap_fields(units, np.mean, n_boot=500, rng=np.random.default_rng(1))
    # Audit 3 (F-5): never narrower than the exact interval over the units, within 0-1.
    # Issue #95: in whole units, only unit c serves 90% of its samples: 1 of 4.
    exact = _exact(1, 4)
    expected = (max(0.0, min(shared[1], exact[0])), min(1.0, max(shared[2], exact[1])))
    assert _bootstrap(block, 500, np.random.default_rng(1), 0.9) == pytest.approx(expected)


def test_f9_a_left_out_unit_with_too_few_fields_is_named_not_counted_as_zero_of_zero():
    curve = {
        ("P1", "A1"): {5: 0.5, 10: 0.95},
        ("P2", "A1"): {5: 0.5, 10: 0.95},
        ("P3", "A1"): {5: 0.5, 10: None},  # too few fields to be judged at 10
    }
    result = protocol_reliability(
        _plate_curve(curve),
        SamplingHierarchy("image_file", "well", "plate"),
        threshold=0.9,
        strata=[],
    )
    left_out = result.validation.set_index("left_out")
    assert left_out.loc["P3", "chosen_fovs"] == 10 and left_out.loc["P3", "eligible"] == 0
    text = " ".join(result.summary(detail="full").split())
    assert "Not checked on 1 plate whose samples had fewer fields than" in text
    assert "P3 (chosen 10)" in text
    short = " ".join(result.summary().split())
    assert "not checked on 1 plate with fewer fields" in short and "P3 (chosen 10)" in short


def test_i8_setup_random_state_reaches_the_protocol_bootstrap(monkeypatch):
    seen = {}
    protocol = workflow_module.protocol_reliability

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return protocol(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "protocol_reliability", spy)
    objects, fovs = _well_acquisition()
    plan = optimise(objects, fovs, _setup(random_state=7, stage_cols=None), **QUICK)
    assert seen["random_state"] == 7
    assert plan.results["protocol"].settings["random_state_protocol"] == 7
    for name in ("random_state", "n_jobs", "novelty_target", "few_fovs"):
        assert f"``{name}``" in Setup.__doc__


TECHNICAL = SamplingHierarchy("position_id", "technical_repeat", "biological_repeat")


def test_f1_a_reliability_of_exactly_the_threshold_is_adequate():
    result = _ninety_percent_result()
    assert result.curve["preserved"].tolist() == [0.9, 0.9]
    table = protocol_reliability(result, TECHNICAL, threshold=0.9, strata=["condition"]).table
    assert table.loc[0, ["eligible", "adequate"]].tolist() == [2, 2]


def test_f2_protocol_summary_runs_when_no_sample_is_judged_anywhere():
    objects, fovs = _tables(wells=("A1", "A2"), repeats=("R1", "R2"))  # no rare state
    result = protocol_reliability(
        _f_reliability(objects, fovs, criteria=[RARE], fov_counts=[2, 4]),
        TECHNICAL,
        threshold=0.9,
        strata=["condition"],
    )
    assert result.per_unit.empty and "unit" in result.per_unit.columns
    assert "Could not be judged at any depth" in result.summary(detail="full")
    assert "not judged" in result.summary() and "at any depth" in result.summary()


def test_f3_an_undefined_criterion_is_not_called_too_few_fields():
    objects, fovs = _tables(wells=("A1", "A2"), repeats=("R1", "R2"))
    result = protocol_reliability(
        _f_reliability(objects, fovs, criteria=[RARE], fov_counts=[2]),
        TECHNICAL,
        threshold=0.9,
        strata=["condition"],
    )
    row = result.table.iloc[0]
    assert (row["undefined"], row["beyond_stream"]) == (4, 0)
    text = " ".join(result.summary(detail="full").split())
    assert "no technical repeat could be judged at this depth" in text
    assert "had that many fields" not in text and "had that many fields" not in result.summary()
    # Beyond the stream, it still says so.
    short = protocol_reliability(
        _f_reliability(objects, fovs, fov_counts=[20]), TECHNICAL, threshold=0.9, strata=[]
    )
    assert short.table.iloc[0]["beyond_stream"] == 4
    assert "no technical repeat had that many fields to judge" in short.summary(detail="full")


def test_f4_fields_needed_median_counts_samples_that_never_got_there():
    reached = {2: 0.95, 4: 0.97, 8: 0.99}
    never = {2: 0.5, 4: 0.6, 8: 0.7}
    curve = {("R1", "A1"): reached, ("R1", "A2"): never, ("R2", "A1"): never}
    result = protocol_reliability(
        _repeat_curve(curve), BY_WELL, threshold=0.9, strata=["condition"]
    )
    text = " ".join(result.summary(detail="full").split())
    assert "median not reached within the listed counts (only 1 of 3 samples got there" in text
    assert "not reached (1 of 3 got there)" in result.summary()
    curve[("R3", "A1")] = {2: 0.5, 4: 0.95, 8: 0.95}
    curve[("R3", "A2")] = reached
    result = protocol_reliability(
        _repeat_curve(curve), BY_WELL, threshold=0.9, strata=["condition"]
    )
    text = " ".join(result.summary(detail="full").split())
    # Reached at 2, 2 and 4, and 2 never: the median of 2, 2, 4, beyond, beyond is 4.
    needed = "Fields needed per sample: median 4, range 2 to 4; 2 needing more counted as more"
    assert needed in text
    assert "4 (2 to 4; 2 more)" in result.summary()


def test_f20_numpy_integers_are_accepted_as_counts_and_seeds():
    objects, fovs = _tables(n_fovs=8)
    result = _f_reliability(
        objects, fovs, n_splits=np.int64(2), draws=np.int64(3), random_state=np.int64(0)
    )
    assert result.settings["n_splits"] == 2 and type(result.settings["random_state"]) is int
    assert _f_reliability(objects, fovs, n_splits=2, draws=3).curve.equals(result.curve)
    replay = _f_replay(objects, fovs, n_splits=np.int64(2), random_state=np.int64(0))
    assert replay.settings["n_splits"] == 2 and type(replay.settings["random_state"]) is int
    table = protocol_reliability(
        _repeat_curve({("R1", "A"): {2: 1.0}, ("R2", "A"): {2: 0.5}}),
        BY_WELL,
        threshold=0.9,
        strata=[],
        n_boot=np.int64(100),
        random_state=np.int64(0),
    ).table
    assert len(table) == 1


def test_f25_docstrings_state_the_interval_and_the_seeding():
    assert "never narrower" in protocol.__doc__ and "percentile interval" not in (
        protocol.ProtocolResult.__doc__
    )
    from morphocoverage.retrospective import reliability, replay

    assert "split_seed" in reliability.__doc__ and "split_seed" in replay.__doc__
    assert "last prefix must be evaluable" in adequacy.__doc__


def test_f7_f12_one_unit_and_half_field_medians_are_stated_plainly():
    one = {("P1", "A1"): {5: 0.95, 10: 0.99}, ("P1", "A2"): {5: 0.8, 10: 0.95}}
    text = protocol_reliability(_curve(one), HIERARCHY, threshold=0.9, strata=["condition"])
    short = " ".join(text.summary().split())
    text = _full(text)
    assert "With a single plate, no number can be chosen on other plates" in text
    assert "the others reached no number" not in text
    assert "median 7.5, range 5 to 10" in text
    assert "with a single plate, no number can be chosen on other plates" in short
    assert "7.5 (5 to 10)" in short


def test_protocol_names_samples_capped_by_their_reference_instead_of_needing_more_fields():
    # idr0028, variance within 10%: every well's curve is flat at about 30%
    # because the 10 held-back fields cannot pin the variance down, yet the
    # protocol summary listed all 12 wells as needing "more than 18" fields.
    per_sample = _curve_with_ceiling(
        {"A1": {5: 0.3, 18: 0.3}, "A2": {5: 0.3, 18: 0.3}, "B1": {5: 0.5, 18: 0.8}},
        {"A1": 0.3, "A2": 0.3, "B1": 1.0},
    )
    result = protocol_reliability(
        per_sample, SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    )
    difficulty = result.difficulty.set_index("well")
    assert difficulty.loc["A1", "ceiling"] == 0.3
    assert list(difficulty["ceiling_below_threshold"]) == [True, True, False]
    text = " ".join(result.summary(detail="full").split())
    assert "Needs more fields than could be judged: B1 · P1 (more than 18)." in text
    assert "For 2 of 3 samples even every available field together was enough in fewer" in text
    assert "A1 · P1 (all fields enough in 30% of hold-outs); A2 · P1" in text
    assert "A1 · P1 (more than 18)" not in text
    assert re.search(r"\n +5 +\d+ of \d+ samples", result.summary(detail="full"))
    assert "1 fields" not in text
    short = " ".join(result.summary().split())
    assert "1 sample needed more fields than could be judged (B1 · P1 (more than 18))" in short
    assert "for 2 of 3 samples every available field together" in short
    assert short.count("cap what can be shown") == 1
    assert "A1 · P1 (more than 18)" not in short


def test_protocol_short_parts_says_a_note_shared_by_every_block_once(monkeypatch):
    block = {
        "samples": 6,
        "units": 3,
        "stratum": {"condition": "A"},
        "label": "the mean",
        "served": 4,
        "needed": "3 (2-4)",
        "check": "4 of 4",
        "short_notes": ["not checked on 1 plate"],
    }
    blocks = [block, {**block, "stratum": {"condition": "B"}}]
    monkeypatch.setattr(protocol_text, "protocol_answers", lambda result: blocks)
    result = SimpleNamespace(settings={"independent_unit": "plate"}, strata=("condition",))
    lines, cautions = protocol_text.protocol_short_parts(result, indent=2)
    assert lines == [
        (
            "  Fields per sample, for experiments like this one (6 samples in 3 plates; a sample "
            "counts when at\n    least 90% of random acquisitions were enough):"
        ),
        (
            "    stratum  to preserve  all served from  needed   left-out check\n"
            "    A        the mean     4 fields         3 (2-4)  4 of 4\n"
            "    B        the mean     4 fields         3 (2-4)  4 of 4"
        ),
        (
            "  All served from: every sample that could be judged was served at that count and "
            "every larger one.\n    Needed: the median (range) of the fields each needed. "
            "Left-out check: the number chosen on the\n    other plates, tried on the one left out."
        ),
        "  For every stratum and target: not checked on 1 plate.",
    ]
    assert cautions == [
        (
            "With 3 plates the field count is a guide: it rests on these plates, so check it on "
            "the next experiment before fixing a protocol."
        )
    ]


def test_protocol_short_answers_per_target_and_states_few_units_once():
    wells = {**_well_values(("P1",), 0.95, 0.99), **_well_values(("P2",), 0.5, 0.95)}
    result = _plate_protocol(wells)
    short, full = result.summary(), result.summary(detail="full")
    assert "all served from" in short and "10 fields" in short
    assert _flat(short).count("With 2 plates the field count is a guide") == 1
    assert _flat(full).count("With 2 plates the field count is a guide") == 1
    # The full form keeps every count of samples served, per unit, at every depth.
    assert "2 of 4 samples" in full and "4 of 4 samples" in full
    assert "2 of 2, 0 of 2" in full
    assert "Fields needed per sample: median" in full
