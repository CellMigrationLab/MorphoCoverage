"""Tests for the benchmark score: fields saved at matched risk (retrospective.frontier)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    BOTH,
    MEAN,
    POPULATION,
    _control,
    _mixed,
    _oracle,
    _plate,
    _replay,
)
from _helpers_workflows import _base, _family_score, _flat, _frontier_rows
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ScalarPrecisionStrategy,
)
from morphocoverage._style import Status
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RareStateDetected
from morphocoverage.retrospective import OutcomeCriterion, replay_acquisitions
from morphocoverage.retrospective.effort import MIN_BOOT_SUCCESS
from morphocoverage.retrospective.frontier import (
    BASELINE,
    FrontierResult,
    _bootstrap,
    _met_not_sustained,
    _saved_interval,
    family,
    frontier,
    sweep,
    transfer,
)
from morphocoverage.retrospective.frontier_text import (
    _interval,
    _setting,
    frontier_status,
    frontier_summary,
)
from morphocoverage.retrospective.replay import ReplayResult


@pytest.fixture(scope="module")
def mixed():
    return _mixed()


def test_the_baseline_is_the_replayed_fixed_rule(mixed):
    fixed = [FixedFOVStrategy(k) for k in range(1, 22)]
    replay = _replay(*mixed, fixed)
    replayed = family(replay, "replayed fixed", {s.config_id: s.target_fovs for s in fixed})
    result = frontier(
        replay, [replayed], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"]
    )
    curve = result.curve.set_index(["family", "setting"])
    for k in range(1, 22):
        base, again = curve.loc[(BASELINE, k)], curve.loc[("replayed fixed", k)]
        assert (base["effort"], base["risk"]) == (again["effort"], again["risk"])
    score = result.scores.set_index("family").loc["replayed fixed"]
    assert score["fields_saved"] == 0
    assert score["saved_low"] == score["saved_high"] == 0


def test_a_sweep_equals_replaying_each_threshold(mixed):
    targets = [0.002, 0.005, 0.02]
    strategies = [ScalarPrecisionStrategy("feature_a", t, "fov", "point") for t in targets]
    replay = _replay(*mixed, strategies)
    swept = sweep(
        replay,
        config_id=strategies[0].config_id,
        quantity="averaging_cv",
        thresholds=targets,
        name="JCB per field",
    )
    for target, strategy in zip(targets, strategies, strict=True):
        direct = replay.decisions[replay.decisions["config_id"] == strategy.config_id]
        redone = swept[swept["setting"] == target]
        keys = ["biological_repeat", "split", "order_id", "fovs"]
        left = direct.sort_values(keys)["decision"].tolist()
        right = redone.sort_values(keys)["decision"].tolist()
        assert left == right


def test_an_adaptive_rule_saves_fields_and_its_stops_follow_the_need(mixed):
    # By construction easy samples need 1 field and hard ones about 7; a rule
    # that knows which is which saves fields at the same risk, and its stops
    # correlate with where held-back fields confirmed the mean.
    oracles = [_oracle(1, hard) for hard in (4, 6, 8, 10, 12, 16, 20)]
    replay = _replay(*mixed, oracles, orders=6)
    rules = family(replay, "oracle", {s.config_id: s.estimators[0].hard for s in oracles})
    result = frontier(
        replay, [rules], criterion=MEAN.name, alphas=[0.05], unit_cols=["biological_repeat"]
    )
    score = result.scores.set_index("family").loc["oracle"]
    base = result.scores.set_index("family").loc[BASELINE]
    assert base["effort"] is not None and score["effort"] is not None
    # 0.5 to 2.5 fields over random_state 0-5 (reservoirs seeded by _seeds.split_seed).
    assert score["fields_saved"] > 0
    assert score["saved_low"] <= score["fields_saved"] <= score["saved_high"]
    assert score["adaptivity"] > 0.3
    assert score["baseline_risk_at_same_effort"] > score["risk"]
    text = result.summary(detail="full")
    assert "a fixed number of fields needed" in text and "oracle at" in text
    assert "saves" in text and "treated as independent" in text
    short = result.summary()
    assert "a fixed count needs" in short and "oracle at" in short and "saves" in short


def test_more_replays_of_the_same_units_do_not_narrow_the_interval(mixed):
    # The interval resamples units; repeating every run of every unit (more
    # hold-outs or orders of the same samples) leaves every resample unchanged.
    oracles = [_oracle(1, hard) for hard in (4, 6, 8, 10, 12, 16)]
    replay = _replay(*mixed, oracles)
    rules = family(replay, "oracle", {s.config_id: s.estimators[0].hard for s in oracles})
    runs = frontier(
        replay, [rules], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"]
    ).runs
    doubled = pd.concat([runs, runs.assign(split=runs["split"] + 100)], ignore_index=True)
    once = _bootstrap(runs, ["biological_repeat"], [0.1], 300, np.random.default_rng(0))
    twice = _bootstrap(doubled, ["biological_repeat"], [0.1], 300, np.random.default_rng(0))
    np.testing.assert_allclose(once[("oracle", 0.1)], twice[("oracle", 0.1)])


def test_one_unit_gives_no_interval(mixed):
    oracles = [_oracle(2, 12)]
    replay = _replay(*mixed, oracles)
    rules = family(replay, "oracle", {oracles[0].config_id: 12})
    result = frontier(replay, [rules], criterion=MEAN.name, alphas=[0.2], unit_cols=[])
    score = result.scores.set_index("family").loc["oracle"]
    assert score["saved_low"] is None and result.settings["units"] == 1


def test_outcomes_that_cannot_be_judged_are_counted_not_scored():
    built = SCENARIOS["rare_clustered"]()
    criterion = OutcomeCriterion(RareStateDetected("true_state", 1), 1, "higher_is_better")
    replay = replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(10)],
        criteria=[criterion],
        objective=POPULATION,
        reservoir_fraction=0.35,
        orders=[AcquisitionOrder.permuted(0)],
        n_splits=10,
    )
    result = frontier(replay, [], criterion=criterion.name, alphas=[0.1], unit_cols=[])
    row = result.curve.set_index("setting").loc[10]
    assert 0 < row["judged"] < row["runs"]
    # #249: the chosen risk carries its denominator, and the summary says it.
    base = result.scores.iloc[0]
    assert 0 < base["judged"] < base["runs"]
    said = f"of the {int(base['judged'])} of {int(base['runs'])} runs judged"
    assert said in " ".join(result.summary().split())


def test_families_from_replays_with_other_streams_are_refused(mixed):
    one = _replay(*mixed, [_oracle(2, 12)], seed=0)
    other = _replay(*mixed, [_oracle(2, 16)], seed=1)
    alien = family(other, "oracle", {_oracle(2, 16).config_id: 16})
    alien = alien.assign(stream_fovs=alien["stream_fovs"] + 1)
    with pytest.raises(ValueError, match="another acquisition"):
        frontier(one, [alien], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])


def test_a_setting_chosen_on_development_is_scored_on_held_out_data(mixed):
    targets = [0.002, 0.005, 0.01, 0.02]
    jcb = ScalarPrecisionStrategy("feature_a", 0.01, "fov", "point")
    scores = {}
    for name, seed in (("development", 0), ("held_out", 7)):
        objects, fovs = _mixed(seed=seed)
        replay = _replay(objects, fovs, [jcb], seed=seed)
        swept = sweep(
            replay,
            config_id=jcb.config_id,
            quantity="averaging_cv",
            thresholds=targets,
            name="JCB per field",
        )
        scores[name] = frontier(
            replay, [swept], criterion=MEAN.name, alphas=[0.2], unit_cols=["biological_repeat"]
        )
    table = transfer(scores["development"], scores["held_out"], 0.2).set_index("family")
    chosen = table.loc["JCB per field"]
    if chosen["setting"] is not None and not pd.isna(chosen["setting"]):
        held = scores["held_out"].curve
        match = held[(held["family"] == "JCB per field") & (held["setting"] == chosen["setting"])]
        assert chosen["held_out_risk"] == match["risk"].iloc[0]
    assert set(table.index) == {BASELINE, "JCB per field"}
    with pytest.raises(ValueError, match="not scored"):
        transfer(scores["development"], scores["held_out"], 0.01)


def test_invalid_requests_are_refused(mixed):
    replay = _replay(*mixed, [_oracle(2, 12)])
    with pytest.raises(ValueError, match="keep_results"):
        no_results = replay_acquisitions(
            mixed[0],
            mixed[1],
            feature_cols=list(FEATURE_COLS),
            strategies=[_oracle(2, 12)],
            criteria=[MEAN],
            objective=POPULATION,
            reservoir_fraction=0.3,
            orders=[AcquisitionOrder.permuted(0)],
            n_splits=1,
            group_cols=["condition", "biological_repeat"],
        )
        sweep(no_results, config_id="x", quantity="q", thresholds=[1], name="n")
    with pytest.raises(ValueError, match="not assessed"):
        frontier(replay, [], criterion="nothing", alphas=[0.1], unit_cols=[])
    with pytest.raises(ValueError, match="alphas"):
        frontier(replay, [], criterion=MEAN.name, alphas=[1.5], unit_cols=[])
    with pytest.raises(ValueError, match="not group columns"):
        frontier(replay, [], criterion=MEAN.name, alphas=[0.1], unit_cols=["well"])
    # #247: with no family no rule was tested, so none can be said not to get there.
    status, text = frontier_status(
        frontier(replay, [], criterion=MEAN.name, alphas=[0.1], unit_cols=[])
    )
    assert status is Status.NOT_ASSESSED and "no rule family given" in text
    renamed = family(replay, BASELINE, {_oracle(2, 12).config_id: 12})
    with pytest.raises(ValueError, match="baseline's name"):
        frontier(replay, [renamed], criterion=MEAN.name, alphas=[0.1], unit_cols=[])


def test_two_configurations_at_one_setting_are_refused(mixed):
    # Merged, they would stop at the earlier of the two stops: a rule never replayed (#230).
    oracles = [_oracle(2, 4), _oracle(2, 16)]
    replay = _replay(*mixed, oracles, orders=1, splits=1)
    with pytest.raises(ValueError, match="settings must differ"):
        family(replay, "oracle", {s.config_id: 0.1 for s in oracles})
    family(replay, "oracle", {s.config_id: s.estimators[0].hard for s in oracles})


def test_the_interval_reads_in_the_direction_of_the_verdict():
    # fields_saved and its interval are negative when a rule costs fields; the sentence
    # "costs 1.3 more fields" must not be followed by an interval of -1.3 to -1.2.
    def row(family, effort, saved, low, high):
        return {
            "family": family,
            "alpha": 0.05,
            "setting": 1.0,
            "effort": effort,
            "risk": 0.0,
            "fields_saved": saved,
            "saved_low": low,
            "saved_high": high,
            "adaptivity": None,
        }

    scores = pd.DataFrame(
        [
            row(BASELINE, 1.0, 0.0, None, None),
            row("costly", 2.3, -1.3, -1.3, -1.2),
            row("thrifty", 0.5, 0.5, 0.2, 0.8),
            row("level", 1.0, 0.0, -0.1, 0.0),
        ]
    )
    settings = {"label": "x", "units": 6}  # an interval is printed from 5 units (audit 4)
    result = FrontierResult(pd.DataFrame(), scores, pd.DataFrame(), settings)
    text = " ".join(result.summary(detail="full").split())
    short = result.summary()
    assert "costs 1.3 (1.2 to 1.3)" in short and "saves 0.5 (0.2 to 0.8)" in short
    assert "as well, no better (-0.1 to 0.0)" in short
    assert "costs 1.3 more fields, approximate 95% interval 1.2 to 1.3" in text
    assert "saves 0.5 fields, approximate 95% interval 0.2 to 0.8" in text
    assert "no better, approximate 95% interval of the fields saved -0.1 to 0.0" in text


def test_the_summary_states_the_efforts_it_subtracts():
    # Fields saved are the baseline's mean effort minus the rule's. When some streams are
    # shorter than the fixed count, that mean is below the setting, and the text says so,
    # so "7 fields" and "saves 1.4" never contradict "5.1 on average".
    scores = pd.DataFrame(
        [
            {
                "family": BASELINE,
                "alpha": 0.05,
                "setting": 7.0,
                "effort": 6.5,
                "risk": 0.04,
                "fields_saved": 0.0,
                "saved_low": None,
                "saved_high": None,
                "adaptivity": None,
                "saved_reached": None,
            },
            {
                "family": "rule",
                "alpha": 0.05,
                "setting": 0.03,
                "effort": 5.1,
                "risk": 0.03,
                "fields_saved": 1.4,
                "saved_low": 0.2,
                "saved_high": 2.0,
                "adaptivity": None,
                "saved_reached": 0.8,
            },
        ]
    )
    settings = {"label": "x", "units": 12, "unit_cols": ["condition", "biological_repeat", "well"]}
    result = FrontierResult(pd.DataFrame(), scores, pd.DataFrame(), settings)
    text = " ".join(result.summary(detail="full").split())
    assert "needed 7 fields, 6.5 on average" in text
    assert "80% of the resamples" in text
    assert "12 units (condition · biological repeat · well)" in text
    short = " ".join(result.summary().split())
    assert "a fixed count needs 7 fields, 6.5 on average" in short
    assert "80% of the resamples" in short


@pytest.fixture(scope="module")
def seeded_mixed():
    return _mixed(seed=0)


def test_a_family_without_a_fingerprint_is_refused(seeded_mixed):
    replay = _replay(*seeded_mixed, [_oracle(2, 12)], seed=0)
    own = family(replay, "oracle", {_oracle(2, 12).config_id: 12}).drop(columns="run_fingerprint")
    with pytest.raises(ValueError, match="no run fingerprint"):
        frontier(replay, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])


def test_a_tampered_fingerprint_is_refused(seeded_mixed):
    replay = _replay(*seeded_mixed, [_oracle(2, 12)], seed=0)
    own = family(replay, "oracle", {_oracle(2, 12).config_id: 12})
    own = own.assign(run_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="another acquisition"):
        frontier(replay, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])


def test_the_replays_own_family_is_accepted(seeded_mixed):
    replay = _replay(*seeded_mixed, [_oracle(2, 12)], seed=0)
    own = family(replay, "oracle", {_oracle(2, 12).config_id: 12})
    result = frontier(
        replay, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"]
    )
    assert "oracle" in set(result.scores["family"])


def test_same_seeds_on_another_set_of_fields_are_refused(seeded_mixed):
    replay = _replay(*seeded_mixed, [_oracle(2, 12)], seed=0)
    other = _replay(*_mixed(n_fovs=30), [_oracle(2, 12)], seed=0)
    alien = family(other, "oracle", {_oracle(2, 12).config_id: 12})
    with pytest.raises(ValueError, match="another acquisition"):
        frontier(
            replay, [alien], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"]
        )


def test_frontier_alphas_are_real_risk_levels():
    for bad in (["0.05"], [True], [float("nan")], [0.0], [1.0]):
        with pytest.raises(ValueError, match="risk levels"):
            frontier(object.__new__(ReplayResult), [], criterion="c", alphas=bad, unit_cols=["g"])


def test_g15_a_rule_that_met_the_risk_but_not_lastingly_is_not_never():
    curve = pd.DataFrame(
        {"setting": ["cheap", "late"], "effort": [1.3, 10.9], "risk": [0.0, 1 / 3]}
    )
    met, exceeded = _met_not_sustained(curve, 0.05)
    assert (met["setting"], exceeded["setting"]) == ("cheap", "late")
    assert _met_not_sustained(curve.assign(risk=[0.2, 0.3]), 0.05) is None
    scores = pd.DataFrame(
        [
            {"family": BASELINE, "alpha": 0.05, "setting": 14, "effort": 14.0, "risk": 0.0},
            {
                "family": "rule",
                "alpha": 0.05,
                "setting": None,
                "effort": None,
                "risk": None,
                "met_setting": "cheap",
                "met_risk": 0.0,
                "exceeded_setting": "late",
                "exceeded_risk": 1 / 3,
            },
        ]
    )
    result = FrontierResult(pd.DataFrame(), scores, pd.DataFrame(), {"label": "the mean"})
    text = " ".join(result.summary(detail="full").split())
    assert "rule: met this risk at cheap (risk 0%) but not from there on: at late" in text
    assert "never got there" not in text


def _score(saved, low, high):
    return {
        "family": "rule",
        "alpha": 0.05,
        "setting": 0.1,
        "effort": 12.0 - saved,
        "risk": 0.05,
        "fields_saved": saved,
        "saved_low": low,
        "saved_high": high,
        "saved_reached": 1.0,
        "baseline_risk_at_same_effort": 0.07,
        "adaptivity": None,
    }


def test_g16_a_saving_whose_interval_includes_zero_is_not_resolved():
    baseline = {"family": BASELINE, "alpha": 0.05, "setting": 12, "effort": 12.0, "risk": 0.04}
    for saved, low, high, resolved in ((1.2, -2.5, 3.0, False), (1.2, 0.4, 3.0, True)):
        scores = pd.DataFrame([baseline, _score(saved, low, high)])
        result = FrontierResult(pd.DataFrame(), scores, pd.DataFrame(), {"units": 6})
        for text in (result.summary(), result.summary(detail="full")):
            assert ("not resolved" in text) != resolved
    costs = pd.DataFrame([baseline, _score(-1.5, -3.0, 0.5)])
    result = FrontierResult(pd.DataFrame(), costs, pd.DataFrame(), {"units": 6})
    text = " ".join(result.summary(detail="full").split())
    assert "costs 1.5 more fields, approximate 95% interval -0.5 to 3.0 (not resolved)" in text
    assert "costs 1.5 (-0.5 to 3.0, not resolved)" in result.summary()


def _frontier_scores(rows, settings=None):
    return FrontierResult(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), settings or {})


def test_g6_a_baseline_that_met_the_risk_but_lost_it_says_so_and_keeps_the_families():
    base = {
        "family": BASELINE,
        "alpha": 0.05,
        "setting": None,
        "effort": None,
        "risk": None,
        "met_setting": 2,
        "met_risk": 0.0,
        "exceeded_setting": 3,
        "exceeded_risk": 0.125,
    }
    rule = {"family": "rule", "alpha": 0.05, "setting": 1, "effort": 2.0, "risk": 0.0}
    result = _frontier_scores([base, rule], {"label": "c", "units": 8})
    text = " ".join(result.summary(detail="full").split())
    assert "a fixed number of fields met this risk at 2 (risk 0%) but not from there on" in text
    assert "never got there" not in text
    assert "rule at 1: 2 fields on average (risk 0%), with no fixed count to compare" in text


def test_g8_a_frontier_interval_from_too_few_resamples_is_withheld():
    statuses = {
        f"u{i}": (["n", "n", "n", "p"] if i == 0 else ["n", "p", "p", "p"]) for i in range(20)
    }
    names = {"p": "preserved", "n": "not_preserved"}
    outcomes, streams, decisions = [], [], []
    for unit, row in statuses.items():
        run = {"unit": unit, "split": 0, "order_id": "recorded", "order": "recorded"}
        streams.append(
            {
                **run,
                "order_seed": None,
                "stream_fovs": 4,
                "reservoir_fovs": 5,
                "run_fingerprint": unit,
            }
        )
        for k, status in enumerate(row, start=1):
            outcomes.append(
                {**run, "fovs": k, "criterion": "c", "value": 0.0, "status": names[status]}
            )
            decisions.append(
                {
                    **run,
                    "strategy": "r",
                    "config_id": "cfg",
                    "fovs": k,
                    "decision": "stop" if k >= 2 else "continue",
                }
            )
    replay = ReplayResult(
        pd.DataFrame(decisions),
        pd.DataFrame(outcomes),
        pd.DataFrame(streams),
        ("unit",),
        {"random_state": 0},
    )
    from morphocoverage.retrospective.frontier import family, frontier

    result = frontier(
        replay,
        [family(replay, "rule", {"cfg": 1})],
        criterion="c",
        alphas=[0.06],
        unit_cols=["unit"],
        n_boot=500,
    )
    row = result.scores.set_index("family").loc["rule"]
    # 72% of the resamples reach the risk for both: below the effort projection's 90%.
    assert MIN_BOOT_SUCCESS == 0.9 and row["saved_reached"] < MIN_BOOT_SUCCESS
    assert row["saved_low"] is None or pd.isna(row["saved_low"])
    reached = f"{100 * row['saved_reached']:.0f}%"
    for text in (result.summary(), result.summary(detail="full")):
        text = " ".join(text.split())
        assert f"no interval: only {reached} of the resamples reached this risk for both" in text
        assert "(at least 90% are needed)" in text


def test_g18_a_fixed_count_beyond_the_held_out_streams_acquires_the_whole_stream():
    development = FrontierResult(
        pd.DataFrame(),
        pd.DataFrame(
            [{"family": BASELINE, "alpha": 0.1, "setting": 5, "risk": 0.05, "effort": 5.0}]
        ),
        pd.DataFrame(),
    )
    held_out = FrontierResult(
        pd.DataFrame(
            {
                "family": BASELINE,
                "setting": [1, 2, 3],
                "effort": [1.0, 2.0, 3.0],
                "risk": [0.5, 0.2, 0.1],
                "judged": [10, 10, 10],
            }
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    row = transfer(development, held_out, 0.1).iloc[0]
    assert row["held_out_setting"] == 3
    assert row["held_out_risk"] == 0.1 and row["held_out_effort"] == 3.0


def _frontier_result(units: int, low: float | None = 1.0, high: float | None = 3.0):
    """A scored frontier with one family saving 2 fields, as ``frontier`` returns it."""
    scores = pd.DataFrame(
        [
            {"family": BASELINE, "alpha": 0.1, "setting": 6, "effort": 6.0, "risk": 0.05},
            {
                "family": "rule",
                "alpha": 0.1,
                "setting": 0.05,
                "effort": 4.0,
                "risk": 0.08,
                "fields_saved": 2.0,
                "saved_low": low,
                "saved_high": high,
                "saved_reached": 1.0,
                "adaptivity": None,
            },
        ]
    )
    settings = {"label": "the mean", "units": units, "unit_cols": ["biological_repeat"]}
    return FrontierResult(pd.DataFrame(), scores, pd.DataFrame(), settings)


def test_p3_3_frontier_interval_is_widened_for_the_units_and_said_to_be_approximate():
    rng = np.random.default_rng(0)
    draws = rng.normal(2.0, 1.0, 2000)
    plain_low, plain_high = np.percentile(draws, [2.5, 97.5])
    low, high = _saved_interval(2.0, draws, 8)
    # Widened by sqrt(8 / 7) t_7 / z, about 1.28 on each side.
    assert low < plain_low - 0.4 and high > plain_high + 0.4
    assert (2.0 - low) / (2.0 - plain_low) == pytest.approx(1.28, abs=0.02)
    text = " ".join(frontier_summary(_frontier_result(units=8), detail="full").split())
    assert "approximate 95% interval 1.0 to 3.0" in text
    assert "with fewer than 20 units they cover the truth less often than stated" in text
    short = " ".join(frontier_summary(_frontier_result(units=8)).split())
    assert "saves 2.0 (1.0 to 3.0)" in short
    assert "with fewer than 20 units they cover the truth less often than stated" in short


def test_p8_1_frontier_gives_no_interval_below_five_units():
    draws = np.random.default_rng(1).normal(7.0, 1.5, 2000)
    assert _saved_interval(7.0, draws, 2) == (None, None)
    assert _saved_interval(7.0, draws, 4) == (None, None)
    assert _saved_interval(7.0, draws, 5)[0] is not None
    # Even a stored interval is not printed from two units (entry 8's plates 1A and 2A).
    for detail in ("short", "full"):
        text = frontier_summary(_frontier_result(units=2), detail=detail)
        [rule_line] = [line for line in text.split("\n") if "rule at 0.05" in line]
        assert "interval" not in rule_line and "to" not in rule_line.split("%")[-1]
        text = " ".join(text.split())
        assert (
            "No interval: only 2 units (biological repeat) to resample, and at least 5 are needed"
            in text
        )


def test_f4_checks_that_could_not_be_judged_are_counted_not_printed_as_nan():
    objects, fovs = _plate()
    rows = objects.index[(objects["biological_repeat"] == "P1") & (objects["condition"] == "KD")][
        :1
    ]
    objects.loc[rows, "feature_a"] = np.nan
    text = _control(objects, fovs, BOTH).summary()
    assert "nan" not in text.lower()
    assert "1 not judged" in text and "could not be judged" in text


def test_f8_f14_frontier_refuses_foreign_replays_and_says_when_there_is_no_interval():
    mixed = _mixed()
    one = _replay(*mixed, [_oracle(2, 12)], seed=0)
    other = _replay(*mixed, [_oracle(2, 12)], seed=1)
    alien = family(other, "oracle", {_oracle(2, 12).config_id: 12})
    with pytest.raises(ValueError, match="another acquisition"):
        frontier(one, [alien], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])
    own = family(one, "oracle", {_oracle(2, 12).config_id: 12})
    result = frontier(one, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=[])
    assert "No interval: there is a single unit to resample" in result.summary()
    assert "resample the 1 units" not in result.summary()


def test_f15_family_settings_are_printed_as_given():
    assert _setting(0.05) == "0.05" and _setting(12) == "12" and _setting("strict") == "strict"


def _saving_score(family, saved, low):
    return {
        "family": family,
        "alpha": 0.05,
        "setting": 0.1,
        "effort": 6.0 - saved,
        "risk": 0.04,
        "fields_saved": saved,
        "saved_low": low,
        "saved_high": saved + 1,
        "saved_reached": 1.0,
        "adaptivity": 0.3,
    }


def _frontier(rows):
    base = {**_saving_score(BASELINE, 0.0, None), "setting": 6.0, "effort": 6.0}
    settings = {"label": "the mean", "units": 8, "unit_cols": ["biological_repeat"]}
    return FrontierResult(pd.DataFrame(), pd.DataFrame([base, *rows]), pd.DataFrame(), settings)


def test_the_family_that_saved_most_is_described_when_several_were_scored():
    one = _frontier([_saving_score("saver", 2.0, 0.5)])
    assert frontier_status(one) == (Status.OK, "at ≤5% risk, saver saves 2.0 of 6 fields")
    several = _frontier([_saving_score("saver", 2.0, 0.5), _saving_score("other", 1.0, 0.2)])
    status, text = frontier_status(several)
    assert status is Status.OK
    assert "saver saves the most here, 2.0 of 6 fields" in text
    unresolved = _frontier([_saving_score("saver", 2.0, -0.5), _saving_score("other", 1.0, -0.2)])
    assert (
        "saver may save the most here, 2.0 fields (not resolved)" in frontier_status(unresolved)[1]
    )


def test_frontier_interval_without_bounds_and_from_part_of_the_resamples():
    assert (
        _interval(SimpleNamespace(saved_reached=1.0, saved_low=None, saved_high=None), 2.0, 6) == ""
    )
    row = SimpleNamespace(saved_reached=0.95, saved_low=1.0, saved_high=3.0)
    assert _interval(row, 2.0, 6) == (
        ", approximate 95% interval 1.0 to 3.0 (from the 95% of the resamples in which both "
        "reached this risk)"
    )
    row = SimpleNamespace(saved_reached=0.95, saved_low=-1.0, saved_high=-3.0)
    assert _interval(row, -2.0, 6) == (
        ", approximate 95% interval 3.0 to 1.0 (from the 95% of the resamples in which both "
        "reached this risk)"
    )


RESAMPLED = "treated as independent"


@pytest.mark.parametrize(
    ("rule", "units", "status", "words"),
    [
        (("saver", 4.0, 2.0, 0.5, 3.0), 8, Status.OK, "saver saves 2.0 of 6 fields"),
        (("saver", 4.0, 2.0, -0.5, 3.0), 8, Status.CAUTION, "may save 2.0 fields (not resolved)"),
        (("saver", 4.0, 2.0, 0.5, 3.0), 3, Status.CAUTION, "may save 2.0 fields (no interval)"),
        (("costly", 8.0, -2.0, -3.0, -1.0), 8, Status.OK, "no rule beats a fixed 6 fields"),
        (("costly", 8.0, -2.0, None, None), 3, Status.CAUTION, "(no interval)"),
    ],
)
def test_frontier_status_follows_its_rule(rule, units, status, words):
    name, effort, saved, low, high = rule
    result = _frontier_rows(
        [_base(0.05), _family_score(name, 0.05, effort, saved, low, high)], units
    )
    got, headline = frontier_status(result)
    assert got is status and words in headline
    assert result.summary().startswith(f"{status.marker} For the mean")


def test_an_unscored_level_is_not_assessed_and_prints_no_interval():
    base = _family_score(BASELINE, 0.05, None, setting=None)
    result = _frontier_rows([base, _family_score("rule", 0.05, None)])
    assert frontier_status(result)[0] is Status.NOT_ASSESSED
    assert RESAMPLED not in result.summary() and RESAMPLED not in result.summary(detail="full")
    assert "No rule got there at any setting tried." in result.summary()


def test_frontier_short_is_a_table_with_levels_merged_and_the_interval_caveat_once():
    rows = []
    for alpha in (0.05, 0.1):
        rows += [_base(alpha), _family_score("rule", alpha, 4.0, 2.0, 0.5, 3.0)]
    result = _frontier_rows(rows)
    short = result.summary()
    assert "at most 5% or 10% of acquisitions: a fixed count needs 6 fields (risk 4%)" in _flat(
        short
    )
    assert "rule at 0.1" in short and "saves 2.0 (0.5 to 3.0)" in short
    assert short.count(RESAMPLED) == 1 and "(biological repeat · well)" in short
    full = _flat(result.summary(detail="full"))
    # Every number of the old sentences, per level, and the caveat once.
    sentence = (
        "rule at 0.1: 4 fields on average (risk 4%); it saves 2.0 fields, approximate 95% "
        "interval 0.5 to 3.0; its stops follow the fields each acquisition needed with "
        "Spearman 0.31."
    )
    assert full.count(sentence) == 2
    assert full.count("a fixed number of fields needed 6 fields (risk 4%).") == 2
    assert full.count(RESAMPLED) == 1


def test_a_lapse_is_stated_in_both_forms():
    lapse = _family_score(
        "rule",
        0.05,
        None,
        met_setting="cheap",
        met_risk=0.0,
        exceeded_setting="late",
        exceeded_risk=0.25,
    )
    result = _frontier_rows([_base(0.05), lapse])
    for text in (result.summary(), result.summary(detail="full")):
        assert "met this risk at cheap (risk 0%) but not from there on" in _flat(text)
