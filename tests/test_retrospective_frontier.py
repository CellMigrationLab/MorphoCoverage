"""Tests for the benchmark score: fields saved at matched risk (retrospective.frontier)."""

from dataclasses import dataclass

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
from morphocoverage._style import Status
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evidence import Component, Direction, Evidence, Quantity
from morphocoverage.outcomes import RareStateDetected, RelativeMeanError
from morphocoverage.policies import ThresholdPolicy
from morphocoverage.retrospective import OutcomeCriterion, replay_acquisitions
from morphocoverage.retrospective.frontier import (
    BASELINE,
    FrontierResult,
    _bootstrap,
    family,
    frontier,
    sweep,
    transfer,
)
from morphocoverage.retrospective.frontier_text import frontier_status
from morphocoverage.strategies.base import Estimator, SamplingStrategy

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
MEAN = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")


def _mixed(seed=0, n_easy=4, n_hard=4, n_fovs=40, cells=20):
    """Samples whose FOVs barely differ (easy) and samples whose FOVs differ a lot (hard).

    The difficulty is known by construction and recorded on the FOV table, so
    a test-only oracle can read it; no packaged strategy sees it.
    """
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for s in range(n_easy + n_hard):
        hard = s >= n_easy
        spread = 0.6 if hard else 0.02
        for f in range(n_fovs):
            name = f"s{s}_f{f}"
            fov_rows.append(
                {
                    "condition": "c",
                    "biological_repeat": f"R{s}",
                    "image_file": name,
                    "difficulty": "hard" if hard else "easy",
                }
            )
            level = 10 + rng.normal(0, spread)
            values = level + rng.normal(0, 0.2, size=(cells, 2))
            object_rows.append(
                pd.DataFrame(values, columns=list(FEATURE_COLS)).assign(
                    condition="c", biological_repeat=f"R{s}", image_file=name
                )
            )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


@dataclass(frozen=True)
class _Oracle(Estimator):
    """Test only: knows each sample's difficulty and asks for ``easy`` or ``hard`` fields."""

    easy: int
    hard: int
    name: str = "oracle"

    @property
    def parameters(self):
        return {"easy": self.easy, "hard": self.hard}

    def estimate(self, data, *, objective):
        need = self.hard if (data.fovs["difficulty"] == "hard").any() else self.easy
        short = max(0, need - len(data.fovs))
        meta = {
            "direction": Direction.LOWER_IS_BETTER,
            "units": "FOVs",
            "component": Component.POPULATION,
        }
        return Evidence({"fields_short": Quantity.available(short, **meta)})


def _oracle(easy, hard):
    return SamplingStrategy([_Oracle(easy, hard)], [ThresholdPolicy("fields_short", 0)])


def _replay(objects, fovs, strategies, *, orders=4, splits=3, seed=0, criteria=(MEAN,)):
    return replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        criteria=list(criteria),
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=[AcquisitionOrder.permuted(s) for s in range(orders)],
        n_splits=splits,
        random_state=seed,
        keep_results=True,
        group_cols=["condition", "biological_repeat"],
    )


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
