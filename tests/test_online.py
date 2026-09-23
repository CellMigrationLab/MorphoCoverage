"""Prospective OnlineSession contracts."""

import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    OnlineSession,
    SamplingObjective,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
)

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _feed_session(scenario, strategies, *, observation_unit=None):
    session = OnlineSession(
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
        observation_unit=observation_unit,
    )
    outputs = []
    ordered = scenario.fovs.sort_values(["condition", "repeat", ORDER_COL], kind="stable")
    for _, fov in ordered.iterrows():
        mask = pd.Series(True, index=scenario.objects.index)
        for column in ("condition", "repeat", "image_file"):
            mask &= scenario.objects[column].eq(fov[column])
        outputs.append(
            session.add_fov(
                scenario.objects.loc[mask].copy(),
                fov.to_dict(),
            )
        )
    return session, outputs


def test_online_session_matches_incremental_replay():
    scenario = SCENARIOS["homogeneous"]()
    strategies = [
        FixedFOVStrategy(4),
        ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
    ]

    replay = evaluate_acquisition_prefixes(
        scenario.objects,
        scenario.fovs,
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=POPULATION,
        incremental=True,
    )
    _, online_steps = _feed_session(scenario, strategies)
    online = [result for step in online_steps for result in step]

    assert len(online) == len(replay)
    for live, retrospective in zip(online, replay, strict=True):
        assert live.without_timing() == retrospective.result.without_timing()


def test_online_session_knows_nothing_about_future_fovs():
    scenario = SCENARIOS["homogeneous"]()
    strategy = FixedFOVStrategy(3)

    ordered = scenario.fovs.sort_values(ORDER_COL)
    first = ordered.iloc[0]
    mask = scenario.objects["image_file"].eq(first["image_file"])

    left = OnlineSession(
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    right = OnlineSession(
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    left_result = left.add_fov(scenario.objects.loc[mask].copy(), first.to_dict())[0]
    right_result = right.add_fov(scenario.objects.loc[mask].copy(), first.to_dict())[0]

    assert left_result.without_timing() == right_result.without_timing()
    assert left_result.n_fovs == 1


def test_online_session_preserves_zero_observation_fov():
    scenario = SCENARIOS["zero_object_fovs"]()
    first = scenario.fovs.nsmallest(1, ORDER_COL).iloc[0]
    observations = scenario.objects.iloc[:0].copy()

    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    [result] = session.add_fov(observations, first.to_dict())

    assert result.n_fovs == 1
    assert result.n_observations == 0
    assert result.decision == "continue"


def test_online_session_keeps_condition_repeat_groups_independent():
    scenario = SCENARIOS["condition_repeat_groups"]()
    session, steps = _feed_session(scenario, [FixedFOVStrategy(2)])

    assert session.group_cols == ("condition", "repeat")
    by_group = {}
    for results in steps:
        [result] = results
        key = (result.group["condition"], result.group["repeat"])
        by_group.setdefault(key, []).append(result)

    assert len(by_group) == scenario.truth["n_groups"]
    for history in by_group.values():
        assert [result.n_fovs for result in history] == list(range(1, len(history) + 1))
        assert history[0].decision == "continue"
        assert history[1].decision == "stop"


def test_online_session_rejects_duplicate_spatial_position():
    scenario = SCENARIOS["homogeneous"]()
    first = scenario.fovs.nsmallest(1, ORDER_COL).iloc[0]
    mask = scenario.objects["image_file"].eq(first["image_file"])
    observations = scenario.objects.loc[mask].copy()

    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    session.add_fov(observations, first.to_dict())
    with pytest.raises(ValueError, match="already been added"):
        session.add_fov(observations, first.to_dict())


def test_online_session_carries_generic_observation_unit():
    scenario = SCENARIOS["homogeneous"]()
    first = scenario.fovs.nsmallest(1, ORDER_COL).iloc[0]
    mask = scenario.objects["image_file"].eq(first["image_file"])

    session = OnlineSession(
        strategies=[FixedFOVStrategy(1)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
        observation_unit="slice",
    )
    [result] = session.add_fov(scenario.objects.loc[mask].copy(), first.to_dict())

    assert result.observation_unit == "slice"
    assert result.n_observations == mask.sum()
    assert result.n_objects == result.n_observations


def test_online_session_accepts_missing_feature_values_for_strategies_that_do_not_use_them():
    scenario = SCENARIOS["homogeneous"]()
    first = scenario.fovs.nsmallest(1, ORDER_COL).iloc[0]
    mask = scenario.objects["image_file"].eq(first["image_file"])
    observations = scenario.objects.loc[mask].copy()
    observations.loc[observations.index[0], "feature_a"] = float("nan")

    session = OnlineSession(
        strategies=[FixedFOVStrategy(1)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    [result] = session.add_fov(observations, first.to_dict())
    assert result.decision == "stop"


def test_online_session_rejects_fov_identifier_contract_drift():
    observations = pd.DataFrame(
        {
            "image_file": ["a.tif"],
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )
    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    session.add_fov(observations, {"image_file": "a.tif"})
    later = observations.assign(image_file="b.tif", position_id="P2")
    with pytest.raises(ValueError, match="higher-priority FOV identifier"):
        session.add_fov(
            later,
            {"image_file": "b.tif", "position_id": "P2"},
        )


def test_online_session_rejects_grouping_contract_drift():
    observations = pd.DataFrame(
        {
            "image_file": ["a.tif"],
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )
    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    session.add_fov(observations, {"image_file": "a.tif"})
    later = observations.assign(image_file="b.tif", repeat="R1")
    with pytest.raises(ValueError, match="changes the automatically resolved grouping contract"):
        session.add_fov(
            later,
            {"image_file": "b.tif", "repeat": "R1"},
        )


def test_online_session_keeps_running_when_a_rule_does_not_fit_the_rows():
    # A cell-based rule on slice rows is not applicable; it must not stop the
    # live session, and FOV-based rules keep deciding.
    scenario = SCENARIOS["homogeneous"]()
    session, outputs = _feed_session(
        scenario,
        [
            ScalarPrecisionStrategy("feature_a", 0.05, "object", "point"),
            ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
            FixedFOVStrategy(3),
        ],
        observation_unit="Slices",
    )
    assert session.observation_unit == "slice"
    object_rule, fov_rule, fixed = outputs[-1]
    assert object_rule.decision is None
    assert object_rule.evidence.quantities["averaging_cv"].availability == "not_applicable"
    assert fov_rule.evidence.quantities["averaging_cv"].availability == "available"
    assert fixed.decision == "stop"


@pytest.mark.parametrize(
    "empty",
    [pd.DataFrame(), pd.DataFrame(columns=list(FEATURE_COLS))],
    ids=["no-columns", "untyped-columns"],
)
def test_zero_observation_fov_accepts_an_untyped_empty_table(empty):
    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)], feature_cols=list(FEATURE_COLS), objective=POPULATION
    )
    [first] = session.add_fov(empty, {"fov_id": "F1"})
    [second] = session.add_fov(
        pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]}), {"fov_id": "F2"}
    )
    assert (first.n_fovs, first.n_observations) == (1, 0)
    assert (second.n_fovs, second.n_observations, second.decision) == (2, 1, "stop")


def test_incremental_sessions_never_join_the_accumulated_data():
    # Joining every FOV on every add would make a long session quadratic.
    scenario = SCENARIOS["homogeneous"]()
    session, _ = _feed_session(
        scenario, [ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point")]
    )
    [group] = session._groups.values()
    assert group._visible is None and len(group._parts) == len(scenario.fovs)
