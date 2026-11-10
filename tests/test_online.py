"""Prospective OnlineSession contracts."""

import warnings

import numpy as np
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
    validate_input_tables,
)
from morphocoverage.transforms import Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _feed_session(scenario, strategies, *, observation_unit=None):
    session = OnlineSession(
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
        observation_unit=observation_unit,
    )
    outputs = []
    ordered = scenario.fovs.sort_values(
        ["condition", "biological_repeat", ORDER_COL], kind="stable"
    )
    for _, fov in ordered.iterrows():
        mask = pd.Series(True, index=scenario.objects.index)
        for column in ("condition", "biological_repeat", "image_file"):
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

    assert session.group_cols == ("condition", "biological_repeat")
    by_group = {}
    for results in steps:
        [result] = results
        key = (result.group["condition"], result.group["biological_repeat"])
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
    later = observations.assign(image_file="b.tif", biological_repeat="R1")
    with pytest.raises(ValueError, match="changes the automatically resolved grouping contract"):
        session.add_fov(
            later,
            {"image_file": "b.tif", "biological_repeat": "R1"},
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


def test_an_empty_position_id_on_every_field_does_not_stop_the_session():
    """CellPatcher writes position_id empty when one file is one field."""
    session = OnlineSession(
        strategies=[FixedFOVStrategy(3)], feature_cols=["a"], objective=POPULATION
    )
    for i in range(4):
        metadata = {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": f"f{i}.tif",
            "position_id": None,
        }
        objects = pd.DataFrame({"a": [1.0, 2.0], "position_id": [None, None]})
        [result] = session.add_fov(objects, metadata)
    assert session.fov_col == "image_file"
    assert result.n_fovs == 4


def test_a_copy_warning_raised_as_an_error_leaves_the_session_as_it_was():
    """#185: the copy warning is part of the transaction, so a warnings-as-errors policy
    refuses the FOV instead of keeping it and raising."""
    import warnings

    scenario = SCENARIOS["homogeneous"]()
    first, second = (row for _, row in scenario.fovs.nsmallest(2, ORDER_COL).iterrows())
    session = OnlineSession(
        strategies=[FixedFOVStrategy(3)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )

    def rows(fov):
        mask = scenario.objects["image_file"].eq(fov["image_file"])
        observations = scenario.objects.loc[mask].copy()
        return observations.assign(label_id=range(1, len(observations) + 1))

    session.add_fov(rows(first), first.to_dict())
    relabelled = rows(second).assign(label_id=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="appear twice in the same field"):
            session.add_fov(relabelled, second.to_dict())
    [result] = session.add_fov(rows(second), second.to_dict())
    assert result.n_fovs == 2


def test_b1_live_session_refuses_z_planes_counted_as_fovs():
    # 3 positions x 4 z-planes, one file per plane and no position_id: the
    # validator refuses the table, so the live session must refuse it too.
    rows, fov_rows = [], []
    for position in range(3):
        for z in range(4):
            name = f"pos{position}_z{z}.tif"
            fov_rows.append({"condition": "a", "biological_repeat": "r1", "image_file": name})
            rows += [
                {"condition": "a", "biological_repeat": "r1", "image_file": name, "z": z, "f": v}
                for v in np.linspace(1, 2, 5)
            ]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fov_rows)
    with pytest.raises(ValueError, match="plane/channel/frame"):
        validate_input_tables(objects, feature_cols=["f"], fovs=fovs)
    session = OnlineSession(
        strategies=[FixedFOVStrategy(5)], feature_cols=["f"], objective=POPULATION
    )
    first = fovs.iloc[0]
    session.add_fov(objects[objects.image_file == first.image_file], first.to_dict())
    second = fovs.iloc[1]
    with pytest.raises(ValueError, match="plane/channel/frame"):
        session.add_fov(objects[objects.image_file == second.image_file], second.to_dict())
    # With fov_col named explicitly, each file is accepted as one position.
    named = OnlineSession(
        strategies=[FixedFOVStrategy(5)],
        feature_cols=["f"],
        objective=POPULATION,
        fov_col="image_file",
    )
    for _, meta in fovs.iterrows():
        (result,) = named.add_fov(objects[objects.image_file == meta.image_file], meta.to_dict())
    assert result.n_fovs == 12


def test_b6_add_fov_is_atomic_when_an_estimator_refuses():
    def observations(position):
        return pd.DataFrame({"position_id": position, "f": np.linspace(1, 2, 5)})

    pilot = ScalarPrecisionStrategy("f", 0.05, "object", "point", pilot_fovs=2, order_col="order")
    session = OnlineSession(
        strategies=[FixedFOVStrategy(10), pilot, FixedFOVStrategy(20)],
        feature_cols=["f"],
        objective=POPULATION,
    )
    session.add_fov(observations("p1"), {"position_id": "p1", "order": 1})
    with pytest.raises(ValueError, match="missing values"):
        session.add_fov(observations("p2"), {"position_id": "p2", "order": None})
    session.add_fov(observations("p2"), {"position_id": "p2", "order": 2})  # the retry
    results = session.add_fov(observations("p3"), {"position_id": "p3", "order": 3})
    assert [(r.n_fovs, r.n_observations) for r in results] == [(3, 15)] * 3
    # A refused first FOV does not fix the session's columns.
    fresh = OnlineSession(
        strategies=[FixedFOVStrategy(3)], feature_cols=["f"], objective=POPULATION
    )
    with pytest.raises(ValueError):
        fresh.add_fov(pd.DataFrame({"f": [np.inf]}), {"position_id": "p1"})
    assert fresh.fov_col is None


def _session(**kwargs):
    return OnlineSession(
        strategies=[FixedFOVStrategy(5)], feature_cols=["f"], objective=POPULATION, **kwargs
    )


def _plane_files(rows_per_file=5):
    """3 positions written as 4 z-plane files each, z only in the FOV table."""
    rows, fov_rows = [], []
    for position in range(3):
        for z in range(4):
            name = f"pos{position}_z{z}.tif"
            fov_rows.append(
                {"condition": "a", "biological_repeat": "r1", "image_file": name, "z": z}
            )
            rows += [
                {"condition": "a", "biological_repeat": "r1", "image_file": name, "f": value}
                for value in np.linspace(1, 2, rows_per_file)
            ]
    return pd.DataFrame(rows), pd.DataFrame(fov_rows)


def _feed(session, objects, fovs):
    for _, meta in fovs.iterrows():
        session.add_fov(objects[objects.image_file == meta.image_file], meta.to_dict())


def test_b1_live_plane_rule_reads_z_from_the_fov_table():
    objects, fovs = _plane_files()
    message = r"holds a single 'z' value \(in the FOV table\)"
    with pytest.raises(ValueError, match=message):
        validate_input_tables(objects, feature_cols=["f"], fovs=fovs)
    # Live, a z is read as planes once one of its values recurs in another field
    # (audit 3, A-14: a stage z distinct in every field is a focus height): the
    # first four files hold z 0-3, and the fifth, z=0 again, is refused.
    session = _session()
    _feed(session, objects, fovs.iloc[:4])
    fifth = fovs.iloc[4]
    with pytest.raises(ValueError, match=message):
        session.add_fov(objects[objects.image_file == fifth.image_file], fifth.to_dict())
    # Refused atomically: the session still holds the first four FOVs only.
    [result] = session.add_fov(
        objects[objects.image_file == fifth.image_file].assign(image_file="other.tif"),
        {**fifth.to_dict(), "image_file": "other.tif", "z": 99},
    )
    assert result.n_fovs == 5

    # One row per field: the validator warns, and so does the live session.
    one, fovs = _plane_files(rows_per_file=1)
    with pytest.warns(UserWarning, match="in the FOV table"):
        validate_input_tables(one, feature_cols=["f"], fovs=fovs)
    with pytest.warns(UserWarning, match="in the FOV table"):
        _feed(_session(), one, fovs)
    # Naming the FOV column says each file is one position: no rule, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _feed(_session(fov_col="image_file"), one, fovs)


def test_b2_live_identity_is_resolved_over_observations_and_fov_row():
    rows, fov_rows = [], []
    for position in range(3):
        for tile in range(4):
            name = f"pos{position}_{tile}.tif"
            fov_rows.append({"condition": "a", "biological_repeat": "r1", "image_file": name})
            rows += [
                {"condition": "a", "biological_repeat": "r1", "position_id": f"pos{position}"}
                | {"image_file": name, "f": value}
                for value in np.linspace(1, 2, 5)
            ]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fov_rows)
    message = "'position_id' is present in objects but absent from fovs"
    with pytest.raises(ValueError, match=message):
        validate_input_tables(objects, feature_cols=["f"], fovs=fovs)
    session = _session()
    with pytest.raises(ValueError, match=message):
        _feed(session, objects, fovs)
    assert session.fov_col is None  # the refused first FOV fixed nothing

    # Later FOVs are resolved the same way, observation columns included.
    session = _session()
    session.add_fov(pd.DataFrame({"f": [1.0]}), {"image_file": "a.tif"})
    with pytest.raises(ValueError, match=message):
        session.add_fov(pd.DataFrame({"f": [1.0], "position_id": ["P2"]}), {"image_file": "b.tif"})
    with pytest.raises(
        ValueError, match="'biological_repeat' is present in objects but absent from fovs"
    ):
        session.add_fov(
            pd.DataFrame({"f": [1.0], "biological_repeat": ["r1"]}), {"image_file": "b.tif"}
        )
    # Identity carried by the FOV row alone is copied onto the observations.
    [result] = session.add_fov(pd.DataFrame({"f": [2.0]}), {"image_file": "b.tif"})
    assert (session.fov_col, session.group_cols, result.n_fovs) == ("image_file", (), 2)
    # An empty position_id (a producer writing every column) is still absent.
    [result] = session.add_fov(
        pd.DataFrame({"f": [3.0], "position_id": [""]}), {"image_file": "c.tif", "position_id": ""}
    )
    assert result.n_fovs == 3


def test_a_result_explains_its_decision_in_one_line():
    from morphocoverage import OnlineSession
    from morphocoverage.strategies import SupportNoveltyStrategy

    built = SCENARIOS["homogeneous"]()
    strategies = [
        FixedFOVStrategy(3),
        SupportNoveltyStrategy(np.sqrt(2), 0.25, 1, Standardize(), 2, 3, "lofo_novel_fov_upper"),
    ]
    session = OnlineSession(
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
        observation_unit="cell",
    )
    for _, fov in built.fovs.iloc[:3].iterrows():
        rows = built.objects[built.objects["image_file"] == fov["image_file"]]
        results = session.add_fov(
            rows.drop(columns=["condition", "biological_repeat"]), fov.to_dict()
        )
    fixed, novelty = results
    # A stop says what it rests on (audit 4, Q2): here, the random-sample assumption.
    assert fixed.explain() == (
        "stop: n_fovs = 3 (stop at >= 3, population); after 3 FOVs, 90 cells; treats the "
        "fields as a random sample of the well's positions; a raster over a spatial trend can "
        "stop early"
    )
    line = novelty.explain()
    # The value, then the rule: a "continue" never reads as a false inequality.
    # Under the population objective the novelty rule cannot settle the decision, and says
    # so beside its own verdict.
    assert line.startswith(
        "unsupported (the rule itself: continue): lofo_novel_fov_upper = 0.632 "
        "(stop at <= 0.25, distribution)"
    )
    assert line.endswith("after 3 FOVs, 90 cells")
