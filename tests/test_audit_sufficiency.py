"""Regressions of the 2026-09-26 function audit: sufficiency, replay and calibration (E1-E10).

Each test names the finding it pins (docs/audit/function_audit_2026-09-26.md)
and fails on the code before the fix.
"""

import numpy as np
import pandas as pd
from test_sufficiency import _novel_acquisition

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage._wording import fields as fields_text
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective.adequacy import _median_point, _tail_point, adequacy_points
from morphocoverage.retrospective.calibration import stopping_calibration
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.retrospective.replay import replay_acquisitions
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.strategies.rare_planning import fovs_to_detect
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.sufficiency_tables import _quantile

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)
PERMUTED = [AcquisitionOrder.permuted(seed) for seed in range(5)]


def _homogeneous(n_fovs=40):
    rng = np.random.default_rng(0)
    rows = [
        {"image_file": f"f{j}", "a": a, "b": b}
        for j in range(n_fovs)
        for a, b in rng.normal(10, 0.1, size=(20, 2))
    ]
    return pd.DataFrame(rows), pd.DataFrame({"image_file": [f"f{j}" for j in range(n_fovs)]})


def test_e1_a_rule_that_says_yes_has_stop_points_under_every_objective():
    objects, fovs = _homogeneous()
    for objective in (POPULATION, ObjectiveConfig(SamplingObjective.PHENOTYPE)):
        report = sufficiency_report(
            objects,
            fovs,
            feature_cols=["a", "b"],
            strategies=[],
            objective=objective,
            orders=PERMUTED,
            novelty=SupportNoveltyEstimator(0.5, 1, None, 2, 3),
            novelty_target=0.25,
        )
        row = report.enough.iloc[0]
        assert row["sampled_enough"] == "yes"
        assert row["enough_after_fovs"] is not None and not pd.isna(row["enough_after_fovs"])
        assert "not met at any point" not in row["answer"]


def _short_group_replay():
    rng = np.random.default_rng(0)
    rows = [
        {"condition": group, "repeat": "r", "image_file": f"{group}{j}", "a": value}
        for group, n in (("A", 8), ("B", 20))
        for j in range(n)
        for value in rng.normal(10, 1, 20)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "repeat", "image_file"]].drop_duplicates()
    return replay_acquisitions(
        objects,
        fovs.reset_index(drop=True),
        feature_cols=["a"],
        strategies=[FixedFOVStrategy(3)],
        criteria=[OutcomeCriterion(RelativeMeanError("a"), 0.05, "lower_is_better")],
        objective=POPULATION,
        reservoir_fraction=0.5,
        min_fovs=6,
        n_splits=2,
    )


def test_e2_a_group_too_short_to_decide_never_stopped_rather_than_vanishing():
    replay = _short_group_replay()
    calibration = stopping_calibration(replay)
    short = calibration.table[calibration.table["condition"] == "A"]
    assert len(short) and (short["never_stopped"] == 1.0).all()
    assert "fewer than the 6 fields it waits for before deciding" in calibration.summary()
    apparent = adequacy_points(replay).apparent
    assert set(apparent["condition"]) == {"A", "B"}
    assert apparent.loc[apparent["condition"] == "A", "first_apparent"].isna().all()


def test_e4_e5_medians_and_tails_say_what_the_runs_did():
    streams = pd.Series([20] * 11)
    half = pd.Series([5, pd.NA], dtype="Int64")
    assert "in half or more of runs" in _median_point(half, streams)
    # 10 of 11 runs reached field 10: at least 90% of runs, so the tail is field 10.
    one_missing = pd.Series([10] * 10 + [pd.NA], dtype="Int64")
    assert _tail_point(one_missing, streams) == "by field 10 in 90% of runs"
    # 6 of 7 runs by field 10 is under 90%: the tail is the next observed count.
    seven = pd.Series([10] * 6 + [11], dtype="Int64")
    assert _tail_point(seven, streams) == "by field 11 in 90% of runs"


def test_e9_stop_counts_are_never_interpolated_or_rounded_away():
    assert _quantile(np.array([2.0, 3.0]), 0.9) == 3.0
    assert _quantile(np.array([]), 0.9) is None
    assert fields_text(2.5) == "2.5 fields" and fields_text(6.0) == "6 fields"


def _few_report(confidence, min_other_fovs):
    objects, fovs = _novel_acquisition(n_fovs=40, state_fovs=(3, 4, 20, 21), outlier_fovs=())
    return sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=DISTRIBUTION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(0.5, min_other_fovs, None, 2, 3, confidence=confidence),
        few_fovs=3,
    )


def test_e3_e6_e7_rare_state_sentences_follow_the_settings():
    report = _few_report(0.8, 1)
    row, answer = report.next_fov.iloc[0], report.next_fov["answer"].iloc[0]
    # E6: the zero-success bound at 80% is about 1.6 / k, not 3 / k.
    assert "about 1.6 / 41" in answer and "3 / 41" not in answer
    # E7: with min_other_fovs=1 a recurrent state is in 2 or more FOVs.
    assert "A state seen in 2 to 3 FOVs is not new" in answer
    # E3: the pooled incidence is not the incidence of each state.
    assert row["fovs_to_see_fewest"] == fovs_to_detect(2 / 41, 0.8, 1)
    assert row["fovs_to_see_fewest"] > row["fovs_to_see_again"]
    assert f"one seen in only 2 FOVs needs about {row['fovs_to_see_fewest']}" in answer
    answer = _few_report(0.95, 2).next_fov["answer"].iloc[0]
    assert "about 3 / 41" in answer and "A state seen in 3 FOVs is not new" in answer


def test_e8_fields_without_observations_are_counted_when_novelty_is_not_estimable():
    objects, fovs = _novel_acquisition()
    objects.loc[0, "a"] = np.nan
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=POPULATION,
        orders=PERMUTED,
        novelty=SupportNoveltyEstimator(0.5, 1, None, 2, 3),
    )
    row = report.next_fov.iloc[0]
    assert pd.isna(row["chance_new"])
    assert row["fovs_without_observations"] == 1
