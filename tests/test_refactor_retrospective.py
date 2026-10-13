"""Characterisation tests for the retrospective functions split up without a change of
behaviour (refactor of 2026-10-12): the branches no golden job and no other test runs, with
their exact text, exceptions and values."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    Summary,
    protocol_text,
    replay_acquisitions,
    report_profile,
    report_status,
    report_status_conditions,
    report_status_controls,
    report_status_plan,
)
from morphocoverage.retrospective.adequacy import adequacy_points
from morphocoverage.retrospective.controls import (
    ControlResult,
    RepresentationControl,
    representation_control,
)
from morphocoverage.retrospective.effect_preservation import effect_preservation
from morphocoverage.retrospective.effort import EffortResult, effort_projection
from morphocoverage.retrospective.effort_text import effort_short_parts
from morphocoverage.retrospective.frontier import family, frontier
from morphocoverage.retrospective.frontier_text import _interval
from morphocoverage.retrospective.hierarchy import SamplingHierarchy
from morphocoverage.retrospective.inspection import inspect_fovs
from morphocoverage.retrospective.overview import dataset_overview
from morphocoverage.retrospective.report import microscopist_report
from morphocoverage.retrospective.spatial import order_efficiency, redundancy_distance
from morphocoverage.strategies.support_novelty import SupportNoveltyEstimator

NOVELTY = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
EMPTY = pd.DataFrame()
MEAN = OutcomeCriterion(RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean")
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
FIXED = [FixedFOVStrategy(3), FixedFOVStrategy(5)]


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    built = SCENARIOS["homogeneous"]()
    return built.objects, built.fovs


def _replay(**kwargs: object):
    objects, fovs = _tables()
    return replay_acquisitions(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=FIXED,
        criteria=[MEAN],
        objective=POPULATION,
        **kwargs,
    )


def test_adequacy_and_frontier_refuse_another_object():
    with pytest.raises(TypeError, match="^replay must be a ReplayResult$"):
        adequacy_points(None)
    with pytest.raises(TypeError, match="^replay must be a ReplayResult$"):
        frontier(None, [], criterion="x", alphas=[0.1], unit_cols=[])


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"novelty": "x"}, TypeError, "novelty must be a SupportNoveltyEstimator"),
        ({"influence_threshold": 0}, ValueError, "influence_threshold must be a positive number"),
        (
            {"count_ratio": 1},
            ValueError,
            "count_ratio must be greater than 1, or None to skip the count reason",
        ),
        ({"neighbour_threshold": 0}, ValueError, "neighbour_threshold must be a positive number"),
        (
            {"influence_spread": 0},
            ValueError,
            "influence_spread must be a positive number, or None",
        ),
        ({"n_neighbours": 0}, ValueError, "n_neighbours must be a positive integer"),
    ],
)
def test_inspect_fovs_checks_its_settings_first(kwargs, error, message):
    with pytest.raises(error) as raised:
        inspect_fovs(EMPTY, EMPTY, feature_cols=["a"], **kwargs)
    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"novelty": "x"}, TypeError, "novelty must be a SupportNoveltyEstimator"),
        ({"control": "x"}, TypeError, "control must be a RepresentationControl"),
        ({"fovs_seen": 0}, ValueError, "fovs_seen must be a positive integer"),
        ({"draws": 0}, ValueError, "draws must be a positive integer"),
        ({"random_state": 1.5}, TypeError, "random_state must be an integer seed"),
    ],
)
def test_representation_control_checks_its_arguments_in_order(kwargs, error, message):
    arguments = {
        "feature_cols": ["a"],
        "novelty": NOVELTY,
        "control": RepresentationControl(),
        "pair_cols": [],
        "fovs_seen": 3,
        **kwargs,
    }
    with pytest.raises(error) as raised:
        representation_control(EMPTY, EMPTY, **arguments)
    assert str(raised.value) == message


def test_effect_preservation_checks_the_control_before_the_tables():
    with pytest.raises(ValueError) as raised:
        effect_preservation(EMPTY, EMPTY, feature="x", control={"well": "A1"})
    assert str(raised.value) == "control must name the condition column 'condition'"
    with pytest.raises(TypeError) as raised:
        effect_preservation(EMPTY, EMPTY, feature="x", comparisons="not pairs")
    assert str(raised.value) == "comparisons must be a sequence of (condition, reference) pairs"
    # Valid comparisons replace the control: the tables are read next.
    with pytest.raises(ValueError, match="^Could not resolve a spatial FOV/position identifier"):
        effect_preservation(EMPTY, EMPTY, feature="x", comparisons=[("a", "b")])


def test_effort_projection_names_the_hierarchy_columns_missing():
    fovs = pd.DataFrame({"fov": ["f1"], "sample": ["s1"]})
    objects = pd.DataFrame({"fov": ["f1"], "sample": ["s1"], "x": [1.0]})
    hierarchy = SamplingHierarchy("fov", "sample", "biological_repeat")
    with pytest.raises(ValueError) as raised:
        effort_projection(objects, fovs, feature="x", hierarchy=hierarchy, strata=[])
    assert str(raised.value) == (
        "hierarchy columns missing from the FOV table: ['biological_repeat']"
    )


def test_replay_acquisitions_checks_the_order_and_stage_columns():
    objects, fovs = _tables()
    with pytest.raises(ValueError) as raised:
        _replay(reservoir_fraction=1.5)
    assert str(raised.value) == "reservoir_fraction must be between 0 and 1, not 1.5"
    with pytest.raises(ValueError) as raised:
        _replay(reservoir_fraction=0.3, acquisition_order_col="feature_a")
    assert str(raised.value) == "acquisition order cannot also be a phenotype feature"
    with pytest.raises(ValueError) as raised:
        _replay(reservoir_fraction=0.3, acquisition_order_col="acquired_at")
    assert str(raised.value) == "fovs is missing required columns: ['acquired_at']"
    with pytest.raises(ValueError) as raised:
        replay_acquisitions(
            objects.assign(stage_x=1.0),
            fovs,
            feature_cols=[*FEATURE_COLS, "stage_x"],
            strategies=FIXED,
            criteria=[MEAN],
            objective=POPULATION,
            reservoir_fraction=0.3,
            stage_cols=("stage_x", "stage_y"),
        )
    assert str(raised.value) == "stage coordinates cannot also be phenotype features"


def test_order_efficiency_frontier_and_redundancy_refuse_what_they_cannot_read():
    objects, fovs = _tables()
    with pytest.raises(TypeError, match="^replay must be a ReplayResult$"):
        order_efficiency(None)
    orders = [AcquisitionOrder.permuted(0)]
    plain = _replay(reservoir_fraction=0.3, n_splits=1, orders=orders)
    with pytest.raises(ValueError) as raised:
        order_efficiency(plain)
    expected = "the replay has no stage paths; run replay_acquisitions with stage_cols"
    assert str(raised.value) == expected
    staged = _replay(
        reservoir_fraction=0.3, n_splits=1, orders=orders, stage_cols=("stage_x", "stage_y")
    )
    with pytest.raises(ValueError) as raised:
        order_efficiency(staged, stage_speed=0)
    assert str(raised.value) == "stage_speed must be a positive, finite number"
    one = family(staged, "fixed", {FIXED[0].config_id: 3})
    two = family(staged, "fixed", {FIXED[1].config_id: 5})
    with pytest.raises(ValueError) as raised:
        frontier(staged, [one, two], criterion=MEAN.name, alphas=[0.2], unit_cols=[])
    assert str(raised.value) == "family names used twice: ['fixed']"
    with pytest.raises(ValueError) as raised:
        redundancy_distance(
            objects,
            fovs,
            feature_cols=["feature_a"],
            stage_cols=("x", "y"),
            bin_edges=[0, 1],
            share=0,
        )
    assert str(raised.value) == "share must be in (0, 1]"


def test_microscopist_report_checks_the_minimum_detection():
    with pytest.raises(ValueError) as raised:
        microscopist_report(minimum_detection=0)
    assert str(raised.value) == "minimum_detection must be a share in (0, 1]"


def test_inspect_fovs_skips_a_summary_without_values_and_checks_the_stage_columns():
    objects, fovs = _tables()
    result = inspect_fovs(
        objects.assign(empty=np.nan),
        fovs,
        feature_cols=[*FEATURE_COLS, "empty"],
        summaries=[Summary("feature_a", "mean"), Summary("empty", "mean")],
        count_ratio=None,
    )
    assert result.settings["skipped_summaries"] == [
        {
            "group": "condition=control, biological_repeat=R1",
            "summary": "mean of empty",
            "reason": "has no value here",
            "label": "control · repeat R1",
        }
    ]
    with pytest.raises(ValueError) as raised:
        inspect_fovs(objects, fovs, feature_cols=list(FEATURE_COLS), stage_cols=("x", "y"))
    assert str(raised.value) == "stage columns missing from the FOV table: ['x', 'y']"


def test_overview_full_names_planes_and_a_missing_condition_or_repeat_column():
    objects, fovs = _tables()
    planes = objects.assign(z=objects.groupby("image_file").cumcount() % 3)
    text = dataset_overview(
        planes, fovs, feature_cols=list(FEATURE_COLS), observation_unit="slice"
    ).summary(detail="full")
    assert text.splitlines()[2] == (
        "'z' takes 3 values per field: slices of several planes, channels or frames of one "
        "field (declared as plane-level rows)."
    )
    no_condition = dataset_overview(
        objects.drop(columns="condition"),
        fovs.drop(columns="condition"),
        feature_cols=list(FEATURE_COLS),
    ).summary(detail="full")
    assert "\nno condition column; 1 repeat (biological_repeat: R1).\n" in no_condition
    no_repeat = dataset_overview(
        objects.drop(columns="biological_repeat"),
        fovs.drop(columns="biological_repeat"),
        feature_cols=list(FEATURE_COLS),
    ).summary(detail="full")
    assert (
        "\n1 condition (condition: control); no repeat column (one implicit repeat).\n" in no_repeat
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


def test_effort_short_parts_says_how_to_answer_a_single_unit_next_time():
    reason = (
        "a single 'biological_repeat' unit holds observations, so the variance at that level "
        "cannot be separated"
    )
    skipped = pd.DataFrame({"stratum": ["all"], "reason": [reason]})
    empty = pd.DataFrame(columns=["stratum"])
    settings = {"feature": "area", "hierarchy": {"fov": "fov", "sample": "well"}}
    result = EffortResult(empty, empty, empty, (), settings, skipped)
    assert effort_short_parts(result) == (
        [
            (
                "  all: more fields or more samples, for the mean area? Not estimable: a single "
                "'biological_repeat'\n    unit holds observations, so the variance at that level "
                "cannot be separated."
            ),
            (
                "  To answer it next time, image at least 2 samples of each condition in one "
                "biological repeat: the\n    variance between them is what this experiment cannot "
                "measure."
            ),
        ],
        [],
    )


def test_next_time_status_without_answers_and_with_a_target_not_judged(monkeypatch):
    protocol = SimpleNamespace(settings={})
    monkeypatch.setattr(report_status_plan, "protocol_answers", lambda protocol: [])
    assert report_status_plan.next_time_status(None, protocol) == (
        report_status.Status.NOT_ASSESSED,
        "not assessed",
    )
    answers = [
        {"label": "the mean", "served": None, "unreached": True, "never": True, "units": 5},
        {"label": "the spread", "served": 3, "units": 5},
    ]
    monkeypatch.setattr(report_status_plan, "protocol_answers", lambda protocol: answers)
    assert report_status_plan.next_time_status(None, protocol) == (
        report_status.Status.CAUTION,
        "3 fields served every sample for 1 of 2 targets; 1 of 2 targets not judged",
    )


def test_controls_status_not_judged_and_without_false_alarms():
    status = report_status_controls.controls_status
    Status = report_status.Status
    unusable = SimpleNamespace(status="not assessed", settings={"reason": "unusable"})
    assert status(unusable) == (Status.CAUTION, "a control was declared, but could not be judged")
    undeclared = SimpleNamespace(status="not assessed", settings={"reason": "none"})
    assert status(undeclared) == (Status.NOT_ASSESSED, "no positive control declared")
    pooled = pd.DataFrame({"candidate": ["reference"], "rate": [0.1]})
    alarms_only = ControlResult(
        pooled.assign(stratum="all FOVs"), pooled, "positive only", (), {"control": {}}
    )
    assert status(alarms_only) == (Status.CAUTION, "positive-control fields: not judged")
    pooled = pd.DataFrame({"candidate": ["positive"], "rate": [0.95]})
    settings = {"control": {"positive": {"condition": ["KD"]}}}
    positive = ControlResult(
        pooled.assign(stratum="all FOVs"), pooled, "positive only", (), settings
    )
    assert status(positive) == (
        Status.CAUTION,
        "KD fields recognized in 95% of checks; false alarms not assessed",
    )


def test_enough_status_with_late_stops_and_nothing_new_not_shown(monkeypatch):
    tallies = [
        {"label": "the mean", "data_rule": True, "groups": 2, "met": 2}
        | {"warned": False, "flagged": False, "lost": False},
        {"label": "nothing new", "data_rule": True, "groups": 2, "met": 0}
        | {"warned": False, "flagged": False, "lost": False},
    ]
    monkeypatch.setattr(report_status, "rule_tallies", lambda sufficiency: tallies)
    monkeypatch.setattr(report_status, "has_resolution", lambda contrast: False)
    monkeypatch.setattr(report_status, "_premature", lambda calibration: 0.5)
    monkeypatch.setattr(report_status, "group_word", lambda result: "group")
    assert report_status.enough_status(object()) == (
        report_status.Status.CAUTION,
        (
            "1 of 2 rules said enough in every group; 'nothing new' not yet shown; held-back "
            "fields: some rules stopped too early (up to 50%)"
        ),
    )


def test_conditions_status_from_contrasts_alone_and_from_a_ranking_alone(monkeypatch):
    module, Status = report_status_conditions, report_status.Status
    monkeypatch.setattr(module, "readings_headline", lambda contrast: "")
    monkeypatch.setattr(module, "condition_readings", lambda c: pd.DataFrame({"verdict": []}))
    monkeypatch.setattr(module, "across_tested", lambda contrast: False)
    monkeypatch.setattr(module, "reagents_headline", lambda contrast: "")
    monkeypatch.setattr(module, "secondary_headline", lambda contrast: "")
    monkeypatch.setattr(module, "opposite_signs", lambda *arguments: True)
    monkeypatch.setattr(module, "across_headline", lambda across, col: f"across ({col})")
    table = pd.DataFrame(
        {
            "status": ["ok", "ok"],
            "d_fields": [0.5, -1.2],
            "p_bonferroni": [0.01, 0.2],
            "p_value": [0.01, 0.2],
            "condition": ["a", "b"],
        }
    )
    settings = {"repeat_col": "biological_repeat"}
    contrast = SimpleNamespace(table=table, repeats=None, settings=settings)
    assert module.conditions_status(contrast, None) == (
        Status.CAUTION,
        (
            "2 contrasts, |d| on field means 0.5 to 1.2; 1 of 2 tested differ (Bonferroni "
            "p ≤ 0.05); opposite signs between repeats"
        ),
    )
    contrast = SimpleNamespace(table=table, repeats=pd.DataFrame({"x": [1]}), settings=settings)
    assert module.conditions_status(contrast, None) == (
        Status.CAUTION,
        (
            "across (biological_repeat); within repeats: 1 of 2 tested differ (Bonferroni p ≤ "
            "0.05); opposite signs between repeats"
        ),
    )
    untested = SimpleNamespace(ranking=pd.DataFrame({"score": [1.0, None], "tests": [0, 0]}))
    assert module.conditions_status(None, untested) == (
        Status.CAUTION,
        "2 features not ranked (no contrast could be tested)",
    )
    tested = SimpleNamespace(ranking=pd.DataFrame({"score": [1.0, None], "tests": [1, 0]}))
    assert module.conditions_status(None, tested) == (Status.OK, "1 of 2 features pass the gate")
    empty = SimpleNamespace(ranking=pd.DataFrame())
    assert module.conditions_status(None, empty) == (Status.NOT_ASSESSED, "not assessed")


def test_detection_adjusted_bounds_never_recognized_and_without_bounds(monkeypatch):
    monkeypatch.setattr(report_profile, "_mismatch", lambda *arguments: None)
    controls = SimpleNamespace(status="assessed", pair_cols=(), table=pd.DataFrame(), settings={})
    frame = pd.DataFrame(
        {"confidence": [0.95], "unseen_state_max_frequency": [float("nan")], "g": ["A"]}
    )
    sufficiency = SimpleNamespace(next_fov=frame, group_cols=("g",))
    adjusted = report_profile._detection_adjusted_bounds
    monkeypatch.setattr(report_profile, "_detection", lambda c: ("positive_lofo", 0, 10))
    assert adjusted(sufficiency, controls) == (
        "The declared positive control was never recognized against the reference fields of an "
        "analysis group, so the unseen-state bound says nothing about a phenotype seen like it."
    )
    monkeypatch.setattr(report_profile, "_detection", lambda c: ("positive_lofo", 8, 10))
    assert adjusted(sufficiency, controls) is None
    frame = pd.DataFrame(
        {"confidence": [0.95, 0.95], "unseen_state_max_frequency": [None, 0.1], "g": ["A", "B"]}
    )
    assert adjusted(SimpleNamespace(next_fov=frame, group_cols=("g",)), controls) == (
        "Adjusted for the declared control's detection rate (80% of its fields recognized "
        "against the reference fields of an analysis group; at least 49% at 95% confidence, of "
        "10 checks): a phenotype seen as well as the control, and shown by no FOV yet, could "
        "still be in\n  g=B: up to 20% of FOVs (the 10% bound above divided by 49%), if "
        "detection is independent between fields.\n  This holds for whole-field phenotypes "
        "like the control only: a small state in an ordinary field is recognized less often "
        "(declare planted_objects to measure it)."
    )
