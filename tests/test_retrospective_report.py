"""Tests for the assembled microscopist's report (docs/report/questions.md)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    KEY,
    _acquisition,
    _adequacy,
    _effort,
    _grid_well,
    _inspect,
    _plate_protocol,
    _pooled,
    _scenario_reliability,
    _syn_inspection,
    _two_plates,
    _weak_control_report,
    _well_values,
)
from _helpers_sampling_unit import CONTROL_CONDITION, _five_wells, _flat_text, _widths
from _helpers_strategies import _flat, _named_scenario_report
from _helpers_workflows import _setup
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import ScalarPrecisionStrategy
from morphocoverage._style import CAUTION, NOT_ASSESSED, OK, PROBLEM, Status, width
from morphocoverage._wording import percent
from morphocoverage.bounds import lower_bound
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    RepresentationControl,
    SamplingHierarchy,
    design_effect,
    effort_projection,
    effort_text,
    microscopist_report,
    redundancy_distance,
    report_profile,
    representation_control,
    sampling_reliability,
)
from morphocoverage.retrospective._text import median_beyond as _median_beyond
from morphocoverage.retrospective._text import order_phrase
from morphocoverage.retrospective.controls import ControlResult
from morphocoverage.retrospective.controls_text import pooled_over
from morphocoverage.retrospective.inspection import InspectionResult
from morphocoverage.retrospective.report import DECLARATION_TITLE, LIMITS_TITLE, QUESTIONS
from morphocoverage.retrospective.report_answers import ASSESSED
from morphocoverage.retrospective.report_enough import detection_line
from morphocoverage.retrospective.report_profile import (
    _detection_adjusted_bounds,
    _new_state_rows,
    _redundancy_rows,
    sampling_profile,
)
from morphocoverage.retrospective.report_status import controls_status, profile_status
from morphocoverage.retrospective.spatial import RedundancyResult
from morphocoverage.sampling_unit import condition_contrast
from morphocoverage.strategies import SupportNoveltyEstimator
from tests.one_acquisition import TEST_ACQUISITION, stamped


def _measured(table: pd.DataFrame) -> RedundancyResult:
    """A hand-made redundancy table as a result recording the shared test acquisition."""
    return RedundancyResult(table, {"acquisition": TEST_ACQUISITION})


def test_an_empty_report_says_what_each_question_needs():
    report = microscopist_report()
    text = report.summary()
    for _, title in QUESTIONS:
        assert title in text
    assert text.count("Not assessed:") == len(QUESTIONS)
    # No definition of new ran: no "nothing new" answer to qualify (audit 10, C24).
    assert "No representation control" not in " ".join(text.split())
    assert "Every answer holds for the stated features." in " ".join(text.split())
    assert report.profile.empty


def test_the_overview_opens_and_the_limits_close_the_report():
    from morphocoverage.retrospective import dataset_overview
    from morphocoverage.retrospective.report import LIMITS_TITLE, OVERVIEW_TITLE

    built = SCENARIOS["condition_repeat_groups"]()
    overview = dataset_overview(built.objects, built.fovs, feature_cols=list(FEATURE_COLS))
    assert overview.counts["objects"] == len(built.objects)
    assert overview.counts["fields"] == len(built.fovs)
    assert overview.counts["features"] == len(FEATURE_COLS)
    report = microscopist_report(overview=overview)
    text = report.summary(detail="full")
    marker = report.statuses["overview"].marker
    assert text.startswith(f"{marker} {OVERVIEW_TITLE}\n")
    assert "feature column" in text and "condition" in text.split(DECLARATION_TITLE)[0]
    assert text.rstrip().split("\n\n")[-1].startswith(LIMITS_TITLE)
    # The short form opens with its title and "At a glance", then the overview.
    short = report.summary()
    assert short.startswith("Microscopist's report — ") and "\nAt a glance\n" in short
    assert f"\n\n{marker} {OVERVIEW_TITLE}\n" in short
    assert short.rstrip().split("\n\n")[-1].startswith(LIMITS_TITLE)
    limits = report.limits()
    # A line pointing to the section, which states the need (owner, 2026-10-06).
    assert "1. Can these features detect what I care about?: not assessed, see question 1." in (
        limits
    )
    reasons = report.limits(pointers=False)
    assert "1. Can these features detect what I care about?: not assessed; needs:" in reasons
    assert "Every answer holds for the stated features" in limits  # the notes close it
    assert "Read with care" not in text
    # An overview with many features lists the first twelve and counts the rest.
    many = built.objects.assign(**{f"f{i}": 0.0 for i in range(15)})
    columns = [*FEATURE_COLS, *(f"f{i}" for i in range(15))]
    wide = dataset_overview(many, built.fovs, feature_cols=columns)
    assert "and 5 more" in wide.summary(detail="full")
    assert "and 14 more" in wide.summary()  # the short form lists three
    single = SCENARIOS["homogeneous"]()
    bare = dataset_overview(
        single.objects.drop(columns=["condition", "biological_repeat"]),
        single.fovs.drop(columns=["condition", "biological_repeat"]),
        feature_cols=list(FEATURE_COLS),
    )
    assert "No condition or repeat column" in bare.summary(detail="full")
    assert "every field in one group (pooled)" in bare.summary()
    # A selection of questions limits the closing section to those questions.
    assert "4. How many fields" not in report.summary(["enough"])


def test_the_declaration_is_printed_first_only_when_given():
    assert DECLARATION_TITLE not in microscopist_report().summary()
    report = microscopist_report(declaration=["Population summary: the mean of x within 5%."])
    text = report.summary(detail="full")
    assert text.startswith(f"{DECLARATION_TITLE}\n  - Population summary")
    assert text.index(DECLARATION_TITLE) < text.index("Foundation: why count")
    assert report.summary(["enough"], detail="full").startswith(DECLARATION_TITLE)
    # In the short form the declaration follows the title and "At a glance".
    short = report.summary()
    assert short.index("At a glance") < short.index(DECLARATION_TITLE)
    assert short.index(DECLARATION_TITLE) < short.index("Foundation: why count fields, and")
    with pytest.raises(TypeError, match="declaration"):
        microscopist_report(declaration="one string, not a sequence")


def test_given_answers_are_placed_under_their_question():
    built = SCENARIOS["condition_repeat_groups"]()
    criterion = OutcomeCriterion(
        RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean of feature_a"
    )
    reliability = sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[criterion],
        fov_counts=[2, 4],
        reservoir_fraction=0.3,
        n_splits=2,
        draws=5,
    )
    report = microscopist_report(reliability=reliability)
    section = report.sections["next_time"]
    assert "how many fields do you need for the mean of feature_a?" in section
    assert report.inputs["next_time"] == "given"
    assert report.sections["effort"].strip().startswith("Not assessed")
    text = report.summary(detail="full")
    assert text.index("4. How many fields") < text.index("how many fields do you need")
    assert text.index("how many fields do you need") < text.index("5. What did")


def test_the_profile_states_measured_numbers_with_their_source():
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for sample in range(6):
        offset = rng.normal(0, 1.0)
        for field in range(5):
            shift = rng.normal(0, 0.5)
            name = f"s{sample}_f{field}"
            fovs.append({"image_file": name, "well": f"W{sample}", "condition": "c"})
            values = offset + shift + rng.normal(0, 1.0, 30)
            rows.append(pd.DataFrame({"image_file": name, "x": values}))
    objects = pd.concat(rows, ignore_index=True)
    fovs = pd.DataFrame(fovs)
    objects = objects.merge(fovs, on="image_file")
    effort = effort_projection(
        objects, fovs, feature="x", hierarchy=SamplingHierarchy("image_file", "well"), strata=[]
    )
    redundancy = pd.DataFrame(
        {"redundancy_distance": [336.0, 520.0], "smallest_lag": [336.0, 336.0], "unit": "µm"}
    )
    report = microscopist_report(effort=stamped(effort), redundancy=_measured(redundancy))
    profile = report.profile.set_index("item")
    effect = design_effect(effort, "all")
    assert profile.loc["field-to-field heterogeneity", "value"] == pytest.approx(effect)
    assert "design effect" in profile.loc["field-to-field heterogeneity", "statement"]
    assert "in 1 of 2 groups, fields closer than about 520 µm" in report.sections["profile"]
    assert set(profile["source"]) == {"effort_projection", "redundancy_distance"}
    # Redundancy alone does not answer question 8 (G9); its caveat stays with the profile.
    assert "sampling-frame bias is not assessed" in report.sections["profile"]
    assert report.inputs["order"] == "not assessed"


def test_a_declared_control_joins_the_first_question_and_the_profile():
    rng = np.random.default_rng(1)
    fov_rows, object_rows = [], []
    for well, condition, offset in (("A1", "MOCK", 0.0), ("B1", "KD", 6.0)):
        for i in range(10):
            name = f"{well}_{i}"
            fov_rows.append(
                {
                    "image_file": name,
                    "condition": condition,
                    "biological_repeat": "P1",
                    "repeat_group": "P1",
                }
            )
            values = rng.normal(0, 1, size=(20, 2))
            values[:, 0] += offset
            object_rows.append(
                pd.DataFrame(values, columns=list(FEATURE_COLS)).assign(
                    image_file=name, condition=condition, biological_repeat="P1"
                )
            )
    objects, fovs = pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)
    control = representation_control(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        novelty=SupportNoveltyEstimator(1.0, 1, None, 3, 3),
        control=RepresentationControl(
            positive={"condition": "KD"}, reference={"condition": "MOCK"}
        ),
        pair_cols=["biological_repeat"],
        fovs_seen=5,
        draws=5,
    )
    report = microscopist_report(controls=control)
    assert "KD fields (positive control)" in report.sections["controls"]
    assert "recognized as different" in report.short["controls"]
    assert "No representation control" not in report.summary()
    assert (report.profile["item"] == "representation").sum() == 2


def test_results_of_the_wrong_kind_are_refused():
    with pytest.raises(TypeError, match="effort"):
        microscopist_report(effort=pd.DataFrame())


def test_fields_alike_at_every_distance_are_the_strongest_redundancy_not_none():
    redundancy = pd.DataFrame(
        {
            "redundancy_distance": [None, 336.0],
            "smallest_lag": [119.0, 336.0],
            "largest_lag": [350.0, 900.0],
            "unit": "µm",
        }
    )
    profile = microscopist_report(redundancy=_measured(redundancy)).sections["profile"]
    assert "in 1 of 2 groups, fields are more alike than random pairs at every distance" in profile
    assert "no group shows" not in profile


def test_the_overview_counts_technical_repeats_within_their_condition_and_repeat():
    from morphocoverage.retrospective import dataset_overview

    built = SCENARIOS["condition_repeat_groups"]()
    fovs = built.fovs.copy()
    # Two wells per condition and repeat, named alike everywhere: still distinct units.
    fovs["technical_repeat"] = ["w1" if i % 2 else "w2" for i in range(len(fovs))]
    objects = built.objects.merge(
        fovs[["condition", "biological_repeat", "image_file", "technical_repeat"]],
        on=["condition", "biological_repeat", "image_file"],
    )
    overview = dataset_overview(objects, fovs, feature_cols=list(FEATURE_COLS))
    groups = fovs.groupby(["condition", "biological_repeat"]).size()
    assert overview.counts["technical_repeats"] == 2 * len(groups)
    assert (overview.layout["technical_repeats"] == 2).all()
    text = overview.summary(detail="full")
    assert f"{2 * len(groups)} technical repeats (technical_repeat, a well or a dish" in text
    assert "2 per condition and repeat" in text and "technical repeats" in text.split("\n")[4]
    assert f"{2 * len(groups)} wells (technical repeats; 2 per condition and repeat)" in (
        " ".join(overview.summary().split())
    )
    plain = dataset_overview(built.objects, built.fovs, feature_cols=list(FEATURE_COLS))
    assert plain.counts["technical_repeats"] == 0 and "technical" not in plain.summary()


def test_f16_order_phrases_are_defined_once():
    from morphocoverage.retrospective import adequacy, calibration

    assert not hasattr(adequacy, "_ORDER_WORDS") and not hasattr(calibration, "_ORDER_WORDS")
    assert order_phrase("permuted") == "in random orders"
    assert order_phrase("recorded", capital=True) == "In the recorded order"
    assert order_phrase("custom") == "custom"


def test_g14_a_sampling_frame_caveat_alone_does_not_assess_question_3():
    redundancy = pd.DataFrame(
        [
            {
                "condition": "c",
                "redundancy_distance": None,
                "smallest_lag": None,
                "largest_lag": None,
                "fov_mean_variance": None,
                "share": 0.9,
                "unit": "µm",
            }
        ]
    )
    report = microscopist_report(redundancy=_measured(redundancy))
    assert report.profile.empty
    assert report.inputs["profile"] == "not assessed"
    assert "3. What is limiting my sampling?: not assessed" in report.limits()
    assert any("sampling-frame bias is not assessed" in note for note in report.notes)


def test_g18_the_split_modules_keep_the_names_callers_use():
    from morphocoverage.retrospective import report, report_profile

    assert report.sampling_profile is report_profile.sampling_profile
    assert "sampling_profile" in report_profile.__all__


def test_g2_the_chance_expectation_counts_only_fields_judged_for_new_states(synthetic):
    objects, fovs = synthetic
    short = (fovs["biological_repeat"] == "2026_01_19") & fovs["image_file"].isin(
        ["field_05.tif", "field_06.tif"]
    )
    fovs = fovs[~short]
    objects = objects.merge(fovs[KEY])
    inspection = _syn_inspection(objects, fovs)
    assert len(inspection.settings["skipped_novelty"]) == 10
    assert inspection.settings["n_fovs"] == 160
    assert inspection.settings["n_fovs_judged"] == 120
    controls = _pooled(reference_lofo=(3, 60))
    statement = _new_state_rows(inspection, controls)[0]["statement"]
    assert "of the 120 fields judged, so these are compatible with chance" in statement


def test_g7_the_redundancy_distance_is_a_point_estimate_with_its_pairs():
    redundancy = pd.DataFrame(
        {
            "redundancy_distance": [336.0, 520.0, 700.0],
            "pairs_closer": [12, 40, 25],
            "smallest_lag": [100.0, 100.0, 100.0],
            "unit": "µm",
        }
    )
    statement = _redundancy_rows(redundancy)[0]["statement"]
    assert "the median of point estimates, not tested against chance" in statement
    assert "from 12 to 40 pairs of fields closer than that per group" in statement


def test_g10_small_shares_are_not_written_as_zero_percent():
    sufficiency = SimpleNamespace(
        next_fov=pd.DataFrame({"g": ["a"], "unseen_state_max_frequency": [0.004]}),
        group_cols=("g",),
    )
    text = _detection_adjusted_bounds(sufficiency, _pooled(positive_lofo=(19, 20)))
    # Audit 4 (P3-9): divided by the one-sided lower bound of 19 of 20 (78%), not by 95%.
    assert "up to 0.5% of FOVs (the 0.4% bound above divided by 78%)" in text
    compared = pd.DataFrame(
        {
            "samples": [40],
            "fovs_per_sample": [3.0],
            "ratio": [0.996],
            "ratio_low": [0.993],
            "ratio_high": [0.999],
            "same_total": [True],
            "resolved": [True],
            "current_samples": [30],
            "current_fovs_per_sample": [4.0],
        }
    )
    line = effort_text._allocation_line(compared, {}, 12, "biological_repeat", "sample")
    assert "0.4% smaller than the acquired" in line


def test_g15_the_closing_notes_do_not_point_below():
    controls = _pooled(positive=(2, 20), reference=(0, 20))
    built = microscopist_report(controls=controls)
    note = next(n for n in built.notes if n.startswith("Weak representation"))
    assert "below" not in note.split(")", 1)[1]
    assert "adequacy answers (questions 2, 4 and 5)" in note
    assert "only 10% of checks (below 90%)" in note


def test_g16_the_profile_labels_each_rate_by_how_it_was_measured():
    controls = _pooled(positive=(18, 20), reference=(1, 20), reference_lofo=(2, 30))
    controls = replace(
        controls,
        settings={**controls.settings, "lofo_group_cols": ["condition", "biological_repeat"]},
    )
    profile = sampling_profile(controls, None, None, None)
    statements = list(profile["statement"])
    assert "90% of declared positive-control fields recognized as different after 10 " in (
        statements[0] + " "
    )
    assert any("after 10 reference fields (false alarms)" in s for s in statements)
    lofo = profile[profile["item"] == "representation, as inspect_fovs judges"]
    assert "their analysis group (condition, biological_repeat)" in lofo["statement"].iloc[0]


def test_g20_rates_of_another_definition_of_new_are_not_combined():
    controls = _pooled(reference_lofo=(1, 30), positive_lofo=(9, 10))
    controls = replace(
        controls, settings={**controls.settings, "novelty_definition": "one definition"}
    )
    inspection = InspectionResult(
        pd.DataFrame({"reason": ["new state"], "condition": ["a"], "f": ["w"]}),
        ("condition",),
        {"fov_key": ["condition", "f"], "n_fovs_judged": 30, "novelty_definition": "another"},
    )
    statement = _new_state_rows(inspection, controls)[0]["statement"]
    assert "not compared with the declared reference's false alarms" in statement
    assert "false alarms alone" not in statement
    sufficiency = SimpleNamespace(
        next_fov=pd.DataFrame({"condition": ["a"], "unseen_state_max_frequency": [0.1]}),
        group_cols=("condition",),
        novelty_definition="another",
    )
    text = _detection_adjusted_bounds(sufficiency, controls)
    assert text.startswith("The unseen-state bound is not adjusted for detection")
    controls = replace(
        controls,
        settings={**controls.settings, "lofo_group_cols": ["condition", "biological_repeat"]},
    )
    sufficiency.novelty_definition = "one definition"
    assert "measured within the groups condition, biological_repeat" in _detection_adjusted_bounds(
        sufficiency, controls
    )


def _controls(settings=None, **rates):
    """A ControlResult holding only pooled counts: kind -> (judged new, judged)."""
    pooled = pd.DataFrame(
        [
            {"candidate": k, "judged_new": n, "candidates": t, "not_judged": 0, "rate": n / t}
            for k, (n, t) in rates.items()
        ]
    )
    return ControlResult(
        pooled.assign(stratum="all FOVs"),
        pooled,
        "assessed",
        ("biological_repeat",),
        {"fovs_seen": 10, "acquisition": TEST_ACQUISITION, **(settings or {})},
    )


def _sufficiency(bound=0.1):
    frame = pd.DataFrame({"g": ["a"], "unseen_state_max_frequency": [bound], "confidence": [0.95]})
    return type("S", (), {"next_fov": frame, "group_cols": ("g",)})()


def test_p6_4_the_unseen_bound_is_adjusted_by_the_planted_rate_when_declared():
    planted = {"control": {"planted_objects": 5}}
    controls = _controls(planted, positive=(29, 30), positive_lofo=(29, 30), planted_lofo=(6, 30))
    text = _detection_adjusted_bounds(_sufficiency(), controls)
    low = lower_bound(6, 30, 0.95)
    assert "for 5 planted positive-control cells" in text
    assert "whole-field phenotypes" not in text
    assert f"up to {percent(min(1.0, 0.1 / low))} of FOVs" in text
    assert f"divided by {percent(low)}" in text
    # Without planted objects the whole-field rate is used, and its scope is said.
    whole = _detection_adjusted_bounds(_sufficiency(), _controls(positive_lofo=(29, 30)))
    assert "whole-field phenotypes like the control only" in whole
    # The weak-representation warning is gated on the planted rate when declared.
    controls = _controls(planted, positive=(29, 30), planted=(6, 30), positive_lofo=(29, 30))
    report = microscopist_report(controls=controls)
    note = next(n for n in report.notes if n.startswith("Weak representation"))
    assert "ordinary fields with 5 planted positive-control objects" in note
    # Without a definition of new, no "nothing new" answer to qualify (audit 10, C24).
    no_control = microscopist_report()
    assert not any("these features and this definition of new" in n for n in no_control.notes)


def test_p3_9_the_adjusted_bound_divides_by_the_low_end_of_the_detection_rate():
    text = _detection_adjusted_bounds(_sufficiency(0.004), _controls(positive_lofo=(19, 20)))
    low = lower_bound(19, 20, 0.95)
    assert f"divided by {percent(low)}" in text and "divided by 95%" not in text
    assert "at least 78% at 95% confidence, of 20 checks" in text


def test_p8_2_redundancy_distance_is_tested_against_shuffled_positions():
    rng = np.random.default_rng(3)
    fovs, objects = [], []
    for well, trend in (("flat", 0.0), ("trend", 1.0)):
        f, o = _grid_well(rng, well, trend)
        fovs += f
        objects += o
    result = redundancy_distance(
        pd.DataFrame(objects),
        pd.DataFrame(fovs),
        feature_cols=["x"],
        stage_cols=("x_um", "y_um"),
        bin_edges=[0, 600, 1200, 1800, 2400, 3000, 4000],
        n_permutations=99,
    )
    table = result.table.set_index("technical_repeat")
    assert bool(table.loc["trend", "beyond_chance"]) and table.loc["trend", "null_share"] < 0.05
    assert not bool(table.loc["flat", "beyond_chance"])
    assert table.loc["flat", "null_share"] > 0.05
    # The report counts only the group beyond chance and says the other is within it.
    assert table.equals(result.table.set_index("technical_repeat"))
    report = microscopist_report(redundancy=result)
    profile = " ".join(report.sections["profile"].split())
    assert "in 1 of 2 groups, fields closer than about" in profile
    assert "beyond what shuffling the positions within each group gives" in profile
    # A point estimate the shuffles do not support is named as such, not as redundancy.
    doubtful = table.reset_index().assign(
        redundancy_distance=[1200.0, 1200.0], beyond_chance=[False, True], null_share=[0.4, 0.01]
    )
    statement = _redundancy_rows(doubtful)[0]["statement"]
    assert "in 1 of 2 groups the point estimate suggested that closer fields" in statement
    assert "in a median of 40% of shuffles, so no spatial redundancy is shown there" in statement


def test_f6_order_medians_are_over_the_same_runs():
    assert _median_beyond(pd.Series([10.0, np.nan, np.nan])) is None
    assert _median_beyond(pd.Series([10.0, 20.0, np.nan])) == 20.0
    assert _median_beyond(pd.Series([None, None])) is None


def test_g9_g10_g14_the_report_marks_questions_by_what_answers_them():
    redundancy = pd.DataFrame(
        {"redundancy_distance": [520.0], "smallest_lag": [336.0], "unit": "µm"}
    )
    report = microscopist_report(redundancy=_measured(redundancy))
    assert report.inputs["order"] == "not assessed"
    assert "8. Could I" in report.limits()
    empty = microscopist_report()
    assert "any of the analyses below" not in empty.summary()
    assert "adequacy_points" not in empty.sections["enough"]


def test_a_weakly_detected_control_prefixes_every_adequacy_answer():
    controls, sufficiency = _weak_control_report()
    detection = float(controls.pooled.set_index("candidate")["rate"]["positive"])
    assert detection < 0.9
    report = microscopist_report(controls=controls, sufficiency=sufficiency)
    warning = "Weak representation: fields of the declared positive control were recognized"
    assert report.sections["enough"].lstrip().startswith(warning)
    assert any(note.startswith(warning) for note in report.notes)
    relaxed = microscopist_report(controls=controls, sufficiency=sufficiency, minimum_detection=0.1)
    assert warning not in relaxed.summary()
    assert any("Technical drift" in note for note in relaxed.notes)


def test_the_unseen_state_bound_is_divided_by_the_controls_detection_rate():
    controls, sufficiency = _weak_control_report()
    # The rate against every reference field, as the bound judges fields (audit 2, G-13).
    row = controls.pooled.set_index("candidate").loc["positive_lofo"]
    detection = float(row["rate"])
    bound = float(sufficiency.next_fov["unseen_state_max_frequency"].iloc[0])
    section = microscopist_report(controls=controls, sufficiency=sufficiency).sections["enough"]
    # Audit 4 (P3-9): divided by the rate's one-sided lower bound, not its point estimate.
    low = lower_bound(int(row["judged_new"]), int(row["candidates"]), 0.95)
    adjusted = min(1.0, bound / low)
    assert f"Adjusted for the declared control's detection rate ({100 * detection:.0f}%" in section
    assert (
        f"up to {100 * adjusted:.0f}% of FOVs (the {100 * bound:.0f}% bound above divided by"
        in section
    )
    assert "if detection is independent between fields" in section
    without = microscopist_report(sufficiency=sufficiency).sections["enough"]
    assert "Adjusted for" not in without


def test_a_strong_plate_and_a_weak_one_are_said_to_be_pooled_wherever_the_rate_is_used():
    # P1 sees the control in every check, P2 in half: one pooled rate for both, said so.
    controls = _two_plates(
        positive=((10, 10), (5, 10)),
        positive_lofo=((10, 10), (5, 10)),
        reference=((0, 10), (1, 10)),
        reference_lofo=((0, 10), (1, 10)),
    )
    assert pooled_over(controls) == "pooled over all plates"
    one_plate = ControlResult(
        controls.table.query("plate == 'P1'"), controls.pooled, "assessed", ("plate",), {}
    )
    assert pooled_over(one_plate) == ""
    sufficiency = SimpleNamespace(
        next_fov=pd.DataFrame(
            {"plate": ["P1", "P2"], "unseen_state_max_frequency": [0.1, 0.1], "confidence": 0.95}
        ),
        group_cols=("plate",),
    )
    bounds = _detection_adjusted_bounds(sufficiency, controls)
    assert "of 20 checks, pooled over all plates):" in bounds
    assert "plate=P1: up to" in bounds and "plate=P2: up to" in bounds
    line, _ = detection_line(sufficiency, controls)
    assert "at 95% confidence, pooled over all plates):" in line
    inspection = InspectionResult(
        pd.DataFrame({"reason": ["new state"], "plate": ["P2"], "f": ["w"]}),
        ("plate",),
        {"fov_key": ["plate", "f"], "n_fovs": 40, "n_fovs_judged": 40},
    )
    statement = _new_state_rows(inspection, controls)[0]["statement"]
    assert "the declared reference's false-alarm rate, pooled over all plates, each" in statement
    status, headline = controls_status(controls)
    assert status is Status.PROBLEM
    assert headline == (
        "weak representation: positive-control fields recognized in 75% of checks, pooled over "
        "all plates"
    )
    report = microscopist_report(controls=controls)
    note = next(n for n in report.notes if n.startswith("Weak representation"))
    assert "recognized in only 75% of checks, pooled over all plates (below 90%)" in note
    strong = _two_plates(positive=((10, 10), (9, 10)), reference=((0, 10), (1, 10)))
    headline = (
        "positive-control fields recognized in 95% of checks; false alarms 5%, both pooled "
        "over all plates"
    )
    assert controls_status(strong) == (Status.OK, headline)


def _redundancy(*groups):
    """One row per group: (group, smallest_lag, redundancy_distance, reason)."""
    return pd.DataFrame(
        [
            {
                "well": well,
                "smallest_lag": lag,
                "largest_lag": None if lag is None else 2000.0,
                "redundancy_distance": distance,
                "beyond_chance": None if lag is None else True,
                "null_share": None if lag is None else 0.01,
                "unit": "µm",
                "reason": reason,
            }
            for well, lag, distance, reason in groups
        ]
    )


ALIKE = ("A1", 100.0, 500.0, None)


UNLIKE = ("A2", 100.0, 100.0, None)


ONE_FIELD = ("A3", None, None, "only 1 field with observations: the semivariogram needs pairs")


def test_a_redundancy_group_with_no_estimate_is_not_counted_as_negative_evidence():
    both = _redundancy_rows(_redundancy(ALIKE, UNLIKE))[0]["statement"]
    assert both.startswith("in 1 of 2 groups, fields closer than about 500 µm")
    assert "not assessed" not in both
    mixed = _redundancy_rows(_redundancy(ALIKE, UNLIKE, ONE_FIELD))[0]["statement"]
    assert mixed.startswith("in 1 of the 2 groups assessed, fields closer than about 500 µm")
    assert mixed.endswith("; 1 more not assessed (too few fields)")
    none = _redundancy_rows(_redundancy(UNLIKE, ONE_FIELD))[0]["statement"]
    assert none.startswith("in the 1 group assessed, no group shows neighbouring fields")
    assert none.endswith("(100 µm); 1 more not assessed (too few fields)")


def test_microscopist_report_checks_the_minimum_detection():
    with pytest.raises(ValueError) as raised:
        microscopist_report(minimum_detection=0)
    assert str(raised.value) == "minimum_detection must be a share in (0, 1]"


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


def test_the_report_prints_the_short_forms_and_its_caveats_once():
    objects, fovs = _five_wells()
    contrast = condition_contrast(objects, fovs, feature="x", control=CONTROL_CONDITION, n_draws=5)
    inspection = _inspect(*_acquisition())
    report = microscopist_report(contrast=stamped(contrast), inspection=stamped(inspection))
    short = report.summary()
    # The answer across biological repeats first, then the contrasts within each repeat.
    # The reading per condition leads, then the table across repeats.
    assert report.short["conditions"].startswith("  Reading, condition by condition")
    assert "Across the 2 biological repeats" in report.short["conditions"]
    assert "Within each repeat: x, each condition against the control" in _flat_text(short)
    assert "user's inference" not in _flat_text(short)
    assert _flat_text(short).count("compare means") == 1
    assert report.short["inspect"] in short
    assert all(width <= 100 for width in _widths(report.short["inspect"]))
    full = report.summary(detail="full")
    assert "Feature x, each condition against the control" in full
    # The closing: what was not assessed, in a line each, and the notes as caveats.
    closing = short.split(LIMITS_TITLE)[1]
    # With a contrast, question 2 is answered from it (the difference resolved or not).
    assert f"{NOT_ASSESSED} 1. Can the features see the phenotype? — not assessed" in closing
    assert "2. Did I image enough? — not assessed" not in closing
    # A standing assumption of every answer: listed under "Assumptions:", not marked.
    assert "Assumptions: Every answer holds for the stated features" in _flat_text(closing)
    assert f"{CAUTION} Every answer holds" not in closing
    limits = report.limits(detail="full")
    # One line per limit pointing to its section, where its need is (owner, 2026-10-06).
    assert "4. How many fields should I image next time?: not assessed, see question 4." in limits
    reasons = report.limits(detail="full", pointers=False)
    assert "4. How many fields should I image next time?: not assessed; needs: run" in reasons
    assert report.limits() == limits


def test_the_closing_marks_a_weak_representation_as_a_problem():
    report = microscopist_report()
    weak = type(report)(
        report.sections,
        report.profile,
        ("Weak representation: fields were recognized in only 50% of checks.",),
        report.inputs,
    )
    assert f"{PROBLEM} Weak representation" in weak.limits(detail="short")


def test_the_workflow_document_prints_the_short_declaration_and_the_full_one_on_demand():
    from _helpers_workflows import _acquisition as _workflow_acquisition

    from morphocoverage.workflows import report

    objects, fovs = _workflow_acquisition()
    result = report(objects, fovs, _setup())
    short = result.summary()
    declared = short.split(DECLARATION_TITLE)[1]
    assert declared.startswith("\n  Population summary:")
    full = result.summary(detail="full")
    assert f"{DECLARATION_TITLE}\n  - Population summary" in full
    assert result.document.short["declaration"] in short


def test_question_2_without_an_order_check_states_a_caution_and_looks_at_it():
    report = _named_scenario_report(
        "homogeneous", [ScalarPrecisionStrategy("feature_a", 0.5, "fov", "upper_95")]
    )
    document = microscopist_report(sufficiency=report)
    section = _flat(document.short["enough"])
    assert f"{CAUTION} The acquisition order was not checked, so every stop treats" in section
    assert document.statuses["enough"] is Status.CAUTION


def test_the_closing_says_every_question_could_be_answered_in_plain_words():
    assert ASSESSED == (
        "Every question could be answered; the assumptions each answer rests on are listed with it."
    )


def test_question_3_is_a_caution_when_a_state_clusters(states):
    clustering, unusual = states
    report = microscopist_report(clustering=clustering, unusual=unusual)
    assert report.statuses["profile"] is Status.CAUTION
    assert "state cells clustered in 1 of 2 groups" in report.headlines["profile"]
    spread = dataclasses.replace(
        clustering, table=clustering.table.assign(p_clustered=1.0, q_clustered=1.0)
    )
    status, _ = profile_status(report.profile, [spread])
    assert status is Status.OK


def test_the_report_sections_are_short_with_each_caveat_once(foundation_checks, controls, states):
    clustering, unusual = states
    mixed = {k: stamped(v) for k, v in {**foundation_checks, "controls": controls}.items()}
    report = microscopist_report(**mixed, clustering=stamped(clustering), unusual=stamped(unusual))
    for key in ("unit", "controls", "profile"):
        short = report.short[key]
        assert short != report.sections[key]
        cautions = [line for line in short.splitlines() if line.lstrip().startswith(CAUTION)]
        assert len(cautions) == len(set(cautions))
        assert all(width(line) <= 100 for line in short.splitlines()), key
        assert all(width(line) <= 100 for line in report.sections[key].splitlines()), key
        assert "condition=" not in short
    text = report.summary()
    assert f"{report.statuses['unit'].marker} Foundation: why count fields" in text
    assert report.sections["profile"] in report.summary(detail="full")
    # The full profile keeps every statement with its source.
    for row in report.profile.itertuples():
        assert _flat_text(row.statement) in _flat_text(report.sections["profile"])


def test_the_report_gives_questions_4_5_and_7_short_with_their_markers():
    reliability = _scenario_reliability()
    protocol = _plate_protocol(_well_values(("P1", "P2"), 0.95, 0.99))
    report = microscopist_report(
        reliability=stamped(reliability),
        protocol=stamped(protocol),
        adequacy=stamped(_adequacy()),
        effort=stamped(_effort()),
    )
    for key in ("next_time", "bought", "effort"):
        assert key in report.short and report.short[key] != report.sections[key]
        assert all(line.startswith("  ") or not line for line in report.short[key].splitlines())
    short = report.summary()
    assert f"{Status.OK.marker} 4. How many fields next time?" in short
    assert f"{Status.OK.marker} 7. More fields or more samples?" in short
    assert report.statuses["bought"] in (Status.OK, Status.CAUTION)
    full = report.summary(detail="full")
    for text in (reliability.summary(detail="full"), protocol.summary(detail="full")):
        assert _flat(text.splitlines()[1]) in _flat(full)
    # Each caveat of question 4 once in the short form.
    section = _flat(report.short["next_time"])
    assert section.count("conservative by construction") == 1
    assert section.count("the field count is a guide") == 1
    assert OK not in report.short["next_time"] and PROBLEM not in report.short["next_time"]
