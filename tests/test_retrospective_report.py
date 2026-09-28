"""Tests for the assembled microscopist's report (docs/questions.md)."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    RepresentationControl,
    SamplingHierarchy,
    design_effect,
    effort_projection,
    microscopist_report,
    representation_control,
    sampling_reliability,
)
from morphocoverage.retrospective.report import DECLARATION_TITLE, QUESTIONS
from morphocoverage.strategies import SupportNoveltyEstimator


def test_an_empty_report_says_what_each_question_needs():
    report = microscopist_report()
    text = report.summary()
    for _, title in QUESTIONS:
        assert title in text
    assert text.count("Not assessed:") == len(QUESTIONS)
    assert "No representation control" in text
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
    text = report.summary()
    assert text.startswith(f"{OVERVIEW_TITLE}\n")
    assert "feature column" in text and "condition" in text.split(DECLARATION_TITLE)[0]
    assert text.rstrip().split("\n\n")[-1].startswith(LIMITS_TITLE)
    limits = report.limits()
    assert "1. Can these features detect what I care about?: not assessed; needs:" in limits
    assert "No representation control" in limits  # the notes close the report
    assert "Read with care" not in text
    # An overview with many features lists the first twelve and counts the rest.
    many = built.objects.assign(**{f"f{i}": 0.0 for i in range(15)})
    columns = [*FEATURE_COLS, *(f"f{i}" for i in range(15))]
    wide = dataset_overview(many, built.fovs, feature_cols=columns)
    assert "and 5 more" in wide.summary()
    single = SCENARIOS["homogeneous"]()
    bare = dataset_overview(
        single.objects.drop(columns=["condition", "repeat"]),
        single.fovs.drop(columns=["condition", "repeat"]),
        feature_cols=list(FEATURE_COLS),
    )
    assert "No condition or repeat column" in bare.summary()
    # A selection of questions limits the closing section to those questions.
    assert "4. How many fields" not in report.summary(["enough"])


def test_the_declaration_is_printed_first_only_when_given():
    assert DECLARATION_TITLE not in microscopist_report().summary()
    report = microscopist_report(declaration=["Population summary: the mean of x within 5%."])
    text = report.summary()
    assert text.startswith(f"{DECLARATION_TITLE}\n  - Population summary")
    assert text.index(DECLARATION_TITLE) < text.index("Foundation: why count")
    assert report.summary(["enough"]).startswith(DECLARATION_TITLE)
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
    text = report.summary()
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
    report = microscopist_report(effort=effort, redundancy=redundancy)
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
            fov_rows.append({"image_file": name, "condition": condition, "repeat": "P1"})
            values = rng.normal(0, 1, size=(20, 2))
            values[:, 0] += offset
            object_rows.append(
                pd.DataFrame(values, columns=list(FEATURE_COLS)).assign(
                    image_file=name, condition=condition, repeat="P1"
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
        pair_cols=["repeat"],
        fovs_seen=5,
        draws=5,
    )
    report = microscopist_report(controls=control)
    assert "declared positive control were recognized" in report.sections["controls"]
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
    profile = microscopist_report(redundancy=redundancy).sections["profile"]
    assert "in 1 of 2 groups, fields are more alike than random pairs at every distance" in profile
    assert "no group shows" not in profile


def test_the_overview_counts_technical_repeats_within_their_condition_and_repeat():
    from morphocoverage.retrospective import dataset_overview

    built = SCENARIOS["condition_repeat_groups"]()
    fovs = built.fovs.copy()
    # Two wells per condition and repeat, named alike everywhere: still distinct units.
    fovs["technical_repeat"] = ["w1" if i % 2 else "w2" for i in range(len(fovs))]
    objects = built.objects.merge(
        fovs[["condition", "repeat", "image_file", "technical_repeat"]],
        on=["condition", "repeat", "image_file"],
    )
    overview = dataset_overview(objects, fovs, feature_cols=list(FEATURE_COLS))
    groups = fovs.groupby(["condition", "repeat"]).size()
    assert overview.counts["technical_repeats"] == 2 * len(groups)
    assert (overview.layout["technical_repeats"] == 2).all()
    text = overview.summary()
    assert f"{2 * len(groups)} technical repeats (technical_repeat, a well or a dish" in text
    assert "2 per condition and repeat" in text and "technical repeats" in text.split("\n")[4]
    plain = dataset_overview(built.objects, built.fovs, feature_cols=list(FEATURE_COLS))
    assert plain.counts["technical_repeats"] == 0 and "technical" not in plain.summary()
