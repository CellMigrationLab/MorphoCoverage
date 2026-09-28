"""Regression tests from the microscopist audit of 2026-09-25 (docs/audit/README.md).

Each test failed before its fix. They protect plain-language answers, not
scientific behaviour: a summary must not crash, must not present a curve
capped by its reference as a field requirement, must not present a variance
share from three samples as established, and must name what it counts.
"""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.bounds import lower_bound
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    ReliabilityResult,
    SamplingHierarchy,
    adequacy_points,
    effort_projection,
    microscopist_report,
    order_efficiency,
    protocol_reliability,
    replay_acquisitions,
    sampling_reliability,
    stopping_calibration,
)
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.transforms import Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
MEAN_A = OutcomeCriterion(
    RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean of feature_a within 2%"
)


def test_sufficiency_summary_works_without_a_novelty_definition():
    # With group columns and no novelty, the next-FOV table is empty and has no
    # group columns; summary() raised KeyError('condition').
    built = SCENARIOS["homogeneous"]()
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(5)],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    text = report.summary()
    assert "condition=control, repeat=R1:" in text
    assert "Did each criterion say enough?" in text and "one more FOV" not in text


def test_the_unseen_state_bound_names_the_fovs_without_observations():
    # 3 fields with cells and 10 empty ones: the "about 3 / 13" count is right
    # for representative draws, but the reader must see that 10 could not
    # have shown anything.
    built = SCENARIOS["homogeneous"]()
    fovs = built.fovs.iloc[:13].copy()
    objects = built.objects[built.objects["image_file"].isin(fovs["image_file"].iloc[:3])]
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3),
    )
    assert report.next_fov["fovs_without_observations"].iloc[0] == 10
    text = report.summary()
    assert "about 3 / 13" in text
    assert "The count includes 10 FOVs without observations, which cannot show a state." in text


def _reliability(name="homogeneous", counts=(2, 5)):
    built = SCENARIOS[name]()
    return sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[MEAN_A],
        fov_counts=list(counts),
        reservoir_fraction=0.3,
        n_splits=3,
        draws=5,
    )


def test_a_curve_rebuilt_without_its_ceiling_says_so():
    result = _reliability()
    assert "the curve cannot be read above this" in result.summary()
    rebuilt = ReliabilityResult(result.curve, pd.DataFrame(), result.group_cols, result.settings)
    text = rebuilt.summary()
    assert "the curve cannot be read above this" not in text
    assert "was not computed" in text and "capped by them rather than by the fields" in text


def _curve_with_ceiling(preserved, ceiling, stream=18):
    rows, ceilings = [], []
    for well, by_k in preserved.items():
        for k, value in by_k.items():
            rows.append(
                {
                    "condition": "ctrl",
                    "plate": "P1",
                    "well": well,
                    "criterion": "var",
                    "fovs": k,
                    "preserved": value,
                    "stream_fovs": stream,
                }
            )
        ceilings.append(
            {
                "condition": "ctrl",
                "plate": "P1",
                "well": well,
                "criterion": "var",
                "splits": 10,
                "defined_splits": 10,
                "whole_stream_preserved": ceiling[well],
                "stream_fovs": stream,
            }
        )
    return ReliabilityResult(
        pd.DataFrame(rows),
        pd.DataFrame(),
        ("condition", "plate", "well"),
        {"labels": {"var": "the variance within 10%"}},
        pd.DataFrame(ceilings),
    )


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
    text = result.summary()
    assert "Needs more fields than could be judged: B1, P1 (more than 18)." in text
    assert "For 2 of 3 samples even every available field together was enough in fewer" in text
    assert "A1, P1 (all fields enough in 30% of hold-outs); A2, P1" in text
    assert "A1, P1 (more than 18)" not in text
    assert "5 fields:" in text and "1 fields" not in text


def test_protocol_without_a_ceiling_table_keeps_the_old_wording():
    per_sample = _curve_with_ceiling({"A1": {5: 0.3, 18: 0.3}}, {"A1": 0.3})
    bare = ReliabilityResult(
        per_sample.curve, pd.DataFrame(), per_sample.group_cols, per_sample.settings
    )
    result = protocol_reliability(
        bare, SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    )
    assert result.difficulty["ceiling_below_threshold"].isna().all()
    assert "Needs more fields than could be judged: A1, P1 (more than 18)." in result.summary()


def _nested(rng, n_samples, n_fovs, cells, v_sample, v_fov=0.25):
    objects, fovs = [], []
    for s in range(n_samples):
        sample = rng.normal(0, np.sqrt(v_sample))
        for f in range(n_fovs):
            name = f"s{s}_f{f}"
            fovs.append({"position_id": name, "well": f"w{s}", "condition": "A"})
            values = 10 + sample + rng.normal(0, np.sqrt(v_fov)) + rng.normal(size=cells)
            objects.append(pd.DataFrame({"position_id": name, "well": f"w{s}", "x": values}))
    return pd.concat(objects, ignore_index=True).assign(condition="A"), pd.DataFrame(fovs)


def test_a_variance_share_from_few_samples_is_marked_as_not_established():
    # Seed 3 of the audit's null simulation: no between-sample variance, yet
    # 70% of the variance of the mean was attributed to the 3 samples. The
    # number may stand, the sentence may not present it as a finding.
    objects, fovs = _nested(np.random.default_rng(3), 3, 10, 30, v_sample=0.0)
    result = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("position_id", "well"),
        strata=["condition"],
        n_boot=100,
        random_state=3,
    )
    text = result.summary()
    assert "Most of the uncertainty of the mean is between samples" in text
    assert "; not established: only 3 samples, its interval includes zero)." in text
    profile = microscopist_report(effort=result).profile
    statement = profile[profile["item"] == "largest source of uncertainty of the mean"]
    assert "not established: only 3 samples" in statement["statement"].iloc[0]


def test_a_variance_share_from_many_samples_with_a_clear_component_is_stated_plainly():
    objects, fovs = _nested(np.random.default_rng(0), 8, 8, 20, v_sample=4.0, v_fov=0.05)
    result = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("position_id", "well"),
        strata=["condition"],
        n_boot=100,
    )
    text = result.summary()
    assert "between samples" in text and "not established" not in text


@pytest.fixture(scope="module")
def gradient_replay():
    built = SCENARIOS["neighbour_correlation"]()
    orders = [AcquisitionOrder.recorded(), AcquisitionOrder.permuted(0)] + [
        AcquisitionOrder.dispersed(s) for s in range(2)
    ]
    return replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(4)],
        criteria=[
            OutcomeCriterion(
                RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean within 5%"
            )
        ],
        objective=POPULATION,
        reservoir_fraction=0.35,
        orders=orders,
        acquisition_order_col=ORDER_COL,
        stage_cols=("stage_x", "stage_y"),
        n_splits=3,
    )


def test_dispersed_orders_are_named_in_every_summary(gradient_replay):
    calibration = stopping_calibration(gradient_replay).summary()
    adequacy = adequacy_points(gradient_replay).summary()
    assert "  spreading the same positions out: it stopped" in calibration
    assert "  Spreading the same positions out: judged on held-back fields" in adequacy
    assert "\n  dispersed:" not in calibration and "\n  dispersed:" not in adequacy


def test_order_efficiency_states_the_runs_that_never_reached_adequacy(gradient_replay):
    result = order_efficiency(gradient_replay)
    table = result.table.set_index("order")
    text = result.summary()
    for order, row in table.iterrows():
        if pd.isna(row["median_fovs"]) or row["reached"] == 1:
            continue
        share = f"{100 * (1 - row['reached']):.0f}% of runs"
        assert f"never within the {row['stream_fovs']} fields in {share}" in text, order
    if (table["reached"] == 1).all():
        pytest.skip("every replayed order reached adequacy; nothing to state")


def test_the_object_unit_jcb_verdict_states_its_independence_assumption():
    # On 3 homogeneous FOVs the published rule is met after 1 FOV; the sentence
    # must say what makes that possible, and the FOV-unit variant must not.
    from morphocoverage import ScalarPrecisionStrategy

    built = SCENARIOS["homogeneous"]()
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[
            ScalarPrecisionStrategy("feature_a", 0.05, "object", "upper_95"),
            ScalarPrecisionStrategy("feature_a", 0.05, "fov", "upper_95"),
        ],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    answers = report.enough.set_index("criterion")["answer"]
    clause = "assumes a spatially uniform sample"
    assert clause in answers["scalar_precision_object_upper95"]
    assert clause not in answers["scalar_precision_fov_upper95"]


def test_fields_with_observations_are_listed_when_most_fields_are_empty():
    # 3 fields with cells among 20: the median count is 0, and nothing was listed,
    # neither the empty fields nor the three that hold every observation.
    from morphocoverage.retrospective import inspect_fovs

    built = SCENARIOS["homogeneous"]()
    kept = built.fovs["image_file"].iloc[:3]
    objects = built.objects[built.objects["image_file"].isin(kept)]
    result = inspect_fovs(objects, built.fovs, feature_cols=list(FEATURE_COLS))
    listed = result.fovs[result.fovs["reason"] == "count"]
    assert sorted(listed["image_file"]) == sorted(kept)
    assert "most fields have none (the median is 0)" in result.summary()


def test_a_fixed_rule_reports_a_reached_target_and_a_threshold_rule_a_met_threshold():
    from morphocoverage import ScalarPrecisionStrategy

    built = SCENARIOS["homogeneous"]()
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[
            FixedFOVStrategy(5),
            ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
        ],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    answers = report.enough.set_index("criterion")["answer"]
    assert "its target was reached by the 20 acquired FOVs" in answers["fixed_fov"]
    assert "yes, its threshold was met by the 20 acquired FOVs" in answers["scalar_precision_fov"]
    assert "yes, met by" not in report.summary()


def test_a_state_seen_in_two_neighbouring_fovs_is_reported_as_seen_in_few():
    # rare_clustered: 10 rare cells in each of two adjacent FOVs of 40. With one
    # supporting FOV they cover each other, so the next-FOV chance is 0 and the
    # state went unmentioned; few_fovs=3 names the two fields.
    built = SCENARIOS["rare_clustered"]()
    novelty = SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3)
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
        few_fovs=3,
    )
    row = report.next_fov.iloc[0]
    assert row["fovs_with_new"] == 0 and row["fovs_with_state_in_few"] == 2
    text = report.summary()
    assert (
        "Seen in few FOVs: 2 of 40 FOVs hold a state that other FOVs show too, but fewer "
        "than 3 of them" in text
    )
    with pytest.raises(ValueError, match="few_fovs must exceed"):
        sufficiency_report(
            built.objects,
            built.fovs,
            feature_cols=list(FEATURE_COLS),
            strategies=[],
            objective=POPULATION,
            orders=[AcquisitionOrder.permuted(0)],
            novelty=novelty,
            few_fovs=1,
        )


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
        results = session.add_fov(rows.drop(columns=["condition", "repeat"]), fov.to_dict())
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


def _weak_control_report():
    from scenarios import _fov_table, _objects_for

    from morphocoverage.retrospective import RepresentationControl, representation_control

    rng = np.random.default_rng(1)
    ref_f = _fov_table(20, condition="MOCK")
    pos_f = _fov_table(20, condition="KD")
    ref_o, pos_o = (
        _objects_for(ref_f, np.full(20, 30), rng),
        _objects_for(pos_f, np.full(20, 30), rng),
    )
    pos_o[list(FEATURE_COLS)] += 0.8
    objects, fovs = pd.concat([ref_o, pos_o], ignore_index=True), pd.concat([ref_f, pos_f])
    novelty = SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3)
    controls = representation_control(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        novelty=novelty,
        control=RepresentationControl(
            positive={"condition": ["KD"]}, reference={"condition": ["MOCK"]}
        ),
        pair_cols=[],
        fovs_seen=10,
        draws=5,
    )
    sufficiency = sufficiency_report(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(10)],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
    )
    return controls, sufficiency


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


def test_adequacy_and_order_summaries_state_the_tail(gradient_replay):
    adequacy = adequacy_points(gradient_replay)
    text = adequacy.summary()
    assert "(medians; by field" in text or "in more than 10% of runs" in text
    realized = adequacy.realized[adequacy.realized["criterion"] == "all criteria"]
    one_order = realized[realized["order"] == "recorded"]["sustained_realized"]
    tail = float(
        np.quantile(one_order.astype("Float64").to_numpy(dtype=float, na_value=np.inf), 0.9)
    )
    if np.isfinite(tail):
        assert f"by field {tail:.0f} in 90% of runs" in text


def test_orders_that_agree_within_one_field_are_called_a_null_result():
    built = SCENARIOS["homogeneous"]()
    replay = replay_acquisitions(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(2)],
        criteria=[
            OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")
        ],
        objective=POPULATION,
        reservoir_fraction=0.3,
        orders=[
            AcquisitionOrder.recorded(),
            AcquisitionOrder.permuted(0),
            AcquisitionOrder.dispersed(0),
        ],
        acquisition_order_col=ORDER_COL,
        stage_cols=("stage_x", "stage_y"),
        n_splits=2,
    )
    result = order_efficiency(replay)
    spread = result.table["median_fovs"].max() - result.table["median_fovs"].min()
    text = result.summary()
    assert ("null result" in text) == (spread <= 1 and (result.table["reached"] == 1).all())


def test_the_whole_stream_point_is_labelled_as_one_acquisition():
    result = _reliability(counts=(2, 14))
    stream = int(result.curve["stream_fovs"].max())
    text = result.summary()
    if stream == 14:
        assert "14 fields: enough in" in text
        assert "(all available fields: one acquisition per hold-out)" in text
    assert "2 fields: enough in" in text and "±" in text.split("2 fields")[1].split("\n")[0]


def test_partly_judged_samples_are_named_in_the_protocol():
    curve = pd.DataFrame(
        [
            {
                "condition": "c",
                "plate": "P1",
                "well": "A",
                "criterion": "rare",
                "fovs": 5,
                "preserved": 0.95,
                "stream_fovs": 18,
                "splits": 10,
                "defined_splits": 3,
            },
            {
                "condition": "c",
                "plate": "P1",
                "well": "B",
                "criterion": "rare",
                "fovs": 5,
                "preserved": 0.95,
                "stream_fovs": 18,
                "splits": 10,
                "defined_splits": 10,
            },
        ]
    )
    per_sample = ReliabilityResult(curve, pd.DataFrame(), ("condition", "plate", "well"), {})
    result = protocol_reliability(
        per_sample, SamplingHierarchy("image_file", "well", "plate"), threshold=0.9, strata=[]
    )
    assert result.table["partly_judged"].tolist() == [1]
    assert "1 judged on only some of the hold-outs" in result.summary()


def test_a_state_seen_in_few_fovs_gets_a_planning_line_from_its_incidence():
    from morphocoverage.strategies.rare_planning import fovs_to_detect

    built = SCENARIOS["rare_clustered"]()
    novelty = SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3)
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
        few_fovs=3,
    )
    row = report.next_fov.iloc[0]
    assert row["few_state_incidence"] == pytest.approx(2 / 40)
    assert 0 < row["few_state_incidence_low"] < 0.05 < row["few_state_incidence_high"] < 0.2
    assert row["fovs_to_see_again"] == fovs_to_detect(0.05, 0.95, 1) == 59
    assert row["fovs_to_see_again_low_end"] == fovs_to_detect(
        row["few_state_incidence_low"], 0.95, 1
    )
    text = report.summary()
    assert "Planning for such a state: these FOVs together are 5% of FOVs (95% interval" in text
    assert "image about 59 independent representative FOVs, up to" in text
    assert "Fields that cluster the state need more." in text


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
