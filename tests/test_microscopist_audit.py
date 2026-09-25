"""Regression tests from the microscopist audit (docs/audit/microscopist_audit_2026-09-25.md).

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
    assert "Did we sample enough?" in text and "one more FOV" not in text


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
    assert "The held-back fields could not judge this for 2 of 3 samples" in text
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
        if row["median_fovs"] > row["stream_fovs"] or row["reached"] == 1:
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
