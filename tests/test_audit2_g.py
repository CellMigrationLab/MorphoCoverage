"""Regression tests of the second audit, group G: effort, inspection, controls, report, frontier."""

import numpy as np
import pandas as pd
import pytest
from test_retrospective_controls import FEATURES, NOVELTY, _plate
from test_retrospective_inspection import FEATURES as INSPECTED
from test_retrospective_inspection import _acquisition

from morphocoverage._wording import percent
from morphocoverage.bounds import lower_bound
from morphocoverage.retrospective import (
    RepresentationControl,
    SamplingHierarchy,
    Summary,
    effort_projection,
    inspect_fovs,
    microscopist_report,
    representation_control,
)
from morphocoverage.retrospective.effort import _acquired_se, _contributions, _se
from morphocoverage.retrospective.frontier import (
    BASELINE,
    FrontierResult,
    _met_not_sustained,
)
from morphocoverage.retrospective.report import _detection_adjusted_bounds
from morphocoverage.strategies import SupportNoveltyEstimator

WELLS = SamplingHierarchy(fov="image_file", sample="well")
V_SAMPLE, V_FIELD, V_OBJECT = 1.0, 0.2, 1.0


def _full(result):
    """The full summary with its line breaks as spaces (the text is wrapped)."""
    return " ".join(result.summary(detail="full").split())


def _wells(fields, cells, seed=0, repeats=None):
    """Tables of wells with ``fields[s]`` fields of ``cells[s]`` objects (0: empty fields)."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for s, (n_fields, m) in enumerate(zip(fields, cells, strict=True)):
        sample = rng.normal(0, np.sqrt(V_SAMPLE))
        for f in range(n_fields):
            name = f"s{s}_f{f}"
            row = {"well": f"W{s}", "image_file": name, "condition": "c"}
            if repeats is not None:
                row["repeat"] = repeats[s]
            fov_rows.append(row)
            values = sample + rng.normal(0, np.sqrt(V_FIELD)) + rng.normal(0, np.sqrt(V_OBJECT), m)
            object_rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fovs = pd.DataFrame(fov_rows)
    objects = pd.concat(object_rows, ignore_index=True).merge(fovs, on="image_file")
    return objects, fovs


def _design_of(row):
    """The design dictionary of a stratum, from its component row."""
    return {
        "has_repeat": False,
        "repeats": 1,
        "samples": int(row["samples"]),
        "total_fovs": int(row["fovs"]),
        "yield": row["fovs_with_observations"] / row["fovs"],
        "per_fov": row["observations_per_fov"],
        "cluster_size": row["cluster_size"],
        "sample_weight": row["sample_weight"],
    }


def test_g1_the_acquired_error_uses_the_sizes_of_unequal_samples():
    # One well of 20 fields and five of 2, 10 objects each (the audit's g02 design).
    fields, cells = [20, 2, 2, 2, 2, 2], [10] * 6
    objects, fovs = _wells(fields, cells)
    result = effort_projection(objects, fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=20)
    design = _design_of(result.components.iloc[0])
    truth = np.array([V_SAMPLE, V_FIELD, V_OBJECT])
    per_well = np.array([n * m for n, m in zip(fields, cells, strict=True)], dtype=float)
    total = per_well.sum()
    parts = np.array([V_SAMPLE * np.sum(per_well**2), V_FIELD * 10 * total, V_OBJECT * total])
    parts /= total**2
    # Analytic 0.690 (a simulation of this design gives 0.688); the old formula gave 0.420.
    assert _acquired_se(truth, design) == pytest.approx(np.sqrt(parts.sum()))
    assert _acquired_se(truth, design) == pytest.approx(0.690, abs=0.001)
    assert _contributions(truth, design) == pytest.approx(parts / parts.sum())
    # Alternatives are compared with the acquired allocation made balanced, never with
    # the imbalance, which is stated apart.
    current = result.projections[result.projections["current"]].set_index("design")["se"]
    acquired, balanced = current["acquired"], current["balanced"]
    assert balanced == pytest.approx(_se(result.components["estimate"].to_numpy(), design, 6, 5))
    assert acquired > balanced
    comparisons = result.comparisons
    assert (comparisons["balanced_se"] == balanced).all()
    assert (comparisons["acquired_se"] == acquired).all()
    assert "The acquired samples differ in size" in _full(result)
    assert "made balanced" in _full(result) or "cannot tell" in _full(result)
    assert "The acquired samples differ in size" in " ".join(result.summary().split())
    even, even_fovs = _wells([5] * 6, [10] * 6)
    same = effort_projection(even, even_fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=20)
    rows = same.projections[same.projections["current"]]
    assert rows["se"].iloc[0] == pytest.approx(rows["se"].iloc[1])
    assert "differ in size" not in _full(same) and "differ in size" not in same.summary()


def test_g2_samples_and_units_are_counted_among_those_with_observations():
    objects, fovs = _wells([6] * 8, [30, 30, 30, 0, 0, 0, 0, 0])
    result = effort_projection(objects, fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=50)
    row = result.components.iloc[0]
    assert (row["units"], row["samples"], row["samples_acquired"], row["fovs"]) == (3, 3, 8, 48)
    assert not result.comparisons["resolved"].any()
    text = _full(result)
    assert "cannot tell from these data (only 3 samples)" in text
    assert "5 of 8 samples held no objects with a value" in text
    short = " ".join(result.summary().split())
    assert "only 3 samples" in short and "5 of 8 samples held no objects with a value" in short


def test_g3_the_repeat_is_left_out_for_the_reason_that_holds():
    # Repeat R1 was imaged but holds no object: one repeat has observations.
    objects, fovs = _wells([4] * 6, [20, 20, 20, 0, 0, 0], repeats=["R0"] * 3 + ["R1"] * 3)
    hierarchy = SamplingHierarchy("image_file", "well", "repeat")
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20
    )
    row = result.components.iloc[0]
    assert row["repeat_dropped_reason"] == "single repeat"
    assert (row["repeats"], row["repeats_acquired"]) == (1, 2)
    text = result.summary()
    assert "Only one of the 2 repeats here holds objects" in text
    assert "Every repeat holds a single sample" not in text


def test_g4_failed_resamples_are_counted_and_withhold_the_interval():
    # Five wells of one field and one of three: many resamples cannot separate the levels.
    objects, fovs = _wells([1, 1, 1, 1, 1, 3], [20] * 6)
    result = effort_projection(objects, fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=500)
    rows = result.components
    failed = int(rows["boot_failed"].iloc[0])
    assert failed > 50 and (rows["n_boot"] == 500).all()
    assert rows["low"].isna().all() and rows["high"].isna().all()
    assert result.comparisons["ratio_low"].isna().all()
    assert not result.comparisons["resolved"].any()
    text = _full(result)
    assert f"not established: no interval, as {failed} of 500 resamples" in text
    assert "cannot tell from these data (no interval" in text
    short = " ".join(result.summary().split())
    assert f"no interval, as {failed} of 500 resamples" in short
    # A design whose resamples all separate the levels keeps its interval.
    fine, fine_fovs = _wells([4] * 6, [20] * 6)
    kept = effort_projection(fine, fine_fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=50)
    assert kept.components["low"].notna().all()


def test_g5_allocations_keep_a_field_per_sample_and_declared_ones_are_checked():
    objects, fovs = _wells([1, 1, 1, 1, 1, 3], [20] * 6)
    result = effort_projection(objects, fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=20)
    assert (result.projections["fovs_per_sample"] >= 1).all()
    assert 12 not in set(result.projections["samples"])  # 8 fields over 12 samples
    for allocation in [(0, 24), (-3, -8), (2, 0), (2, -1), ("a", 4), (3,), (2.5, 4)]:
        with pytest.raises(ValueError, match="allocation"):
            effort_projection(
                objects,
                fovs,
                feature="x",
                hierarchy=WELLS,
                strata=[],
                allocations=[allocation],
                n_boot=20,
            )


def test_g6_an_empty_or_unlabelled_stratum_is_skipped_with_its_reason():
    objects, fovs = _wells([3] * 8, [15] * 4 + [0] * 4)
    fovs["condition"] = np.where(fovs["well"].isin(["W0", "W1", "W2", "W3"]), "ctrl", "drug")
    objects = objects.drop(columns="condition").merge(fovs[["image_file", "condition"]])
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=WELLS, strata=["condition"], n_boot=20
    )
    skipped = result.skipped.set_index("condition")["reason"]
    assert skipped["drug"] == "no field holds an observation with a value of x"
    assert set(result.components["condition"]) == {"ctrl"}
    fovs["batch"] = "b1"
    fovs.loc[fovs["well"] == "W0", "batch"] = np.nan
    batched = effort_projection(
        objects, fovs, feature="x", hierarchy=WELLS, strata=["batch"], n_boot=20
    )
    reason = batched.skipped["reason"].iloc[0]
    assert reason.startswith("batch is missing for 3 fields")
    assert list(batched.components["stratum"].unique()) == ["batch=b1"]


def test_g7_most_of_the_uncertainty_only_above_half():
    rng = np.random.default_rng(2)
    fov_rows, object_rows = [], []
    for s in range(8):
        well = rng.normal(0, 0.35)
        for f in range(5):
            name = f"s{s}_f{f}"
            fov_rows.append({"well": f"W{s}", "image_file": name, "condition": "c"})
            values = well + rng.normal(0, 0.6) + rng.normal(0, 3, 20)
            object_rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fovs = pd.DataFrame(fov_rows)
    objects = pd.concat(object_rows, ignore_index=True).merge(fovs, on="image_file")
    result = effort_projection(objects, fovs, feature="x", hierarchy=WELLS, strata=[], n_boot=50)
    assert result.components["share"].max() < 0.5
    text = _full(result)
    assert "The largest part of the uncertainty of the mean is" in text
    assert "Most of the uncertainty" not in text


def _spots():
    """Two conditions of 12 fields: 3 fields of A without spots; B's w is negative."""
    rng = np.random.default_rng(0)
    fov_rows, object_rows = [], []
    for condition in ("A", "B"):
        for i in range(12):
            name = f"{condition}{i:02d}"
            fov_rows.append({"condition": condition, "image_file": name})
            spots = np.zeros(20) if i < 3 else rng.poisson(5, 20).astype(float) + 1
            z = rng.normal(5, 1, 20) if condition == "A" else rng.normal(-3, 1, 20)
            object_rows.append(
                pd.DataFrame({"image_file": name, "condition": condition, "spots": spots, "w": z})
            )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def test_g8_a_summary_that_is_zero_is_reported_not_dropped():
    objects, fovs = _spots()
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["spots"],
        count_ratio=None,
        summaries=[Summary("spots", 0.1)],
        group_cols=["condition"],
    )
    skipped = pd.DataFrame(result.settings["skipped_summaries"]).set_index("group")
    assert skipped.loc["condition=A", "reason"].startswith("is 0 here")
    # The group by its value (the section names its column), once per group in full.
    assert "was not checked in A: it is 0 here" in result.summary(detail="full")
    assert "was not checked in 2 groups (A and B): it is 0 here" in " ".join(
        result.summary().split()
    )


def test_g9_a_summary_skipped_in_one_group_names_that_group():
    objects, fovs = _spots()
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["w"],
        count_ratio=None,
        summaries=[Summary("w", "mean")],
        group_cols=["condition"],
    )
    skipped = result.settings["skipped_summaries"]
    assert [(s["group"], s["summary"]) for s in skipped] == [("condition=B", "mean of w")]
    assert result.settings["checked"] == ["influence"]  # checked in A
    text = result.summary()
    assert "The influence on the mean of w was not checked in B" in text
    assert "(influence checked)" in text
    assert "Analysis groups: condition." in result.summary(detail="full")


def test_g10_nothing_checked_is_not_called_nothing_stands_out():
    objects, fovs = _spots()
    result = inspect_fovs(
        objects, fovs, feature_cols=["spots"], count_ratio=None, group_cols=["condition"]
    )
    assert result.settings["checked"] == []
    assert result.summary().startswith("No field was checked")
    assert "No field stands out" not in result.summary()
    # Only a reason that was skipped everywhere is not a check either.
    negative = inspect_fovs(
        objects[objects["condition"] == "B"],
        fovs[fovs["condition"] == "B"],
        feature_cols=["w"],
        summaries=[Summary("w", "mean")],
        count_ratio=None,
    )
    assert negative.summary().startswith("No field was checked")


def test_g11_the_influence_threshold_is_a_positive_number():
    objects, fovs = _acquisition()
    for threshold in (None, 0, -0.1, "0.1", np.inf, True):
        with pytest.raises(ValueError, match="influence_threshold"):
            inspect_fovs(objects, fovs, feature_cols=INSPECTED, influence_threshold=threshold)
    with pytest.raises(ValueError, match="neighbour_threshold"):
        inspect_fovs(objects, fovs, feature_cols=INSPECTED, neighbour_threshold=None)
    inspect_fovs(objects, fovs, feature_cols=INSPECTED, influence_threshold=np.float32(0.2))


def test_g12_a_planting_that_cannot_happen_is_said():
    objects, fovs = _plate()  # 25 objects per field
    control = RepresentationControl(
        positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_objects=30
    )
    result = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=NOVELTY,
        control=control,
        pair_cols=["repeat"],
        fovs_seen=6,
        draws=5,
    )
    assert "planted" not in set(result.pooled["candidate"])
    gaps = result.settings["planted_skipped"]
    assert [g["stratum"] for g in gaps] == ["repeat=P1", "repeat=P2"]
    assert (
        "no positive-control field holds 30 objects to plant (the largest holds 25)"
        in (gaps[0]["reason"])
    )
    assert "Planted objects: not measured in repeat P1" in result.summary()
    assert "Planted objects: not measured in repeat P2" in result.summary()


def test_g13_the_chance_expectation_uses_the_leave_one_field_out_rate():
    novelty = SupportNoveltyEstimator(0.6, 1, None, 3, 2)
    objects, fovs = _plate(seed=3, plates=("P1",), fovs_per_well=30)
    control = RepresentationControl(positive={"condition": "KD"}, reference={"condition": "MOCK"})
    controls = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        control=control,
        pair_cols=["repeat"],
        fovs_seen=3,
        draws=20,
    )
    rates = controls.pooled.set_index("candidate")["rate"]
    inspection = inspect_fovs(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        count_ratio=None,
        group_cols=["condition", "repeat"],
    )
    new = inspection.fovs[inspection.fovs["reason"] == "new state"]
    flagged_reference = int((new["condition"] == "MOCK").sum())
    # The same test as the inspection: its rate is the share of reference fields it flags.
    assert rates["reference_lofo"] == pytest.approx(flagged_reference / 30)
    assert rates["reference"] > 0.3  # after 3 fields, the rate the report used to multiply
    statement = next(
        s
        for s in microscopist_report(controls=controls, inspection=inspection).profile["statement"]
        if s.startswith("new states")
    )
    expected = rates["reference_lofo"] * 60
    # Audit 3 (G-3): from the rate to its upper bound, over the fields judged.
    assert f"{expected:.0f} to " in statement and "of the 60 fields judged" in statement
    assert f"was {flagged_reference} of 30 fields" in statement


def _sufficiency():
    """The part of a sufficiency report the adjusted bound reads."""
    from types import SimpleNamespace

    # The analysis groups of the controls' tables (audit 3, G-20: other groups are refused).
    next_fov = pd.DataFrame(
        [{"condition": "MOCK", "repeat": "P1", "unseen_state_max_frequency": 0.12}]
    )
    return SimpleNamespace(group_cols=("condition", "repeat"), next_fov=next_fov)


def test_g13_the_adjusted_bound_uses_the_rate_against_every_reference_field():
    objects, fovs = _plate(shift=1.0)
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
    control = RepresentationControl(positive={"condition": "KD"}, reference={"condition": "MOCK"})
    controls = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        control=control,
        pair_cols=["repeat"],
        fovs_seen=6,
        draws=10,
    )
    row = controls.pooled.set_index("candidate").loc["positive_lofo"]
    detection = float(row["rate"])
    text = _detection_adjusted_bounds(_sufficiency(), controls)
    # Audit 3 (G-1, J-2): against the reference fields of an analysis group, via percent().
    assert f"({percent(detection)} of its fields recognized against the reference" in text
    # Audit 4 (P3-9): divided by the rate's one-sided 95% lower bound, not its point estimate.
    low = lower_bound(int(row["judged_new"]), int(row["candidates"]), 0.95)
    assert f"up to {percent(min(1.0, 0.12 / low))} of FOVs" in text


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
    report = microscopist_report(redundancy=redundancy)
    assert report.profile.empty
    assert report.inputs["profile"] == "not assessed"
    assert "3. What is limiting my sampling?: not assessed" in report.limits()
    assert any("sampling-frame bias is not assessed" in note for note in report.notes)


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


def test_g18_the_split_modules_keep_the_names_callers_use():
    from morphocoverage.retrospective import report, report_profile

    assert report._detection_adjusted_bounds is report_profile._detection_adjusted_bounds
    assert "sampling_profile" in report_profile.__all__


def test_g19_the_adjusted_bound_is_given_with_a_positive_control_only():
    objects, fovs = _plate(shift=1.0)
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
    controls = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        control=RepresentationControl(positive={"condition": "KD"}),
        pair_cols=["repeat"],
        fovs_seen=6,
        draws=10,
    )
    assert controls.status == "positive only"
    text = _detection_adjusted_bounds(_sufficiency(), controls)
    assert text is not None and text.startswith("Adjusted for the declared control's detection")
