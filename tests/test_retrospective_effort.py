"""Tests for the scalar hierarchical effort projection (P10 of docs/optimise/optimise.md)."""

from __future__ import annotations

import inspect as source

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import TRUTH, V_FIELD, V_OBJECT, V_SAMPLE, _plate, _simulate, _wells

from morphocoverage import statistics
from morphocoverage.retrospective import (
    SamplingHierarchy,
    design_effect,
    effort_design,
    effort_projection,
    effort_text,
    microscopist_report,
    nested_variance_components,
)
from morphocoverage.retrospective.effort import EffortResult, _contributions
from morphocoverage.retrospective.effort_design import _acquired_se, _se
from morphocoverage.retrospective.effort_design import _interval as effort_interval
from morphocoverage.retrospective.effort_text import effort_short_parts


def _codes(objects):
    keys = [
        objects["biological_repeat"],
        objects["biological_repeat"] + objects["well"],
        objects["image_file"],
    ]
    return [pd.factorize(k)[0] for k in keys]


def test_components_are_recovered_on_average_in_unbalanced_designs():
    rng = np.random.default_rng(3)
    estimates = []
    for _ in range(60):
        objects, _ = _simulate(rng, TRUTH, 6, 4, 6, 20, balanced=False)
        estimates.append(nested_variance_components(objects["x"].to_numpy(), _codes(objects)))
    mean = np.mean(estimates, axis=0)
    assert mean == pytest.approx(TRUTH, rel=0.2, abs=0.08)


def test_no_between_sample_variance_is_estimated_near_zero():
    rng = np.random.default_rng(4)
    estimates = [
        nested_variance_components(
            (o := _simulate(rng, (0.0, 0.0, 0.3, 4.0), 4, 5, 6, 20)[0])["x"].to_numpy(),
            _codes(o),
        )
        for _ in range(40)
    ]
    mean = np.mean(estimates, axis=0)
    assert abs(mean[0]) < 0.05 and abs(mean[1]) < 0.05
    assert mean[2] == pytest.approx(0.3, abs=0.06)


def test_projected_standard_errors_match_simulated_designs():
    rng = np.random.default_rng(5)
    components = np.array(TRUTH)
    for samples, fovs in ((4, 12), (8, 6)):
        means = [
            _simulate(rng, TRUTH, 2, samples // 2, fovs, 15)[0]["x"].mean() for _ in range(300)
        ]
        # 2 repeats of samples // 2 samples each, 15 cells per FOV.
        design = {"has_repeat": True, "repeats": 2, "yield": 1.0, "per_fov": 15, "cluster_size": 15}
        expected = _se(components, design, samples, fovs)
        assert np.std(means, ddof=1) == pytest.approx(expected, rel=0.15)


def test_few_independent_units_give_no_recommendation():
    rng = np.random.default_rng(6)
    objects, fovs = _simulate(rng, TRUTH, 3, 4, 6, 20)
    result = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat"),
        strata=["condition"],
        n_boot=100,
    )
    text = " ".join(result.summary(detail="full").split())
    assert "cannot tell from these data (only 3 repeats)" in text
    short = " ".join(result.summary().split())
    assert "Cannot tell whether more fields or more samples would help" in short
    assert "only 3 repeats" in short
    assert not result.comparisons["resolved"].any()
    levels = result.components.set_index("level")
    assert list(levels.index) == ["biological_repeat", "well", "image_file", "observation"]
    assert levels["share"].sum() == pytest.approx(1)
    assert (levels["low"] <= levels["high"]).all()


def test_a_dominant_between_sample_component_favours_more_samples():
    rng = np.random.default_rng(7)
    objects, fovs = _simulate(rng, (0.0, 2.0, 0.05, 1.0), 1, 12, 8, 20)
    result = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well"),
        strata=[],
        allocations=[(6, 16), (24, 4)],
        n_boot=200,
    )
    comparisons = result.comparisons.set_index("samples")
    assert comparisons.loc[24, "resolved"] and comparisons.loc[24, "ratio"] < 1
    assert comparisons.loc[6, "resolved"] and comparisons.loc[6, "ratio"] > 1
    full = " ".join(result.summary(detail="full").split())
    assert "smallest standard error that resolved here is with 24 samples of about 4 fields" in full
    assert "Most of the uncertainty of the mean is between samples" in full
    assert "24 samples of about 4 fields" in result.summary()
    assert "between samples" in result.summary()


def test_fields_without_observations_count_as_acquired_effort():
    rng = np.random.default_rng(8)
    objects, fovs = _simulate(rng, TRUTH, 1, 6, 6, 20)
    extra = pd.DataFrame(
        {
            "biological_repeat": "R0",
            "well": "W0",
            "image_file": ["empty1", "empty2"],
            "condition": "ctrl",
        }
    )
    with_empty = pd.concat([fovs, extra], ignore_index=True)
    hierarchy = SamplingHierarchy("image_file", "well")
    plain = effort_projection(objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20)
    empty = effort_projection(
        objects, with_empty, feature="x", hierarchy=hierarchy, strata=[], n_boot=20
    )
    # The same observations over more acquired FOVs: the acquired design's SE is
    # unchanged (empty FOVs average out nothing), and they are reported as effort.
    se = plain.projections.set_index("design")["se"]["acquired"]
    se_empty = empty.projections.set_index("design")["se"]["acquired"]
    assert se_empty == pytest.approx(se)
    level = empty.components.iloc[0]
    assert level["fovs"] == 38 and level["fovs_with_observations"] == 36
    assert "2 of 38 acquired fields had no cells" in empty.summary()


def test_empty_fields_do_not_shrink_the_projected_error_or_design_effect():
    # Audit regression: with 60% empty FOVs the projection was about 19% too small
    # and the design effect halved.
    rng = np.random.default_rng(12)
    objects, fovs = _simulate(rng, (0.0, 0.5, 0.5, 1.0), 1, 6, 10, 30)
    hierarchy = SamplingHierarchy("image_file", "well")
    full = effort_projection(objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20)
    kept = fovs["image_file"].iloc[::5].tolist() + fovs["image_file"].iloc[1::5].tolist()
    sparse = objects[objects["image_file"].isin(kept)]
    thin = effort_projection(sparse, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20)
    assert design_effect(thin, "all") == pytest.approx(design_effect(full, "all"), rel=0.4)
    se_full = full.projections.set_index("design")["se"]["acquired"]
    se_thin = thin.projections.set_index("design")["se"]["acquired"]
    assert se_thin > se_full  # fewer fields with observations: a less precise mean


def test_one_observation_per_field_is_reported_clearly():
    rng = np.random.default_rng(13)
    objects, fovs = _simulate(rng, TRUTH, 1, 4, 5, 1)
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=SamplingHierarchy("image_file", "well"), strata=[]
    )
    assert result.components.empty
    assert "single observation" in result.skipped["reason"].iloc[0]
    for detail in ("short", "full"):
        text = " ".join(result.summary(detail=detail).split())
        assert "Not estimable: every FOV holds a single observation" in text


def test_invalid_declarations_are_refused():
    rng = np.random.default_rng(9)
    objects, fovs = _simulate(rng, TRUTH, 2, 3, 3, 5)
    with pytest.raises(ValueError, match="technical levels"):
        effort_projection(
            objects,
            fovs,
            feature="x",
            hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat", ("condition",)),
            strata=[],
        )
    with pytest.raises(ValueError, match="strata missing"):
        effort_projection(
            objects,
            fovs,
            feature="x",
            hierarchy=SamplingHierarchy("image_file", "well"),
            strata=["plate"],
        )
    with pytest.raises(ValueError):
        nested_variance_components(np.array([1.0]), [np.array([0])])
    # One sample per repeat: sample and repeat variance cannot be told apart, so
    # the repeat is left out of that stratum's components and the summary says so.
    single, single_fovs = _simulate(rng, TRUTH, 3, 1, 4, 5)
    result = effort_projection(
        single,
        single_fovs,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat"),
        strata=[],
        n_boot=20,
    )
    assert list(result.components["level"]) == ["well", "image_file", "observation"]
    assert result.components["repeat_dropped"].all()
    assert (result.components["unit"] == "well").all()
    assert result.components["units"].iloc[0] == 3  # W0 of R0, R1 and R2 are three samples
    text = " ".join(result.summary(detail="full").split())
    assert "Every biological repeat holds a single sample here" in text
    assert "cannot tell from these data (only 3 samples)" in text
    short = " ".join(result.summary().split())
    assert (
        "Every biological repeat holds a single sample here" in short and "only 3 samples" in short
    )


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_sample_labels_are_refused_rather_than_pooled_into_one_sample(blank):
    # Several wells with a blank label would otherwise become one sample, and
    # change the sample count and the variance components (#240).
    rng = np.random.default_rng(3)
    objects, fovs = _simulate(rng, TRUTH, 3, 4, 3, 5)
    fovs.loc[fovs["well"].isin(["W2", "W3"]), "well"] = blank
    with pytest.raises(ValueError, match="'well' contains missing values or blank text"):
        effort_projection(
            objects.drop(columns="well"),
            fovs,
            feature="x",
            hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat"),
            strata=[],
            n_boot=20,
        )


WELLS = SamplingHierarchy(fov="image_file", sample="well")


def _full(result):
    """The full summary with its line breaks as spaces (the text is wrapped)."""
    return " ".join(result.summary(detail="full").split())


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
    assert "5 of 8 samples held no cells with a value" in text
    short = " ".join(result.summary().split())
    assert "only 3 samples" in short and "5 of 8 samples held no cells with a value" in short


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


def test_g9_the_imbalance_line_names_the_level_that_differs():
    projections = pd.DataFrame(
        {"current": [True, True], "design": ["acquired", "balanced"], "se": [1.2, 1.0]}
    )
    first = {
        "repeat_dropped": False,
        "repeat_weight": 0.6,
        "repeats": 2,
        "sample_weight": 1 / 6,
        "samples": 6,
    }
    line = effort_text._imbalance_line(projections, first, "technical repeat", "cells")
    assert line.startswith("  The acquired repeats differ in size")
    both = {**first, "sample_weight": 0.3}
    line = effort_text._imbalance_line(projections, both, "technical repeat", "cells")
    assert "The acquired repeats and technical repeats differ in size" in line


def test_g13_the_effort_summary_has_no_line_for_a_setting_it_never_records():
    objects, fovs = _plate()
    result = effort_projection(
        objects,
        fovs,
        feature="feature_a",
        hierarchy=SamplingHierarchy("image_file", "well"),
        strata=["condition"],
        n_boot=20,
    )
    assert "normalized" not in result.settings
    assert "normalised" not in source.getsource(effort_text.effort_summary)


def test_g14_projections_assume_the_same_share_of_fields_would_be_empty_again():
    objects, fovs = _plate()
    empty = fovs["image_file"].isin(["P1_A1_0", "P2_A1_0"])
    objects = objects[~objects["image_file"].isin(fovs.loc[empty, "image_file"])]
    result = effort_projection(
        objects,
        fovs,
        feature="feature_a",
        hierarchy=SamplingHierarchy("image_file", "well", "biological_repeat"),
        strata=["condition"],
        n_boot=20,
    )
    assert "the same share would be empty again." in " ".join(result.summary().split())
    assert "the same share would be empty again." in " ".join(result.summary(detail="full").split())


def test_j_shared_helpers_and_declared_privates():
    counts = np.array([0, 3, 10, 25])
    design = effort_design._design(
        pd.DataFrame({"f": np.repeat(["a", "b", "c"], [3, 10, 25]), "s": "w", "value": 1.0}),
        ["s", "f"],
        False,
        4,
        SamplingHierarchy("f", "s"),
    )
    assert design["cluster_size"] == pytest.approx(statistics.cluster_size(counts))  # J-4
    assert effort_design.__all__ == []  # J-16
    assert "Shared-private:" in (effort_design.__doc__ or "")  # J-12
    from morphocoverage.retrospective import report_profile

    assert "Shared-private:" in (report_profile.__doc__ or "")


def test_p3_10_effort_intervals_are_widened_for_few_units():
    draws = np.random.default_rng(2).normal(1.0, 0.1, 1000)
    plain_low, plain_high = np.quantile(draws, [0.025, 0.975])
    low, high = effort_interval(draws, 1.0, 3)
    # sqrt(3 / 2) t_2 / z = 2.69 on each side with 3 units.
    assert (1.0 - low) / (1.0 - plain_low) == pytest.approx(2.69, abs=0.05)
    assert (high - 1.0) / (plain_high - 1.0) == pytest.approx(2.69, abs=0.05)
    # A standard error cannot be negative, however wide the interval.
    assert effort_interval(np.abs(draws - 1.0), 0.01, 3, floor=0.0)[0] >= 0.0
    assert effort_interval(draws, 1.0, 1) == (None, None)


def test_f1_unequal_fields_weigh_the_field_term_by_their_cluster_size():
    # 10 fields of 2 cells and 2 of 100: N = 220, sum n^2 = 20040.
    sizes = np.array([2] * 10 + [100] * 2, dtype=float)
    n, cluster = sizes.sum(), float(np.sum(sizes**2) / sizes.sum())
    design = {
        "has_repeat": False,
        "repeats": 1,
        "samples": 1,
        "total_fovs": len(sizes),
        "yield": 1.0,
        "per_fov": n / len(sizes),
        "cluster_size": cluster,
        "sample_weight": 1.0,  # a single sample holds every observation
    }
    components = np.array([0.0, 1.0, 1.0])
    truth = np.sqrt(1.0 * np.sum(sizes**2) / n**2 + 1.0 / n)
    assert _se(components, design, 1, len(sizes)) == pytest.approx(truth)
    shares = _contributions(components, design)
    field_part, cell_part = np.sum(sizes**2) / n**2, 1 / n
    assert shares[1] == pytest.approx(field_part / (field_part + cell_part))


def test_f2_one_repeat_or_one_sample_is_explained_not_a_linear_algebra_error():
    rng = np.random.default_rng(1)
    objects, fovs = _simulate(rng, TRUTH, 1, 4, 5, 10)
    hierarchy = SamplingHierarchy("image_file", "well", "biological_repeat")
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20
    )
    assert "There is a single biological repeat here" in result.summary()
    objects, fovs = _simulate(rng, TRUTH, 1, 1, 5, 10)
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=SamplingHierarchy("image_file", "well"), strata=[]
    )
    assert result.components.empty
    assert "a single 'well' unit holds observations" in result.skipped["reason"].iloc[0]
    # One stratum that cannot be estimated does not stop the others.
    first, first_fovs = _simulate(rng, TRUTH, 3, 3, 4, 10)
    second, second_fovs = _simulate(rng, TRUTH, 1, 1, 4, 10)
    second, second_fovs = second.assign(condition="B"), second_fovs.assign(condition="B")
    result = effort_projection(
        pd.concat([first, second]),
        pd.concat([first_fovs, second_fovs]),
        feature="x",
        hierarchy=hierarchy,
        strata=["condition"],
        n_boot=20,
    )
    assert set(result.components["condition"]) == {"ctrl"}
    assert list(result.skipped["condition"]) == ["B"]


def test_f3_no_alternative_at_the_same_total_is_not_called_an_overlap():
    rng = np.random.default_rng(2)
    objects, fovs = _simulate(rng, (0.0, 1.0, 0.3, 4.0), 1, 12, 6, 10)
    result = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "well"),
        strata=[],
        allocations=[(6, 20)],
        n_boot=20,
    )
    assert "no allocation compared has the acquired total of fields" in _full(result)
    assert "no allocation compared has the acquired total of fields" in " ".join(
        result.summary().split()
    )
    assert "overlap" not in _full(result) and "overlap" not in result.summary()


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
    text = " ".join(result.summary(detail="full").split())
    assert "Most of the uncertainty of the mean is between samples" in text
    assert "; not established: only 3 samples, its interval includes zero)." in text
    short = " ".join(result.summary().split())
    assert "between samples" in short
    assert "The largest share is not established: only 3 samples, its interval includes" in short
    profile = microscopist_report(effort=result).profile
    statement = profile[profile["item"] == "largest source of uncertainty of the mean"]
    assert "not established: only 3 samples" in statement["statement"].iloc[0]


def test_a_variance_share_from_many_samples_with_a_clear_component_is_stated_plainly():
    # 12 samples: from 8, the interval widened for the number of samples (audit 4,
    # P3-10) rightly reaches zero for a variance estimated from 8 values.
    objects, fovs = _nested(np.random.default_rng(0), 12, 8, 20, v_sample=4.0, v_fov=0.05)
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


def _allocations(ratios):
    """Equal-total allocations, each resolved by its own interval, with these SE ratios."""
    return pd.DataFrame(
        {
            "samples": [10 * (i + 2) for i in range(len(ratios))],
            "fovs_per_sample": [4.0] * len(ratios),
            "ratio": ratios,
            "ratio_low": [r - 0.05 for r in ratios],
            "ratio_high": [r - 0.01 for r in ratios],
            "same_total": [True] * len(ratios),
            "resolved": [True] * len(ratios),
            "current_samples": [12] * len(ratios),
            "current_fovs_per_sample": [4.0] * len(ratios),
        }
    )


def test_the_smallest_resolved_allocation_is_described_not_called_the_answer():
    compared = _allocations([0.9, 0.8, 0.85])
    line = effort_text._allocation_line(compared, {}, 12, "biological_repeat", "sample")
    assert "the smallest standard error that resolved here is with 30 samples" in line
    assert "20% smaller than the acquired 12 samples" in line
    assert "would give" not in line and "best" not in line
    answer, reason = effort_text._allocation_cell(compared, {}, 12, "biological_repeat", "sample")
    assert reason is None
    assert answer == "30 samples of about 4 fields: 20% smaller SE, the smallest that resolved here"


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
