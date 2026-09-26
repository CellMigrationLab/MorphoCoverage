"""Tests for the scalar hierarchical effort projection (roadmap Step 7)."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective import (
    SamplingHierarchy,
    design_effect,
    effort_projection,
    nested_variance_components,
)
from morphocoverage.retrospective.effort import _se


def _simulate(rng, components, repeats, samples, fovs, cells, balanced=True):
    """Observation and FOV tables of a nested design with known variance components."""
    v_rep, v_sample, v_fov, v_obs = components
    object_rows, fov_rows = [], []
    for r in range(repeats):
        rep = rng.normal(0, np.sqrt(v_rep))
        n_samples = samples if balanced else rng.integers(max(2, samples - 2), samples + 3)
        for s in range(n_samples):
            sample = rng.normal(0, np.sqrt(v_sample))
            n_fovs = fovs if balanced else rng.integers(max(2, fovs - 3), fovs + 4)
            for f in range(n_fovs):
                field = rng.normal(0, np.sqrt(v_fov))
                name = f"r{r}_s{s}_f{f}"
                fov_rows.append({"repeat": f"R{r}", "well": f"W{s}", "image_file": name})
                m = cells if balanced else rng.integers(max(1, cells - 10), cells + 11)
                values = rep + sample + field + rng.normal(0, np.sqrt(v_obs), m)
                object_rows.append(pd.DataFrame({"image_file": name, "x": values}))
    fovs_table = pd.DataFrame(fov_rows).assign(condition="ctrl")
    objects = pd.concat(object_rows, ignore_index=True)
    objects = objects.merge(fovs_table, on="image_file")
    return objects, fovs_table


def _codes(objects):
    keys = [
        objects["repeat"],
        objects["repeat"] + objects["well"],
        objects["image_file"],
    ]
    return [pd.factorize(k)[0] for k in keys]


TRUTH = (0.5, 1.0, 0.3, 4.0)


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
        hierarchy=SamplingHierarchy("image_file", "well", "repeat"),
        strata=["condition"],
        n_boot=100,
    )
    text = result.summary()
    assert "cannot tell from these data (only 3 repeats)" in text
    assert not result.comparisons["resolved"].any()
    levels = result.components.set_index("level")
    assert list(levels.index) == ["repeat", "well", "image_file", "observation"]
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
    assert "24 samples of about 4 fields would give a" in result.summary()
    assert "Most of the uncertainty of the mean is between samples" in result.summary()


def test_fields_without_observations_count_as_acquired_effort():
    rng = np.random.default_rng(8)
    objects, fovs = _simulate(rng, TRUTH, 1, 6, 6, 20)
    extra = pd.DataFrame(
        {"repeat": "R0", "well": "W0", "image_file": ["empty1", "empty2"], "condition": "ctrl"}
    )
    with_empty = pd.concat([fovs, extra], ignore_index=True)
    hierarchy = SamplingHierarchy("image_file", "well")
    plain = effort_projection(objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20)
    empty = effort_projection(
        objects, with_empty, feature="x", hierarchy=hierarchy, strata=[], n_boot=20
    )
    # The same observations over more acquired FOVs: the acquired design's SE is
    # unchanged (empty FOVs average out nothing), and they are reported as effort.
    se = plain.projections.set_index("current")["se"][True]
    se_empty = empty.projections.set_index("current")["se"][True]
    assert se_empty == pytest.approx(se)
    level = empty.components.iloc[0]
    assert level["fovs"] == 38 and level["fovs_with_observations"] == 36
    assert "2 of 38 acquired fields had no objects" in empty.summary()


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
    se_full = full.projections.set_index("current")["se"][True]
    se_thin = thin.projections.set_index("current")["se"][True]
    assert se_thin > se_full  # fewer fields with observations: a less precise mean


def test_one_observation_per_field_is_reported_clearly():
    rng = np.random.default_rng(13)
    objects, fovs = _simulate(rng, TRUTH, 1, 4, 5, 1)
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=SamplingHierarchy("image_file", "well"), strata=[]
    )
    assert result.components.empty
    assert "single observation" in result.skipped["reason"].iloc[0]
    assert "Not estimable: every FOV holds a single observation" in result.summary()


def test_invalid_declarations_are_refused():
    rng = np.random.default_rng(9)
    objects, fovs = _simulate(rng, TRUTH, 2, 3, 3, 5)
    with pytest.raises(ValueError, match="technical levels"):
        effort_projection(
            objects,
            fovs,
            feature="x",
            hierarchy=SamplingHierarchy("image_file", "well", "repeat", ("condition",)),
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
        hierarchy=SamplingHierarchy("image_file", "well", "repeat"),
        strata=[],
        n_boot=20,
    )
    assert list(result.components["level"]) == ["well", "image_file", "observation"]
    assert result.components["repeat_dropped"].all()
    assert (result.components["unit"] == "well").all()
    assert result.components["units"].iloc[0] == 3  # W0 of R0, R1 and R2 are three samples
    text = result.summary()
    assert "Every repeat holds a single sample here" in text
    assert "cannot tell from these data (only 3 samples)" in text
