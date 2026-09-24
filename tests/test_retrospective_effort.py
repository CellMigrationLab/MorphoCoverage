"""Tests for the scalar hierarchical effort projection (roadmap Step 7)."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective import (
    SamplingHierarchy,
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
        expected = _se(components, 2, samples, fovs, 15, True)
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
    assert "24 samples x 4 fields would give a" in result.summary()
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
    # Same observations over more acquired FOVs: fewer observations per FOV, higher SE.
    se = plain.projections.set_index(["samples", "fovs_per_sample"])["se"]
    se_empty = empty.projections.set_index(["samples", "fovs_per_sample"])["se"]
    common = se.index.intersection(se_empty.index)
    assert len(common) and (se_empty[common] >= se[common]).all()


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
    # One sample per repeat: sample and repeat variance cannot be told apart.
    single, single_fovs = _simulate(rng, TRUTH, 3, 1, 4, 5)
    with pytest.raises(ValueError, match="cannot be separated"):
        effort_projection(
            single,
            single_fovs,
            feature="x",
            hierarchy=SamplingHierarchy("image_file", "well", "repeat"),
            strata=[],
        )
