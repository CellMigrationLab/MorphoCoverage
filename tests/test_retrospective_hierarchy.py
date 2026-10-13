"""Tests of ``retrospective.hierarchy``: the sampling hierarchy and its levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import _plate_curve, _wells
from _helpers_tables import FOVS, OBJECTS, _quiet
from _helpers_workflows import QUICK, _setup, _tables, _well_acquisition

from morphocoverage import integrity_checks
from morphocoverage.normalize import normalize_to_control
from morphocoverage.retrospective import (
    SamplingHierarchy,
    dataset_overview,
    effort_projection,
    inspect_fovs,
    protocol_reliability,
)
from morphocoverage.workflows import optimise

HIERARCHY = SamplingHierarchy("position_id", "condition")


CALLS = {
    "integrity feature_cols": lambda v: integrity_checks(OBJECTS, FOVS, feature_cols=v),
    "overview feature_cols": lambda v: dataset_overview(OBJECTS, FOVS, feature_cols=v),
    "inspect novelty_cols": lambda v: inspect_fovs(
        OBJECTS, FOVS, feature_cols=["x", "y"], novelty_cols=v, fov_col="position_id"
    ),
    "hierarchy technical_levels": lambda v: SamplingHierarchy("f", "s", None, v),
    "effort strata": lambda v: effort_projection(
        OBJECTS, FOVS, feature="x", hierarchy=HIERARCHY, strata=v
    ),
    "normalize feature_cols": lambda v: normalize_to_control(
        OBJECTS, FOVS, feature_cols=v, control={"condition": "ctrl"}
    ),
}


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("bad", ["xy", {"x", "y"}, {"x": 1}], ids=["string", "set", "mapping"])
def test_strings_sets_and_mappings_are_refused_not_reinterpreted(name, bad):
    with pytest.raises(TypeError, match="column names"):
        _quiet(lambda: CALLS[name](bad))


def test_technical_levels_keep_their_order():
    assert SamplingHierarchy("f", "s", None, ["plate", "well"]).technical_levels == (
        "plate",
        "well",
    )
    assert SamplingHierarchy("f", "s").technical_levels == ()


def test_f12_a_bare_string_is_not_split_into_letters():
    with pytest.raises(TypeError, match="technical_levels must be a sequence"):
        SamplingHierarchy("position_id", "well", "biological_repeat", technical_levels="plate")
    with pytest.raises(TypeError, match="strata must be a sequence"):
        protocol_reliability(
            _plate_curve({("P1", "A1"): {5: 0.9}}),
            SamplingHierarchy("image_file", "well", "plate"),
            threshold=0.9,
            strata="condition",
        )


def test_g3_the_repeat_is_left_out_for_the_reason_that_holds():
    # Repeat R1 was imaged but holds no object: one repeat has observations.
    objects, fovs = _wells([4] * 6, [20, 20, 20, 0, 0, 0], repeats=["R0"] * 3 + ["R1"] * 3)
    hierarchy = SamplingHierarchy("image_file", "well", "biological_repeat")
    result = effort_projection(
        objects, fovs, feature="x", hierarchy=hierarchy, strata=[], n_boot=20
    )
    row = result.components.iloc[0]
    assert row["repeat_dropped_reason"] == "single repeat"
    assert (row["repeats"], row["repeats_acquired"]) == (1, 2)
    text = result.summary()
    assert "Only one of the 2 biological repeats here holds cells" in text
    assert "Every repeat holds a single sample" not in text


def test_i2_a_declared_hierarchy_keeps_one_stratum_per_condition():
    objects, fovs = _well_acquisition()
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    declared = _setup(hierarchy=hierarchy, stage_cols=None)
    assert declared.resolved_effort_strata(fovs) == ["condition"]
    assert _setup(hierarchy=hierarchy, effort_strata=()).resolved_effort_strata(fovs) == []
    plan = optimise(objects, fovs, declared, **QUICK)
    strata = set(plan.results["effort"].components["stratum"])
    assert strata == {"condition=control", "condition=drug"}
    assert "condition" in plan.results["protocol"].strata


def test_g4_same_named_dishes_of_pooled_conditions_are_different_samples(synthetic):
    objects, fovs = synthetic

    def rename(table):
        table = table.copy()
        table["technical_repeat"] = "dish" + table["technical_repeat"].str[-1]
        table["image_file"] = (
            table["condition"]
            + "_"
            + table["biological_repeat"]
            + "_"
            + table["technical_repeat"]
            + "_"
        ) + table["image_file"]
        return table

    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    result = effort_projection(
        rename(objects),
        rename(fovs),
        feature="area",
        hierarchy=hierarchy,
        strata=[],
        n_boot=20,
        observation_unit="cell",
    )
    first = result.components.iloc[0]
    assert (first["samples"], first["samples_acquired"], first["repeats"]) == (30, 30, 3)


def test_g5_an_allocation_with_fewer_samples_than_repeats_covers_fewer_repeats(synthetic):
    objects, fovs = synthetic
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    result = effort_projection(
        objects,
        fovs,
        feature="area",
        hierarchy=hierarchy,
        strata=["condition"],
        allocations=[(2, 45.0)],
        n_boot=20,
        observation_unit="cell",
    )
    comp = result.components[result.components["condition"] == "drug"].set_index("level")
    v_rep, v_sample, v_fov, v_obs = np.maximum(comp["estimate"].to_numpy(), 0)
    first = comp.iloc[0]
    observations = first["fovs_with_observations"] * first["observations_per_fov"]
    size = first["cluster_size"]
    expected = np.sqrt(v_rep / 2 + v_sample / 2 + (v_fov * size + v_obs) / observations)
    projections = result.projections
    row = projections[(projections["condition"] == "drug") & (projections["samples"] == 2)]
    assert float(row["se"].iloc[0]) == pytest.approx(expected)


def test_p9_4_effort_intervals_do_not_depend_on_the_row_order_of_the_tables():
    objects, fovs = _tables(
        [
            ("ctrl", r, w, 4, 10.0 + i, 0.3)
            for i, (r, w) in enumerate(
                [("r1", "A"), ("r1", "B"), ("r2", "A"), ("r2", "B"), ("r3", "A"), ("r3", "B")]
            )
        ],
        per_field=8,
    )
    hierarchy = SamplingHierarchy("position_id", "technical_repeat", "biological_repeat")

    def components(obj, fov):
        result = effort_projection(
            obj, fov, feature="x", hierarchy=hierarchy, strata=["condition"], n_boot=100
        )
        return result.components[["level", "estimate", "low", "high"]].reset_index(drop=True)

    first = components(objects, fovs)
    shuffled = components(
        objects.sample(frac=1, random_state=1), fovs.sample(frac=1, random_state=2)
    )
    pd.testing.assert_frame_equal(first, shuffled, check_exact=False, rtol=1e-9)


def test_effort_projection_names_the_hierarchy_columns_missing():
    fovs = pd.DataFrame({"fov": ["f1"], "sample": ["s1"]})
    objects = pd.DataFrame({"fov": ["f1"], "sample": ["s1"], "x": [1.0]})
    hierarchy = SamplingHierarchy("fov", "sample", "biological_repeat")
    with pytest.raises(ValueError) as raised:
        effort_projection(objects, fovs, feature="x", hierarchy=hierarchy, strata=[])
    assert str(raised.value) == (
        "hierarchy columns missing from the FOV table: ['biological_repeat']"
    )
