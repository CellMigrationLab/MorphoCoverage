"""Tests for JCB/Zweifach multi-sample planning and the planned cell-count strategy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_strategies import _area_replay, _mixture, _words

from morphocoverage import (
    Decision,
    ObjectCountPlan,
    ObjectiveConfig,
    PlannedObjectCountStrategy,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    plan_objects_to_average,
    run_strategies,
)
from morphocoverage.strategies import jcb_planning
from morphocoverage.strategies.scalar_precision import sd_upper_bound_factor

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"


def _centred(n, sd, seed):
    """n values with mean exactly 0 and SD (ddof=1) exactly sd."""
    z = np.random.default_rng(seed).normal(size=n)
    z = z - z.mean()
    return sd * z / np.std(z, ddof=1)


def _pilot(samples):
    """Objects and FOVs for pilot samples given as (condition, dish, values), one FOV each."""
    rows = [
        {
            "condition": condition,
            "biological_repeat": dish,
            "image_file": f"{dish}.tif",
            "area": value,
        }
        for condition, dish, values in samples
        for value in values
    ]
    objects = pd.DataFrame(rows)
    fovs = objects.drop_duplicates("image_file").drop(columns="area")
    return objects, fovs.reset_index(drop=True)


def _plan(objects, fovs, **kwargs):
    settings = {
        "feature": "area",
        "target_cv": 0.025,
        "sd_estimate": "point",
        "sample_cols": ("condition", "biological_repeat"),
        "control": {"condition": "control"},
    }
    settings.update(kwargs)
    return plan_objects_to_average(objects, fovs=fovs, **settings)


def _two_dishes(sd=0.1778, means=(1.0, 1.1)):
    # Reproduce the paper's reported worked example with one control and one
    # treated dish, 10 cells each. The centred data set is engineered to have
    # SD ``sd`` relative to the control mean of 1.0, matching the reported
    # 17.8% point estimate rather than the inconsistent visible Data S1 inputs.
    centred = _centred(20, sd, seed=3)
    centred[:10] -= centred[:10].mean()
    centred[10:] -= centred[10:].mean()
    centred *= sd / np.std(centred, ddof=1)
    return _pilot(
        [
            ("control", "dish1", means[0] + centred[:10]),
            ("drug", "dish2", means[1] + centred[10:]),
        ]
    )


def test_reproduces_the_papers_worked_example_from_two_dishes():
    objects, fovs = _two_dishes()
    counts = {}
    for target in (0.025, 0.05):
        for sd_estimate in ("point", "upper_95"):
            plan = _plan(objects, fovs, target_cv=target, sd_estimate=sd_estimate)
            counts[target, sd_estimate] = plan.required_objects
    assert round(plan.cv, 3) == 0.178
    assert round(plan.cv_upper_95, 3) == 0.260
    assert counts == {
        (0.025, "point"): 51,
        (0.025, "upper_95"): 108,
        (0.05, "point"): 13,
        (0.05, "upper_95"): 27,
    }
    assert (plan.n_samples, plan.n_control_samples, plan.n_objects) == (2, 1, 20)


def test_centring_removes_differences_between_sample_means():
    # Keep the control normalization fixed and move only the treated mean:
    # centring each sample removes that between-sample offset from the SD.
    close = _plan(*_two_dishes(means=(1.0, 1.01)))
    apart = _plan(*_two_dishes(means=(1.0, 1.5)))
    assert close.cv == pytest.approx(apart.cv, rel=1e-12)
    # Without centring, the dish offset would inflate the raw SD.
    objects, _ = _two_dishes(means=(1.0, 1.5))
    assert np.std(objects["area"], ddof=1) > apart.cv


def test_sd_is_expressed_relative_to_all_control_observations():
    # Supplementary Data S1/S3 normalize to the mean of all control-cell
    # observations. With unequal control sample sizes that differs from the
    # equal-weight mean of control sample means.
    rng = np.random.default_rng(0)
    parts = [
        ("control", "c1", 10 + _centred(5, 1.0, 1)),
        ("control", "c2", 20 + _centred(15, 1.0, 2)),
        ("drug", "d1", 40 + rng.normal(0, 3, 10)),
    ]
    objects, fovs = _pilot(parts)
    plan = _plan(objects, fovs)
    assert plan.control_mean == pytest.approx(17.5)
    centred = np.concatenate([values - values.mean() for _, _, values in parts])
    assert plan.cv == pytest.approx(np.std(centred, ddof=1) / 17.5)
    assert plan.n_control_samples == 2 and plan.n_samples == 3


def test_upper_bound_uses_all_cells_minus_one_degrees_of_freedom():
    plan = _plan(*_two_dishes())
    assert plan.cv_upper_95 == pytest.approx(plan.cv * sd_upper_bound_factor(19))


def test_datas1_visible_inputs_recalculate_differently_from_cached_reported_result():
    # Supplementary Data S1 ships with cached result cells matching the paper's
    # reported 17.8% / 26.0% example, but its visible 10 control and 10 treated
    # values recalculate to 20.3808% / 29.7676%. Keep this discrepancy explicit
    # so a future refactor does not silently "validate" against stale cache data.
    control = np.array(
        [
            19.162246666666665,
            27.77174888888889,
            26.188026666666666,
            19.359966666666665,
            25.781626666666668,
            22.64094,
            20.364784444444446,
            16.210504444444446,
            20.26476,
            17.732388888888888,
        ]
    )
    treated = np.array(
        [
            35.442235555555555,
            21.09789111111111,
            30.06036222222222,
            25.09115111111111,
            17.61352,
            27.524764444444443,
            28.43596888888889,
            25.922217777777778,
            28.28141111111111,
            21.952986666666664,
        ]
    )
    objects, fovs = _pilot(
        [
            ("control", "dish1", control),
            ("drug", "dish2", treated),
        ]
    )
    plan = _plan(objects, fovs, target_cv=0.025, sd_estimate="upper_95")
    assert plan.control_mean == pytest.approx(21.547699333333334)
    assert plan.cv == pytest.approx(0.20380765852997446)
    assert plan.cv_upper_95 == pytest.approx(0.2976756852727519)
    assert plan.required_objects == 142

    point = _plan(objects, fovs, target_cv=0.025, sd_estimate="point")
    assert point.required_objects == 67


@pytest.mark.parametrize(
    "change, message",
    [
        ({"control": {"condition": "none"}}, "no pilot sample matches the control"),
        ({"control": {}}, "control must map"),
        ({"sample_cols": ("condition", "well")}, "not columns"),
        ({"sample_cols": "condition"}, "non-empty sequence"),
        ({"target_cv": 0}, "target_cv"),
        ({"sd_estimate": "mean"}, "sd_estimate"),
        ({"observation_unit": "patch"}, "patch"),
    ],
)
def test_invalid_plans_are_refused(change, message):
    objects, fovs = _two_dishes()
    with pytest.raises((TypeError, ValueError), match=message):
        _plan(objects, fovs, **change)


@pytest.mark.parametrize("value, message", [(np.nan, "missing values"), (-1.0, "negative")])
def test_missing_or_negative_pilot_values_are_refused(value, message):
    objects, fovs = _two_dishes()
    objects.loc[0, "area"] = value
    with pytest.raises(ValueError, match=message):
        _plan(objects, fovs)


def test_planning_needs_two_cells_and_a_positive_control_mean():
    objects, fovs = _pilot([("control", "c1", [5.0])])
    with pytest.raises(ValueError, match="at least 2"):
        _plan(objects, fovs)
    objects, fovs = _pilot([("control", "c1", [0.0, 0.0]), ("drug", "d1", [1.0, 2.0])])
    with pytest.raises(ValueError, match="positive"):
        _plan(objects, fovs)


def _evaluation(n_fovs=6, cells_per_fov=10, repeat="R1"):
    """An evaluated control repeat, not used for planning."""
    rows = [
        {
            "condition": "control",
            "biological_repeat": repeat,
            "image_file": f"{repeat}_f{j}.tif",
            "area": 1.0 + 0.01 * k,
        }
        for j in range(n_fovs)
        for k in range(cells_per_fov)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects.drop_duplicates("image_file").drop(columns="area").reset_index(drop=True)
    return objects, fovs.assign(order=range(n_fovs))


def _replay(objects, fovs, strategy, **kwargs):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[strategy],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=POPULATION,
        **kwargs,
    )


def test_strategy_stops_at_the_first_fov_that_reaches_the_planned_count():
    plan = _plan(*_two_dishes(), target_cv=0.05)  # 13 cells
    objects, fovs = _evaluation()
    results = _replay(objects, fovs, PlannedObjectCountStrategy(plan))
    decisions = [(item.result.n_observations, item.result.decision) for item in results]
    assert decisions == [
        (10, Decision.CONTINUE),
        (20, Decision.STOP),
        (30, Decision.STOP),
        (40, Decision.STOP),
        (50, Decision.STOP),
        (60, Decision.STOP),
    ]
    assert results[0].result.evidence.value("planned_required_objects") == 13


def test_groups_containing_pilot_samples_are_not_judged_on_their_own_plan():
    objects, fovs = _two_dishes()
    plan = _plan(objects, fovs)
    results = run_strategies(
        objects,
        fovs=fovs,
        strategies=[PlannedObjectCountStrategy(plan)],
        feature_cols=["area"],
        objective=POPULATION,
        group_cols=["condition"],
    )
    assert len(results) == 2
    for result in results:
        assert result.decision is None
        quantity = result.evidence.quantities["planned_required_objects"]
        assert quantity.availability.value == "not_applicable"
        assert "pilot samples" in quantity.reason


def test_plan_does_not_apply_when_samples_cannot_be_identified_or_rows_are_not_cells():
    plan = _plan(*_two_dishes(), sample_cols=("condition", "biological_repeat", "image_file"))
    objects, fovs = _evaluation()
    [unidentified] = run_strategies(
        objects.drop(columns="image_file").assign(position_id=objects["image_file"]),
        fovs=fovs.drop(columns="image_file").assign(position_id=fovs["image_file"]),
        strategies=[PlannedObjectCountStrategy(plan)],
        feature_cols=["area"],
        objective=POPULATION,
    )
    assert unidentified.decision is None
    assert (
        "cannot be excluded" in unidentified.evidence.quantities["planned_required_objects"].reason
    )
    [patches] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[PlannedObjectCountStrategy(plan)],
        feature_cols=["area"],
        objective=POPULATION,
        observation_unit="patch",
    )
    assert patches.decision is None


def test_plan_is_provenance():
    objects, fovs = _two_dishes()
    point, upper = _plan(objects, fovs), _plan(objects, fovs, sd_estimate="upper_95")
    record = json.loads(json.dumps(point.parameters))
    assert record["pilot_samples"] == [["control", "dish1"], ["drug", "dish2"]]
    assert record["control"] == {"condition": "control"}
    first = PlannedObjectCountStrategy(point)
    assert first.config_id == PlannedObjectCountStrategy(_plan(objects, fovs)).config_id
    assert first.config_id != PlannedObjectCountStrategy(upper).config_id
    with pytest.raises(TypeError, match="ObjectCountPlan"):
        PlannedObjectCountStrategy(point.parameters)
    assert isinstance(point, ObjectCountPlan)


def test_real_pilot_wells_plan_the_other_wells():
    # idr0139: plan on two DMSO control wells and two Leptomycin b wells, one
    # sample per well, then stop every other well at the planned count.
    plate = {"biological_repeat": str}
    fovs = pd.read_csv(DATA / "idr0139_fovs.csv.gz", dtype=plate)
    objects = pd.read_csv(DATA / "idr0139_nuclei.csv.gz", dtype=plate)
    nuclear = objects["Nuclei_Intensity_MeanIntensity_Fascin"]
    objects["Fascin_Ratio"] = nuclear / (
        nuclear + objects["Cytoplasm_Intensity_MeanIntensity_Fascin"]
    )
    pilot_wells = ["B02", "N12", "H13", "L18"]
    plan = plan_objects_to_average(
        objects[objects["technical_repeat"].isin(pilot_wells)],
        fovs=fovs[fovs["technical_repeat"].isin(pilot_wells)],
        feature="Fascin_Ratio",
        target_cv=0.05,
        sd_estimate="upper_95",
        sample_cols=("biological_repeat", "technical_repeat"),
        control={"condition": "DMSO"},
        group_cols=["condition", "biological_repeat", "technical_repeat"],
    )
    assert (plan.n_samples, plan.n_control_samples) == (4, 2)
    assert 0 < plan.cv < plan.cv_upper_95 < 1
    assert plan.required_objects >= 1

    fovs = fovs.assign(order=fovs.groupby("technical_repeat").cumcount())
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[PlannedObjectCountStrategy(plan)],
        feature_cols=["Fascin_Ratio"],
        acquisition_order_col="order",
        objective=POPULATION,
        group_cols=["condition", "biological_repeat", "technical_repeat"],
    )
    for item in results:
        well = item.result.group["technical_repeat"]
        if well in pilot_wells:
            assert item.result.decision is None
        else:
            reached = item.result.n_observations >= plan.required_objects
            assert item.result.decision is (Decision.STOP if reached else Decision.CONTINUE)


def test_c2_a_plan_whose_samples_are_constant_is_refused():
    pilot = pd.DataFrame(
        {
            "condition": ["ctrl"] * 4 + ["drug"] * 4,
            "biological_repeat": ["r1", "r1", "r2", "r2"] * 2,
            "image_file": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "n_nuclei": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="within-sample SD .* is zero"):
        plan_objects_to_average(
            pilot,
            feature="n_nuclei",
            target_cv=0.05,
            sd_estimate="upper_95",
            sample_cols=["condition", "biological_repeat"],
            control={"condition": "ctrl"},
        )
    # A hand-made plan of 0 objects cannot drive the strategy either.
    fields = {name: None for name in ObjectCountPlan.__dataclass_fields__}
    fields.update(sample_cols=("condition",), control=(), pilot_samples=(), required_objects=0)
    with pytest.raises(ValueError, match="requires no object"):
        PlannedObjectCountStrategy(ObjectCountPlan(**fields))


def _area_plan():
    rng = np.random.default_rng(0)
    pilot = pd.DataFrame(
        {
            "condition": ["ctrl"] * 20 + ["drug"] * 20,
            "biological_repeat": "p",
            "image_file": [f"p{i // 5}" for i in range(40)],
            "area": rng.normal(100, 30, 40),
        }
    )
    return plan_objects_to_average(
        pilot,
        feature="area",
        target_cv=0.05,
        sd_estimate="point",
        sample_cols=["condition", "biological_repeat"],
        control={"condition": "ctrl"},
    )


def test_c3_the_planned_count_counts_measured_values_of_the_planned_feature():
    plan = _area_plan()
    n = plan.required_objects
    area = np.random.default_rng(1).normal(100, 30, n)
    area[: n // 2] = np.nan
    objects = pd.DataFrame(
        {
            "condition": "ctrl",
            "biological_repeat": "e",
            "image_file": [f"e{i % 4}" for i in range(n)],
            "area": area,
        }
    )
    fovs = pd.DataFrame(
        {"condition": "ctrl", "biological_repeat": "e", "image_file": [f"e{j}" for j in range(4)]}
    ).assign(order=range(4))
    last = _area_replay(objects, fovs, PlannedObjectCountStrategy(plan))[-1].result
    assert last.n_observations == n
    assert last.evidence.value("measured_objects") == n - n // 2
    assert last.diagnostics["objects_without_value"] == n // 2
    assert last.decision == "continue"
    # The planned feature must be evaluated: another feature is not counted instead.
    with pytest.raises(ValueError, match="'area' is not one of the explicitly selected"):
        _area_replay(
            objects.drop(columns="area").assign(other=1.0),
            fovs,
            PlannedObjectCountStrategy(plan),
            features=("other",),
        )


def test_c12_a_blank_sample_column_in_the_pilot_is_refused_with_a_reason():
    pilot = pd.DataFrame(
        {
            "condition": ["ctrl"] * 6 + ["drug"] * 6,
            "biological_repeat": "r1",
            "well": ["A1", "A1", "A1", "A2", "A2", np.nan] + ["B1"] * 3 + ["B2"] * 3,
            "image_file": [f"p{i}" for i in range(12)],
            "area": [10, 12, 11, 9, 13, 30, 20, 22, 21, 19, 23, 18.0],
        }
    )
    with pytest.raises(ValueError, match=r"sample_cols \['well'\] have missing values"):
        plan_objects_to_average(
            pilot,
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=["condition", "biological_repeat", "well"],
            control={"condition": "ctrl"},
        )


def test_c5_categorical_sample_columns_plan_only_the_samples_in_the_pilot(monkeypatch):
    # pandas 2 groups categoricals over every combination of categories unless told not to.
    groupby = pd.DataFrame.groupby

    def pandas2_groupby(self, *args, **kwargs):
        kwargs.setdefault("observed", False)
        return groupby(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", pandas2_groupby)
    rng = np.random.default_rng(0)
    rows = [
        {
            "condition": c,
            "biological_repeat": r,
            "position_id": f"{c}{r}{j}",
            "area": rng.normal(10, 2),
        }
        for c, r in (("control", "r1"), ("control", "r2"), ("treated", "r1"))
        for j in range(3)
        for _ in range(20)
    ]
    pilot = pd.DataFrame(rows).astype({"condition": "category", "biological_repeat": "category"})
    plan = plan_objects_to_average(
        pilot,
        feature="area",
        target_cv=0.05,
        sd_estimate="point",
        sample_cols=["condition", "biological_repeat"],
        control={"condition": "control"},
        fov_col="position_id",
    )
    assert plan.n_samples == 3
    assert ("treated", "r2") not in plan.pilot_samples


def _two_well_pilot():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "condition": np.repeat(["DMSO", "drug"], 40),
            "biological_repeat": "r1",
            "well": np.tile(np.repeat(["A1", "A2"], 20), 2),
            "position_id": np.repeat([f"p{i}" for i in range(8)], 10),
            "area": rng.normal(10, 2, 80),
        }
    )


def test_c12_a_control_column_or_value_the_pilot_lacks_is_named():
    def plan(control):
        return plan_objects_to_average(
            _two_well_pilot(),
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=["condition", "biological_repeat", "well"],
            control=control,
            fov_col="position_id",
        )

    with pytest.raises(ValueError, match=r"control columns \['treatment'\] are not columns"):
        plan({"treatment": "DMSO"})
    with pytest.raises(ValueError, match="the pilot holds condition: 'DMSO' and 'drug'"):
        plan({"condition": "dmso"})
    assert plan({"condition": "DMSO"}).n_control_samples == 2


def test_c13_the_spreadsheet_sd_is_kept_and_its_small_pilot_bias_is_documented():
    pilot = _two_well_pilot()
    plan = plan_objects_to_average(
        pilot,
        feature="area",
        target_cv=0.05,
        sd_estimate="point",
        sample_cols=["condition", "biological_repeat", "well"],
        control={"condition": "DMSO"},
        fov_col="position_id",
    )
    keys = ["condition", "biological_repeat", "well"]
    centred = pilot["area"] - pilot.groupby(keys)["area"].transform("mean")
    control_mean = pilot.loc[pilot["condition"] == "DMSO", "area"].mean()
    # ddof=1 over all objects after centring, as the paper's spreadsheet computes it.
    assert plan.cv == pytest.approx(np.std(centred, ddof=1) / control_mean)
    doc = jcb_planning.__doc__
    assert "(n_objects - n_samples) / (n_objects - 1)" in _words(doc)
    assert "3 of 3, 76% and 90%" in _words(doc)


def _sample_pilot(values_by_sample):
    rows = [
        {
            "condition": condition,
            "biological_repeat": "r1",
            "image_file": f"{condition}{j}",
            "x": value,
        }
        for condition, samples in values_by_sample.items()
        for j, values in enumerate(samples)
        for value in values
    ]
    return pd.DataFrame(rows)


def test_p4_7_the_plan_carries_cautions_for_values_far_from_normal_and_a_cv_above_50():
    rng = np.random.default_rng(9)
    settings = {
        "feature": "x",
        "target_cv": 0.05,
        "sample_cols": ["condition", "image_file"],
        "control": {"condition": "ctrl"},
    }
    mixed = _sample_pilot({"ctrl": [_mixture(rng, 30) for _ in range(3)]})
    plan = plan_objects_to_average(mixed, sd_estimate="upper_95", **settings)
    assert any(w.startswith("the upper-95 SD bound assumes normal values") for w in plan.warnings)
    assert "warnings" not in plan.parameters  # the cautions do not define the plan
    point = plan_objects_to_average(mixed, sd_estimate="point", **settings)
    assert not any("normal values" in w for w in point.warnings)
    wide = _sample_pilot({"ctrl": [list(rng.exponential(10.0, 30)) for _ in range(3)]})
    assert any(
        "exceeds 50% CV of the control mean" in w
        for w in plan_objects_to_average(wide, sd_estimate="point", **settings).warnings
    )
    normal = _sample_pilot({"ctrl": [list(rng.normal(10, 1, 30)) for _ in range(3)]})
    assert plan_objects_to_average(normal, sd_estimate="upper_95", **settings).warnings == ()
    # The planned rule shows the plan's cautions beside its evidence.
    treated = pd.DataFrame(
        {"condition": "drug", "biological_repeat": "r1", "image_file": "d0", "x": [1.0, 2.0, 3.0]}
    )
    [result] = run_strategies(
        treated,
        strategies=[PlannedObjectCountStrategy(plan)],
        feature_cols=["x"],
        objective=POPULATION,
    )
    assert result.evidence.diagnostics["warning"] == "plan: " + "; ".join(plan.warnings)


def test_c2_planning_refuses_a_pilot_of_one_object_per_sample():
    objects = pd.DataFrame(
        {
            "condition": ["control", "control", "drug", "drug"],
            "biological_repeat": ["r1", "r2", "r3", "r4"],
            "image_file": ["a", "b", "c", "d"],
            "area": [10.0, 30.0, 5.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="2 or more cells"):
        plan_objects_to_average(
            objects,
            fovs=objects.drop(columns="area"),
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=("condition", "biological_repeat"),
            control={"condition": "control"},
        )
