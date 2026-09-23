"""Tests for JCB/Zweifach multi-sample planning and the planned cell-count strategy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage import (
    Decision,
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    run_strategies,
)
from morphocoverage.strategies import (
    CellCountPlan,
    PlannedCellCountStrategy,
    plan_cells_to_average,
)
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
        {"condition": condition, "repeat": dish, "image_file": f"{dish}.tif", "area": value}
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
        "sample_cols": ("condition", "repeat"),
        "control": {"condition": "control"},
    }
    settings.update(kwargs)
    return plan_cells_to_average(objects, fovs=fovs, **settings)


def _two_dishes(sd=0.1778, means=(0.9, 1.1)):
    # Zweifach 2024's worked example: 20 cells, 10 per dish, pooled after
    # centring each dish on its own mean. The within-dish spread is set so the
    # centred data set has SD ``sd`` relative to the control mean of 1.
    centred = _centred(20, sd, seed=3)
    centred[:10] -= centred[:10].mean()
    centred[10:] -= centred[10:].mean()
    centred *= sd / np.std(centred, ddof=1)
    return _pilot(
        [
            ("control", "dish1", means[0] + centred[:10]),
            ("control", "dish2", means[1] + centred[10:]),
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
    assert (plan.n_samples, plan.n_control_samples, plan.n_objects) == (2, 2, 20)


def test_centring_removes_differences_between_sample_means():
    close = _plan(*_two_dishes(means=(0.99, 1.01)))
    apart = _plan(*_two_dishes(means=(0.5, 1.5)))
    assert close.cv == pytest.approx(apart.cv, rel=1e-12)
    # Without centring, the dish offset would inflate the SD.
    objects, _ = _two_dishes(means=(0.5, 1.5))
    assert np.std(objects["area"], ddof=1) > 2 * apart.cv


def test_sd_is_expressed_relative_to_the_mean_of_control_sample_means():
    # Control dishes at 10 (5 cells) and 20 (15 cells): the control average is
    # 15, the mean of the dish means, not the cell-weighted 17.5. The treated
    # sample adds to the pooled SD but not to the normalizing mean.
    rng = np.random.default_rng(0)
    parts = [
        ("control", "c1", 10 + _centred(5, 1.0, 1)),
        ("control", "c2", 20 + _centred(15, 1.0, 2)),
        ("drug", "d1", 40 + rng.normal(0, 3, 10)),
    ]
    objects, fovs = _pilot(parts)
    plan = _plan(objects, fovs)
    assert plan.control_mean == pytest.approx(15.0)
    centred = np.concatenate([values - values.mean() for _, _, values in parts])
    assert plan.cv == pytest.approx(np.std(centred, ddof=1) / 15.0)
    assert plan.n_control_samples == 2 and plan.n_samples == 3


def test_upper_bound_uses_all_cells_minus_one_degrees_of_freedom():
    plan = _plan(*_two_dishes())
    assert plan.cv_upper_95 == pytest.approx(plan.cv * sd_upper_bound_factor(19))


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
            "repeat": repeat,
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
    results = _replay(objects, fovs, PlannedCellCountStrategy(plan))
    decisions = [(item.result.n_objects, item.result.decision) for item in results]
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
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[PlannedCellCountStrategy(plan)],
        feature_cols=["area"],
        objective=POPULATION,
        group_cols=["condition"],
    )
    assert result.decision is None
    quantity = result.evidence.quantities["planned_required_objects"]
    assert quantity.availability.value == "not_applicable"
    assert "pilot samples" in quantity.reason


def test_plan_does_not_apply_when_samples_cannot_be_identified_or_rows_are_not_cells():
    plan = _plan(*_two_dishes(), sample_cols=("condition", "repeat", "image_file"))
    objects, fovs = _evaluation()
    [unidentified] = run_strategies(
        objects.drop(columns="image_file").assign(position_id=objects["image_file"]),
        fovs=fovs.drop(columns="image_file").assign(position_id=fovs["image_file"]),
        strategies=[PlannedCellCountStrategy(plan)],
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
        strategies=[PlannedCellCountStrategy(plan)],
        feature_cols=["area"],
        objective=POPULATION,
        observation_unit="patch",
    )
    assert patches.decision is None


def test_plan_is_provenance():
    objects, fovs = _two_dishes()
    point, upper = _plan(objects, fovs), _plan(objects, fovs, sd_estimate="upper_95")
    record = json.loads(json.dumps(point.parameters))
    assert record["pilot_samples"] == [["control", "dish1"], ["control", "dish2"]]
    assert record["control"] == {"condition": "control"}
    first = PlannedCellCountStrategy(point)
    assert first.config_id == PlannedCellCountStrategy(_plan(objects, fovs)).config_id
    assert first.config_id != PlannedCellCountStrategy(upper).config_id
    with pytest.raises(TypeError, match="CellCountPlan"):
        PlannedCellCountStrategy(point.parameters)
    assert isinstance(point, CellCountPlan)


def test_real_pilot_wells_plan_the_other_wells():
    # idr0139: plan on two DMSO control wells and two Leptomycin b wells, one
    # sample per well, then stop every other well at the planned count.
    fovs = pd.read_csv(DATA / "idr0139_fovs.csv.gz", dtype={"repeat": str})
    objects = pd.read_csv(DATA / "idr0139_nuclei.csv.gz", dtype={"repeat": str})
    nuclear = objects["Nuclei_Intensity_MeanIntensity_Fascin"]
    objects["Fascin_Ratio"] = nuclear / (
        nuclear + objects["Cytoplasm_Intensity_MeanIntensity_Fascin"]
    )
    pilot_wells = ["B02", "N12", "H13", "L18"]
    plan = plan_cells_to_average(
        objects[objects["well"].isin(pilot_wells)],
        fovs=fovs[fovs["well"].isin(pilot_wells)],
        feature="Fascin_Ratio",
        target_cv=0.05,
        sd_estimate="upper_95",
        sample_cols=("repeat", "well"),
        control={"condition": "DMSO"},
        group_cols=["condition", "repeat", "well"],
    )
    assert (plan.n_samples, plan.n_control_samples) == (4, 2)
    assert 0 < plan.cv < plan.cv_upper_95 < 1
    assert plan.required_objects >= 1

    fovs = fovs.assign(order=fovs.groupby("well").cumcount())
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[PlannedCellCountStrategy(plan)],
        feature_cols=["Fascin_Ratio"],
        acquisition_order_col="order",
        objective=POPULATION,
        group_cols=["condition", "repeat", "well"],
    )
    for item in results:
        well = item.result.group["well"]
        if well in pilot_wells:
            assert item.result.decision is None
        else:
            reached = item.result.n_objects >= plan.required_objects
            assert item.result.decision is (Decision.STOP if reached else Decision.CONTINUE)
