"""The owner's decisions of 2026-10-02 on the scientific issues an audit raised: planning on
the values the paired test compares, the change beside a verdict, the influence on the
result, the per-field rule for patches and tracks, each sub-experiment's own control, the
checks of the values, the split into two groups, the share plotted per field, and a reason
shared by every group said once."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage._wording import change
from morphocoverage.objectives import RareTarget
from morphocoverage.retrospective._value_checks import alike_fields, few_values
from morphocoverage.sampling_unit import unusual_objects
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.distribution import distribution_contrast
from morphocoverage.sampling_unit.object_rules import object_rule_check
from morphocoverage.sampling_unit.repeat_plan import repeat_plan
from morphocoverage.schema import validate_input_tables
from morphocoverage.statistics import BIMODAL, bimodality_coefficient
from morphocoverage.workflows import Setup
from morphocoverage.workflows._distribution import split_words
from morphocoverage.workflows._rare_share import share_by_field, with_share
from morphocoverage.workflows._result_influence import with_result_influence
from morphocoverage.workflows.retrospective import inspect_fovs


def _tables(conditions, repeats, fields=6, cells=20, effect=None, seed=0, shift=None):
    """``conditions`` per repeat, ``fields`` fields of ``cells`` cells each; ``effect`` the
    mean of each condition, ``shift`` a multiplier of each repeat's values."""
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for repeat in repeats:
        for condition in conditions[repeat]:
            for field in range(fields):
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": f"{condition}_{repeat}_{field}",
                }
                fovs.append(key)
                mean = (effect or {}).get(condition, 10.0) * (shift or {}).get(repeat, 1.0)
                rows += [{**key, "x": v} for v in rng.normal(mean, 1.0, cells)]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _three(effect=None, seed=0):
    repeats = ["r1", "r2", "r3"]
    return _tables({r: ["ctrl", "drug"] for r in repeats}, repeats, effect=effect, seed=seed)


def test_the_plan_powers_the_paired_test_on_the_log_ratios():
    objects, fovs = _three({"drug": 12.0})
    contrast = condition_contrast(
        objects, fovs, feature="x", control={"condition": "ctrl"}, observation_unit="cell"
    )
    [row] = contrast.plan.to_dict("records")
    assert row["planned_on"] == "log ratio"
    table = contrast.table
    ratios = np.log(table["mean"] / table["control_mean"])
    assert row["change"] == pytest.approx(np.exp(ratios.mean()))
    # A 20% change, consistent across repeats: the paired test needs few repeats.
    assert 2 <= row["repeats_needed"] <= 4
    assert table["log_ratio_se"].notna().all() and table["difference_se"].notna().all()
    # A mean at or below zero: the plan uses the differences, as the paired test does.
    signed = table.assign(control_mean=-table["control_mean"])
    assert repeat_plan(signed).iloc[0]["planned_on"] == "difference"


def test_the_change_is_written_so_a_tiny_change_never_reads_as_none():
    assert change(1.25) == "×1.25"
    assert change(0.987) == "×0.987"
    assert change(1.002) == "×1.002"
    assert change(-3.21, "raw") == "-3.21"
    assert change(np.nan) == "–"


def test_a_field_is_influential_when_it_moves_the_result():
    objects, fovs = _three({"drug": 12.0})
    # One drug field three times as bright: it lifts its repeat's mean of cells.
    bright = objects["position_id"] == "drug_r1_0"
    objects.loc[bright, "x"] *= 3
    setup = Setup(readout="x", control={"condition": "ctrl"}, observation_unit="cell")
    contrast = condition_contrast(
        objects, fovs, feature="x", control=setup.control, observation_unit="cell"
    )
    inspection = inspect_fovs(objects, fovs, feature_cols=["x"], **setup.common)
    result = with_result_influence(inspection, objects, fovs, contrast, setup)
    found = result.fovs[result.fovs["reason"] == "influence"]
    assert list(found["position_id"]) == ["drug_r1_0"]
    detail = found["detail"].iloc[0]
    assert "removing it lowers drug's mean of x in this repeat" in detail
    assert "its change against the control goes from ×" in detail
    assert result.settings["influence_on"] == "result"


@pytest.mark.parametrize(("unit", "rules"), [("cell", 2), ("patch", 1), ("track", 1)])
def test_the_rule_per_object_is_a_default_only_for_objects(unit, rules):
    chosen = Setup(readout="x", observation_unit=unit).stopping_rules()
    assert len(chosen) == rules and chosen[-1].resampling_unit == "fov"


def _screen():
    """Two sets, each its own control and repeats; set B's whole repeats are twice as bright."""
    conditions = {r: ["ctrl", "drugA"] for r in ("a1", "a2", "a3")}
    conditions.update({r: ["ctrl", "drugB"] for r in ("b1", "b2", "b3")})
    shift = {"b1": 2.0, "b2": 2.0, "b3": 2.0}
    return _tables(conditions, list(conditions), fields=4, cells=15, shift=shift)


def test_unusual_cells_are_judged_against_their_own_sets_control():
    objects, fovs = _screen()
    result = unusual_objects(
        objects, fovs, feature_cols=["x"], reference={"condition": "ctrl"}, seed_checks=0
    )
    for row in result.groups.to_dict("records"):
        own = row["biological_repeat"][0]
        assert {g["biological_repeat"][0] for g in row["reference_groups"]} == {own}
    pooled = unusual_objects(
        objects,
        fovs,
        feature_cols=["x"],
        reference={"condition": "ctrl"},
        seed_checks=0,
        within_experiment=False,
    )
    assert result.settings["within_experiment"] and not pooled.settings["within_experiment"]
    sets = {g["biological_repeat"][0] for g in pooled.groups["reference_groups"].iloc[0]}
    assert sets == {"a", "b"}


def test_control_means_are_compared_within_a_sub_experiment():
    objects, fovs = _screen()
    check = object_rule_check(
        objects, fovs, feature="x", control={"condition": "ctrl"}, observation_unit="cell"
    )
    # Set B is twice as bright as set A, but each set's repeats agree: no caution.
    assert "control means of the repeats differ" not in check.summary(detail="full")


def test_fields_normalised_per_image_are_more_alike_than_chance():
    objects, fovs = _three()
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    assert alike_fields(data) is None
    per_image = objects.assign(
        x=objects["x"] / objects.groupby("position_id")["x"].transform("mean")
    )
    data = validate_input_tables(per_image, fovs=fovs, feature_cols=["x"])
    assert "Fields more alike than chance: x (in 6 of 6 groups)" in alike_fields(data)


def test_a_feature_of_few_values_is_named():
    objects, fovs = _three()
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    assert few_values(data) is None
    sparse = objects.assign(x=np.where(np.arange(len(objects)) % 10 == 0, 1e-5, 0.0))
    data = validate_input_tables(sparse, fovs=fovs, feature_cols=["x"], observation_unit="patch")
    assert "Few distinct values: x (2 values, 0 in 90% of patches)" in few_values(data)


def test_a_condition_that_splits_its_cells_is_said_to():
    rng = np.random.default_rng(3)
    assert bimodality_coefficient(rng.normal(size=400)) < BIMODAL
    assert (
        bimodality_coefficient(np.r_[rng.normal(-2.5, 1, 200), rng.normal(2.5, 1, 200)]) > BIMODAL
    )
    objects, fovs = _three()
    split = objects["condition"] == "drug"
    halves = np.where(np.arange(split.sum()) % 2 == 0, -3.0, 3.0)
    objects.loc[split, "x"] = objects.loc[split, "x"] + halves
    result = distribution_contrast(
        objects, fovs, features=["x"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    [row] = result.splits().to_dict("records")
    assert row["condition"] == "drug" and row["split_repeats"] == 3
    assert split_words(result, "drug").startswith("its cells may split into two groups on x")
    assert split_words(result, "ctrl") == ""


def test_the_labelled_share_is_plotted_per_field():
    objects, fovs = _three()
    objects["mitotic"] = np.random.default_rng(1).random(len(objects)) < 0.2
    rare = RareTarget(
        detection_probability=0.9, min_fov_occurrences=2, phenotype="mitotic", fov_incidence=0.2
    )
    setup = Setup(readout="x", rare=rare, rare_label="mitotic", observation_unit="cell")
    shared = with_share(objects, setup)
    table = share_by_field(shared, fovs, setup)
    assert len(table) == len(fovs)
    assert table["mitotic share"].between(0, 1).all() and table["mitotic share"].nunique() > 1


def test_a_reason_every_group_shares_is_said_once():
    from morphocoverage import ObjectiveConfig, SamplingObjective
    from morphocoverage.benchmark import AcquisitionOrder
    from morphocoverage.strategies import ScalarPrecisionStrategy
    from morphocoverage.sufficiency import sufficiency_report

    objects, fovs = _three()
    rules = [
        ScalarPrecisionStrategy("x", 0.05, "object", "upper_95"),
        ScalarPrecisionStrategy("x", 0.05, "fov", "upper_95"),
    ]
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["x"],
        strategies=rules,
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        orders=[AcquisitionOrder.permuted(0)],
        observation_unit="track",
    )
    full = " ".join(report.summary(detail="full").split())
    assert full.count("is undefined in every group: the object-unit mean-precision") == 1
    assert "undefined (the object-unit" not in full
    assert "(cell, nucleus, object), and these rows are of another unit (track)" in full
