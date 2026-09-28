"""Tests of the sampling-unit checks: can objects be counted, or must FOVs be?

Constructed acquisitions with a known field structure: no field effect (the
FOVs behave like random objects), a strong FOV effect, rare states spread
over the fields or clustered in a few, and planted out-of-distribution objects.
The expectations are the properties of the construction, not performance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from morphocoverage.sampling_unit import (
    condition_contrast,
    descriptor_spread_check,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"


def acquisition(
    *,
    n_fields: int = 40,
    per_field: int = 50,
    field_sd: float = 0.0,
    object_sd: float = 1.0,
    mean: float = 10.0,
    condition: str = "A",
    repeat: str = "r1",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One group: FOV means drawn with ``field_sd``, objects around them with ``object_sd``."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, field_sd, n_fields)
    rows = []
    for j in range(n_fields):
        values = mean + offsets[j] + rng.normal(0, object_sd, per_field)
        for v in values:
            rows.append({"condition": condition, "repeat": repeat, "image_file": f"f{j}", "x": v})
    objects = pd.DataFrame(rows)
    fovs = pd.DataFrame(
        {
            "condition": condition,
            "repeat": repeat,
            "image_file": [f"f{j}" for j in range(n_fields)],
        }
    )
    return objects, fovs


def combined(*parts):
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


# -- field variability -----------------------------------------------------------------


def test_fovs_like_random_objects_have_a_design_effect_near_one():
    objects, fovs = acquisition(field_sd=0.0)
    row = fov_variability(objects, fovs, feature="x").groups.iloc[0]
    assert row["status"] == "ok"
    assert 0.6 < row["design_effect"] < 1.6
    assert row["unusual_fov_share"] <= 0.15


def test_a_strong_field_effect_gives_the_design_effect_of_its_construction():
    # v_fov = v_obs = 1 and 50 objects per FOV: 1 + 49 * 0.5 = 25.5
    objects, fovs = acquisition(field_sd=1.0, n_fields=60, seed=3)
    row = fov_variability(objects, fovs, feature="x").groups.iloc[0]
    assert 0.3 < row["icc"] < 0.7
    assert 12 < row["design_effect"] < 40
    assert row["error_ratio"] == pytest.approx(math.sqrt(row["design_effect"]))
    assert row["unusual_fov_share"] > 0.5
    assert row["anova_p"] < 1e-6


def test_zero_observation_fields_count_as_acquired_and_missing_values_are_left_out():
    objects, fovs = acquisition(n_fields=10, per_field=20)
    fovs = pd.concat(
        [fovs, pd.DataFrame({"condition": ["A"], "repeat": ["r1"], "image_file": ["empty"]})],
        ignore_index=True,
    )
    objects.loc[0, "x"] = np.nan
    result = fov_variability(objects, fovs, feature="x")
    row = result.groups.iloc[0]
    assert row["fovs"] == 11
    assert row["fovs_with_observations"] == 10
    assert row["observations"] == 199
    assert row["missing_values"] == 1
    assert row["observations_per_fov"] == pytest.approx(199 / 11)
    empty = result.fovs[result.fovs["image_file"] == "empty"].iloc[0]
    assert empty["observations"] == 0 and pd.isna(empty["mean"])


def test_groups_are_assessed_independently():
    a = acquisition(field_sd=0.5, condition="A", seed=1)
    b = acquisition(field_sd=2.0, condition="B", seed=2)
    alone = fov_variability(*a, feature="x").groups.iloc[0]
    both = fov_variability(*combined(a, b), feature="x").groups
    together = both[both["condition"] == "A"].iloc[0]
    assert together["design_effect"] == pytest.approx(alone["design_effect"])
    assert len(both) == 2


def test_too_few_fields_is_reported_not_guessed():
    objects, fovs = acquisition(n_fields=1)
    row = fov_variability(objects, fovs, feature="x").groups.iloc[0]
    assert row["status"] == "fewer than 2 fields with observations"
    assert pd.isna(row["design_effect"])


# -- the JCB rule with whole FOVs -----------------------------------------------------


def test_the_rule_asks_for_the_papers_number_of_objects():
    objects, fovs = acquisition(object_sd=2.0, field_sd=0.0)
    row = object_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=200).table.iloc[0]
    sd = objects["x"].std(ddof=1) / objects["x"].mean()
    assert row["rule_objects"] == math.ceil((sd / 0.05) ** 2)


def test_with_fovs_like_random_objects_the_rule_keeps_its_promise():
    # A mean far from zero: the JCB rule applies to non-negative measurements.
    objects, fovs = acquisition(field_sd=0.0, per_field=10, n_fields=80, object_sd=3.0, mean=30.0)
    row = object_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=3000)
    row = row.table.iloc[0]
    assert 0.8 < row["error_ratio"] < 1.25


def test_with_a_field_effect_the_real_error_exceeds_the_promise():
    objects, fovs = acquisition(field_sd=1.0, object_sd=2.0, n_fields=40, per_field=50)
    row = object_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=2000)
    row = row.table.iloc[0]
    assert row["error_ratio"] > 2
    assert not row["meets_target"]
    assert row["fov_rule_fovs"] > 1


def test_control_groups_set_the_reference_mean():
    a = acquisition(mean=10.0, condition="A", seed=1)
    b = acquisition(mean=20.0, condition="B", seed=2)
    objects, fovs = combined(a, b)
    table = object_rule_check(
        objects, fovs, feature="x", targets=[0.05], control={"condition": "A"}, n_draws=100
    ).table.set_index("condition")
    reference = a[0]["x"].mean()
    for condition, part in (("A", a), ("B", b)):
        sd = part[0]["x"].std(ddof=1) / reference
        assert table.loc[condition, "object_sd"] == pytest.approx(sd)
    with pytest.raises(ValueError, match="no analysis group matches"):
        object_rule_check(objects, fovs, feature="x", control={"condition": "C"})


def test_the_control_mean_pools_every_control_object():
    # control groups of 10 x 5 and 10 x 45 objects at means 10 and 20: the pooled
    # control mean is 19, not the mean of the two group means (15)
    small = acquisition(mean=10.0, n_fields=10, per_field=5, condition="A", repeat="r1", seed=1)
    large = acquisition(mean=20.0, n_fields=10, per_field=45, condition="A", repeat="r2", seed=2)
    objects, fovs = combined(small, large)
    # repeat_col=None pools the control objects of every repeat (by default each repeat
    # is judged against its own control, audit 4 D3).
    check = object_rule_check(
        objects,
        fovs,
        feature="x",
        targets=[0.05],
        control={"condition": "A"},
        n_draws=50,
        repeat_col=None,
    )
    reference = objects["x"].mean()
    assert reference == pytest.approx(19.0, abs=0.1)
    for (part, _), row in zip([small, large], check.table.to_dict("records"), strict=True):
        assert row["object_sd"] == pytest.approx(part["x"].std(ddof=1) / reference)


def test_the_rule_check_is_deterministic_and_group_local():
    a = acquisition(field_sd=0.5, condition="A", seed=1)
    b = acquisition(field_sd=0.5, condition="B", seed=2)
    first = object_rule_check(*combined(a, b), feature="x", n_draws=300).table
    again = object_rule_check(*combined(a, b), feature="x", n_draws=300).table
    pd.testing.assert_frame_equal(first, again)
    alone = object_rule_check(*a, feature="x", n_draws=300).table
    np.testing.assert_allclose(
        first[first["condition"] == "A"]["real_error"].to_numpy(), alone["real_error"].to_numpy()
    )


# -- the JCS descriptor spread with whole FOVs ----------------------------------------


def test_descriptor_spread_is_alike_when_fovs_behave_like_random_objects():
    objects, fovs = acquisition(field_sd=0.0, n_fields=40, per_field=30)
    result = descriptor_spread_check(
        objects, fovs, feature="x", threshold=100, sizes=[50, 100, 200], n_draws=200
    )
    mean = result.at_threshold.set_index("descriptor").loc["mean"]
    assert 0.6 < mean["ratio"] < 1.6


def test_descriptor_spread_is_wider_with_a_field_effect():
    objects, fovs = acquisition(field_sd=1.0, n_fields=40, per_field=50)
    result = descriptor_spread_check(
        objects, fovs, feature="x", threshold=100, sizes=[50, 150], n_draws=200
    )
    assert 100 in result.settings["sizes"]
    mean = result.at_threshold.set_index("descriptor").loc["mean"]
    assert mean["ratio"] > 1.5


# -- rare states: spread or clustered by field ------------------------------------------


def _labelled(objects: pd.DataFrame, fields_with_state: list[str], per_field: int) -> pd.DataFrame:
    labels = np.full(len(objects), "common", dtype=object)
    for name in fields_with_state:
        where = np.flatnonzero(objects["image_file"].to_numpy() == name)[:per_field]
        labels[where] = "rare"
    return objects.assign(state=labels)


def test_a_state_spread_over_the_fields_is_not_called_clustered():
    objects, fovs = acquisition(n_fields=40, per_field=50)
    objects = _labelled(objects, [f"f{j}" for j in range(0, 40, 2)], per_field=1)
    row = fov_clustering(objects, fovs, label_col="state", states=["rare"]).table.iloc[0]
    assert row["state_objects"] == 20 and row["fovs_with_state"] == 20
    assert row["clustering_ratio"] < 1  # one per FOV: more spread than random objects
    assert row["p_clustered"] > 0.5


def test_a_state_clustered_in_two_fields_is_called_clustered():
    objects, fovs = acquisition(n_fields=40, per_field=50)
    objects = _labelled(objects, ["f3", "f17"], per_field=10)
    row = fov_clustering(objects, fovs, label_col="state", states=["rare"]).table.iloc[0]
    assert row["fovs_with_state"] == 2
    assert row["expected_fovs_if_random"] > 10
    assert row["p_clustered"] <= 0.01
    assert row["largest_fov_share"] == pytest.approx(0.5)
    assert row["fovs_needed"] > row["fovs_needed_if_random"]


def test_expected_fields_are_the_hypergeometric_expectation():
    objects, fovs = acquisition(n_fields=4, per_field=5)
    objects = _labelled(objects, ["f0"], per_field=3)
    row = fov_clustering(objects, fovs, label_col="state", states=["rare"]).table.iloc[0]
    expected = 4 * (1 - stats.hypergeom(20, 3, 5).pmf(0))
    assert row["expected_fovs_if_random"] == pytest.approx(expected)
    # 15% prevalence at 5 objects per FOV: 19 objects, so 4 FOVs, if objects were random
    assert row["fovs_needed_if_random"] == math.ceil(
        math.ceil(math.log(0.05) / math.log(1 - 3 / 20)) / 5
    )


def test_unlabelled_objects_are_left_out_and_empty_fovs_count():
    objects, fovs = acquisition(n_fields=10, per_field=10)
    objects = _labelled(objects, ["f0"], per_field=2)
    objects.loc[objects["image_file"] == "f9", "state"] = None
    fovs = pd.concat(
        [fovs, pd.DataFrame({"condition": ["A"], "repeat": ["r1"], "image_file": ["empty"]})],
        ignore_index=True,
    )
    table = fov_clustering(objects, fovs, label_col="state").table.set_index("state")
    assert set(table.index) == {"common", "rare"}
    assert table.loc["rare", "labelled"] == 90
    # f9 was not labelled at all, so it was not looked at; the empty FOV was.
    assert table.loc["rare", "fovs_acquired"] == 11
    assert table.loc["rare", "fovs_unlabelled"] == 1
    assert table.loc["rare", "fovs"] == 10
    assert table.loc["rare", "fov_incidence"] == pytest.approx(1 / 10)


def test_fov_clustering_is_deterministic():
    objects, fovs = acquisition(n_fields=20, per_field=20)
    objects = _labelled(objects, ["f1", "f2", "f3"], per_field=4)
    first = fov_clustering(objects, fovs, label_col="state", random_state=3).table
    again = fov_clustering(objects, fovs, label_col="state", random_state=3).table
    pd.testing.assert_frame_equal(first, again)


# -- unusual (out-of-distribution) objects ------------------------------------------------


def _with_planted(n_planted: int = 10):
    parts = [
        acquisition(condition="ctrl", repeat=f"r{r}", n_fields=10, per_field=30, seed=r)
        for r in range(3)
    ]
    treated = acquisition(condition="treated", n_fields=10, per_field=30, seed=9)
    objects, fovs = combined(*parts, treated)
    objects["y"] = np.random.default_rng(5).normal(0, 1, len(objects))
    planted = np.flatnonzero(objects["condition"].to_numpy() == "treated")[:n_planted]
    objects.loc[planted, ["x", "y"]] += 12.0
    objects.index = objects.index + 1000  # labels must follow the caller's index
    return objects, fovs, objects.index[planted]


def test_planted_far_objects_are_unusual_and_typical_objects_mostly_not():
    objects, fovs, planted = _with_planted()
    result = unusual_objects(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, seed_checks=1
    )
    assert result.labels.loc[planted].all()
    typical = result.labels.drop(planted)
    assert typical.mean() < 0.05
    treated = result.groups.set_index("condition").loc["treated"]
    assert treated["status"] == "ok" and treated["unusual"] >= 10


def test_a_group_is_never_its_own_reference():
    objects, fovs, _ = _with_planted()
    groups = unusual_objects(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).groups
    for row in groups.to_dict("records"):
        own = {"condition": row["condition"], "repeat": row["repeat"]}
        assert own not in row["reference_groups"]
    single = objects[(objects["condition"] != "ctrl") | (objects["repeat"] == "r0")]
    single_fovs = fovs[(fovs["condition"] != "ctrl") | (fovs["repeat"] == "r0")]
    rows = unusual_objects(
        single, single_fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).groups.set_index("condition")
    assert rows.loc["ctrl", "status"] == "no other reference group to score against"
    assert rows.loc["treated", "status"] == "ok"


def test_unusual_objects_use_only_the_explicit_features_and_are_deterministic():
    objects, fovs, _ = _with_planted()
    first = unusual_objects(objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"})
    noisy = objects.assign(stage_x=np.arange(len(objects)) * 1e6)
    again = unusual_objects(noisy, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"})
    pd.testing.assert_series_equal(first.labels, again.labels)
    pd.testing.assert_series_equal(first.distances, again.distances)


def test_unusual_labels_feed_fov_clustering():
    objects, fovs, _ = _with_planted()
    labels = unusual_objects(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).labels
    table = fov_clustering(
        objects.assign(unusual=labels), fovs, label_col="unusual", states=[True]
    ).table.set_index("condition")
    treated = table.loc["treated"]
    # the ten planted objects sit in the first FOV of the treated well
    assert treated["fovs_with_state"] <= 3
    assert treated["p_clustered"] <= 0.05


# -- real data: the JCS case study (idr0139) --------------------------------------------


@pytest.fixture(scope="module")
def idr0139():
    sys.path.insert(0, str(DATA))
    try:
        from datasets import load
    finally:
        sys.path.remove(str(DATA))
    dataset = load("idr0139")
    return dataset.objects, dataset.fovs, dataset.readout


def test_idr0139_fovs_are_not_random_objects(idr0139):
    """Regression of docs/preliminary_findings.md entry 9 on the committed wells."""
    objects, fovs, ratio = idr0139
    groups = ["condition", "repeat", "technical_repeat"]
    variability = fov_variability(objects, fovs, feature=ratio, group_cols=groups).groups.set_index(
        "technical_repeat"
    )
    assert variability.loc["J05", "design_effect"] > 5
    check = object_rule_check(
        objects,
        fovs,
        feature=ratio,
        targets=[0.025],
        control={"condition": "Untreated"},
        n_draws=1000,
        group_cols=groups,
    ).table.set_index("technical_repeat")
    assert check.loc["J05", "fovs_used"] == pytest.approx(1.0)
    assert 0.03 < check.loc["J05", "real_error"] < 0.055
    assert not check.loc["J05", "meets_target"]


# -- the report places the checks ---------------------------------------------------------


def test_the_report_opens_with_the_sampling_unit_and_profiles_clustering():
    from morphocoverage.retrospective import microscopist_report

    objects, fovs = acquisition(field_sd=1.0, n_fields=20, per_field=30)
    objects = _labelled(objects, ["f3", "f7"], per_field=8)
    report = microscopist_report(
        variability=fov_variability(objects, fovs, feature="x"),
        object_rules=object_rule_check(objects, fovs, feature="x", n_draws=200),
        clustering=fov_clustering(objects, fovs, label_col="state", states=["rare"]),
    )
    text = report.summary()
    unit = text.split("1. Can these features")[0]
    assert unit.startswith("Foundation: why count fields, and not only objects?")
    assert "How variable are the fields?" in unit
    assert "Would the JCB object-count rule hold" in unit
    items = set(report.profile["item"])
    assert {"fields against random objects", "rare-state clustering"} <= items
    assert report.inputs["unit"] == "given"
    assert "Not assessed" in microscopist_report().sections["unit"]


def test_the_report_states_the_design_effect_once():
    from morphocoverage.retrospective import (
        SamplingHierarchy,
        effort_projection,
        microscopist_report,
    )

    parts = [
        acquisition(field_sd=1.0, n_fields=10, condition="A", repeat=f"r{r}", seed=r)
        for r in range(3)
    ]
    objects, fovs = combined(*parts)
    objects["sample"] = objects["repeat"]
    fovs["sample"] = fovs["repeat"]
    effort = effort_projection(
        objects,
        fovs,
        feature="x",
        hierarchy=SamplingHierarchy("image_file", "sample"),
        strata=[],
        n_boot=5,
    )
    variability = fov_variability(objects, fovs, feature="x")
    alone = microscopist_report(effort=effort).profile
    both = microscopist_report(effort=effort, variability=variability).profile
    assert "field-to-field heterogeneity" in set(alone["item"])
    assert "field-to-field heterogeneity" not in set(both["item"])
    assert "fields against random objects" in set(both["item"])


# -- condition contrast ------------------------------------------------------------------


def _two_conditions(
    shift: float, field_sd: float, seed: int = 3, n_fields: int = 12, n_wells: int = 2
):
    """Control and treated wells in two repeats; ``shift`` moves the treated condition's ``x``.

    ``y`` never differs between the conditions.
    """
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for repeat in ("R1", "R2"):
        for condition, well_shift in (("control", 0.0), ("treated", shift)):
            for well in [f"w{i + 1}" for i in range(n_wells)]:
                for j in range(n_fields):
                    name = f"{repeat}_{condition}_{well}_f{j}"
                    fovs.append(
                        {"condition": condition, "repeat": repeat, "well": well, "image_file": name}
                    )
                    offset = well_shift + rng.normal(0, field_sd)
                    for _ in range(25):
                        objects.append(
                            {
                                "condition": condition,
                                "repeat": repeat,
                                "well": well,
                                "image_file": name,
                                "x": 10 + offset + rng.normal(),
                                "y": rng.normal(),
                            }
                        )
    return pd.DataFrame(objects), pd.DataFrame(fovs)


CONTRAST_GROUPS = ["condition", "repeat", "well"]


def test_cohen_d_is_the_pooled_sd_effect_size():
    from morphocoverage.statistics import cohen_d

    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])
    assert cohen_d(b, a) == pytest.approx(2.0 / np.sqrt(5 / 3))
    assert np.isnan(cohen_d(np.array([1.0]), a))
    assert np.isnan(cohen_d(np.full(3, 2.0), np.full(3, 2.0)))


def test_a_shifted_condition_differs_confidently_on_objects_and_on_fields():
    objects, fovs = _two_conditions(shift=2.0, field_sd=0.3)
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=CONTRAST_GROUPS,
        n_permutations=500,
        n_draws=50,
        object_counts=(50, 200, 10_000),
    )
    table = result.table
    assert list(zip(table["repeat"], table["condition"])) == [("R1", "treated"), ("R2", "treated")]
    assert (table["status"] == "ok").all()
    # Both wells of a condition are pooled: 24 fields of 25 objects on each side.
    assert (table["fields"] == 24).all() and (table["observations"] == 600).all()
    assert (table["d_objects"] > 1).all() and (table["d_fields"] > 1).all()
    # The well column in group_cols is the shuffled unit (audit 4 Q1): 2 wells a side
    # give 6 assignments, so p cannot fall below 1/3 however large the shift.
    assert (table["permutation_unit"] == "well").all()
    assert table["p_value"].tolist() == pytest.approx([1 / 3] * 2)
    assert table["p_bonferroni"].tolist() == pytest.approx([2 / 3] * 2)
    effects = result.effects
    # 10,000 objects exceed what a side holds and are left out; 50 and 200 are kept.
    assert sorted(effects["n_objects"].unique()) == [50, 200]
    assert set(effects["sampling"]) == {"random objects", "whole FOVs"}
    assert (effects["same_sign"] == 1.0).all()
    whole = effects[effects["sampling"] == "whole FOVs"]
    assert (whole["fields_treated"] >= 2).all() and whole["fields_treated"].max() <= 9
    text = result.summary()
    assert "repeat=R1, treated vs control" in text
    assert "Cohen's d on objects" in text and "on field means" in text
    assert "Randomization test over whole technical repeats (well" in text
    assert "JCS effect size" in text and "whole FOVs of resampled technical repeats" in text


def test_a_field_effect_alone_does_not_make_the_conditions_differ_by_fields():
    # No condition difference, strong field-to-field variation: the field-unit
    # randomization test does not call the conditions different.
    objects, fovs = _two_conditions(shift=0.0, field_sd=1.0, seed=5)
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=CONTRAST_GROUPS,
        n_permutations=500,
        n_draws=20,
        object_counts=(100,),
    )
    assert (result.table["p_value"] > 0.05).all()


def test_condition_contrast_is_deterministic_and_needs_a_control_and_other_conditions():
    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3)
    common = {
        "feature": "x",
        "control": {"condition": "control"},
        "group_cols": CONTRAST_GROUPS,
        "n_permutations": 100,
        "n_draws": 10,
        "object_counts": (50,),
    }
    first, second = (condition_contrast(objects, fovs, **common) for _ in range(2))
    pd.testing.assert_frame_equal(first.table, second.table)
    pd.testing.assert_frame_equal(first.effects, second.effects)
    with pytest.raises(ValueError, match="no analysis group matches"):
        condition_contrast(objects, fovs, **{**common, "control": {"condition": "none"}})
    with pytest.raises(ValueError, match="condition column"):
        condition_contrast(objects, fovs, **{**common, "control": {"well": "w1"}})
    only_control = fovs[fovs["condition"] == "control"]
    alone = condition_contrast(objects[objects["condition"] == "control"], only_control, **common)
    assert alone.table.empty and "no other condition to compare" in alone.summary()
    # A repeat without a control group is reported, not compared.
    fovs_missing = fovs[~((fovs["repeat"] == "R2") & (fovs["condition"] == "control"))]
    objects_missing = objects[objects["image_file"].isin(fovs_missing["image_file"])]
    partial = condition_contrast(objects_missing, fovs_missing, **common)
    r2 = partial.table[partial.table["repeat"] == "R2"].iloc[0]
    assert r2["status"] == "no control group in repeat=R2" and np.isnan(r2["p_value"])


def test_the_report_places_the_contrast_last():
    from morphocoverage.retrospective import microscopist_report

    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3)
    contrast = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=CONTRAST_GROUPS,
        n_permutations=50,
        n_draws=5,
        object_counts=(50,),
    )
    report = microscopist_report(contrast=contrast)
    text = report.summary()
    assert "9. Do the conditions differ, and how confidently?" in text
    assert text.index("8. Could I have sampled") < text.index("9. Do the conditions differ")
    assert "generalizing it needs independent biological repeats" in text
    assert "Not assessed" in microscopist_report().sections["conditions"]


def _technical(objects, fovs):
    """The same tables with the well named as the MorphoBricks technical repeat."""
    rename = {"well": "technical_repeat"}
    return objects.rename(columns=rename), fovs.rename(columns=rename)


def test_technical_repeats_are_the_unit_when_each_side_holds_two():
    objects, fovs = _technical(*_two_conditions(shift=2.0, field_sd=0.3))
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        n_permutations=300,
        n_draws=10,
        object_counts=(50,),
    )
    table = result.table
    assert (table["permutation_unit"] == "technical_repeat").all()
    assert (table["technical"] == 2).all() and (table["control_technical"] == 2).all()
    assert (table["distinct_assignments"] == 6).all()
    assert np.isfinite(table["d_technical"]).all() and (table["d_technical"] > 1).all()
    # Two wells a side: the observed split and its mirror are always as extreme.
    assert (table["p_value"] > 0.2).all()
    text = result.summary()
    assert "with the technical repeat (technical_repeat) as the unit" in text
    assert "in 2 and 2 technical repeats" in text and "on technical-repeat means" in text
    assert "whole technical repeats (technical_repeat; only 6 distinct assignments exist)" in text
    # Without the column among the groups, the technical repeats stay the unit (audit 3, H-1).
    pooled = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=["condition", "repeat"],
        n_permutations=100,
        n_draws=5,
        object_counts=(50,),
    )
    assert (pooled.table["permutation_unit"] == "technical_repeat").all()
    assert pooled.table["p_value"].tolist() == table["p_value"].tolist()


def test_feature_ranking_gates_by_q_value_and_ranks_by_effect_size():
    from morphocoverage.sampling_unit import feature_ranking

    # Five wells a side: 252 assignments, a floor of 2/252 (four a side give 2/70,
    # which two tests of the readout cannot pass after the correction over four).
    objects, fovs = _technical(
        *_two_conditions(shift=2.0, field_sd=0.3, seed=7, n_wells=5, n_fields=6)
    )
    result = feature_ranking(
        objects,
        fovs,
        features=["y", "x"],
        control={"condition": "control"},
        n_permutations=400,
    )
    tests = result.tests
    assert len(tests) == 4  # two features, two repeats
    assert (tests["permutation_unit"] == "technical_repeat").all()
    assert (tests["distinct_assignments"] == 252).all() and tests["p_exact"].all()
    x_rows, y_rows = tests[tests["feature"] == "x"], tests[tests["feature"] == "y"]
    assert x_rows["passes_gate"].all() and not y_rows["passes_gate"].any()
    assert (tests["q_value"] >= tests["p_value"]).all()
    ranking = result.ranking
    assert ranking["feature"].tolist() == ["x", "y"]
    assert ranking.loc[0, "score"] > 1 and np.isnan(ranking.loc[1, "score"])
    assert result.top() == ["x"] and result.top(0) == []
    text = result.summary()
    assert "Which features separate the conditions? Each of the 2 declared features" in text
    assert "1. x: |d|" in text and "passes in 2 of 2 contrasts" in text
    assert "No contrast passes the gate for 1 feature: y." in text
    assert "Plotted by default: the readout and up to 5 best-ranked other features" in text
    # Deterministic, and the same numbers as the contrast for the readout.
    again = feature_ranking(
        objects, fovs, features=["y", "x"], control={"condition": "control"}, n_permutations=400
    )
    pd.testing.assert_frame_equal(result.tests, again.tests)
    contrast = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        n_permutations=400,
        n_draws=5,
        object_counts=(50,),
    )
    assert x_rows["d_fields"].tolist() == pytest.approx(contrast.table["d_fields"].tolist())


def test_feature_ranking_says_when_few_technical_repeats_cannot_pass_the_gate():
    from morphocoverage.sampling_unit import feature_ranking

    objects, fovs = _technical(*_two_conditions(shift=3.0, field_sd=0.3))
    result = feature_ranking(
        objects, fovs, features=["x"], control={"condition": "control"}, n_permutations=100
    )
    assert not result.tests["passes_gate"].any() and result.top() == []
    text = result.summary()
    assert "No contrast passes the gate for 1 feature: x." in text
    # Two wells a side: the observed split and its mirror, 2 of 6 assignments.
    assert "with as few as 6 distinct assignments a p-value cannot fall below 0.33" in text
    with pytest.raises(ValueError, match="alpha"):
        feature_ranking(objects, fovs, features=["x"], control={"condition": "control"}, alpha=1)
    with pytest.raises(ValueError, match="at least one"):
        feature_ranking(objects, fovs, features=[], control={"condition": "control"})
    only = fovs[fovs["condition"] == "control"]
    controls = objects[objects["condition"] == "control"]
    alone = feature_ranking(controls, only, features=["x"], control={"condition": "control"})
    assert alone.tests.empty and "no other condition to compare" in alone.summary()
