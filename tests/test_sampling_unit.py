"""Tests of the sampling-unit checks: can objects be counted, or must FOVs be?

Constructed acquisitions with a known field structure: no field effect (the
FOVs behave like random objects), a strong FOV effect, rare states spread
over the fields or clustered in a few, and planted out-of-distribution objects.
The expectations are the properties of the construction, not performance.
"""

from __future__ import annotations

import math
import re
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
from morphocoverage.sampling_unit.contrast import UNREACHABLE

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
            rows.append(
                {"condition": condition, "biological_repeat": repeat, "image_file": f"f{j}", "x": v}
            )
    objects = pd.DataFrame(rows)
    fovs = pd.DataFrame(
        {
            "condition": condition,
            "biological_repeat": repeat,
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
        [
            fovs,
            pd.DataFrame(
                {"condition": ["A"], "biological_repeat": ["r1"], "image_file": ["empty"]}
            ),
        ],
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


# -- the mean-precision rule with whole FOVs -----------------------------------------------------


def test_the_rule_asks_for_the_papers_number_of_objects():
    objects, fovs = acquisition(object_sd=2.0, field_sd=0.0)
    row = object_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=200).table.iloc[0]
    sd = objects["x"].std(ddof=1) / objects["x"].mean()
    assert row["rule_objects"] == math.ceil((sd / 0.05) ** 2)


def test_with_fovs_like_random_objects_the_rule_keeps_its_promise():
    # A mean far from zero: the mean-precision rule applies to non-negative measurements.
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
        [
            fovs,
            pd.DataFrame(
                {"condition": ["A"], "biological_repeat": ["r1"], "image_file": ["empty"]}
            ),
        ],
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


def test_a_states_p_value_depends_neither_on_the_other_states_nor_on_the_row_order():
    """The states of a group shared one generator in their order, and the shuffles followed
    the row order, so reordering rows or states changed an unchanged state's p (#296)."""
    objects, fovs = acquisition(n_fields=30, per_field=10)
    draw = np.random.default_rng(1).random(len(objects))
    objects["state"] = np.where(draw < 0.04, "rare", np.where(draw < 0.1, "other", "common"))

    def p_values(table, states=None):
        result = fov_clustering(table, fovs, label_col="state", states=states)
        return dict(zip(result.table["state"], result.table["p_clustered"], strict=True))

    first = p_values(objects)
    assert 0.05 < first["rare"] < 1 and 0.05 < first["other"] < 1
    assert p_values(objects.sample(frac=1, random_state=2)) == first
    assert p_values(objects, ["other", "rare"]) == {s: first[s] for s in ("other", "rare")}
    assert p_values(objects, ["rare"]) == {"rare": first["rare"]}


def test_clustered_is_called_at_q_corrected_over_every_group_and_state():
    """#287: 20 groups with a state spread at random and one clustered in one field. A raw
    p <= 0.05 among the null groups is not called clustered; the q is Benjamini-Hochberg over
    the 21 tests of the analysis, in the table and in the answer (owner, 2026-10-05)."""
    rng = np.random.default_rng(5)
    rows, fields = [], []
    for group in range(21):
        for j in range(10):
            image = f"c{group}_f{j}"
            fields.append(
                {"condition": f"c{group}", "biological_repeat": "r1", "image_file": image}
            )
            for i in range(10):
                state = j == 0 and i < 8 if group == 20 else bool(rng.random() < 0.1)
                rows.append(
                    {"condition": f"c{group}", "biological_repeat": "r1", "image_file": image}
                    | {"state": state}
                )
    result = fov_clustering(
        pd.DataFrame(rows),
        pd.DataFrame(fields),
        label_col="state",
        states=[True],
        n_permutations=999,
    )
    table = result.table
    expected = stats.false_discovery_control(table["p_clustered"].astype(float), method="bh")
    np.testing.assert_allclose(table["q_clustered"], expected)
    null = table[table["condition"] != "c20"]
    assert (null["p_clustered"] <= 0.05).any() and (null["q_clustered"] > 0.05).all()
    assert table.loc[table["condition"] == "c20", "q_clustered"].iloc[0] <= 0.05
    text = " ".join(result.summary().split())
    assert "q ≤ 0.05 after Benjamini-Hochberg over the 21 tests" in text
    assert "in 1 of 21 groups where seen (c20 · repeat r1)" in text


@pytest.mark.parametrize(
    ("states", "message"),
    [("rare", "must be a list"), ([], "must be a list"), (["rare", "rare"], "twice")],
)
def test_fov_clustering_refuses_states_it_would_misread(states, message):
    """A bare string was read letter by letter, a state named twice gave one row with the
    state and one without, and no state gave a table the summary could not read (#288)."""
    objects, fovs = acquisition(n_fields=10, per_field=10)
    objects = _labelled(objects, ["f1"], per_field=2)
    with pytest.raises(ValueError, match=message):
        fov_clustering(objects, fovs, label_col="state", states=states)


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


def test_a_failed_seed_refit_is_counted_and_said(monkeypatch):
    """#314: a refit that failed (a feature constant in that reference sample) was dropped, so
    the refits that succeeded alone gave the agreement, or none was shown."""
    from morphocoverage.sampling_unit import unusual

    objects, fovs, _ = _with_planted()
    calls = []
    real = unusual.score

    def failing_second(*args, **kwargs):
        calls.append(1)
        if len(calls) % 3 == 2:  # the first of the two refits of every group
            return "zero SD on the visible data for ['y']"
        return real(*args, **kwargs)

    monkeypatch.setattr(unusual, "score", failing_second)
    result = unusual.unusual_objects(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, seed_checks=2
    )
    ok = result.groups[result.groups["status"] == "ok"]
    assert (ok["seed_failures"] == 1).all() and ok["seed_agreement"].notna().all()
    text = " ".join(result.summary().split())
    assert f"{len(ok)} refits with other reference samples failed in" in text
    assert "zero SD on the visible data for ['y']" in text


def test_a_group_is_never_its_own_reference():
    objects, fovs, _ = _with_planted()
    groups = unusual_objects(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).groups
    for row in groups.to_dict("records"):
        own = {"condition": row["condition"], "biological_repeat": row["biological_repeat"]}
        assert own not in row["reference_groups"]
    single = objects[(objects["condition"] != "ctrl") | (objects["biological_repeat"] == "r0")]
    single_fovs = fovs[(fovs["condition"] != "ctrl") | (fovs["biological_repeat"] == "r0")]
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


def test_unusual_labels_do_not_depend_on_row_order_when_observations_tie():
    """Identical observations in different fields (discrete features) used to swap their
    left-out reference group, and so their scores, when the rows were reordered (#282)."""
    objects, fovs, _ = _with_planted(n_planted=0)
    objects[["x", "y"]] = objects[["x", "y"]].round()

    def scored(table):
        result = unusual_objects(
            table, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, quantile=0.05
        )
        return result.distances.sort_index(), result.labels.sort_index()

    shuffled = scored(objects.sample(frac=1, random_state=7))
    for first, again in zip(scored(objects), shuffled, strict=True):
        pd.testing.assert_series_equal(first, again)


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
    groups = ["condition", "biological_repeat", "technical_repeat"]
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
        # Wells of 4 fields: below the 5 the check asks by default since audit 10 (owner,
        # 2026-10-06), so the entry is reproduced with the rule before, 2 fields.
        min_fields=2,
    ).table.set_index("technical_repeat")
    assert check.loc["J05", "fovs_used"] == pytest.approx(1.0)
    assert 0.03 < check.loc["J05", "real_error"] < 0.055
    assert not check.loc["J05", "meets_target"]
    default = object_rule_check(
        objects,
        fovs,
        feature=ratio,
        targets=[0.025],
        control={"condition": "Untreated"},
        n_draws=10,
        group_cols=groups,
    ).table
    assert default["status"].eq("fewer than 5 fields with cells").all()


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
    text = report.summary(detail="full")
    unit = text.split("1. Can these features")[0]
    marker = report.statuses["unit"].marker
    # Undeclared observations are called cells (owner, 2026-10-02).
    assert unit.startswith(f"{marker} Foundation: why count fields, and not only cells?")
    assert "How variable are the fields?" in unit
    assert "Would the mean-precision object-count rule hold" in unit
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
    objects["sample"] = objects["biological_repeat"]
    fovs["sample"] = fovs["biological_repeat"]
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
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "well": well,
                            "image_file": name,
                        }
                    )
                    offset = well_shift + rng.normal(0, field_sd)
                    for _ in range(25):
                        objects.append(
                            {
                                "condition": condition,
                                "biological_repeat": repeat,
                                "well": well,
                                "image_file": name,
                                "x": 10 + offset + rng.normal(),
                                "y": rng.normal(),
                            }
                        )
    return pd.DataFrame(objects), pd.DataFrame(fovs)


CONTRAST_GROUPS = ["condition", "biological_repeat", "well"]


def test_cohen_d_is_the_pooled_sd_effect_size():
    from morphocoverage.statistics import cohen_d

    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])
    assert cohen_d(b, a) == pytest.approx(2.0 / np.sqrt(5 / 3))
    assert np.isnan(cohen_d(np.array([1.0]), a))
    assert np.isnan(cohen_d(np.full(3, 2.0), np.full(3, 2.0)))


def test_a_shifted_condition_differs_confidently_on_objects_and_on_fields():
    objects, fovs = _two_conditions(shift=2.0, field_sd=0.3, n_wells=5)
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
    assert list(zip(table["biological_repeat"], table["condition"])) == [
        ("R1", "treated"),
        ("R2", "treated"),
    ]
    assert (table["status"] == "ok").all()
    # The five wells of a condition are pooled: 60 fields of 25 objects on each side.
    assert (table["fields"] == 60).all() and (table["observations"] == 1500).all()
    assert (table["d_objects"] > 1).all() and (table["d_fields"] > 1).all()
    # The well column in group_cols is the shuffled unit (audit 4 Q1): 5 wells a side
    # give 252 assignments; the shift puts the observed split at the floor, 2/252.
    assert (table["permutation_unit"] == "well").all()
    assert table["p_value"].tolist() == pytest.approx([2 / 252] * 2)
    assert table["p_bonferroni"].tolist() == pytest.approx([4 / 252] * 2)
    effects = result.effects
    # 10,000 objects exceed what a side holds and are left out; 50 and 200 are kept.
    assert sorted(effects["n_objects"].unique()) == [50, 200]
    assert set(effects["sampling"]) == {"random objects", "whole FOVs"}
    assert (effects["same_sign"] == 1.0).all()
    whole = effects[effects["sampling"] == "whole FOVs"]
    assert (whole["fields_treated"] >= 2).all() and whole["fields_treated"].max() <= 9
    text = " ".join(result.summary().split())
    assert "treated vs control, repeat R1" in text
    assert "d on cells d on field means" in text
    assert "Randomization tests over whole technical repeats (well)" in text
    full = " ".join(result.summary(detail="full").split())
    # Across repeats the JCS draws are in the full form only.
    assert "JCS effect size" not in text and "JCS effect size" in full
    assert "fields of technical repeats (well) drawn with" in full
    assert (
        "treated vs control, repeat R1: randomization test over whole technical repeats (well"
        in full
    )
    assert re.search(r"whole fields fields resampled", full)


def test_a_field_effect_alone_does_not_make_the_conditions_differ_by_fields():
    # No condition difference, strong field-to-field variation: the field-unit
    # randomization test does not call the conditions different.
    objects, fovs = _two_conditions(shift=0.0, field_sd=1.0, seed=5, n_wells=5)
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
    fovs_missing = fovs[~((fovs["biological_repeat"] == "R2") & (fovs["condition"] == "control"))]
    objects_missing = objects[objects["image_file"].isin(fovs_missing["image_file"])]
    partial = condition_contrast(objects_missing, fovs_missing, **common)
    r2 = partial.table[partial.table["biological_repeat"] == "R2"].iloc[0]
    assert r2["status"] == "no control group in biological_repeat=R2" and np.isnan(r2["p_value"])


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
    text = report.summary(detail="full")
    assert "9. Do the conditions differ, and how confidently?" in text
    assert text.index("8. Could I have sampled") < text.index("9. Do the conditions differ")
    assert "user's inference" not in " ".join(text.split())
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
    # Two wells a side: the exact test's smallest p is 1/3 (the observed split and its mirror
    # are always as extreme), so it cannot help and is not run; neither is a d from two
    # well means, which can be very large by chance.
    assert table["p_floor"].tolist() == pytest.approx([1 / 3] * 2)
    assert table["p_value"].isna().all() and (table["status"] == UNREACHABLE).all()
    assert table["d_technical"].isna().all()
    text = " ".join(result.summary(detail="full").split())
    assert "with the technical repeat (technical_repeat) as the unit" in text
    assert "tech. repeats d on cells d on field means" in text and " 2 vs 2 " in text
    assert "⚠️" not in result.summary()
    assert "too few technical repeats on a side for the randomization test" in text
    # Without the column among the groups, the technical repeats stay the unit (audit 3, H-1).
    pooled = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=["condition", "biological_repeat"],
        n_permutations=100,
        n_draws=5,
        object_counts=(50,),
    )
    assert (pooled.table["permutation_unit"] == "technical_repeat").all()
    assert pooled.table["status"].tolist() == table["status"].tolist()


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
    text = result.summary(detail="full")
    assert "Which features separate the conditions? Each of the 2 declared features" in text
    assert re.search(r"^ +1 +x +\d+\.\d\d .* 2 of 2$", text, re.MULTILINE)
    assert "No contrast passes the gate for 1 feature: y." in text
    assert "Plotted by default: up to 5 best-ranked features (the readout when none ranks)" in text
    short = result.summary()
    assert "1 of 2 features pass the gate" in short and "Plotted by default" in short
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
    text = " ".join(result.summary().split())
    # Two wells a side: the observed split and its mirror, 2 of 6 assignments, a floor of 1/3
    # above the gate, so the test cannot help and is not run (as in condition_contrast).
    assert result.tests["p_value"].isna().all()
    assert (result.tests["status"] == UNREACHABLE).all()
    assert "None is ranked" in text and "so it is not run" in text
    with pytest.raises(ValueError, match="alpha"):
        feature_ranking(objects, fovs, features=["x"], control={"condition": "control"}, alpha=1)
    with pytest.raises(ValueError, match="at least one"):
        feature_ranking(objects, fovs, features=[], control={"condition": "control"})
    only = fovs[fovs["condition"] == "control"]
    controls = objects[objects["condition"] == "control"]
    alone = feature_ranking(controls, only, features=["x"], control={"condition": "control"})
    assert alone.tests.empty and "no other condition to compare" in alone.summary()


@pytest.mark.parametrize("independent", [False, True])
def test_feature_ranking_says_the_fields_are_independent_only_when_declared(independent):
    """#306: the wording was reversed, printing the declaration when it was not made."""
    from morphocoverage.sampling_unit import feature_ranking

    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3)
    result = feature_ranking(
        objects,
        fovs,
        features=["x"],
        control={"condition": "control"},
        technical_col=None,
        fields_independent=independent,
        n_permutations=100,
    )
    for detail in ("short", "full"):
        text = " ".join(result.summary(detail=detail).split())
        assert ("declared independent draws" in text) is independent
