"""Tests of the sampling-unit checks: cells or fields (docs/questions.md, section 4).

Constructed acquisitions with a known field structure: no field effect (the
fields behave like random cells), a strong field effect, rare states spread
over the fields or clustered in a few, and planted out-of-distribution cells.
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
    cell_rule_check,
    descriptor_spread_check,
    field_clustering,
    field_variability,
    unusual_cells,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"


def acquisition(
    *,
    n_fields: int = 40,
    per_field: int = 50,
    field_sd: float = 0.0,
    cell_sd: float = 1.0,
    mean: float = 10.0,
    condition: str = "A",
    repeat: str = "r1",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One group: field means drawn with ``field_sd``, cells around them with ``cell_sd``."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, field_sd, n_fields)
    rows = []
    for j in range(n_fields):
        values = mean + offsets[j] + rng.normal(0, cell_sd, per_field)
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


def test_fields_like_random_cells_have_a_design_effect_near_one():
    objects, fovs = acquisition(field_sd=0.0)
    row = field_variability(objects, fovs, feature="x").groups.iloc[0]
    assert row["status"] == "ok"
    assert 0.6 < row["design_effect"] < 1.6
    assert row["unusual_field_share"] <= 0.15


def test_a_strong_field_effect_gives_the_design_effect_of_its_construction():
    # v_fov = v_obs = 1 and 50 cells per field: 1 + 49 * 0.5 = 25.5
    objects, fovs = acquisition(field_sd=1.0, n_fields=60, seed=3)
    row = field_variability(objects, fovs, feature="x").groups.iloc[0]
    assert 0.3 < row["icc"] < 0.7
    assert 12 < row["design_effect"] < 40
    assert row["error_ratio"] == pytest.approx(math.sqrt(row["design_effect"]))
    assert row["unusual_field_share"] > 0.5
    assert row["anova_p"] < 1e-6


def test_zero_observation_fields_count_as_acquired_and_missing_values_are_left_out():
    objects, fovs = acquisition(n_fields=10, per_field=20)
    fovs = pd.concat(
        [fovs, pd.DataFrame({"condition": ["A"], "repeat": ["r1"], "image_file": ["empty"]})],
        ignore_index=True,
    )
    objects.loc[0, "x"] = np.nan
    result = field_variability(objects, fovs, feature="x")
    row = result.groups.iloc[0]
    assert row["fovs"] == 11
    assert row["fovs_with_observations"] == 10
    assert row["observations"] == 199
    assert row["missing_values"] == 1
    assert row["observations_per_fov"] == pytest.approx(199 / 11)
    empty = result.fields[result.fields["image_file"] == "empty"].iloc[0]
    assert empty["observations"] == 0 and pd.isna(empty["mean"])


def test_groups_are_assessed_independently():
    a = acquisition(field_sd=0.5, condition="A", seed=1)
    b = acquisition(field_sd=2.0, condition="B", seed=2)
    alone = field_variability(*a, feature="x").groups.iloc[0]
    both = field_variability(*combined(a, b), feature="x").groups
    together = both[both["condition"] == "A"].iloc[0]
    assert together["design_effect"] == pytest.approx(alone["design_effect"])
    assert len(both) == 2


def test_too_few_fields_is_reported_not_guessed():
    objects, fovs = acquisition(n_fields=1)
    row = field_variability(objects, fovs, feature="x").groups.iloc[0]
    assert row["status"] == "fewer than 2 fields with observations"
    assert pd.isna(row["design_effect"])


# -- the JCB rule with whole fields -----------------------------------------------------


def test_the_rule_asks_for_the_papers_number_of_cells():
    objects, fovs = acquisition(cell_sd=2.0, field_sd=0.0)
    row = cell_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=200).table.iloc[0]
    sd = objects["x"].std(ddof=1) / objects["x"].mean()
    assert row["rule_cells"] == math.ceil((sd / 0.05) ** 2)


def test_with_fields_like_random_cells_the_rule_keeps_its_promise():
    objects, fovs = acquisition(field_sd=0.0, per_field=10, n_fields=80, cell_sd=3.0)
    row = cell_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=3000)
    row = row.table.iloc[0]
    assert 0.8 < row["error_ratio"] < 1.25


def test_with_a_field_effect_the_real_error_exceeds_the_promise():
    objects, fovs = acquisition(field_sd=1.0, cell_sd=2.0, n_fields=40, per_field=50)
    row = cell_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=2000)
    row = row.table.iloc[0]
    assert row["error_ratio"] > 2
    assert not row["meets_target"]
    assert row["field_rule_fovs"] > 1


def test_control_groups_set_the_reference_mean():
    a = acquisition(mean=10.0, condition="A", seed=1)
    b = acquisition(mean=20.0, condition="B", seed=2)
    objects, fovs = combined(a, b)
    table = cell_rule_check(
        objects, fovs, feature="x", targets=[0.05], control={"condition": "A"}, n_draws=100
    ).table.set_index("condition")
    reference = a[0]["x"].mean()
    for condition, part in (("A", a), ("B", b)):
        sd = part[0]["x"].std(ddof=1) / reference
        assert table.loc[condition, "cell_sd"] == pytest.approx(sd)
    with pytest.raises(ValueError, match="no analysis group matches"):
        cell_rule_check(objects, fovs, feature="x", control={"condition": "C"})


def test_the_rule_check_is_deterministic_and_group_local():
    a = acquisition(field_sd=0.5, condition="A", seed=1)
    b = acquisition(field_sd=0.5, condition="B", seed=2)
    first = cell_rule_check(*combined(a, b), feature="x", n_draws=300).table
    again = cell_rule_check(*combined(a, b), feature="x", n_draws=300).table
    pd.testing.assert_frame_equal(first, again)
    alone = cell_rule_check(*a, feature="x", n_draws=300).table
    np.testing.assert_allclose(
        first[first["condition"] == "A"]["real_error"].to_numpy(), alone["real_error"].to_numpy()
    )


# -- the JCS descriptor spread with whole fields ----------------------------------------


def test_descriptor_spread_is_alike_when_fields_behave_like_random_cells():
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
    row = field_clustering(objects, fovs, label_col="state", states=["rare"]).table.iloc[0]
    assert row["state_cells"] == 20 and row["fovs_with_state"] == 20
    assert row["clustering_ratio"] < 1  # one per field: more spread than random cells
    assert row["p_clustered"] > 0.5


def test_a_state_clustered_in_two_fields_is_called_clustered():
    objects, fovs = acquisition(n_fields=40, per_field=50)
    objects = _labelled(objects, ["f3", "f17"], per_field=10)
    row = field_clustering(objects, fovs, label_col="state", states=["rare"]).table.iloc[0]
    assert row["fovs_with_state"] == 2
    assert row["expected_fovs_if_random"] > 10
    assert row["p_clustered"] <= 0.01
    assert row["largest_field_share"] == pytest.approx(0.5)
    assert row["fovs_needed"] > row["fovs_needed_if_random"]


def test_expected_fields_are_the_hypergeometric_expectation():
    objects, fovs = acquisition(n_fields=4, per_field=5)
    objects = _labelled(objects, ["f0"], per_field=3)
    row = field_clustering(objects, fovs, label_col="state", states=["rare"]).table.iloc[0]
    expected = 4 * (1 - stats.hypergeom(20, 3, 5).pmf(0))
    assert row["expected_fovs_if_random"] == pytest.approx(expected)
    # 5% prevalence at 5 cells per field: 59 cells, so 12 fields, if cells were random
    assert row["fovs_needed_if_random"] == math.ceil(
        math.ceil(math.log(0.05) / math.log(1 - 3 / 20)) / 5
    )


def test_unlabelled_cells_are_left_out_and_empty_fields_count():
    objects, fovs = acquisition(n_fields=10, per_field=10)
    objects = _labelled(objects, ["f0"], per_field=2)
    objects.loc[objects["image_file"] == "f9", "state"] = None
    fovs = pd.concat(
        [fovs, pd.DataFrame({"condition": ["A"], "repeat": ["r1"], "image_file": ["empty"]})],
        ignore_index=True,
    )
    table = field_clustering(objects, fovs, label_col="state").table.set_index("state")
    assert set(table.index) == {"common", "rare"}
    assert table.loc["rare", "labelled"] == 90
    assert table.loc["rare", "fovs"] == 11
    assert table.loc["rare", "fov_incidence"] == pytest.approx(1 / 11)


def test_field_clustering_is_deterministic():
    objects, fovs = acquisition(n_fields=20, per_field=20)
    objects = _labelled(objects, ["f1", "f2", "f3"], per_field=4)
    first = field_clustering(objects, fovs, label_col="state", random_state=3).table
    again = field_clustering(objects, fovs, label_col="state", random_state=3).table
    pd.testing.assert_frame_equal(first, again)


# -- unusual (out-of-distribution) cells ------------------------------------------------


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


def test_planted_far_cells_are_unusual_and_typical_cells_mostly_not():
    objects, fovs, planted = _with_planted()
    result = unusual_cells(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, seed_checks=1
    )
    assert result.labels.loc[planted].all()
    typical = result.labels.drop(planted)
    assert typical.mean() < 0.05
    treated = result.groups.set_index("condition").loc["treated"]
    assert treated["status"] == "ok" and treated["unusual"] >= 10


def test_a_group_is_never_its_own_reference():
    objects, fovs, _ = _with_planted()
    groups = unusual_cells(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).groups
    for row in groups.to_dict("records"):
        own = {"condition": row["condition"], "repeat": row["repeat"]}
        assert own not in row["reference_groups"]
    single = objects[(objects["condition"] != "ctrl") | (objects["repeat"] == "r0")]
    single_fovs = fovs[(fovs["condition"] != "ctrl") | (fovs["repeat"] == "r0")]
    rows = unusual_cells(
        single, single_fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).groups.set_index("condition")
    assert rows.loc["ctrl", "status"] == "no other reference group to score against"
    assert rows.loc["treated", "status"] == "ok"


def test_unusual_cells_use_only_the_explicit_features_and_are_deterministic():
    objects, fovs, _ = _with_planted()
    first = unusual_cells(objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"})
    noisy = objects.assign(stage_x=np.arange(len(objects)) * 1e6)
    again = unusual_cells(noisy, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"})
    pd.testing.assert_series_equal(first.labels, again.labels)
    pd.testing.assert_series_equal(first.distances, again.distances)


def test_unusual_labels_feed_field_clustering():
    objects, fovs, _ = _with_planted()
    labels = unusual_cells(
        objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}
    ).labels
    table = field_clustering(
        objects.assign(unusual=labels), fovs, label_col="unusual", states=[True]
    ).table.set_index("condition")
    treated = table.loc["treated"]
    # the ten planted cells sit in the first field of the treated well
    assert treated["fovs_with_state"] <= 3
    assert treated["p_clustered"] <= 0.05


# -- real data: the JCS case study (idr0139) --------------------------------------------


@pytest.fixture(scope="module")
def idr0139():
    sys.path.insert(0, str(DATA))
    try:
        from datasets import FASCIN_RATIO, load_idr0139
    finally:
        sys.path.remove(str(DATA))
    objects, fovs, _ = load_idr0139()
    return objects, fovs, FASCIN_RATIO


def test_idr0139_fields_are_not_random_cells(idr0139):
    """Regression of docs/preliminary_findings.md entry 9 on the committed wells."""
    objects, fovs, ratio = idr0139
    groups = ["condition", "repeat", "well"]
    variability = field_variability(
        objects, fovs, feature=ratio, group_cols=groups, observation_unit="cell"
    ).groups.set_index("well")
    assert variability.loc["J05", "design_effect"] > 5
    check = cell_rule_check(
        objects,
        fovs,
        feature=ratio,
        targets=[0.025],
        control={"condition": "Untreated"},
        n_draws=1000,
        group_cols=groups,
        observation_unit="cell",
    ).table.set_index("well")
    assert check.loc["J05", "fields_used"] == pytest.approx(1.0)
    assert 0.03 < check.loc["J05", "real_error"] < 0.055
    assert not check.loc["J05", "meets_target"]
