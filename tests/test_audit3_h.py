"""Third audit, group H (with J-1, J-2, J-3, J-4, J-9, J-11, J-14, J-15, J-16): the
sampling-unit checks and the plots.

One test per finding; the reproduction scripts are in the audit's H folder.
"""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import morphocoverage.plots as plots_module
from morphocoverage import schema
from morphocoverage._seeds import group_identity
from morphocoverage.plots import plots_of_data
from morphocoverage.sampling_unit import (
    condition_contrast,
    feature_ranking,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
    variability,
)
from morphocoverage.sampling_unit._units import (
    ENUMERATION_BUDGET,
    permutation_p,
    smallest_p,
)
from morphocoverage.statistics import cluster_size, design_effect

CONTROL = {"condition": "ctrl"}


def _tables(spec, seed=0, per_field=20, obs_sd=1.0, empty_every=0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "technical_repeat": well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            if empty_every and j % empty_every == empty_every - 1:
                continue
            offset = mean + rng.normal(0, field_sd)
            objects += [{**key, "x": v} for v in offset + rng.normal(0, obs_sd, per_field)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _without_technical(objects, fovs):
    return objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")


def _spy(monkeypatch):
    """The units every plotted interval resampled: per call, the fields of each unit."""
    calls = []
    original = plots_module.bootstrap_nested

    def spy(units, *args, **kwargs):
        calls.append([len(unit) for unit in units])
        return original(units, *args, **kwargs)

    monkeypatch.setattr(plots_module, "bootstrap_nested", spy)
    return calls


def test_h1_technical_repeats_are_the_unit_whatever_group_cols_say():
    # Three wells a side, no condition effect, wells far apart: shuffling fields as if
    # independent passed a well-to-well shift for a condition effect (42% at p <= 0.05).
    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        spec = [("ctrl", "r1", f"c{i}", 5, rng.normal(0, 1.0), 0.2) for i in range(3)]
        spec += [("drug", "r1", f"d{i}", 5, rng.normal(0, 1.0), 0.2) for i in range(3)]
        objects, fovs = _tables(spec, seed=seed, per_field=10)
        result = condition_contrast(
            objects,
            fovs,
            feature="x",
            control=CONTROL,
            object_counts=[],
            n_permutations=199,
            group_cols=["condition", "biological_repeat"],
        )
        row = result.table.iloc[0]
        assert row["permutation_unit"] == "technical_repeat"
        assert row["technical"] == row["control_technical"] == 3
        hits += row["p_value"] <= 0.05
    # 20 assignments of 3 wells a side: p never falls below 0.1, so the test cannot help and
    # is not run (it never passed a well-to-well shift for a condition effect either way).
    assert hits == 0 and np.isnan(row["p_value"]) and row["p_floor"] == pytest.approx(0.1)
    assert result.settings["technical_col"] == "technical_repeat"
    assert "with the technical repeat (technical_repeat) as the unit" in " ".join(
        result.summary(detail="full").split()
    )
    assert "too few technical repeats on a side" in " ".join(result.summary().split())
    # The same with the ranking, and with a technical column only in the FOV table.
    objects["y"] = objects["x"]
    ranking = feature_ranking(
        objects,
        fovs,
        features=["x", "y"],
        control=CONTROL,
        group_cols=["condition", "biological_repeat"],
    )
    assert (ranking.tests["permutation_unit"] == "technical_repeat").all()
    bare, bare_fovs = _without_technical(objects, fovs)
    bare_fovs["well"] = fovs["technical_repeat"]
    named = condition_contrast(
        bare, bare_fovs, feature="x", control=CONTROL, technical_col="well", object_counts=[]
    )
    assert named.table["permutation_unit"].iloc[0] == "well"
    bare_fovs.loc[bare_fovs.index[0], "well"] = None
    with pytest.raises(ValueError, match="have no 'well' value"):
        condition_contrast(bare, bare_fovs, feature="x", control=CONTROL, technical_col="well")


def test_h2_pooled_repeats_resample_the_repeats_alone(monkeypatch):
    calls = _spy(monkeypatch)
    spec = [("ctrl", f"r{r}", f"w{i}", 3, float(r), 0.2) for r in range(10) for i in range(2)]
    objects, fovs = _tables(spec, per_field=5)
    title = plots_of_data(objects, fovs, features=["x"], n_boot=20)["x"][0].axes[0]
    # One call for the one pooled cluster: ten repeats, each with its 6 fields (2 wells).
    assert calls == [[6] * 10]
    text = title.figure.caption
    assert "resampling whole biological repeats with all their fields" in text
    assert not hasattr(plots_module, "_bootstrap_repeats")


def test_h3_plot_intervals_are_the_widened_nested_ones_with_the_small_unit_caveat(
    monkeypatch,
):
    calls = _spy(monkeypatch)
    spec = [("ctrl", "r1", f"w{i}", 4, 0.3 * i, 0.2) for i in range(3)]
    objects, fovs = _tables(spec, per_field=10)
    ax = plots_of_data(objects, fovs, features=["x"], n_boot=50)["x"][0].axes[0]
    assert calls == [[4, 4, 4]]  # the wells, each with its fields, into bootstrap_nested
    assert "with 2-3 wells the interval is only a guide" in (ax.figure.caption)


def test_h4_repeats_left_out_of_the_layout_are_pooled_not_merged(monkeypatch):
    calls = _spy(monkeypatch)
    spec = [("ctrl", f"r{r}", w, 4, 2.0 * r, 0.2) for r in range(3) for w in ("A", "B")]
    objects, fovs = _tables(spec, per_field=15)
    ax = plots_of_data(objects, fovs, features=["x"], repeat_col=None, n_boot=20)["x"][0].axes[0]
    # Three repeats, each with its two wells of four fields: never well A of all repeats.
    assert calls == [[8, 8, 8]]
    title = ax.figure.caption
    assert "repeat not shown apart, pooled within each cluster" in title
    assert "whole biological repeats with all their fields" in title
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["1st", "2nd"]


def test_h5_only_the_technical_repeat_column_is_taken_by_itself():
    spec = [(c, "r1", "w", 6, m, 0.2) for c, m in (("ctrl", 0.0), ("drug", 3.0))]
    objects, fovs = _without_technical(*_tables(spec, per_field=15))
    renamed = objects.rename(columns={"condition": "treatment"})
    renamed_fovs = fovs.rename(columns={"condition": "treatment"})
    with pytest.raises(ValueError, match=r"hold \['treatment'\], which the figure does not"):
        plots_of_data(
            renamed, renamed_fovs, features=["x"], group_cols=["treatment", "biological_repeat"]
        )
    figures = plots_of_data(
        renamed,
        renamed_fovs,
        features=["x"],
        condition_col="treatment",
        group_cols=["treatment", "biological_repeat"],
        n_boot=10,
    )
    ax = figures["x"][0].axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ctrl", "drug"]
    assert "coloured by technical repeat" not in ax.figure.caption


def test_h6_a_missing_technical_repeat_value_raises():
    spec = [("ctrl", "r1", w, 4, 0.0, 0.2) for w in ("A", "B", "C")]
    objects, fovs = _without_technical(*_tables(spec, per_field=15))
    fovs["well"] = [p.split("_")[2] for p in fovs["position_id"]]
    fovs.loc[fovs.index[:2], "well"] = None
    with pytest.raises(ValueError, match="2 rows of the FOV table have no 'well' value"):
        plots_of_data(objects, fovs, features=["x"], technical_col="well")


def test_h7_a_field_test_that_cannot_pass_the_gate_is_stated():
    spec = [("ctrl", "r1", "w", 3, 0.0, 0.1), ("drug", "r1", "w", 3, 5.0, 0.1)]
    objects, fovs = _without_technical(*_tables(spec, per_field=30))
    objects["y"] = objects["x"] * 2
    result = feature_ranking(
        objects, fovs, features=["x", "y"], control=CONTROL, fields_independent=True
    )
    assert result.tests["p_floor"].tolist() == pytest.approx([0.1, 0.1])
    # A floor of 0.1 is above the gate: the tests cannot help, so they are not run.
    assert result.tests["p_value"].isna().all()
    text = " ".join(result.summary().split())
    assert "None is ranked" in text and "so it is not run" in text


def test_h8_a_constant_feature_is_reported_not_a_crash():
    objects, fovs = _tables([("ctrl", "r1", "w", 6, 0.0, 0.3)], per_field=20)
    for value in (0.1, 1.0):
        objects["x"] = value
        result = object_rule_check(objects, fovs, feature="x", n_draws=50)
        assert result.table["status"].str.startswith("the feature does not vary").all()
        assert "the feature does not vary" in " ".join(result.summary().split())


def _repeat_shifted(shift, seed, n_repeats):
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for repeat in range(n_repeats):
        offset = rng.normal(0, shift, 2)
        for condition in ("ctrl", "drug"):
            for j in range(10):
                key = {
                    "condition": condition,
                    "biological_repeat": f"r{repeat}",
                    "position_id": f"{condition}{repeat}_{j}",
                }
                fovs.append(key)
                values = offset + rng.normal(0, 0.3, 2) + rng.normal(0, 1, (40, 2))
                rows += [{**key, "a": a, "b": b} for a, b in values]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def test_h9_unusual_thresholds_leave_one_group_out_for_every_group():
    shares = {"ctrl": [], "drug": []}
    for seed in range(3):
        objects, fovs = _repeat_shifted(0.5, seed, n_repeats=6)
        result = unusual_objects(
            objects, fovs, feature_cols=["a", "b"], reference=CONTROL, seed_checks=0
        )
        groups = result.groups
        assert (groups["calibration"] == "leave-one-group-out").all()
        for condition in shares:
            shares[condition] += list(groups[groups["condition"] == condition]["unusual_share"])
    # A repeat shift of half an SD flagged 2.9% of control objects for a nominal 1%
    # when thresholds came from fields of the same repeat.
    assert np.mean(shares["ctrl"]) < 0.02 and np.mean(shares["drug"]) < 0.02
    # A single reference group cannot leave one out: the fields are, and it is said.
    single, single_fovs = _repeat_shifted(0.5, 0, n_repeats=1)
    one = unusual_objects(
        single, single_fovs, feature_cols=["a", "b"], reference=CONTROL, seed_checks=0
    )
    row = one.groups.set_index("condition").loc["drug"]
    assert row["calibration"] == "leave-one-field-out (a single reference group)"
    # The reference named as the reference (audit 10, A24).
    said = " ".join(one.summary().split())
    assert "Against a single reference group, ctrl · repeat r0, the threshold comes from" in said


def test_h10_whole_field_draws_count_acquired_empty_fields():
    spec = [("ctrl", "r1", "w", 40, 10.0, 0.1), ("drug", "r1", "w", 40, 11.0, 0.1)]
    objects, fovs = _without_technical(*_tables(spec, per_field=10, empty_every=2))
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, object_counts=[100], n_draws=20
    )
    row = result.effects[result.effects["sampling"] == "whole FOVs"].iloc[0]
    # Half the 40 fields are empty: 100 objects take about 19-20 acquired fields, not 10.
    assert 17 < row["fields_treated"] < 21


def test_h11_the_field_rule_is_also_given_in_acquired_fields():
    objects, fovs = _tables([("ctrl", "r1", "w", 40, 10.0, 1.0)], per_field=10, empty_every=2)
    result = object_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=200)
    row = result.table.iloc[0]
    assert row["fovs_with_observations"] == 20 and row["fovs"] == 40
    assert row["fov_rule_acquired"] == 2 * row["fov_rule_fovs"]
    text = result.summary(detail="full")
    rule = f"{int(row['fov_rule_fovs'])} ({int(row['fov_rule_fovs_upper95'])})"
    acquired = f"{int(row['fov_rule_acquired'])} ({int(row['fov_rule_acquired_upper95'])})"
    assert "fields with cells" in text and "20 of 40" in text
    assert rule in text and acquired in text
    # The short form gives the fields to acquire.
    assert acquired in result.summary()


def test_h12_a_zero_control_mean_is_not_too_few_fields():
    spec = [("ctrl", "r1", "w", 6, 0.0, 0.0), ("drug", "r1", "w", 6, 3.0, 0.5)]
    objects, fovs = _tables(spec, per_field=30)
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, len(objects)).astype(float)
    objects["mitoses"] = np.where(objects["condition"] == "ctrl", 0.0, counts)
    result = object_rule_check(objects, fovs, feature="mitoses", control=CONTROL, n_draws=50)
    assert result.table["status"].str.contains("the reference mean is 0").all()
    few = object_rule_check(*_tables([("ctrl", "r1", "w", 1, 10.0, 0.1)]), feature="x")
    assert few.table["status"].eq("fewer than 2 fields with cells").all()


def test_h13_a_sparse_group_does_not_read_zero_per_field():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"position_id": [f"f{j}" for j in range(50)]})
    rows = [{"position_id": f"f{j}", "x": v} for j in range(10) for v in rng.normal(j, 1, 2)]
    text = fov_variability(pd.DataFrame(rows), fovs, feature="x").summary(detail="full")
    row = next(line for line in text.splitlines() if line.startswith("  all fields (pooled)"))
    assert row.split()[5] == "0.4"  # fields, objects, then per field


def test_h14_monte_carlo_p_values_never_fall_below_the_floor():
    t = [[np.array([1.0 + 0.01 * i]) for i in range(7)]]
    c = [[np.array([5.0 + 0.01 * i]) for i in range(7)]]
    # 3432 assignments, more than 2000 shuffles but within the budget: enumerated, exact.
    assert 3432 <= ENUMERATION_BUDGET
    p = permutation_p(t, c, False, 2000, np.random.default_rng(0))
    assert p == pytest.approx(smallest_p(7, 7))
    # Beyond the budget the Monte Carlo p-value is floored at the exact test's smallest p.
    t = [[np.array([1.0 + 0.01 * i]) for i in range(12)]]
    c = [[np.array([5.0 + 0.01 * i]) for i in range(12)]]
    p = permutation_p(t, c, False, 99, np.random.default_rng(0))
    assert p == pytest.approx(1 / 100) and p >= smallest_p(12, 12)


def test_h15_numpy_integers_are_accepted_as_counts_and_seeds():
    spec = [("ctrl", "r1", "w", 4, 0.0, 0.2), ("drug", "r1", "v", 4, 1.0, 0.2)]
    objects, fovs = _tables(spec, per_field=10)
    first = condition_contrast(
        objects,
        fovs,
        feature="x",
        control=CONTROL,
        n_permutations=np.int64(99),
        random_state=np.int64(1),
        n_draws=np.int32(5),
        object_counts=[np.int64(20)],
    )
    again = condition_contrast(
        objects,
        fovs,
        feature="x",
        control=CONTROL,
        n_permutations=99,
        random_state=1,
        n_draws=5,
        object_counts=[20],
    )
    pd.testing.assert_frame_equal(first.effects, again.effects)
    feature_ranking(objects, fovs, features=["x"], control=CONTROL, random_state=np.int64(2))
    object_rule_check(objects, fovs, feature="x", n_draws=np.int64(10), random_state=np.int64(0))
    objects["state"] = np.where(np.arange(len(objects)) % 7 == 0, "rare", "common")
    fov_clustering(objects, fovs, label_col="state", n_permutations=np.int64(9))
    plots_of_data(objects, fovs, features=["x"], n_boot=np.int64(10), max_repeats=np.int8(2))


def test_h16_a_technical_column_of_the_fov_table_only_is_used():
    spec = [("ctrl", "r1", w, 4, 0.0, 0.2) for w in ("A", "B", "C")]
    objects, fovs = _without_technical(*_tables(spec, per_field=10))
    fovs["well"] = [p.split("_")[2] for p in fovs["position_id"]]
    ax = plots_of_data(objects, fovs, features=["x"], technical_col="well", n_boot=10)["x"][0]
    title = ax.axes[0].figure.caption
    assert "its shape the well" in title
    assert "whole wells with their fields (3 per repeat" in title


def test_h17_contrast_and_ranking_share_one_test_and_variability_the_design_effect():
    spec = [
        (c, "r1", f"{c}{i}", 5, 2.0 if c == "drug" else 0.0, 0.2)
        for c in ("ctrl", "drug")
        for i in range(4)  # 4 wells a side: a floor of 2/70, so both run the test
    ]
    objects, fovs = _tables(spec)
    contrast = condition_contrast(objects, fovs, feature="x", control=CONTROL, object_counts=[])
    ranking = feature_ranking(objects, fovs, features=["x"], control=CONTROL)
    columns = ["d_fields", "permutation_unit", "distinct_assignments", "p_value", "p_floor"]
    for column in columns:
        assert contrast.table[column].iloc[0] == ranking.tests[column].iloc[0]
    row = fov_variability(objects, fovs, feature="x").groups.iloc[0]
    sizes = objects.groupby("position_id").size().to_numpy()
    assert row["cluster_size"] == pytest.approx(cluster_size(sizes))
    assert row["design_effect"] == pytest.approx(
        design_effect(row["v_fov"], row["v_obs"], row["cluster_size"])
    )


def test_h18_the_variance_components_are_cited_where_they_live():
    assert "statistics.nested_variance_components" in variability.__doc__
    assert "retrospective.effort.nested_variance_components" not in variability.__doc__


def test_h19_an_untested_contrast_keeps_its_effect_sizes():
    spec = [
        ("ctrl", "r1", "c0", 10, 0.0, 0.2),
        ("drug", "r1", "d0", 5, 1.0, 0.2),
        ("drug", "r1", "d1", 5, 1.0, 0.2),
    ]
    objects, fovs = _tables(spec, per_field=30)
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, object_counts=[50, 100], n_draws=20
    )
    row = result.table.iloc[0]
    assert "only one technical repeat" in row["status"] and np.isnan(row["p_value"])
    assert np.isfinite(row["effect_full"]) and np.isfinite(row["d_objects"])
    assert len(result.effects) == 4
    text = " ".join(result.summary().split())
    assert "No well-level test within a repeat in the contrast: only one technical repeat" in text
    assert "d on cells" in text and "JCS effect size" in text


def test_j1_groups_whose_seeds_collide_are_still_told_apart():
    first, second, repeat = "plate24_wellI03_KO", "plate15_wellP15_siGENE", "2024_01_15"
    assert group_identity({"condition": first, "biological_repeat": repeat}) == group_identity(
        {"condition": second, "biological_repeat": repeat}
    )
    rng = np.random.default_rng(0)
    rows, fov_rows = [], []
    for condition, shift in ((first, 0.0), (second, 5.0)):
        for f in range(8):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "position_id": f"{condition}_{f}",
            }
            fov_rows.append(key)
            rows += [{**key, "area": 10 + shift + v} for v in rng.normal(size=30)]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fov_rows)
    control = {"condition": first}
    contrast = condition_contrast(objects, fovs, feature="area", control=control, n_draws=5)
    assert contrast.table["d_objects"].iloc[0] > 3  # not the control against itself
    rules = object_rule_check(objects, fovs, feature="area", control=control, n_draws=20)
    assert rules.table["status"].eq("ok").all()
    unusual = unusual_objects(objects, fovs, feature_cols=["area"], reference=control)
    treated = unusual.groups.set_index("condition").loc[second]
    assert treated["reference_groups"] == [{"condition": first, "biological_repeat": repeat}]
    assert treated["unusual_share"] > 0.5


def test_j2_j16_shares_are_worded_once_and_modules_list_their_names():
    objects = pd.DataFrame(
        {
            "position_id": [f"f{j}" for j in range(90)],
            "state": ["rare"] + ["common"] * 89,
        }
    )
    fovs = objects[["position_id"]]
    text = fov_clustering(objects, fovs, label_col="state", states=["rare"], n_permutations=9)
    line = text.summary()
    assert "1 (1%)" in line and "1 objects" not in line
    assert "Are the state 'rare' cells spread over the fields?" in line
    # No contrast can be tested: the ranking does not speak of BH over 0 tests.
    spec = [("ctrl", "r1", "c0", 1, 0.0, 0.2), ("drug", "r1", "d0", 1, 1.0, 0.2)]
    ranking = feature_ranking(*_tables(spec), features=["x"], control=CONTROL)
    assert "over the 0 tests" not in ranking.summary(detail="full")
    assert "though no contrast could be tested" in " ".join(ranking.summary(detail="full").split())
    assert "no contrast could be tested" in " ".join(ranking.summary().split())
    import morphocoverage.sampling_unit as package

    for name in ("clustering", "contrast", "object_rules", "ranking", "variability"):
        module = getattr(package, name)
        assert all(hasattr(module, public) for public in module.__all__)


def test_j3_j9_j11_plots_use_the_shared_helpers():
    source = inspect.getsource(plots_module)
    assert "sampling_unit" not in source.split('"""', 2)[2]  # no private import from there
    assert not hasattr(plots_module, "_technical_count")
    assert not hasattr(plots_module, "_ordinal")
    assert plots_module.fov_values is schema.fov_values
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        plots_of_data(*_tables([("ctrl", "r1", "w", 2, 0, 0)]), features=["x"], n_boot=True)
