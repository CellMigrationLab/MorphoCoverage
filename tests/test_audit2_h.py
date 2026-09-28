"""Second audit, group H (with J-8, J-12, J-17, D-5): the sampling-unit checks and the plots.

One test per finding; the reproduction scripts are in the audit's H, J and D folders.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

import morphocoverage.plots as plots_module
from morphocoverage.plots import plots_of_data
from morphocoverage.sampling_unit import (
    condition_contrast,
    feature_ranking,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.sampling_unit._units import (
    field_means,
    permutation_p,
    separation_d,
)
from morphocoverage.statistics import bootstrap_fields, bootstrap_nested, first_detection_fovs

CONTROL = {"condition": "ctrl"}


def _tables(spec, seed=0, per_field=20, obs_sd=1.0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "repeat": repeat,
                "technical_repeat": well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            offset = mean + rng.normal(0, field_sd)
            n = per_field if np.isscalar(per_field) else per_field[j % len(per_field)]
            objects += [{**key, "x": v} for v in offset + rng.normal(0, obs_sd, n)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _exact_p(t_units, c_units):
    """The exact p-value over every assignment of whole units, by brute force."""
    units = [*t_units, *c_units]
    observed = abs(separation_d(field_means(t_units), field_means(c_units)))
    stats = []
    for chosen in itertools.combinations(range(len(units)), len(t_units)):
        first = [units[i] for i in chosen]
        second = [units[i] for i in range(len(units)) if i not in chosen]
        stats.append(abs(separation_d(field_means(first), field_means(second))))
    return float(np.mean(np.array(stats) >= observed * (1 - 1e-9)))


def test_h1_perfectly_separating_shuffles_count_as_extreme():
    # Two wells a side, every object of a well the same value: a perfect separation.
    spec = [("ctrl", "r1", f"c{i}", 4, 1.0, 0) for i in range(2)]
    spec += [("drug", "r1", f"d{i}", 4, 3.0, 0) for i in range(2)]
    objects, fovs = _tables(spec, obs_sd=0.0)
    result = feature_ranking(objects, fovs, features=["x"], control=CONTROL, n_permutations=1999)
    row = result.tests.iloc[0]
    assert np.isinf(row["d_fields"])
    # 6 assignments; the observed one and its mirror are both perfect: p = 2/6.
    assert row["p_value"] == pytest.approx(1 / 3)
    assert not row["passes_gate"] and result.top() == []
    t = [[np.array([3.0]), np.array([3.0])], [np.array([3.0]), np.array([3.0])]]
    c = [[np.array([1.0]), np.array([1.0])], [np.array([1.0]), np.array([1.0])]]
    # 8 fields shuffled: of 70 assignments, the two that keep the sides whole are infinite.
    p = permutation_p(t, c, False, 100, np.random.default_rng(0))
    assert p == pytest.approx(2 / 70)
    shuffled = permutation_p(t, c, False, 69, np.random.default_rng(0))
    assert shuffled > 1 / 70  # at random too, the infinite shuffles count


def test_h2_float_ties_count_and_few_assignments_are_enumerated():
    t = [[np.array([3.0]), np.array([3.1])], [np.array([3.2]), np.array([3.3])]]
    c = [[np.array([1.0]), np.array([1.1])], [np.array([1.2]), np.array([1.3])]]
    # Six assignments, fewer than the shuffles asked for: enumerated, exactly 1/3.
    exact = permutation_p(t, c, True, 5999, np.random.default_rng(0))
    assert exact == pytest.approx(_exact_p(t, c)) and exact == pytest.approx(1 / 3)
    # Under no difference no p-value falls below the attainable floor (2/20 with 3 v 3 fields).
    rng = np.random.default_rng(5)
    ps = []
    for _ in range(300):
        tt = [[rng.normal(size=5) for _ in range(3)]]
        cc = [[rng.normal(size=5) for _ in range(3)]]
        ps.append(permutation_p(tt, cc, False, 199, rng))
    assert min(ps) == pytest.approx(0.1)
    # More assignments than shuffles: the Monte Carlo (extreme + 1) / (n + 1).
    many = [[rng.normal(size=4) for _ in range(12)]]
    other = [[rng.normal(size=4) for _ in range(12)]]
    p = permutation_p(many, other, False, 99, np.random.default_rng(1))
    assert (p * 100) == pytest.approx(round(p * 100))


def test_h3_one_technical_repeat_on_a_side_gets_no_test():
    spec = [("drug", "r1", "d0", 8, 0.3, 0.2)] + [
        ("ctrl", "r1", f"c{i}", 8, m, 0.2) for i, m in enumerate((-0.4, 0.1, 0.5))
    ]
    objects, fovs = _tables(spec, per_field=15)
    common = {"n_permutations": 99, "control": CONTROL}
    contrast = condition_contrast(
        objects, fovs, feature="x", n_draws=2, object_counts=(10,), **common
    )
    row = contrast.table.iloc[0]
    assert np.isnan(row["p_value"]) and "only one technical repeat" in row["status"]
    assert "cannot stand in for technical repeats" in contrast.summary()
    objects["y"] = objects["x"] * 2
    ranking = feature_ranking(objects, fovs, features=["x", "y"], **common)
    assert ranking.tests["p_value"].isna().all() and not ranking.tests["passes_gate"].any()
    text = ranking.summary()
    assert "Not ranked because no contrast could be tested (2 features)" in text
    assert "No contrast passes the gate" not in text
    # Without the technical-repeat column among the groups it is still the unit (audit 3,
    # H-1): the single treated well gets no test either.
    pooled = condition_contrast(
        objects,
        fovs,
        feature="x",
        n_draws=2,
        object_counts=(10,),
        group_cols=["condition", "repeat"],
        **common,
    )
    assert "only one technical repeat" in pooled.table["status"].iloc[0]
    assert np.isnan(pooled.table["p_value"].iloc[0])


def test_h4_a_constant_feature_is_never_separated_or_variable():
    sizes = [3, 7, 11, 5, 9, 13]
    for value in (0.1, 0.7):
        objects = pd.DataFrame(
            {"image_file": np.repeat([f"f{j}" for j in range(6)], sizes), "x": value}
        )
        fovs = pd.DataFrame({"image_file": [f"f{j}" for j in range(6)]})
        row = fov_variability(objects, fovs, feature="x").groups.iloc[0]
        assert row["status"] == "the feature is constant"
        assert row["icc"] is None and row["design_effect"] is None
        assert row["unusual_fov_share"] is None
    # Field means of a constant 0.1 differ by rounding only: no separation.
    first = np.array([np.full(3, 0.1).mean(), np.full(7, 0.1).mean()])
    second = np.array([np.full(11, 0.1).mean(), np.full(5, 0.1).mean()])
    assert np.isnan(separation_d(first, second))
    t_objects, t_fovs = _tables([("drug", "r1", "w2", 6, 0, 0)], per_field=[3, 7])
    c_objects, c_fovs = _tables([("ctrl", "r1", "w1", 6, 0, 0)], per_field=[3, 7, 11])
    objects = pd.concat([t_objects, c_objects], ignore_index=True)
    fovs = pd.concat([t_fovs, c_fovs], ignore_index=True)
    objects["x"] = 0.1
    ranking = feature_ranking(
        objects,
        fovs,
        features=["x"],
        control=CONTROL,
        technical_col=None,
        n_permutations=99,
        fields_independent=True,
    )
    assert np.isnan(ranking.tests["d_fields"].iloc[0]) and ranking.top() == []
    assert "do not vary" in ranking.tests["status"].iloc[0]


def test_h5_the_ranking_floor_is_the_smallest_attainable_p():
    spec = [("ctrl", "r1", f"c{i}", 5, 0.0, 0.3) for i in range(3)]
    spec += [("drug", "r1", f"d{i}", 5, 5.0 + 0.3 * i, 0.3) for i in range(3)]
    objects, fovs = _tables(spec, seed=2)
    result = feature_ranking(objects, fovs, features=["x"], control=CONTROL, n_permutations=1999)
    # 20 assignments of 3 wells a side; the split and its mirror: 2/20.
    assert result.tests["p_value"].iloc[0] == pytest.approx(0.1)
    text = result.summary()
    assert "with as few as 20 distinct assignments a p-value cannot fall below 0.1," in text
    assert "cannot pass the gate however large the difference" in text


def _perfect_separation():
    rows, fields = [], []
    for condition, mean in (("ctrl", 10.0), ("drug", 12.0)):
        for j in range(6):
            name = f"{condition}_{j}"
            fields.append({"condition": condition, "image_file": name})
            rows += [
                {"condition": condition, "image_file": name, "x": v} for v in (mean - 1, mean + 1)
            ]
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_h6_the_contrast_scores_a_perfect_separation_as_the_ranking_does():
    objects, fovs = _perfect_separation()
    common = {
        "control": CONTROL,
        "group_cols": ["condition"],
        "n_permutations": 199,
        "fields_independent": True,
    }
    contrast = condition_contrast(
        objects, fovs, feature="x", n_draws=5, object_counts=(4,), **common
    )
    row = contrast.table.iloc[0]
    assert row["status"] == "ok" and np.isinf(row["d_fields"]) and row["d_fields"] > 0
    ranking = feature_ranking(objects, fovs, features=["x"], **common)
    assert ranking.tests["d_fields"].iloc[0] == row["d_fields"]
    # Two perfect assignments (the split and its mirror) of 924: p is 2/924, exact.
    assert row["p_value"] > 0 and "do not vary" not in contrast.summary()
    assert "on field means infinite (no variation within either side)" in contrast.summary()


def test_h7_unlabelled_fields_are_not_fields_without_the_state():
    objects, fovs = _tables([("ctrl", "r1", "w1", 40, 0, 0)], per_field=20)
    rng = np.random.default_rng(0)
    state_fields = set(fovs["position_id"].iloc[::4])
    objects["label"] = np.where(
        objects["position_id"].isin(state_fields) & (rng.random(len(objects)) < 0.3),
        "rare",
        "common",
    )
    half = objects.copy()
    annotated = set(fovs["position_id"].iloc[:20])
    half.loc[~half["position_id"].isin(annotated), "label"] = None
    row = fov_clustering(
        half, fovs, label_col="label", states=["rare"], n_permutations=99
    ).table.iloc[0]
    assert row["fovs_acquired"] == 40 and row["fovs_unlabelled"] == 20 and row["fovs"] == 20
    seen = half[half["label"] == "rare"]["position_id"].nunique()
    assert row["fov_incidence"] == pytest.approx(seen / 20)
    # An acquired field without observations still counts as a field without the state.
    empty = pd.DataFrame([{**fovs.iloc[0].to_dict(), "position_id": "empty"}])
    with_empty = fov_clustering(
        half, pd.concat([fovs, empty]), label_col="label", states=["rare"], n_permutations=9
    ).table.iloc[0]
    assert with_empty["fovs"] == 21 and with_empty["fovs_unlabelled"] == 20


def _two_repeats():
    spec = [
        (c, r, f"{c}{r}{i}", 4, (1.0 if c == "drug" else 0) + (2.0 if r == "r2" else 0), 0.2)
        for r in ("r1", "r2")
        for c in ("ctrl", "drug")
        for i in range(2)
    ]
    return _tables(spec)


def test_h8_a_misspelt_repeat_or_technical_column_raises():
    objects, fovs = _two_repeats()
    common = {"control": CONTROL, "n_permutations": 9}
    contrast = {"feature": "x", "n_draws": 2, "object_counts": (20,), **common}
    with pytest.raises(ValueError, match="repeat_col 'Repeat' is not a column"):
        condition_contrast(objects, fovs, repeat_col="Repeat", **contrast)
    with pytest.raises(ValueError, match="technical_col 'Technical_repeat' is not a column"):
        condition_contrast(objects, fovs, technical_col="Technical_repeat", **contrast)
    with pytest.raises(ValueError, match="repeat_col 'Repeat'"):
        feature_ranking(objects, fovs, features=["x"], repeat_col="Repeat", **common)
    objects["well"], fovs["well"] = objects["technical_repeat"], fovs["technical_repeat"]
    # A technical-repeat column of the FOV table is the unit whatever group_cols says
    # (audit 3, H-1), so naming one outside the groups is no longer refused.
    named = feature_ranking(objects, fovs, features=["x"], technical_col="well", **common)
    assert named.settings["technical_col"] == "well"
    # Only the defaults may be absent: without a repeat column, one implicit repeat.
    alone = condition_contrast(
        objects.drop(columns="repeat"),
        fovs.drop(columns="repeat"),
        group_cols=["condition", "technical_repeat"],
        **contrast,
    )
    assert alone.settings["repeat_col"] is None and len(alone.table) == 1


def test_h9_plots_raise_on_a_misspelt_column():
    objects, fovs = _two_repeats()
    common = {"features": ["x"], "n_boot": 10}
    for argument in ("condition_col", "repeat_col", "technical_col"):
        with pytest.raises(ValueError, match=f"{argument} 'Wrong' is not a column"):
            plots_of_data(objects, fovs, **{argument: "Wrong"}, **common)
    # The default condition and repeat columns may be absent.
    figures = plots_of_data(
        objects.drop(columns="repeat"),
        fovs.drop(columns="repeat"),
        group_cols=["condition", "technical_repeat"],
        **common,
    )
    assert len(figures["x"]) == 1


def test_h10_pooled_repeats_resample_the_repeats_first():
    rng = np.random.default_rng(0)
    # Ten repeats far apart, each one well of tight fields.
    repeats = [[[rng.normal(3 * r, 0.1, 10) for _ in range(6)]] for r in range(10)]
    fields = [f for repeat in repeats for unit in repeat for f in unit]
    # Pooled repeats resample the repeats alone, each with all its fields (audit 3, H-2).
    units = [[f for unit in repeat for f in unit] for repeat in repeats]
    _, low, high = bootstrap_nested(units, np.mean, n_boot=400, rng=np.random.default_rng(1))
    _, f_low, f_high = bootstrap_fields(fields, np.mean, n_boot=400, rng=np.random.default_rng(1))
    # Repeat means have SD 9: the mean of 10 has SE 2.9, a 95% interval about 11 wide.
    # Resampling the 60 fields instead treats them as independent: about sqrt(6) narrower.
    assert high - low > 9 and high - low > 2 * (f_high - f_low)
    spec = [("ctrl", f"r{r}", "w", 3, float(r), 0.2) for r in range(10)]
    objects, fovs = _tables(spec, per_field=5)
    title = (
        plots_of_data(objects, fovs, features=["x"], n_boot=20)["x"][0]
        .axes[0]
        .get_title(loc="left")
    )
    assert "resampling whole biological repeats (repeat), each with all its fields" in title


def test_h11_jcb_rules_on_negative_values_are_not_applicable():
    objects, fovs = _tables([("ctrl", "r1", "w1", 20, -2.0, 0.5)], per_field=30)
    result = object_rule_check(objects, fovs, feature="x", n_draws=50)
    assert result.table["status"].str.startswith("not applicable").all()
    assert "non-negative measurements" in result.summary()
    # A positive group normalized by a control with negative values: also not applicable.
    treated, t_fovs = _tables([("drug", "r1", "w2", 10, 20.0, 0.5)], per_field=30)
    control, c_fovs = _tables([("ctrl", "r1", "w1", 10, 0.5, 0.5)], per_field=30)
    both = object_rule_check(
        pd.concat([treated, control]),
        pd.concat([t_fovs, c_fovs]),
        feature="x",
        control=CONTROL,
        n_draws=50,
    )
    assert both.table["status"].str.startswith("not applicable").all()
    fine = object_rule_check(treated, t_fovs, feature="x", n_draws=50)
    assert (fine.table["status"] == "ok").all()


def test_h12_the_legend_counts_wells_within_each_repeat():
    spec = [("ctrl", f"r{r}", f"r{r}_w{i}", 3, 0, 0.1) for r in range(9) for i in range(2)]
    objects, fovs = _tables(spec, per_field=5)
    ax = plots_of_data(objects, fovs, features=["x"], n_boot=10)["x"][0].axes[0]
    assert not ax.texts
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["1st technical_repeat", "2nd technical_repeat"]


def test_h14_d5_fields_if_random_use_the_shared_detection_count():
    # 25 fields of one labelled object, 5 in the state: prevalence 0.2 and beta 0.36,
    # where the closed form rounds 2.0000000000000004 up to 3.
    objects = pd.DataFrame(
        {"position_id": [f"f{j}" for j in range(25)], "state": ["rare"] * 5 + ["common"] * 20}
    )
    fovs = pd.DataFrame({"position_id": [f"f{j}" for j in range(25)]})
    row = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=["rare"],
        detection_probability=0.36,
        n_permutations=9,
    ).table.iloc[0]
    assert row["fovs_needed_if_random"] == first_detection_fovs(0.2, 0.36) == 2
    assert row["fovs_needed"] == 2


def test_h15_untested_features_are_listed_as_untested():
    spec = [
        (c, "r1", f"{c}{i}", 5, 2.0 if c == "drug" else 0, 0.2)
        for c in ("ctrl", "drug")
        for i in range(3)
    ]
    objects, fovs = _tables(spec)
    empty = {"condition": "drug", "repeat": "r1", "technical_repeat": "drug0"}
    fovs = pd.concat([fovs, pd.DataFrame([{**empty, "position_id": "empty"}])], ignore_index=True)
    objects["y"] = np.where(objects["condition"] == "drug", np.nan, objects["x"])
    objects["w"] = np.random.default_rng(0).normal(size=len(objects))
    # Fields compared as independent draws (audit 4 Q1): the wells are left out of the
    # groups and the unit, and the fields declared independent.
    common = {
        "control": CONTROL,
        "technical_col": None,
        "group_cols": ["condition", "repeat"],
        "fields_independent": True,
    }
    result = feature_ranking(objects, fovs, features=["x", "y", "w"], n_permutations=199, **common)
    text = result.summary()
    assert "No contrast passes the gate for 1 feature: w." in text
    assert (
        "Not ranked because no contrast could be tested (1 feature): y (fewer than 2 fields "
        "with observations on one side)."
    ) in text
    contrast = condition_contrast(
        objects, fovs, feature="x", n_permutations=99, n_draws=5, object_counts=(10,), **common
    )
    x_row = result.tests[result.tests["feature"] == "x"].iloc[0]
    assert x_row["fields"] == contrast.table["fields"].iloc[0] == 16
    assert x_row["control_fields"] == contrast.table["control_fields"].iloc[0]


def test_h16_the_variability_summary_states_missing_values():
    objects, fovs = _tables([("ctrl", "r1", "w1", 10, 0, 0.5)], per_field=20)
    objects.loc[np.random.default_rng(0).random(len(objects)) < 0.4, "x"] = np.nan
    missing = int(objects["x"].isna().sum())
    result = fov_variability(objects, fovs, feature="x")
    assert f"Objects with no value were left out: {missing}." in result.summary()
    full = result.summary(detail="full")
    assert "no value, left out" in full and f" {missing} " in full
    complete = fov_variability(objects.dropna(), fovs, feature="x")
    assert "left out" not in complete.summary()
    assert "left out" not in complete.summary(detail="full")


def _caps(figure):
    segments = [
        s
        for c in figure.axes[0].collections
        if isinstance(c, LineCollection)
        for s in c.get_segments()
    ]
    return sorted({round(float(s[0][1]), 6) for s in segments if abs(s[0][1] - s[1][1]) < 1e-12})


def test_h17_every_cluster_draws_from_its_own_generator():
    objects, fovs = _tables([("ctrl", "r1", f"w{i}", 6, i * 0.5, 0.5) for i in range(3)])
    objects["y"] = objects["x"] * 2
    alone = _caps(plots_of_data(objects, fovs, features=["y"])["y"][0])
    after = _caps(plots_of_data(objects, fovs, features=["x", "y"])["y"][0])
    assert alone == after
    more, more_fovs = _tables(
        [("ctrl", "r1", f"w{i}", 6, i * 0.5, 0.5) for i in range(3)]
        + [("drug", "r1", f"v{i}", 6, 1.0, 0.5) for i in range(3)]
    )
    more["y"] = more["x"] * 2
    ctrl = _caps(
        plots_of_data(more[more["condition"] == "ctrl"], more_fovs.iloc[:18], features=["y"])["y"][
            0
        ]
    )
    both = plots_of_data(more, more_fovs, features=["y"])["y"][0]
    assert set(ctrl) <= set(_caps(both))


def test_h18_a_control_naming_an_absent_column_raises_clearly():
    objects, fovs = _tables([("ctrl", "r1", "w1", 4, 0, 0.2), ("drug", "r1", "w2", 4, 1, 0.2)])
    control = {"condition": "ctrl", "plate": "P1"}
    with pytest.raises(ValueError, match=r"control names columns absent .*\['plate'\]"):
        condition_contrast(objects, fovs, feature="x", control=control, n_permutations=9)
    with pytest.raises(ValueError, match=r"control names columns absent .*\['plate'\]"):
        feature_ranking(objects, fovs, features=["x"], control=control, n_permutations=9)


def test_j8_plot_intervals_include_empty_fields(monkeypatch):
    rng = np.random.default_rng(0)
    fov_rows, rows = [], []
    for j in range(12):
        fov_rows.append({"condition": "A", "position_id": f"f{j}"})
        if j < 4:  # 4 fields with objects, 8 acquired empty fields
            rows += [
                {"condition": "A", "position_id": f"f{j}", "x": v}
                for v in rng.normal(10 + 2 * j, 1, size=20)
            ]
    captured = []
    original = plots_module.bootstrap_nested

    def spy(units, *args, **kwargs):
        captured.append([len(unit) for unit in units])
        return original(units, *args, **kwargs)

    monkeypatch.setattr(plots_module, "bootstrap_nested", spy)
    plots_of_data(
        pd.DataFrame(rows), pd.DataFrame(fov_rows), features=["x"], group_cols=[], n_boot=20
    )
    assert captured == [[12]]


def test_j12_a_group_without_complete_features_is_not_assessed():
    rng = np.random.default_rng(0)
    rows = []
    for condition in ("MOCK", "MOCK2", "KD"):
        for j in range(4):
            for v, w in rng.normal(0, 1, size=(20, 2)):
                y = np.nan if condition == "KD" else w
                rows.append(
                    {"condition": condition, "position_id": f"{condition}{j}", "x": v, "y": y}
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "position_id"]].drop_duplicates()
    result = unusual_objects(
        objects,
        fovs,
        feature_cols=["x", "y"],
        reference={"condition": "MOCK"},
        group_cols=["condition"],
    )
    kd = result.groups.set_index("condition").loc["KD"]
    assert kd["status"].startswith("not assessed") and kd["status"] != "ok"
    text = " ".join(result.summary().split())
    assert (
        "Unusual objects not scored in KD: not assessed: none of its 80 objects has a value of "
        "every feature." in text
    )
    assert "nan%" not in text and "0 of 0" not in text


def test_j17_clustering_words_are_proper_plurals():
    objects = pd.DataFrame(
        {
            "condition": ["A"] * 4 + ["B"] * 4,
            "position_id": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
            "state": ["x", None, "x", "y", None, None, None, None],
        }
    )
    fovs = objects[["condition", "position_id"]].drop_duplicates()
    for unit, words in (("patch", "patches"), ("nucleus", "nuclei")):
        text = fov_clustering(
            objects,
            fovs,
            label_col="state",
            states=["x"],
            n_permutations=9,
            observation_unit=unit,
            group_cols=["condition"],
        ).summary()
        text = " ".join(text.split())
        assert f"none of the {words} of its 2 fields is labelled" in text
        assert "patche " not in text
