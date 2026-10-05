"""Fourth audit, group C: the sampling-unit analyses (contrast, ranking, object rule).

One test per finding, on small deterministic tables. Owner decisions: Q1 (the
shuffled unit: technical_repeat, else a nested group column, else no p-value
unless ``fields_independent=True``), D3 (each repeat's own control mean), D4
(the reachable q of the ranking), D20 (the same-sign draws).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from morphocoverage._plots_text import title as _title
from morphocoverage.plots import _Layout
from morphocoverage.sampling_unit import (
    condition_contrast,
    descriptor_spread_check,
    feature_ranking,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.sampling_unit._units import NOT_INDEPENDENT
from morphocoverage.workflows import Setup
from morphocoverage.workflows._declaration import declaration
from morphocoverage.workflows._shared import conditions

CONTROL = {"condition": "ctrl"}


def _tables(spec, seed=0, per_field=20, well_col="technical_repeat"):
    """``spec``: (condition, repeat, well, fields, well mean) per well; field SD 0.3."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                well_col: well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            level = mean + rng.normal(0, 0.3)
            objects += [
                {**key, "x": v, "y": 2 * v + 1} for v in level + rng.normal(0, 1, per_field)
            ]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _wells(n, *, repeats=("r1",), shift=0.0, fields=4, well_col="technical_repeat", seed=0):
    spec = [
        (condition, repeat, f"{condition}{i}", fields, 10 + (shift if condition == "drug" else 0))
        for repeat in repeats
        for condition in ("ctrl", "drug")
        for i in range(n)
    ]
    return _tables(spec, seed=seed, well_col=well_col)


def _groups(well_col):
    return ["condition", "biological_repeat", well_col]


# --- Q1: the shuffled unit ---------------------------------------------------------


def test_p1_1_a_well_column_in_group_cols_is_the_shuffled_unit():
    objects, fovs = _wells(4, well_col="well")
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, group_cols=_groups("well"), n_draws=5
    )
    row = result.table.iloc[0]
    assert result.settings["unit"] == "well"
    assert row["permutation_unit"] == "well"
    assert (row["technical"], row["control_technical"]) == (4, 4)
    # 4 wells a side: 70 assignments, mirrors pair up, so p cannot fall below 2/70.
    assert row["p_floor"] == pytest.approx(2 / 70) and row["status"] == "ok"
    assert "whole technical repeats (well" in " ".join(result.summary().split())
    assert "whole technical repeats (well" in result.summary(detail="full")
    ranking = feature_ranking(
        objects, fovs, features=["x", "y"], control=CONTROL, group_cols=_groups("well")
    )
    assert set(ranking.tests["permutation_unit"]) == {"well"}


def test_p3_2_without_a_unit_there_is_no_p_value_unless_fields_are_declared_independent():
    objects, fovs = _wells(1, shift=0.5)
    objects, fovs = objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")
    refused = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    row = refused.table.iloc[0]
    assert np.isnan(row["p_value"]) and row["status"] == NOT_INDEPENDENT
    text = " ".join(refused.summary().split())
    assert "fields of one dish or well are not independent draws" in text
    assert "fields_independent=True" in text
    assert np.isfinite(row["d_fields"])  # the effect sizes are still reported
    declared = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, n_draws=5, fields_independent=True
    )
    row = declared.table.iloc[0]
    assert row["status"] == "ok" and row["permutation_unit"] == "field"
    assert np.isfinite(row["p_value"])
    assert "treated as independent draws as declared" in " ".join(declared.summary().split())
    assert "treated as independent draws as declared" in " ".join(
        declared.summary(detail="full").split()
    )
    ranking = feature_ranking(objects, fovs, features=["x", "y"], control=CONTROL)
    assert ranking.tests["p_value"].isna().all()
    assert "not independent draws" in " ".join(ranking.summary().split())
    with pytest.raises(TypeError, match="fields_independent"):
        condition_contrast(objects, fovs, feature="x", control=CONTROL, fields_independent=1)


def test_p5_3_a_metadata_well_column_with_one_well_on_a_side_gets_no_test():
    spec = [
        ("ctrl", "r1", "A1", 4, 10.0),
        ("ctrl", "r1", "A2", 4, 10.0),
        ("drug", "r1", "B1", 4, 12.0),
        ("drug", "r1", "B2", 4, 12.0),
        ("single", "r1", "C1", 4, 12.0),
    ]
    objects, fovs = _tables(spec, well_col="Metadata_Well")
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control=CONTROL,
        group_cols=_groups("Metadata_Well"),
        n_draws=5,
    )
    table = result.table.set_index("condition")
    assert table.loc["drug", "permutation_unit"] == "Metadata_Well"
    assert np.isnan(table.loc["single", "p_value"])
    assert "only one technical repeat (Metadata_Well)" in table.loc["single", "status"]
    # Well names restarting in every condition are distinct wells, identified with their
    # condition and repeat; a column with one value per condition and repeat is no unit.
    restarted = fovs.assign(Metadata_Well=fovs["Metadata_Well"].str[1])
    restarted_objects = objects.assign(Metadata_Well=objects["Metadata_Well"].str[1])
    result = condition_contrast(
        restarted_objects,
        restarted,
        feature="x",
        control=CONTROL,
        group_cols=_groups("Metadata_Well"),
        n_draws=5,
    )
    assert result.table.set_index("condition").loc["drug", "permutation_unit"] == "Metadata_Well"
    one = fovs.assign(dish="d")
    result = condition_contrast(
        objects.assign(dish="d"), one, feature="x", control=CONTROL, group_cols=_groups("dish")
    )
    assert result.settings["unit"] is None
    assert "holds one value per condition and repeat" in result.summary()
    assert result.table["p_value"].isna().all()


def test_q1_setup_fields_independent_reaches_the_report_contrast():
    objects, fovs = _wells(1, shift=0.5)
    objects, fovs = objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")
    for independent, tested in ((False, False), (True, True)):
        setup = Setup("x", features=("y",), control=CONTROL, fields_independent=independent)
        results = conditions(objects, fovs, setup, {})
        assert bool(results["contrast"].table["p_value"].notna().all()) is tested
        assert bool(results["ranking"].tests["p_value"].notna().all()) is tested
    with pytest.raises(TypeError, match="fields_independent"):
        Setup("x", fields_independent="yes")


def test_p1_1_contrasts_passing_with_opposite_signs_are_flagged():
    spec = []
    for repeat, drug in (("r1", 14.0), ("r2", 6.0)):
        # 5 wells a side: a floor of 2/252, so both contrasts can pass after Bonferroni.
        spec += [("ctrl", repeat, f"c{i}", 3, 10.0) for i in range(5)]
        spec += [("drug", repeat, f"d{i}", 3, drug) for i in range(5)]
    objects, fovs = _tables(spec)
    result = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    assert (result.table["p_value"] <= 0.05).all()
    assert "in opposite directions" in result.summary()
    ranking = feature_ranking(objects, fovs, features=["x"], control=CONTROL)
    assert "passes the gate for drug in opposite directions" in ranking.summary()


# --- D20: the same-sign draws --------------------------------------------------------


def test_p1_6_same_sign_draws_resample_whole_technical_repeats():
    objects, fovs = _wells(3, shift=0.5)
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, object_counts=(50,), n_draws=20
    )
    whole = result.effects[result.effects["sampling"] == "whole FOVs"]
    assert set(whole["resampled"]) == {"technical_repeat"}
    text = " ".join(result.summary().split())
    assert "the fields of technical repeats (technical_repeat) drawn with replacement" in text
    assert "not a confidence in the effect" in text
    full = result.summary(detail="full")
    assert "resampled" in full and full.count("technical_repeat") >= 3
    objects, fovs = objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")
    fields_only = condition_contrast(
        objects,
        fovs,
        feature="x",
        control=CONTROL,
        object_counts=(50,),
        n_draws=20,
        fields_independent=True,
    )
    assert fields_only.effects["resampled"].isna().all()
    assert "The draws resample these fields and cells, not wells" in " ".join(
        fields_only.summary().split()
    )


# --- D3: each repeat's own control mean ----------------------------------------------


def _shifted(scale):
    objects, fovs = _wells(2, repeats=("r1", "r2"), fields=6)
    objects.loc[objects["biological_repeat"] == "r2", "x"] *= scale
    return objects, fovs


def test_p1_4_the_object_rule_divides_by_the_repeats_own_control_mean():
    objects, fovs = _shifted(2.0)
    result = object_rule_check(objects, fovs, feature="x", control=CONTROL, n_draws=50)
    table = result.table[result.table["target"] == 0.05]
    for repeat in ("r1", "r2"):
        in_repeat = objects[objects["biological_repeat"] == repeat]
        reference = in_repeat.loc[in_repeat["condition"] == "ctrl", "x"].mean()
        for row in table[table["biological_repeat"] == repeat].to_dict("records"):
            group = in_repeat[in_repeat["technical_repeat"] == row["technical_repeat"]]
            sd = (group["x"] / reference).std(ddof=1)
            assert row["rule_objects"] == int(np.ceil((sd / 0.05) ** 2))
    # The same CV in both repeats: the same rule, although the levels differ 2x.
    by_repeat = table.groupby("biological_repeat")["rule_objects"].median()
    assert by_repeat["r1"] == pytest.approx(by_repeat["r2"], rel=0.35)
    text = result.summary()
    assert "of the same biological repeat" in text
    assert "differ by 2.0x" in text


def test_p5_6_a_unit_change_in_one_repeat_is_flagged_and_a_repeat_without_control_refused():
    objects, fovs = _shifted(0.1)
    text = object_rule_check(objects, fovs, feature="x", control=CONTROL, n_draws=50).summary()
    text = " ".join(text.split())
    assert "differ by 10.0x" in text and "change of units" in text
    keep = ~((objects["biological_repeat"] == "r2") & (objects["condition"] == "ctrl"))
    result = object_rule_check(objects[keep], fovs, feature="x", control=CONTROL, n_draws=50)
    r2 = result.table[result.table["biological_repeat"] == "r2"]
    assert (r2["status"].str.contains("no control object in this group's repeat")).all()
    assert (result.table[result.table["biological_repeat"] == "r1"]["status"] == "ok").all()


# --- D4: the reachable q of the ranking ------------------------------------------------


def test_p1_3_the_ranking_states_the_smallest_reachable_q():
    # 4 wells a side: a p-value cannot fall below 2/70 = 0.029, under the 0.05 gate, but
    # corrected over 3 features x 1 contrast a lone contrast at the floor gets q = 0.086.
    objects, fovs = _wells(4, shift=8.0)
    objects["noise"] = np.random.default_rng(1).normal(size=len(objects))
    ranking = feature_ranking(objects, fovs, features=["x", "noise", "y"], control=CONTROL)
    text = " ".join(ranking.summary().split())
    assert "a lone contrast at the floor cannot pass" in text
    assert "q = 0.086" in text
    assert "at least 2 of the 3 tests" in text


def test_p3_6_a_field_floor_below_the_gate_is_stated_when_the_correction_puts_it_out_of_reach():
    # 4 fields a side shuffled as independent: floor 2/70 = 0.029; 4 tests make q 0.11.
    objects, fovs = _wells(1, shift=8.0)
    objects, fovs = objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")
    rng = np.random.default_rng(2)
    objects["n1"], objects["n2"] = rng.normal(size=len(objects)), rng.normal(size=len(objects))
    ranking = feature_ranking(
        objects,
        fovs,
        features=["x", "y", "n1", "n2"],
        control=CONTROL,
        fields_independent=True,
    )
    text = ranking.summary()
    assert "shuffled so few whole fields" in text
    assert "cannot pass: one at the floor passes only if at least 3 of the 4 tests" in text


# --- P9-1: a constant offset --------------------------------------------------------------


def test_p9_1_a_constant_offset_never_moves_the_p_value():
    objects, fovs = _wells(5, repeats=("r1", "r2"))  # 5 wells a side: the test runs
    p_values = []
    for offset in (0.0, 1e6, 1e10):
        shifted = objects.assign(x=objects["x"] + offset)
        table = condition_contrast(shifted, fovs, feature="x", control=CONTROL, n_draws=5).table
        assert (table["p_value"] >= table["p_floor"]).all()
        p_values.append(table["p_value"].tolist())
        ranking = feature_ranking(shifted, fovs, features=["x"], control=CONTROL)
        assert (ranking.tests["q_value"] > 0).all()
    assert p_values[1] == p_values[0]
    # At 1e10 the values keep about 6 digits of their spread, so near-ties between the 252
    # assignments are counted as ties (the tolerance widens with the offset): p can only grow.
    assert all(far >= near for far, near in zip(p_values[2], p_values[0], strict=True))


# --- P1-10, P1-12, P3-5: wording ----------------------------------------------------------


def test_p1_10_the_contrast_says_it_compares_means_only():
    objects, fovs = _wells(3)
    text = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5).summary()
    assert "compare means: a difference confined to a subpopulation" in " ".join(text.split())


def test_p1_12_small_unit_d_plural_and_rare_declaration_wording():
    objects, fovs = _wells(2, shift=3.0)
    contrast = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    # From 2 well means a side a d can be very large by chance: it is not computed.
    assert contrast.table["d_technical"].isna().all()
    assert "unstable" not in " ".join(contrast.summary().split())
    assert " 2 vs 2 " in contrast.summary(detail="full")
    # A field rule asking for one field says "1 field", not "1 fields".
    spec = [("ctrl", "r1", "A", 4, 10.0), ("drug", "r1", "B", 4, 10.0)]
    objects, fovs = _tables(spec, per_field=200)
    fovs = pd.concat([fovs, fovs.assign(position_id=fovs["position_id"] + "_empty")])
    result = object_rule_check(objects, fovs, feature="x", control=CONTROL, n_draws=20)
    full = result.summary(detail="full")
    assert "fields with cells" in full and "1 fields" not in full
    row = result.table.iloc[0]
    assert f"{int(row['fov_rule_fovs'])} ({int(row['fov_rule_fovs_upper95'])})" in full
    from morphocoverage.objectives import RareTarget

    rare = RareTarget(
        detection_probability=0.9, min_fov_occurrences=2, phenotype="state", fov_incidence=0.2
    )
    setup = Setup("x", rare=rare, rare_label="r")
    rare_line = next(line for line in declaration(setup) if line.startswith("Rare phenotype"))
    assert "by optimise and benchmark" in rare_line


def test_p3_5_a_single_technical_repeat_interval_is_captioned():
    layout = _Layout(
        "condition", "biological_repeat", "technical_repeat", ["a"], ["r1"], False, 4, ()
    )
    title = _title("x", "", layout, "cell", "median", 0.95, clusters=[(1, [1]), (1, [3])])
    assert "with one well the interval shows field-to-field variation only" in title


# --- P9-4: the row order of the tables ------------------------------------------------------


def _sorted(table):
    """A result table in a fixed row order (groups come in order of first appearance)."""
    keys = [
        c for c in ("condition", "biological_repeat", "technical_repeat", "target", "n", "sampling")
    ]
    keys = [c for c in keys if c in table.columns]
    return table.sort_values(keys, kind="stable").reset_index(drop=True)


def test_p9_4_row_order_never_changes_the_contrast_draws_or_unusual_objects():
    objects, fovs = _wells(2, repeats=("r1", "r2"))
    rng = np.random.default_rng(3)
    shuffled_objects = objects.iloc[rng.permutation(len(objects))]
    shuffled_fovs = fovs.iloc[rng.permutation(len(fovs))]
    runs = [
        condition_contrast(o, f, feature="x", control=CONTROL, object_counts=(30,), n_draws=10)
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_frame_equal(runs[0].effects, runs[1].effects)
    pd.testing.assert_frame_equal(runs[0].table, runs[1].table)
    labels = [
        unusual_objects(o, f, feature_cols=["x", "y"], reference=CONTROL, per_fov=5).labels
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_series_equal(labels[0].sort_index(), labels[1].sort_index())
    rules = [
        object_rule_check(o, f, feature="x", control=CONTROL, n_draws=50).table
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_frame_equal(*(_sorted(r) for r in rules))
    spreads = [
        descriptor_spread_check(o, f, feature="x", threshold=40, n_draws=10).curves
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_frame_equal(*(_sorted(r) for r in spreads))
