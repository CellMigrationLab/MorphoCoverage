"""Tests of ``sampling_unit.contrast``: the condition contrast, its units, draws and words."""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import (
    CONTRAST_GROUPS,
    CONTROL,
    CONTROL_CONDITION,
    CONTROL_ONLY,
    _constant_field_means,
    _error,
    _field_rows,
    _five_wells,
    _flat,
    _flat_text,
    _join,
    _signed,
    _technical,
    _three,
    _two_conditions,
    _two_repeats,
    _well_column_tables,
    _well_tables,
    _well_tables_with_empty_fields,
    _wells,
    _without_technical,
)
from _helpers_sampling_unit import _tables as _discovery_tables
from _helpers_workflows import DOCS, READOUT, _acquisition
from scipy import stats

from morphocoverage._style import CAUTION, Status
from morphocoverage._wording import WITHIN_FAMILY
from morphocoverage.plots import plots_of_data
from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit import (
    condition_contrast,
    feature_ranking,
    fov_clustering,
    object_rule_check,
)
from morphocoverage.sampling_unit._groups import contrasts_of
from morphocoverage.sampling_unit._units import (
    ENUMERATION_BUDGET,
    NOT_INDEPENDENT,
    field_means,
    permutation_p,
    separation_d,
    smallest_p,
)
from morphocoverage.sampling_unit.contrast import UNREACHABLE
from morphocoverage.sampling_unit.contrast_text import _no_within_test, contrast_summary
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.reading_text import (
    LIMITED_REPLICATION,
    condition_readings,
    readings_parts,
)
from morphocoverage.sampling_unit.repeat_plan import repeat_plan
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._html_cards import significant_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "lab"))
import catalog

ONE_WELL = "only one technical repeat (well) with observations on one side, no replication"


def _contrast(statuses: list[str], technical: list[int], control: list[int]) -> SimpleNamespace:
    table = pd.DataFrame({"status": statuses, "technical": technical, "control_technical": control})
    return SimpleNamespace(table=table)


def test_no_test_within_a_repeat_gives_the_status_reason_never_one_well_each_a6_c5_b18():
    # idr0028: the control has one well per repeat, the treatments two.
    said = _no_within_test(_contrast([ONE_WELL] * 2, [2, 2], [1, 1]))
    assert said.startswith("the control has one well in each repeat")
    both = _no_within_test(_contrast([ONE_WELL] * 2, [1, 1], [1, 1]))
    assert both.startswith("each condition and its control have one well in each repeat")
    # Two wells a side: the test cannot reach 0.05; no well column: not recorded.
    assert "too few wells a side" in _no_within_test(_contrast([UNREACHABLE], [2], [2]))
    unknown = _no_within_test(_contrast([NOT_INDEPENDENT], [0], [0]))
    assert unknown == "no well recorded, so treated as one well per condition and repeat"
    for text in (said, unknown):
        assert "each condition has one well" not in text


def test_contrasts_are_ordered_by_identity_not_by_set_iteration():
    described = [{"repeat": r, "condition": "drug", "is_control": False} for r in (1, "1", 2, "2")]
    expected = contrasts_of(described)
    for order in ([3, 2, 1, 0], [1, 3, 0, 2]):
        assert contrasts_of([described[i] for i in order]) == expected
    assert len(expected) == 4


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
    objects, fovs = _well_tables(spec, obs_sd=0.0)
    result = feature_ranking(objects, fovs, features=["x"], control=CONTROL, n_permutations=1999)
    row = result.tests.iloc[0]
    assert np.isinf(row["d_fields"])
    # 6 assignments; the observed one and its mirror are both perfect: p could not fall below
    # 2/6, above the gate, so the ranking does not run a test that cannot help.
    assert row["p_floor"] == pytest.approx(1 / 3) and np.isnan(row["p_value"])
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
    objects, fovs = _well_tables(spec, per_field=15)
    common = {"n_permutations": 99, "control": CONTROL}
    contrast = condition_contrast(
        objects, fovs, feature="x", n_draws=2, object_counts=(10,), **common
    )
    row = contrast.table.iloc[0]
    assert np.isnan(row["p_value"]) and "only one technical repeat" in row["status"]
    assert "the fields of one well are not independent replicates" in " ".join(
        contrast.summary().split()
    )
    objects["y"] = objects["x"] * 2
    ranking = feature_ranking(objects, fovs, features=["x", "y"], **common)
    assert ranking.tests["p_value"].isna().all() and not ranking.tests["passes_gate"].any()
    text = ranking.summary()
    assert "Not ranked because no contrast could be tested (2 features:" in text
    assert "No contrast passes the gate" not in text
    # Without the technical-repeat column among the groups it is still the unit (audit 3,
    # H-1): the single treated well gets no test either.
    pooled = condition_contrast(
        objects,
        fovs,
        feature="x",
        n_draws=2,
        object_counts=(10,),
        group_cols=["condition", "biological_repeat"],
        **common,
    )
    assert "only one technical repeat" in pooled.table["status"].iloc[0]
    assert np.isnan(pooled.table["p_value"].iloc[0])


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
    assert "An infinite d is a perfect separation: no variation within either side" in " ".join(
        contrast.summary().split()
    )
    assert "inf" in contrast.summary(detail="full")


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
        objects.drop(columns=["biological_repeat", "repeat_group"], errors="ignore"),
        fovs.drop(columns=["biological_repeat", "repeat_group"], errors="ignore"),
        group_cols=["condition", "technical_repeat"],
        **contrast,
    )
    assert alone.settings["repeat_col"] is None and len(alone.table) == 1


def test_h18_a_control_naming_an_absent_column_raises_clearly():
    objects, fovs = _well_tables([("ctrl", "r1", "w1", 4, 0, 0.2), ("drug", "r1", "w2", 4, 1, 0.2)])
    control = {"condition": "ctrl", "plate": "P1"}
    with pytest.raises(ValueError, match=r"control names columns absent .*\['plate'\]"):
        condition_contrast(objects, fovs, feature="x", control=control, n_permutations=9)
    with pytest.raises(ValueError, match=r"control names columns absent .*\['plate'\]"):
        feature_ranking(objects, fovs, features=["x"], control=control, n_permutations=9)


def test_h1_technical_repeats_are_the_unit_whatever_group_cols_say():
    # Three wells a side, no condition effect, wells far apart: shuffling fields as if
    # independent passed a well-to-well shift for a condition effect (42% at p <= 0.05).
    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        spec = [("ctrl", "r1", f"c{i}", 5, rng.normal(0, 1.0), 0.2) for i in range(3)]
        spec += [("drug", "r1", f"d{i}", 5, rng.normal(0, 1.0), 0.2) for i in range(3)]
        objects, fovs = _well_tables_with_empty_fields(spec, seed=seed, per_field=10)
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


def test_h10_whole_field_draws_count_acquired_empty_fields():
    spec = [("ctrl", "r1", "w", 40, 10.0, 0.1), ("drug", "r1", "w", 40, 11.0, 0.1)]
    objects, fovs = _without_technical(
        *_well_tables_with_empty_fields(spec, per_field=10, empty_every=2)
    )
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, object_counts=[100], n_draws=20
    )
    row = result.effects[result.effects["sampling"] == "whole FOVs"].iloc[0]
    # Half the 40 fields are empty: 100 objects take about 19-20 acquired fields, not 10.
    assert 17 < row["fields_treated"] < 21


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
    objects, fovs = _well_tables_with_empty_fields(spec, per_field=10)
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


def test_h19_an_untested_contrast_keeps_its_effect_sizes():
    spec = [
        ("ctrl", "r1", "c0", 10, 0.0, 0.2),
        ("drug", "r1", "d0", 5, 1.0, 0.2),
        ("drug", "r1", "d1", 5, 1.0, 0.2),
    ]
    objects, fovs = _well_tables_with_empty_fields(spec, per_field=30)
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


def _groups(well_col):
    return ["condition", "biological_repeat", well_col]


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
    objects, fovs = _well_column_tables(spec, well_col="Metadata_Well")
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


def test_p1_1_contrasts_passing_with_opposite_signs_are_flagged():
    spec = []
    for repeat, drug in (("r1", 14.0), ("r2", 6.0)):
        # 5 wells a side: a floor of 2/252, so both contrasts can pass after Bonferroni.
        spec += [("ctrl", repeat, f"c{i}", 3, 10.0) for i in range(5)]
        spec += [("drug", repeat, f"d{i}", 3, drug) for i in range(5)]
    objects, fovs = _well_column_tables(spec)
    result = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    assert (result.table["p_value"] <= 0.05).all()
    assert "in opposite directions" in result.summary()
    ranking = feature_ranking(objects, fovs, features=["x"], control=CONTROL)
    assert "passes the gate for drug in opposite directions" in ranking.summary()


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


def test_p1_10_the_contrast_says_it_compares_means_only():
    objects, fovs = _wells(3)
    text = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5).summary()
    assert "compare means: a difference confined to a subpopulation" in " ".join(text.split())


def test_a7_the_main_test_reads_one_scale_per_feature():
    from morphocoverage.sampling_unit.discovery import feature_discovery

    # A7: b's values below 0 put area on differences for every condition, a included.
    objects, fovs = _signed(offset_b=-1.0)
    tests = feature_discovery(
        objects,
        fovs,
        features=["area", "positive"],
        control={"condition": "ctrl"},
        n_permutations=9,
        fov_col="image_file",
    ).tests.set_index(["feature", "condition"])
    assert set(tests.loc["area", "paired_values"]) == {"raw"}
    assert set(tests.loc["positive", "paired_values"]) == {"normalised"}
    means = objects.groupby(["condition", "biological_repeat"])["area"].mean()
    controls = means["ctrl"].to_numpy()
    expected = stats.ttest_1samp(means["a"].to_numpy() - controls, 0.0).pvalue
    assert np.isclose(tests.loc[("area", "a"), "p_paired"], expected)
    # The contrast of the readout, its plan and its words read the same scale.
    from morphocoverage.sampling_unit.contrast import condition_contrast

    contrast = condition_contrast(
        objects,
        fovs,
        feature="area",
        control={"condition": "ctrl"},
        n_permutations=9,
        n_draws=5,
        fov_col="image_file",
    )
    assert contrast.settings["log_scale"] is False
    assert contrast.repeats["p_paired_t_normalised"].isna().all()
    assert contrast.repeats["fold_change"].isna().all()
    assert set(contrast.plan["planned_on"]) <= {"difference", "d"}


COMMON = {
    "feature": "x",
    "control": {"condition": "control"},
    "group_cols": CONTRAST_GROUPS,
    "n_permutations": 100,
    "n_draws": 10,
    "object_counts": (50,),
}


def test_g3_bonferroni_counts_the_contrasts_tested():
    # Five wells a side, so the test within a repeat can reach p <= 0.05 and is run.
    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3, n_wells=5)
    kept = fovs[~((fovs["biological_repeat"] == "R2") & (fovs["condition"] == "control"))]
    result = condition_contrast(
        objects[objects["image_file"].isin(kept["image_file"])], kept, **COMMON
    )
    assert "Bonferroni over 1 comparison:" in " ".join(result.summary(detail="full").split())
    assert "Bonferroni-corrected over 1 comparison." in " ".join(result.summary().split())


def test_g4_the_jcs_line_is_kept_without_a_repeat_column():
    objects, fovs = _two_conditions(shift=1.0, field_sd=0.3)
    one = objects[objects["biological_repeat"] == "R1"].drop(
        columns=["biological_repeat", "repeat_group"], errors="ignore"
    )
    one_fovs = fovs[fovs["biological_repeat"] == "R1"].drop(
        columns=["biological_repeat", "repeat_group"], errors="ignore"
    )
    result = condition_contrast(one, one_fovs, **{**COMMON, "group_cols": ["condition", "well"]})
    assert "JCS effect size" in result.summary()


def test_g5_an_undefined_field_d_gets_no_p_value():
    objects, fovs = _constant_field_means()
    # Every field holds 9 and 11, so every field mean is 10 on both sides (a perfect
    # separation, treated means 11 against 10, is an infinite d instead: H-6).
    above = objects["x"] > objects.groupby("condition")["x"].transform("mean")
    objects["x"] = np.where(above, 11.0, 9.0)
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control={"condition": "control"},
        group_cols=["condition"],
        n_permutations=99,
        n_draws=5,
        object_counts=(4,),
        fields_independent=True,
    )
    row = result.table.iloc[0]
    assert np.isnan(row["p_value"]) and "do not vary" in row["status"]


def test_contrast_summary_without_any_object_count_that_fits():
    from morphocoverage.sampling_unit import condition_contrast

    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    result = condition_contrast(
        objects, fovs, feature=READOUT, control={"condition": "control"}, n_permutations=20
    )
    # The effect size from every object is still given (audit 3, H-19), with no draws.
    assert result.effects.empty and "from 50 objects" not in result.summary()
    # Across repeats the short form leaves it to the full one's table.
    assert "over control SD, every cell" in result.summary(detail="full")


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


@pytest.fixture(scope="module")
def screen():
    """The lab's filopodia screen: 3 biological repeats, one coverslip a side and repeat."""
    data = catalog.load("cct8_filopodia_screen")
    return condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )


def test_the_report_says_once_that_the_false_discovery_rate_is_within_each_family(screen):
    joined = " ".join(screen.summary().split())
    assert joined.count(WITHIN_FAMILY) == 1
    docs = (DOCS / "common" / "what_happens.md").read_text()
    assert "controls the false discovery rate within each family" in " ".join(docs.split())


def test_one_well_per_condition_is_no_within_repeat_technical_replication(screen):
    joined = " ".join(screen.summary().split())
    said = "its control have one well in each repeat, no within-repeat technical replication"
    assert said in joined


def test_three_repeats_show_each_repeats_change_and_the_interval_once_said_limited(screen):
    row = screen.repeats.set_index("condition").loc["siCCT8#5"]
    assert row["repeats"] == 3 and np.isfinite(row["ci_low"]) and row["ci_low"] < row["ci_high"]
    assert row["repeat_contrasts"].count(": ") == 3
    joined = " ".join(screen.summary().split())
    assert "in the 3 repeats, ×0.56 overall, 95% t interval ×0.35 to ×0.917" in joined
    assert joined.count("very limited biological replication") == 1
    assert LIMITED_REPLICATION in joined


def test_more_repeats_show_no_limited_replication_line():
    objects, fovs = _discovery_tables(n_repeats=4)
    contrast = condition_contrast(
        objects, fovs, feature="area", control={"condition": "ctrl"}, n_permutations=99
    )
    joined = " ".join(contrast.summary().split())
    assert "very limited biological replication" not in joined
    assert "in the 3 repeats" not in joined
    assert np.isfinite(contrast.repeats["ci_low"]).all()  # the CSV carries it at any n


def test_a_declared_ratio_on_a_value_that_is_not_positive_is_refused():
    objects, fovs = _discovery_tables(n_repeats=3)
    objects.loc[0, "area"] = -1.0
    with pytest.raises(ValueError, match="declared on the ratio scale.*not positive"):
        condition_contrast(
            objects, fovs, feature="area", control={"condition": "ctrl"}, scale="ratio"
        )
    setup = Setup(readout="area", control={"condition": "ctrl"}, scales={"area": "ratio"})
    with pytest.raises(ValueError, match="'area' is declared on the ratio scale"):
        report(objects, fovs, setup)


def test_a_declared_difference_replaces_the_automatic_ratio():
    objects, fovs = _discovery_tables(n_repeats=3)
    automatic = condition_contrast(objects, fovs, feature="area", control={"condition": "ctrl"})
    declared = condition_contrast(
        objects, fovs, feature="area", control={"condition": "ctrl"}, scale="difference"
    )
    assert automatic.repeats["fold_change"].notna().all()
    assert declared.repeats["fold_change"].isna().all()
    assert declared.repeats["q_paired_t_normalised"].isna().all()
    joined = " ".join(declared.summary().split())
    assert "the differences from each repeat's control, as declared in the Setup" in joined
    assert "the raw q: the feature is declared on the difference scale" in joined
    discovery = feature_discovery(
        objects,
        fovs,
        features=["area", "noise"],
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
        scales={"area": "difference"},
    )
    tests = discovery.tests.set_index("feature")
    assert tests.loc["area", "paired_values"] == "raw"
    assert tests.loc["noise", "paired_values"] == "normalised"  # the automatic rule
    assert "declared in the Setup: area, the difference" in " ".join(discovery.summary().split())


def _contrast_edges():
    from morphocoverage.sampling_unit import condition_contrast

    objects, fovs = _join(
        _field_rows("ctrl", "r1", 3, 5, 1.0, 0), _field_rows("ctrl", "r2", 3, 5, 1.0, 1)
    )
    error = _error(
        lambda: condition_contrast(objects, fovs, feature=["x"], control={"condition": "ctrl"})
    )
    result = condition_contrast(objects, fovs, feature="x", control={"condition": "ctrl"})
    return error, result.settings["reason"], result.table.shape, result.effects.shape


def _draw_rows_missing_sampling():
    from morphocoverage.sampling_unit.contrast_text import _draw_rows

    table = pd.DataFrame(
        {"biological_repeat": ["r1"], "condition": ["A"], "control_condition": ["c"]}
    )
    effects = pd.DataFrame(
        [
            {
                "biological_repeat": "r1",
                "condition": "A",
                "n_objects": 50,
                "sampling": "random objects",
                "median": 0.5,
                "q25": 0.25,
                "q75": 0.75,
                "same_sign": 0.9,
                "fields_treated": np.nan,
                "resampled": None,
            },
            {
                "biological_repeat": "r1",
                "condition": "A",
                "n_objects": 100,
                "sampling": "whole FOVs",
                "median": 0.4,
                "q25": 0.2,
                "q75": 0.6,
                "same_sign": np.nan,
                "fields_treated": 4.5,
                "resampled": "technical_repeat",
            },
        ]
    )
    contrast = SimpleNamespace(
        table=table, effects=effects, settings={"repeat_col": "biological_repeat"}
    )
    return _draw_rows(contrast)


EXPECTED_CONTRAST_EDGES = (
    ("TypeError", "feature must be one column name"),
    ("Every analysis group is the control, ctrl: no other condition to compare."),
    (0, 0),
    (0, 0),
)


EXPECTED_DRAW_ROWS_MISSING_SAMPLING = (
    [
        ["A vs c, repeat r1", "50", "0.50 (0.25 to 0.75), 90%", "–", "–", "fields only"],
        ["A vs c, repeat r1", "100", "–", "0.40 (0.20 to 0.60)", "about 4.5", "technical_repeat"],
    ],
    ["contrast", "cells", "random cells", "whole fields", "fields", "resampled"],
)


def test_contrast_edges():
    assert _contrast_edges() == EXPECTED_CONTRAST_EDGES


def test_draw_rows_missing_sampling():
    assert _draw_rows_missing_sampling() == EXPECTED_DRAW_ROWS_MISSING_SAMPLING


def test_the_plan_gives_the_verdicts_change_over_every_paired_repeat():
    objects, fovs = _three({"drug": 12.0})
    contrast = condition_contrast(
        objects, fovs, feature="x", control={"condition": "ctrl"}, observation_unit="cell"
    )
    [plan] = contrast.plan.to_dict("records")
    [across] = contrast.repeats.to_dict("records")
    assert plan["change"] == pytest.approx(across["fold_change"])


def test_a_condition_without_the_control_is_named_once(designs):
    contrast, discovery = designs["E4"]
    line = "Not compared: cond3 (no biological repeat holds both it and ctrl)."
    reading = _flat(readings_parts(contrast))
    assert reading.count(line) == 1
    assert " ".join(discovery.summary().split()).count(line) == 1
    summary = " ".join(contrast_summary(contrast, "full").split())
    assert "No well-level test within a repeat" not in summary
    assert summary.count("Not compared: cond3") == 1

    class Result:
        pass

    result = Result()
    result.discovery, result.results = discovery, {}
    assert significant_summary(result, str).count(line) == 1


def test_a_repeat_without_the_control_is_said_where_the_repeats_are(designs):
    contrast, _ = designs["E5"]
    reading = condition_readings(contrast).set_index("condition")["reading"]
    assert "3 of its 4 biological repeats hold ctrl; 2026_01_26 left out" in reading["cond1"]
    assert "left out" not in reading["cond2"]
    assert "No well-level test" not in contrast_summary(contrast, "full")


def test_a_control_only_repeat_is_said_in_the_balance_line(designs):
    contrast, _ = designs["E4"]
    assert contrast.settings["control_only_repeats"] == [CONTROL_ONLY]
    text = " ".join(contrast_summary(contrast, "full").split())
    assert f"The ctrl of biological repeat {CONTROL_ONLY}, which holds no other condition" in text


def test_counts_are_each_conditions_own(designs):
    contrast, discovery = designs["E2"]
    text = " ".join(contrast_summary(contrast, "full").split())
    assert "Across the biological repeats, 1 to 4 per condition (each repeat's value" in text
    assert "With 2 and 1 biological repeats holding both sides, cond2 and cond4 are" in text
    assert "the control as provided is pooled from biological repeats the condition was not" in (
        text
    )
    assert "(1, 2 and 3 of them)" in text

    class Result:
        pass

    result = Result()
    result.discovery, result.results = discovery, {}
    words = significant_summary(result, str)
    assert "significant across the biological repeats, 1 to 4 per condition" in words


def test_one_repeat_is_read_as_one_repeat(designs):
    contrast, _ = designs["E1"]
    reading = _flat(readings_parts(contrast))
    assert "Significant across repeats:" not in reading
    assert "in one biological repeat: the direction and Cohen's d of that repeat" in reading
    assert "Cohen's d +" in reading and "per repeat, mean" not in reading
    text = " ".join(contrast_summary(contrast, "full").split())
    assert "each of 1 repeat" not in text and "the fields a side the one repeat needs" in text


def test_the_contrasts_are_one_table_each_caveat_once_and_every_number_in_full():
    objects, fovs = _five_wells()
    result = condition_contrast(
        objects, fovs, feature="x", control=CONTROL_CONDITION, n_permutations=400, n_draws=5
    )
    short = result.summary()
    header = r"contrast\s+fields\s+cells\s+tech\.\s+d on cells\s+d on field means"
    assert re.search(header, short)
    assert _flat_text(short).count("compare means") == 1
    assert _flat_text(short).count("drawn with replacement") == 1
    full = result.summary(detail="full")
    for row in result.table.to_dict("records"):
        for column in ("d_objects", "d_fields", "d_technical", "effect_full"):
            assert f"{row[column]:.2f}" in full
        assert f"p = {row['p_value']:.3g}" in full
        assert f"{int(row['distinct_assignments'])} distinct assignments" in full
    for row in result.effects.to_dict("records"):
        assert f"{row['median']:.2f} ({row['q25']:.2f} to {row['q75']:.2f})" in full
    assert full.count("compare means") == 1
    assert conditions_status(result, None)[0] is Status.OK


def test_a_contrast_the_design_cannot_test_is_left_out_said_once_not_flagged():
    # One technical repeat a side and two biological repeats: no test within or across
    # repeats. That is a usual pilot design, so the answer is "not tested", not a warning.
    objects, fovs = _technical(*_two_conditions(shift=2.0, field_sd=0.3, n_wells=1))
    result = condition_contrast(objects, fovs, feature="x", control=CONTROL_CONDITION, n_draws=5)
    status, headline = conditions_status(result, None)
    assert status is Status.NOT_ASSESSED
    assert "no test (one well on a side)" in headline
    short = result.summary()
    assert " ".join(short.split()).count("only one technical repeat") == 1
    assert "no test" in short and CAUTION not in short
