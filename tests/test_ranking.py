"""Tests of ``sampling_unit.ranking``: the features ranked by how they separate the conditions."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import (
    CONTROL,
    CONTROL_CONDITION,
    _constant_field_means,
    _error,
    _field_rows,
    _five_wells,
    _join,
    _technical,
    _two_conditions,
    _well_tables,
    _well_tables_with_empty_fields,
    _wells,
    _without_technical,
)
from _helpers_tables import FOVS, OBJECTS, _quiet

from morphocoverage.sampling_unit import condition_contrast, feature_ranking, fov_variability
from morphocoverage.sampling_unit._units import separation_d


@pytest.mark.parametrize("bad", ["xy", {"x", "y"}, {"x": 1}], ids=["string", "set", "mapping"])
def test_the_sampling_unit_feature_selections_refuse_what_they_cannot_read(bad):
    from morphocoverage.sampling_unit import feature_ranking, unusual_objects

    control = {"condition": "ctrl"}
    with pytest.raises(TypeError, match="column names"):
        _quiet(lambda: feature_ranking(OBJECTS, FOVS, features=bad, control=control))
    with pytest.raises(TypeError, match="column names"):
        _quiet(
            lambda: unusual_objects(
                OBJECTS, FOVS, feature_cols=bad, reference=control, fov_col="position_id"
            )
        )


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
    t_objects, t_fovs = _well_tables([("drug", "r1", "w2", 6, 0, 0)], per_field=[3, 7])
    c_objects, c_fovs = _well_tables([("ctrl", "r1", "w1", 6, 0, 0)], per_field=[3, 7, 11])
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
    objects, fovs = _well_tables(spec, seed=2)
    result = feature_ranking(objects, fovs, features=["x"], control=CONTROL, n_permutations=1999)
    # 20 assignments of 3 wells a side; the split and its mirror: a floor of 2/20, above the
    # gate, so the test cannot help and is not run.
    assert result.tests["p_floor"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(result.tests["p_value"].iloc[0])
    text = " ".join(result.summary().split())
    assert "None is ranked" in text and "so it is not run" in text


def test_h15_untested_features_are_listed_as_untested():
    spec = [
        (c, "r1", f"{c}{i}", 5, 2.0 if c == "drug" else 0, 0.2)
        for c in ("ctrl", "drug")
        for i in range(3)
    ]
    objects, fovs = _well_tables(spec)
    empty = {"condition": "drug", "biological_repeat": "r1", "technical_repeat": "drug0"}
    fovs = pd.concat([fovs, pd.DataFrame([{**empty, "position_id": "empty"}])], ignore_index=True)
    objects["y"] = np.where(objects["condition"] == "drug", np.nan, objects["x"])
    objects["w"] = np.random.default_rng(0).normal(size=len(objects))
    # Fields compared as independent draws (audit 4 Q1): the wells are left out of the
    # groups and the unit, and the fields declared independent.
    common = {
        "control": CONTROL,
        "technical_col": None,
        "group_cols": ["condition", "biological_repeat"],
        "fields_independent": True,
    }
    result = feature_ranking(objects, fovs, features=["x", "y", "w"], n_permutations=199, **common)
    text = " ".join(result.summary().split())
    assert "No contrast passes the gate for 1 feature: w." in text
    assert (
        "Not ranked because no contrast could be tested (1 feature: y): fewer than 2 fields "
        "with observations on one side."
    ) in text
    contrast = condition_contrast(
        objects, fovs, feature="x", n_permutations=99, n_draws=5, object_counts=(10,), **common
    )
    x_row = result.tests[result.tests["feature"] == "x"].iloc[0]
    assert x_row["fields"] == contrast.table["fields"].iloc[0] == 16
    assert x_row["control_fields"] == contrast.table["control_fields"].iloc[0]


def test_h7_a_field_test_that_cannot_pass_the_gate_is_stated():
    spec = [("ctrl", "r1", "w", 3, 0.0, 0.1), ("drug", "r1", "w", 3, 5.0, 0.1)]
    objects, fovs = _without_technical(*_well_tables_with_empty_fields(spec, per_field=30))
    objects["y"] = objects["x"] * 2
    result = feature_ranking(
        objects, fovs, features=["x", "y"], control=CONTROL, fields_independent=True
    )
    assert result.tests["p_floor"].tolist() == pytest.approx([0.1, 0.1])
    # A floor of 0.1 is above the gate: the tests cannot help, so they are not run.
    assert result.tests["p_value"].isna().all()
    text = " ".join(result.summary().split())
    assert "None is ranked" in text and "so it is not run" in text


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


def test_g6_a_perfect_separation_ranks_first_at_the_cap():
    objects, fovs = _constant_field_means()
    result = feature_ranking(
        objects,
        fovs,
        features=["x"],
        control={"condition": "control"},
        group_cols=["condition"],
        n_permutations=199,
        fields_independent=True,
    )
    assert result.tests["status"].iloc[0] == "ok"
    assert result.ranking["score"].iloc[0] == 20.0 and result.top() == ["x"]


def test_g7_the_gate_sentence_follows_the_floor():
    objects, fovs = _technical(
        *_two_conditions(shift=2.0, field_sd=0.3, seed=7, n_wells=4, n_fields=6)
    )
    text = feature_ranking(
        objects, fovs, features=["y", "x"], control={"condition": "control"}, n_permutations=400
    ).summary()
    # 70 assignments, 4 wells a side: the split and its mirror give a floor of 2/70,
    # under the gate; corrected over the 4 tests a lone contrast there gets q = 0.11
    # (audit 4 D4), which the sentence now says.
    assert "cannot fall below 0.029;" in text and "cannot pass the gate" not in text
    assert "a lone contrast at the floor cannot pass" in text


def _discovery_and_ranking_errors():
    from morphocoverage.sampling_unit import feature_discovery, feature_ranking

    objects, fovs = _join(
        _field_rows("ctrl", "r1", 3, 5, 1.0, 0), _field_rows("A", "r1", 3, 5, 2.0, 1)
    )
    control = {"condition": "ctrl"}
    out = []
    for function in (feature_discovery, feature_ranking):
        out.append(_error(lambda f=function: f(objects, fovs, features=[], control=control)))
        out.append(
            _error(lambda f=function: f(objects, fovs, features=["x"], control=control, alpha=1.5))
        )
    only, only_fovs = _field_rows("ctrl", "r1", 3, 5, 1.0, 0)
    ranked = feature_ranking(only, only_fovs, features=["x"], control=control)
    out.append((ranked.settings["reason"], ranked.tests.shape, ranked.ranking.shape))
    return out


EXPECTED_DISCOVERY_AND_RANKING_ERRORS = [
    ("ValueError", "features must name at least one column"),
    ("ValueError", "alpha must be between 0 and 1"),
    ("ValueError", "features must name at least one column"),
    ("ValueError", "alpha must be between 0 and 1"),
    (("Every analysis group is the control, ctrl: no other condition to compare."), (0, 0), (0, 0)),
]


def test_discovery_and_ranking_errors():
    assert _discovery_and_ranking_errors() == EXPECTED_DISCOVERY_AND_RANKING_ERRORS


def test_the_ranking_is_a_table_of_the_best_features_with_its_caveats_once():
    objects, fovs = _five_wells()
    ranking = feature_ranking(
        objects, fovs, features=["y", "x"], control=CONTROL_CONDITION, n_permutations=400
    )
    short = ranking.summary()
    assert "1 of 2 features pass the gate" in short
    assert re.search(r"rank\s+feature\s+\|d\|\s+best contrast\s+q\s+passes in", short)
    assert short.count("distinct assignments") == 1
    full = ranking.summary(detail="full")
    assert "Ranking within repeats (wells as the unit): each of the 2 declared features" in full
    best = ranking.ranking.iloc[0]
    assert f"{best['score']:.2f}" in full and f"{best['q_value']:.3g}" in full
    assert "No contrast passes the gate for 1 feature: y." in full
