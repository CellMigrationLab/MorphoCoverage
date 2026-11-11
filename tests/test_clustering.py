"""Tests of ``sampling_unit.clustering``: whether a state clusters in a few fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import (
    CONTROL,
    _error,
    _field_rows,
    _flat_text,
    _well_tables,
    _well_tables_with_empty_fields,
)

from morphocoverage._wording import percent
from morphocoverage.objectives import RareTarget
from morphocoverage.sampling_unit import feature_ranking, fov_clustering
from morphocoverage.statistics import first_detection_fovs, fovs_to_detect
from morphocoverage.workflows import Setup
from morphocoverage.workflows._shared import rare_clustering


def test_h7_unlabelled_fields_are_not_fields_without_the_state():
    objects, fovs = _well_tables([("ctrl", "r1", "w1", 40, 0, 0)], per_field=20)
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
        assert f"none of the {words} of its 2 fields scored" in text
        assert "patche " not in text


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
    ranking = feature_ranking(
        *_well_tables_with_empty_fields(spec), features=["x"], control=CONTROL
    )
    assert "over the 0 tests" not in ranking.summary(detail="full")
    assert "though no contrast could be tested" in " ".join(ranking.summary(detail="full").split())
    assert "no contrast could be tested" in " ".join(ranking.summary().split())
    import morphocoverage.sampling_unit as package

    for name in ("clustering", "contrast", "object_rules", "ranking", "variability"):
        module = getattr(package, name)
        assert all(hasattr(module, public) for public in module.__all__)


def _fields(labels_per_fov, condition="drug", repeat="R1", well="A"):
    """Objects with a boolean label, one list of labels per field, and the FOV table."""
    objects, fovs = [], []
    for j, labels in enumerate(labels_per_fov):
        name = f"{condition}_{repeat}_{well}_f{j:02d}"
        identity = {
            "condition": condition,
            "biological_repeat": repeat,
            "repeat_group": repeat,
            "technical_repeat": well,
        }
        fovs.append({**identity, "position_id": name})
        for value in labels:
            objects.append(
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "technical_repeat": well,
                    "position_id": name,
                    "state": value,
                    "x": 0.0,
                }
            )
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _concat(*tables):
    return tuple(pd.concat(parts, ignore_index=True) for parts in zip(*tables, strict=True))


def test_p6_1_a_field_holds_an_unusual_state_only_with_m_objects():
    # A 15-object state in 1 of 30 drug fields, and one stray "unusual" object in
    # 6 other drug fields and in 5 of 30 control fields, as unusual_objects flags
    # a share of ordinary objects by construction.
    drug = [[False] * 20 for _ in range(30)]
    drug[11] = [True] * 15 + [False] * 20
    for j in (1, 4, 9, 17, 22, 28):
        drug[j] = [True] + [False] * 19
    control = [[False] * 20 for _ in range(30)]
    for j in (2, 7, 13, 19, 25):
        control[j] = [True] + [False] * 19
    objects, fovs = _concat(_fields(drug), _fields(control, condition="control"))
    # 99 shuffles: with 9, no p could reach 0.05 and the test is left out (audit 10).
    plain = fov_clustering(objects, fovs, label_col="state", states=[True], n_permutations=99)
    held = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=[True],
        n_permutations=99,
        min_objects_per_fov=3,
        reference={"condition": "control"},
    )
    before = plain.table.set_index("condition").loc["drug"]
    after = held.table.set_index("condition").loc["drug"]
    # Before: 7 of 30 fields "hold" it, so a handful of fields seem enough.
    assert before["fovs_needed"] == first_detection_fovs(7 / 30, 0.95) < 15
    # After: only the field with the state holds it, and the truth (89) is given.
    assert after["fovs_holding_state"] == 1
    assert after["fovs_needed"] == first_detection_fovs(1 / 30, 0.95) == 89
    # The clustering test still counts every field with one of the objects.
    assert after["fovs_with_state"] == 7 and after["p_clustered"] == before["p_clustered"]
    # The reference groups' own incidence is printed beside it.
    assert after["reference_fovs"] == 30 and after["reference_fovs_holding"] == 0
    text = held.summary()
    assert "holding ≥3" in text and "1 of 30" in text
    assert "reference holding" in held.summary(detail="full")
    assert "0 of 30 (0%)" in text
    assert "89 (up to" in text


def test_p6_5_rare_state_fields_report_the_technical_repeats_holding_it():
    well_a = [[True] + [False] * 9 if j < 5 else [False] * 10 for j in range(10)]
    empty = [[False] * 10 for _ in range(10)]
    objects, fovs = _concat(_fields(well_a), _fields(empty, well="B"), _fields(empty, well="C"))
    result = fov_clustering(objects, fovs, label_col="state", states=[True], n_permutations=9)
    assert set(result.table["technical_repeats"]) == {1}
    text = result.summary()
    assert "drug · repeat R1" in text and "1 of 3 technical repeats" in text
    assert "The field counts assume fields from independent technical repeats" in text
    pooled = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=[True],
        n_permutations=9,
        group_cols=["condition", "biological_repeat"],
    )
    row = pooled.table.iloc[0]
    assert (row["technical_repeats"], row["technical_repeats_holding"]) == (3, 1)


def test_p1_5_the_rare_field_plan_uses_min_fov_occurrences():
    labels = [[True, False] if j < 2 else [False, False] for j in range(8)]
    objects, fovs = _fields(labels)
    result = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=[True],
        detection_probability=0.9,
        min_fov_occurrences=2,
        n_permutations=9,
    )
    assert result.table.iloc[0]["fovs_needed"] == fovs_to_detect(0.25, 0.9, 2) == 15
    text = result.summary()
    assert "to see in ≥2, 90%" in text and " 15 (up to" in text
    rare = RareTarget(
        detection_probability=0.9, min_fov_occurrences=2, phenotype="state", fov_incidence=0.25
    )
    setup = Setup("x", rare=rare, rare_label="state")
    workflow = rare_clustering(objects, fovs, setup)
    assert workflow.table.iloc[0]["fovs_needed"] == 15


def _clustering_errors():
    from morphocoverage.sampling_unit import fov_clustering

    objects, fovs = _field_rows("A", "r1", 4, 3, 1.0, 0)
    objects["state"] = ["a", "b"] * 6
    calls = [
        {"label_col": "missing"},
        {"label_col": "state", "detection_probability": 1.5},
        {"label_col": "state", "states": "a"},
        {"label_col": "state", "states": []},
        {"label_col": "state", "states": ["a", "a"]},
    ]
    return [_error(lambda c=c: fov_clustering(objects, fovs, **c)) for c in calls]


EXPECTED_CLUSTERING_ERRORS = [
    ("ValueError", "label_col 'missing' is not a column of objects"),
    ("ValueError", "detection_probability and confidence must be between 0 and 1"),
    ("ValueError", "states must be a list of 'state' values, such as ['rare']"),
    ("ValueError", "states must be a list of 'state' values, such as ['rare']"),
    ("ValueError", "states names a state twice: ['a', 'a']"),
]


def test_clustering_errors():
    assert _clustering_errors() == EXPECTED_CLUSTERING_ERRORS


def test_clustering_short_answers_per_state_and_full_keeps_every_number(states):
    clustering, _ = states
    short = _flat_text(clustering.summary())
    assert "Are the state cells spread over the fields?" in short
    assert "in 1 of 2 groups where tested (MOCK · repeat r1)" in short
    full = clustering.summary(detail="full")
    for row in clustering.table[clustering.table["state_objects"] > 0].to_dict("records"):
        for value in (
            percent(row["prevalence"]),
            f"{row['expected_fovs_if_random']:.1f}",
            f"{row['clustering_ratio']:.1f}",
            f"{row['p_clustered']:.2g}",
            f"{int(row['fovs_with_state'])} of {int(row['fovs'])}",
        ):
            assert value in full
