"""The owner's decisions of 2026-10-06 on the other feature tables, after audit B: novelty on
the explainable features, populations per space, a fingerprint per table, MFA weights, the
separation by repeat, the centred view, and the clear fixes (findings 2-16)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from test_feature_tables import _tables

from morphocoverage._integrity_tables import integrity_by_table
from morphocoverage.embedding import _standardised
from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.integrity import integrity_checks
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.sampling_unit.populations import Populations, find_populations
from morphocoverage.separation import Separation, separation
from morphocoverage.workflows import Setup
from morphocoverage.workflows._enough_targets import fingerprint_lines, fingerprint_precision
from morphocoverage.workflows._feature_space import centred_tables, features_words
from morphocoverage.workflows._guided_tables import changed, entries
from morphocoverage.workflows._html_map_text import comparison_table, share_lines
from morphocoverage.workflows._population_spaces import (
    _merged,
    found_words,
    populations_in_spaces,
)
from morphocoverage.workflows._populations import PopulationsResult, populations_text
from morphocoverage.workflows.guided import Answers, rough_runtime

KEY = ["condition", "biological_repeat", "image_file", "label_id"]


def _cells(
    fields: int = 4,
    cells: int = 15,
    conditions: tuple[str, ...] = ("ctrl", "a", "b"),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.random.Generator]:
    """Conditions in three repeats, ``fields`` fields of ``cells`` labelled cells each."""
    rng = np.random.default_rng(seed)
    rows = [
        {"condition": c, "biological_repeat": r, "image_file": f"{c}_{r}_{f}.tif", "label_id": k}
        for c in conditions
        for r in ("R1", "R2", "R3")
        for f in range(fields)
        for k in range(1, cells + 1)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    return objects, fovs, rng


def _table(objects: pd.DataFrame, values: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = {f"{prefix}{j}": values[:, j] for j in range(values.shape[1])}
    return pd.concat([objects[KEY].reset_index(drop=True), pd.DataFrame(columns)], axis=1)


# --- D2: populations in each space, merged -------------------------------------------------


def _found(index: pd.Index, members: dict[str, tuple[range, str]]) -> Populations:
    """A ``Populations`` with these groups (cells by position) of these kinds."""
    labels = pd.Series([None] * len(index), index=index, dtype=object)
    rows, kept = [], {}
    for name, (cells, kind) in members.items():
        inside = index[list(cells)]
        labels[inside] = name
        kept[name] = (inside, index)
        rows.append(
            {
                "population": name,
                "kind": kind,
                "observations": len(inside),
                "share": len(inside) / len(index),
                "defining": [("f1", 2.0)],
                "defined_by": "high f1 (d +2.0)",
                "rule": "",
            }
        )
    settings = {
        "features": ["f1"],
        "observations": len(index),
        "incomplete_observations": 0,
        "clustered": True,
        "sampled": len(index),
    }
    return Populations(labels, pd.DataFrame(rows), settings, kept)


def test_a_group_found_in_two_spaces_is_reported_once_naming_both():
    index = pd.RangeIndex(100)
    results = {
        "explainable": _found(index, {"G1": (range(20), "group")}),
        # Jaccard 18/22 with the explainable G1: the same group; 50-69 is another.
        "DINO": _found(index, {"G1": (range(2, 22), "group"), "G2": (range(50, 70), "group")}),
        # Jaccard 11/20 with DINO's second group: the same.
        "combined": _found(index, {"G1": (range(50, 61), "group")}),
    }
    merged = _merged(results, index)
    table = merged.table.set_index("population")
    assert list(table.index) == ["G1", "G2"]
    assert table.loc["G1", "found_in"] == ["explainable", "DINO"]
    assert table.loc["G2", "found_in"] == ["DINO", "combined"]
    # Each keeps the cells of the first space that found it.
    assert merged.membership("G1").sum() == 20 and merged.membership("G2").eq(1).sum() == 20
    assert found_words(table.loc["G1"]) == "found in the explainable features and DINO"
    assert found_words(table.loc["G2"]) == (
        "found in DINO and the combined space only; no rule in the explainable features "
        "describes it"
    )


def test_a_group_living_in_one_table_is_found_there_and_not_named_with_noise():
    # Audit B, finding 8: found alone (J about 1), lost in the combined space, and described
    # by explainable features at d about 0.25. Owner, 2026-10-06: each space searched.
    objects, fovs, rng = _cells(fields=6, cells=20)
    n = len(objects)
    member = rng.random(n) < 0.06
    for j in range(5):
        objects[f"x{j}"] = rng.normal(size=n)
    embedding = rng.normal(size=(n, 40))
    embedding[np.ix_(member, np.arange(12))] += 4.0
    table = _table(objects, embedding, "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    setup = Setup(
        readout="x0",
        features=tuple(f"x{j}" for j in range(5)),
        control={"condition": "ctrl"},
        feature_tables={"DINO": tuple(table.columns[4:])},
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        found = populations_in_spaces(merged, fovs, setup)
    best = max(
        found.of_kind("group"),
        key=lambda name: (found.membership(name).eq(1) & member).sum(),
    )
    inside = found.membership(best).eq(1).to_numpy()
    assert (inside & member).sum() / (inside | member).sum() > 0.9
    row = found.table.set_index("population").loc[best]
    assert "DINO" in row["found_in"] and "explainable" not in row["found_in"]
    assert row["defining"] == [] and not row["rule"]
    result = PopulationsResult(found, None, None, None)
    text = " ".join(populations_text(result, setup)[2].split())
    assert f"so no rule in them describes it: {best} (DINO" in text
    assert "clustered in each feature space apart: the 5 explainable features" in text


def test_populations_with_no_complete_observation_are_not_assessed():
    # Audit B, finding 5: a population search with no complete observation crashed.
    objects, fovs, rng = _cells(fields=2, cells=5)
    objects["x0"] = rng.normal(size=len(objects))
    objects["x1"] = np.nan
    found = find_populations(objects, fovs, features=["x0", "x1"])
    assert not found.names and found.settings["observations"] == 0
    setup = Setup(readout="x0", features=("x0", "x1"), observation_unit="cell")
    status, headline, text = populations_text(PopulationsResult(found, None, None, None), setup)
    assert status.marker == "➖" and "not assessed" in headline
    assert "no cell has a value of every feature" in text


# --- D3: a fingerprint per table -----------------------------------------------------------


def test_the_declared_fingerprint_runs_per_table():
    answers = Answers(
        loaded_correctly=True,
        control={"condition": "ctrl"},
        aim="fingerprint",
        features=["area", "shape"],
        readout="area",
        observation_unit="cell",
        fingerprint_tolerance=0.2,
    )
    criteria = answers.fingerprint({"DINO": ("e0", "e1"), "DINOv3": ("v0", "v1")})
    assert [c.metric.features for c in criteria] == [
        ("area", "shape"),
        ("e0", "e1"),
        ("v0", "v1"),
    ]
    assert {c.threshold for c in criteria} == {0.2} and all(
        c.metric.weights is None for c in criteria
    )


def test_each_tables_fingerprint_has_its_own_verdict_and_line():
    objects, fovs, _ = _cells(fields=6, cells=5, conditions=("ctrl", "a"))
    objects["area"] = np.random.default_rng(2).normal(size=len(objects))
    objects["shape"] = np.random.default_rng(3).normal(size=len(objects))
    rng = np.random.default_rng(1)
    table = _table(objects, rng.normal(size=(len(objects), 3)), "e")
    table.loc[0, "e1"] = np.nan  # a missing value: that cell left out of DINO's fingerprint
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    fingerprint = tuple(
        OutcomeCriterion(CovarianceError(columns, True), 5.0, "lower_is_better", words)
        for columns, words in (
            (("area", "shape"), "the covariance of the standardised features within 500%"),
            (("e0", "e1", "e2"), "the covariance of the standardised DINO columns within 500%"),
        )
    )
    setup = Setup(
        readout="area",
        features=("area", "shape"),
        fingerprint=fingerprint,
        feature_tables={"DINO": ("e0", "e1", "e2")},
        observation_unit="cell",
    )
    precision = fingerprint_precision(merged, fovs, setup)
    assert set(precision["table"]) == {"the explainable features", "DINO"}
    assert precision.loc[precision["table"] == "DINO", "left_out"].sum() == 1
    text = " ".join(" ".join(fingerprint_lines(precision, "well", "fields", unit="cell")).split())
    assert "The fingerprint of the explainable features (the covariance" in text
    assert "The fingerprint of DINO (the covariance of the standardised DINO" in text
    assert "1 cell with a missing value of its columns left out." in text


# --- D5 and findings 3, 4: the separation scores -------------------------------------------


def test_a_field_effect_does_not_read_as_conditions_separating():
    # Audit B, finding 3: 99% against 33% at a field ICC of 0.2 with no condition effect.
    objects, fovs, rng = _cells(fields=6, cells=20)
    n = len(objects)
    field = objects.groupby(KEY[:3], sort=False).ngroup().to_numpy()
    shifts = rng.normal(size=(field.max() + 1, 60))
    values = np.sqrt(0.8) * rng.normal(size=(n, 60)) + np.sqrt(0.2) * shifts[field]
    merged = attach_feature_tables(objects, fovs, {"T": _table(objects, values, "t")}).objects
    found = separation(merged, fovs, spaces={"T": [f"t{j}" for j in range(60)]})
    row = found.spaces.iloc[0]
    assert abs(row["share"] - row["chance"]) < 0.05
    assert abs(row["repeat_share"] - row["repeat_chance"]) < 0.05


def test_a_perfect_separation_of_few_images_reads_100_percent():
    # Audit B, finding 4: six images a condition and repeat, k = 10: a ceiling of 50%.
    objects, fovs, rng = _cells(fields=6, cells=1)
    objects = objects.drop(columns="label_id")
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    centre = objects["condition"].map({"ctrl": 0.0, "a": 10.0, "b": 20.0}).to_numpy()
    for j in range(3):
        objects[f"x{j}"] = centre + rng.normal(size=len(objects))
    found = separation(objects, fovs, spaces={"X": ["x0", "x1", "x2"]})
    assert found.spaces.loc[0, "share"] == pytest.approx(1.0)
    by_condition = found.table[found.table["score"] == "condition"]
    assert by_condition["k"].eq(5).all()  # the five other images of its condition


def test_repeats_separating_more_than_conditions_are_said_plainly():
    objects, fovs, rng = _cells(fields=4, cells=10)
    n = len(objects)
    repeat = objects["biological_repeat"].map({"R1": 0.0, "R2": 6.0, "R3": 12.0}).to_numpy()
    batch = rng.normal(size=(n, 8)) + repeat[:, None]
    merged = attach_feature_tables(objects, fovs, {"B": _table(objects, batch, "b")}).objects
    merged["x0"] = rng.normal(size=n)
    for j in range(1, 3):
        merged[f"x{j}"] = rng.normal(size=n)
    found = separation(
        merged, fovs, spaces={"B": [f"b{j}" for j in range(8)], "X": ["x0", "x1", "x2"]}
    )
    assert found.repeats_first("B") and not found.repeats_first("X")
    line = found.line("B")
    assert "Within a condition, 100% share the repeat" in line
    assert line.endswith("Here the repeats separate more than the conditions.")
    page = comparison_table(found)
    assert "The repeats separate more than the conditions in B" in page


def test_the_combined_space_is_shown_apart_under_feature_table():
    spaces = pd.DataFrame(
        [
            {"space": "Combined", "features": 9, "share": 0.9, "chance": 0.3, "repeats": 3},
            {"space": "Explainable", "features": 3, "share": 0.4, "chance": 0.3, "repeats": 3},
            {"space": "DINO", "features": 6, "share": 0.8, "chance": 0.3, "repeats": 3},
        ]
    ).assign(repeat_share=0.3, repeat_chance=0.3, conditions=3, left_out=0, single=0)
    found = Separation(pd.DataFrame(), spaces, {})
    lines = found.comparison().splitlines()
    assert lines[0].split()[:2] == ["feature", "table"]
    assert [line.split()[0] for line in lines[1:]] == ["DINO", "Explainable", "Combined"]
    page = comparison_table(found)
    assert "<th>feature table</th>" in page and "Combined: all the tables at once" in page
    assert page.index("DINO") < page.index("Explainable") < page.index("Combined")


# --- D4, D5b: the maps ---------------------------------------------------------------------


def test_the_combined_map_says_each_tables_share_of_each_axis():
    shares = {"explainable": [0.42, 0.9], "DINOv2": [0.58, 0.1]}
    assert share_lines(shares, (0.3, 0.2)) == [
        "PC1: explainable 42%, DINOv2 58%.",
        "PC2: explainable 90%, DINOv2 10%.",
    ]


def test_the_centred_view_subtracts_each_repeats_control_median():
    objects, fovs, rng = _cells(fields=2, cells=3)
    values = rng.normal(size=(len(objects), 2))
    table = _table(objects, values, "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    merged["x0"] = 1.0
    setup = Setup(
        readout="x0", control={"condition": "ctrl"}, feature_tables={"DINO": ("d0", "d1")}
    )
    centred = centred_tables(merged, fovs, setup)
    for (condition, repeat), rows in merged.groupby(["condition", "biological_repeat"]):
        control = merged[(merged["condition"] == "ctrl") & (merged["biological_repeat"] == repeat)]
        expected = rows["d0"] - control["d0"].median()
        np.testing.assert_allclose(centred.loc[rows.index, "d0"], expected)
    assert (centred["x0"] == 1.0).all()  # the explainable features are left as they are
    assert centred_tables(merged, fovs, Setup(readout="x0")) is None


# --- Findings 2, 5, 6, 11, 12, 14, 15, 16 ---------------------------------------------------


def test_a_copy_confined_to_one_table_is_found_and_named():
    # Audit B, finding 2: pooled, the copy in the embedding alone was missed.
    objects, fovs, rng = _cells(fields=3, cells=12)
    n = len(objects)
    for j in range(3):
        objects[f"x{j}"] = rng.normal(size=n)
    values = rng.normal(size=(n, 6))
    first = (objects["image_file"] == "a_R1_0.tif").to_numpy()
    second = (objects["image_file"] == "a_R1_1.tif").to_numpy()
    values[second] = values[first]  # the embedding of one image computed twice
    merged = attach_feature_tables(objects, fovs, {"DINO": _table(objects, values, "d")}).objects
    columns = {"explainable": ["x0", "x1", "x2"], "DINO": [f"d{j}" for j in range(6)]}
    pooled = integrity_checks(merged, fovs, feature_cols=[c for v in columns.values() for c in v])
    assert pooled.findings.empty
    found = integrity_by_table(merged, fovs, tables=columns)
    assert list(found.findings["check"]) == ["copied field"]
    assert list(found.findings["table"]) == ["DINO"]
    assert "copied field in DINO" in found.summary()
    assert "each feature table apart: explainable (3 columns) and DINO (6 columns)" in " ".join(
        found.summary().split()
    )


def test_a_column_without_a_value_is_refused_and_a_constant_one_listed():
    objects, fovs, embedding = _tables()
    with pytest.raises(ValueError, match="columns without a single value: e1"):
        attach_feature_tables(objects, fovs, {"DINO": embedding.assign(e1=np.nan)})
    tables = attach_feature_tables(objects, fovs, {"DINO": embedding.assign(e2=3.0)})
    assert tables.constant == {"DINO": ("e2",)}
    assert "1 column with one value everywhere (e2), which no analysis can use" in (
        tables.summary()
    )


def test_a_mostly_zero_column_does_not_outweigh_the_others():
    # Audit B, finding 6: just under half the values at 0, a tiny MAD, a variance of 830,000.
    rng = np.random.default_rng(0)
    relu = np.abs(rng.normal(size=2000))
    relu[:980] = 0.0
    normal = rng.normal(size=2000)
    z, _ = _standardised(np.column_stack([relu, normal]))
    assert z[:, 0].var() <= 16
    # A column with an ordinary MAD is scaled by it as before.
    mad = 1.4826 * np.median(np.abs(normal - np.median(normal)))
    np.testing.assert_allclose(z[:, 1], (normal - np.median(normal)) / mad)


@pytest.mark.parametrize("which", ["table", "objects"])
def test_a_label_written_as_a_decimal_matches_its_whole_number(which):
    # Audit B, finding 11: a CSV column with one empty cell is read as 1.0, 2.0, ...
    objects, fovs, embedding = _tables()
    if which == "table":
        embedding["label_id"] = embedding["label_id"].astype(float)
    else:
        objects["label_id"] = objects["label_id"].astype(float)
    assert "e0" in attach_feature_tables(objects, fovs, {"DINO": embedding}).objects


def test_the_columns_are_said_as_explainable_features_and_other_tables():
    setup = Setup(
        readout="a",
        features=("a", "b"),
        feature_tables={"DINOv2": ("d0", "d1", "d2"), "DINOv3": ("v0", "v1")},
    )
    assert features_words(setup) == "2 explainable features and 5 columns of 2 other tables"
    one = Setup(readout="a", features=("a", "b"), feature_tables={"DINOv2": ("d0",)})
    assert features_words(one) == "2 explainable features and 1 column of the other table DINOv2"


def test_a_reordered_table_keeps_its_checksum_and_a_changed_one_is_named():
    objects, _, embedding = _tables()
    first = entries({"DINO": (embedding, None)}, objects)
    shuffled = entries({"DINO": (embedding.sample(frac=1, random_state=0), None)}, objects)
    assert first == shuffled
    edited = embedding.assign(e0=embedding["e0"] + 1e-3)
    later = entries({"DINO": (edited, None)}, objects)
    saved = {"checksum": "a", "fovs_checksum": None, "feature_tables": first}
    now = {"checksum": "a", "fovs_checksum": None, "feature_tables": later}
    assert changed(saved, now) == ["the feature table 'DINO'"]
    assert changed(saved, {**now, "feature_tables": []}) == [
        "the feature table 'DINO', not given this time"
    ]


def test_the_rough_runtime_counts_the_other_tables():
    alone = rough_runtime(17, 3)
    assert rough_runtime(17, 3, tables=(400,)) > alone
    assert rough_runtime(17, 3, tables=(1500,)) > rough_runtime(17, 3, tables=(400,))
    assert rough_runtime(17, 3, tables=()) == alone
