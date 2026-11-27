"""Question 10 (owner, 2026-10-02): populations are found on every cell pooled, on all the
features, and only then is a condition asked whether it is enriched in one; the PCA and
UMAP ring them, colour by any feature and describe a selection by its features."""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import SEVERAL, _several
from _helpers_feature_tables import _cells, _table
from _helpers_rare_fingerprint import FEATURES, _screen, _setup

from morphocoverage._style import Status
from morphocoverage.embedding import embed
from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.sampling_unit import find_populations
from morphocoverage.sampling_unit.populations import Populations
from morphocoverage.workflows import Setup, html_report, report
from morphocoverage.workflows._feature_space import attached
from morphocoverage.workflows._html_cards import chip_words
from morphocoverage.workflows._html_render import chip
from morphocoverage.workflows._population_spaces import _merged, found_words, populations_in_spaces
from morphocoverage.workflows._populations import (
    PopulationsResult,
    _side,
    enrichment_tests,
    labelled_populations,
    one_sided,
    populations_in_report,
    populations_text,
)


@pytest.fixture(scope="module")
def screen():
    objects, fovs = _screen()
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    return objects, fovs, setup, result


def test_the_planted_population_is_found_on_every_condition_pooled():
    objects, fovs = _screen()
    found = find_populations(objects, fovs, features=FEATURES, observation_unit="cell")
    labels = found.labels.reindex(objects.index)
    # The mitoses of every condition form one population of their own, named by size.
    small = found.table.sort_values("share").iloc[0]
    inside = labels == small["population"]
    assert objects.loc[inside, "mitotic"].mean() > 0.95
    assert inside[objects["mitotic"]].mean() > 0.9
    assert set(objects.loc[inside, "condition"]) == {"siCTRL", "narrow", "arrest"}
    assert "high dna" in small["defined_by"] and small["repeats"] == 3
    # Spread over many fields and every well (#79, owner, 2026-10-05).
    assert small["largest_field"] < 0.1 and small["technical_repeats"] == 18
    assert found.names == list(
        found.table.sort_values("observations", ascending=False)["population"]
    )


def test_a_condition_more_common_in_a_population_is_said_with_its_shares(screen):
    """With 3 repeats the arrest's mitoses (10% against 2% of the control's cells) are higher
    in every repeat; their q is borderline, and every repeat on one side is a trend only from
    6 repeats (owner, 2026-10-06, audit 10), so the answer gives their numbers in the table,
    and no trend."""
    result = screen[3]
    status = result.document.statuses["populations"]
    headline = result.document.headlines["populations"]
    assert headline.endswith("no condition consistently enriched")
    short = result.document.short["populations"]
    found = r"arrest: higher in 3 of 3, \d+(\.\d)?% vs \d\.\d%, q 0\.\d+"
    assert re.search(found, " ".join(short.split()))
    assert "in every repeat, not significant" not in short
    # "of every condition pooled" read as a number per condition (ChatGPT, 2026-10-05).
    assert "cells in all, every condition pooled, clustered on the 3 declared features (" in short
    assert "the rest (not tested)" in short and "to validate on the images" in short
    # The share of a group's cells in its largest field is a column of the table (#79).
    assert "in largest field" in short
    # Each group comes with the rule that picks out its cells, fitted on all the cells; its
    # procedure is checked on held-out fields (#88, owner, 2026-10-05).
    flat = " ".join(short.split()).lower()
    assert "picked by dna >" in flat and "fitted on all the cells" in flat
    assert "refitted with whole fields held out" in flat
    # Population enrichment is exploratory: ⚠️, never "Enriched" (audit 10).
    assert status is Status.CAUTION
    assert chip(status, "populations", headline).endswith("No consistent difference</span>")


def test_the_enrichment_is_tested_on_the_log_odds_across_repeats():
    rows = []
    for r in range(4):
        for condition, share in (("ctrl", 0.02), ("drug", 0.20)):
            n, k = 1000, int(1000 * share) + r
            rows += [
                {
                    "condition": condition,
                    "biological_repeat": f"r{r}",
                    "repeat_group": f"r{r}",
                    "P2": 1.0,
                }
            ] * k
            rows += [
                {
                    "condition": condition,
                    "biological_repeat": f"r{r}",
                    "repeat_group": f"r{r}",
                    "P2": 0.0,
                }
            ] * (n - k)
    tests = enrichment_tests(pd.DataFrame(rows), ["P2"], "ctrl")
    row = tests.iloc[0]
    assert row["higher"] == 4 and row["q_paired"] < 0.001
    assert row["d"] == pytest.approx(np.log((200.5 / 800.5) / (20.5 / 980.5)), abs=0.1)


def test_the_main_population_is_listed_but_not_tested(screen):
    populations = screen[3].results["populations"]
    main = populations.populations.table.iloc[0]
    assert main["share"] > 0.5
    assert main["population"] not in set(populations.enrichment["feature"])


def test_a_lean_observation_table_gives_the_same_populations_and_enrichment():
    """#86: condition and repeats only in the FOV table (a lean table, which the validator
    attaches) gave no enrichment, as if there were no control, or raised in pooled_columns."""
    objects, fovs = _screen()
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    full = populations_in_report(objects, fovs, setup)
    lean = objects.drop(columns=["condition", "biological_repeat", "technical_repeat"])
    found = populations_in_report(lean, fovs, setup)
    assert found.populations.table.equals(full.populations.table)
    assert found.enrichment is not None and len(found.enrichment)
    pd.testing.assert_frame_equal(found.enrichment, full.enrichment)


def test_no_group_found_is_said_neutrally():
    rng = np.random.default_rng(0)
    rows = []
    for repeat in ("r1", "r2", "r3"):
        for condition in ("ctrl", "drug"):
            for field in range(6):
                for label in range(40):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "repeat_group": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            **dict(zip(FEATURES, rng.normal(0, 1, 3), strict=True)),
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    setup = _setup(control={"condition": "ctrl"}, fingerprint=(), rare=None, rare_label=None)
    result = populations_in_report(objects, fovs, setup)
    status, headline, _text = populations_text(result, setup)
    if not result.populations.names:
        # #94 (owner, 2026-10-05): no group found is no evidence of one continuum.
        assert status is Status.NOT_ASSESSED and headline.startswith("no group:")
        assert "no candidate group passed the check on these features" in headline
        assert "continuum" not in headline + _text
        assert chip(status, "populations", headline).endswith("➖</span> No group</span>")
    else:  # one blob found as one population: nothing enriched
        assert status is not Status.OK


def test_the_settings_record_the_umap_layout_the_populations_were_found_on():
    objects, fovs = _screen()
    found = find_populations(objects, fovs, features=FEATURES, observation_unit="cell")
    assert found.settings["compatibility_layout"] is True


def test_above_the_sample_size_every_cell_still_gets_a_population():
    objects, fovs = _screen()
    full = find_populations(objects, fovs, features=FEATURES, observation_unit="cell")
    sampled = find_populations(
        objects, fovs, features=FEATURES, observation_unit="cell", max_points=1500
    )
    assert sampled.settings["sampled"] == 1500
    mitotic = objects["mitotic"]
    small = sampled.table.sort_values("share").iloc[0]["population"]
    assert (sampled.labels.reindex(objects.index)[mitotic] == small).mean() > 0.9
    assert full.settings["sampled"] == len(objects)


def test_the_smallest_cluster_is_a_share_of_the_sample_not_of_the_whole_table():
    """#87: the default min_cells was 0.5% of every observation, applied to the sample, so a
    table ten times larger than max_points lost the planted mitoses (5%) altogether."""
    tables = [_screen(seed) for seed in range(10)]
    objects = pd.concat(
        [o.assign(image_file=f"s{s}_" + o["image_file"]) for s, (o, _) in enumerate(tables)],
        ignore_index=True,
    )
    fovs = pd.concat(
        [f.assign(image_file=f"s{s}_" + f["image_file"]) for s, (_, f) in enumerate(tables)],
        ignore_index=True,
    )
    found = find_populations(
        objects, fovs, features=FEATURES, observation_unit="cell", max_points=500
    )
    assert len(found.table)
    small = found.table.sort_values("share").iloc[0]["population"]
    assert (found.labels[objects["mitotic"]] == small).mean() > 0.9
    assert found.settings["sampled"] <= 501 and found.settings["min_cells"] == 15


def test_the_points_carry_their_population_and_feature_values():
    objects, fovs = _screen()
    marked = objects.assign(population=np.where(objects["mitotic"], "P2", "P1"))
    embedding = embed(
        marked, fovs, features=FEATURES, carry=["population"], umap=False, observation_unit="cell"
    )
    assert {"population", "z:area", "z:dna", "z:ratio"} <= set(embedding.points)
    assert (embedding.points["population"] == "P2").sum() == objects["mitotic"].sum()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_maps_ring_populations_colour_by_feature_and_describe_a_selection(screen, tmp_path):
    objects, fovs, _, result = screen
    page = html_report(result, objects, fovs, tmp_path).read_text()
    assert 'id="populations"' in page and "Are there populations or groups of cells?" in page
    data = json.loads(
        re.search(r'id="embedding-data">(.*?)</script>', page, re.DOTALL)
        .group(1)
        .replace("<\\/", "</")
    )
    assert [p["name"] for p in data["populations"]] == result.results[
        "populations"
    ].populations.names
    first = data["sets"][data["values"][0]]["PCA"]
    assert len(first["pop"]) == len(first["x"]) and set(first["z"]) == set(FEATURES)
    assert '<select class="colour-by">' in page and '<option value="population">' in page
    assert "Features that set them apart" in page and "circle-open" in page


def test_no_group_reads_neutrally_in_the_text_the_chip_and_the_report():
    """#94 (owner, 2026-10-05): with no group, question 10 says no candidate passed the
    check, ➖ "No group", never that the cells form one continuum."""
    from morphocoverage.sampling_unit.populations import _COLUMNS, Populations
    from morphocoverage.workflows._populations import PopulationsResult

    settings = {
        "features": list(FEATURES),
        "map_components": 3,
        "min_cells": 15,
        "purity": 0.94,
        "neighbours": 10,
        "observations": 500,
        "clustered": True,
    }
    empty = Populations(pd.Series(dtype=object), pd.DataFrame(columns=_COLUMNS), settings)
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    status, headline, text = populations_text(PopulationsResult(empty, None, None, None), setup)
    assert status is Status.NOT_ASSESSED and status.marker == "➖"
    assert headline == "no group: no candidate group passed the check on these features"
    flat = " ".join(text.split())
    assert "no candidate group passed the neighbours-inside check on these features" in flat
    assert "continuum" not in flat
    assert chip(status, "populations", headline).endswith("No group</span>")


def test_the_populations_name_the_cells_without_their_own_control_never_a_missing_value():
    objects, fovs, tables = _several()
    objects.loc[objects.index[:5], "area"] = np.nan  # a true missing value, counted as one
    setup = Setup(
        readout="area",
        features=("area", "fib_len", "fib_w"),
        control=SEVERAL,
        normalize="control_mean",
        observation_unit="cell",
    )
    merged, setup, _ = attached(objects, fovs, setup, {"fiber": tables["fiber"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = setup.prepared(merged, fovs)
        found = populations_in_report(prepared, fovs, setup, merged)
    settings = found.populations.settings
    # b1 in R1, where cB was not imaged: 3 fields of 6 cells (audit 8, seed 4).
    assert [(b["reason"], b["observations"]) for b in settings["left_out_by"]] == [
        ("no own control", 18)
    ]
    assert settings["incomplete_observations"] == 5
    text = " ".join(populations_text(found, setup)[2].split())
    assert "(18 cells of b1 in R1 left out: no own control)" in text


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


def test_enrichment_without_a_repeat_column_skips_a_condition_without_cells():
    marked = pd.DataFrame(
        {"condition": ["ctrl"] * 4 + ["drug"] * 4, "G1": [0, 0, 0, 1, 1, 1, 0, 0]}
    )
    tests = enrichment_tests(marked, ["G1"], (("drug", "ctrl"), ("other", "ctrl")))
    assert tests[["feature", "condition", "repeats", "higher", "lower"]].to_dict("records") == [
        {"feature": "G1", "condition": "drug vs ctrl", "repeats": 1, "higher": 1, "lower": 0}
    ]
    assert np.isnan(tests["p_paired"].iloc[0])
    assert tests["share"].iloc[0] == 0.5 and tests["control_share"].iloc[0] == 0.25
    assert tests.attrs["unpaired"] == []


def _populations_edges():
    from morphocoverage.sampling_unit import find_populations

    rng = np.random.default_rng(0)
    rows = []
    for condition, shift in (("ctrl", 0.0), ("A", 0.0)):
        for j in range(15):
            for _ in range(10):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": "r1",
                        "repeat_group": "r1",
                        "image_file": f"{condition}{j}",
                        "a": rng.normal(shift, 1),
                        "b": rng.normal(0, 1),
                    }
                )
    objects = pd.DataFrame(rows)
    planted = objects.index[:90]
    objects.loc[planted, ["a", "b"]] += 9.0
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    sampled = find_populations(objects, fovs, features=["a", "b"], max_points=150, min_cells=20)
    missing = objects.assign(a=np.nan)
    none = find_populations(missing, fovs, features=["a", "b"])
    keep = ["population", "kind", "observations", "fields", "repeats", "defined_by"]
    return (
        sampled.table[keep].to_dict("records"),
        {k: sampled.settings[k] for k in ("sampled", "clustered", "assigned", "min_cells")},
        sampled.labels.value_counts(dropna=False).to_dict(),
        none.table.shape,
        {
            k: none.settings[k]
            for k in (
                "features",
                "observations",
                "incomplete_observations",
                "clustered",
                "assigned",
            )
        },
    )


EXPECTED_POPULATIONS_EDGES = (
    [
        {
            "population": "P1",
            "kind": "main",
            "observations": 210,
            "fields": 21,
            "repeats": 1,
            "defined_by": "low b (d -9.2), low a (d -9.0)",
        },
        {
            "population": "G1",
            "kind": "group",
            "observations": 90,
            "fields": 9,
            "repeats": 1,
            "defined_by": "high b (d +9.2), high a (d +9.0)",
        },
    ],
    {"sampled": 150, "clustered": True, "assigned": 1.0, "min_cells": 20},
    {"P1": 210, "G1": 90},
    (0, 17),
    {
        "features": [],
        "observations": 0,
        "incomplete_observations": 300,
        "clustered": False,
        "assigned": 0.0,
    },
)


def test_populations_edges():
    assert _populations_edges() == EXPECTED_POPULATIONS_EDGES


def _shares(counts):
    """``counts``: (condition, repeat) -> (cells in the population, cells)."""
    rows = []
    for (condition, repeat), (inside, cells) in counts.items():
        rows += [
            {
                "condition": condition,
                "biological_repeat": repeat,
                "repeat_group": repeat,
                "P2": float(i < inside),
            }
            for i in range(cells)
        ]
    return pd.DataFrame(rows)


def test_an_enrichment_reads_the_way_its_test_goes():
    # 1 of 5000 cells against 0 of 300, in every repeat: higher, never "lower".
    counts = {}
    for repeat in ("r1", "r2", "r3"):
        counts[("drug", repeat)] = (1, 5000)
        counts[("ctrl", repeat)] = (0, 300)
    [row] = enrichment_tests(_shares(counts), ["P2"], "ctrl").to_dict("records")
    assert row["higher"] == 3 and row["d"] > 0


def test_a_tie_between_repeats_is_mixed_and_one_repeat_is_no_trend():
    assert _side({"higher": 1, "lower": 1, "repeats": 3}) == "mixed (1 higher, 1 lower, of 3)"
    assert _side({"higher": 0, "lower": 0, "repeats": 3}) == "no change in 3 of 3"
    single = pd.DataFrame({"higher": [1], "lower": [0], "repeats": [1]})
    assert not one_sided(single).any()


def test_a_condition_without_a_shared_repeat_is_named():
    counts = {("drug", "r9"): (5, 50), ("ctrl", "r1"): (1, 50), ("ctrl", "r2"): (1, 50)}
    tests = enrichment_tests(_shares(counts), ["P2"], "ctrl")
    assert tests.empty and tests.attrs["unpaired"] == ["drug"]


def test_a_repeat_where_the_group_is_absent_on_both_sides_counts_as_no_change():
    rows = []
    for r, (control_k, drug_k) in enumerate(((5, 30), (6, 28), (0, 0), (4, 25))):
        for condition, k in (("ctrl", control_k), ("drug", drug_k)):
            rows += [
                {
                    "condition": condition,
                    "biological_repeat": f"r{r}",
                    "repeat_group": f"r{r}",
                    "P2": 1.0,
                }
            ] * k
            rows += [
                {
                    "condition": condition,
                    "biological_repeat": f"r{r}",
                    "repeat_group": f"r{r}",
                    "P2": 0.0,
                }
            ] * (300 - k)
    [row] = enrichment_tests(pd.DataFrame(rows), ["P2"], "ctrl").to_dict("records")
    assert row["repeats"] == 4 and row["higher"] == 3 and row["lower"] == 0


@pytest.fixture(scope="module")
def labelled():
    from _helpers_rare_fingerprint import _screen, _setup

    objects, fovs = _screen()
    setup = _setup(fingerprint=())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = populations_in_report(objects, fovs, setup)
    return found, setup


def test_the_declared_labels_group_is_tested_once_as_the_label(labelled):
    found, setup = labelled
    own = labelled_populations(found.label)
    assert own and not set(own) & set(found.enrichment["feature"])
    status, headline, text = populations_text(found, setup)
    flat = " ".join(text.split())
    name = own[0]
    said = f"{name} is the declared rare phenotype, mitotic: most of its cells carry the label"
    # Said once, in the headline (the page's In short); the text gives its numbers only.
    assert said in headline and said not in flat
    held = rf"{name}: \d+% of its cells carry the label mitotic, and it holds \d+% of the "
    assert re.search(held + "labelled ones; not tested again here.", flat)
    assert "its share is compared in “Do the conditions differ?”" in headline
    assert "q " not in headline and status is Status.OK

    class Result:
        pass

    result = Result()
    result.results = {"populations": found}
    assert chip_words(result)["populations"] == "The declared label"


def test_an_embedding_only_report_finds_and_tests_the_populations_of_its_table():
    """With no readout and no explainable feature (the tables alone), the populations are
    found on the table, every cell is marked and tested (#649: no refusal for want of a
    feature column), the same tests as on the marks computed directly."""
    objects, fovs = _screen()
    key = ["condition", "biological_repeat", "technical_repeat", "image_file", "label_id"]
    table = objects[[*key, *FEATURES]]
    observations = objects.drop(columns=list(FEATURES))
    setup = Setup(control={"condition": "siCTRL"}, observation_unit="cell", random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(observations, fovs, setup, feature_tables={"emb": table})
    found = result.results["populations"]
    assert setup.embedding_only and found.control == "siCTRL"
    groups = found.populations.of_kind("group")
    assert groups, found.populations.table
    # The planted mitoses are a group of their own, in every condition.
    marks = pd.DataFrame({g: found.populations.membership(g) for g in groups})
    mitoses = max(groups, key=lambda g: objects.loc[marks[g].eq(1), "mitotic"].mean())
    assert objects.loc[marks[mitoses].eq(1), "mitotic"].mean() > 0.9
    # Every cell is marked (none lost), and the tests are those of the marks themselves.
    assert marks.notna().all(axis=1).reindex(objects.index).all()
    expected = enrichment_tests(objects.assign(**marks), groups, "siCTRL")
    pd.testing.assert_frame_equal(found.enrichment, expected)
    arrest = found.enrichment.set_index(["feature", "condition"]).loc[(mitoses, "arrest")]
    assert arrest["share"] > arrest["control_share"]
