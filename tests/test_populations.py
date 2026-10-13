"""Question 10 (owner, 2026-10-02): populations are found on every cell pooled, on all the
features, and only then is a condition asked whether it is enriched in one; the PCA and
UMAP ring them, colour by any feature and describe a selection by its features."""

import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_rare_fingerprint import FEATURES, _screen, _setup

from morphocoverage._style import Status
from morphocoverage.embedding import embed
from morphocoverage.sampling_unit import find_populations
from morphocoverage.workflows import html_report, report
from morphocoverage.workflows._html_render import chip
from morphocoverage.workflows._populations import (
    enrichment_tests,
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
            rows += [{"condition": condition, "biological_repeat": f"r{r}", "P2": 1.0}] * k
            rows += [{"condition": condition, "biological_repeat": f"r{r}", "P2": 0.0}] * (n - k)
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
