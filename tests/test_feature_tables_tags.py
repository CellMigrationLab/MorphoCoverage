"""Feature tables tagged explainable or non-explainable (owner, 2026-10-06): the tags and
their refusal, the column names found in two tables, one correction family over every
explainable table, MFA weights across them, the result tables grouped by table, the guided
session recording each tag; a bare DataFrame stays non-explainable, as before."""

import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from morphocoverage.feature_tables import attach_feature_tables, mfa_weights, read_tag
from morphocoverage.sampling_unit._experiments import corrected
from morphocoverage.transforms import BlockWeights, Chain
from morphocoverage.workflows import Setup, guide, html_report, report, scan_design
from morphocoverage.workflows._feature_space import (
    centred_tables,
    copy_tables,
    explainable_space,
    novelty_space,
    space_names,
    unusual_space,
)
from morphocoverage.workflows._population_rule_words import rule_words
from morphocoverage.workflows._population_spaces import rule_tables
from morphocoverage.workflows.guided import ANSWERS_FILE
from morphocoverage.workflows.setup import default_novelty

KEY = ["condition", "biological_repeat", "image_file", "label_id"]


def _cells(seed: int = 0):
    """3 conditions in 3 repeats, 4 fields of 6 labelled cells; ``area`` in the observation
    table; fiberfeatures (``length`` shifted in ``a``, and an ``area`` of its own),
    CellProfiler (3 correlated columns) and a DINO embedding of 4 columns."""
    rng = np.random.default_rng(seed)
    rows = [
        {"condition": c, "biological_repeat": f"R{r}", "image_file": f"{c}_{r}_{f}.tif"}
        | {"label_id": k}
        for c in ("ctrl", "a", "b")
        for r in range(3)
        for f in range(4)
        for k in range(1, 7)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    n, shifted = len(objects), objects["condition"].eq("a").to_numpy()
    objects["area"] = rng.normal(10, 1, n) + 1.5 * shifted
    fiber = (
        objects[KEY]
        .assign(length=rng.normal(5, 1, n) + shifted, area=rng.normal(3, 1, n))
        .iloc[::-1]
    )
    base = rng.normal(size=n)
    profiler = objects[KEY].assign(
        **{f"int{j}": base + 0.3 * rng.normal(size=n) + 5 for j in range(3)}
    )
    dino = objects[KEY].assign(**{f"f{j}": rng.normal(size=n) for j in range(4)})
    return objects, fovs, fiber, profiler, dino


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


SETUP = Setup(readout="area", control={"condition": "ctrl"}, observation_unit="cell")


@pytest.fixture(scope="module")
def tagged():
    objects, fovs, fiber, profiler, dino = _cells()
    tables = {
        "fiberfeatures": (fiber, "explainable"),
        "CellProfiler": (profiler, "Explainable"),
        "DINOv2": dino,
    }
    return objects, fovs, _quiet(report, objects, fovs, SETUP, feature_tables=tables), tables


# --- 1. API: tags ---------------------------------------------------------------------------


def test_a_tag_is_one_of_two_words_in_any_case_and_any_other_is_refused():
    assert read_tag("Non_Explainable") == read_tag("non explainable") == "non-explainable"
    assert read_tag(" EXPLAINABLE ") == "explainable"
    assert read_tag("embedding") is None and read_tag(1) is None
    objects, fovs, fiber, _, dino = _cells()
    with pytest.raises(ValueError) as error:
        attach_feature_tables(objects, fovs, {"fiber": (fiber, "readable"), "DINO": dino})
    message = str(error.value)
    assert "'fiber' is tagged 'readable'" in message
    assert "'explainable' (each column read and tested one by one)" in message
    tables = attach_feature_tables(objects, fovs, {"fiber": (fiber, "explainable"), "DINO": dino})
    assert tables.tags == {"fiber": "explainable", "DINO": "non-explainable"}
    assert set(tables.explainable) == {"fiber"} and set(tables.other) == {"DINO"}


def test_a_bare_table_is_non_explainable_and_gives_the_report_of_before():
    objects, fovs, _, _, dino = _cells()
    setup = Setup(readout="area", control={"condition": "ctrl"}, observation_unit="cell")
    bare = _quiet(report, objects, fovs, setup, feature_tables={"DINOv2": dino})
    named = _quiet(
        report, objects, fovs, setup, feature_tables={"DINOv2": (dino, "non-explainable")}
    )
    assert bare.summary(detail="full") == named.summary(detail="full")
    assert bare.setup.explainable_tables is None and bare.setup.feature_tables
    # No explainable table, no table column: the result tables as before.
    assert "(non-explainable)" not in bare.summary(detail="full")


# --- 2. Names -------------------------------------------------------------------------------


def test_a_name_found_twice_is_prefixed_outside_the_observation_table_and_said_once(tagged):
    _, _, result, _ = tagged
    assert "area" in result.setup.columns and "fiberfeatures: area" in result.setup.features
    assert result.setup.explainable_tables["fiberfeatures"] == ("length", "fiberfeatures: area")
    full = result.summary(detail="full")
    assert full.count('so named "fiberfeatures: area" here') == 1


def test_two_embeddings_with_the_same_column_names_load_together():
    objects, fovs, _, _, dino = _cells()
    tables = attach_feature_tables(objects, fovs, {"DINOv2": dino, "DINOv3": dino})
    assert tables.columns["DINOv2"][0] == "DINOv2: f0"
    assert tables.columns["DINOv3"][0] == "DINOv3: f0"
    assert tables.renamed["DINOv3"]["f0"] == "DINOv3: f0"
    with pytest.raises(ValueError, match="metadata columns"):
        attach_feature_tables(objects, fovs, {"DINO": (dino.assign(z=0), "explainable")})


# --- 3. Tests: one family over every explainable table --------------------------------------


def test_both_explainable_tables_are_tested_in_one_family_named_by_table(tagged):
    _, _, result, _ = tagged
    tests = result.discovery.tests
    tables = dict(zip(tests["feature"], tests["table"], strict=True))
    assert tables["area"] == "observation table"
    assert tables["length"] == tables["fiberfeatures: area"] == "fiberfeatures"
    assert tables["int0"] == "CellProfiler"
    assert not any(f.startswith("f") and f[1:].isdigit() for f in tables)  # DINOv2 never
    assert list(tests.columns).index("table") == list(tests.columns).index("feature") + 1
    # One Benjamini-Hochberg family over every explainable feature, per sub-experiment.
    expected = corrected(tests["p_cells"], tests["experiment"])
    np.testing.assert_allclose(tests["q_cells"], expected, equal_nan=True)
    assert len(tests) == 2 * len(result.setup.columns)
    # The whole-table test reads the non-explainable table only.
    whole = result.results["whole_table"]
    assert set(whole.tests["table"]) == {"DINOv2"}


def test_the_empty_features_rule_and_a_setup_without_readout():
    every = Setup(readout="area", explainable_tables={"fiber": ("length", "fiber: area")})
    assert every.features == ("length", "fiber: area")
    declared = Setup(
        readout="area", features=("length",), explainable_tables={"fiber": ("length",)}
    )
    assert declared.features == ("length",)
    with pytest.raises(ValueError, match="declare the readout"):
        Setup(explainable_tables={"fiber": ("length",)})


# --- 4. MFA across explainable tables -------------------------------------------------------


def _wide(n: int = 400):
    rng = np.random.default_rng(1)
    rows = [
        {"condition": "ctrl", "biological_repeat": "R1", "image_file": f"f{i // 10}.tif"}
        | {"label_id": i % 10}
        for i in range(n)
    ]
    objects = pd.DataFrame(rows)
    for j in range(3):
        objects[f"a{j}"] = rng.normal(size=n)
    base = rng.normal(size=n)
    for j in range(60):
        objects[f"b{j}"] = base + 0.2 * rng.normal(size=n)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    return objects, fovs


def test_a_wide_correlated_table_weighs_as_much_as_a_narrow_one():
    objects, fovs = _wide()
    narrow, wide = tuple(f"a{j}" for j in range(3)), tuple(f"b{j}" for j in range(60))
    setup = Setup(
        readout="a0", features=(*narrow, *wide), explainable_tables={"A": narrow, "B": wide}
    )
    columns, blocks = explainable_space(setup)
    assert blocks == {"A": narrow, "B": wide}
    columns, weights, words = unusual_space(objects, fovs, setup)
    assert "each of their 2 tables weighted by MFA" in words
    z = objects[columns].to_numpy()
    z = (z - np.median(z, axis=0)) / (1.4826 * np.median(np.abs(z - np.median(z, axis=0)), axis=0))
    weighted = z * np.array([weights[c] for c in columns])

    def total(names):
        part = weighted[:, [columns.index(c) for c in names]]
        return float(((part - part.mean(axis=0)) ** 2).sum())

    unweighted = 60 / 3
    assert unweighted > 10 and 0.2 < total(wide) / total(narrow) < 2
    assert weights["a0"] == 1.0
    # The transform the definition of new gets: the same weights, fitted on the visible.
    novelty = default_novelty(setup.features)
    weighted_novelty, _ = novelty_space(replace(setup, novelty=novelty))
    assert isinstance(weighted_novelty.transform, Chain)
    assert isinstance(weighted_novelty.transform.transforms[-1], BlockWeights)
    state, _ = BlockWeights((narrow, wide)).fit_matrix(z, tuple(columns))
    np.testing.assert_allclose(state["weights"], mfa_weights(z, columns, blocks))


def test_one_explainable_table_is_not_weighted():
    objects, fovs = _wide()
    features = tuple(f"a{j}" for j in range(3))
    alone = Setup(readout="a0", features=features, novelty=default_novelty(features))
    assert explainable_space(alone)[1] is None
    assert novelty_space(alone)[0] is alone.novelty
    assert unusual_space(objects, fovs, alone)[1] is None
    # All the features from one explainable table: one table, nothing weighted either.
    one = Setup(readout="a0", features=features, explainable_tables={"A": features})
    assert explainable_space(one)[1] is None and copy_tables(one) == {"explainable": features}


# --- 5. Maps and copy checks per table ------------------------------------------------------


def test_each_explainable_table_has_its_map_and_its_copy_checks(tagged):
    _, _, result, _ = tagged
    names = list(space_names(result.setup))
    assert names[:2] == ["Combined", "Explainable"]
    assert {"observation table", "fiberfeatures", "CellProfiler", "DINOv2"} <= set(names)
    assert space_names(result.setup)["Explainable"][1] is not None  # weighted by MFA
    assert list(copy_tables(result.setup)) == [
        "observation table",
        "fiberfeatures",
        "CellProfiler",
        "DINOv2",
    ]
    overview = result.summary(detail="full")
    assert "fiberfeatures (explainable): 2 feature columns" in overview
    assert "Explainable features by table: observation table 1, fiberfeatures 2" in overview
    assert "every column of these tables when none is declared" in overview


# --- 7. The scan and the guided session ------------------------------------------------------


def test_the_scan_lists_each_table_with_its_tag_and_offers_the_explainable_columns():
    objects, fovs, fiber, _, dino = _cells()
    tables = {"fiber": (fiber, "explainable"), "DINO": dino}
    scan = _quiet(scan_design, objects, fovs, feature_tables=tables)
    text = scan.summary()
    assert "Other feature table fiber (explainable): 2 feature columns" in text
    assert "Other feature table DINO (non-explainable): 4 feature columns" in text
    assert {"length", "fiber: area"} <= set(scan.settings["feature_cols"])
    setup = scan.setup("area", control={"condition": "ctrl"})
    assert setup.explainable_tables == {"fiber": ("length", "fiber: area")}
    assert setup.feature_tables == {"DINO": ("f0", "f1", "f2", "f3")}


class Script:
    def __init__(self, answers):
        self.answers, self.said = list(answers), []

    def ask(self, prompt):
        if not self.answers:
            raise AssertionError(f"no scripted answer left for {prompt!r}")
        return self.answers.pop(0)

    def say(self, text):
        self.said.append(text)


def _guide(objects, folder, answers, **kwargs):
    script = Script(answers)
    result = guide(objects, folder=folder, ask=script.ask, say=script.say, **kwargs)
    return result, "\n".join(script.said)


QUESTIONS = ["y", "", "", "", "", "", "all", ""]


def test_the_guided_session_asks_records_and_reloads_each_tag(tmp_path):
    objects, _, fiber, _, dino = _cells()
    cells, fibers, dinos = tmp_path / "cells.csv", tmp_path / "fiber.csv", tmp_path / "dino.csv"
    objects.to_csv(cells, index=False)
    fiber.to_csv(fibers, index=False)
    dino.to_csv(dinos, index=False)
    folder = tmp_path / "results"
    loaded = ["y", str(fibers), "fiber", "y", str(dinos), "DINO", "n", ""]
    result, text = _guide(str(cells), folder, [*loaded, *QUESTIONS], run_report=False)
    assert "fiber: 2 feature columns, explainable, checked." in text
    tags = {e["name"]: e["tag"] for e in result.answers.dataset["feature_tables"]}
    assert tags == {"fiber": "explainable", "DINO": "non-explainable"}
    assert result.setup.explainable_tables == {"fiber": ("length", "fiber: area")}
    assert {"length", "fiber: area"} <= set(result.setup.features)
    assert "fiber (explainable, " in "\n".join(result.answers.lines())
    # The same tables and tags: the saved answers are offered again.
    again, text = _guide(str(cells), folder, [*loaded, ""], run_report=False)
    assert "were found for this table" in text
    assert again.setup.explainable_tables == result.setup.explainable_tables


def test_answers_saved_without_tags_read_as_non_explainable(tmp_path):
    objects, _, _, _, dino = _cells()
    _guide(objects, tmp_path, QUESTIONS, run_report=False, feature_tables={"DINO": dino})
    path = tmp_path / ANSWERS_FILE
    saved = json.loads(path.read_text())
    for entry in saved["dataset"]["feature_tables"]:
        del entry["tag"]  # as saved before 2026-10-06
    path.write_text(json.dumps(saved))
    result, text = _guide(objects, tmp_path, [""], run_report=False, feature_tables={"DINO": dino})
    assert "were found for this table" in text
    assert result.setup.feature_tables == {"DINO": ("f0", "f1", "f2", "f3")}
    assert "DINO (non-explainable, a DataFrame)" in "\n".join(result.answers.lines())


def test_the_page_lists_features_under_their_table_and_its_tables_name_it(tagged, tmp_path):
    objects, fovs, result, tables = tagged
    page = _quiet(html_report, result, objects, fovs, tmp_path, feature_tables=tables)
    text = page.read_text()
    assert '<span class="tab-group" role="presentation">CellProfiler</span>' in text
    for name in ("tests_across_repeats.csv", "tests_cell_level.csv", "heatmap_d.csv"):
        frame = pd.read_csv(next(tmp_path.rglob(name)))
        assert list(frame.columns)[:2] == ["feature", "table"], name
        assert set(frame["table"]) == {"observation table", "fiberfeatures", "CellProfiler"}


def test_a_rule_names_the_table_of_each_feature():
    owner = {"length": "fiberfeatures", "int0": "CellProfiler"}
    terms = (("length", ">", 5.0), ("int0", "<=", 2.0))
    assert rule_tables(terms, owner) == "length from fiberfeatures, int0 from CellProfiler"
    row = pd.Series(
        {"rule": "length > 5 and int0 ≤ 2", "rule_recall": np.nan, "rule_tables": "x from y"}
    )
    assert rule_words(row, "cells") == "picked by length > 5 and int0 ≤ 2, x from y"


# --- 6. Normalisation ------------------------------------------------------------------------


def test_explainable_tables_are_normalised_as_features_and_embeddings_centred_as_tables(tagged):
    objects, fovs, result, tables = tagged
    attached_objects = attach_feature_tables(objects, fovs, tables).objects
    setup = replace(result.setup, normalize="control_mean")
    divided = setup.prepared(attached_objects, fovs)
    control = attached_objects["condition"].eq("ctrl") & attached_objects["biological_repeat"].eq(
        "R0"
    )
    assert divided.loc[control, "int0"].mean() == pytest.approx(1.0)
    assert divided["f0"].equals(attached_objects["f0"])  # an embedding is never divided
    centred = centred_tables(attached_objects, fovs, setup)
    assert centred["int0"].equals(attached_objects["int0"])  # only the other tables centred
    assert not centred["f0"].equals(attached_objects["f0"])
