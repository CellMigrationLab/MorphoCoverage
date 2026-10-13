"""The guided session: load, scan, ask, save the answers, continue or start over, report
(workflows.guided)."""

import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
import pytest
from _helpers_guided import Script, _cells

from morphocoverage.workflows import Answers, guide, load_answers, save_answers
from morphocoverage.workflows._guided_ask import _selection
from morphocoverage.workflows._guided_tables import entries
from morphocoverage.workflows.guided import ANSWERS_FILE, table_checksum


def _run(objects, folder, answers, **kwargs):
    script = Script(answers)
    result = guide(objects, folder=folder, ask=script.ask, say=script.say, **kwargs)
    return result, script


def test_the_questions_are_asked_and_the_answers_saved(tmp_path):
    # loaded? unit (default), one control? control (default: siCTRL), positive control,
    # aim, features, readout, tolerance
    result, script = _run(
        _cells(),
        tmp_path,
        ["y", "", "", "", "siA#1", "both", "1,3", "area", "15%"],
        run_report=False,
    )
    answers = result.answers
    assert answers.control == {"condition": "siCTRL"}
    assert answers.positive == {"condition": "siA#1"}
    assert result.setup.positive.selector == {"condition": "siA#1"}
    assert answers.aim == "both"
    assert answers.features == ["area", "aspect"]
    assert answers.readout == "area"
    assert answers.observation_unit == "cell"
    assert answers.fingerprint_tolerance == pytest.approx(0.15)
    saved = json.loads((tmp_path / ANSWERS_FILE).read_text())
    assert saved["dataset"]["checksum"] == table_checksum(_cells())
    assert load_answers(tmp_path) == answers
    assert (tmp_path / "scan.txt").exists()
    # The guided session prints the scan without its API questions, then asks them itself.
    assert "Before the analysis" not in script.text
    assert "Design scan" in script.text
    setup = result.setup
    assert setup.readout == "area" and setup.features == ("area", "aspect")
    assert setup.fingerprint[0].threshold == pytest.approx(0.15)
    assert setup.novelty is not None


def test_saved_answers_are_continued_or_deleted(tmp_path):
    _run(_cells(), tmp_path, ["y", "", "", "", "", "", "all", ""], run_report=False)
    result, script = _run(_cells(), tmp_path, [""], run_report=False)
    assert "were found for this table" in script.text
    assert result.answers.aim == "population" and result.setup.fingerprint == ()
    # Start over: the file is deleted and every question is asked again.
    result, script = _run(
        _cells(), tmp_path, ["start over", "y", "", "", "4", "", "intensity"], run_report=False
    )
    assert "The saved answers were deleted" in script.text
    assert result.answers.control is None and result.answers.features == ["intensity"]
    assert load_answers(tmp_path).features == ["intensity"]


def test_answers_for_another_table_are_never_reused(tmp_path):
    _run(_cells(), tmp_path, ["y", "", "", "", "", "", "all", ""], run_report=False)
    other = _cells(seed=1)
    result, script = _run(other, tmp_path, ["n"], run_report=False)
    assert "cannot be reused for this one" in script.text
    assert result.stopped.endswith("choose another folder for this table")
    assert (tmp_path / ANSWERS_FILE).exists()
    _run(other, tmp_path, ["y", "y", "", "", "", "", "", "1"], run_report=False)
    assert load_answers(tmp_path).dataset["checksum"] == table_checksum(other)


def test_a_dataset_that_did_not_load_stops_without_saving(tmp_path):
    result, script = _run(_cells(), tmp_path, ["n"], run_report=False)
    assert result.answers is None and result.stopped
    assert "Fix the tables" in script.text
    assert not (tmp_path / ANSWERS_FILE).exists()


def test_paths_and_the_folder_are_asked_and_bad_answers_asked_again(tmp_path):
    path = tmp_path / "cells.csv"
    _cells().to_csv(path, index=False)
    folder = tmp_path / "results"
    script = Script(
        [
            str(path),
            "",
            "",  # no other feature tables (owner, 2026-10-05)
            str(folder),
            "maybe",
            "y",
            "",
            "",
            "9",
            "",
            "",
            "2",
            "0",
            "1-2",
            "",
            "200",
            "0.2",
        ]
    )
    result = guide(ask=script.ask, say=script.say, run_report=False)
    assert result.folder == folder
    assert result.answers.dataset["objects"] == str(path)
    assert result.answers.features == ["area", "intensity"]
    assert result.answers.fingerprint_tolerance == pytest.approx(0.2)
    assert script.text.count("Please answer again.") == 4


def test_selections():
    options = ["a", "b", "c", "d"]
    assert _selection("all", options) == options
    assert _selection("4, 1-2", options) == ["a", "b", "d"]
    assert _selection("C,a", options) == ["a", "c"]
    assert "outside" in _selection("2-9", options)
    assert "not one of the options" in _selection("e", options)


def test_answers_are_checked():
    with pytest.raises(ValueError, match="readout"):
        Answers(True, None, "population", ["area"], "intensity", "cell")
    with pytest.raises(ValueError, match="fingerprint_tolerance"):
        Answers(True, None, "fingerprint", ["area"], "area", "cell")
    with pytest.raises(ValueError, match="aim must be"):
        Answers(True, None, "rare", ["area"], "area", "cell")


def test_an_unreadable_answers_file_is_offered_for_deletion(tmp_path):
    (tmp_path / ANSWERS_FILE).write_text("{not json")
    with pytest.raises(ValueError, match="not a saved-answers file"):
        load_answers(tmp_path)
    _, script = _run(_cells(), tmp_path, ["y", "y", "", "", "", "", "", "1"], run_report=False)
    assert "cannot be read" in script.text
    assert load_answers(tmp_path).features == ["area"]


def test_the_session_ends_with_the_report_saved(tmp_path):
    # A figure of an earlier session in this folder is not this report's.
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "stale.png").write_bytes(b"png")
    os.utime(tmp_path / "figures" / "stale.png", (time.time() - 3600,) * 2)
    result, script = _run(_cells(), tmp_path, ["y", "", "", "", "", "", "all", "", ""])
    with zipfile.ZipFile(tmp_path / f"{tmp_path.name}.zip") as archive:
        names = archive.namelist()
    assert f"{tmp_path.name}/figures/area.png" in names
    assert f"{tmp_path.name}/figures/stale.png" not in names
    for name in ("scan.txt", "report_short.txt", "report_full.txt", ANSWERS_FILE):
        assert (tmp_path / name).exists()
    # The page, with the readout and the best-ranked features, then the PCA and UMAP of all
    # three, raw and normalised, each as PNG and PDF.
    assert "report.html, one file to open with a double click" in script.text
    assert (tmp_path / f"{tmp_path.name}.zip").exists()
    page = (tmp_path / "report.html").read_text()
    assert "All 3 features at once" in page and "embedding-data" in page
    for name in ("area", "pca_raw", "umap_raw", "pca_normalised", "umap_normalised"):
        for kind in ("png", "pdf"):
            assert (tmp_path / "figures" / f"{name}.{kind}").exists()
    assert result.report.discovery.top()[0] == "area"
    assert "Microscopist's report" in (tmp_path / "report_short.txt").read_text()
    assert result.report is not None
    save_answers(tmp_path, result.answers)
    assert load_answers(tmp_path).readout == "area"


POPULATION = ["y", "", "", "", "", "", "all", ""]  # loaded, unit, one control?, ...


def _fields(objects):
    """The FOV table of ``objects``: one row per field with its condition and repeat."""
    columns = ["image_file", "condition", "biological_repeat"]
    return objects[columns].drop_duplicates().reset_index(drop=True)


def _guided(objects, fovs, folder, answers):
    script = Script(answers)
    result = guide(objects, fovs, folder=folder, ask=script.ask, say=script.say, run_report=False)
    return result, script


def test_357_saved_answers_need_the_same_fov_table_too(tmp_path):
    cells = _cells()
    fovs = _fields(cells)
    _guided(cells, fovs, tmp_path, POPULATION)
    saved = json.loads((tmp_path / ANSWERS_FILE).read_text())
    assert saved["version"] == 2 and saved["dataset"]["fovs_checksum"] == table_checksum(fovs)
    _, script = _guided(cells, fovs.copy(), tmp_path, [""])
    assert "were found for this table" in script.text
    moved = fovs.copy()
    moved.loc[0, "biological_repeat"] = "2026_02_01"
    empty = pd.concat([fovs, fovs.iloc[[0]].assign(image_file="empty.tif")], ignore_index=True)
    for other in (moved, empty, None):
        result, script = _guided(cells, other, tmp_path, ["n"])
        assert "cannot be reused for this one" in script.text
        assert result.stopped.endswith("choose another folder for this table")


def test_357_answers_saved_without_a_fov_table_are_not_reused_with_one(tmp_path):
    cells = _cells()
    _guided(cells, None, tmp_path, POPULATION)
    assert load_answers(tmp_path).dataset["fovs_checksum"] is None
    result, script = _guided(cells, _fields(cells), tmp_path, ["n"])
    assert "cannot be reused for this one" in script.text and result.stopped


def test_357_the_same_tables_under_other_paths_are_the_same_dataset(tmp_path):
    cells = _cells()
    paths = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        cells.to_csv(folder / "cells.csv", index=False)
        _fields(cells).to_csv(folder / "fovs.csv", index=False)
        paths.append((folder / "cells.csv", folder / "fovs.csv"))
    results = tmp_path / "results"
    # Given as paths, the session asks for other feature tables first: none here.
    _guided(*paths[0], results, ["", *POPULATION])
    result, script = _guided(*paths[1], results, ["", ""])
    assert "were found for this table" in script.text and result.answers is not None


def test_357_answers_of_an_earlier_version_are_asked_again_once(tmp_path):
    cells = _cells()
    _guided(cells, None, tmp_path, POPULATION)
    path = tmp_path / ANSWERS_FILE
    old = json.loads(path.read_text())
    old["version"] = 1
    del old["dataset"]["fovs_checksum"]
    path.write_text(json.dumps(old))
    result, script = _guided(cells, None, tmp_path, POPULATION)
    assert "saved by an earlier version, which did not check the FOV table" in script.text
    assert result.answers is not None and load_answers(tmp_path).version == 2
    _, script = _guided(cells, None, tmp_path, [""])
    assert "were found for this table" in script.text


# --- Other feature tables (owner, 2026-10-05) --------------------------------------------


def _embedding(cells, drop=0, seed=3):
    key = cells[["condition", "biological_repeat", "image_file", "label_id"]]
    rng = np.random.default_rng(seed)
    return key.assign(**{f"dino_{j}": rng.normal(size=len(key)) for j in range(4)}).iloc[drop:]


def test_other_feature_tables_are_asked_for_checked_named_and_saved(tmp_path):
    cells = _cells()
    path, good, short = tmp_path / "cells.csv", tmp_path / "dino.csv", tmp_path / "short.csv"
    cells.to_csv(path, index=False)
    _embedding(cells).to_csv(good, index=False)
    _embedding(cells, drop=2).to_csv(short, index=False)
    # Each table's tag asked after its name, "no" (non-explainable) by default (2026-10-06).
    tables = [
        *("y", str(tmp_path / "missing.csv"), str(short), "short", ""),
        *(str(good), "DINOv2", "", ""),
    ]
    questions = ["y", "", "", "", "", "", "all", ""]
    result, script = _run(str(path), tmp_path / "results", tables + questions, run_report=False)
    assert "other feature tables for the same observations" in script.prompts[0]
    assert "It cannot be read" in script.text
    assert "'short' lacks a row for 2 observations" in script.text
    assert "DINOv2: 4 feature columns, non-explainable, checked." in script.text
    entry = result.answers.dataset["feature_tables"]
    # The checksum of the rows sorted by their identity (owner, 2026-10-06).
    checksum = entry[0]["checksum"]
    assert entry == [
        {"name": "DINOv2", "path": str(good), "tag": "non-explainable", "checksum": checksum}
    ]
    tables = {"DINOv2": (pd.read_csv(good), None, "non-explainable")}
    assert checksum == entries(tables, cells)[0]["checksum"]
    assert result.setup.feature_tables == {"DINOv2": tuple(f"dino_{j}" for j in range(4))}
    # Listed in the scan, which is printed before "Did your dataset load correctly?".
    scan = (tmp_path / "results" / "scan.txt").read_text()
    line = "Other feature table DINOv2 (non-explainable): 4 feature columns, every observation"
    assert line in scan
    assert script.text.index("Other feature table DINOv2") < script.text.index("Loaded correctly?")
    assert "Other feature tables: DINOv2" in "\n".join(result.answers.lines())


def test_saved_answers_are_reused_only_with_the_same_other_tables(tmp_path):
    cells = _cells()
    first = {"DINOv2": _embedding(cells)}
    _run(
        cells,
        tmp_path,
        ["y", "", "", "", "", "", "all", ""],
        run_report=False,
        feature_tables=first,
    )
    result, script = _run(cells, tmp_path, [""], run_report=False, feature_tables=first)
    assert "were found for this table" in script.text and result.setup.feature_tables
    other = {"DINOv2": _embedding(cells, seed=4)}
    result, script = _run(cells, tmp_path, ["n"], run_report=False, feature_tables=other)
    assert "cannot be reused for this one" in script.text and result.answers is None
    # The refusal names the table that changed (owner, 2026-10-06).
    assert "what differs: the feature table 'DINOv2'." in " ".join(script.text.split())


def test_saved_answers_are_reused_for_a_table_in_another_row_order(tmp_path):
    cells = _cells()
    first = {"DINOv2": _embedding(cells)}
    _run(
        cells,
        tmp_path,
        ["y", "", "", "", "", "", "all", ""],
        run_report=False,
        feature_tables=first,
    )
    shuffled = {"DINOv2": first["DINOv2"].sample(frac=1, random_state=0)}
    result, script = _run(cells, tmp_path, [""], run_report=False, feature_tables=shuffled)
    assert "were found for this table" in script.text and result.setup.feature_tables


def test_a_fingerprint_with_other_tables_has_one_criterion_per_table():
    answers = Answers(
        loaded_correctly=True,
        control=None,
        aim="fingerprint",
        features=["area", "aspect"],
        readout="area",
        observation_unit="cell",
        fingerprint_tolerance=0.2,
    )
    # Owner, 2026-10-06: the explainable features and each table apart, each its verdict.
    explainable, dino = answers.fingerprint({"DINOv2": tuple(f"d{i}" for i in range(8))})
    assert explainable.metric.features == ("area", "aspect")
    assert dino.metric.features == tuple(f"d{i}" for i in range(8))
    assert dino.metric.weights is None and dino.threshold == explainable.threshold == 0.2
    assert dino.description == "the covariance of the standardised DINOv2 columns within 20%"
    assert answers.fingerprint() == (explainable,)
