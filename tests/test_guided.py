"""The guided session: load, scan, ask, save the answers, continue or start over, report
(workflows.guided)."""

import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

from morphocoverage.workflows import Answers, guide, load_answers, save_answers
from morphocoverage.workflows._guided_ask import _selection
from morphocoverage.workflows.guided import ANSWERS_FILE, table_checksum


def _cells(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for condition, shift in (("siCTRL", 0), ("siA#1", 8), ("siA#2", 9)):
        for repeat in ("2026_01_01", "2026_02_01", "2026_03_01"):
            for field in range(4):
                for label in range(6):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(100 + shift, 10),
                            "intensity": rng.normal(5, 1),
                            "aspect": rng.normal(2, 0.2),
                        }
                    )
    return pd.DataFrame(rows)


class Script:
    """Answers the prompts in order and records what was said."""

    def __init__(self, answers):
        self.answers, self.said, self.prompts = list(answers), [], []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"no scripted answer left for {prompt!r}")
        return self.answers.pop(0)

    def say(self, text):
        self.said.append(text)

    @property
    def text(self):
        return "\n".join(self.said)


def _run(objects, folder, answers, **kwargs):
    script = Script(answers)
    result = guide(objects, folder=folder, ask=script.ask, say=script.say, **kwargs)
    return result, script


def test_the_questions_are_asked_and_the_answers_saved(tmp_path):
    # loaded? control (default: siCTRL), positive control, aim, features, readout, unit
    # (default), tolerance
    result, script = _run(
        _cells(), tmp_path, ["y", "", "siA#1", "both", "1,3", "area", "", "15%"], run_report=False
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
    _run(_cells(), tmp_path, ["y", "", "", "", "all", "", ""], run_report=False)
    result, script = _run(_cells(), tmp_path, [""], run_report=False)
    assert "were found for this table" in script.text
    assert result.answers.aim == "population" and result.setup.fingerprint == ()
    # Start over: the file is deleted and every question is asked again.
    result, script = _run(
        _cells(), tmp_path, ["start over", "y", "4", "", "intensity", ""], run_report=False
    )
    assert "The saved answers were deleted" in script.text
    assert result.answers.control is None and result.answers.features == ["intensity"]
    assert load_answers(tmp_path).features == ["intensity"]


def test_answers_for_another_table_are_never_reused(tmp_path):
    _run(_cells(), tmp_path, ["y", "", "", "", "all", "", ""], run_report=False)
    other = _cells(seed=1)
    result, script = _run(other, tmp_path, ["n"], run_report=False)
    assert "cannot be reused for this one" in script.text
    assert result.stopped.endswith("choose another folder for this table")
    assert (tmp_path / ANSWERS_FILE).exists()
    _run(other, tmp_path, ["y", "y", "", "", "", "1", ""], run_report=False)
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
            str(folder),
            "maybe",
            "y",
            "9",
            "",
            "",
            "2",
            "0",
            "1-2",
            "",
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
    _, script = _run(_cells(), tmp_path, ["y", "y", "", "", "", "1", ""], run_report=False)
    assert "cannot be read" in script.text
    assert load_answers(tmp_path).features == ["area"]


def test_the_session_ends_with_the_report_saved(tmp_path):
    # A figure of an earlier session in this folder is not this report's.
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "stale.png").write_bytes(b"png")
    os.utime(tmp_path / "figures" / "stale.png", (time.time() - 3600,) * 2)
    result, script = _run(_cells(), tmp_path, ["y", "", "", "", "all", "", ""])
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
