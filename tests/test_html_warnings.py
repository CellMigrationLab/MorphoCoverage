"""The page silences only the repeated data-check warning while it builds the figures and maps
(owner, 2026-10-12): any other warning raised there reaches the user."""

import importlib
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "scenarios"))

import make_scenarios

from morphocoverage.workflows import Setup, report

PAGE = importlib.import_module("morphocoverage.workflows.html_report")


@pytest.fixture(scope="module")
def built():
    scenario = make_scenarios.load("a06_expl_none_image_none_tol")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            scenario.objects,
            scenario.fovs,
            Setup(**scenario.setup),
            feature_tables=scenario.feature_tables,
        )
    return scenario, result


def _page(built, tmp_path, monkeypatch, message):
    scenario, result = built
    original = PAGE.figure_legends

    def legends(*args, **kwargs):
        warnings.warn(message, UserWarning, stacklevel=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(PAGE, "figure_legends", legends)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PAGE.html_report(
            result,
            scenario.objects,
            scenario.fovs,
            tmp_path,
            feature_tables=scenario.feature_tables,
        )
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_another_warning_while_building_the_figures_reaches_the_user(built, tmp_path, monkeypatch):
    shown = _page(built, tmp_path, monkeypatch, "a figure could not be drawn")
    assert "a figure could not be drawn" in shown


def test_the_repeated_data_check_warning_is_silenced(built, tmp_path, monkeypatch):
    shown = _page(built, tmp_path, monkeypatch, "check_tables found 1 warning in these tables")
    assert not any(text.startswith("check_tables found") for text in shown)
