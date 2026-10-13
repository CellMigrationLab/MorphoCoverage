"""The scenarios of ``data/scenarios`` run as a user would run them, for the matrix and pages."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from morphocoverage.workflows import Setup, html_report, report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios

NAMES = make_scenarios.names()


# The CSVs whose rows name a feature, which name its table with explainable tables.
FEATURE_CSVS = ("tests_", "heatmap_d", "_pairs", "pca_loadings_")


def _report(loaded: Any, setup: Setup) -> Any:
    """``report`` on the scenario's tables, its warnings kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(loaded.objects, loaded.fovs, setup, feature_tables=loaded.feature_tables)


def _page(result: Any, loaded: Any, truth: dict[str, Any], folder: Path) -> None:
    """The page is built, and its CSVs name each feature's table with explainable tables."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(
            result, loaded.objects, loaded.fovs, folder, feature_tables=loaded.feature_tables
        )
    assert page.exists()
    named = []
    for path in sorted((folder / "figures").glob("*.csv")):
        if not any(part in path.name for part in FEATURE_CSVS):
            continue
        columns = pd.read_csv(path, nrows=0).columns
        if "feature" in columns:
            named.append((path.name, "table" in columns))
    if truth.get("table_column"):
        assert named and all(has for _, has in named), named
    else:
        assert not any(has for _, has in named), named
