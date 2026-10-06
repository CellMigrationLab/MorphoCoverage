"""Every scenario of ``data/scenarios`` run as a user would run it, and its truth checked.

``data/scenarios/make_scenarios.py`` builds one small acquisition per kind of input: the
axes of what a user provides (tables, controls, normalisation, unit, wells, targets, every
pair of values covered), designs at the edges, and the mistakes users make. Each folder's
``scenario.json`` holds the Setup as plain JSON and the truth. The test builds the Setup,
runs ``report`` (and ``html_report`` for a covering subset: every kind of table, several
controls, normalisation and a rare label; ``test_scenarios_pages.py`` builds the page of every
other scenario that runs) and checks the truth: no crash, each refusal with
its message, the analyses that must run and those that must not, the reason each question
not applicable gives, the ``table`` column of the CSVs with explainable tables, and the
planted effects where the design makes the answer near-certain. It checks structure and
reasons, not numbers.
"""

from __future__ import annotations

import dataclasses
import itertools
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from morphocoverage.sampling_unit.contrast_repeats import TREND_REPEATS
from morphocoverage.sampling_unit.reading_text import CONFIRMED, CONSISTENT
from morphocoverage.workflows import Setup, html_report, optimise, report
from morphocoverage.workflows._enough_targets import NO_TARGET
from morphocoverage.workflows._html_tests import across_frame

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"
sys.path.insert(0, str(SCENARIOS))
try:
    import make_scenarios
finally:
    sys.path.remove(str(SCENARIOS))

NAMES = make_scenarios.names()
ALPHA = 0.05
# The CSVs whose rows name a feature, which name its table with explainable tables.
FEATURE_CSVS = ("tests_", "heatmap_d", "_pairs", "pca_loadings_")


def _report(loaded: Any, setup: Setup) -> Any:
    """``report`` on the scenario's tables, its warnings kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(loaded.objects, loaded.fovs, setup, feature_tables=loaded.feature_tables)


def _refused(loaded: Any, truth: dict[str, Any]) -> None:
    """The refusal expected, where it is expected, with its message."""
    error = {"ValueError": ValueError, "TypeError": TypeError}[truth["error"]]
    if truth["at"] == "setup":
        with pytest.raises(error, match=truth["match"]):
            Setup(**loaded.setup)
        return
    setup = Setup(**loaded.setup)
    with pytest.raises(error, match=truth["match"]):
        _report(loaded, setup)


def _sections(result: Any, truth: dict[str, Any]) -> None:
    """The analyses that run, those that do not, and why each question cannot be answered."""
    ran = {key for key, value in result.results.items() if value is not None}
    missing = sorted(set(truth.get("results", ())) - ran)
    assert not missing, f"must run: {missing}"
    extra = sorted(set(truth.get("absent", ())) & ran)
    assert not extra, f"must not run: {extra}"
    for question, reason in truth.get("not_applicable", {}).items():
        said = " | ".join(
            (result.not_assessed.get(question, ""), result.document.headlines.get(question, ""))
        )
        assert reason in said, f"{question}: {said!r} does not say {reason!r}"
    for question, part in truth.get("headlines", {}).items():
        assert part in result.document.headlines[question]
    for part in truth.get("overview", ()):
        assert part in result.document.overview
    # Flattened: a sentence wraps where its words fall, so a part may straddle two lines.
    full = " ".join(result.summary("full").split()) if "full" in truth else ""
    for part in truth.get("full", ()):
        assert part in full, part
    if "controls_status" in truth:
        assert result.results["controls"].status == truth["controls_status"]
    if "contrast_kinds" in truth:
        kinds = set(result.setup.contrast_table["kind"])
        assert kinds == set(truth["contrast_kinds"])
    if "separation_spaces" in truth:
        spaces = set(result.results["separation"].table["space"])
        assert spaces == set(truth["separation_spaces"])


def _tests(result: Any, truth: dict[str, Any]) -> None:
    """The tests across repeats: what is tested, what is found, what is not."""
    across = across_frame(result)
    whole = result.results.get("whole_table")
    if not across.empty:  # the trend line from 6 repeats only (owner, 2026-10-06)
        trend = across["verdict"] == CONSISTENT
        assert (across.loc[trend, "repeats"] >= TREND_REPEATS).all()
    if whole is not None and "same way in every" in result.document.headlines["conditions"]:
        assert (whole.tests["repeats"] >= TREND_REPEATS).any()
    if truth.get("untested_across"):
        assert across.empty or across["adjusted_p"].isna().all()
    if "untested_whole" in truth:
        statuses = whole.tests["status"]
        assert (statuses != "tested").all()
        assert statuses.str.contains(truth["untested_whole"]).all()
    if truth.get("left_out"):
        assert (whole.tests["left_out"] > 0).all()
    if "compared" in truth:
        assert across["condition"].nunique() == truth["compared"]
    if "tested_features" in truth:
        tested = set(result.discovery.tests["feature"])
        assert set(truth["tested_features"]) <= tested
    if "balanced" in truth:
        cells = result.results["contrast"].cells.set_index("condition")
        for condition in truth["balanced"]:
            assert cells.loc[condition, "unbalanced"]
            assert pd.notna(cells.loc[condition, "p_balanced"])
    for feature, conditions in truth.get("planted", {}).items():
        for condition in conditions:
            row = across[(across["feature"] == feature) & (across["condition"] == condition)]
            assert len(row) == 1 and row["adjusted_p"].iloc[0] <= ALPHA
            assert (row["higher"] == row["repeats"]).all()
            assert row["verdict"].iloc[0] == CONFIRMED
    for feature, conditions in truth.get("trend", {}).items():
        for condition in conditions:
            row = across[(across["feature"] == feature) & (across["condition"] == condition)]
            assert len(row) == 1 and row["adjusted_p"].iloc[0] > ALPHA
            assert row["verdict"].iloc[0] == CONSISTENT
    if not across.empty:
        nulls = across["condition"].isin(truth.get("null_conditions", ()))
        nulls |= across["feature"].isin(truth.get("null_features", ()))
        assert not (across.loc[nulls, "adjusted_p"] <= ALPHA).any()
        assert not across.loc[nulls, "verdict"].isin([CONFIRMED, CONSISTENT]).any()
    for table, conditions in truth.get("whole_table_planted", {}).items():
        rows = whole.tests[whole.tests["table"] == table].set_index("condition")
        assert all(rows.loc[c, "q_repeats"] <= ALPHA for c in conditions)
        null = rows.index.isin(truth.get("null_conditions", ()))
        assert not (rows.loc[null, "q_repeats"] <= ALPHA).any()


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


@pytest.mark.parametrize("name", NAMES)
def test_the_scenario_gives_its_truth(name: str, tmp_path: Path) -> None:
    loaded = make_scenarios.load(name)
    truth = loaded.truth
    if truth["outcome"] == "refused":
        _refused(loaded, truth)
        return
    result = _report(loaded, Setup(**loaded.setup))
    _sections(result, truth)
    _tests(result, truth)
    if truth.get("order_free"):
        tables = dict(reversed(loaded.feature_tables.items()))
        reversed_order = dataclasses.replace(loaded, feature_tables=tables)
        again = _report(reversed_order, Setup(**loaded.setup))
        pd.testing.assert_frame_equal(
            again.results["separation"].table, result.results["separation"].table
        )
    if truth.get("html"):
        _page(result, loaded, truth, tmp_path)


def test_the_whole_distribution_alone_runs_when_its_fingerprint_is_not_assessable() -> None:
    # a08 with the guided session's "whole distribution" aim, tolerances=(): its groups of 4
    # images are too small for the components; the report says so, it does not crash.
    loaded = make_scenarios.load("a08_embed1_several_norm_image_tech_fp_dino")
    setup = Setup(**{**loaded.setup, "tolerances": ()})
    result = _report(loaded, setup)
    enough = " ".join(result.document.sections["enough"].split())
    assert "not assessable: its median group holds 4 images" in enough
    assert NO_TARGET in enough
    merged = loaded.objects.merge(loaded.feature_tables["DINO1"])
    with pytest.raises(ValueError, match="No declared target can be judged") as error:
        optimise(merged, loaded.fovs, setup, fov_counts=[2], reservoir_fraction=0.5)
    assert "not assessable: its median group holds 4 images" in str(error.value)


def test_the_page_says_nothing_was_compared_when_no_repeat_holds_a_condition_and_its_control(
    tmp_path: Path,
) -> None:
    # e02: the discovery compares nothing, so its ranking is empty; the page was refused a
    # KeyError on its missing score column. It builds and says what was not compared.
    loaded = make_scenarios.load("e02_control_in_other_repeats")
    result = _report(loaded, Setup(**loaded.setup))
    assert result.discovery.top() == []
    _page(result, loaded, {"table_column": False}, tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert "Not compared: a (no " in page


def test_the_committed_scenarios_are_what_the_generator_writes(tmp_path: Path) -> None:
    built = make_scenarios.main(tmp_path)
    assert make_scenarios.names(tmp_path) == NAMES
    for folder in sorted(tmp_path.iterdir()):
        written = sorted(p.name for p in folder.iterdir())
        assert written == sorted(p.name for p in (SCENARIOS / folder.name).iterdir())
        for file in written:
            assert (folder / file).read_bytes() == (SCENARIOS / folder.name / file).read_bytes()
    assert make_scenarios.readme_rows(built) in (SCENARIOS / "README.md").read_text()


def test_the_axes_cover_every_pair_of_values_at_least_once_and_each_value_twice() -> None:
    cases = [make_scenarios.load(n).spec["axes"] for n in NAMES if n.startswith("a")]
    values = {axis: {str(c[axis]) for c in cases} for axis in cases[0]}
    for axis, seen in values.items():
        for value in seen:
            assert sum(str(c[axis]) == value for c in cases) >= 2, (axis, value)
    covered = {(a, str(c[a]), b, str(c[b])) for c in cases for a, b in itertools.combinations(c, 2)}
    for a, b in itertools.combinations(values, 2):
        for va, vb in itertools.product(values[a], values[b]):
            if _allowed({a: va, b: vb}):
                assert (a, va, b, vb) in covered, (a, va, b, vb)


def _allowed(pair: dict[str, str]) -> bool:
    """Whether two axis values can go together: normalisation and a positive control need a
    control, and a positive control or a fingerprint on explainable features need them."""
    embedding = pair.get("tables", "").startswith("embed")
    if pair.get("normalize") == "True" and pair.get("controls") == "none":
        return False
    if pair.get("target") == "positive" and (pair.get("controls") == "none" or embedding):
        return False
    return not (pair.get("target") == "fp_expl" and embedding)
