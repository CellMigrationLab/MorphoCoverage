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
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
from _helpers_scenarios import NAMES, _page, _report

matplotlib.use("Agg")

import pandas as pd
import pytest

from morphocoverage._style import Status, width
from morphocoverage.retrospective.report_status_conditions import NOTHING_COMPARED
from morphocoverage.sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS
from morphocoverage.sampling_unit.contrast_text import ONE_UNIT
from morphocoverage.sampling_unit.reading_text import CONFIRMED, CONSISTENT, NO_CONTROL
from morphocoverage.sampling_unit.repeat_plan import repeats_needed
from morphocoverage.workflows import Setup, optimise
from morphocoverage.workflows._enough_targets import NO_TARGET
from morphocoverage.workflows._html_map_text import NEIGHBOURS, ONE_CONDITION
from morphocoverage.workflows._html_tests import across_frame

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"
sys.path.insert(0, str(SCENARIOS))
try:
    import make_scenarios
finally:
    sys.path.remove(str(SCENARIOS))

ALPHA = 0.05


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


def _words(result: Any) -> None:
    """The report's words agree with what ran (audit 10, owner, 2026-10-06): what is said
    once is said once (A2, B12), and no sentence contradicts the analysis it describes (T4)."""
    full = " ".join(result.summary("full").split())
    assert full.count("never a biological verdict: open their images") <= 1
    assert full.count("may have been imaged or processed differently") <= 1
    contrast = result.results.get("contrast")
    table = contrast.table if contrast is not None else pd.DataFrame({"status": []})
    statuses = table["status"].astype(str)
    if "one well in each repeat" in full:  # A6, C5: the reason is the status's
        assert statuses.str.startswith(ONE_UNIT).any()
    for raw in (result.summary(), result.summary("full")):  # A22: prose wrapped, tables apart
        assert all(width(line) <= 100 or "  " in line.strip() for line in raw.splitlines())
    assert full.count("Shown: the 95% interval") <= 1  # A22: said once, not per feature
    across = across_frame(result)
    if not across.empty and across["repeats"].max() != 3:  # C24: the 3-repeat clause only at 3
        assert "with 3 biological repeats only large" not in full
        assert "with 3 repeats only large" not in full
    short = " ".join(result.summary().split())
    for text in (short, full):
        assert not re.search(r"\b0 vs 0\b", text)  # B6, C7: no own control, never 0 vs 0
        assert "{'" not in text  # A22, C8, C22: no dict repr
        assert not ("No condition to compare" in text and "Not compared:" in text)  # C8
        assert text.count("Which features separate the conditions?") <= 1  # A9
        assert not re.search(r"\d+ of \d+ features and the", text)  # C20: 4 of 3 features
        assert "any of all" not in text  # C19
        assert "no spatial autocorrelation (" not in text  # A21: flagged at its threshold
        assert "fov_variability]" not in text  # A22: no internal names
        if result.document.units != "cells":  # C18: the unit's words, never cells
            assert "random cells" not in text and "not only cells" not in text
        assert ", with 2 biological repeats holding both sides" not in text  # C21: no "A, with"
        if result.setup.reference_control is None:  # C24: generic lines only where they apply
            assert "compared with its own control" not in text
        if result.setup.rare_label is None:
            assert "and the rare label)" not in text
        if result.setup.novelty is None:
            assert "No representation control" not in text
        if not result.setup.columns and result.setup.normalize:  # T10, B7, C11: centred
            assert "as normalised" not in text
        if result.not_assessed.get("profile"):  # T4, C13: the reason, never what to run
            assert "needs: run representation_control" not in text
        if "also given balanced" in text:  # C6: only beside a balanced test
            assert contrast.cells["p_balanced"].notna().any()
        if statuses.str.startswith(NO_CONTROL).any():  # B4, C9: the reason is the control
            assert "has a value of every feature" not in text


def _tolerance_counts(result: Any) -> None:
    """T1: "within its tolerance in N of M" says the fingerprint's own counts, per criterion:
    N groups meeting it of M judged (audit 10)."""
    precision = result.results.get("fingerprint_precision")
    headline = result.document.headlines.get("enough", "")
    said = {
        (int(n), int(m)) for n, m in re.findall(r"within its tolerance in (\d+) of (\d+)", headline)
    }
    if precision is None or not said:
        return
    judged = precision[precision["met"].notna()]
    counted = {
        (int(rows["met"].astype(bool).sum()), len(rows))
        for _, rows in judged.groupby("criterion", sort=False)
    }
    assert said <= counted, (said, counted)


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
    headline = result.document.headlines.get("conditions", "")
    if "beyond the mean" in headline:  # C27: the exploratory part marked at a glance
        assert "beyond the mean (exploratory)" in headline
    for feature, conditions in truth.get("trend", {}).items():
        for condition in conditions:
            row = across[(across["feature"] == feature) & (across["condition"] == condition)]
            assert len(row) == 1 and row["adjusted_p"].iloc[0] > ALPHA
            assert row["verdict"].iloc[0] == CONSISTENT
            # T5, C15: the glance names the trend, never only the significant conditions.
            assert "trend: " in headline and f"{condition} higher in every repeat" in headline
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


@pytest.mark.parametrize("name", NAMES)
def test_the_scenario_gives_its_truth(name: str, tmp_path: Path) -> None:
    loaded = make_scenarios.load(name)
    truth = loaded.truth
    if truth["outcome"] == "refused":
        _refused(loaded, truth)
        return
    result = _report(loaded, Setup(**loaded.setup))
    _sections(result, truth)
    _words(result)
    _tolerance_counts(result)
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
    # Nothing was compared: ➖ with the reason, never "No consistent difference" (F12).
    assert result.document.headlines["conditions"] == NOTHING_COMPARED
    assert "Nothing compared" in page and "No consistent difference" not in page


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


def test_the_page_of_one_condition_says_there_is_nothing_to_separate(tmp_path: Path) -> None:
    # T6, C29: e03's page explained the separation line though one condition gives none.
    loaded = make_scenarios.load("e03_one_condition")
    result = _report(loaded, Setup(**loaded.setup))
    _page(result, loaded, {"table_column": False}, tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert ONE_CONDITION in page and NEIGHBOURS not in page


def test_the_rows_across_experiments_of_the_acquisition_block_scenario() -> None:
    # T7: e16's several controls give the main rows and the rows across experiments.
    loaded = make_scenarios.load("e16_acquisition_block")
    result = _report(loaded, Setup(**loaded.setup))
    kinds = set(result.setup.contrast_table["kind"])
    assert {"own_control", "across_normalised"} <= kinds


def test_a_positive_control_no_better_than_false_alarms_is_not_judged() -> None:
    # T3, C1: a02's positive control is recognized no more often than reference fields are
    # called new; the answer cannot tell them apart, it is not a weak representation.
    loaded = make_scenarios.load("a02_expl_several_norm_cells_tech_positive")
    result = _report(loaded, Setup(**loaded.setup))
    headline = result.document.headlines["controls"]
    assert "cannot tell the positive control from the reference" in headline
    assert result.document.statuses["controls"] is not Status.PROBLEM


def test_the_repeats_needed_are_never_fewer_than_the_paired_test_needs() -> None:
    # T8, C14: "Repeats needed ... 2" while the paired test needs 3.
    assert repeats_needed(5.0, 0.1, 0.05) >= MIN_REPEATS


def test_a_pure_shift_does_not_differ_beyond_the_mean() -> None:
    # T9, C2: a25's conditions are shifted on the log scale, nothing more; beyond the mean
    # reads the quantile shifts minus the mean shift, so no distribution differs.
    loaded = make_scenarios.load("a25_mix_several_norm_cells_tech_fp_expl")
    result = _report(loaded, Setup(**loaded.setup))
    headline = result.document.headlines["conditions"]
    assert "the distribution differs" not in headline
    assert "same way in every repeat" not in " ".join(result.summary("full").split())


def test_the_live_decision_with_no_target_says_nothing_decides_a_stop() -> None:
    # T4, C12: a10 declares no target, yet the declaration said a stop needs the fingerprint.
    loaded = make_scenarios.load("a10_embed2_several_norm_cells_none_tol")
    result = _report(loaded, Setup(**loaded.setup))
    short = " ".join(result.summary().split())
    assert "a stop needs every declared target (phenotypic fingerprint)" not in short
    assert "no target: nothing decides a stop" in short
