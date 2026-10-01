"""Readable reports, step R4: optimise, benchmark, the integrity summary and the data check.

Each summary opens with a status marker that follows its rule (``_style``: ✅
answered and good, ⚠️ look at this, ❌ a problem, ➖ not assessed), is short by
default with every caveat once, and keeps every number of the old text in
``detail="full"``.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import check_tables, integrity_checks
from morphocoverage._style import CAUTION, NOT_ASSESSED, OK, PROBLEM, Status
from morphocoverage.retrospective.frontier import BASELINE, FrontierResult
from morphocoverage.retrospective.frontier_text import frontier_status
from morphocoverage.retrospective.report import LIMITS_TITLE
from morphocoverage.retrospective.report_status import calibration_status
from morphocoverage.workflows import benchmark, default_families, optimise
from morphocoverage.workflows.benchmarks import Benchmark, grid_warnings
from morphocoverage.workflows.retrospective import CALIBRATION_SHORT, CALIBRATION_TITLE

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load
from test_integrity import FEATURES, _check, _field, _replace_field
from test_workflows import READOUT, _acquisition, _setup

OUT_OF_REACH = "Out of reach of these checks"
RESAMPLED = "treated as independent"


def _flat(text: str) -> str:
    """Text with its wrapping undone, for substrings that may span a wrapped line."""
    return " ".join(text.split())


# --- Integrity --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integrity():
    objects, fovs = load("idr0028_screenB").tables
    clean = _check(objects, fovs)
    source = objects[_field(objects, fovs, 0)]
    copied = _replace_field(objects, fovs, 40, source.copy())
    copied = _replace_field(copied, fovs, 41, source.sample(frac=1, random_state=0))
    twice = objects[_field(objects, fovs, 60)]
    copied = _replace_field(copied, fovs, 60, pd.concat([twice, twice]))
    return clean, _check(copied, fovs)


def test_integrity_marker_follows_the_findings(integrity):
    clean, dirty = integrity
    assert clean.status is Status.OK and clean.summary().startswith(f"{OK} Data integrity")
    assert dirty.status is Status.PROBLEM
    assert dirty.summary().startswith(f"{PROBLEM} Data integrity — 3 findings in 4 fields")
    with pytest.raises(ValueError, match="short"):
        clean.summary(detail="brief")


def test_integrity_short_groups_the_findings_in_a_table_and_states_caveats_once(integrity):
    _, dirty = integrity
    text = dirty.summary()
    assert "structure to inspect, not a judgment" in text
    assert "Fields are named condition · biological repeat · technical repeat · image file." in text
    # One row per check and pair of groups, the groups named as group labels.
    assert "copied field             between MOCK · repeat 1A · B6 and" in text
    assert "repeated within a field  within " in text
    assert "condition=" not in text
    assert text.count(OUT_OF_REACH) == 1
    assert "Check where these fields came from" in text
    assert "on 3 features" in text


def test_integrity_full_keeps_every_detail_of_the_findings(integrity):
    _, dirty = integrity
    full = _flat(dirty.summary(detail="full"))
    for finding in dirty.findings.itertuples():
        assert _flat(finding.detail) in full
        assert f"{finding.observations} observations" in full
    settings = dirty.settings
    assert f"{settings['n_fovs']} fields and {settings['n_observations']} nuclei" in full
    for feature in FEATURES:
        assert feature in full
    assert full.count(OUT_OF_REACH) == 1


def test_integrity_with_no_applicable_check_is_not_assessed():
    files = [f"f{i}" for i in range(4)]
    fovs = pd.DataFrame({"condition": "a", "biological_repeat": "r", "image_file": files})
    objects = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "image_file": [f"f{i // 10}" for i in range(40)],
            "count": [float(i % 3) for i in range(40)],
        }
    )
    result = integrity_checks(objects, fovs, feature_cols=["count"])
    assert result.status is Status.NOT_ASSESSED
    text = result.summary()
    assert text.startswith(f"{NOT_ASSESSED} Data integrity")
    assert "cannot be told from a coincidence" in _flat(text)


# --- Data check --------------------------------------------------------------------------


def test_data_check_marker_and_forms():
    built = SCENARIOS["condition_repeat_groups"]()
    clean = check_tables(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), observation_unit="cell"
    )
    assert clean.status is Status.OK and clean.summary().startswith(f"{OK} MorphoCoverage")
    warned = check_tables(built.objects, built.fovs, feature_cols=list(FEATURE_COLS))
    assert warned.status is Status.CAUTION
    short, full = warned.summary(), warned.summary(detail="full")
    assert short.startswith(f"{CAUTION} MorphoCoverage data check")
    for problem in warned.problems:
        assert _flat(problem.title) in _flat(short) and _flat(problem.fix) in _flat(short)
        assert f"Where: {_flat(problem.where)}" in _flat(full)
    assert "Where:" not in short and "summary(detail='full')" in short
    broken = check_tables(built.objects, built.fovs, feature_cols=["nope"])
    assert broken.status is Status.PROBLEM and broken.summary().startswith(PROBLEM)
    with pytest.raises(ValueError, match="Where: missing"):
        broken.raise_if_errors()  # a refusal carries the full form


# --- Frontier and benchmark ------------------------------------------------------------------


def _score(family, alpha, effort, saved=None, low=None, high=None, setting=0.1, **extra):
    return {
        "family": family,
        "alpha": alpha,
        "setting": setting,
        "effort": effort,
        "risk": 0.04,
        "fields_saved": saved,
        "saved_low": low,
        "saved_high": high,
        "saved_reached": 1.0,
        "adaptivity": 0.31,
        **extra,
    }


def _frontier(rows, units=8):
    settings = {"label": "the mean", "units": units, "unit_cols": ["biological_repeat", "well"]}
    return FrontierResult(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), settings)


def _base(alpha, effort=6.0):
    return _score(BASELINE, alpha, effort, setting=effort)


@pytest.mark.parametrize(
    ("rule", "units", "status", "words"),
    [
        (("saver", 4.0, 2.0, 0.5, 3.0), 8, Status.OK, "saver saves 2.0 of 6 fields"),
        (("saver", 4.0, 2.0, -0.5, 3.0), 8, Status.CAUTION, "may save 2.0 fields (not resolved)"),
        (("saver", 4.0, 2.0, 0.5, 3.0), 3, Status.CAUTION, "may save 2.0 fields (no interval)"),
        (("costly", 8.0, -2.0, -3.0, -1.0), 8, Status.OK, "no rule beats a fixed 6 fields"),
        (("costly", 8.0, -2.0, None, None), 3, Status.CAUTION, "(no interval)"),
    ],
)
def test_frontier_status_follows_its_rule(rule, units, status, words):
    name, effort, saved, low, high = rule
    result = _frontier([_base(0.05), _score(name, 0.05, effort, saved, low, high)], units)
    got, headline = frontier_status(result)
    assert got is status and words in headline
    assert result.summary().startswith(f"{status.marker} For the mean")


def test_an_unscored_level_is_not_assessed_and_prints_no_interval():
    base = _score(BASELINE, 0.05, None, setting=None)
    result = _frontier([base, _score("rule", 0.05, None)])
    assert frontier_status(result)[0] is Status.NOT_ASSESSED
    assert RESAMPLED not in result.summary() and RESAMPLED not in result.summary(detail="full")
    assert "No rule got there at any setting tried." in result.summary()


def test_frontier_short_is_a_table_with_levels_merged_and_the_interval_caveat_once():
    rows = []
    for alpha in (0.05, 0.1):
        rows += [_base(alpha), _score("rule", alpha, 4.0, 2.0, 0.5, 3.0)]
    result = _frontier(rows)
    short = result.summary()
    assert "at most 5% or 10% of acquisitions: a fixed count needs 6 fields (risk 4%)" in _flat(
        short
    )
    assert "rule at 0.1" in short and "saves 2.0 (0.5 to 3.0)" in short
    assert short.count(RESAMPLED) == 1 and "(biological repeat · well)" in short
    full = _flat(result.summary(detail="full"))
    # Every number of the old sentences, per level, and the caveat once.
    sentence = (
        "rule at 0.1: 4 fields on average (risk 4%); it saves 2.0 fields, approximate 95% "
        "interval 0.5 to 3.0; its stops follow the fields each acquisition needed with "
        "Spearman 0.31."
    )
    assert full.count(sentence) == 2
    assert full.count("a fixed number of fields needed 6 fields (risk 4%).") == 2
    assert full.count(RESAMPLED) == 1


def test_a_lapse_is_stated_in_both_forms():
    lapse = _score(
        "rule",
        0.05,
        None,
        met_setting="cheap",
        met_risk=0.0,
        exceeded_setting="late",
        exceeded_risk=0.25,
    )
    result = _frontier([_base(0.05), lapse])
    for text in (result.summary(), result.summary(detail="full")):
        assert "met this risk at cheap (risk 0%) but not from there on" in _flat(text)


@pytest.fixture(scope="module")
def bench():
    objects, fovs = _acquisition()
    return benchmark(
        objects,
        fovs,
        _setup(),
        reservoir_fraction=0.25,
        families=default_families(READOUT, n_resamples=20),
        n_splits=2,
        permuted_orders=2,
        n_boot=50,
    )


def test_benchmark_opens_with_the_worst_status_and_groups_grid_warnings(bench):
    warnings = grid_warnings(bench)
    assert warnings  # the default grids do not bracket these constructed data
    short, full = bench.summary(), bench.summary(detail="full")
    statuses = [frontier_status(result)[0] for result in bench.frontiers.values()]
    assert short.startswith(f"{CAUTION} Benchmark — fields saved against a fixed count")
    assert Status.CAUTION in statuses or warnings
    # Short: the grid advice once, each family's problem on its own line.
    assert _flat(short).count("widen the grid") == 1
    for warning in warnings:
        name = warning.split("the grid of ", 1)[1].split(" (", 1)[0]
        assert f"- {name} (" in short
        assert _flat(warning.removeprefix("Warning: the")) in _flat(full)
    for text in (short, full):
        assert "risk levels 5% and 10% of stopping too early (alphas" in _flat(text)
    for criterion in bench.frontiers:
        assert f"For {criterion}" in short


def test_benchmark_without_a_grid_problem_is_not_cautioned_for_one():
    scored = _frontier([_base(0.05), _score("saver", 0.05, 4.0, 2.0, 0.5, 3.0)])
    # The family stopped at its setting, after the first field: its grid brackets the data.
    runs = pd.DataFrame({"family": ["saver"], "setting": [0.1], "stopped": [True], "effort": [3]})
    result = FrontierResult(scored.curve, scored.scores, runs, scored.settings)
    bench = Benchmark({"the mean": result}, ("saver",), {})
    assert not grid_warnings(bench)
    assert bench.summary().startswith(f"{OK} Benchmark")


# --- Optimise ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plan():
    objects, fovs = _acquisition()
    return optimise(
        objects,
        fovs,
        _setup(),
        fov_counts=[2, 4],
        reservoir_fraction=0.25,
        n_splits=2,
        draws=5,
        permuted_orders=2,
        dispersed_orders=1,
    )


def test_plan_has_its_own_title_and_the_rule_check_at_a_glance(plan):
    short = plan.summary()
    assert short.startswith("Plan for the next experiment — 768 objects in 64 fields")
    status, answer = calibration_status(plan.results["calibration"])
    glance = short.split("\n\n")[1]
    assert f"{status.marker} {CALIBRATION_SHORT}" in glance
    assert _flat(answer) in _flat(glance)
    # The rule check is a section before the limits, and the settings close the plan.
    section = short.index(f"{status.marker} {CALIBRATION_TITLE}")
    assert section < short.index(f"\n{LIMITS_TITLE}\n") < short.index("Settings: a sample")
    full = plan.summary(detail="full")
    assert full.index(CALIBRATION_TITLE) < full.index(f"\n{LIMITS_TITLE}\n")
    assert "90% of random acquisitions" in _flat(full) and "90% of random" in _flat(short)
    with pytest.raises(ValueError, match="short"):
        plan.summary(detail="everything")


def test_calibration_status_follows_its_rule(plan):
    calibration = plan.results["calibration"]
    shares = calibration.table["not_preserved_at_stop"].dropna()
    status = calibration_status(calibration)[0]
    assert status is (Status.CAUTION if shares.max() > 0.1 else Status.OK)
    safe = calibration.table.assign(not_preserved_at_stop=0.0)
    ok = type(calibration)(calibration.runs, safe, calibration.group_cols, calibration.settings)
    assert calibration_status(ok)[0] is Status.OK
    empty = type(calibration)(calibration.runs, safe.iloc[0:0], calibration.group_cols, {})
    assert calibration_status(empty)[0] is Status.NOT_ASSESSED
    assert calibration_status(None)[0] is Status.NOT_ASSESSED
