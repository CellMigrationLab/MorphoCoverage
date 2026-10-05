"""Fourth (adversarial) audit, group H: the retrospective analyses and the benchmark.

One test per finding (P3-3, P8-1, P2-9, P9-2, P9-6, P3-10, P3-12, P8-2, P9-5,
P9-8, P1-7, P1-8, P1-9 and the effort part of P9-4); the reproduction
scripts are in the audit's P1, P2, P3, P8 and P9 folders.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import ObjectiveConfig, SamplingObjective, ScalarPrecisionStrategy
from morphocoverage._caveats import is_assumption
from morphocoverage._seeds import group_rng, group_seed
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    SamplingHierarchy,
    Summary,
    dataset_overview,
    effort_projection,
    inspect_fovs,
    microscopist_report,
    redundancy_distance,
    sampling_reliability,
)
from morphocoverage.retrospective.calibration import (
    StoppingCalibration,
    _table,
    calibration_summary,
)
from morphocoverage.retrospective.effort_design import _interval as effort_interval
from morphocoverage.retrospective.frontier import BASELINE, FrontierResult, _saved_interval
from morphocoverage.retrospective.frontier_text import frontier_summary
from morphocoverage.retrospective.report import MicroscopistReport
from morphocoverage.retrospective.report_profile import _redundancy_rows
from morphocoverage.sampling_unit import fov_variability
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import Setup
from morphocoverage.workflows._document import _technical_repeats, coverage_notes
from morphocoverage.workflows.benchmarks import Benchmark, grid_warnings

MEAN = OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better", "the mean")


def _tables(spec, seed=0, per_field=20, obs_sd=1.0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "technical_repeat": well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            offset = mean + rng.normal(0, field_sd)
            objects += [{**key, "x": v} for v in offset + rng.normal(0, obs_sd, per_field)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _frontier_result(units: int, low: float | None = 1.0, high: float | None = 3.0):
    """A scored frontier with one family saving 2 fields, as ``frontier`` returns it."""
    scores = pd.DataFrame(
        [
            {"family": BASELINE, "alpha": 0.1, "setting": 6, "effort": 6.0, "risk": 0.05},
            {
                "family": "rule",
                "alpha": 0.1,
                "setting": 0.05,
                "effort": 4.0,
                "risk": 0.08,
                "fields_saved": 2.0,
                "saved_low": low,
                "saved_high": high,
                "saved_reached": 1.0,
                "adaptivity": None,
            },
        ]
    )
    settings = {"label": "the mean", "units": units, "unit_cols": ["biological_repeat"]}
    return FrontierResult(pd.DataFrame(), scores, pd.DataFrame(), settings)


def test_p3_3_frontier_interval_is_widened_for_the_units_and_said_to_be_approximate():
    rng = np.random.default_rng(0)
    draws = rng.normal(2.0, 1.0, 2000)
    plain_low, plain_high = np.percentile(draws, [2.5, 97.5])
    low, high = _saved_interval(2.0, draws, 8)
    # Widened by sqrt(8 / 7) t_7 / z, about 1.28 on each side.
    assert low < plain_low - 0.4 and high > plain_high + 0.4
    assert (2.0 - low) / (2.0 - plain_low) == pytest.approx(1.28, abs=0.02)
    text = " ".join(frontier_summary(_frontier_result(units=8), detail="full").split())
    assert "approximate 95% interval 1.0 to 3.0" in text
    assert "with fewer than 20 units they cover the truth less often than stated" in text
    short = " ".join(frontier_summary(_frontier_result(units=8)).split())
    assert "saves 2.0 (1.0 to 3.0)" in short
    assert "with fewer than 20 units they cover the truth less often than stated" in short


def test_p8_1_frontier_gives_no_interval_below_five_units():
    draws = np.random.default_rng(1).normal(7.0, 1.5, 2000)
    assert _saved_interval(7.0, draws, 2) == (None, None)
    assert _saved_interval(7.0, draws, 4) == (None, None)
    assert _saved_interval(7.0, draws, 5)[0] is not None
    # Even a stored interval is not printed from two units (entry 8's plates 1A and 2A).
    for detail in ("short", "full"):
        text = frontier_summary(_frontier_result(units=2), detail=detail)
        [rule_line] = [line for line in text.split("\n") if "rule at 0.05" in line]
        assert "interval" not in rule_line and "to" not in rule_line.split("%")[-1]
        text = " ".join(text.split())
        assert (
            "No interval: only 2 units (biological repeat) to resample, and at least 5 are needed"
            in text
        )


def _calibration_runs(premature: int, beyond: int, correct: int) -> pd.DataFrame:
    """Runs of one rule: stops fixed later (premature), not fixable, and preserved."""
    rows = []
    kinds = [("not_preserved", "preserved")] * premature
    kinds += [("not_preserved", "not_preserved")] * beyond
    kinds += [("preserved", "preserved")] * correct
    for i, (at_stop, at_end) in enumerate(kinds):
        rows.append(
            {
                "split": i,
                "order_id": "permuted-0",
                "order": "permuted",
                "strategy": "rule",
                "config_id": "c1",
                "criterion": "mean",
                "stopped": True,
                "stop_fovs": 3 + i % 3,
                "stream_fovs": 20,
                "status_at_stop": at_stop,
                "status_at_end": at_end,
            }
        )
    return pd.DataFrame(rows)


def test_p2_9_calibration_headlines_the_stops_not_preserved_with_bounds():
    runs = _calibration_runs(premature=2, beyond=4, correct=4)
    table = _table(runs, [])
    row = table.iloc[0]
    assert row["not_preserved_at_stop"] == pytest.approx(0.6)
    assert row["premature"] == pytest.approx(0.2)
    total = row["premature"] + row["not_preserved_by_stream"]
    assert total == pytest.approx(row["not_preserved_at_stop"])
    text = calibration_summary(StoppingCalibration(runs, table, (), {"labels": {}}))
    assert "60%, 20%-60% too early" in text
    assert "held-back fields' own noise can also cause" in " ".join(text.split())
    # A median stop carries its Monte Carlo error.
    assert "median (MC SE)" in text and re.search(r"  4 \(\d", text)


def test_p9_2_retyped_labels_keep_their_draws_and_shares_carry_their_monte_carlo_error():
    # Owner decision: seed by the labels written canonically, so a retyped label is the
    # same group; other groups keep independent draws and never enter a group's seed.
    assert group_seed(0, {"biological_repeat": 1}) == group_seed(0, {"biological_repeat": "1"})
    assert group_seed(0, {"biological_repeat": 1}) == group_seed(0, {"biological_repeat": 1.0})
    assert group_seed(3, {"well": "A1"}, 2)[::2] == [3, 2]
    first = group_rng(0, {"condition": "ctrl"}).permutation(12)
    assert not (group_rng(0, {"condition": "drug"}).permutation(12) == first).all()
    built = SCENARIOS["homogeneous"]()
    settings = {
        "feature_cols": list(FEATURE_COLS),
        "criteria": [MEAN],
        "fov_counts": [2, 5],
        "reservoir_fraction": 0.3,
        "n_splits": 4,
        "draws": 10,
    }
    numbered = {"R1": 1, "R2": 2, "R3": 3}

    def retyped(table: pd.DataFrame, as_text: bool) -> pd.DataFrame:
        labels = table["biological_repeat"].map(lambda r: numbered.get(r, r))
        return table.assign(biological_repeat=labels.astype(str) if as_text else labels.astype(int))

    as_numbers = sampling_reliability(
        retyped(built.objects, False), retyped(built.fovs, False), **settings
    )
    as_text = sampling_reliability(
        retyped(built.objects, True), retyped(built.fovs, True), **settings
    )
    columns = ["fovs", "preserved", "preserved_split_sd", "preserved_mc_se"]
    pd.testing.assert_frame_equal(as_numbers.curve[columns], as_text.curve[columns])
    row = as_numbers.curve.iloc[0]
    assert row["preserved_mc_se"] == pytest.approx(row["preserved_split_sd"] / 2)
    assert "Monte Carlo SE" in as_numbers.summary(detail="full")


def test_p9_6_benchmark_warns_when_a_grid_does_not_bracket_the_data():
    rows = []
    for setting in (0.001, 0.01, 0.1):
        for run in range(10):
            rows.append(
                {
                    "family": "JCS spread per object",
                    "setting": setting,
                    "stopped": False,
                    "effort": 20,
                }
            )
            rows.append(
                {
                    "family": "JCB per object",
                    "setting": setting,
                    "stopped": run < 5,
                    "effort": 1 + run,
                }
            )
    runs = pd.DataFrame(rows)
    result = FrontierResult(pd.DataFrame(), pd.DataFrame(), runs, {"min_fovs": 1})
    grids = {
        "JCS spread per object": {
            "quantity": "max_descriptor_spread",
            "thresholds": [0.001, 0.01, 0.1],
        },
        "JCB per object": {"quantity": "averaging_cv", "thresholds": [0.001, 0.01, 0.1]},
    }
    bench = Benchmark(
        {"the mean": result},
        ("JCS spread per object", "JCB per object"),
        {"families": grids, "readout": "area"},
    )
    warnings = grid_warnings(bench)
    assert len(warnings) == 1
    assert "the grid of JCS spread per object (max_descriptor_spread 0.001 to 0.1)" in warnings[0]
    assert "does not bracket these data: it never stopped at any setting tried" in warnings[0]
    assert "units of area" in warnings[0] and "widen the grid" in warnings[0]


def test_p3_10_effort_intervals_are_widened_for_few_units():
    draws = np.random.default_rng(2).normal(1.0, 0.1, 1000)
    plain_low, plain_high = np.quantile(draws, [0.025, 0.975])
    low, high = effort_interval(draws, 1.0, 3)
    # sqrt(3 / 2) t_2 / z = 2.69 on each side with 3 units.
    assert (1.0 - low) / (1.0 - plain_low) == pytest.approx(2.69, abs=0.05)
    assert (high - 1.0) / (plain_high - 1.0) == pytest.approx(2.69, abs=0.05)
    # A standard error cannot be negative, however wide the interval.
    assert effort_interval(np.abs(draws - 1.0), 0.01, 3, floor=0.0)[0] >= 0.0
    assert effort_interval(draws, 1.0, 1) == (None, None)


def test_p3_12_reliability_says_its_curve_is_conservative_by_construction():
    built = SCENARIOS["homogeneous"]()
    result = sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[MEAN],
        fov_counts=[2],
        reservoir_fraction=0.3,
        n_splits=3,
        draws=5,
    )
    assert "conservative by construction" in result.summary()


def _grid_well(rng, well, trend, n=6, pitch=500.0):
    fovs, objects = [], []
    for i in range(n):
        for j in range(n):
            key = {
                "condition": "c",
                "biological_repeat": "r",
                "technical_repeat": well,
                "position_id": f"{well}_{i}_{j}",
            }
            fovs.append({**key, "x_um": i * pitch, "y_um": j * pitch})
            level = 10 + trend * i + rng.normal(0, 0.3)
            objects += [{**key, "x": v} for v in level + rng.normal(0, 0.2, 10)]
    return fovs, objects


def test_p8_2_redundancy_distance_is_tested_against_shuffled_positions():
    rng = np.random.default_rng(3)
    fovs, objects = [], []
    for well, trend in (("flat", 0.0), ("trend", 1.0)):
        f, o = _grid_well(rng, well, trend)
        fovs += f
        objects += o
    table = redundancy_distance(
        pd.DataFrame(objects),
        pd.DataFrame(fovs),
        feature_cols=["x"],
        stage_cols=("x_um", "y_um"),
        bin_edges=[0, 600, 1200, 1800, 2400, 3000, 4000],
        n_permutations=99,
    ).set_index("technical_repeat")
    assert bool(table.loc["trend", "beyond_chance"]) and table.loc["trend", "null_share"] < 0.05
    assert not bool(table.loc["flat", "beyond_chance"])
    assert table.loc["flat", "null_share"] > 0.05
    # The report counts only the group beyond chance and says the other is within it.
    report = microscopist_report(redundancy=table.reset_index())
    profile = " ".join(report.sections["profile"].split())
    assert "in 1 of 2 groups, fields closer than about" in profile
    assert "beyond what shuffling the positions within each group gives" in profile
    # A point estimate the shuffles do not support is named as such, not as redundancy.
    doubtful = table.reset_index().assign(
        redundancy_distance=[1200.0, 1200.0], beyond_chance=[False, True], null_share=[0.4, 0.01]
    )
    statement = _redundancy_rows(doubtful)[0]["statement"]
    assert "in 1 of 2 groups the point estimate suggested that closer fields" in statement
    assert "in a median of 40% of shuffles, so no spatial redundancy is shown there" in statement


def test_p9_5_neighbour_ties_do_not_depend_on_the_row_order_of_the_fov_table():
    rng = np.random.default_rng(4)
    fovs, objects = _grid_well(rng, "w", 0.0, n=5)
    fovs, objects = pd.DataFrame(fovs), pd.DataFrame(objects)
    # One deviant field in the middle of a regular grid: its 4 nearest are tied with others.
    objects.loc[objects["position_id"] == "w_2_2", "x"] += 3.0
    objects.loc[objects["position_id"] == "w_1_3", "x"] -= 2.0

    def flagged(fov_table):
        result = inspect_fovs(
            objects,
            fov_table,
            feature_cols=["x"],
            summaries=[Summary("x", "mean")],
            stage_cols=("x_um", "y_um"),
            n_neighbours=3,
            neighbour_threshold=2.0,
        )
        rows = result.fovs[result.fovs["reason"] == "neighbours"]
        return sorted(zip(rows["position_id"], rows["detail"], strict=True))

    first = flagged(fovs)
    assert first and any("fields tied in distance taken by field name" in d for _, d in first)
    for seed in range(3):
        assert flagged(fovs.sample(frac=1, random_state=seed)) == first


def test_p9_8_sufficiency_report_runs_its_replays_on_n_jobs_with_identical_answers():
    built = SCENARIOS["homogeneous"]()
    arguments = {
        "feature_cols": list(FEATURE_COLS),
        "strategies": [ScalarPrecisionStrategy("feature_a", 0.05, "fov", "upper_95")],
        "objective": ObjectiveConfig(SamplingObjective.POPULATION),
        "orders": [AcquisitionOrder.permuted(s) for s in range(3)],
    }
    serial = sufficiency_report(built.objects, built.fovs, **arguments)
    parallel = sufficiency_report(built.objects, built.fovs, n_jobs=2, **arguments)
    pd.testing.assert_frame_equal(serial.stops, parallel.stops)
    assert serial.summary() == parallel.summary()
    assert "stop_fov_median_mc_se" in serial.stops.columns


def _report_parts(objects, fovs, group_cols):
    overview = dataset_overview(objects, fovs, feature_cols=["x"], group_cols=group_cols)
    variability = fov_variability(objects, fovs, feature="x", group_cols=group_cols)
    document = microscopist_report(overview=overview, variability=variability)
    return {"overview": overview, "variability": variability}, document


_WELLS = [
    ("ctrl", "r1", "A", 6, 10.0, 0.1),
    ("ctrl", "r1", "B", 6, 14.0, 0.1),
    ("ctrl", "r1", "C", 6, 8.0, 0.1),
    ("drug", "r1", "A", 6, 12.0, 0.1),
    ("drug", "r1", "B", 6, 16.0, 0.1),
]


def test_p1_7_the_report_names_the_variance_between_technical_repeats():
    objects, fovs = _tables(_WELLS)
    results, document = _report_parts(
        objects, fovs, ["condition", "biological_repeat", "technical_repeat"]
    )
    document = _technical_repeats(document, results)
    profile = document.sections["profile"]
    assert "Between technical repeats (x): in 2 condition and repeat groups" in profile
    assert "as much as their fields alone predict" in profile
    assert "only more technical repeats reduce (optimise, question 7)" in profile
    assert any("Each 'enough' is per technical repeat" in note for note in document.notes)


def test_p1_8_a_group_that_pools_technical_repeats_says_so_beside_enough():
    objects, fovs = _tables(_WELLS)
    results, document = _report_parts(objects, fovs, ["condition", "biological_repeat"])
    document = _technical_repeats(document, results)
    enough = document.sections["enough"]
    assert "Each analysis group pools the technical repeats of its condition and repeat" in enough
    assert "(2 to 3 per group" in enough and "add 'technical_repeat' to group_cols" in enough
    assert any("pools technical repeats (2 to 3 per group)" in note for note in document.notes)


def test_p1_9_the_limits_say_what_was_not_read_or_not_recorded():
    setup = Setup(readout="x", features=("x", "y"))
    notes = coverage_notes({"variability": object()}, setup)
    assert any("read the readout (x) only" in n and "(y)" in n for n in notes)
    assert any("No stage coordinates:" in n for n in notes)
    assert any("No acquisition order:" in n for n in notes)
    placed = Setup(readout="x", stage_cols=("sx", "sy"), acquisition_order_col="t")
    assert coverage_notes({}, placed) == []
    # Every note is standing: options left at their defaults (no stage coordinates, no
    # acquisition order) are notes on what was not assessed, not cautions about these data.
    kinds = {n.split(" ")[0]: is_assumption(n) for n in notes}
    assert kinds == {"The": True, "No": True}
    # Nothing left unassessed still reads as "on the assumptions stated", never "everything".
    assessed = MicroscopistReport({}, pd.DataFrame(), inputs={})
    closing = assessed.limits()
    assert "Every question could be answered; the assumptions each answer rests on" in closing
    assert "everything" not in closing


def test_p9_4_effort_intervals_do_not_depend_on_the_row_order_of_the_tables():
    objects, fovs = _tables(
        [
            ("ctrl", r, w, 4, 10.0 + i, 0.3)
            for i, (r, w) in enumerate(
                [("r1", "A"), ("r1", "B"), ("r2", "A"), ("r2", "B"), ("r3", "A"), ("r3", "B")]
            )
        ],
        per_field=8,
    )
    hierarchy = SamplingHierarchy("position_id", "technical_repeat", "biological_repeat")

    def components(obj, fov):
        result = effort_projection(
            obj, fov, feature="x", hierarchy=hierarchy, strata=["condition"], n_boot=100
        )
        return result.components[["level", "estimate", "low", "high"]].reset_index(drop=True)

    first = components(objects, fovs)
    shuffled = components(
        objects.sample(frac=1, random_state=1), fovs.sample(frac=1, random_state=2)
    )
    pd.testing.assert_frame_equal(first, shuffled, check_exact=False, rtol=1e-9)


def test_issue_148_a_caution_answering_a_question_not_assessed_raises_its_status():
    from morphocoverage._caveats import assumption, caveat_lines
    from morphocoverage._style import Status
    from morphocoverage.workflows._document import with_text

    def added(text: str, key: str = "profile", status: Status = Status.OK) -> Status:
        document = MicroscopistReport(
            sections={key: "  needs it"},
            profile=pd.DataFrame(),
            inputs={key: "not assessed"},
            statuses={key: Status.NOT_ASSESSED},
        )
        short = "\n".join(caveat_lines([text], indent=2))
        return with_text(document, key, text, status=status, short_text=short).statuses[key]

    assert added("a caution about these data") is Status.CAUTION
    assert added("a caution about these data", "populations") is Status.OK
    assert added("a caution", status=Status.PROBLEM) is Status.PROBLEM
    assert added(assumption("a standing note")) is Status.OK
