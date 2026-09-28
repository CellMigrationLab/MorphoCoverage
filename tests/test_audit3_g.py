"""Regression tests of audit 3, group G: effort, report, controls, inspection, frontier.

One test per finding; the reproductions behind them were run on data/synthetic.
"""

from __future__ import annotations

import inspect as source
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_retrospective_controls import FEATURES, NOVELTY, _plate

from morphocoverage import statistics
from morphocoverage._wording import percent
from morphocoverage.bounds import upper_bound
from morphocoverage.retrospective import (
    RepresentationControl,
    SamplingHierarchy,
    Summary,
    effort_design,
    effort_projection,
    effort_text,
    inspect_fovs,
    microscopist_report,
    report,
    representation_control,
)
from morphocoverage.retrospective.controls import ControlResult
from morphocoverage.retrospective.effort import MIN_BOOT_SUCCESS
from morphocoverage.retrospective.frontier import BASELINE, FrontierResult, transfer
from morphocoverage.retrospective.inspection import InspectionResult
from morphocoverage.retrospective.replay import ReplayResult
from morphocoverage.retrospective.report_profile import (
    _detection_adjusted_bounds,
    _new_state_rows,
    _redundancy_rows,
    sampling_profile,
)
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SYN_FEATURES = ["area", "intensity", "eccentricity"]
SYN_NOVELTY = SupportNoveltyEstimator(float(np.sqrt(3)), 1, Standardize(), 5, 3)
KEY = ["condition", "repeat", "technical_repeat", "image_file"]


@pytest.fixture(scope="module")
def synthetic():
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv")
    return objects, fovs


def _syn_control(objects, fovs, **kwargs):
    settings = {"pair_cols": ["repeat"], "fovs_seen": 10, "draws": 2}
    settings.update(kwargs)
    return representation_control(
        objects,
        fovs,
        feature_cols=SYN_FEATURES,
        novelty=SYN_NOVELTY,
        control=RepresentationControl(
            positive={"condition": "drug"}, reference={"condition": "control"}
        ),
        observation_unit="cell",
        **settings,
    )


def _syn_inspection(objects, fovs, **kwargs):
    return inspect_fovs(
        objects,
        fovs,
        feature_cols=["area"],
        novelty=SYN_NOVELTY,
        novelty_cols=SYN_FEATURES,
        count_ratio=None,
        observation_unit="cell",
        **kwargs,
    )


def _pooled(**rates):
    """A ControlResult holding only pooled counts: kind -> (judged new, judged)."""
    pooled = pd.DataFrame(
        [
            {"candidate": k, "judged_new": n, "candidates": t, "not_judged": 0, "rate": n / t}
            for k, (n, t) in rates.items()
        ]
    )
    table = pooled.assign(stratum="all FOVs")
    return ControlResult(table, pooled, "assessed", ("repeat",), {"fovs_seen": 10})


def test_g1_false_alarms_are_measured_within_the_inspections_groups(synthetic):
    objects, fovs = synthetic
    controls = _syn_control(objects, fovs)
    inspection = _syn_inspection(objects, fovs)
    assert controls.settings["lofo_group_cols"] == list(inspection.group_cols)
    new = inspection.fovs[inspection.fovs["reason"] == "new state"][KEY].drop_duplicates()
    flagged = int((new["condition"] == "control").sum())
    reference = controls.pooled.set_index("candidate").loc["reference_lofo"]
    # The inspection's own share of control fields called new (7 of 90), not 0 of 90 as
    # when each field was judged against the whole repeat's reference fields.
    assert (int(reference["judged_new"]), int(reference["candidates"])) == (flagged, 90)
    assert flagged > 0
    positive = controls.pooled.set_index("candidate").loc["positive_lofo"]
    # Every drug field holding cells against each control well; since audit 4 (P1-2) the
    # 15 empty drug fields are counted apart, never judged as misses.
    assert int(positive["candidates"]) == 75 * 5
    assert controls.settings["positive_empty"] == 15
    # Measured over each whole stratum, as before, on request.
    pooled = _syn_control(objects, fovs, lofo_group_cols=[])
    assert pooled.pooled.set_index("candidate").loc["reference_lofo", "judged_new"] < flagged


def test_g2_the_chance_expectation_counts_only_fields_judged_for_new_states(synthetic):
    objects, fovs = synthetic
    short = (fovs["repeat"] == "2026_01_19") & fovs["image_file"].isin(
        ["field_05.tif", "field_06.tif"]
    )
    fovs = fovs[~short]
    objects = objects.merge(fovs[KEY])
    inspection = _syn_inspection(objects, fovs)
    assert len(inspection.settings["skipped_novelty"]) == 10
    assert inspection.settings["n_fovs"] == 160
    assert inspection.settings["n_fovs_judged"] == 120
    controls = _pooled(reference_lofo=(3, 60))
    statement = _new_state_rows(inspection, controls)[0]["statement"]
    assert "of the 120 fields judged expected by chance" in statement


def test_g3_the_expected_count_gives_the_rates_upper_bound():
    inspection = InspectionResult(
        pd.DataFrame({"reason": ["new state"] * 4, "condition": list("abcd"), "f": list("wxyz")}),
        ("condition",),
        {"fov_key": ["condition", "f"], "n_fovs": 336, "n_fovs_judged": 336},
    )
    statement = _new_state_rows(inspection, _pooled(reference_lofo=(0, 56)))[0]["statement"]
    high = upper_bound(0, 56, 0.95)
    assert "0 to 17.5 of the 336 fields judged expected by chance" in statement
    assert f"0 of 56 fields (0%; at most {percent(high)} at 95% confidence)" in statement
    assert high == pytest.approx(0.052, abs=1e-3)
    assert "about 0 of 336" not in statement


def test_g4_same_named_dishes_of_pooled_conditions_are_different_samples(synthetic):
    objects, fovs = synthetic

    def rename(table):
        table = table.copy()
        table["technical_repeat"] = "dish" + table["technical_repeat"].str[-1]
        table["image_file"] = (
            table["condition"] + "_" + table["repeat"] + "_" + table["technical_repeat"] + "_"
        ) + table["image_file"]
        return table

    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "repeat")
    result = effort_projection(
        rename(objects),
        rename(fovs),
        feature="area",
        hierarchy=hierarchy,
        strata=[],
        n_boot=20,
        observation_unit="cell",
    )
    first = result.components.iloc[0]
    assert (first["samples"], first["samples_acquired"], first["repeats"]) == (30, 30, 3)


def test_g5_an_allocation_with_fewer_samples_than_repeats_covers_fewer_repeats(synthetic):
    objects, fovs = synthetic
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "repeat")
    result = effort_projection(
        objects,
        fovs,
        feature="area",
        hierarchy=hierarchy,
        strata=["condition"],
        allocations=[(2, 45.0)],
        n_boot=20,
        observation_unit="cell",
    )
    comp = result.components[result.components["condition"] == "drug"].set_index("level")
    v_rep, v_sample, v_fov, v_obs = np.maximum(comp["estimate"].to_numpy(), 0)
    first = comp.iloc[0]
    observations = first["fovs_with_observations"] * first["observations_per_fov"]
    size = first["cluster_size"]
    expected = np.sqrt(v_rep / 2 + v_sample / 2 + (v_fov * size + v_obs) / observations)
    projections = result.projections
    row = projections[(projections["condition"] == "drug") & (projections["samples"] == 2)]
    assert float(row["se"].iloc[0]) == pytest.approx(expected)


def _frontier_scores(rows, settings=None):
    return FrontierResult(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), settings or {})


def test_g6_a_baseline_that_met_the_risk_but_lost_it_says_so_and_keeps_the_families():
    base = {
        "family": BASELINE,
        "alpha": 0.05,
        "setting": None,
        "effort": None,
        "risk": None,
        "met_setting": 2,
        "met_risk": 0.0,
        "exceeded_setting": 3,
        "exceeded_risk": 0.125,
    }
    rule = {"family": "rule", "alpha": 0.05, "setting": 1, "effort": 2.0, "risk": 0.0}
    text = _frontier_scores([base, rule], {"label": "c", "units": 8}).summary()
    assert "a fixed number of fields met this risk at 2 (risk 0%) but not from there on" in text
    assert "never got there" not in text
    assert "rule at 1: 2 fields on average (risk 0%), with no fixed count to compare" in text


def test_g7_the_redundancy_distance_is_a_point_estimate_with_its_pairs():
    redundancy = pd.DataFrame(
        {
            "redundancy_distance": [336.0, 520.0, 700.0],
            "pairs_closer": [12, 40, 25],
            "smallest_lag": [100.0, 100.0, 100.0],
            "unit": "µm",
        }
    )
    statement = _redundancy_rows(redundancy)[0]["statement"]
    assert "the median of point estimates, not tested against chance" in statement
    assert "from 12 to 40 pairs of fields closer than that per group" in statement


def test_g8_a_frontier_interval_from_too_few_resamples_is_withheld():
    statuses = {
        f"u{i}": (["n", "n", "n", "p"] if i == 0 else ["n", "p", "p", "p"]) for i in range(20)
    }
    names = {"p": "preserved", "n": "not_preserved"}
    outcomes, streams, decisions = [], [], []
    for unit, row in statuses.items():
        run = {"unit": unit, "split": 0, "order_id": "recorded", "order": "recorded"}
        streams.append({**run, "order_seed": None, "stream_fovs": 4, "reservoir_fovs": 5})
        for k, status in enumerate(row, start=1):
            outcomes.append(
                {**run, "fovs": k, "criterion": "c", "value": 0.0, "status": names[status]}
            )
            decisions.append(
                {
                    **run,
                    "strategy": "r",
                    "config_id": "cfg",
                    "fovs": k,
                    "decision": "stop" if k >= 2 else "continue",
                }
            )
    replay = ReplayResult(
        pd.DataFrame(decisions),
        pd.DataFrame(outcomes),
        pd.DataFrame(streams),
        ("unit",),
        {"random_state": 0},
    )
    from morphocoverage.retrospective.frontier import family, frontier

    result = frontier(
        replay,
        [family(replay, "rule", {"cfg": 1})],
        criterion="c",
        alphas=[0.06],
        unit_cols=["unit"],
        n_boot=500,
    )
    row = result.scores.set_index("family").loc["rule"]
    # 72% of the resamples reach the risk for both: below the effort projection's 90%.
    assert MIN_BOOT_SUCCESS == 0.9 and row["saved_reached"] < MIN_BOOT_SUCCESS
    assert row["saved_low"] is None or pd.isna(row["saved_low"])
    reached = f"{100 * row['saved_reached']:.0f}%"
    text = result.summary()
    assert f"no interval: only {reached} of the resamples reached this risk for both" in text
    assert "(at least 90% are needed)" in text


def test_g9_the_imbalance_line_names_the_level_that_differs():
    projections = pd.DataFrame(
        {"current": [True, True], "design": ["acquired", "balanced"], "se": [1.2, 1.0]}
    )
    first = {
        "repeat_dropped": False,
        "repeat_weight": 0.6,
        "repeats": 2,
        "sample_weight": 1 / 6,
        "samples": 6,
    }
    line = effort_text._imbalance_line(projections, first, "technical repeat", "cells")
    assert line.startswith("  The acquired repeats differ in size")
    both = {**first, "sample_weight": 0.3}
    line = effort_text._imbalance_line(projections, both, "technical repeat", "cells")
    assert "The acquired repeats and technical repeats differ in size" in line


def test_g10_small_shares_are_not_written_as_zero_percent():
    sufficiency = SimpleNamespace(
        next_fov=pd.DataFrame({"g": ["a"], "unseen_state_max_frequency": [0.004]}),
        group_cols=("g",),
    )
    text = _detection_adjusted_bounds(sufficiency, _pooled(positive_lofo=(19, 20)))
    # Audit 4 (P3-9): divided by the one-sided lower bound of 19 of 20 (78%), not by 95%.
    assert "up to 0.5% of FOVs (the 0.4% bound above divided by 78%)" in text
    compared = pd.DataFrame(
        {
            "samples": [40],
            "fovs_per_sample": [3.0],
            "ratio": [0.996],
            "ratio_low": [0.993],
            "ratio_high": [0.999],
            "same_total": [True],
            "resolved": [True],
            "current_samples": [30],
            "current_fovs_per_sample": [4.0],
        }
    )
    line = effort_text._allocation_line(compared, {}, 12, "repeat", "sample")
    assert "a 0.4% smaller standard error" in line


def _counted(counts, stage=False):
    names = [f"f{i}" for i in range(len(counts))]
    fovs = pd.DataFrame({"condition": "a", "repeat": "r", "image_file": names})
    if stage:
        fovs["x"] = [float(i % 5) for i in range(len(names))]
        fovs["y"] = [float(i // 5) for i in range(len(names))]
    rng = np.random.default_rng(len(counts))
    rows = [
        {"condition": "a", "repeat": "r", "image_file": name, "v": 10.0 + value}
        for name, n in zip(names, counts, strict=True)
        for value in rng.normal(0.0, 1.0, n)
    ]
    return pd.DataFrame(rows), fovs


def test_g11_one_observation_is_one_cell_not_one_cells():
    objects, fovs = _counted([1, 12, 15, 14, 13, 16])
    result = inspect_fovs(objects, fovs, feature_cols=["v"], observation_unit="cell")
    assert "f0: 1 cell; the median is 13.5." in result.summary()


def test_g12_an_empty_neighbour_is_not_counted_among_the_fields_compared():
    counts = [6] * 25
    counts[7] = 0  # the field above the centre holds no cell
    objects, fovs = _counted(counts, stage=True)
    objects.loc[objects["image_file"] == "f12", "v"] += 30.0
    result = inspect_fovs(
        objects, fovs, feature_cols=["v"], stage_cols=("x", "y"), count_ratio=None
    )
    rows = result.fovs[(result.fovs["reason"] == "neighbours")].set_index("image_file")
    detail = rows.loc["f12", "detail"]
    assert "above 3 of its 4 nearest fields (1 field without a value)" in detail


def test_g13_the_effort_summary_has_no_line_for_a_setting_it_never_records():
    objects, fovs = _plate()
    result = effort_projection(
        objects,
        fovs,
        feature="feature_a",
        hierarchy=SamplingHierarchy("image_file", "well"),
        strata=["condition"],
        n_boot=20,
    )
    assert "normalized" not in result.settings
    assert "normalised" not in source.getsource(effort_text.effort_summary)


def test_g14_projections_assume_the_same_share_of_fields_would_be_empty_again():
    objects, fovs = _plate()
    empty = fovs["image_file"].isin(["P1_A1_0", "P2_A1_0"])
    objects = objects[~objects["image_file"].isin(fovs.loc[empty, "image_file"])]
    result = effort_projection(
        objects,
        fovs,
        feature="feature_a",
        hierarchy=SamplingHierarchy("image_file", "well", "repeat"),
        strata=["condition"],
        n_boot=20,
    )
    assert "the same share would be empty again." in " ".join(result.summary().split())
    assert "the same share would be empty again." in " ".join(result.summary(detail="full").split())


def test_g15_the_closing_notes_do_not_point_below():
    controls = _pooled(positive=(2, 20), reference=(0, 20))
    built = microscopist_report(controls=controls)
    note = next(n for n in built.notes if n.startswith("Weak representation"))
    assert "below" not in note.split(")", 1)[1]
    assert "adequacy answers (questions 2, 4 and 5)" in note
    assert "only 10% of checks (below 90%)" in note


def test_g16_the_profile_labels_each_rate_by_how_it_was_measured():
    controls = _pooled(positive=(18, 20), reference=(1, 20), reference_lofo=(2, 30))
    controls.settings["lofo_group_cols"] = ["condition", "repeat"]
    profile = sampling_profile(controls, None, None, None)
    statements = list(profile["statement"])
    assert "90% of declared positive-control fields recognized as different after 10 " in (
        statements[0] + " "
    )
    assert any("after 10 reference fields (false alarms)" in s for s in statements)
    lofo = profile[profile["item"] == "representation, as inspect_fovs judges"]
    assert "their analysis group (condition, repeat)" in lofo["statement"].iloc[0]


def test_g17_the_leave_one_field_out_false_alarms_are_measured_without_a_positive():
    objects, fovs = _plate()
    keep = ~((fovs["repeat"] == "P2") & (fovs["condition"] == "KD"))
    fovs = fovs[keep]
    objects = objects[objects["image_file"].isin(fovs["image_file"])]
    control = RepresentationControl(positive={"condition": "KD"}, reference={"condition": "MOCK"})
    result = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=NOVELTY,
        control=control,
        pair_cols=["repeat"],
        fovs_seen=6,
        draws=3,
    )
    rows = result.table.set_index(["repeat", "candidate"])
    assert ("P2", "reference_lofo") in rows.index
    assert ("P2", "positive") not in rows.index
    assert [s["stratum"] for s in result.settings["skipped_strata"]] == ["repeat=P2"]


def test_g18_a_fixed_count_beyond_the_held_out_streams_acquires_the_whole_stream():
    development = FrontierResult(
        pd.DataFrame(),
        pd.DataFrame(
            [{"family": BASELINE, "alpha": 0.1, "setting": 5, "risk": 0.05, "effort": 5.0}]
        ),
        pd.DataFrame(),
    )
    held_out = FrontierResult(
        pd.DataFrame(
            {
                "family": BASELINE,
                "setting": [1, 2, 3],
                "effort": [1.0, 2.0, 3.0],
                "risk": [0.5, 0.2, 0.1],
                "judged": [10, 10, 10],
            }
        ),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    row = transfer(development, held_out, 0.1).iloc[0]
    assert row["held_out_setting"] == 3
    assert row["held_out_risk"] == 0.1 and row["held_out_effort"] == 3.0


def test_g19_the_influence_reason_names_the_field_weighted_summary():
    objects, fovs = _counted([20] * 6)
    objects.loc[objects["image_file"] == "f0", "v"] += 20.0
    result = inspect_fovs(
        objects, fovs, feature_cols=["v"], summaries=[Summary("v", "mean")], count_ratio=None
    )
    detail = result.fovs.set_index("image_file").loc["f0", "detail"]
    assert detail.startswith("removing it lowers the field-weighted mean of v by")


def test_g20_rates_of_another_definition_of_new_are_not_combined():
    controls = _pooled(reference_lofo=(1, 30), positive_lofo=(9, 10))
    controls.settings["novelty_definition"] = "one definition"
    inspection = InspectionResult(
        pd.DataFrame({"reason": ["new state"], "condition": ["a"], "f": ["w"]}),
        ("condition",),
        {"fov_key": ["condition", "f"], "n_fovs_judged": 30, "novelty_definition": "another"},
    )
    statement = _new_state_rows(inspection, controls)[0]["statement"]
    assert "not compared with the declared reference's false alarms" in statement
    assert "expected by chance" not in statement
    sufficiency = SimpleNamespace(
        next_fov=pd.DataFrame({"condition": ["a"], "unseen_state_max_frequency": [0.1]}),
        group_cols=("condition",),
        novelty_definition="another",
    )
    text = _detection_adjusted_bounds(sufficiency, controls)
    assert text.startswith("The unseen-state bound is not adjusted for detection")
    controls.settings["lofo_group_cols"] = ["condition", "repeat"]
    sufficiency.novelty_definition = "one definition"
    assert "measured within the groups condition, repeat" in _detection_adjusted_bounds(
        sufficiency, controls
    )


def test_g21_percentile_labels_use_the_shared_ordinal_and_rates_one_lookup():
    labels = [Summary("area", q).label for q in (0.01, 0.22, 0.11, 0.025)]
    assert labels == [
        "1st percentile of area",
        "22nd percentile of area",
        "11th percentile of area",
        "2.5th percentile of area",
    ]
    assert "_ordinal_suffix" not in source.getsource(source.getmodule(Summary))
    assert 'set_index("candidate")' not in source.getsource(report.microscopist_report)


def test_i_note_empty_reference_fields_do_not_leave_checks_unjudged(synthetic):
    objects, fovs = synthetic
    result = _syn_control(objects, fovs, pair_cols=[], fovs_seen=5, draws=4)
    pooled = result.pooled.set_index("candidate")
    assert (pooled["not_judged"] == 0).all()
    assert int(pooled.loc["positive", "candidates"]) == 4 * 75  # drug fields with cells (P1-2)
    assert "(after 5 reference fields with cells)" in result.summary()


def test_j_shared_helpers_and_declared_privates():
    counts = np.array([0, 3, 10, 25])
    design = effort_design._design(
        pd.DataFrame({"f": np.repeat(["a", "b", "c"], [3, 10, 25]), "s": "w", "value": 1.0}),
        ["s", "f"],
        False,
        4,
        SamplingHierarchy("f", "s"),
    )
    assert design["cluster_size"] == pytest.approx(statistics.cluster_size(counts))  # J-4
    assert effort_design.__all__ == []  # J-16
    assert "Shared-private:" in (effort_design.__doc__ or "")  # J-12
    from morphocoverage.retrospective import report_profile

    assert "Shared-private:" in (report_profile.__doc__ or "")
