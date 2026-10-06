"""Audit 10, group G2: the tests and their families (owner, 2026-10-06).

Each test pins one decision of the audit's report findings on the scenario of
``data/scenarios`` that shows it, or on a small constructed table: what the report says now,
and that it never says the old, misleading thing.
"""

from __future__ import annotations

import sys
import warnings
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from morphocoverage import (
    Component,
    Decision,
    Effort,
    EffortPolicy,
    Evidence,
    SamplingObjective,
)
from morphocoverage._control import ACROSS, MASTER_RAW
from morphocoverage._style import Status
from morphocoverage.plots_pairwise import PANEL_KEYS, verdict_note
from morphocoverage.policies import decide
from morphocoverage.sampling_unit import object_rule_check
from morphocoverage.sampling_unit.contrast_across import across_rows
from morphocoverage.sampling_unit.contrast_cells import repeats_to_balance, trusted_test
from morphocoverage.sampling_unit.contrast_sampling import (
    MIN_FIELDS,
    noncentral_d_interval,
    sampling_row,
)
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.distribution import distribution_contrast
from morphocoverage.sampling_unit.whole_table import sign_floor, whole_table_tests
from morphocoverage.statistics import benjamini_hochberg, cohen_d
from morphocoverage.sufficiency_short import met_count
from morphocoverage.workflows import Setup, attach_feature_tables, plots, report
from morphocoverage.workflows._guided_controls import ask_experiment
from morphocoverage.workflows._html_render import chip
from morphocoverage.workflows._whole_table import across_reason, whole_table_lines

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"
sys.path.insert(0, str(SCENARIOS))
try:
    import make_scenarios
finally:
    sys.path.remove(str(SCENARIOS))


@cache
def _scenario_report(name: str) -> Any:
    """``report`` on a scenario's tables, as a user runs it, its warnings kept quiet."""
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), feature_tables=loaded.feature_tables
        )


def test_the_rule_counts_say_how_many_groups_were_judged() -> None:
    # B19: "0 of 40 (1 not defined)" hid 12 groups with too few fields to judge.
    tally = {"met": 0, "groups": 40, "undefined": 1, "waiting": 12}
    assert met_count(tally) == "0 of 27 judged; 12 too few fields; 1 undefined"
    assert met_count({"met": 3, "groups": 4, "undefined": 0, "waiting": 0}) == "3 of 4"
    assert met_count({"met": 0, "groups": 4, "undefined": 0, "waiting": 4}, "group") == (
        "none judged; 4 too few fields"
    )
    # e01: three fields per group, too few for the per-field bootstrap bound in every group.
    text = _scenario_report("e01_two_repeats").summary("full")
    row = next(line for line in text.splitlines() if "mean-precision rule, per field, 5%  " in line)
    assert "none judged; 4 too few fields" in row
    assert " 0 of 4 " not in row


@pytest.mark.parametrize(
    "name", ["a10_embed2_several_norm_cells_none_tol", "a21_embed1_one_norm_cells_none_tol"]
)
def test_a_setup_with_no_target_has_no_component_and_nothing_decides_a_stop(name: str) -> None:
    # C12: "a stop needs every declared target (phenotypic fingerprint)" with none declared.
    setup = Setup(**make_scenarios.load(name).setup)
    objective = setup.objective()
    assert objective.objective is SamplingObjective.NO_TARGET
    assert objective.required_components == frozenset()
    said = [line for line in setup.declaration() if line.startswith("Live decision")]
    assert said == ["Live decision: no target: nothing decides a stop."]
    for detail in ("short", "full"):
        text = " ".join(_scenario_report(name).summary(detail).split())
        assert "a stop needs every declared target" not in text
    assert "Live decision: no target: nothing decides a stop." in text
    # Whatever a rule says, no target means no stop.
    stop = EffortPolicy("n_fovs", 1, Component.DISTRIBUTION)
    overall, per_component = decide(Evidence(), Effort(n_fovs=10, n_objects=100), [stop], objective)
    assert per_component == {"distribution": Decision.STOP}
    assert overall is Decision.UNSUPPORTED


def test_panel_c_says_its_q_is_the_figures_and_gives_the_verdicts_q() -> None:
    # A3: panel C's q, adjusted over the figure's pairs, was green where the verdict's q,
    # adjusted over the feature family, was not significant (and the other way round).
    name = "a15_expl_one_cells_tech_rare"
    loaded, result = make_scenarios.load(name), _scenario_report(name)
    discovery = result.discovery
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = plots(loaded.objects, loaded.fovs, result.setup, ranking=discovery)
    tests = discovery.tests
    checked = 0
    for feature, parts in drawn.items():
        for figure in parts:
            mine = tests[tests["feature"].eq(feature) & tests["q_paired"].notna()]
            expected = dict(zip(mine["condition"], mine["q_paired"], strict=True))
            assert figure.pairwise_verdicts == pytest.approx(expected)
            legend = " ".join(figure.legend_artist.get_text().split())
            assert "adjusted over the pairs of this figure only" in legend
            if expected:
                checked += 1
                assert verdict_note(expected) in legend
                assert "Verdict q (the feature's family): " in legend
    assert checked
    assert PANEL_KEYS["C"].endswith("adjusted over this figure only")


def test_a_balanced_test_left_with_one_repeat_gives_way_to_the_test_as_provided() -> None:
    # B5: the image-level "balanced" test kept R3 alone (the other repeats had fewer than 5
    # control images) and the reading quoted it as the condition's image-level q.
    rng = np.random.default_rng(3)
    by_t = {"R1": rng.normal(1, 1, 3), "R2": rng.normal(1, 1, 4), "R3": rng.normal(1, 1, 12)}
    by_c = {"R1": rng.normal(0, 1, 3), "R2": rng.normal(0, 1, 4), "R3": rng.normal(0, 1, 12)}
    assert repeats_to_balance(by_t, {"R1", "R2"}, unbalanced=True) == []
    assert repeats_to_balance(by_t, {"R1"}, unbalanced=True) == ["R2", "R3"]
    test = trusted_test(by_t, by_c, by_c, 199, np.random.default_rng(0))
    assert not test["balanced"]
    assert test["observations"] == 19 and test["control_observations"] == 19
    assert np.isfinite(test["p"])


def test_a_control_against_the_master_is_tested_on_the_scale_of_its_fold_change() -> None:
    # B17: the vs_master_raw row printed a fold change (x0.88) but tested the differences.
    rows = pd.DataFrame(
        [
            {
                "kind": MASTER_RAW,
                "label": "cB vs cA (raw)",
                "condition": "cB",
                "reference": "cA",
                "condition_control": None,
                "reference_control": None,
                "family": MASTER_RAW,
            }
        ]
    )
    means = {("cA", r): m for r, m in zip("123", (10.0, 100.0, 1000.0), strict=True)}
    means |= {("cB", r): m for r, m in zip("123", (8.0, 90.0, 880.0), strict=True)}
    [row] = across_rows(means, rows, list("123"), 0).to_dict("records")
    expected = np.log([0.8, 0.9, 0.88])
    assert row["values"] == "normalised"
    assert row["effects"] == pytest.approx(expected)
    assert row["fold_change"] == pytest.approx(np.exp(expected.mean()))
    # A signed feature keeps the difference, and says so.
    signed = {key: value - 50 for key, value in means.items()}
    [raw] = across_rows(signed, rows, list("123"), 0).to_dict("records")
    assert raw["values"] == "raw" and np.isnan(raw["fold_change"])
    assert raw["effects"] == pytest.approx([-2.0, -10.0, -120.0])


def test_each_kind_of_row_across_experiments_gives_its_own_reason_for_no_whole_table_test():
    # B16: "a difference of log ratios has no meaning" was said of the master rows too, which
    # compare raw centres.
    loaded = make_scenarios.load("a23_embed2_several_image_tech_fp_dino")
    setup = Setup(**{**loaded.setup, "control_parents": None})
    assert set(setup.contrast_table["kind"]) >= {ACROSS, MASTER_RAW}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(loaded.objects, loaded.fovs, setup, feature_tables=loaded.feature_tables)
    full = " ".join(result.summary("full").split())
    assert "a difference of log ratios has no meaning in a table such as an embedding" in full
    assert "not the controls with the master" in full
    assert across_reason([MASTER_RAW], features=True) == (
        "The rows across experiments of the contrast table are tested feature by feature "
        "only: the whole-table test compares each condition with its own control, not the "
        "controls with the master."
    )
    assert across_reason(["own_control"], features=True) == ""


def test_the_exact_interval_of_d_covers_at_its_level() -> None:
    # B1: every new interval with a coverage test (AGENTS.md).
    rng = np.random.default_rng(10)
    covered, reps, true = 0, 1500, 0.8
    for _ in range(reps):
        first, second = rng.normal(true, 1, 6), rng.normal(0, 1, 5)
        low, high = noncentral_d_interval(cohen_d(first, second), 6, 5)
        covered += low <= true <= high
    assert 0.93 <= covered / reps <= 0.97
    assert all(np.isnan(noncentral_d_interval(0.5, 1, 6)))


def _fields(values: np.ndarray) -> list[list[np.ndarray]]:
    """One side, no wells: one field per value."""
    return [[np.array([v]) for v in values]]


def test_shown_within_a_repeat_holds_its_false_alarm_rate_and_needs_five_fields() -> None:
    # B1, A16: the percentile bootstrap called a null difference "shown" in 22% of 3-vs-3
    # comparisons and 11-15% from 5 vs 4; one image per field and no wells now gets the
    # exact interval, and fewer than 5 fields a side is not assessable.
    rng = np.random.default_rng(11)
    shown = 0
    for _ in range(400):
        t, c = rng.normal(0, 1, 5), rng.normal(0, 1, 5)
        row = sampling_row(_fields(t), _fields(c), cohen_d(t, c), rng, n_boot=50)
        assert row["interval"] == "noncentral t" and row["assessable"]
        shown += row["resolved"]
    assert shown / 400 <= 0.08
    few = sampling_row(_fields([5.0, 6, 7]), _fields([0.0, 1, 2]), 5.0, rng, n_boot=50)
    assert not few["assessable"] and not few["resolved"]
    assert np.isfinite(few["d_low"])  # kept for the plan of the next experiment
    # Wells or several observations per field: the bootstrap, as before.
    wells = [[np.array([1.0, 2.0]), np.array([1.5, 2.5])]] * 3
    row = sampling_row(wells, wells, 0.1, rng, n_boot=50)
    assert row["interval"] == "bootstrap"


def test_question_2_says_not_assessable_below_five_fields_a_side() -> None:
    # e01: three fields a side in each repeat.
    text = " ".join(_scenario_report("e01_two_repeats").summary("full").split())
    assert f"not assessable: fewer than {MIN_FIELDS} fields a side" in text
    assert "yes (shown)" not in text
    assert "not assessed: too few fields" not in text


def test_the_count_rule_with_whole_fields_says_its_draws_repeat_fields() -> None:
    # A16: "fields taken" 4-18 of 3 acquired fields, the draws with replacement unsaid.
    rng = np.random.default_rng(12)
    rows = []
    for well, fields in (("w1", 3), ("w2", 8)):
        for field in range(fields):
            for label in range(20):
                rows.append(
                    {
                        "condition": "ctrl",
                        "biological_repeat": "r1",
                        "technical_repeat": well,
                        "image_file": f"{well}_{field}.tif",
                        "label_id": label,
                        "x": 10 + rng.normal(0, 1 + field % 3),
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "technical_repeat", "image_file"]]
    result = object_rule_check(
        objects, fovs.drop_duplicates(), feature="x", observation_unit="cell", n_draws=50
    )
    status = dict(zip(result.table["technical_repeat"], result.table["status"], strict=False))
    assert status["w1"] == f"fewer than {MIN_FIELDS} fields with cells"
    assert status["w2"] == "ok"
    short = " ".join(result.summary().split())
    full = " ".join(result.summary("full").split())
    assert "drawn at random with replacement" in short
    assert "with replacement, until that many, so it can exceed the fields acquired" in full


def _distribution_tables(factor: float, spread: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three repeats of 6 fields of 20 cells a side: ``a`` the control's lognormal cells
    times ``factor``, their log spread times ``spread``."""
    rng = np.random.default_rng(13)
    rows = []
    for repeat in ("r1", "r2", "r3"):
        for condition in ("ctrl", "a"):
            for field in range(6):
                logs = rng.normal(0, 0.5, 20)
                if condition == "a":
                    logs = logs * spread + np.log(factor)
                for label, value in enumerate(np.exp(logs)):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{repeat}_{condition}_{field}.tif",
                            "label_id": label,
                            "area": 100 * value,
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


def test_beyond_the_mean_takes_out_the_mean_shift_on_the_features_scale() -> None:
    # C2: a condition that only moved its mean read "differs beyond the mean", the quantile
    # shifts holding the mean's shift (a25), and a fold change moves every quantile of a raw
    # positive feature.
    objects, fovs = _distribution_tables(2.2)
    result = distribution_contrast(
        objects, fovs, features=["area"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    assert result.settings["log_scale"] == {"area": True}
    shifts = result.tests[result.tests["statistic"].isin(["q10", "q50", "q90"])]
    assert (shifts["mean"].abs() < 0.2).all()
    assert result.differing().empty
    # On the difference scale, the quantile shifts beyond the mean are the skew's alone.
    raw = distribution_contrast(
        objects,
        fovs,
        features=["area"],
        control={"condition": "ctrl"},
        observation_unit="cell",
        scales={"area": "difference"},
    )
    assert raw.settings["log_scale"] == {"area": False}
    # A wider distribution, the same median on the log scale: its spread differs.
    objects, fovs = _distribution_tables(1.0, spread=2.0)
    wide = distribution_contrast(
        objects, fovs, features=["area"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    assert set(wide.differing()["statistic"].iloc[0]) >= {"spread"}


def test_beyond_the_mean_says_not_significant_and_no_trend_from_two_repeats() -> None:
    # C16: "the same way in every repeat, not tested (fewer than 3 repeats)" read as a trend
    # from 2 repeats (a25); C2: "no difference" for "not significant" (a24).
    a25 = " ".join(_scenario_report("a25_mix_several_norm_cells_tech_fp_expl").summary().split())
    beyond = a25[a25.index("Beyond the mean") :]
    assert "differs beyond the mean" not in beyond
    assert "the same way in every repeat, not tested" not in beyond
    assert "in both repeats" in beyond
    a24 = _scenario_report("a24_expl_one_norm_image_none_fp_expl")
    text = " ".join(a24.summary().split())
    assert "no difference beyond the mean" not in text
    assert "no difference in the distributions" not in a24.document.headlines["conditions"]
    assert "beyond the mean (exploratory), not significant" in a24.document.headlines["conditions"]


def test_the_whole_table_verdict_names_thin_controls_and_the_sign_flip_floor() -> None:
    # B10: the whole-DINO verdicts rested on a repeat whose control held one image; and
    # with n repeats an exact sign-flip test cannot go below 2^(1-n).
    assert [sign_floor(n) for n in (3, 5)] == [0.25, 0.0625]
    loaded = make_scenarios.load("a03_mix_one_norm_image_none_fp_dino")
    repeats = sorted(set(loaded.fovs["biological_repeat"]))
    keys = ["condition", "biological_repeat", "image_file"]

    def kept(table: pd.DataFrame) -> pd.DataFrame:
        # The control keeps one field in the last repeat.
        last = (table["condition"] == "ctrl") & (table["biological_repeat"] == repeats[-1])
        first = table.loc[last, "image_file"].min()
        return table[~last | (table["image_file"] == first)]

    attached = attach_feature_tables(loaded.objects, loaded.fovs, loaded.feature_tables)
    fovs, objects = kept(loaded.fovs), kept(attached.objects)
    assert len(fovs[keys].drop_duplicates()) == len(loaded.fovs) - 3
    tables = {"DINO1": list(attached.columns["DINO1"])}
    result = whole_table_tests(objects, fovs, tables=tables, pairs=[("a", "ctrl")])
    [row] = result.tests.to_dict("records")
    assert row["few_control"] == f"{repeats[-1]} (1)"
    assert row["sign_floor"] == 0.25
    line = " ".join(" ".join(whole_table_lines(result)).split())
    assert "an exact sign-flip test could give no p below 0.25 with 3 repeats" in line
    assert f"the control holds fewer than 3 fields in {repeats[-1]} (1)" in line


def _three_repeats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three repeats, ctrl and kd, 4 fields of 30 cells a side: ``steady`` changes by x0.83
    in every repeat almost exactly (each repeat's cells the control's own, scaled), the other
    features by a factor that varies between the repeats."""
    rng = np.random.default_rng(14)
    factors = {"steady": (0.83, 0.83, 0.83), "f1": (0.6, 0.9, 1.0), "f2": (0.7, 1.2, 0.95)}
    factors |= {"f3": (1.3, 0.8, 1.1), "f4": (0.75, 1.0, 0.7)}
    rows = []
    for index, repeat in enumerate(("r1", "r2", "r3")):
        cells = {f: np.exp(rng.normal(0, 0.3, (4, 30))) * 10 for f in factors}
        for condition in ("ctrl", "kd"):
            for field in range(4):
                for label in range(30):
                    row = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "image_file": f"{repeat}_{condition}_{field}.tif",
                        "label_id": label,
                    }
                    for feature, values in cells.items():
                        scale = factors[feature][index] if condition == "kd" else 1.0
                        row[feature] = values[field, label] * scale
                    rows.append(row)
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


def test_a_verdict_on_three_near_identical_repeats_is_said_and_cautioned() -> None:
    # A10: a glioma verdict rested on 3 repeats whose log ratios agreed to 0.0018 (median
    # 0.18 over the family): p 3e-5 from its own tiny SD.
    objects, fovs = _three_repeats()
    discovery = feature_discovery(
        objects,
        fovs,
        features=["steady", "f1", "f2", "f3", "f4"],
        control={"condition": "ctrl"},
        observation_unit="cell",
        n_permutations=199,
    )
    tests = discovery.tests.set_index("feature")
    assert tests.loc["steady", "near_identical"]
    assert not tests.drop(index="steady")["near_identical"].any()
    assert tests.loc["steady", "repeat_sd"] < 0.2 * tests.loc["steady", "family_sd"]
    resting = discovery.resting()
    assert list(resting["feature"]) == ["steady"]
    text = " ".join(discovery.summary().split())
    assert "Rests on 3 near-identical repeats: steady for kd" in text
    assert "under a fifth of its median" in text


def test_the_moderated_paired_t_borrows_the_familys_variance() -> None:
    # A10: the P1 candidate (limma-style empirical Bayes) for 3 near-identical repeats.
    sys.path.insert(0, str(SCENARIOS.parent / "simulations"))
    try:
        from decisions.candidates import repeat_means
    finally:
        sys.path.remove(str(SCENARIOS.parent / "simulations"))
    rng = np.random.default_rng(15)
    family = {f"f{i}": rng.normal(0, 0.18, 3) for i in range(30)}
    family["steady"] = np.array([0.180, 0.182, 0.184])
    found = repeat_means.moderated_t(family)
    assert repeat_means.t_summary(family["steady"])["p"] < 1e-3
    assert found["steady"]["p"] > 0.01
    assert found["steady"]["prior_df"] > 0
    # Calibrated under the null at 3 repeats: about 5% of the features at p <= 0.05.
    rejected = [
        result["p"] <= 0.05
        for _ in range(200)
        for result in repeat_means.moderated_t(
            {f"f{i}": rng.normal(0, rng.choice([0.1, 0.2]), 3) for i in range(20)}
        ).values()
    ]
    assert 0.03 <= np.mean(rejected) <= 0.07


def _sets(layout: dict[str, list[str]], shifts: dict[str, float]) -> tuple[pd.DataFrame, ...]:
    """Conditions imaged in the repeats ``layout`` gives, 3 fields of 20 cells each, two
    lognormal features; ``shifts``: each condition's log shift of both."""
    rng = np.random.default_rng(16)
    rows = []
    for condition, repeats in layout.items():
        for repeat in repeats:
            batch = rng.normal(0, 0.1)
            for field in range(3):
                for label in range(20):
                    row = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "image_file": f"{condition}_{repeat}_{field}.tif",
                        "label_id": label,
                    }
                    for feature in ("area", "length"):
                        noise = rng.normal(0, 0.3) + batch + shifts.get(condition, 0.0)
                        row[feature] = 10 * np.exp(noise)
                    rows.append(row)
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


SCREEN = {
    "ctrl": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "siA": ["R1", "R2", "R3"],
    "siB": ["R4", "R5", "R6"],
}


def test_a_screen_against_one_control_gives_the_screen_wide_q_beside_each_verdict() -> None:
    # A11: the filopodia hits were significant within families of 2 tests each; over the
    # 12 tests against siCTRL, q 0.063 and 0.054.
    objects, fovs = _sets(SCREEN, {"siA": 0.4, "siB": 0.3})
    common = {"features": ["area", "length"], "observation_unit": "cell", "n_permutations": 199}
    apart = feature_discovery(objects, fovs, control={"condition": "ctrl"}, **common)
    tests = apart.tests
    assert tests["experiment"].nunique() == 2
    expected = benjamini_hochberg(tests["p_paired"].to_numpy(float))
    assert tests["q_screen"].to_numpy() == pytest.approx(expected)
    assert "(screen-wide q " in " ".join(apart.summary().split())
    # Declared one experiment (a screen): one family, its q the screen's, none beside it.
    one = feature_discovery(
        objects,
        fovs,
        control={"condition": "ctrl"},
        families={"siA": "screen", "siB": "screen"},
        **common,
    )
    assert one.tests["experiment"].nunique() == 1
    assert one.tests["q_paired"].to_numpy() == pytest.approx(expected)
    assert one.tests["q_screen"].isna().all()
    # The readout's reading carries it too.
    setup = Setup(readout="area", features=("area", "length"), control={"condition": "ctrl"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = " ".join(report(objects, fovs, setup).summary("full").split())
    assert "(screen-wide q " in text


def test_declared_experiments_are_the_correction_families() -> None:
    # B11: THBS5 vs VEC (untransfected) was corrected with the siRNA set imaged in the same
    # repeats; declared apart, each experiment is its own family.
    layout = {
        "VEC": ["R1", "R2", "R3"],
        "THBS5": ["R1", "R2", "R3"],
        "VEC_SiNC": ["R1", "R2", "R3", "R4"],
        "VEC_SiA": ["R1", "R2", "R3", "R4"],
        "VEC_SiB": ["R1", "R2", "R3", "R4"],
    }
    objects, fovs = _sets(layout, {"THBS5": 0.3, "VEC_SiA": 0.5})
    fovs = fovs.assign(set=np.where(fovs["condition"].isin(["VEC", "THBS5"]), "untr", "siRNA"))
    control = {"condition": {"VEC": ["THBS5"], "VEC_SiNC": ["VEC_SiA", "VEC_SiB"]}}
    declared = Setup(readout="area", features=("area", "length"), control=control)
    assert declared.families(fovs) is None
    setup = Setup(readout="area", features=("area", "length"), control=control, experiment="set")
    families = setup.families(fovs)
    assert families["THBS5"] == "untr" and families["VEC_SiA"] == "siRNA"
    assert "VEC" not in families or families["VEC"] == "untr"
    common = {"features": ["area", "length"], "observation_unit": "cell", "n_permutations": 199}
    common |= setup.contrast_arguments
    together = feature_discovery(objects, fovs, **common).tests
    apart = feature_discovery(objects, fovs, families=families, **common).tests
    assert together["experiment"].nunique() == 1
    split = apart.groupby("experiment")["condition"].agg(lambda c: tuple(sorted(set(c))))
    assert sorted(split) == [("THBS5 vs VEC",), ("VEC_SiA vs VEC_SiNC", "VEC_SiB vs VEC_SiNC")]
    mine = apart[apart["condition"].astype(str).str.startswith("THBS5")]
    expected = benjamini_hochberg(mine["p_paired"].to_numpy(float))
    assert mine["q_paired"].to_numpy() == pytest.approx(expected)
    words = " ".join(setup.declaration())
    assert "Experiments: declared by the column 'set', each corrected on its own." in words
    with pytest.raises(TypeError, match="experiment must be"):
        Setup(readout="area", control={"condition": "VEC"}, experiment=3)
    with pytest.raises(ValueError, match="not a column of the FOV table"):
        Setup(readout="area", control=control, experiment="plate").families(fovs)


def test_the_report_runs_with_declared_experiments_and_the_guided_session_asks() -> None:
    # B11: the Setup's experiment column reaches every family; the guided session asks
    # whether sets of conditions are separate experiments, No by default.
    layout = {"ctrl": SCREEN["ctrl"], "siA": SCREEN["siA"], "siB": SCREEN["siB"]}
    objects, fovs = _sets(layout, {"siA": 0.4, "siB": 0.3})
    fovs = fovs.assign(screen="one")
    setup = Setup(
        readout="area",
        features=("area", "length"),
        control={"condition": "ctrl"},
        experiment="screen",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    tests = result.discovery.tests
    assert tests["experiment"].nunique() == 1 and tests["q_screen"].isna().all()
    assert "screen-wide" not in result.summary("full")
    scan = SimpleNamespace(fovs=fovs)
    prompts: list[str] = []

    def answer(text: str) -> str:
        prompts.append(text)
        return "y"

    found = ask_experiment(answer, lambda text: None, scan, {"condition": "ctrl"})
    assert found == {"siA": "screen", "siB": "screen"}
    assert prompts[0].startswith("Are they one experiment against ctrl")
    assert ask_experiment(lambda text: "", print, scan, {"condition": "ctrl"}) is None
    several = {"condition": {"c1": ["a"], "c2": {"k": "b"}}}
    assert ask_experiment(answer, print, scan, several) == {"a": "c1", "b": "c2"}


def test_nothing_compared_is_said_never_no_consistent_difference() -> None:
    # F12 note (e02): the chip read "⚠️ No consistent difference" when nothing was compared.
    result = _scenario_report("e02_control_in_other_repeats")
    status = result.document.statuses["conditions"]
    headline = result.document.headlines["conditions"]
    assert status is Status.NOT_ASSESSED
    assert headline.startswith("nothing compared: no biological repeat holds both a condition")
    assert "Nothing compared" in chip(status, "conditions", headline)
    assert "No consistent difference" not in chip(status, "conditions", headline)
