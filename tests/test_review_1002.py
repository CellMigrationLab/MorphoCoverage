"""Regressions of the 2026-10-02 code review of the report: each test pins a finding the
review confirmed with a script, and fails on the code before its fix."""

import json

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import _three

from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.distribution import distribution_contrast
from morphocoverage.statistics import paired_t, repeat_values
from morphocoverage.workflows import Setup
from morphocoverage.workflows._html_data import json_island
from morphocoverage.workflows._html_figures import file_stem
from morphocoverage.workflows._html_render import adjusted_p
from morphocoverage.workflows._populations import _side, enrichment_tests, one_sided
from morphocoverage.workflows._result_influence import influence_on_result, with_result_influence
from morphocoverage.workflows.retrospective import inspect_fovs


def _influence(objects, fovs, control):
    setup = Setup(readout="x", control=control, observation_unit="cell")
    contrast = condition_contrast(
        objects, fovs, feature="x", control=control, observation_unit="cell"
    )
    inspection = inspect_fovs(objects, fovs, feature_cols=["x"], **setup.common)
    return with_result_influence(inspection, objects, fovs, contrast, setup)


def test_a_centred_readout_is_not_judged_by_relative_change():
    objects, fovs = _three()
    objects["x"] = objects["x"] - 10.0  # centred near 0: a relative change explodes
    result = _influence(objects, fovs, {"condition": "ctrl"})
    assert result.fovs.empty or not (result.fovs["reason"] == "influence").any()
    reasons = {entry["reason"] for entry in result.settings["skipped_summaries"]}
    assert reasons == {"has negative values, so a relative change is undefined"}


def test_a_mean_of_zero_is_recorded_as_not_checked():
    objects, fovs = _three()
    zero = (objects["condition"] == "drug") & (objects["biological_repeat"] == "r1")
    objects.loc[zero, "x"] = 0.0
    result = _influence(objects, fovs, {"condition": "ctrl"})
    [entry] = result.settings["skipped_summaries"]
    assert entry["reason"] == "is 0 here, so a relative change is undefined"
    assert "drug" in entry["group"] and "r1" in entry["group"]


def test_a_field_outside_a_two_column_control_is_not_a_control_field():
    rows, fovs = [], []
    rng = np.random.default_rng(4)
    for repeat in ("r1", "r2", "r3"):
        for condition, well in (("ctrl", "A"), ("ctrl", "B"), ("drug", "A")):
            for field in range(4):
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "technical_repeat": well,
                    "position_id": f"{condition}{well}_{repeat}_{field}",
                }
                fovs.append(key)
                shift = 30.0 if (condition, well) == ("ctrl", "B") else 10.0
                rows += [{**key, "x": v} for v in rng.normal(shift, 1.0, 20)]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fovs)
    result = _influence(objects, fovs, {"condition": "ctrl", "technical_repeat": "A"})
    flagged = result.fovs[result.fovs["reason"] == "influence"] if len(result.fovs) else []
    # Well B of ctrl is not the declared control: none of its fields is scored as one.
    assert not len(flagged) or not flagged["position_id"].str.startswith("ctrlB").any()


def test_without_repeats_the_influence_stays_on_the_summary():
    objects, fovs = _three()
    pooled = objects.drop(columns="biological_repeat")
    contrast = condition_contrast(
        pooled,
        fovs.drop(columns="biological_repeat"),
        feature="x",
        control={"condition": "ctrl"},
        observation_unit="cell",
        repeat_col=None,
    )
    assert not influence_on_result(contrast)


def test_the_plan_gives_the_verdicts_change_over_every_paired_repeat():
    objects, fovs = _three({"drug": 12.0})
    contrast = condition_contrast(
        objects, fovs, feature="x", control={"condition": "ctrl"}, observation_unit="cell"
    )
    [plan] = contrast.plan.to_dict("records")
    [across] = contrast.repeats.to_dict("records")
    assert plan["change"] == pytest.approx(across["fold_change"])


def test_one_rule_for_the_values_the_paired_test_compares():
    values, ratios, fold = repeat_values([2.0, 4.0], [1.0, 2.0])
    assert values == "normalised" and fold == pytest.approx(2.0)
    assert np.allclose(ratios, np.log(2.0))
    values, _, mean = repeat_values([1.0, -1.0], [0.5, 0.5])
    assert values == "raw" and mean == pytest.approx(-0.5)
    assert np.isnan(paired_t(np.array([0.1, 0.1, 0.1])))


def test_a_skewed_condition_is_not_said_to_split_in_two():
    objects, fovs = _three()
    drug = objects["condition"] == "drug"
    rng = np.random.default_rng(8)
    objects.loc[drug, "x"] = 10 + rng.exponential(2.0, drug.sum())
    result = distribution_contrast(
        objects, fovs, features=["x"], control={"condition": "ctrl"}, observation_unit="cell"
    )
    assert result.splits().empty


def _shares(counts):
    """``counts``: (condition, repeat) -> (cells in the population, cells)."""
    rows = []
    for (condition, repeat), (inside, cells) in counts.items():
        rows += [
            {"condition": condition, "biological_repeat": repeat, "P2": float(i < inside)}
            for i in range(cells)
        ]
    return pd.DataFrame(rows)


def test_an_enrichment_reads_the_way_its_test_goes():
    # 1 of 5000 cells against 0 of 300, in every repeat: higher, never "lower".
    counts = {}
    for repeat in ("r1", "r2", "r3"):
        counts[("drug", repeat)] = (1, 5000)
        counts[("ctrl", repeat)] = (0, 300)
    [row] = enrichment_tests(_shares(counts), ["P2"], "ctrl").to_dict("records")
    assert row["higher"] == 3 and row["d"] > 0


def test_a_tie_between_repeats_is_mixed_and_one_repeat_is_no_trend():
    assert _side({"higher": 1, "lower": 1, "repeats": 3}) == "mixed (1 higher, 1 lower, of 3)"
    assert _side({"higher": 0, "lower": 0, "repeats": 3}) == "no change in 3 of 3"
    single = pd.DataFrame({"higher": [1], "lower": [0], "repeats": [1]})
    assert not one_sided(single).any()


def test_a_condition_without_a_shared_repeat_is_named():
    counts = {("drug", "r9"): (5, 50), ("ctrl", "r1"): (1, 50), ("ctrl", "r2"): (1, 50)}
    tests = enrichment_tests(_shares(counts), ["P2"], "ctrl")
    assert tests.empty and tests.attrs["unpaired"] == ["drug"]


def test_two_features_of_one_file_name_get_two_files():
    taken: set[str] = set()
    assert [file_stem(name, taken) for name in ("a b", "a_b", "a/b")] == ["a_b", "a_b_2", "a_b_3"]


def test_names_cannot_close_the_data_script():
    text = json_island({"feature": "x<!--<script></script>&"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert json.loads(text) == {"feature": "x<!--<script></script>&"}


def test_the_adjusted_p_wording_leaves_the_figures_data_alone():
    markup = '<p>q 0.04</p><script type="application/json">{"name":"raw q 1"}</script>'
    assert adjusted_p(markup) == (
        '<p>adjusted p 0.04</p><script type="application/json">{"name":"raw q 1"}</script>'
    )


def test_a_sign_flip_test_reads_the_same_at_any_scale():
    from morphocoverage.sampling_unit.contrast_repeats import sign_flip_test

    for scale in (1.0, 1e-12, 1e-13):
        p, floor, exact = sign_flip_test(scale * np.arange(1.0, 8.0), np.random.default_rng(0))
        assert exact and p == pytest.approx(2 / 2**7) and floor == pytest.approx(2 / 2**7)


def test_benjamini_hochberg_keeps_nan_and_matches_the_definition():
    from morphocoverage.statistics import benjamini_hochberg

    p = np.array([0.01, np.nan, 0.04, 0.03, 1.0])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    assert np.allclose(q[[0, 2, 3, 4]], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 1.0])
