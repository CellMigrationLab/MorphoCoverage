"""Declared comparisons and two conditions over their own references (owner, 2026-10-05).

A design like the ECM dataset (``ecm_design``): eight conditions in six biological repeats,
unevenly. Each ``(condition, reference)`` pair is one test, per repeat, in the repeats where
both are imaged; a condition may be in several; ``(a, ref_a, b, ref_b)`` compares two
conditions each normalised to its own reference, across the repeats that hold all four.
"""

import numpy as np
import pandas as pd
import pytest
from ecm_design import CONDITIONS, tables

from morphocoverage._control import (
    check_comparisons,
    comparison_label,
    control_mask,
    reference_population,
    versus,
)
from morphocoverage.retrospective.effect_preservation import effect_preservation
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_relative import relative_rows, relative_values
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.distribution import distribution_contrast
from morphocoverage.sampling_unit.ranking import feature_ranking

FAST = {"n_permutations": 99, "n_draws": 10, "object_counts": ()}
PAIRS = [
    ("THBS5_SiCol6a1", "THBS5_SiNC"),
    ("THBS5_SiCol6a1", "VEC_SiCol6a1"),
    ("THBS5_SiNC", "VEC_SiNC"),
    ("VEC", "VEC_SiNC"),
]
RELATIVE = [
    ("THBS5_SiCol6a1", "THBS5_SiNC", "VEC_SiCol6a1", "VEC_SiNC"),
    ("THBS5_SiFBN2", "THBS5_SiNC", "VEC_SiFBN2", "VEC_SiNC"),
]


@pytest.fixture(scope="module")
def ecm():
    return tables(effects={"THBS5_SiCol6a1": 0.4})


@pytest.fixture(scope="module")
def contrast(ecm):
    objects, fovs = ecm
    return condition_contrast(
        objects, fovs, feature="area", comparisons=PAIRS, relative=RELATIVE, **FAST
    )


def test_each_comparison_is_paired_in_the_repeats_holding_both(contrast):
    table = contrast.table
    labels = [comparison_label(c, r) for c, r in PAIRS]
    assert set(table["condition"]) == set(labels)
    for (condition, reference), label in zip(PAIRS, labels, strict=True):
        rows = table[table["condition"] == label]
        assert set(rows["control_condition"]) == {reference}
        compared = rows[rows["control_mean"].notna()]
        assert len(compared) == (5 if condition == "VEC" else 6)
    # VEC was not imaged in R5: its comparison has no R5 row; VEC_SiNC was, so the cell-level
    # test pools a reference repeat VEC lacks and is given balanced.
    vec = contrast.cells.set_index("condition").loc["VEC vs VEC_SiNC"]
    assert vec["other_control_repeats"] == 1 and vec["unbalanced"]


def test_a_condition_in_two_comparisons_is_tested_twice_and_a_reference_is_tested(contrast):
    across = contrast.repeats.set_index("condition")
    assert across.loc["THBS5_SiCol6a1 vs THBS5_SiNC", "repeats"] == 6
    assert across.loc["THBS5_SiCol6a1 vs VEC_SiCol6a1", "repeats"] == 6
    assert across.loc["THBS5_SiNC vs VEC_SiNC", "repeats"] == 6
    assert across.loc["THBS5_SiCol6a1 vs THBS5_SiNC", "q_paired_t_normalised"] <= 0.05
    assert across.loc["THBS5_SiNC vs VEC_SiNC", "q_paired_t_normalised"] > 0.05
    # Every comparison shares the repeats: one Benjamini-Hochberg family.
    assert across["experiment"].nunique() == 1
    assert set(contrast.relative["experiment"]) == set(across["experiment"])


def test_comparisons_against_one_control_give_the_control_s_paired_tests():
    objects, fovs = tables(seed=3, effects={"VEC_SiFBN2": 0.3})
    plain = condition_contrast(
        objects, fovs, feature="area", control={"condition": "VEC_SiNC"}, **FAST
    )
    pairs = [(c, "VEC_SiNC") for c in CONDITIONS if c != "VEC_SiNC"]
    paired = condition_contrast(objects, fovs, feature="area", comparisons=pairs, **FAST)
    left = plain.repeats.set_index("condition")
    right = paired.repeats.assign(
        condition=paired.repeats["condition"].str.replace(" vs VEC_SiNC", "", regex=False)
    ).set_index("condition")
    for column in ("repeats", "p_paired_t", "q_paired_t", "p_paired_t_normalised"):
        assert np.allclose(left[column], right.loc[left.index, column], equal_nan=True)
    assert plain.settings["comparisons"] == [] and "THBS5" in plain.repeats["condition"].values


def test_a_relative_comparison_finds_a_planted_difference_of_effects(contrast):
    rows = contrast.relative.set_index("comparison")
    planted = rows.loc["THBS5_SiCol6a1/THBS5_SiNC vs VEC_SiCol6a1/VEC_SiNC"]
    assert planted["values"] == "normalised" and planted["repeats"] == 6
    assert planted["effect"] == pytest.approx(0.4, abs=0.1)
    assert planted["q_paired_t"] <= 0.05
    # THBS5_SiFBN2 was not imaged in R6: 5 repeats hold all four, and nothing is planted.
    other = rows.loc["THBS5_SiFBN2/THBS5_SiNC vs VEC_SiFBN2/VEC_SiNC"]
    assert other["repeats"] == 5 and other["q_paired_t"] > 0.05


def test_without_a_difference_of_effects_the_relative_test_holds_its_rate():
    """Repeat-to-repeat shifts shared by the four conditions and noise of each condition's
    mean: the paired t-test on the log-ratio differences gives p <= 0.05 in about 5%."""
    rng = np.random.default_rng(11)
    conditions = ("a", "ra", "b", "rb")
    repeats = [f"R{r}" for r in range(6)]
    hits, runs = 0, 2000
    for _ in range(runs):
        means = {}
        for r in repeats:
            shift = rng.normal(0, 0.3)
            for c in conditions:
                means[(c, r)] = 10 * np.exp(shift + rng.normal(0, 0.1))
        row = relative_rows(means, [conditions], repeats, 0).iloc[0]
        hits += row["p_paired_t"] <= 0.05
    assert 0.035 <= hits / runs <= 0.065


def test_a_mean_that_is_not_positive_gives_differences_of_differences():
    values, effects = relative_values(
        np.array([1.0, 2.0]), np.array([-1.0, 1.0]), np.array([0.5, 0.5]), np.array([0.0, 0.0])
    )
    assert values == "raw"
    assert np.allclose(effects, [(1 + 1) - 0.5, (2 - 1) - 0.5])


def test_the_other_analyses_run_per_comparison(ecm):
    objects, fovs = ecm
    kwargs = {"comparisons": PAIRS[:2], "n_permutations": 99}
    ranking = feature_ranking(objects, fovs, features=["area", "order"], **kwargs)
    assert set(ranking.tests["condition"]) == {comparison_label(*p) for p in PAIRS[:2]}
    discovery = feature_discovery(objects, fovs, features=["area", "order"], **kwargs)
    assert set(discovery.tests["condition"]) == {comparison_label(*p) for p in PAIRS[:2]}
    distribution = distribution_contrast(objects, fovs, features=["area"], comparisons=PAIRS[:2])
    assert set(distribution.tests["condition"]) <= {comparison_label(*p) for p in PAIRS[:2]}
    kept = effect_preservation(
        objects, fovs, feature="area", comparisons=PAIRS[:1], fov_counts=(2,), n_draws=5
    )
    assert set(kept.table["condition"]) == {comparison_label(*PAIRS[0])}
    assert set(kept.table["control_condition"]) == {"THBS5_SiNC"}


def test_a_comparison_naming_an_absent_condition_is_refused(ecm):
    objects, fovs = ecm
    with pytest.raises(ValueError, match="absent from the tables: 'VEC_SiXX'"):
        condition_contrast(objects, fovs, feature="area", comparisons=[("VEC_SiXX", "VEC")], **FAST)
    with pytest.raises(ValueError, match="with itself"):
        check_comparisons([("VEC", "VEC")])
    with pytest.raises(TypeError):
        check_comparisons("VEC")


def test_the_resolver_names_and_pools_the_references():
    assert versus("a", "ctrl") == "a vs ctrl"
    assert versus("a vs ctrl", "ctrl") == "a vs ctrl"
    pooled = reference_population(None, PAIRS, "condition")
    assert pooled == {"condition": ("THBS5_SiNC", "VEC_SiCol6a1", "VEC_SiNC")}
    table = pd.DataFrame({"condition": ["VEC", "VEC_SiNC", "THBS5_SiNC"]})
    assert control_mask(table, pooled).tolist() == [False, True, True]
    assert reference_population({"condition": "VEC"}, PAIRS, "condition") == {"condition": "VEC"}
