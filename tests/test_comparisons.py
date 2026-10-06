"""Several declared controls and the contrast table (owner, 2026-10-05).

A design like the ECM dataset (``ecm_design``): eight conditions in six biological repeats,
unevenly (VEC and THBS5 in R1-R3 only, R5 without a SiNC control), three controls keyed by
treatment, two parents keyed ``transfection``, the master VEC_SiNC. Per feature: 7 main
verdicts (own_control), 1 control against the master on raw values, 3 conditions of
different experiments with the same treatment each normalised to its own control: 11 rows.
One control behaves exactly as before.
"""

import numpy as np
import pandas as pd
import pytest
from ecm_design import CONDITIONS, CONTROLS, MASTER, PARENTS, tables

from morphocoverage._control import (
    CONTRAST_COLUMNS,
    check_contrast_table,
    comparison_label,
    control_mask,
    versus,
)
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_across import _effects, across_rows
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.ranking import feature_ranking
from morphocoverage.statistics import benjamini_hochberg
from morphocoverage.workflows import Setup, plots, report

FAST = {"n_permutations": 99, "n_draws": 10, "object_counts": ()}
OWN = [
    ("VEC_SiCol6a1", "VEC_SiNC"),
    ("VEC_SiFBN2", "VEC_SiNC"),
    ("THBS5_SiCol6a1", "THBS5_SiNC"),
    ("THBS5_SiFBN2", "THBS5_SiNC"),
    ("THBS5", "VEC"),
    ("VEC_SiNC", "VEC"),
    ("THBS5_SiNC", "THBS5"),
]
ACROSS = [
    "SiCol6a1 effect: THBS5_SiCol6a1/THBS5_SiNC vs VEC_SiCol6a1/VEC_SiNC",
    "SiFBN2 effect: THBS5_SiFBN2/THBS5_SiNC vs VEC_SiFBN2/VEC_SiNC",
    "transfection effect: THBS5_SiNC/THBS5 vs VEC_SiNC/VEC",
]


def _setup(**kwargs) -> Setup:
    arguments = {
        "control": CONTROLS,
        "control_parents": PARENTS,
        "master_control": MASTER,
        "observation_unit": "cell",
        **kwargs,
    }
    return Setup(readout="area", features=("area", "order"), **arguments)


@pytest.fixture(scope="module")
def ecm():
    return tables(effects={"THBS5_SiCol6a1": 0.4})


@pytest.fixture(scope="module")
def ecm_report(ecm):
    objects, fovs = ecm
    return report(objects, fovs, _setup(normalize="control_mean"))


def test_the_ecm_declaration_gives_eleven_contrasts():
    table = _setup().contrast_table
    assert list(table.columns) == CONTRAST_COLUMNS
    own = table[table["kind"] == "own_control"]
    assert list(zip(own["condition"], own["reference"], strict=True)) == OWN
    raw = table[table["kind"] == "vs_master_raw"]
    # VEC against VEC_SiNC is the parent link already: only THBS5_SiNC.
    assert list(raw["label"]) == ["THBS5_SiNC vs VEC_SiNC (raw)"]
    across = table[table["kind"] == "across_normalised"]
    assert list(across["label"]) == ACROSS
    row = across.iloc[2]
    assert (row["condition"], row["condition_control"]) == ("THBS5_SiNC", "THBS5")
    assert (row["reference"], row["reference_control"]) == ("VEC_SiNC", "VEC")
    assert len(table) == 11
    assert set(table["family"]) == {"own_control", "vs_master_raw", "across_normalised"}


def test_without_a_master_or_keys_there_are_fewer_contrasts():
    listed = {"condition": {"A": ["a1", "a2"], "B": ["b1"]}}
    table = Setup(readout="area", control=listed).contrast_table
    assert set(table["kind"]) == {"own_control"} and len(table) == 3
    keyed = Setup(readout="area", control=CONTROLS).contrast_table
    assert (keyed["kind"] == "vs_master_raw").sum() == 0
    assert (keyed["kind"] == "across_normalised").sum() == 2  # no parents, no transfection


def test_one_control_is_unchanged():
    setup = Setup(readout="area", control={"condition": "VEC_SiNC"})
    assert setup.contrast_table is None and setup.comparisons is None
    assert setup.contrast_arguments == {"control": {"condition": "VEC_SiNC"}}
    assert setup.reference_control == setup.figure_control == setup.single_control
    assert not any(line.startswith("Controls:") for line in setup.declaration())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control": {"condition": {"A": ["x"], "B": ["x"]}}}, "belongs to two controls"),
        ({"control": {"condition": {"A": ["A"]}}}, "its own control"),
        ({"control": {"condition": {"A": "x"}}}, "treatment keys"),
        (
            {
                "control": {"condition": {"A": ["a"], "B": ["b"]}},
                "control_parents": {"A": "B", "B": "a"},
            },
            "cycle",
        ),
        (
            {"control": {"condition": {"A": ["a"]}}, "control_parents": {"Z": "a"}},
            "not a declared control",
        ),
        (
            {"control": {"condition": {"A": ["a"]}}, "master_control": "a"},
            "not one of the declared",
        ),
        ({"control": {"condition": "A"}, "control_parents": {"A": "B"}}, "several controls"),
    ],
)
def test_several_controls_are_checked(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        Setup(readout="area", **kwargs)


def test_a_member_absent_from_the_data_is_refused(ecm):
    objects, fovs = ecm
    # Two controls: one control in the new forms is one control (owner, 2026-10-05).
    control = {"condition": {"VEC_SiNC": ["VEC_SiXX"], "THBS5_SiNC": ["THBS5_SiCol6a1"]}}
    with pytest.raises(ValueError, match="'VEC_SiXX'"):
        report(objects, fovs, Setup(readout="area", control=control))


def test_a_user_table_replaces_the_generated_one(ecm):
    objects, fovs = ecm
    table = pd.DataFrame(
        {
            "kind": ["own_control", "across_normalised"],
            "condition": ["THBS5_SiCol6a1", "THBS5_SiCol6a1"],
            "reference": ["THBS5_SiNC", "VEC_SiCol6a1"],
            "condition_control": [None, "THBS5_SiNC"],
            "reference_control": [None, "VEC_SiNC"],
        }
    )
    setup = Setup(readout="area", contrasts=table)
    assert setup.comparisons == (("THBS5_SiCol6a1", "THBS5_SiNC"),)
    assert list(setup.contrast_table["family"]) == ["own_control", "across_normalised"]
    contrast = condition_contrast(objects, fovs, feature="area", contrasts=table, **FAST)
    assert list(contrast.table["condition"].unique()) == ["THBS5_SiCol6a1 vs THBS5_SiNC"]
    assert list(contrast.across["comparison"]) == [
        "THBS5_SiCol6a1/THBS5_SiNC vs VEC_SiCol6a1/VEC_SiNC"
    ]
    with pytest.raises(ValueError, match="needs condition_control"):
        check_contrast_table(table.assign(condition_control=None))


def test_main_verdicts_pair_each_condition_with_its_own_control(ecm_report):
    contrast = ecm_report.results["contrast"]
    labels = {comparison_label(c, r) for c, r in OWN}
    assert set(contrast.table["condition"]) == labels
    across = contrast.repeats.set_index("condition")
    assert across.loc["VEC_SiNC vs VEC", "repeats"] == 3  # VEC in R1-R3 only
    assert across.loc["THBS5 vs VEC", "repeats"] == 3
    assert across.loc["VEC_SiCol6a1 vs VEC_SiNC", "repeats"] == 5  # no VEC_SiNC in R5
    assert across.loc["THBS5_SiCol6a1 vs THBS5_SiNC", "q_paired_t_normalised"] <= 0.05
    assert set(contrast.cells["condition"]) <= labels  # cell-level tests: main verdicts only
    assert set(ecm_report.results["discovery"].tests["condition"]) <= labels


def test_secondary_rows_are_their_own_families(ecm_report):
    across = ecm_report.results["contrast"].across.set_index("comparison")
    assert len(across) == 4
    assert across.loc["THBS5_SiNC vs VEC_SiNC (raw)", "repeats"] == 5
    assert across.loc[ACROSS[2], "repeats"] == 3  # R1-R3 hold all four
    planted = across.loc[ACROSS[0]]
    assert planted["values"] == "normalised" and planted["q_paired_t"] <= 0.05
    # Each family is corrected over every feature's rows (owner, 2026-10-05).
    every = ecm_report.results["contrast"].settings["across_features"]
    for family, block in every.groupby("family"):
        expected = benjamini_hochberg(block["p_paired_t"].to_numpy(float))
        assert np.allclose(block["q_paired_t"], expected, equal_nan=True), family


def test_the_report_leads_with_the_main_verdicts_then_compares_across(ecm_report):
    text = ecm_report.summary()
    section = text[text.index("9. Do the conditions differ, and how confidently?") :]
    assert section.index("THBS5_SiCol6a1 vs THBS5_SiNC") < section.index(
        "Compared across experiments"
    )
    flat = " ".join(section.split())
    assert "raw comparisons across experiments assume they were imaged and processed" in flat
    assert "VEC_SiNC was imaged in 5 of the 6 biological repeats, not in R5." in flat
    assert "The contrasts tested, one row per test (contrasts.csv)" in flat
    assert "Controls: VEC_SiNC for VEC_SiCol6a1 and VEC_SiFBN2" in " ".join(text.split())


def test_normalised_figures_show_each_condition_over_its_own_control(ecm):
    objects, fovs = ecm
    setup = _setup(normalize="control_mean")
    prepared = setup.prepared(objects, fovs)
    condition, repeat = objects["condition"], objects["biological_repeat"]
    for name, own in [("VEC_SiCol6a1", "VEC_SiNC"), ("VEC_SiNC", "VEC"), ("VEC", "VEC")]:
        mine = (condition == name).to_numpy()
        inside = mine & repeat.isin(set(repeat[(condition == own).to_numpy()])).to_numpy()
        assert prepared.loc[inside, "area"].notna().all()
        assert prepared.loc[mine & ~inside, "area"].isna().all()
    vec = (condition == "VEC").to_numpy()
    means = prepared[vec].groupby(repeat[vec])["area"].mean()
    assert np.allclose(means, 1.0)  # a control without a parent reads 1
    sentence = setup.normalized_sentence(fovs)
    assert "levels across experiments are read" in sentence and "VEC_SiNC in R4" in sentence
    assert setup.compared_values(objects, fovs) is objects  # the tests read raw values
    figures = plots(objects, fovs, _setup(), pairwise=False)
    shown = {c for part in figures["area"] for c in part.cluster_summaries["condition"]}
    assert shown == set(CONDITIONS)


def test_unusual_cells_use_each_condition_s_own_control(ecm_report):
    unusual = ecm_report.results["unusual"].groups
    row = unusual[unusual["condition"] == "VEC_SiCol6a1"].iloc[0]
    assert {g["condition"] for g in row["reference_groups"]} == {"VEC_SiNC"}
    row = unusual[unusual["condition"] == "VEC"].iloc[0]
    assert {g["condition"] for g in row["reference_groups"]} == {"VEC"}


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
    assert plain.settings["comparisons"] == [] and plain.across.empty


def test_the_other_analyses_run_per_comparison(ecm):
    objects, fovs = ecm
    kwargs = {"comparisons": OWN[:2], "n_permutations": 99}
    labels = {comparison_label(*p) for p in OWN[:2]}
    ranking = feature_ranking(objects, fovs, features=["area", "order"], **kwargs)
    assert set(ranking.tests["condition"]) == labels
    discovery = feature_discovery(objects, fovs, features=["area", "order"], **kwargs)
    assert set(discovery.tests["condition"]) == labels


def _rows(**row) -> pd.DataFrame:
    return check_contrast_table(pd.DataFrame([row]))


def test_a_planted_difference_of_effects_is_found_and_none_holds_its_rate():
    """Repeat-to-repeat shifts shared by the four conditions and noise of each mean: the
    paired t-test on the log-ratio differences gives p <= 0.05 in about 5% without a
    difference, and finds one planted in every repeat."""
    rng = np.random.default_rng(11)
    rows = _rows(
        kind="across_normalised",
        condition="a",
        condition_control="ra",
        reference="b",
        reference_control="rb",
    )
    repeats = [f"R{r}" for r in range(6)]

    def means(effect: float) -> dict:
        found = {}
        for r in repeats:
            shift = rng.normal(0, 0.3)
            for c in ("a", "ra", "b", "rb"):
                bump = effect if c == "a" else 0.0
                found[(c, r)] = 10 * np.exp(shift + rng.normal(0, 0.1) + bump)
        return found

    runs = 2000
    hits = sum(
        across_rows(means(0.0), rows, repeats, 0)["p_paired_t"].iloc[0] <= 0.05 for _ in range(runs)
    )
    assert 0.035 <= hits / runs <= 0.065
    found = across_rows(means(0.5), rows, repeats, 0).iloc[0]
    assert found["values"] == "normalised" and found["p_paired_t"] <= 0.05


def test_a_mean_that_is_not_positive_gives_differences_of_differences():
    sides = [np.array([1.0, 2.0]), np.array([-1.0, 1.0]), np.array([0.5, 0.5]), np.zeros(2)]
    values, effects, fold = _effects("across_normalised", sides)
    assert values == "raw" and np.isnan(fold)
    assert np.allclose(effects, [(1 + 1) - 0.5, (2 - 1) - 0.5])


def test_a_row_held_in_too_few_repeats_is_not_compared():
    means = {("a", "R1"): 1.0, ("b", "R1"): 2.0, ("a", "R2"): 1.0, ("b", "R2"): 2.0}
    rows = _rows(kind="vs_master_raw", condition="a", reference="b")
    row = across_rows(means, rows, ["R1", "R2"], 0).iloc[0]
    assert row["comparison"] == "a vs b (raw)" and row["repeats"] == 2
    assert row["status"].startswith("fewer than 3") and np.isnan(row["p_paired_t"])


def test_the_resolver_names_and_pools_the_controls():
    assert versus("a", "ctrl") == "a vs ctrl"
    assert versus("a vs ctrl", "ctrl") == "a vs ctrl"
    table = pd.DataFrame({"condition": ["VEC", "VEC_SiNC", "THBS5_SiNC"]})
    pooled = {"condition": ("VEC_SiNC", "THBS5_SiNC")}
    assert control_mask(table, pooled).tolist() == [False, True, True]


# --- The audit of 2026-10-05 (owner, 2026-10-05) ----------------------------------------


@pytest.mark.parametrize(
    "declared",
    [
        {"control": {"condition": {"VEC_SiNC": ["VEC_SiCol6a1", "VEC_SiFBN2"]}}},
        {"control": {"condition": {"VEC_SiNC": {"Col6": "VEC_SiCol6a1"}}}},
        {
            "control": {"condition": {"VEC_SiNC": ["VEC_SiCol6a1"]}},
            "master_control": "VEC_SiNC",
        },
    ],
)
def test_one_control_in_the_new_forms_is_one_control(declared):
    setup = Setup(readout="area", normalize="control_mean", **declared)
    old = Setup(readout="area", normalize="control_mean", control={"condition": "VEC_SiNC"})
    assert setup.control == {"condition": "VEC_SiNC"} and setup.master_control is None
    assert setup == old
    assert setup.contrast_table is None and setup.declaration() == old.declaration()


def test_a_condition_with_two_own_controls_is_refused():
    table = pd.DataFrame(
        {
            "kind": ["own_control", "own_control"],
            "condition": ["THBS5_SiCol6a1", "THBS5_SiCol6a1"],
            "reference": ["THBS5_SiNC", "VEC_SiCol6a1"],
        }
    )
    with pytest.raises(ValueError, match="'THBS5_SiCol6a1' two own controls"):
        Setup(readout="area", contrasts=table)


def test_the_own_control_comes_from_the_declaration():
    table = _setup().contrast_table
    row = table["condition"] == "THBS5_SiCol6a1"
    moved = table.assign(reference=table["reference"].where(~row, "VEC_SiCol6a1"))
    with pytest.raises(ValueError, match="the declared controls give 'THBS5_SiNC'"):
        _setup(contrasts=moved)
    # A table without the row keeps the declared own control, as the Controls line says.
    kept = _setup(contrasts=table[~row | (table["kind"] != "own_control")])
    assert kept.own_references["THBS5_SiCol6a1"] == "THBS5_SiNC"
    assert any("THBS5_SiNC for THBS5_SiCol6a1" in line for line in kept.declaration())


def test_the_refusals_say_what_was_declared():
    with pytest.raises(ValueError, match="'x' is listed twice under control 'A'"):
        Setup(readout="area", control={"condition": {"A": ["x", "x"], "B": ["y"]}})
    with pytest.raises(ValueError, match="linked to the parent 'Z'") as raised:
        Setup(
            readout="area",
            control={"condition": {"A": ["B"], "B": ["y"]}},
            control_parents={"B": "Z"},
        )
    assert "control_parents" not in str(raised.value)


def test_the_secondary_rows_run_for_every_feature(ecm_report):
    contrast = ecm_report.results["contrast"]
    every = contrast.settings["across_features"]
    assert set(every["feature"]) == {"area", "order"} and len(every) == 8
    # Each family over every feature and its rows, as the main rows over the features.
    for family, block in every.groupby("family"):
        expected = benjamini_hochberg(block["p_paired_t"].to_numpy(float))
        assert np.allclose(block["q_paired_t"], expected, equal_nan=True), family
    readout = every[every["feature"] == "area"].reset_index(drop=True)
    assert np.allclose(contrast.across["q_paired_t"], readout["q_paired_t"], equal_nan=True)
    # Discovery and the heatmap keep the main rows only.
    labels = {comparison_label(c, r) for c, r in OWN}
    assert set(ecm_report.results["discovery"].tests["condition"]) <= labels
    text = " ".join(ecm_report.summary().split())
    section = text[text.index("Compared across experiments") :]
    assert "first area, the readout" in section
    assert "Of the other feature's 4 rows compared" in section
    assert "every row is in tests_across_experiments.csv" in section
    # Each row names its repeats, and an interaction names the larger ratio.
    assert "R1, R2 and R3" in section and "larger ratio" in section
    glance = text[: text.index("What is in the dataset")]
    assert "across experiments: " in glance and " significant" in glance


def test_not_compared_names_the_comparison_s_own_reference():
    from morphocoverage.sampling_unit._experiments import not_compared

    text = not_compared(["THBS5_SiFBN2 vs THBS5_SiNC"], "THBS5, THBS5_SiNC, VEC and 1 more")
    assert text == "Not compared: THBS5_SiFBN2 vs THBS5_SiNC (no biological repeat holds both)."
    assert not_compared(["a"], "ctrl") == (
        "Not compared: a (no biological repeat holds both it and ctrl)."
    )


def test_the_unusual_cells_are_said_against_each_own_control(ecm_report):
    text = ecm_report.summary(detail="full")
    assert "('VEC" not in text and "('THBS5" not in text and "()" not in text
    flat = " ".join(ecm_report.summary().split())
    assert "each against its own control" in flat
    assert "against each condition's own control" in flat


def test_a_contrast_table_alone_says_the_unusual_cells_it_reports():
    table = _setup().contrast_table
    setup = Setup(readout="area", features=("area", "order"), contrasts=table)
    assert setup.reference_control is not None
    words = " ".join(setup.declaration())
    assert "unusual cells are not assessed" not in words
    assert "for unusual cells (against each condition's own control" in words


def test_the_figure_legend_pairs_each_condition_with_its_own_control(ecm, ecm_report):
    from morphocoverage.workflows._html_tests import figure_legends

    objects, fovs = ecm
    legends = figure_legends(ecm_report, objects, fovs)
    # A condition is "not compared" only in repeats without its own reference.
    assert legends and all(", in repeats without" not in text for text in legends.values())
    assert "VEC_SiCol6a1: n = " in legends["area"]
