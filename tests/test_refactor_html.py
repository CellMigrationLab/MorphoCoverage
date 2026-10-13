"""Characterisation tests of the HTML page's and question 10's branches that neither the golden
runs nor the rest of the suite execute, written on the code before its refactor (2026-10-12):
the exact output of each, so the refactor is shown not to change it."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from morphocoverage._style import Status
from morphocoverage.embedding import Embedding
from morphocoverage.workflows._html_maps import _set
from morphocoverage.workflows._html_render import chip
from morphocoverage.workflows._html_tests import _cells
from morphocoverage.workflows._population_tests import enrichment_tests
from morphocoverage.workflows.html_report import html_report


def test_the_chip_words_the_answers_decide():
    beyond = chip(Status.OK, "conditions", "beyond the mean: spread higher")
    assert beyond == (
        '<span class="chip ok"><span aria-hidden="true">✅</span> Differs beyond the mean</span>'
    )
    nothing = chip(Status.CAUTION, "conditions", "nothing compared: one condition")
    assert nothing.endswith("> Nothing compared</span>")
    assert chip(Status.CAUTION, "populations", "a (exploratory; q 0.01)").endswith(
        "> More common (exploratory)</span>"
    )
    assert chip(Status.CAUTION, "populations", "mostly a1: not a subpopulation").endswith(
        "> A condition's shift</span>"
    )
    assert chip(Status.CAUTION, "populations", "higher in every repeat").endswith(
        "> Trend only</span>"
    )
    assert chip(Status.CAUTION, "conditions", "differs in 1 repeat").endswith(
        "> No consistent difference</span>"
    )
    assert chip(Status.OK, "conditions", "n = 2, higher").endswith("> No conclusion (n = 2)</span>")
    assert chip(Status.CAUTION, "enough", "shown in 0 of 4").endswith("> Not yet</span>")
    assert chip(None, "conditions", "anything") == ""


def _embedding(points: pd.DataFrame) -> Embedding:
    settings = {
        "features": ["a", "b"],
        "umap_points": len(points),
        "n_neighbors": 5,
        "min_dist": 0.1,
    }
    loadings = pd.DataFrame({"feature": ["a", "b"], "pc1": [0.6, -0.8], "pc2": [0.8, 0.6]})
    return Embedding(points, loadings.assign(top=False), (0.7, 0.3), settings)


def test_a_map_without_points_is_none_and_one_without_conditions_draws_them_alike():
    empty = pd.DataFrame({"pc1": [np.nan], "pc2": [np.nan], "field": ["f1"]})
    assert _set(_embedding(empty), "pc", {}, "raw") is None
    points = pd.DataFrame(
        {
            "pc1": [1.0, -2.0],
            "pc2": [0.5, 1.5],
            "umap1": [0.1, 0.2],
            "umap2": [0.3, 0.4],
            "field": ["f1", "f2"],
            "observation": [0, 1],
        }
    )
    drawn = _set(_embedding(points), "umap", {}, "raw", line="Neighbours: none.")
    assert drawn == {
        "x": [0.1, 0.2],
        "y": [0.3, 0.4],
        "cond": [0, 0],
        "labels": ["f1", "f2"],
        "caption": (
            "UMAP of 2 standardised features (raw values), 2 observations (n_neighbors 5, "
            "min_dist 0.1); its axes have no units.\nNeighbours: none."
        ),
        "xlabel": "UMAP 1",
        "ylabel": "UMAP 2",
        "arrows": [],
        "obs": [0, 1],
        "z": {},
    }
    pca = _set(_embedding(points), "pc", {}, "raw")
    assert pca["cond"] == [0, 0]
    assert pca["arrows"] == [["a", 1.2, 1.6], ["b", -1.6, 1.2]]
    assert pca["caption"] == (
        "PCA of 2 standardised features (raw values): PC1 70% and PC2 30% of the variance; "
        "arrows: the 2 largest loadings, as no feature ranks.\n"
        "PC1, strongest loadings: b -0.80, a +0.60; a feature with + is higher to the right.\n"
        "PC2, strongest loadings: a +0.80, b +0.60; a feature with + is higher towards the top."
    )


def test_the_cell_level_screen_is_empty_without_tests_or_cells():
    result = SimpleNamespace(
        discovery=None, results={"contrast": None}, setup=SimpleNamespace(readout="area")
    )
    assert _cells(result, None) == ""


def test_enrichment_without_a_repeat_column_skips_a_condition_without_cells():
    marked = pd.DataFrame(
        {"condition": ["ctrl"] * 4 + ["drug"] * 4, "G1": [0, 0, 0, 1, 1, 1, 0, 0]}
    )
    tests = enrichment_tests(marked, ["G1"], (("drug", "ctrl"), ("other", "ctrl")))
    assert tests[["feature", "condition", "repeats", "higher", "lower"]].to_dict("records") == [
        {"feature": "G1", "condition": "drug vs ctrl", "repeats": 1, "higher": 1, "lower": 0}
    ]
    assert np.isnan(tests["p_paired"].iloc[0])
    assert tests["share"].iloc[0] == 0.5 and tests["control_share"].iloc[0] == 0.25
    assert tests.attrs["unpaired"] == []


def test_the_page_refuses_anything_but_a_report(tmp_path):
    with pytest.raises(TypeError, match="^result must be the Report that report\\(\\) returned$"):
        html_report(object(), pd.DataFrame(), pd.DataFrame(), tmp_path)


def test_a_field_flagged_only_for_new_states_is_not_a_field_to_check():
    from morphocoverage.workflows._html_cards import flagged_fields

    inspection = SimpleNamespace(
        group_cols=[],
        settings={"fov_key": ["field"]},
        fovs=pd.DataFrame(
            {
                "field": ["f1", "f2", "f2"],
                "reason": ["new state", "influence", "new state"],
                "detail": ["a new state", "moves the mean", "another state"],
            }
        ),
    )
    result = SimpleNamespace(results={"inspection": inspection})
    assert flagged_fields(result) == ([("f2", "moves the mean")], [("f2",)], ["field"])


def test_text_html_joins_a_wrapped_sentence_and_makes_a_table():
    from morphocoverage.workflows._html_render import text_html

    text = (
        "Readings:\n"
        "  condition  d      p\n"
        "  drugA      +0.50  0.010\n"
        "  drugB      -0.20  0.300\n"
        "A sentence that is wrapped\n"
        "  onto a second line.\n"
        "  - an item\n"
        "✅ Fine: nothing to say."
    )
    assert text_html(text) == (
        "<p>Readings:</p>\n"
        '<div class="table-wrap"><table class="sortable"><thead><tr><th>condition</th>'
        '<th>d</th><th>p</th></tr></thead><tbody><tr><td class="">drugA</td>'
        '<td class="num">+0.50</td><td class="num">0.010</td></tr><tr><td class="">drugB</td>'
        '<td class="num">-0.20</td><td class="num">0.300</td></tr></tbody></table></div>\n'
        "<p>A sentence that is wrapped onto a second line.</p>\n"
        '<p class="item" style="margin-left:1.1em">an item</p>\n'
        '<p class="status-line"><span class="mark ok" role="img" aria-label="Answered">✅'
        "</span> Fine: nothing to say.</p>"
    )


def test_no_declared_target_judged_is_said_among_the_targets():
    from morphocoverage.workflows._html_enough import _targets

    precision = pd.DataFrame(
        {
            "criterion": ["covariance", "covariance"],
            "met": [np.nan, np.nan],
            "not_assessable": ["too few fields for its components", np.nan],
        }
    )
    setup = SimpleNamespace(rare=None, mean_target=False, unit_word="cell")
    result = SimpleNamespace(setup=setup, results={"fingerprint_precision": precision})
    lines, tabs = _targets(result, "well", "field")
    assert lines == [
        "The fingerprint is not assessable: too few fields for its components.",
        "No declared target can be judged on these data.",
    ]
    assert [(name, mark) for name, mark, _ in tabs] == [("fingerprint", "➖")]


def test_a_condition_without_a_repeat_plan_is_left_out_of_the_table():
    from morphocoverage.workflows._html_plan import _difference_goal

    discovery = SimpleNamespace(
        tests=pd.DataFrame(
            {
                "feature": ["area"] * 3,
                "condition": ["a", "b", "c"],
                "repeats": [3, 3, 3],
                "confirmed": [True, False, False],
            }
        ),
        ranking=pd.DataFrame({"feature": ["area"]}),
    )
    plan = pd.DataFrame(
        {
            "condition": ["a", "b", "c"],
            "repeats_needed": [4.0, np.nan, 5.0],
            "repeats_needed_twice_fields": [3.0, np.nan, np.inf],
            "planned_on": ["log ratio", "difference", "log ratio"],
            "change": [0.2, np.nan, -0.1],
        }
    )
    contrast = SimpleNamespace(plan=plan, table=pd.DataFrame(), settings={})
    result = SimpleNamespace(
        discovery=discovery,
        setup=SimpleNamespace(readout="area", columns=("area",)),
        results={"contrast": contrast},
    )
    option, panel = _difference_goal(result)
    assert option == "A difference from the control"
    assert panel == (
        "<p>Optimise plans each well&#x27;s own measurements, not a comparison: a difference "
        "from the control is planned in biological repeats. For each condition not yet "
        "significant across repeats, the repeats a paired t-test on the repeat means needs for "
        "80% power, if the change and the variation seen here recur, each repeat with its own "
        "control. Significant across repeats already: area.</p>"
        '<div class="table-wrap"><table class="sortable"><thead><tr><th>feature</th>'
        "<th>condition</th><th>change seen</th><th>repeats needed</th>"
        "<th>with twice the fields per repeat</th></tr></thead><tbody><tr><td>area</td>"
        "<td>c</td><td>×-0.10</td><td>5</td><td>more than 30</td></tr></tbody></table></div>"
    )


@pytest.fixture(scope="module")
def screened():
    import warnings

    from _helpers_rare_fingerprint import _screen, _setup

    from morphocoverage.workflows import report

    objects, fovs = _screen()
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    return objects, fovs, setup, result


def test_a_trend_in_six_repeats_and_an_unpaired_condition_in_question_10(screened):
    from dataclasses import replace

    from morphocoverage.workflows._populations import populations_text

    _, _, setup, result = screened
    found = result.results["populations"]
    enrichment = found.enrichment.copy()
    enrichment.loc[1, ["repeats", "higher", "lower", "q_paired"]] = [6, 6, 0, 0.2]
    enrichment.attrs["unpaired"] = ["late"]
    status, headline, text = populations_text(replace(found, enrichment=enrichment), setup)
    assert status is Status.CAUTION
    assert headline == (
        "1 group of cells to validate on the images; G1 (4.7% of cells; high dna): higher in "
        "arrest in every repeat, not significant (11% vs 1.5%, q 0.20)"
    )
    flat = " ".join(text.split())
    assert "Not compared: late (no biological repeat shared with the control)." in flat


def test_the_page_refuses_tables_of_other_sizes_than_the_reports(screened, tmp_path):
    objects, fovs, _, result = screened
    with pytest.raises(ValueError) as raised:
        html_report(result, objects.iloc[:-1], fovs, tmp_path, check_source=False)
    assert str(raised.value) == (
        "objects and fovs must be the tables the report was run on (4320 observations, "
        "108 fields), not 4319 rows and 108"
    )
