"""The report guides what to do next (owner, 2026-10-02): "Do the conditions differ?" opens
with the significant features per condition (the candidates, said as such, with fewer than 3
biological repeats), the best-ranked features are plotted without forcing the readout, and
each answer ends with the analysis that follows from it, as code that runs."""

import html
import re
import warnings
from dataclasses import replace

import pandas as pd
import pytest
from test_rare_fingerprint import _screen, _setup

from morphocoverage import workflows
from morphocoverage.workflows import report
from morphocoverage.workflows._html_cards import significant_summary
from morphocoverage.workflows._html_next import fields_next, fov_counts, populations_next
from morphocoverage.workflows._shared import check_workflow_inputs
from morphocoverage.workflows.figures import plots


@pytest.fixture(scope="module")
def screen():
    objects, fovs = _screen()
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    return objects, fovs, result


class _Result:
    """A report's discovery alone, as significant_summary reads it."""

    def __init__(self, tests):
        self.discovery = type("Discovery", (), {"tests": tests})()
        self.results = {}


def _tests(repeats, lower, higher, confirmed=False):
    return pd.DataFrame(
        {
            "feature": ["area", "dna"],
            "condition": ["drug", "drug"],
            "control_condition": ["ctrl", "ctrl"],
            "repeats": [repeats, repeats],
            "lower": [lower, 0],
            "higher": [higher, repeats],
            "d": [-1.2, 0.8],
            "passes": [True, True],
            "q_cells": [0.001, 0.01],
            "q_paired": [0.01 if confirmed else float("nan")] * 2,
            "confirmed": [confirmed, False],
        }
    )


def _words(markup):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup))


def test_the_summary_names_the_significant_features_per_condition():
    words = _words(significant_summary(_Result(_tests(3, 3, 0, confirmed=True)), str))
    assert "features significant across the 3 biological repeats" in words
    assert "drug : 1 feature" in words
    assert "area: lower in 3 of 3 repeats (adjusted p 0.010; cell-level d -1.20)" in words


def test_without_a_significant_feature_the_trends_are_named_once():
    words = _words(significant_summary(_Result(_tests(3, 3, 0)), str))
    assert "drug : no feature significant across repeats" in words
    assert words.count("area lower") == 1 and words.count("dna higher") == 1


def test_with_two_repeats_the_summary_gives_the_candidates_that_agree():
    words = _words(significant_summary(_Result(_tests(2, 1, 1)), str))
    assert "with 2 biological repeats, no feature can be called significant" in words
    # area goes down in one repeat and up in the other: left out.
    assert "1 candidate that go the same way in both repeats" in words
    assert "dna: higher" in words and "area" not in words.split("drug")[-1]


def test_with_one_repeat_every_candidate_is_given_as_exploratory():
    words = _words(significant_summary(_Result(_tests(1, 1, 0)), str))
    assert "with 1 biological repeat, no feature can be called significant" in words
    assert "2 candidates" in words and "exploratory" in words


def test_the_plots_are_the_best_ranked_the_readout_only_when_it_ranks(screen):
    objects, fovs, result = screen
    ranked = result.discovery.top()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = plots(objects, fovs, result.setup, ranking=result.discovery, pairwise=False)
    assert list(drawn) == list(ranked)[:5]

    # No feature ranks: the readout alone.
    nothing = replace(result.discovery, ranking=result.discovery.ranking.iloc[:0])
    assert not nothing.top()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = plots(objects, fovs, result.setup, ranking=nothing, pairwise=False)
    assert list(alone) == [result.setup.readout]


def _run(markup, namespace):
    """Run the code of a "What to do next" box, as a user would paste it."""
    code = re.search(r"<pre><code>(.*?)</code></pre>", markup, re.DOTALL).group(1)
    exec(html.unescape(code), namespace)  # noqa: S102 - the page's own code, as a user runs it


def test_the_populations_next_step_declares_the_population_as_the_rare_phenotype(
    screen, monkeypatch
):
    objects, fovs, result = screen
    seen = {}

    def fake_report(objects, fovs, setup):
        # The setup is valid for these tables, the population's label among them.
        check_workflow_inputs(objects, fovs, setup)
        seen.update(objects=objects, setup=setup)
        return result

    monkeypatch.setattr(workflows, "report", fake_report)
    markup = populations_next(result, fovs, objects)
    table = result.results["populations"].populations.table
    small = table.sort_values("share").iloc[0]
    words = _words(markup)
    # Validate on the images first, its fields named; then label it with its rule.
    assert f"First validate {small['population']}" in words and "open the images" in words
    assert "label its cells with its rule, picked by" in words
    _run(markup, {"result": result, "objects": objects.copy(), "fovs": fovs})
    setup = seen["setup"]
    assert setup.rare_label == small["population"]
    labelled = seen["objects"][small["population"]]
    # The rule labels about the group's share of the cells (it is checked on held-out cells),
    # and the code declares the share it labels.
    assert labelled.dtype == bool and labelled.mean() == pytest.approx(small["share"], abs=0.03)
    assert setup.rare.object_prevalence == pytest.approx(labelled.mean(), abs=1e-4)
    # Planning its fields is the plan section's (_html_plan), not this step's.
    assert "optimise" not in markup


def test_the_fields_next_step_runs_the_report_without_the_fields(screen, monkeypatch):
    objects, fovs, result = screen
    key = ["condition", "biological_repeat", "technical_repeat", "image_file"]
    flagged = [tuple(row) for row in fovs[key].head(2).itertuples(index=False)]
    seen = {}
    monkeypatch.setattr(
        workflows,
        "report",
        lambda objects, fovs, setup, **kw: seen.update(objects=objects, fovs=fovs, **kw),
    )
    markup = fields_next(flagged, key)
    # The sentence says what to do once the images confirm it; the code is folded.
    assert "If you confirm that some should be excluded" in _words(markup)
    assert "<details><summary>Show the code</summary><pre><code>" in markup
    _run(markup, {"result": result, "objects": objects, "fovs": fovs})
    assert len(seen["fovs"]) == len(fovs) - 2
    gone = set(fovs["image_file"].head(2))
    assert not gone & set(seen["objects"]["image_file"])
    assert len(seen["objects"]) == len(objects) - objects["image_file"].isin(gone).sum()


def test_too_few_fields_per_well_give_no_counts_to_replay():
    fovs = pd.DataFrame({"condition": ["a", "a", "b"], "image_file": ["1", "2", "3"]})
    assert fov_counts(fovs) == []
    many = pd.DataFrame({"condition": ["a"] * 40, "image_file": [str(i) for i in range(40)]})
    counts = fov_counts(many)
    assert len(counts) == 4 and max(counts) <= 26


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_with_one_feature_the_summary_gives_the_conditions_by_verdict():
    from morphocoverage.workflows._html_cards import readout_summary

    objects, fovs = _screen()
    setup = _setup(features=("area",), fingerprint=(), rare=None, rare_label=None, novelty=None)
    result = report(objects, fovs, setup)
    assert significant_summary(result, str) == ""
    words = _words(readout_summary(result, str))
    assert words.strip().startswith("area, compared with siCTRL, condition by condition:")
    assert re.search(r"(Significant across repeats|Same direction|No consistent)", words)
    assert "arrest" in words and "narrow" in words


def test_the_integrity_fields_are_listed_in_the_code_with_the_others():
    from morphocoverage.workflows._html_cards import _field_values

    key = ["condition", "biological_repeat", "technical_repeat", "position_id"]
    name = "condition=si#5, biological_repeat=1, technical_repeat=single, position_id=si#5|1|c7"
    assert _field_values(name, key) == ("si#5", "1", "single", "si#5|1|c7")
    assert _field_values("condition=a", key) is None
    code = html.unescape(fields_next([("a", "1", "w", "f1"), ("a", "1", "w", "f1")], key))
    assert code.count("('a', '1', 'w', 'f1')") == 1 and "# ..." not in code
