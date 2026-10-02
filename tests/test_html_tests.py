"""The tests behind the reading (owner, 2026-10-02, after an audit): one correction family on
the page, the tests explained before their tables, p and the adjusted p in their own columns,
and no version of the cell-level screen whose control comes from other repeats."""

import html
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from test_rare_fingerprint import _screen, _setup

from morphocoverage.workflows import html_report, report
from morphocoverage.workflows._families import _CELL_VERSIONS
from morphocoverage.workflows._html_render import adjusted_p
from morphocoverage.workflows._html_tests import _other_versions

# A q left on the page: before a number or comparison, in brackets, before a colon, or
# corrected / uncorrected (docs/workflows.md: the adjusted p throughout).
_Q = re.compile(r"(?<![\w-])q(?: ?[<≤=:]| \d| (?:un)?corrected\b|\))|\(q\)")


@pytest.fixture(scope="module")
def screen():
    objects, fovs = _screen()
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    return objects, fovs, result


def test_the_readout_takes_the_correction_of_every_feature(screen):
    _, _, result = screen
    contrast, tests = result.results["contrast"], result.discovery.tests
    mine = tests[tests["feature"] == result.setup.readout].set_index("condition")
    for row in contrast.repeats.itertuples():
        assert row.q_paired_t_normalised == pytest.approx(mine.loc[row.condition, "q_paired"])
        assert row.p_paired_t == pytest.approx(mine.loc[row.condition, "p_paired_raw"])
    cells = contrast.cells.set_index("condition")
    for condition, test in mine.iterrows():
        version = _CELL_VERSIONS[(bool(test["normalised"]), bool(test["balanced"]))]
        assert cells.loc[condition, f"q_{version}"] == pytest.approx(test["q_cells"])
        assert cells.loc[condition, f"d_{version}"] == pytest.approx(test["d"])
    assert contrast.settings["family_features"] == len(set(tests["feature"]))
    short = " ".join(result.document.short["conditions"].split())
    assert "within the condition's sub-experiment, over its 3 features and their" in short


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_tests_are_explained_before_their_tables(screen, tmp_path):
    objects, fovs, result = screen
    page = html_report(result, objects, fovs, tmp_path).read_text()
    body = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    section = body[body.index('id="conditions"') : body.index('id="enough"')]
    tests = section[section.index("The tests behind the reading") :]
    assert tests.index("The tests, in short") < tests.index("<table")
    assert "Paired t-test on the biological repeat means: decides" in tests
    assert "Cell-level screen: exploratory" in tests
    assert "<th>p</th><th>adjusted p</th>" in tests
    # The q-value is called the adjusted p in the page's text, its tables and sentences:
    # "q 0.26", "q ≤", "(q)", "q:" and "q corrected" alike.
    text = html.unescape(re.sub(r"<[^>]+>", " ", section))
    assert not _Q.search(text), _Q.search(text)
    for name in ("tests_across_repeats", "tests_cell_level"):
        frame = pd.read_csv(tmp_path / "figures" / f"{name}.csv")
        assert {"p", "adjusted_p"} <= set(frame.columns)


def test_the_screen_as_provided_is_not_shown_with_another_repeats_control():
    row = {
        "condition": "drug",
        "other_control_repeats": 3,
        **{
            f"{k}_{s}": v
            for s in ("cells", "cells_normalised")
            for k, v in (("d", 0.5), ("p", 0.01), ("q", 0.02))
        },
        "d_balanced": -0.2,
        "p_balanced": 0.3,
        "q_balanced": 0.3,
        "p_balanced_normalised": np.nan,
    }
    frame = pd.DataFrame(
        {"feature": ["x"], "condition": ["drug"], "version": ["balanced, normalised"]}
    )
    markup = _other_versions({"drug": pd.Series(row)}, frame, "x")
    assert "Not shown as provided for drug" in markup
    assert "as provided, raw" not in markup and "balanced, raw" in markup


def test_the_page_calls_the_q_value_the_adjusted_p():
    markup = "<p>paired t-test p 0.21, q 0.26; q ≤ 0.05; raw: d, p (q); the q-value</p>"
    assert adjusted_p(markup) == (
        "<p>paired t-test p 0.21, adjusted p 0.26; adjusted p ≤ 0.05; raw: d, p (adjusted p); "
        "the q-value</p>"
    )
    assert adjusted_p('<a data-q="q 1">x</a>') == '<a data-q="q 1">x</a>'
    # The forms an audit found left over (2026-10-02).
    fixed = adjusted_p("<p>q: the version the reading uses; p uncorrected, q corrected</p>")
    assert fixed == (
        "<p>adjusted p: the version the reading uses; p uncorrected, the adjusted p corrected</p>"
    )
    assert not _Q.search(fixed) and _Q.search("p uncorrected, q corrected")
