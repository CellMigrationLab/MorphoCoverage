"""The tests behind the reading (owner, 2026-10-02, after an audit): one correction family on
the page, the tests explained before their tables, p and the adjusted p in their own columns,
and no version of the cell-level screen whose control comes from other repeats."""

import html
import re
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from _helpers_rare_fingerprint import _screen, _setup
from _helpers_sampling_unit import _tables as _discovery_tables

from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.reading_text import LIMITED_REPLICATION
from morphocoverage.workflows import html_report, report
from morphocoverage.workflows._families import _CELL_VERSIONS, shared_family
from morphocoverage.workflows._html_across import _across, across_frame
from morphocoverage.workflows._html_render import adjusted_p
from morphocoverage.workflows._html_tests import _cells, _other_versions

# A q left on the page: before a number or comparison, in brackets, before a colon, or
# corrected / uncorrected (docs/report/conditions.md: the adjusted p throughout).
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


def test_the_page_table_and_its_csv_carry_each_repeats_change_and_the_interval(tmp_path):
    objects, fovs = _discovery_tables(n_repeats=3)
    discovery = feature_discovery(
        objects,
        fovs,
        features=["area", "noise"],
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
    )
    result = SimpleNamespace(
        discovery=discovery,
        results={"contrast": None},
        setup=SimpleNamespace(readout="area", unit_word="cell"),
    )
    frame = across_frame(result)
    assert {"repeat_contrasts", "ci_low", "ci_high"} <= set(frame.columns)
    markup, _ = _across(result, tmp_path)
    assert "<th>change in each repeat</th><th>95% t interval</th>" in markup
    assert html.escape(LIMITED_REPLICATION) in markup
    saved = pd.read_csv(tmp_path / "tests_across_repeats.csv")
    assert {"repeat_contrasts", "ci_low", "ci_high"} <= set(saved.columns)


def test_the_cell_level_screen_is_empty_without_tests_or_cells():
    result = SimpleNamespace(
        discovery=None, results={"contrast": None}, setup=SimpleNamespace(readout="area")
    )
    assert _cells(result, None) == ""


@dataclass(frozen=True)
class _Contrast:
    repeats: pd.DataFrame
    cells: pd.DataFrame
    settings: dict


class _Discovery:
    tests = pd.DataFrame(
        {
            "feature": ["area"],
            "condition": ["A"],
            "p_paired": [0.01],
            "q_paired": [0.02],
            "paired_values": ["raw"],
            "p_cells": [0.03],
            "q_cells": [0.04],
            "d": [0.5],
            "normalised": [False],
            "balanced": [False],
        }
    )
    settings: ClassVar[dict] = {}


def test_shared_family_leaves_conditions_the_discovery_did_not_test():
    contrast = _Contrast(
        pd.DataFrame({"condition": ["A", "B"], "p_paired_t": [0.5, 0.6], "q_paired_t": [0.5, 0.6]}),
        pd.DataFrame(
            {
                "condition": ["A", "B"],
                "d_cells": [0.0, 0.0],
                "p_cells": [1.0, 1.0],
                "q_cells": [1.0, 1.0],
            }
        ),
        {"x": 1},
    )
    shared = shared_family(contrast, _Discovery(), "area")
    assert shared.repeats.to_csv(index=False) == (
        "condition,p_paired_t,q_paired_t\nA,0.01,0.02\nB,0.6,0.6\n"
    )
    assert shared.cells.to_csv(index=False) == (
        "condition,d_cells,p_cells,q_cells\nA,0.5,0.03,0.04\nB,0.0,1.0,1.0\n"
    )
    assert shared.settings == {"x": 1, "family_features": 1, "family_share": None}
