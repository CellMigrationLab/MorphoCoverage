"""Did I image enough? for every feature alike (owner, 2026-10-01): the report is neutral, so
question 2 asks each feature the same questions, a summary table first and then one feature
at a time, and says when it is enough as well as when it is not."""

import html
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_html_report import _tables

from morphocoverage._style import Status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.workflows import _enough_features, _html_tabs, html_report, report
from morphocoverage.workflows._enough_features import (
    ENOUGH,
    NOT_YET,
    FeatureEnough,
    features_status,
    features_table,
)
from morphocoverage.workflows._html_enough import answer_lines, enough_body
from morphocoverage.workflows._html_tabs import TABS_AT_MOST, tabs_html
from morphocoverage.workflows.setup import Setup

SETUP = Setup(
    readout="area",
    features=("area", "speed", "aspect"),
    control={"condition": "ctrl"},
    observation_unit="cell",
)


@pytest.fixture(scope="module")
def result():
    objects, fovs = _tables()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(objects, fovs, SETUP)


def test_the_light_contrast_leaves_out_the_cell_level_test_only():
    objects, fovs = _tables()
    common = {"feature": "area", "control": {"condition": "ctrl"}, "observation_unit": "cell"}
    full = condition_contrast(objects, fovs, **common)
    light = condition_contrast(objects, fovs, cell_test=False, **common)
    assert light.cells.empty and not full.cells.empty
    columns = ["d_low", "d_high", "resolved", "too_small"]
    pd.testing.assert_frame_equal(light.table[columns], full.table[columns])


def test_each_feature_is_asked_and_the_answer_says_enough_too(result):
    features = result.results["enough_features"]
    rows = features.table.set_index("feature")
    assert list(rows.index) == ["area", "speed", "aspect"]
    assert rows.loc["area", "verdict"] == ENOUGH and rows.loc["area", "shown"] == 3
    assert (rows.loc[["speed", "aspect"], "verdict"] == NOT_YET).all()
    # Each feature's own mean, by the readout's mean-precision rule run on it.
    assert set(features.precision) == {"area", "speed", "aspect"}
    status = result.document.statuses["enough"]
    headline = result.document.headlines["enough"]
    assert status is Status.CAUTION
    assert headline.startswith(
        "to compare with the control, within single repeats, of 3 features: enough for 1 "
        "(area); not yet for 2 (speed and aspect)"
    )
    # aspect differs in no test: no projection of the fields it would need (audit 10, C25).
    assert "needs the most" not in headline
    assert not features.table.set_index("feature").loc["aspect", "flagged"]


def test_every_feature_enough_is_said_and_marked_enough(result):
    features = result.results["enough_features"]
    area = features.table[features.table["feature"] == "area"].reset_index(drop=True)
    only = FeatureEnough(area, {"area": features.contrasts["area"]}, features.precision)
    status, headline = features_status(only, "group")
    assert status is Status.OK
    assert "of 1 feature: enough for 1 (area)" in headline and "needs the most" not in headline
    lines = answer_lines(only, "fields", 3)
    assert lines == [
        (
            "Within each biological repeat, the fields imaged show the difference from the "
            "control in every comparison, for area."
        )
    ]


def test_the_short_form_has_the_table_and_the_rules_the_full_form_every_feature(result):
    short = result.document.short["enough"]
    assert short.index("Each feature alike, 3 features") < short.index(
        "The stopping rules, as optimise and live run them (on the readout):"
    )
    assert "enough: shown in 3 of 3" in short and "not yet: shown in 0 of 3" in short
    # The readout's own resolution is one row of the table, not repeated in the short form.
    assert "does the difference from the control show? area" not in short
    full = result.document.sections["enough"]
    for feature in ("area", "speed", "aspect"):
        assert full.count(f"does the difference from the control show? {feature},") == 1


def test_the_table_drops_the_precision_column_with_no_rule(result):
    features = result.results["enough_features"]
    table = features.table.assign(precision_rule=None, precision_met=float("nan"))
    text = features_table(FeatureEnough(table, features.contrasts, {}), "group")
    assert "meets the rule" not in text and "to show the rest" in text


def test_the_columns_say_what_each_is(result):
    """ "compared with the control" beside "own mean" read as contradictory ("enough: shown in
    6 of 6" beside "0 of 9 wells", ChatGPT audit B6): each header names its goal."""
    text = features_table(result.results["enough_features"], "group")
    head = text.splitlines()[0]
    for words in ("difference from the control shown", "to show the rest"):
        assert words in head
    assert "groups whose mean meets the rule" in head
    assert "compared with the control" not in head and "own mean" not in head


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_page_shows_a_summary_then_one_feature_at_a_time(result, tmp_path):
    objects, fovs = _tables()
    page = html_report(result, objects, fovs, tmp_path).read_text()
    enough = page[page.index('id="enough"') :]
    assert enough.index('class="reading"') < enough.index("One feature at a time")
    tabs = re.findall(r'role="tab" id="enough-tab-\d+"[^>]*>([^<]+)<span class="mark">(.)', enough)
    assert tabs == [("area", "✅"), ("speed", "⚠"), ("aspect", "⚠")]
    panel = enough[enough.index('id="enough-panel-0"') : enough.index('id="enough-panel-1"')]
    words = html.unescape(re.sub(r"<span class=\"tip\".*?</span>|<[^>]+>", "", panel))
    assert "Enough fields to compare with the control? Yes" in words
    assert "Enough to pin down each group's own mean of area?" in words
    assert "was met in 6 of 6 groups: enough." in words


def test_up_to_twelve_tabs_then_a_list_to_search():
    items = [(f"f{i}", "✅", f"<p>{i}</p>") for i in range(TABS_AT_MOST)]
    assert tabs_html("t", items, "Features").count('role="tab"') == TABS_AT_MOST
    many = tabs_html("t", [*items, ("last", "", "<p>last</p>")], "Features")
    assert 'role="tab"' not in many and 'class="feature-tabs picker"' in many
    assert 'placeholder="Find one of 13 features"' in many
    assert many.count("<option") == 13 and '<option value="0" selected>✅ f0</option>' in many
    assert many.count('class="tab-panel"') == 13 and many.count(" hidden>") == 12
    assert '<h4 class="panel-name">✅ f0</h4>' not in many and '"panel-name">f0 ✅</h4>' in many


def test_a_long_summary_table_shows_its_first_rows_and_a_button(result):
    features = result.results["enough_features"]
    rows = pd.concat([features.table] * 8, ignore_index=True)
    rows["feature"] = [f"f{i:02d}" for i in range(len(rows))]
    contrasts = {f: features.contrasts["area"] for f in rows["feature"]}
    many = FeatureEnough(rows, contrasts, {})
    page = SimpleNamespace(setup=SETUP, results={**result.results, "enough_features": many})
    body = enough_body(page, lambda m: m)
    assert '<div class="long-table">' in body and "Show all 24 features</button>" in body
    assert 'class="feature-tabs picker"' in body and "Each feature alike" not in body


def test_the_tests_behind_the_reading_are_cut_by_kind():
    text = (
        "  Analysis: per field.\n  Across the repeats: paired t.\n    a row\n"
        "  Cell-level screen: 3 candidates.\n  ⚠️ one caution.\n"
        "  Assumptions: fields independent."
    )
    head, blocks, tail = _html_tabs.test_blocks(text)
    assert head == ["  Analysis: per field."]
    assert list(blocks) == ["Across repeats", "Cell-level screen"]
    assert "a row" in blocks["Across repeats"][0]
    assert tail == ["  ⚠️ one caution.", "  Assumptions: fields independent."]


def test_the_precision_split_over_workers_is_the_one_replay():
    objects, fovs = _tables()
    rng = np.random.default_rng(1)
    for i in range(16):
        objects[f"f{i:02d}"] = objects["area"] + rng.normal(0, 1 + i, len(objects))
    features = [f"f{i:02d}" for i in range(16)]
    setup = Setup(
        readout="f00",
        features=tuple(features),
        control={"condition": "ctrl"},
        observation_unit="cell",
        n_jobs=2,
    )
    split = _enough_features._precision(objects, fovs, setup, features)
    assert split == _enough_features._replay(objects, fovs, setup, features, 1)
    assert split and list(split) == [f for f in features if f in split]


def test_a_repeat_the_folders_did_not_have_is_not_named():
    from morphocoverage.retrospective.report_resolved import _name

    row = {"condition": "drug", "control_condition": "ctrl"}
    assert _name({**row, "biological_repeat": "single"}) == "drug vs ctrl"
    assert _name({**row, "biological_repeat": "r1"}) == "drug vs ctrl, repeat r1"
    from morphocoverage.sampling_unit.contrast_text import _name as contrast_name

    single = {**row, "biological_repeat": "single"}
    assert contrast_name(single, "biological_repeat") == "drug vs ctrl"


def test_the_rule_is_named_once_above_the_table(result):
    short = " ".join(result.document.short["enough"].split())
    own = "first); the rule of the last column: the mean-precision rule, per cell"
    assert own in short
    assert "own mean (" not in short


def _card(page: str) -> str:
    """The markup of "Did I image enough?"."""
    start = page.index('<section class="card" id="enough"')
    return page[start : page.index("</section>", start)]


def _words(markup: str) -> str:
    """The text a reader sees: no tags, no glossary tooltips, no "?" help."""
    text = re.sub(r'<span class="tip".*?</span>|<[^>]+>', "", markup, flags=re.DOTALL)
    return " ".join(html.unescape(text).split())


def _help(card: str) -> str:
    return html.unescape(re.search(r'data-help="([^"]*)"', card).group(1))


@pytest.fixture(scope="module")
def card(result, tmp_path_factory):
    objects, fovs = _tables()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = html_report(result, objects, fovs, tmp_path_factory.mktemp("enough"))
    return _card(path.read_text())


def test_the_top_is_one_sentence_then_a_compact_table(card):
    # The owner's layout (2026-10-02): one answer sentence, then a table, one row per
    # feature: the comparisons shown and what would show the rest; no own-mean column.
    top = card[: card.index("One feature at a time")]
    readings = re.findall(r'<div class="reading">(.*?)</div>', top, flags=re.DOTALL)
    assert len(readings) == 1 and readings[0].count("<p>") == 1
    assert _words(readings[0]) == (
        "Within each biological repeat, the fields imaged show the difference from the control "
        "in every comparison for area, and in none for the other 2 features."
    )
    assert re.findall(r"<th>([^<]+)</th>", top) == [
        "feature",
        "shown, of 3 comparisons",
        "to show the rest",
    ]
    words = _words(top)
    assert "own mean" not in words and "still needed" not in words
    cells = [html.unescape(c) for c in re.findall(r"<td[^>]*>([^<]*)</td>", top)]
    assert cells[:3] == ["area", "3", "–"]
    # aspect differs in no test: no projection of its fields (audit 10, C25), and the comparison
    # not projected is counted so the row adds up (audit A4).
    assert cells[6:] == [
        "aspect",
        "0",
        "1 not projected (no candidate difference); 2 too small even with 4 times the fields",
    ]


def test_what_every_tab_would_repeat_is_said_once_in_the_help(card):
    words, help_ = _words(card), _help(card)
    for repeated in (
        "This is per repeat",
        "answered in “Do the conditions differ?”",
        "A different goal from the comparison",
        "is the lever",
        "every one in the full report",
        "a condition in one repeat; shown",
    ):
        assert repeated not in words
    assert help_.count("This is per repeat: whether the conditions differ across") == 1
    assert help_.count("a different goal from the comparison") == 1
    assert help_.count("more fields per group is the lever") == 1
    # Each tab keeps its comparisons, its own mean and its table.
    panel = card[card.index('id="enough-panel-1"') : card.index('id="enough-panel-2"')]
    assert "Enough to pin down each group's own mean of speed?" in _words(panel)
    assert "<table" in panel
    # The assumptions of these answers, once, at the end of the card.
    assert words.count("Assumptions") == 1
    assert words.index("Assumptions") > words.index("Within each repeat, does the difference")
    assert "treats cells as independent draws" in words


def test_the_card_has_no_stopping_rules_and_no_code(card):
    # The stopping rules, the repeats needed and optimise belong to the plan of the next
    # experiment (owner, 2026-10-02).
    words = _words(card)
    assert "stopping rules" not in words.lower() and "as optimise and live run them" not in words
    assert "What to do next" not in words and "optimise(" not in card and "<pre>" not in card
    assert "biological repeats in all for" not in words
    assert "Whether unseen cell states remain" not in words and "Show more" not in words


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_with_one_repeat_it_says_first_and_once_that_none_can_be_significant(tmp_path):
    objects, fovs = _tables()
    objects = objects[objects["biological_repeat"] == "r1"]
    fovs = fovs[fovs["biological_repeat"] == "r1"]
    result = report(objects, fovs, SETUP)
    card = _card(html_report(result, objects, fovs, tmp_path).read_text())
    words = _words(card)
    said = "With 1 biological repeat, no difference can be called significant"
    assert words.count(said) == 1
    first = re.search(r'<div class="reading"><p>(.*?)</p>', card, flags=re.DOTALL).group(1)
    assert _words(first).startswith(said)
    assert "Within the biological repeat, the fields imaged" in words
    # Nothing across repeats to point to.
    assert "answered in" not in _help(card) and "This is per repeat" not in _help(card)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_with_one_feature_the_card_has_the_same_layout_and_one_table(tmp_path):
    objects, fovs = _tables()
    setup = Setup(readout="area", control={"condition": "ctrl"}, observation_unit="cell")
    result = report(objects, fovs, setup)
    card = _card(html_report(result, objects, fovs, tmp_path).read_text())
    words = _words(card)
    assert words.index("Within each biological repeat, the fields imaged show") < words.index(
        "Within each repeat, does the difference from the control show? area"
    )
    # Every comparison once, then the precision of each group's own mean, the assumptions.
    assert card.count("<table") == 1 and "Show more" not in words
    assert "Enough to pin down each group's own mean of area?" in words
    assert words.count("Assumptions") == 1 and "stopping rules" not in words.lower()
    assert "What to do next" not in words


def test_comparisons_too_small_are_counted_over_the_features_they_sum(result):
    """Stress test C: "solidity, area, perimeter and 1 more: too small to show in 10
    comparisons" summed over 4 features while each has 6: the headline says over how many."""
    features = result.results["enough_features"]
    table = features.table.assign(too_small=[2, 3] + [0] * (len(features.table) - 2))
    _, text = features_status(FeatureEnough(table, features.contrasts, {}), "group")
    names = list(table["feature"][:2])
    assert f"5 comparisons over {names[0]} and {names[1]} too small to show" in text
    one = features.table.assign(too_small=[2] + [0] * (len(features.table) - 1))
    _, text = features_status(FeatureEnough(one, features.contrasts, {}), "group")
    assert f"2 comparisons of {names[0]} too small to show even" in text
