"""Did I image enough? for every feature alike (owner, 2026-10-01): the report is neutral, so
question 2 asks each feature the same questions, a summary table first and then one feature
at a time, and says when it is enough as well as when it is not."""

import html
import re
import warnings

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
from morphocoverage.workflows._html_enough import enough_body, summary_lines
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
    assert "aspect needs the most (about 10 fields a side)" in headline


def test_every_feature_enough_is_said_and_marked_enough(result):
    features = result.results["enough_features"]
    area = features.table[features.table["feature"] == "area"].reset_index(drop=True)
    only = FeatureEnough(area, {"area": features.contrasts["area"]}, features.precision)
    status, headline = features_status(only, "group")
    assert status is Status.OK
    assert "of 1 feature: enough for 1 (area)" in headline and "needs the most" not in headline
    lines = summary_lines(only, "group", "fields")
    assert lines[0].startswith("Enough fields to compare with the control, within single")
    assert "Enough for area (every comparison shown)" in lines[0]


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
    assert "own mean" not in text and "still needed" in text


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
    short = "  Each feature alike, 24 features, ordered as in x (significant across\n"
    short += "    repeats first):\n"
    short += features_table(many, "group", indent=4)
    body = enough_body(many, short, "group", "fields", lambda m: m, lambda m: m)
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
    own = "first); own mean: the groups whose mean meets the mean-precision rule, per cell"
    assert own in short
    assert "own mean (" not in short
