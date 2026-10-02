"""The "Do the conditions differ?" card (owner decisions of 2026-10-02): no planning tab and
no plan on the page, no readout block repeating the readings, "Show more" holding the
assumptions only, each figure with its legend whose numbers are its Across repeats rows,
the per-repeat d as a column of that tab, no "limited by" verdict and no q on the page."""

import html
import re
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest
from test_rare_fingerprint import _screen, _setup

from morphocoverage._wording import change, p_value
from morphocoverage.sampling_unit.legend_text import _design
from morphocoverage.workflows import html_report, report
from morphocoverage.workflows._html_tests import _family, across_frame, d_per_repeat


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    objects, fovs = _screen()
    setup = _setup(fingerprint=(), rare=None, rare_label=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
        page = html_report(result, objects, fovs, tmp_path_factory.mktemp("page")).read_text()
    body = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    section = body[body.index('id="conditions"') : body.index('id="enough"')]
    return result, body, section


def _text(markup):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", markup)).split())


def test_no_planning_tab_and_no_plan_on_the_page(card):
    _, body, section = card
    tabs = re.findall(r'role="tab"[^>]*>([^<]+)<', section)
    assert tabs[:2] == ["Across repeats", "Cell-level screen"] and "Planning" not in tabs
    text = _text(body)
    assert "Planning the next experiment" not in text
    # No "limited by" verdict anywhere (owner, 2026-10-02).
    for words in ("limited by", "what limits it", "add repeats, not", "more cells help"):
        assert words not in text


def test_no_readout_block_repeats_the_readings(card):
    _, _, section = card
    text = _text(section)
    assert "condition by condition</summary>" not in section
    assert "Reading, condition by condition" not in text
    assert "Which features separate the conditions?" not in text
    assert "Plotted by default" not in text


def test_show_more_holds_the_assumptions_only(card):
    _, _, section = card
    more = section[section.index("<summary>Show more</summary>") :]
    more = more[: more.index("</details>")]
    text = _text(more)
    assert text.startswith("Show more Assumptions")
    assert "<table" not in more and "Within each repeat:" not in text
    assert "Repeats needed" not in text and "Most repeats searched" not in text


def test_each_figure_has_a_legend_whose_numbers_are_its_across_rows(card):
    result, _, section = card
    frame = across_frame(result)
    blocks = re.split(r'<div class="feature" data-spec=', section)[1:]
    assert blocks
    for block in blocks:
        feature = html.unescape(re.search(r"<h4>([^<]*)</h4>", block).group(1)).rsplit(": ", 1)[0]
        legend = re.search(
            r'<summary>For a figure legend</summary><p class="legend-box">(.*?)</p>', block
        )
        assert legend, feature
        words = _text(legend.group(1))
        rows = frame[frame["feature"] == feature]
        assert len(rows)
        assert f"n = {int(rows['repeats'].max())} independent biological repeats" in words
        for r in rows.itertuples():
            clause = (
                f"{r.condition}: change {change(r.change, r.values)}, "
                f"d {r.d_per_repeat_mean:+.2f}, adjusted p {p_value(r.adjusted_p)} "
                f"(p {p_value(r.p)})"
            )
            assert clause in words


def test_the_across_repeats_tab_gives_the_d_per_repeat_of_every_row(card):
    result, _, section = card
    panel = section[section.index('id="tests-panel-0"') :]
    table = panel[: panel.index("</table>")]
    heads = [_text(h) for h in re.findall(r"<th>(.*?)</th>", table)]
    column = heads.index("d per repeat: mean (range)")
    cells = [
        [_text(c) for c in re.findall(r"<td>(.*?)</td>", row)]
        for row in re.findall(r"<tr>(.*?)</tr>", table)[1:]
    ]
    frame = across_frame(result)
    assert [row[column] for row in cells] == [d_per_repeat(r) for r in frame.itertuples()]
    assert frame["d_per_repeat_mean"].notna().all()
    # The readout's per-repeat d is the one its own contrast computes.
    contrast = result.results["contrast"].repeats.set_index("condition")
    mine = frame[frame["feature"] == result.setup.readout].set_index("condition")
    for condition, row in mine.iterrows():
        assert row["d_per_repeat_mean"] == pytest.approx(contrast.loc[condition, "d_mean"])
        assert row["d_per_repeat_min"] == pytest.approx(contrast.loc[condition, "d_min"])


def test_the_balance_is_said_once_in_the_cell_level_tab(card):
    _, _, section = card
    assert section.count("Balance: ") == 1
    panel = section[section.index('id="tests-panel-1"') :]
    assert panel.index("Balance: ") < panel.index("<table")


def test_unranked_features_are_named_under_the_heatmap(card):
    result, _, section = card
    lines = result.discovery._unranked(8)
    for line in lines:
        assert section.index('class="heatmap"') < section.index(html.escape(line))
        assert section.index(html.escape(line)) < section.index("The best-ranked")


def test_wells_and_the_correction_family_are_said_as_the_legend_says_them():
    table = pd.DataFrame(
        {
            "biological_repeat": ["a", "b", "c"],
            "technical": [3, 3, 3],
            "control_technical": [3, 3, 3],
            "observations": [40, 50, 60],
            "control_observations": [45, 55, 65],
        }
    )
    contrast = SimpleNamespace(
        table=table,
        settings={"repeat_col": "biological_repeat", "observation_unit": "cell"},
    )
    rows = pd.DataFrame({"repeats": [3]})
    assert _design(contrast, rows) == (
        "n = 3 independent biological repeats, 3 wells per condition in each "
        "(40 to 65 cells a condition and repeat)"
    )
    repeats = pd.DataFrame(
        {
            "condition": [f"c{i}" for i in range(12)],
            "experiment": [i // 2 for i in range(12)],
            "p_paired_t": [0.01] * 12,
            "p_paired_t_normalised": [0.01] * 12,
        }
    )
    result = SimpleNamespace(discovery=None, results={"contrast": SimpleNamespace(repeats=repeats)})
    assert _family(result) == "2 conditions at once in each of 6 sub-experiments"


def test_the_heatmap_names_its_axes(card):
    _, _, section = card
    heatmap = section[section.index('class="heatmap"') : section.index("</svg>")]
    assert "condition, against siCTRL" in heatmap and "feature, best-ranked first" in heatmap
