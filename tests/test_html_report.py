"""The report as one HTML page (workflows.html_report) and its text conversion."""

from __future__ import annotations

import base64
import json
import os
import re
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import MITOTIC, SEVERAL, _cells, _embedding_only, _several
from _helpers_html import _tables
from _helpers_workflows import _report, _scenario_report

from morphocoverage import FixedFOVStrategy
from morphocoverage._style import Status
from morphocoverage.glossary import GLOSSARY, PHRASES, glossary_markdown, slug
from morphocoverage.workflows import Setup, _html_sections, html_report, report, zip_results
from morphocoverage.workflows._html_data import json_island
from morphocoverage.workflows._html_figures import COLOURS, file_stem
from morphocoverage.workflows._html_render import Terms, adjusted_p, chip, text_html
from morphocoverage.workflows._html_sections import heatmap, hidden, unnumbered, what_checked


def test_text_becomes_paragraphs_status_lines_and_tables():
    lines = [
        "  Planning the next experiment, with 3 biological repeats as here: the fields a",
        "    side each repeat needs:",
        "    condition  d cells, mean  repeats needed",
        "    drug       -0.57          7",
        "    other      0.20           more than 30",
        "  ✅ Data checks: no warning.",
        "  - siCCT8#5 — lower in 3 of 3",
        "  Assumptions: Cells are independent draws.",
    ]
    page = text_html("\n".join(lines))
    # A wrapped sentence is joined again.
    assert "the fields a side each repeat needs:</p>" in page
    assert page.count("<tr>") == 3 and "<th>d cells, mean</th>" in page
    assert '<td class="num">-0.57</td>' in page
    assert 'aria-label="Answered">✅</span> Data checks: no warning.' in page
    assert '<p class="item">siCCT8#5 — lower in 3 of 3</p>' in page
    assert "<strong>Assumptions</strong> Cells are independent draws." in page


def test_text_is_escaped_and_ragged_tables_stay_preformatted():
    page = text_html("a <b> & c\n  x  y\n  1  2  3")
    assert "a &lt;b&gt; &amp; c" in page
    assert "<pre>" in page


def test_chips_carry_a_word_besides_the_colour():
    assert "Caution" in chip(Status.CAUTION) and "⚠️" in chip(Status.CAUTION)
    assert chip(None) == ""


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_page_holds_every_answer_and_its_figures(tmp_path):
    objects, fovs = _tables()
    setup = Setup(
        readout="area",
        features=("area", "speed", "aspect"),
        control={"condition": "ctrl"},
        observation_unit="cell",
    )
    result = report(objects, fovs, setup)
    path = html_report(result, objects, fovs, tmp_path)
    page = path.read_text()
    assert path.name == "report.html"
    for anchor in ("glance", "overview", "enough", "conditions", "limits"):
        assert f'id="{anchor}"' in page
    assert "Show more" in page
    # Embedded images are base64, which can spell anything: left out with the scripts.
    text = re.sub(r"<script.*?</script>|data:[\w/+.-]+;base64,[\w+/=]+", "", page, flags=re.DOTALL)
    assert "None" not in text
    # Every figure is embedded and saved as PNG and PDF.
    assert page.count("data:image/png;base64,") >= 4
    for name in ("area", "pca_raw", "umap_raw", "pca_normalised", "umap_normalised"):
        assert (tmp_path / "figures" / f"{name}.pdf").exists()
    data = json.loads(re.search(r'id="embedding-data">(.*?)</script>', page).group(1))
    assert data["conditions"] == ["ctrl", "drug"] and data["values"] == ["normalised", "raw"]
    pca = data["sets"]["raw"]["PCA"]
    assert len(pca["x"]) == len(objects) and pca["arrows"][0][0] == "area"
    # The control is drawn first, so the condition sits over it.
    assert pca["cond"] == sorted(pca["cond"])
    # The page carries every file for download, so it can be shared alone.
    files = json.loads(re.search(r'id="files-data">(.*?)</script>', page).group(1))
    assert {"report_short.txt", "report_full.txt", "feature_discovery.csv"} <= set(files)
    assert base64.b64decode(files["figures/area.pdf"][1]).startswith(b"%PDF")
    assert 'data-file="figures/area.pdf"' in page and 'href="figures/' not in page
    assert 'id="glossary"' in page and "Benjamini-Hochberg q-value" in page
    # Every figure carries the data it draws, to draw it again outside (owner, 2026-10-02).
    figures = tmp_path / "figures"
    for name in ("area_data", "area_repeats", "heatmap_d", "embedding_raw", "pca_loadings_raw"):
        assert f"figures/{name}.csv" in files and f'data-file="figures/{name}.csv"' in page
    drawn = pd.read_csv(figures / "area_data.csv")
    assert set(drawn["values"]) == {"raw", "normalised"}
    assert (drawn["values"] == "raw").sum() == objects["area"].notna().sum()
    assert {"condition", "biological_repeat", "field", "area"} <= set(drawn.columns)
    points = pd.read_csv(figures / "embedding_raw.csv")
    assert len(points) == len(objects) and {"PC1", "PC2", "condition"} <= set(points.columns)
    # Each point names its row in the observation table and its label_id (#74).
    assert sorted(points["observation"]) == list(objects.index)
    assert (objects.loc[points["observation"], "label_id"].to_numpy() == points["label_id"]).all()
    assert sorted(pca["obs"]) == list(objects.index) and len(pca["lid"]) == len(objects)
    assert pca["lid"][0] == objects.loc[pca["obs"][0], "label_id"] and "' · label_id '" in page
    assert set(pd.read_csv(figures / "heatmap_d.csv")["feature"]) <= set(setup.columns)
    # Terms in the answers open their meaning, and link to their entry.
    assert 'class="term"' in page and 'id="g-mean-precision-rule"' in page
    with pytest.raises(TypeError, match="Report"):
        html_report("not a report", objects, fovs, tmp_path)
    # Figures from other tables than the answers' are refused.
    with pytest.raises(ValueError, match="the tables the report was run on"):
        html_report(result, objects.iloc[1:], fovs, tmp_path)
    with pytest.raises(ValueError, match="the tables the report was run on"):
        html_report(result, objects, fovs.iloc[1:], tmp_path)


def test_docs_glossary_is_the_rendering_of_the_glossary():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "common" / "glossary.md").read_text() == glossary_markdown()
    names = [term.name for term in GLOSSARY]
    assert len(names) == len(set(names))
    assert "Mean-precision rule" in names


def test_zip_holds_every_file_of_the_folder(tmp_path):
    folder = tmp_path / "results"
    (folder / "figures").mkdir(parents=True)
    (folder / "report.html").write_text("page")
    (folder / "figures" / "a.png").write_bytes(b"png")
    archive = zip_results(folder)
    zip_results(folder)  # a second zip replaces the first, never contains it
    with zipfile.ZipFile(archive) as opened:
        assert sorted(opened.namelist()) == ["results/figures/a.png", "results/report.html"]
    # A figure an earlier run left is not this report's: left out with since.
    (folder / "figures" / "old.png").write_bytes(b"png")
    os.utime(folder / "figures" / "old.png", (time.time() - 3600,) * 2)
    with zipfile.ZipFile(zip_results(folder, since=time.time() - 60)) as opened:
        assert sorted(opened.namelist()) == ["results/figures/a.png", "results/report.html"]


def test_terms_link_to_the_glossary_once_per_section_and_never_in_tables():
    terms = Terms()
    page = terms.annotate(
        text_html(
            "  The paired t-test, the cell-level test (balanced); the paired test again.\n"
            "    condition  paired t p\n"
            "    drug       0.02"
        )
    )
    assert page.count('class="term"') == 3
    assert 'href="#g-paired-t-test-on-the-repeat-means">paired t-test<span class="tip"' in page
    # The table keeps its plain header: a tooltip there would be clipped by its scrolling.
    assert "<th>paired t p</th>" in page
    # Every phrase names a glossary entry, and every anchor exists in the glossary.
    names = {term.name for term in GLOSSARY}
    assert set(PHRASES) <= names
    assert len({slug(term) for term in GLOSSARY}) == len(GLOSSARY)


def test_a_hanging_item_is_its_own_paragraph():
    text = (
        "  2 fields to reopen in 1 group: sways a\n"
        "    declared summary (2 fields).\n"
        "    ctrl · r1 · f01: removing it lowers the mean of area\n"
        "      by 13%.\n"
        "    ctrl · r1 · f02: removing it lowers the mean of area\n"
        "      by 11%.\n"
    )
    page = text_html(text)
    assert page.count("<p") == 3
    assert "sways a declared summary (2 fields).</p>" in page
    assert "f02: removing it lowers the mean of area by 11%.</p>" in page


def test_the_page_has_no_question_numbers():
    assert unnumbered("whether it holds across repeats is question 9.") == (
        "whether it holds across repeats is “Do the conditions differ?”."
    )
    assert unnumbered("➖ 1. Can the features see the phenotype? — not assessed") == (
        "➖ Can these features detect what I care about? — not assessed"
    )
    # A planning question keeps its words, without its number.
    assert unnumbered("4. How many fields should I image next time?: not assessed") == (
        "How many fields should I image next time?: not assessed"
    )


def test_a_section_this_design_cannot_answer_is_left_out():
    assert hidden(Status.NOT_ASSESSED, "not applicable: each field holds one cell")
    assert not hidden(Status.NOT_ASSESSED, "not assessed; needs a positive control")
    assert not hidden(Status.CAUTION, "not applicable")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_page_reads_in_order_with_a_heatmap_a_drawer_and_no_grey(tmp_path):
    objects, fovs = _tables()
    setup = Setup(
        readout="area",
        features=("area", "speed", "aspect"),
        control={"condition": "ctrl"},
        observation_unit="cell",
    )
    result = report(objects, fovs, setup)
    page = html_report(result, objects, fovs, tmp_path).read_text()
    body = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    # QC first, then the whole dataset, its populations, the conditions, and how far the
    # sampling supports them (owner, 2026-10-02).
    order = ("overview", "inspect", "dataset", "populations", "conditions", "enough")
    where = [body.index(f'id="{key}"') for key in order]
    assert where == sorted(where)
    assert not re.search(r"<h2>[^<]*\d\. ", body) and "Foundation" not in body
    # The data: one row per check; what was checked under "Show more"; each "?" its help.
    data = body[body.index('id="overview"') : body.index('id="inspect"')]
    assert '<ul class="check-rows">' in data and "Data checks" in data
    assert data.index("Show more") < data.index("What was checked")
    assert "Negative control: ctrl" in data and 'class="help"' in data
    assert '<p class="intro">' not in body
    # Did I image enough? answers in sentences, then one feature at a time, as tabs.
    enough = body[body.index('id="enough"') :]
    assert enough.index('class="reading"') < enough.index("One feature at a time")
    assert enough.count('role="tab"') == 3 and 'id="enough-panel-2"' in enough
    # Question 9: the significant features per condition, the heatmap, the best-ranked
    # features (not the readout first), each with its legend, then the tests; the readings
    # are not repeated on the card (owner, 2026-10-02).
    conditions = body[body.index('id="conditions"') : body.index('id="enough"')]
    assert conditions.index('class="significant"') < conditions.index('class="heatmap"')
    assert conditions.index('class="heatmap"') < conditions.index("The tests behind")
    tests = conditions.index("The tests behind")
    assert conditions.index("For a figure legend") < tests
    assert "Reading, condition by condition" not in conditions
    assert 'aria-label="Cohen' in conditions and 'class="selection"' not in conditions
    assert 'class="selection"' in body[body.index('id="dataset"') : body.index('id="populations"')]
    # The glossary is a drawer with a search.
    assert '<aside class="drawer" id="glossary"' in body and 'type="search"' in body
    data = json.loads(re.search(r'id="embedding-data">(.*?)</script>', page).group(1))
    assert data["colours"][0] == COLOURS[0] and "#999999" not in data["colours"]


def test_the_heatmap_marks_the_gates():
    tests = pd.DataFrame(
        {
            "feature": ["a", "a", "b", "b"],
            "condition": ["x", "y", "x", "y"],
            "d": [2.5, -0.4, np.nan, 0.9],
            "passes": [True, False, False, True],
            "confirmed": [True, False, False, False],
            "q_cells": [0.001, 0.4, np.nan, 0.01],
            "q_paired": [0.01, 0.6, np.nan, 0.2],
        }
    )

    class Discovery:
        pass

    discovery = Discovery()
    discovery.tests = tests
    discovery.ranking = pd.DataFrame({"feature": ["a", "b"]})
    svg = heatmap(discovery, ["x", "y"])
    assert "+2.50 ✓" in svg and "+0.90 ●" in svg and "-0.40<" in svg and "–<" in svg
    assert "cell-level q 0.001; repeat means q 0.010" in svg
    assert heatmap(None, []) == ""


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_interactive_figures_draw_the_static_figures_numbers(tmp_path):
    objects, fovs = _tables()
    setup = Setup(
        readout="area",
        features=("area", "speed", "aspect"),
        control={"condition": "ctrl"},
        observation_unit="cell",
    )
    result = report(objects, fovs, setup)
    page = html_report(result, objects, fovs, tmp_path).read_text()
    # plotly.js is inside the page, so it works offline; and only once.
    assert page.count("plotly.js (cartesian - minified)") == 1
    specs = json.loads(re.search(r'id="feature-data">(.*?)</script>', page).group(1))
    entry = next(s for s in specs if s["feature"] == "area")
    # Declared raw, with a control: the normalised values are the other view.
    assert entry["current"] == "raw" and set(entry["views"]) == {"raw", "normalised"}
    readout, normalised = entry["views"]["raw"], entry["views"]["normalised"]
    assert "relative to control" in normalised["ylabel"]
    control = [c["summary"] for c in normalised["clusters"] if c["condition"] == 0]
    assert control == pytest.approx([1.0], abs=0.15) or all(0.8 < v < 1.2 for v in control)
    assert 'data-value="normalised"' in page
    assert [c["name"] for c in readout["conditions"]] == ["ctrl", "drug"]
    assert readout["conditions"][0]["colour"] == COLOURS[0]
    # One cluster per condition and repeat, every observation drawn (few here).
    assert len(readout["clusters"]) == 6
    assert readout["points"]["shown"] == readout["points"]["total"] == len(objects)
    assert all(c["low"] <= c["summary"] <= c["high"] for c in readout["clusters"])
    # The matrix: d below the diagonal, the corrected p above it.
    matrix = readout["pairwise"]
    assert matrix["names"] == ["ctrl", "drug"]
    assert "randomization test on d" in matrix["b"]["hover"][0][1]
    assert "paired t-test" in matrix["c"]["hover"][1][0]
    assert "<strong>Legend:</strong>" in page and "<p>(C) The test that decides" in page
    assert matrix["b"]["text"][1][0].startswith("+") and matrix["b"]["text"][0][1]
    assert 'class="pl-pair"' in page and "The static figure (as in the PDF)" in page
    # Its caption says how the interactive figure is coloured.
    assert "a point in its condition&#x27;s colour" in page


def test_a_missing_attachment_is_named_and_two_of_one_name_are_both_carried(tmp_path):
    """#366/#339: no requested attachment is dropped or overwritten without a word."""
    from morphocoverage.workflows._html_data import files_card, page_files

    (tmp_path / "figures").mkdir()
    first, second = tmp_path / "a" / "scan.txt", tmp_path / "b" / "scan.txt"
    for path, text in ((first, "first"), (second, "second")):
        path.parent.mkdir()
        path.write_text(text)
    taken = tmp_path / "report_short.txt"
    taken.write_text("attached")
    missing = tmp_path / "answers.json"
    attach = [first, second, first, taken, missing]
    files, _ = page_files(tmp_path, {"report_short.txt": "text"}, attach, time.time())
    carried = {name: base64.b64decode(entry[1]).decode() for name, entry in files.items()}
    assert carried == {
        "report_short.txt": "text",
        "scan.txt": "first",
        "scan_2.txt": "second",
        "report_short_2.txt": "attached",
    }
    card = files_card(files, attach)
    assert card.count("Not found, so not attached") == 1 and "answers.json" in card
    assert "Not found" not in files_card(files, attach[:-1])


def test_the_zip_leaves_out_only_itself(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "source_data.zip").write_bytes(b"zip")
    with zipfile.ZipFile(zip_results(folder)) as opened:
        assert opened.namelist() == ["results/source_data.zip"]


def test_nothing_compared_is_said_never_no_consistent_difference() -> None:
    # F12 note (e02): the chip read "⚠️ No consistent difference" when nothing was compared.
    result = _scenario_report("e02_control_in_other_repeats")
    status = result.document.statuses["conditions"]
    headline = result.document.headlines["conditions"]
    assert status is Status.NOT_ASSESSED
    assert headline.startswith("nothing compared: no biological repeat holds both a condition")
    assert "Nothing compared" in chip(status, "conditions", headline)
    assert "No consistent difference" not in chip(status, "conditions", headline)


CONFIG = re.compile(r"config [0-9a-f]{16}")


@pytest.mark.slow
def test_the_page_and_the_short_report_carry_no_provenance(tmp_path: Path) -> None:
    loaded, result = _report("a06_expl_none_image_none_tol")
    assert CONFIG.search(result.summary(detail="full"))
    assert not CONFIG.search(result.summary())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(
            result, loaded.objects, loaded.fovs, tmp_path, feature_tables=loaded.feature_tables
        )
    html = page.read_text()
    assert not CONFIG.search(html)
    assert "_upper95" not in html


def test_the_maps_have_raw_and_normalised_buttons_and_a_method_menu(tmp_path):
    import warnings

    from _helpers_embedding import _embedding_only

    from morphocoverage.workflows import html_report

    objects, fovs, table, result = _embedding_only(normalize="control_mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    text = page.read_text()
    assert 'data-value="@corrected"' in text and 'data-value="raw"' in text
    assert "Divided by the control mean" in text or "Minus the repeat median" in text
    assert 'aria-label="Values"' in text


def _visible(page: str) -> str:
    text = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


@pytest.mark.parametrize(
    "fields",
    [{"normalize": "control_mean"}, {"rare": MITOTIC, "rare_label": "mitotic"}],
    ids=["normalize", "rare_label"],
)
def test_b1_b2_an_embedding_only_page_is_built_with_normalize_or_a_rare_label(fields, tmp_path):
    objects, fovs, table, result = _embedding_only(**fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    visible = _visible(page.read_text())
    assert "The whole tables" in visible and "Plan the next experiment" in visible
    if "normalize" in fields:  # the populations counted the centred values: said on them
        assert (tmp_path / "figures" / "separation.csv").exists()


def test_b7_the_csvs_name_each_features_table(tmp_path):
    objects, fovs, tables = _several()
    setup = Setup(
        readout="area",
        features=("area", "fib_len", "fib_w"),
        control=SEVERAL,
        master_control="cA",
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables=tables)
        html_report(result, objects, fovs, tmp_path, feature_tables=tables)
    figures = tmp_path / "figures"
    across = pd.read_csv(figures / "tests_across_experiments.csv")
    assert list(across.columns[2:4]) == ["feature", "table"]
    assert set(across.loc[across["feature"] == "fib_len", "table"]) == {"fiber"}
    pairs = sorted(figures.glob("*_pairs.csv"))
    assert pairs and all(pd.read_csv(p).columns[0] == "table" for p in pairs)
    loadings = pd.read_csv(next(figures.glob("pca_loadings_raw_Combined.csv")))
    assert dict(zip(loadings["feature"], loadings["table"], strict=True))["d0"] == "DINO"
    assert dict(zip(loadings["feature"], loadings["table"], strict=True))["fib_w"] == "fiber"


def test_the_page_says_the_repeats_are_scored_on_raw_values_and_the_one_well_line(tmp_path):
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell", normalize="control_mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    visible = _visible(page.read_text())
    assert "Within a condition, the same with the repeat, always on the raw values" in visible
    assert "The wells are not given, so a difference between the wells" in visible


def test_340_the_rules_line_names_an_other_rule_and_raises_on_a_broken_label(monkeypatch):
    class Counted:
        name = "my own rule"

    setup = Setup(readout="x", observation_unit="cell", rules=(FixedFOVStrategy(4), Counted()))
    assert "Stopping rules checked: fixed 4 fields and my own rule." in what_checked(setup)

    def broken(*args, **kwargs):
        raise ValueError("a broken label")

    monkeypatch.setattr(_html_sections, "rule_label", broken)
    with pytest.raises(ValueError, match="a broken label"):
        what_checked(setup)


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
    with pytest.raises(ValueError, match="not the tables the report was run on"):
        html_report(result, objects.iloc[:-1], fovs, tmp_path)
    # The source check has no opt-out (compatibility sweep MC-2).
    with pytest.raises(TypeError, match="check_source"):
        html_report(result, objects, fovs, tmp_path, check_source=False)


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
