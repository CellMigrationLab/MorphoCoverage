"""The report as one HTML page (workflows.html_report) and its text conversion."""

import base64
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage._style import Status
from morphocoverage.glossary import GLOSSARY, PHRASES, glossary_markdown, slug
from morphocoverage.workflows import Setup, html_report, report, zip_results
from morphocoverage.workflows._html_figures import COLOURS
from morphocoverage.workflows._html_render import Terms, chip, text_html
from morphocoverage.workflows._html_sections import heatmap, hidden, unnumbered


def test_text_becomes_paragraphs_status_lines_and_tables():
    lines = [
        "  Planning the next experiment, with 3 biological repeats as here: the fields a",
        "    side each repeat needs:",
        "    condition  d cells, mean  limited by",
        "    drug       -0.57          sampling (more fields help)",
        "    other      0.20           the repeats",
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


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in ("r1", "r2", "r3"):
        for condition, shift in (("ctrl", 0.0), ("drug", 2.0)):
            for field in range(5):
                for label in range(6):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(20 + shift, 2),
                            "speed": rng.normal(5, 1),
                            "aspect": rng.normal(2, 0.2),
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


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
    assert "Full detail" in page
    assert "None" not in re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    # Every figure is embedded and saved as PNG and PDF.
    assert page.count("data:image/png;base64,") >= 4
    for name in ("area", "pca_raw", "umap_raw", "pca_normalised", "umap_normalised"):
        assert (tmp_path / "figures" / f"{name}.pdf").exists()
    data = json.loads(re.search(r'id="embedding-data">(.*?)</script>', page).group(1))
    assert data["conditions"] == ["ctrl", "drug"] and data["values"] == ["raw", "normalised"]
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
    # Terms in the answers open their meaning, and link to their entry.
    assert 'class="term"' in page and 'id="g-mean-precision-rule"' in page
    with pytest.raises(TypeError, match="Report"):
        html_report("not a report", objects, fovs, tmp_path)


def test_docs_glossary_is_the_rendering_of_the_glossary():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "glossary.md").read_text() == glossary_markdown()
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
    # QC before the result, the result before how far the sampling supports it.
    where = [body.index(f'id="{key}"') for key in ("overview", "inspect", "conditions", "enough")]
    assert where == sorted(where)
    assert not re.search(r"<h2>[^<]*\d\. ", body) and "Foundation" not in body
    assert "What was checked" in body and "Negative control: ctrl" in body
    # Did I image enough? answers in sentences before its numbers.
    enough = body[body.index('id="enough"') :]
    assert enough.index('class="reading"') < enough.index("The numbers")
    # Question 9: the reading, then the heatmap and figures, then the tests.
    conditions = body[body.index('id="conditions"') : body.index('id="enough"')]
    assert conditions.index("Reading, condition by condition") < conditions.index('class="heatmap"')
    assert conditions.index('class="heatmap"') < conditions.index("The tests behind")
    assert 'aria-label="Cohen' in conditions and 'class="selection"' in conditions
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
    assert "q cells 0.001; q paired 0.01" in svg
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
    assert "randomization test on d" in matrix["hover"][0][1]
    assert matrix["text"][1][0].startswith("+") and matrix["text"][0][1]
    assert 'class="pl-pair"' in page and "The static figure (as in the PDF)" in page
    # Its caption says how the interactive figure is coloured.
    assert "a point in its condition&#x27;s colour" in page
