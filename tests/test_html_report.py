"""The report as one HTML page (workflows.html_report) and its text conversion."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from morphocoverage._style import Status
from morphocoverage.workflows import Setup, html_report, report
from morphocoverage.workflows._html_render import chip, text_html


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
    assert "Full detail" in page and "None" not in re.sub(r"<script.*?</script>", "", page)
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
    with pytest.raises(TypeError, match="Report"):
        html_report("not a report", objects, fovs, tmp_path)
