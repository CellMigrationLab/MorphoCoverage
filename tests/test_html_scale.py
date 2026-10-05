"""The HTML report at scale (owner decisions of 2026-10-05): many conditions, many features
and long names. Small fixtures: 3 features and 16 conditions, a few fields each."""

import base64
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage._palette import COLOURS, GREY, condition_palette
from morphocoverage._wording import shortened
from morphocoverage.embedding import _pca, _standardised, embed
from morphocoverage.plots import plots_of_data
from morphocoverage.plots_embedding import embedding_figures
from morphocoverage.plots_pairwise import pairwise_contrasts
from morphocoverage.statistics import benjamini_hochberg
from morphocoverage.workflows import Setup, html_report, plots, report
from morphocoverage.workflows._html_data import LINKED, files_card, inline_images, page_files
from morphocoverage.workflows._html_enough import _needs
from morphocoverage.workflows._html_plotly_script import PLOTLY_SCRIPT
from morphocoverage.workflows._html_sections import heatmap
from morphocoverage.workflows.figures import colour_note, highlighted
from morphocoverage.workflows.guided import rough_runtime

CONDITIONS = ["ctrl"] + [f"c{i:02d}" for i in range(1, 16)]


def _tables(conditions=CONDITIONS, features=("area", "speed", "aspect"), fields=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in ("2026_03_02", "2026_03_09", "2026_03_16"):
        batch = rng.normal(0, 0.3)
        for condition in conditions:
            shift = 4.0 if condition in ("c03", "c07") else 0.0
            for field in range(fields):
                for label in range(6):
                    row = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "image_file": f"{condition}_{repeat}_{field}.tif",
                        "label_id": label + 1,
                    }
                    for k, feature in enumerate(features):
                        row[feature] = rng.normal(20 + (shift if k == 0 else 0) + batch, 2)
                    rows.append(row)
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


def _setup(features=("area", "speed", "aspect")):
    return Setup(
        readout=features[0],
        features=tuple(features),
        control={"condition": "ctrl"},
        observation_unit="cell",
    )


def test_up_to_12_conditions_each_keep_their_own_colour():
    names = [f"c{i}" for i in range(12)]
    assert list(condition_palette(names, "c0").values()) == list(COLOURS)


def test_past_12_the_control_and_the_hits_are_coloured_the_rest_grey():
    names = ["ctrl", *[f"c{i:02d}" for i in range(1, 40)]]
    hits = [f"c{i:02d}" for i in range(39, 0, -1)]  # more hits than colours
    palette = condition_palette(names, "ctrl", hits)
    assert palette["ctrl"] == COLOURS[0]
    coloured = [c for c, colour in palette.items() if colour != GREY]
    assert coloured[0] == "ctrl" and len(coloured) == len(COLOURS)
    # The best hits keep a colour, each its own, and none the control's blue.
    assert [palette[h] for h in hits[:11]] == list(COLOURS[1:])
    assert list(palette.values()).count(COLOURS[0]) == 1
    # Without the control drawn, its blue stays unused.
    without = condition_palette(names[1:], "ctrl", hits)
    assert COLOURS[0] not in without.values()
    note = colour_note(palette, "significant across repeats")
    assert "the 11 conditions significant across repeats are in colour" in note
    assert "the other 28 are grey" in note
    assert colour_note(condition_palette(names[:5], "ctrl"), "x") == ""


def test_each_part_of_a_split_figure_colours_its_own_conditions():
    objects, fovs = _tables()
    figures = plots_of_data(
        objects, fovs, features=["area"], control={"condition": "ctrl"}, n_boot=50
    )["area"]
    assert len(figures) == 2
    for figure in figures:
        conditions = list(dict.fromkeys(figure.cluster_summaries["condition"]))
        assert conditions[0] == "ctrl" and figure.colours["ctrl"] == COLOURS[0]
        assert list(figure.colours.values()) == list(COLOURS[: len(conditions)])
        assert GREY not in figure.colours.values()


def test_a_figure_past_12_conditions_greys_all_but_the_control_and_the_highlighted():
    objects, fovs = _tables()
    (figure,) = plots_of_data(
        objects,
        fovs,
        features=["area"],
        control={"condition": "ctrl"},
        max_conditions=16,
        highlight=["c07", "c03"],
        n_boot=50,
    )["area"]
    assert figure.colours["c07"] == COLOURS[1] and figure.colours["c03"] == COLOURS[2]
    assert sum(colour == GREY for colour in figure.colours.values()) == 13
    assert "the other 13 in grey" in figure.caption


def test_repeat_names_wider_than_their_slot_are_slanted():
    objects, fovs = _tables(conditions=CONDITIONS[:12], features=("area",), fields=1)
    (figure,) = plots_of_data(
        objects, fovs, features=["area"], control={"condition": "ctrl"}, n_boot=20
    )["area"]
    names = [t for t in figure.axes[0].texts if t.get_text().startswith("2026_")]
    assert names and all(t.get_rotation() == 40 for t in names)


def test_a_split_figures_pairs_are_adjusted_over_its_own_pairs():
    objects, fovs = _tables(features=("area",), fields=1)
    setup = _setup(("area",))
    figures = plots(objects, fovs, setup)["area"]
    assert len(figures) == 2
    for figure in figures:
        conditions = list(dict.fromkeys(figure.cluster_summaries["condition"]))
        table = figure.pairwise
        n = len(conditions)
        assert len(table) == n * (n - 1) // 2
        assert set(table["first"]) | set(table["second"]) == set(conditions)
        assert np.allclose(table["q"], benjamini_hochberg(table["p"].to_numpy(float)))
    # The p-values are those of the pairs tested over every condition: only q changes.
    every = pairwise_contrasts(
        objects, fovs, feature="area", control="ctrl", random_state=setup.random_state
    )
    second = figures[1].pairwise.merge(every, on=["first", "second"], suffixes=("", "_all"))
    assert len(second) == len(figures[1].pairwise)
    assert np.allclose(second["p"], second["p_all"], equal_nan=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_maps_of_many_conditions_grey_all_but_the_control_and_the_hits():
    objects, fovs = _tables()
    embedding = embed(objects, fovs, features=["area", "speed", "aspect"], umap=False)
    figures = embedding_figures(embedding, control={"condition": "ctrl"}, highlight=["c03"])
    labels = [t.get_text() for t in figures["pca"].axes[0].get_legend().get_texts()]
    assert labels[0].startswith("the other 14 conditions")
    assert any(label.startswith("ctrl") for label in labels)
    assert any(label.startswith("c03") for label in labels) and len(labels) == 3


def test_the_embedding_is_built_at_once_and_holds_every_feature():
    features = [f"f{i:03d}" for i in range(120)]
    objects, fovs = _tables(conditions=CONDITIONS[:3], features=features, fields=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        embedding = embed(objects, fovs, features=features, umap=False)
    points = embedding.points
    z = [c for c in points.columns if c.startswith("z:")]
    assert z == [f"z:{f}" for f in features]
    assert list(points.columns[:2]) == ["condition", "biological_repeat"]
    assert list(points.columns[-4:]) == ["pc1", "pc2", "umap1", "umap2"]
    # The same values as built one column at a time before.
    z_values, _ = _standardised(objects[features].to_numpy(float))
    assert np.array_equal(points[z].to_numpy(), z_values)
    scores, _, _ = _pca(z_values)
    assert np.array_equal(points["pc1"].to_numpy(), scores[:, 0])
    assert points["umap1"].isna().all() and len(points) == len(objects)


def test_highlighted_takes_the_significant_conditions_else_the_candidates():
    class Discovery:
        pass

    from morphocoverage.sampling_unit import FeatureDiscovery

    tests = pd.DataFrame(
        {
            "condition": ["a", "b", "c", "a"],
            "d": [0.5, -2.0, 1.0, 3.0],
            "passes": [True, True, False, True],
            "confirmed": [False, True, False, False],
        }
    )
    discovery = FeatureDiscovery(tests, pd.DataFrame(), {})
    assert highlighted(discovery) == (["b"], "significant across repeats")
    none = FeatureDiscovery(tests.assign(confirmed=False), pd.DataFrame(), {})
    assert highlighted(none)[0] == ["a", "b"]
    assert highlighted(None) == ([], "") and highlighted(Discovery()) == ([], "")


def test_the_heatmap_scrolls_shortens_long_names_and_turns_when_conditions_outnumber():
    long = "siRNA_pool_against_CCT8_targeting_exon_3_batch_2026_03_15_plate_A_well_B07_4471"
    conditions = [long, *[f"c{i}" for i in range(5)]]
    tests = pd.DataFrame(
        {
            "feature": ["area"] * 6 + ["speed"] * 6,
            "condition": conditions * 2,
            "control_condition": "ctrl",
            "d": np.linspace(-1, 1, 12),
            "passes": False,
            "confirmed": False,
            "q_cells": 0.5,
            "q_paired": np.nan,
        }
    )

    class Discovery:
        pass

    discovery = Discovery()
    discovery.tests = tests
    discovery.ranking = pd.DataFrame({"feature": ["area", "speed"]})
    svg = heatmap(discovery, conditions)
    assert 'class="heatmap-scroll"' in svg and "table-wrap" not in svg
    # More conditions than features: the conditions are the rows.
    rows = re.findall(r'class="hm-row"[^>]*>([^<]*)<', svg)
    columns = re.findall(r'class="hm-col"[^>]*>([^<]*)<', svg)
    assert columns == ["area", "speed"] and rows[1:] == conditions[1:]
    assert rows[0] == shortened(long, 30) and len(rows[0]) == 30 and "…" in rows[0]
    # Cut in the middle: names of a series that differ only at their end stay apart.
    assert rows[0].endswith(long[-5:])
    assert f"<title>{long}</title>" in svg
    # The axis titles follow their axes.
    titles = re.findall(r'class="hm-axis"[^>]*>([^<]*)<', svg)
    assert titles == ["feature, best-ranked first", "condition, against ctrl"]
    wide = heatmap(discovery, conditions[:1])
    assert re.findall(r'class="hm-axis"[^>]*>([^<]*)<', wide) == [
        "condition, against ctrl",
        "feature, best-ranked first",
    ]


def _contrast(fields):
    class Contrast:
        table = pd.DataFrame(
            {
                "fields_to_resolve": fields,
                "d_low": 0.1,
                "resolved": False,
                "too_small": [False] * len(fields),
            }
        )

    return Contrast()


def test_the_fields_to_show_the_rest_are_a_range_with_many_values():
    many = _needs(_contrast([6, 6, 7, 9, 12, 24]), "fields")
    assert many == "6 need about 6-24 fields a side in their repeat"
    few = _needs(_contrast([6, 7, 7]), "fields")
    assert few == "1 needs about 6 and 2 about 7 fields a side in their repeat"


def test_files_past_the_budget_are_linked_and_each_png_is_carried_once(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "a.png").write_bytes(b"\x89PNG small")
    (figures / "b.csv").write_text("x\n1\n")
    (figures / "c.pdf").write_bytes(b"%PDF" + b"0" * 5000)
    files, images = page_files(tmp_path, {"report_short.txt": "text"}, [], 0, budget=1000)
    assert files["figures/a.png"][1] == "img" and set(images) == {"figures/a.png"}
    assert base64.b64decode(files["figures/b.csv"][1]) == b"x\n1\n"
    assert files["figures/c.pdf"][1] == LINKED and files["figures/c.pdf"][2] == 5004
    assert files["report_short.txt"][1]
    card = files_card(files)
    assert "the 1 largest" in card and "open from the figures folder" in card
    markup = inline_images('<img src="figures/a.png" data-file="figures/a.png">', images)
    assert markup.startswith('<img src="data:image/png;base64,')
    everything, _ = page_files(tmp_path, {}, [], 0)
    assert all(entry[1] != LINKED for entry in everything.values())
    assert "inside this page" in files_card(everything)


def test_plotly_strings_escape_only_what_html_reads():
    script = PLOTLY_SCRIPT
    assert "'&quot;'" not in script and "/[&<>]/g" in script


def test_the_runtime_estimate_is_said_in_minutes():
    # As measured on 4 cores: about 13 min and 50 min (stress runs, 2026-10-02 and -05).
    assert 9 * 60 <= rough_runtime(300, 3) <= 14 * 60
    assert 40 * 60 <= rough_runtime(100, 41) <= 55 * 60
    assert rough_runtime(100, 41, n_jobs=1) > rough_runtime(100, 41, n_jobs=4)


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    objects, fovs = _tables()
    setup = _setup()
    folder = tmp_path_factory.mktemp("scale")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
        path = html_report(result, objects, fovs, folder)
    return path.read_text(), folder


def test_the_page_of_16_conditions_is_written_with_grey_maps_and_a_search(page):
    text, _ = page
    data = json.loads(re.search(r'id="embedding-data">(.*?)</script>', text).group(1))
    assert len(data["conditions"]) == 16 and data["colours"][0] == COLOURS[0]
    assert data["colours"].count(GREY) >= 4 and data["grey"] == GREY
    assert 'class="find"' in text and 'id="embedding-conditions"' in text
    assert "are in colour; the other" in text
    # Each part of a split feature figure colours its own conditions, the control blue.
    specs = json.loads(re.search(r'id="feature-data">(.*?)</script>', text).group(1))
    for spec in specs:
        colours = [c["colour"] for c in spec["views"][spec["current"]]["conditions"]]
        assert colours == list(COLOURS[: len(colours)])
    # Each PNG once: as its image, the download reading it from there.
    files = json.loads(re.search(r'id="files-data">(.*?)</script>', text).group(1))
    pngs = [name for name in files if name.endswith(".png")]
    assert pngs and all(files[name][1] == "img" for name in pngs)
    for name in pngs:
        assert text.count(f'data-file="{name}"') >= 1
    assert text.count("data:image/png;base64,") == len(pngs)


def test_static_pngs_of_wide_figures_are_drawn_at_most_2400_pixels(page):
    from PIL import Image

    _, folder = page
    widths = [Image.open(path).size[0] for path in (folder / "figures").glob("*.png")]
    assert widths and max(widths) <= 2400


def test_the_colour_menu_says_when_it_offers_only_the_best_ranked_features():
    from morphocoverage.workflows._html_maps import _colour_menu

    data = {"populations": [], "features": [f"f{i}" for i in range(60)]}
    assert "the 60 best-ranked of 300 features" in _colour_menu(data, 300)
    assert "best-ranked of" not in _colour_menu(data, 60)


def test_long_condition_names_are_cut_in_the_static_matrices():
    long = "siRNA_pool_against_CCT8_targeting_exon_3_batch_2026_03_15_plate_A_well_B07_4471"
    objects, fovs = _tables(conditions=["ctrl", long, "c01"], features=("area",), fields=1)
    (figure,) = plots(objects, fovs, _setup(("area",)))["area"]
    labels = [t.get_text() for ax in figure.axes for t in ax.get_xticklabels()]
    assert long not in labels and shortened(long, 24) in labels and shortened(long, 30) in labels
    assert long in set(figure.pairwise["second"]) | set(figure.pairwise["first"])
