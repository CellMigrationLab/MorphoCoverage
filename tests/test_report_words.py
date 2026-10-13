"""The report's words for the design and the observations (owner decisions of 2026-10-02):
the overview says the layout as it is when the conditions are not all in every repeat, the
words for the observations follow the declared unit (and are cells when none is declared),
and the plan's options do not read as planning a significance."""

import html
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_html import _tables

from morphocoverage import RareTarget
from morphocoverage.retrospective.overview import dataset_overview
from morphocoverage.workflows import Setup, html_report, report
from morphocoverage.workflows._html_render import adjusted_p

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
sys.path.insert(0, str(LAB))
import catalog

REPEATS = ("r1", "r2", "r3", "r4")


def _design(plan, seed=0):
    """Tables of a design: each condition in the repeats ``plan`` gives it, 3 fields of 5."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, held in plan.items():
        for repeat in held:
            for field in range(3):
                for label in range(5):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "technical_repeat": "A1",
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(20, 2),
                        }
                    )
    objects = pd.DataFrame(rows)
    keys = ["condition", "biological_repeat", "technical_repeat", "image_file"]
    return objects, objects[keys].drop_duplicates()


def _design_line(plan):
    objects, fovs = _design(plan)
    overview = dataset_overview(objects, fovs, feature_cols=["area"], control={"condition": "ctrl"})
    return overview.short_lines()[2]


def test_a_complete_layout_keeps_its_words():
    line = _design_line({c: REPEATS[:3] for c in ("ctrl", "a", "b")})
    assert "3 conditions (ctrl, a, b), 3 biological repeats (r1, r2, r3), one well per " in line
    assert "condition in each repeat (9 wells); each well" in line
    assert "sub-experiment" not in line and " of the " not in line


def test_an_uneven_layout_says_the_repeats_each_condition_holds():
    """Stress design E2: "one well per condition in each repeat (14 wells)" misread a
    design in which cond2 and cond4 are not in every repeat."""
    plan = {"ctrl": REPEATS, "c1": REPEATS[:3], "c2": REPEATS[:2], "c3": REPEATS, "c4": ["r1"]}
    line = _design_line(plan)
    assert "; c1 in 3, c2 in 2, c4 in 1 of the 4 repeats; one well per condition in each " in line
    assert "repeat, 14 wells;" in line


def test_a_condition_never_beside_the_control_is_named():
    """Stress design E4: a condition imaged only in repeats without the control."""
    plan = {"ctrl": REPEATS[:3], "c1": REPEATS[:3], "c3": ["r5", "r6", "r7"]}
    line = _design_line(plan)
    assert "c3 in no repeat with ctrl" in line


def test_sub_experiments_are_said_with_their_own_control_and_repeats():
    """The glioma screen: two sub-experiments, each with its own siCTRL in its own 3
    repeats, not "5 conditions ... 6 biological repeats ... one well per condition in each
    repeat"."""
    data = catalog.load("cct7_cct8_glioma_cdm_tracks")
    overview = dataset_overview(
        *data.tables,
        feature_cols=data.feature_cols,
        control=data.control,
        observation_unit=data.observation_unit,
    )
    line = overview.short_lines()[2]
    assert (
        "; 2 sub-experiments, each of 2 conditions and its own siCTRL in its own 3 biological "
        "repeats (siCCT7#3 and siCCT7#6; siCCT8#5 and siCCT8#8); one well per condition in "
        "each repeat, 18 wells;"
    ) in line
    full = overview.summary(detail="full")
    assert "2 sub-experiments, each of 2 conditions" in " ".join(full.split())


def _visible(page):
    """The page's text a reader can see, tooltips included; no scripts, styles or tags."""
    page = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.DOTALL)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())


# "cell" where it is not the report's observations: the papers' titles and journals, the
# lab's name, and the simulations behind the cell-level test (which simulate cells).
ALLOWED = (
    r"J Cell (?:Biol|Sci)",
    r"Cell Migration Lab",
    r"in cell biology",
    r"how many cells to average",
    r"breast cancer cell invasion",
    r"30 cells a well",
    r"0\.5 cell SD",
    r"times the cells of the others",
)


def _stray(text):
    """Each "cell" or "cells" in ``text`` (with its context) that is not an allowed use;
    "cell-level" is one, since the test is named for the observations."""
    stray = re.sub("|".join(ALLOWED), " ", " ".join(text.split()))
    return [m.group(0) for m in re.finditer(r".{0,40}\b[Cc]ells?\b.{0,40}", stray)]


def _reported(objects, fovs, setup, folder):
    """The report, its page's visible text, and its short and full text, without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
        page = html_report(result, objects, fovs, folder).read_text()
    return result, _visible(page), result.summary(), result.summary(detail="full")


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    """A report on nuclei and one whose unit is not declared."""
    objects, fovs = _tables()
    made = {}
    for unit in ("nucleus", None):
        setup = Setup(
            readout="area",
            features=("area", "speed", "aspect"),
            control={"condition": "ctrl"},
            observation_unit=unit,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = report(objects, fovs, setup)
            folder = tmp_path_factory.mktemp(str(unit))
            page = html_report(result, objects, fovs, folder).read_text()
        made[unit] = result, page
    return made


def test_a_page_on_nuclei_says_nuclei_never_cells(pages):
    """Owner, 2026-10-02: "it could be something else segmented in the image"; the text
    report too, which the page no longer rewrites (the sources say the unit's word)."""
    result, page = pages["nucleus"]
    text = _visible(page)
    for words in (text, result.summary(), result.summary(detail="full")):
        assert not _stray(words), _stray(words)
    for words in (
        "every nucleus a point",
        "Are there populations or groups of nuclei?",
        "Why count fields, and not only nuclei?",
        "Nucleus-level randomization test",
        "nucleus-level screen",
    ):
        assert words in text, words


def test_undeclared_observations_are_called_cells(pages):
    result, page = pages[None]
    assert not result.results["overview"].settings["observation_unit_declared"]
    text = result.summary()
    assert text.splitlines()[0].startswith("Microscopist's report — 180 cells in 30 fields")
    assert "10. Groups of cells?" in text and "180 cells (assumed, not declared)" in text
    assert "every cell a point" in _visible(page)


def test_the_plan_options_do_not_read_as_planning_significance(pages):
    """ChatGPT audit A1: "The mean of Round in every well, significant across repeats" read
    as if optimise planned the significance; those features come first instead."""
    _, page = pages["nucleus"]
    text = _visible(page)
    assert "The mean of area in every well" in text
    assert "in every well, significant across repeats" not in text


@pytest.fixture(scope="module")
def tracks(tmp_path_factory):
    """The synthetic acquisition reported as tracks, with its planted rare label declared:
    every answer runs, the populations and the label's group included."""
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv")
    rare = RareTarget(
        fov_incidence=0.05, detection_probability=0.95, min_fov_occurrences=3, phenotype="rare"
    )
    setup = Setup(
        readout="area",
        features=("area", "intensity", "eccentricity"),
        control={"condition": "control"},
        observation_unit="track",
        rare=rare,
        rare_label="rare",
    )
    return _reported(objects, fovs, setup, tmp_path_factory.mktemp("tracks"))


def test_a_report_on_tracks_says_tracks_never_cells(tracks):
    _, page, short, full = tracks
    for text in (short, full, page):
        assert not _stray(text), _stray(text)
    flat = " ".join(short.split())
    for words in (
        "Track-level screen (exploratory, as in Plot&Stats): Cohen's d on every track",
        "track-level screen (exploratory) q",
        "q track level",
        "The track-level test treats the tracks as independent",
        "Candidates come from the track-level test",
        # "of every condition pooled" read as a number per condition (ChatGPT, 2026-10-05).
        "2421 tracks in all, every condition pooled, clustered on",
        "most of its tracks carry the label",
    ):
        assert words in flat, words
    assert "track-level test does not" in " ".join(full.split())
    assert "Track-level screen" in page and "track-level adjusted p" in page


def test_the_declared_labels_group_is_said_once(tracks):
    """The synthetic demo said "G1 is the declared rare phenotype, rare: most of its cells
    carry the label; ..." in In short and again in the card's text under it."""
    result, page, _, _ = tracks
    headline = result.document.headlines["populations"]
    said = "is the declared rare phenotype, rare: most of its tracks carry the label"
    assert said in headline
    assert said not in result.document.short["populations"]
    start = re.search(r"groups of tracks\?\W+In short:", page)
    assert start, page
    card = page[start.start() :]
    card = card[: card.index("Assumptions")]
    assert card.count(said) == 1, card
    assert "of its tracks carry the label rare, and it holds" in card


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    """Patches in two biological repeats: the candidates of the screen are the answer."""
    objects, fovs = _tables()
    objects = objects[objects["biological_repeat"] != "r3"]
    fovs = fovs[fovs["biological_repeat"] != "r3"]
    setup = Setup(
        readout="area",
        features=("area", "speed", "aspect"),
        control={"condition": "ctrl"},
        observation_unit="patch",
    )
    return _reported(objects, fovs, setup, tmp_path_factory.mktemp("pilot"))


def test_a_pilot_on_patches_says_patches_never_cells(pilot):
    _, page, short, full = pilot
    for text in (short, full, page):
        assert not _stray(text), _stray(text)
    assert "exploratory patch-level screen" in " ".join(short.split())
    assert "every patch an independent draw" in page
    assert "patch-level d" in page


def test_the_page_names_the_screen_once():
    """The text report already says the unit's word: the page's pass leaves it alone and
    names only what is still written of cells (the tab, the glossary)."""
    markup = "<p>Track-level screen; Cell-level screen; q track level</p>"
    assert adjusted_p(markup, "tracks", "track") == (
        "<p>Track-level screen; Track-level screen; adjusted p, tracks</p>"
    )
    assert adjusted_p("<th>q cell level</th>") == "<th>adjusted p, cells</th>"
