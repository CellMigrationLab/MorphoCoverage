"""The rare phenotype and the fingerprint in the report (owner, 2026-10-02, after reading a
synthetic screen with both declared): the labelled share is compared like a feature, the
distributions are compared beyond the mean, both targets are answered in "Did I image
enough?", and the label checks the features."""

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage import RareTarget
from morphocoverage._style import Status
from morphocoverage.multivariate_outcomes import QuantileError
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.sampling_unit.distribution import distribution_contrast
from morphocoverage.workflows import Setup, default_novelty, html_report, report
from morphocoverage.workflows._label_control import label_answer, label_control

FEATURES = ("area", "dna", "ratio")


def _screen(seed=0, narrow=0.6, mitotic=(0.02, 0.10)):
    """siCTRL, narrow (the same mean, a narrower spread) and arrest (more mitoses), 3
    repeats, 2 wells each, 6 fields of 40 cells; mitoses are bright and small."""
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for r, repeat in enumerate(("2026_01_05", "2026_01_12", "2026_01_19")):
        batch = rng.normal(0, 0.1, 3)
        for c, condition in enumerate(("siCTRL", "narrow", "arrest")):
            for w in range(2):
                well = f"{'ABC'[c]}{w + 1}"
                for f in range(6):
                    image = f"{condition}_{repeat}_{well}_{f}.tif"
                    z = rng.normal(0, 1, (40, 3)) + batch
                    if condition == "narrow":
                        z = z * narrow
                    share = mitotic[1] if condition == "arrest" else mitotic[0]
                    marked = rng.random(40) < share
                    z[marked] = rng.normal([-2.5, 4.0, 2.5], 0.3, (marked.sum(), 3))
                    for i in range(40):
                        rows.append(
                            {
                                "condition": condition,
                                "biological_repeat": repeat,
                                "technical_repeat": well,
                                "image_file": image,
                                "label_id": i + 1,
                                "area": 200 + 20 * z[i, 0],
                                "dna": 1000 + 100 * z[i, 1],
                                "ratio": 0.4 + 0.05 * z[i, 2],
                                "mitotic": bool(marked[i]),
                            }
                        )
                    fovs.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "technical_repeat": well,
                            "image_file": image,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _setup(**changes):
    settings = {
        "readout": "area",
        "features": FEATURES,
        "control": {"condition": "siCTRL"},
        "fingerprint": (
            OutcomeCriterion(
                QuantileError(FEATURES, True, quantiles=(0.1, 0.5, 0.9)),
                0.1,
                "lower_is_better",
                "the 10-90% quantiles within 0.1 SD",
            ),
        ),
        "rare": RareTarget(
            object_prevalence=0.02,
            detection_probability=0.95,
            min_fov_occurrences=3,
            phenotype="mitotic (pH3)",
        ),
        "rare_label": "mitotic",
        "novelty": default_novelty(FEATURES),
        "observation_unit": "cell",
        "random_state": 0,
    }
    return Setup(**{**settings, **changes})


@pytest.fixture(scope="module")
def screen():
    objects, fovs = _screen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, _setup())
    return objects, fovs, result


def test_the_labelled_share_is_compared_like_a_feature(screen):
    _, _, result = screen
    ranking = result.results["discovery"].ranking
    assert "mitotic share" in set(ranking["feature"])
    tests = result.results["discovery"].tests
    arrest = tests[(tests["feature"] == "mitotic share") & (tests["condition"] == "arrest")]
    assert int(arrest["higher"].iloc[0]) == 3
    headline = result.document.headlines["conditions"]
    assert "of 4 (3 features and the mitotic share) significant across repeats" in headline


def test_a_narrower_distribution_with_the_same_mean_is_found_beyond_the_mean(screen):
    _, _, result = screen
    distribution = result.results["distribution"]
    spread = distribution.tests.query("condition == 'narrow' and statistic == 'spread'")
    # The planted spread: SD 0.6x the control's (a little more with the few mitoses).
    assert np.exp(spread["mean"]).between(0.55, 0.75).all()
    assert (spread["lower"] == 3).all() and (spread["q"] <= 0.05).all()
    assert "narrow" in set(distribution.differing()["condition"])
    short = result.document.short["conditions"]
    assert "Beyond the mean (the fingerprint is declared)" in short
    assert "- narrow — differs beyond the mean:" in short
    headline = result.document.headlines["conditions"]
    assert "beyond the mean, the distribution differs for" in headline
    assert "narrow (area, dna and ratio)" in headline
    assert result.document.statuses["conditions"] is Status.OK


def test_the_spread_and_quantiles_are_read_per_repeat_against_its_own_control():
    objects, fovs = _screen(narrow=1.0)
    same = distribution_contrast(
        objects[objects["condition"] != "arrest"],
        fovs[fovs["condition"] != "arrest"],
        features=["area"],
        control={"condition": "siCTRL"},
        observation_unit="cell",
    )
    assert same.tests["repeats"].eq(3).all() and not len(same.differing())
    assert set(same.tests["statistic"]) == {"spread", "q10", "q50", "q90"}


def test_both_targets_are_answered_in_did_i_image_enough(screen):
    _, _, result = screen
    short = result.document.short["enough"]
    assert "The rare phenotype, mitotic (pH3), to be seen in at least 3 fields: seen in" in short
    assert "arrest: 6 of 6 wells" in short
    assert "The fingerprint, the 10-90% quantiles within 0.1 SD: its error with the fields" in short
    headline = result.document.headlines["enough"]
    assert "mitotic seen in at least 3 fields in" in headline
    assert "the fingerprint within its tolerance in" in headline
    # A declared prevalence with a label is a usual declaration: no warning about it.
    assert "No field incidence" not in result.summary()
    assert "no field incidence of the rare phenotype declared" not in result.summary()


def test_more_fields_give_a_smaller_fingerprint_error():
    from morphocoverage.workflows._enough_targets import fingerprint_precision

    objects, fovs = _screen()
    few = fovs.groupby(["condition", "biological_repeat", "technical_repeat"]).head(4)
    keep = objects.merge(few[["image_file"]], on="image_file")
    setup = _setup()
    many = fingerprint_precision(objects, fovs, setup)["error"].median()
    fewer = fingerprint_precision(keep, few, setup)["error"].median()
    assert fewer > many


def test_the_labelled_cells_are_named_and_counted_where_they_cluster(screen):
    _, _, result = screen
    glance = result.document.headlines["profile"]
    assert "labelled states" not in glance and "mitotic cells" in glance
    inspection = result.document.short.get("inspect", "")
    if "a state no other field shows" in inspection:
        assert "labelled mitotic)" in inspection


def test_the_label_checks_what_the_features_see(screen):
    objects, fovs, result = screen
    check = result.results["label_control"]
    assert check.recognized >= 0.9 and check.chance < 0.1
    assert result.document.statuses["controls"] is Status.OK
    assert result.document.headlines["controls"].startswith("mitotic cells recognized by their")
    # A label the features cannot see: its cells' neighbours are labelled about by chance.
    rng = np.random.default_rng(3)
    blind = objects.assign(mitotic=rng.random(len(objects)) < 0.05)
    status, _, _ = label_answer(label_control(blind, fovs, _setup()))
    assert status is Status.CAUTION


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_page_has_a_tab_beyond_the_mean(screen, tmp_path):
    objects, fovs, result = screen
    page = html_report(result, objects, fovs, tmp_path).read_text()
    conditions = page[page.index('id="conditions"') : page.index('id="enough"')]
    assert ">Beyond the mean<" in conditions


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_targets_have_their_own_tabs_after_the_features(screen, tmp_path):
    # Each declared target answers in a line at the top of "Did I image enough?" and in its
    # own tab after the features; what optimise's held-back fields can confirm is for the
    # plan of the next experiment (owner, 2026-10-02).
    import html
    import re

    objects, fovs, result = screen
    page = html_report(result, objects, fovs, tmp_path).read_text()
    start = page.index('<section class="card" id="enough"')
    card = page[start : page.index("</section>", start)]
    tabs = re.findall(r'role="tab" id="enough-tab-\d+"[^>]*>([^<]+)<', card)
    features = list(result.results["enough_features"].table["feature"])
    assert tabs == [*features, "rare phenotype", "fingerprint"]
    top = html.unescape(re.sub(r"<[^>]+>", "", card[: card.index("One feature at a time")]))
    assert "The rare phenotype, mitotic, was seen in at least 3 fields in" in top
    assert "The fingerprint is within its tolerance in" in top
    last = len(tabs) - 1
    fingerprint = card[card.index(f'id="enough-panel-{last}"') :]
    assert "The fingerprint, the 10-90% quantiles within 0.1 SD: its error" in html.unescape(
        fingerprint
    )
    assert "Optimise judges it" not in card and "held-back" not in card


def test_issue_145_a_comparison_not_tested_is_not_called_not_significant():
    from morphocoverage.workflows._distribution import distribution_headline, distribution_text

    objects, fovs = _screen()
    dropped = "narrow_2026_01_19"
    objects = objects[~objects["image_file"].str.startswith(dropped)]
    fovs = fovs[~fovs["image_file"].str.startswith(dropped)]
    result = distribution_contrast(
        objects, fovs, features=FEATURES, control={"condition": "siCTRL"}
    )
    assert result.tests.query("condition == 'narrow'")["q"].isna().all()
    paragraph = distribution_text(result)
    line = " ".join(paragraph.split("- narrow — ")[1].split("\n    condition")[0].split())
    assert line.startswith("the same way in every repeat, not tested (fewer than 3 repeats): ")
    assert "not significant" not in line and "q –" not in line
    assert "    narrow  " in paragraph  # its values are in the table
    assert "not significant" not in distribution_headline(result)


def test_issue_147_the_table_keeps_the_smallest_q_when_it_is_cut():
    from morphocoverage.sampling_unit.distribution import STATISTICS, DistributionContrast
    from morphocoverage.workflows._distribution import MAX_ROWS, distribution_text

    rows = []
    # Sub-experiment 1: 30 features, each p 0.01 on one statistic (q 0.04 after BH over 120).
    for i in range(30):
        for statistic, p in zip(STATISTICS, (0.01, 0.9, 0.9, 0.9), strict=True):
            rows.append(("A", f"f{i:02d}", statistic, p, 0.04 if p < 0.5 else 0.9, 1))
    # Sub-experiment 2: a larger p but a smaller q (0.02 over 4 tests).
    for statistic in STATISTICS:
        rows.append(("B", "g", statistic, 0.02, 0.02, 2))
    tests = pd.DataFrame(
        rows, columns=["condition", "feature", "statistic", "p", "q", "experiment"]
    )
    tests = tests.assign(repeats=3, lower=0, higher=3, mean=0.1, low=0.0, high=0.2)
    settings = {"alpha": 0.05, "families": 2, "observation_unit": "cell"}
    text = distribution_text(DistributionContrast(tests, settings))
    table = text.split("90% shift")[1]
    assert table.split("\n")[1].split()[:2] == ["B", "g"]
    assert table.count(" (q ") == 4 * MAX_ROWS
    assert "and 7 more features" in text
