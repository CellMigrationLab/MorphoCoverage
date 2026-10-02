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
