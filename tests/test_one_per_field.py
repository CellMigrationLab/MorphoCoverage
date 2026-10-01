"""One observation per field (one cell per image): the questions about fields do not apply."""

import sys
from pathlib import Path

import pytest

from morphocoverage._style import CAUTION
from morphocoverage.workflows import Setup, default_novelty, report
from morphocoverage.workflows._shared import one_per_field, one_per_field_reasons

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


@pytest.fixture(scope="module")
def cdm():
    data = catalog.load("cct8_cdm_morphology")
    features = tuple(data.feature_cols[:3])
    setup = Setup(
        readout="Area",
        features=features,
        control=data.control,
        novelty=default_novelty(features),
        observation_unit="cell",
    )
    return data, setup, report(*data.tables, setup)


def test_one_cell_per_field_is_recognized(cdm):
    data, setup, _ = cdm
    assert one_per_field(*data.tables, setup)
    tracks = catalog.load("cct7_cct8_glioma_cdm_tracks")
    track_setup = Setup(readout="Mean Speed", observation_unit="track")
    assert not one_per_field(*tracks.tables, track_setup)


def test_the_field_questions_are_not_applicable_said_once_not_warned(cdm):
    _, _, result = cdm
    # Neither the field variability, the object rule with whole fields nor the clustering
    # of unusual cells is run: with one cell per field they cannot be measured.
    for name in ("variability", "object_rules", "unusual", "clustering"):
        assert name not in result.results
    short = result.summary()
    glance = short.split("What is in the dataset")[0]
    assert "➖ Foundation. Why count fields? — not applicable: each field holds one cell" in glance
    assert "➖ 3. What limits my sampling? — not applicable: with one cell per field" in glance
    foundation = short.split("Foundation: why count fields")[1].split("1. Can these")[0]
    assert CAUTION not in foundation


def test_nothing_new_is_left_out_when_new_needs_several_cells_in_a_field(cdm):
    _, setup, result = cdm
    reasons = one_per_field_reasons(setup)
    assert "whether unseen cell states remain is not asked" in reasons["enough"]
    enough = " ".join(result.summary().split("2. Did I image enough?")[2].split())
    assert "a new state needs 3 similar cells in one field" in enough
    assert "nothing new (≤5%)" not in enough


def test_the_contrast_shows_one_d_when_fields_and_cells_coincide(cdm):
    _, _, result = cdm
    full = result.summary(detail="full")
    within = full.split("Within each repeat:")[1]
    assert "d cells" in within and "d fields" not in within.split("\n\n")[0]
