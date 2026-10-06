"""One observation per field (one cell per image): the field questions run on the rows as
fields; only the ideas within a field are not applicable (owner, 2026-10-06, plan v2)."""

import sys
from pathlib import Path

import pytest

from morphocoverage._style import CAUTION
from morphocoverage.sampling_unit.variability import ONE_PER_FIELD
from morphocoverage.workflows import Setup, default_novelty, report
from morphocoverage.workflows._one_per_field import (
    field_novelty,
    one_per_field,
    one_per_field_reasons,
)

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


def test_the_field_questions_run_on_the_rows_as_fields(cdm):
    _, _, result = cdm
    # The spread over fields and the unusual images run; the clustering within fields and
    # the count rule with whole fields (the same draws as random cells) do not.
    assert set(result.results["variability"].groups["status"]) == {ONE_PER_FIELD}
    assert result.results["unusual"].settings["words"] == "images"
    for name in ("object_rules", "clustering"):
        assert name not in result.results
    short = result.summary()
    glance = short.split("What is in the dataset")[0]
    assert "✅ Foundation. Why count fields? — each field holds one of the cells" in glance
    foundation = short.split("Foundation: why count fields")[1].split("1. Can these")[0]
    assert "a mean from whole fields is a mean of as many random cells (design effect 1)" in (
        " ".join(foundation.split())
    )
    assert CAUTION not in foundation
    profile = short.split("3. What is limiting my sampling?")[1].split("6. Which")[0]
    profile = " ".join(profile.split())
    assert "Unusual images (exploratory, farther from the siCTRL reference" in profile
    assert "cluster in a few fields is not applicable: each field holds one cell" in profile


def test_new_is_judged_across_fields(cdm):
    _, setup, result = cdm
    # A new state needs 3 similar cells: on fields of one cell, 3 other fields.
    field = field_novelty(setup.novelty)
    assert (field.min_novel_observations, field.min_other_fovs) == (1, 3)
    assert "enough" not in one_per_field_reasons(setup)
    assert "controls" not in one_per_field_reasons(setup)
    enough = " ".join(result.summary().split("2. Did I image enough?")[2].split())
    assert "nothing new (≤5%)" in enough
    assert "'New': a cell that fewer than 3 other fields show" in enough
    declaration = " ".join(result.summary().split("What this acquisition was asked")[1].split())
    assert "new phenotypes (new: a cell that fewer than 3 other fields show)" in declaration
    assert "unusual images (against the control" in declaration


def test_the_contrast_shows_one_d_when_fields_and_cells_coincide(cdm):
    _, _, result = cdm
    full = result.summary(detail="full")
    within = full.split("Within each repeat:")[1].split("\n  Planning")[0]
    assert "d on cells" in within and "d on fields" not in within
