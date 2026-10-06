"""Several controls apart in intensity (owner, 2026-10-05).

On the owner's ECM data the untransfected controls (VEC, THBS5) were about 10 times dimmer
than the siRNA controls: imaged or processed differently. With several controls, the
controls are compared on their intensity-like features, and a 2-fold gap or more is said
where the data are described, in "Do the conditions differ?" and beside each comparison
across experiments that joins such controls. With one control nothing changes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ecm_design import CONTROLS, MASTER, PARENTS, tables

from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._acquisition_gap import (
    POINTER,
    acquisition_gaps,
    gap_caution,
    gap_features,
)

INTENSITY = "Intensity (mean)"
DIM = ("VEC", "THBS5")


def _design(dim: float = 0.08, cells: int = 3):
    """The ECM-like design with an intensity: the untransfected conditions ``dim`` times
    the others'."""
    objects, fovs = tables(cells=cells)
    scale = np.where(objects["condition"].isin(DIM), dim, 1.0)
    objects[INTENSITY] = 20000.0 * scale * objects["area"] / 10
    return objects, fovs


def _setup(**kwargs) -> Setup:
    arguments = {
        "readout": "area",
        "features": ("area", INTENSITY),
        "control": CONTROLS,
        "control_parents": PARENTS,
        "master_control": MASTER,
        "observation_unit": "cell",
        **kwargs,
    }
    return Setup(**arguments)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_controls_apart_in_intensity_are_named_with_their_fold():
    objects, fovs = _design()
    setup = _setup()
    assert gap_features(setup) == [INTENSITY]
    gaps = acquisition_gaps(objects, fovs, setup)
    pairs = {frozenset(g["controls"]) for g in gaps}
    # Each dim control against each bright one; the two dim ones, and the two bright, alike.
    assert pairs == {frozenset((d, b)) for d in DIM for b in ("VEC_SiNC", "THBS5_SiNC")}
    assert all(g["feature"] == INTENSITY and 11 < g["fold"] < 14 for g in gaps)
    text = gap_caution(gaps)
    assert text.startswith(
        "VEC_SiNC and VEC differ 13-fold in median Intensity (mean); VEC_SiNC and THBS5 13-fold;"
    )
    # Audit 10 (B14): centring per repeat leaves the gap, so its scope names it too.
    assert (
        "raw comparisons across them, the maps on raw values or centred per repeat, the "
        "separation of the conditions and the populations, may show acquisition" in text
    )


def test_a_gap_under_two_fold_or_one_control_says_nothing():
    objects, fovs = _design(dim=0.7)  # well under 2-fold
    assert acquisition_gaps(objects, fovs, _setup()) == []
    objects, fovs = _design()
    one = Setup(
        readout="area",
        features=("area", INTENSITY),
        control={"condition": "VEC_SiNC"},
        observation_unit="cell",
    )
    assert acquisition_gaps(objects, fovs, one) == []
    assert gap_caution([]) is None


def test_a_listed_feature_is_compared_and_must_be_declared():
    objects, fovs = _design()
    objects["brightness"] = objects[INTENSITY]
    setup = _setup(features=("area", "brightness"), intensity_features=("brightness",))
    assert gap_features(setup) == ["brightness"]
    assert {g["feature"] for g in acquisition_gaps(objects, fovs, setup)} == {"brightness"}
    assert gap_features(_setup(features=("area", "brightness"))) == []
    with pytest.raises(ValueError, match="intensity_features"):
        _setup(intensity_features=("brightness",))


@pytest.fixture(scope="module")
def gap_report():
    objects, fovs = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(objects, fovs, _setup())


def test_the_report_says_the_gap_where_the_data_and_the_comparisons_are(gap_report):
    short = gap_report.summary()
    overview = _flat(short[: short.index("What this acquisition was asked to preserve")])
    assert "⚠️ Controls apart in intensity: " in overview
    glance = _flat(short[: short.index("What is in the dataset")])
    # The fixture's intensity is 160 x area in the dim experiment (VEC, THBS5): an exact
    # relation within that experiment, named too (owner, 2026-10-05).
    assert "values: duplicated or derived features" in glance
    # The gap is said in the acquisition block, with the fields far from their group (plan v2).
    assert "acquisition: controls apart in intensity" in glance or (
        "acquisition: " in glance and ", controls apart in intensity" in glance
    )
    section = _flat(short[short.index("9. Do the conditions differ, and") :])
    # Said once, in the acquisition block; the comparisons point to it (audit 10).
    caution = gap_caution(gap_report.results["contrast"].settings["acquisition_gaps"])
    assert caution not in section and f"⚠️ {POINTER}" in section
    assert _flat(gap_report.summary(detail="full")).count(caution) == 1
    # Beside the row across experiments that joins dim and bright controls, not the others;
    # normalised to their own controls, a softer note, no caution (audit 10, C31).
    assert "transfection effect: THBS5_SiNC/THBS5 vs VEC_SiNC/VEC: joins controls apart" in section
    assert "; each side is normalised to its own control, which takes out a gap that scales" in (
        section
    )
    assert "⚠️ transfection effect: THBS5_SiNC/THBS5 vs VEC_SiNC/VEC: joins" not in section
    assert "THBS5_SiNC vs VEC_SiNC (raw): join" not in section
    gaps = gap_report.results["contrast"].settings["acquisition_gaps"]
    assert len(gaps) == 4


def test_with_one_control_the_report_is_unchanged():
    objects, fovs = _design()
    keep = objects["condition"].isin(["VEC_SiNC", "VEC_SiCol6a1"])
    objects, fovs = objects[keep], fovs[fovs["condition"].isin(["VEC_SiNC", "VEC_SiCol6a1"])]
    setup = Setup(
        readout="area",
        features=("area", INTENSITY),
        control={"condition": "VEC_SiNC"},
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    assert "apart in intensity" not in result.summary(detail="full")
    assert "acquisition_gaps" not in result.results["contrast"].settings


def test_the_caution_is_beside_a_control_against_its_parent_too(gap_report):
    """A main verdict joining two controls apart in intensity, a control against its parent,
    is named under the readings too (owner, 2026-10-05)."""
    section = _flat(gap_report.summary()[gap_report.summary().index("9. Do the conditions") :])
    reading = section[: section.index("Across the biological repeats")]
    assert "VEC_SiNC vs VEC and THBS5_SiNC vs THBS5: join controls apart in intensity" in reading
    assert "VEC_SiCol6a1 vs VEC_SiNC: join" not in reading


def test_the_maps_say_the_gap_centring_leaves():
    """Audit 10 (B14): under the maps, the centred view keeps a gap between experiments
    within a repeat, as do the separation and the populations."""
    from morphocoverage.workflows._acquisition_gap import gap_maps_line

    gaps = [{"controls": ("cA", "cB"), "feature": INTENSITY, "fold": 2.4}]
    line = gap_maps_line(gaps)
    assert line.startswith("Controls apart in intensity (cA and cB): centring per repeat leaves")
    assert "the separation of the conditions and the populations may show acquisition" in line
    assert gap_maps_line(gaps, centred=False).startswith(
        "Controls apart in intensity (cA and cB): the maps, the separation"
    )
    assert gap_maps_line([]) == ""
