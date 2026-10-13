"""Tests for representation controls (R1 of docs/report/questions.md)."""

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import BOTH, FEATURES, NOVELTY, _control, _plate

from morphocoverage.retrospective import RepresentationControl, representation_control


def test_a_strong_positive_is_detected_and_rates_are_counted():
    result = _control(*_plate(), BOTH)
    assert result.status == "assessed"
    rates = result.pooled.set_index("candidate")["rate"]
    assert rates["positive"] == 1.0
    assert rates["reference"] < 0.2
    # Five scattered planted cells do not always link into one new state.
    assert rates["reference"] < rates["planted"] < rates["positive"]
    per = result.table[result.table["candidate"] == "positive"]
    assert set(per["biological_repeat"]) == {"P1", "P2"}
    assert (per["candidates"] == 10 * 12).all()
    text = result.summary()
    assert "KD fields (positive control)" in text and "100% of 240 checks" in text
    assert "false alarms" in text


def test_a_positive_identical_to_the_reference_is_detected_at_the_false_alarm_rate():
    objects, fovs = _plate(seed=1, shift=0.0, plates=("P1",), fovs_per_well=20)
    result = _control(objects, fovs, BOTH, draws=40, fovs_seen=8)
    rates = result.pooled.set_index("candidate")["rate"]
    assert abs(rates["positive"] - rates["reference"]) < 0.06
    assert abs(rates["planted"] - rates["reference"]) < 0.06


def test_without_a_reference_the_false_alarm_rate_is_not_assessed():
    result = _control(*_plate(), RepresentationControl(positive={"condition": ["KD"]}))
    assert result.status == "positive only"
    assert "reference" not in set(result.pooled["candidate"])
    assert "not assessed (no reference control was declared)" in result.summary()


def test_without_a_control_nothing_is_assessed():
    result = _control(*_plate(), RepresentationControl())
    assert result.status == "not assessed"
    assert "Representation sensitivity: not assessed" in result.summary()


def test_declarations_are_checked():
    with pytest.raises(ValueError, match="positive and a reference"):
        RepresentationControl(positive={"condition": "KD"}, planted_objects=3)
    with pytest.raises(ValueError, match="both a positive and a reference"):
        _control(
            *_plate(),
            RepresentationControl(
                positive={"biological_repeat": "P1"}, reference={"condition": "MOCK"}
            ),
        )
    with pytest.raises(ValueError, match="not in the FOV table"):
        _control(*_plate(), RepresentationControl(positive={"genotype": "KD"}))
    with pytest.raises(ValueError, match="pair_cols missing"):
        _control(*_plate(), BOTH, pair_cols=["plate"])


def test_deterministic_under_a_seed():
    first = _control(*_plate(), BOTH, random_state=3)
    again = _control(*_plate(), BOTH, random_state=3)
    pd.testing.assert_frame_equal(first.table, again.table)


def test_controls_on_different_plates_are_declared_but_not_judged():
    objects, fovs = _plate()
    unpaired = fovs[~((fovs["biological_repeat"] == "P1") & (fovs["condition"] == "MOCK"))]
    unpaired = unpaired[
        ~((unpaired["biological_repeat"] == "P2") & (unpaired["condition"] == "KD"))
    ]
    objects = objects[objects["image_file"].isin(unpaired["image_file"])]
    result = _control(objects, unpaired, BOTH)
    assert result.status == "not assessed"
    text = " ".join(result.summary().split())
    assert "a control was declared, but no stratum could be judged" in text
    assert "repeat P1: no positive-control" not in text  # P1 has positives, no reference
    assert "repeat P2: the positive control was not imaged here" in text  # A24


def test_missing_values_leave_checks_unjudged_instead_of_crashing():
    objects, fovs = _plate()
    positive = objects["condition"] == "KD"
    objects.loc[positive, "feature_b"] = np.nan
    result = _control(objects, fovs, BOTH)
    assert result.status == "not assessed"
    assert "no check could be made" in result.summary()
    assert result.table["not_judged"].sum() > 0


def test_fields_with_a_missing_pairing_value_are_refused_not_dropped():
    # Equality never matches a missing value: plate P2's fields would vanish and be
    # reported as a stratum with no positive-control field (#227).
    objects, fovs = _plate()
    fovs["plate"] = fovs["biological_repeat"].where(fovs["biological_repeat"] == "P1")
    with pytest.raises(ValueError, match=r"pair_cols \['plate'\] have missing values"):
        _control(objects, fovs, BOTH, pair_cols=["plate"])


def test_a_generator_of_features_is_read_once():
    # The settings text used to consume it before validation (#222).
    objects, fovs = _plate(plates=("P1",))
    listed = _control(objects, fovs, BOTH, draws=3)
    generated = representation_control(
        objects,
        fovs,
        feature_cols=(c for c in FEATURES),
        novelty=NOVELTY,
        control=BOTH,
        pair_cols=["biological_repeat"],
        fovs_seen=6,
        draws=3,
    )
    pd.testing.assert_frame_equal(listed.table, generated.table)


def test_adding_a_plate_does_not_change_another_plates_rates():
    objects, fovs = _plate()
    one = fovs[fovs["biological_repeat"] == "P1"]
    alone = _control(objects[objects["biological_repeat"] == "P1"], one, BOTH).table
    both = _control(objects, fovs, BOTH).table
    pd.testing.assert_frame_equal(
        both[both["biological_repeat"] == "P1"].reset_index(drop=True), alone.reset_index(drop=True)
    )


def test_worker_processes_give_identical_rates():
    objects, fovs = _plate()
    one = _control(objects, fovs, BOTH)
    two = _control(objects, fovs, BOTH, n_jobs=3)
    pd.testing.assert_frame_equal(one.table, two.table)
    pd.testing.assert_frame_equal(one.pooled, two.pooled)
    assert one.status == two.status
    with pytest.raises(ValueError, match="n_jobs"):
        _control(objects, fovs, BOTH, n_jobs=0)
