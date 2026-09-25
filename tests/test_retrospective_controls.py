"""Tests for representation controls (roadmap Step 6)."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective import RepresentationControl, representation_control
from morphocoverage.strategies import SupportNoveltyEstimator

FEATURES = ["feature_a", "feature_b"]
NOVELTY = SupportNoveltyEstimator(1.0, 1, None, 3, 3)


def _plate(seed=0, shift=6.0, plates=("P1", "P2"), fovs_per_well=12):
    """Per plate: a reference well and a positive well shifted by ``shift`` in feature_a."""
    rng = np.random.default_rng(seed)
    fov_rows, object_rows = [], []
    for plate in plates:
        for well, condition, offset in (("A1", "MOCK", 0.0), ("B1", "KD", shift)):
            for i in range(fovs_per_well):
                name = f"{plate}_{well}_{i}"
                fov_rows.append(
                    {"condition": condition, "repeat": plate, "well": well, "image_file": name}
                )
                values = rng.normal(0.0, 1.0, size=(25, 2))
                values[:, 0] += offset
                object_rows.append(
                    pd.DataFrame(values, columns=FEATURES).assign(
                        condition=condition, repeat=plate, well=well, image_file=name
                    )
                )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def _control(objects, fovs, control, **kwargs):
    settings = {"pair_cols": ["repeat"], "fovs_seen": 6, "draws": 10}
    settings.update(kwargs)
    return representation_control(
        objects, fovs, feature_cols=FEATURES, novelty=NOVELTY, control=control, **settings
    )


BOTH = RepresentationControl(
    positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_cells=5
)


def test_a_strong_positive_is_detected_and_rates_are_counted():
    result = _control(*_plate(), BOTH)
    assert result.status == "assessed"
    rates = result.pooled.set_index("candidate")["rate"]
    assert rates["positive"] == 1.0
    assert rates["reference"] < 0.2
    # Five scattered planted cells do not always link into one new state.
    assert rates["reference"] < rates["planted"] < rates["positive"]
    per = result.table[result.table["candidate"] == "positive"]
    assert set(per["repeat"]) == {"P1", "P2"}
    assert (per["candidates"] == 10 * 12).all()
    text = result.summary()
    assert "recognized as different in 100%" in text and "false alarms" in text


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
    assert "False-alarm rate: not assessed" in result.summary()


def test_without_a_control_nothing_is_assessed():
    result = _control(*_plate(), RepresentationControl())
    assert result.status == "not assessed"
    assert "Representation sensitivity: not assessed" in result.summary()


def test_declarations_are_checked():
    with pytest.raises(ValueError, match="positive and a reference"):
        RepresentationControl(positive={"condition": "KD"}, planted_cells=3)
    with pytest.raises(ValueError, match="both a positive and a reference"):
        _control(
            *_plate(),
            RepresentationControl(positive={"repeat": "P1"}, reference={"condition": "MOCK"}),
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
    unpaired = fovs[~((fovs["repeat"] == "P1") & (fovs["condition"] == "MOCK"))]
    unpaired = unpaired[~((unpaired["repeat"] == "P2") & (unpaired["condition"] == "KD"))]
    objects = objects[objects["image_file"].isin(unpaired["image_file"])]
    result = _control(objects, unpaired, BOTH)
    assert result.status == "not assessed"
    text = result.summary()
    assert "a control was declared, but no stratum could be judged" in text
    assert "repeat=P1: no positive-control FOV" not in text  # P1 has positives, no reference
    assert "repeat=P2: no positive-control FOV" in text


def test_missing_values_leave_checks_unjudged_instead_of_crashing():
    objects, fovs = _plate()
    positive = objects["condition"] == "KD"
    objects.loc[positive, "feature_b"] = np.nan
    result = _control(objects, fovs, BOTH)
    assert result.status == "not assessed"
    assert "no check could be made" in result.summary()
    assert result.table["not_judged"].sum() > 0


def test_adding_a_plate_does_not_change_another_plates_rates():
    objects, fovs = _plate()
    one = fovs[fovs["repeat"] == "P1"]
    alone = _control(objects[objects["repeat"] == "P1"], one, BOTH).table
    both = _control(objects, fovs, BOTH).table
    pd.testing.assert_frame_equal(
        both[both["repeat"] == "P1"].reset_index(drop=True), alone.reset_index(drop=True)
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
