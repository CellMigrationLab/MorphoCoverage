"""Tests for representation controls (R1 of docs/report/questions.md)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    BOTH,
    EMPTY,
    FEATURES,
    KEY,
    NOVELTY,
    SYN_FEATURES,
    SYN_NOVELTY,
    _control,
    _plate,
    _syn_inspection,
    _two_plates,
)
from _helpers_sampling_unit import _flat_text

from morphocoverage._style import Status
from morphocoverage._wording import percent
from morphocoverage.bounds import lower_bound
from morphocoverage.retrospective import (
    RepresentationControl,
    inspect_fovs,
    microscopist_report,
    report_status,
    report_status_controls,
    representation_control,
)
from morphocoverage.retrospective.controls import ControlResult
from morphocoverage.retrospective.controls_text import NOT_PLANTED, representation_gate
from morphocoverage.retrospective.report_profile import _detection_adjusted_bounds
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize


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


def test_g12_a_planting_that_cannot_happen_is_said():
    objects, fovs = _plate()  # 25 objects per field
    control = RepresentationControl(
        positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_objects=30
    )
    result = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=NOVELTY,
        control=control,
        pair_cols=["biological_repeat"],
        fovs_seen=6,
        draws=5,
    )
    assert "planted" not in set(result.pooled["candidate"])
    gaps = result.settings["planted_skipped"]
    assert [g["stratum"] for g in gaps] == ["biological_repeat=P1", "biological_repeat=P2"]
    assert (
        "no positive-control field holds 30 objects to plant (the largest holds 25)"
        in (gaps[0]["reason"])
    )
    assert "Planted cells: not measured in biological repeat P1" in result.summary()
    assert "Planted cells: not measured in biological repeat P2" in result.summary()


def test_g13_the_chance_expectation_uses_the_leave_one_field_out_rate():
    novelty = SupportNoveltyEstimator(0.6, 1, None, 3, 2)
    objects, fovs = _plate(seed=3, plates=("P1",), fovs_per_well=30)
    control = RepresentationControl(positive={"condition": "KD"}, reference={"condition": "MOCK"})
    controls = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        control=control,
        pair_cols=["biological_repeat"],
        fovs_seen=3,
        draws=20,
    )
    rates = controls.pooled.set_index("candidate")["rate"]
    inspection = inspect_fovs(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        count_ratio=None,
        group_cols=["condition", "biological_repeat"],
    )
    new = inspection.fovs[inspection.fovs["reason"] == "new state"]
    flagged_reference = int((new["condition"] == "MOCK").sum())
    # The same test as the inspection: its rate is the share of reference fields it flags.
    assert rates["reference_lofo"] == pytest.approx(flagged_reference / 30)
    assert rates["reference"] > 0.3  # after 3 fields, the rate the report used to multiply
    statement = next(
        s
        for s in microscopist_report(controls=controls, inspection=inspection).profile["statement"]
        if s.startswith("new states")
    )
    expected = rates["reference_lofo"] * 60
    # Audit 3 (G-3): from the rate to its upper bound, over the fields judged.
    # Audit 10 (A20): "false alarms alone would give about 2, up to 9, of the 60 fields".
    assert f"would give about {round(expected)}, up to " in statement
    assert "of the 60 fields judged" in statement
    assert f"was {flagged_reference} of 30 fields" in statement


def _sufficiency():
    """The part of a sufficiency report the adjusted bound reads."""
    from types import SimpleNamespace

    # The analysis groups of the controls' tables (audit 3, G-20: other groups are refused).
    next_fov = pd.DataFrame(
        [{"condition": "MOCK", "biological_repeat": "P1", "unseen_state_max_frequency": 0.12}]
    )
    return SimpleNamespace(group_cols=("condition", "biological_repeat"), next_fov=next_fov)


def test_g13_the_adjusted_bound_uses_the_rate_against_every_reference_field():
    objects, fovs = _plate(shift=1.0)
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
    control = RepresentationControl(positive={"condition": "KD"}, reference={"condition": "MOCK"})
    controls = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        control=control,
        pair_cols=["biological_repeat"],
        fovs_seen=6,
        draws=10,
    )
    row = controls.pooled.set_index("candidate").loc["positive_lofo"]
    detection = float(row["rate"])
    text = _detection_adjusted_bounds(_sufficiency(), controls)
    # Audit 3 (G-1, J-2): against the reference fields of an analysis group, via percent().
    assert f"({percent(detection)} of its fields recognized against the reference" in text
    # Audit 4 (P3-9): divided by the rate's one-sided 95% lower bound, not its point estimate.
    low = lower_bound(int(row["judged_new"]), int(row["candidates"]), 0.95)
    assert f"up to {percent(min(1.0, 0.12 / low))} of FOVs" in text


def test_g19_the_adjusted_bound_is_given_with_a_positive_control_only():
    objects, fovs = _plate(shift=1.0)
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
    controls = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=novelty,
        control=RepresentationControl(positive={"condition": "KD"}),
        pair_cols=["biological_repeat"],
        fovs_seen=6,
        draws=10,
    )
    assert controls.status == "positive only"
    text = _detection_adjusted_bounds(_sufficiency(), controls)
    assert text is not None and text.startswith("Adjusted for the declared control's detection")


def _syn_control(objects, fovs, **kwargs):
    settings = {"pair_cols": ["biological_repeat"], "fovs_seen": 10, "draws": 2}
    settings.update(kwargs)
    return representation_control(
        objects,
        fovs,
        feature_cols=SYN_FEATURES,
        novelty=SYN_NOVELTY,
        control=RepresentationControl(
            positive={"condition": "drug"}, reference={"condition": "control"}
        ),
        observation_unit="cell",
        **settings,
    )


def test_g1_false_alarms_are_measured_within_the_inspections_groups(synthetic):
    objects, fovs = synthetic
    controls = _syn_control(objects, fovs)
    inspection = _syn_inspection(objects, fovs)
    assert controls.settings["lofo_group_cols"] == list(inspection.group_cols)
    new = inspection.fovs[inspection.fovs["reason"] == "new state"][KEY].drop_duplicates()
    flagged = int((new["condition"] == "control").sum())
    reference = controls.pooled.set_index("candidate").loc["reference_lofo"]
    # The inspection's own share of control fields called new (7 of 90), not 0 of 90 as
    # when each field was judged against the whole repeat's reference fields.
    assert (int(reference["judged_new"]), int(reference["candidates"])) == (flagged, 90)
    assert flagged > 0
    positive = controls.pooled.set_index("candidate").loc["positive_lofo"]
    # Every drug field holding cells against each control well; since audit 4 (P1-2) the
    # 15 empty drug fields are counted apart, never judged as misses.
    assert int(positive["candidates"]) == 75 * 5
    assert controls.settings["positive_empty"] == 15
    # Measured over each whole stratum, as before, on request.
    pooled = _syn_control(objects, fovs, lofo_group_cols=[])
    assert pooled.pooled.set_index("candidate").loc["reference_lofo", "judged_new"] < flagged


def test_g17_the_leave_one_field_out_false_alarms_are_measured_without_a_positive():
    objects, fovs = _plate()
    keep = ~((fovs["biological_repeat"] == "P2") & (fovs["condition"] == "KD"))
    fovs = fovs[keep]
    objects = objects[objects["image_file"].isin(fovs["image_file"])]
    control = RepresentationControl(positive={"condition": "KD"}, reference={"condition": "MOCK"})
    result = representation_control(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=NOVELTY,
        control=control,
        pair_cols=["biological_repeat"],
        fovs_seen=6,
        draws=3,
    )
    rows = result.table.set_index(["biological_repeat", "candidate"])
    assert ("P2", "reference_lofo") in rows.index
    assert ("P2", "positive") not in rows.index
    assert [s["stratum"] for s in result.settings["skipped_strata"]] == ["biological_repeat=P2"]


def test_i_note_empty_reference_fields_do_not_leave_checks_unjudged(synthetic):
    objects, fovs = synthetic
    result = _syn_control(objects, fovs, pair_cols=[], fovs_seen=5, draws=4)
    pooled = result.pooled.set_index("candidate")
    assert (pooled["not_judged"] == 0).all()
    assert int(pooled.loc["positive", "candidates"]) == 4 * 75  # drug fields with cells (P1-2)
    assert "After 5 reference fields with cells" in result.summary()


def test_p6_4_representation_control_measures_the_planted_rate_leaving_one_field_out():
    rng = np.random.default_rng(3)
    rows, fields = [], []
    for condition, shift in (("MOCK", 0.0), ("KD", 8.0)):
        for j in range(8):
            name = f"{condition}{j}"
            fields.append({"condition": condition, "biological_repeat": "R1", "position_id": name})
            for x, y in rng.normal(size=(20, 2)):
                rows.append(
                    {"condition": condition, "biological_repeat": "R1", "position_id": name}
                    | {"a": x + shift, "b": y}
                )
    controls = representation_control(
        pd.DataFrame(rows),
        pd.DataFrame(fields),
        feature_cols=["a", "b"],
        novelty=SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 3, 3),
        control=RepresentationControl(
            positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_objects=5
        ),
        pair_cols=["biological_repeat"],
        fovs_seen=3,
        draws=2,
    )
    planted = controls.pooled.set_index("candidate").loc["planted_lofo"]
    assert planted["candidates"] + planted["not_judged"] == 8  # one check per reference field
    text = controls.summary()
    assert "MOCK fields with 5 KD cells planted" in text
    assert f"of {int(planted['candidates'])} checks" in text


def test_p1_2_empty_positive_fields_are_not_counted_as_misses():
    rng = np.random.default_rng(5)
    rows, fields = [], []
    for condition, shift in (("MOCK", 0.0), ("KD", 8.0)):
        for j in range(10):
            name = f"{condition}{j}"
            fields.append({"condition": condition, "biological_repeat": "R1", "position_id": name})
            if condition == "KD" and j < 3:
                continue  # 30% of the positive fields hold no cells
            for x, y in rng.normal(size=(20, 2)):
                rows.append(
                    {"condition": condition, "biological_repeat": "R1", "position_id": name}
                    | {"a": x + shift, "b": y}
                )
    controls = representation_control(
        pd.DataFrame(rows),
        pd.DataFrame(fields),
        feature_cols=["a", "b"],
        novelty=SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 3, 3),
        control=RepresentationControl(
            positive={"condition": "KD"}, reference={"condition": "MOCK"}
        ),
        pair_cols=["biological_repeat"],
        fovs_seen=3,
        draws=3,
    )
    rates = controls.pooled.set_index("candidate")["rate"]
    assert rates["positive"] == 1.0 and rates["positive_lofo"] == 1.0
    assert controls.settings["positive_empty"] == 3
    assert "3 positive-control fields without cells were not judged" in controls.summary()


PLANTED = {"control": {"planted_objects": 5}}


def test_the_planted_gate_follows_the_declaration_and_never_reads_the_whole_fields():
    strong = _two_plates(PLANTED, positive=((9, 10), (9, 10)), planted=((10, 10), (9, 10)))
    assert representation_gate(strong, 0.9) == ("", False)
    weak = _two_plates(PLANTED, positive=((10, 10), (10, 10)), planted=((5, 10), (3, 10)))
    gate, is_weak = representation_gate(weak, 0.9)
    assert is_weak and "ordinary fields with 5 planted positive-control objects" in gate
    assert "only 40% of checks, pooled over all plates" in gate
    # No planting could be judged: not assessed, whether the whole fields are strong or weak.
    for whole in ((10, 10), (2, 10)):
        unplanted = _two_plates(PLANTED, positive=(whole, whole))
        assert representation_gate(unplanted, 0.9) == (NOT_PLANTED, False)
    undeclared = _two_plates(positive=((5, 10), (5, 10)))
    gate, is_weak = representation_gate(undeclared, 0.9)
    assert is_weak and "fields of the declared positive control" in gate


def test_a_report_with_no_planting_says_the_gate_was_not_assessed():
    from _helpers_retrospective import _weak_control_report

    _, sufficiency = _weak_control_report()
    plain = microscopist_report(sufficiency=sufficiency)
    # Whole positive fields recognized in only 20% of checks: never the planted answer.
    controls = _two_plates(PLANTED, positive=((2, 10), (2, 10)), reference=((0, 10), (0, 10)))
    report = microscopist_report(controls=controls, sufficiency=sufficiency)
    assert report.statuses["enough"] is plain.statuses["enough"]
    assert report.headlines["enough"] == (
        f"small-state representation not assessed; {plain.headlines['enough']}"
    )
    assert report.sections["enough"].lstrip().startswith(f"{NOT_PLANTED}.")
    assert f"{Status.NOT_ASSESSED.marker} {NOT_PLANTED}." in report.summary()
    assert not any(n.startswith("Weak representation") for n in report.notes)
    assert f"{NOT_PLANTED}, so the adequacy answers (questions 2, 4 and 5) are not checked" in (
        " ".join(report.notes)
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"novelty": "x"}, TypeError, "novelty must be a SupportNoveltyEstimator"),
        ({"control": "x"}, TypeError, "control must be a RepresentationControl"),
        ({"fovs_seen": 0}, ValueError, "fovs_seen must be a positive integer"),
        ({"draws": 0}, ValueError, "draws must be a positive integer"),
        ({"random_state": 1.5}, TypeError, "random_state must be an integer seed"),
    ],
)
def test_representation_control_checks_its_arguments_in_order(kwargs, error, message):
    arguments = {
        "feature_cols": ["a"],
        "novelty": NOVELTY,
        "control": RepresentationControl(),
        "pair_cols": [],
        "fovs_seen": 3,
        **kwargs,
    }
    with pytest.raises(error) as raised:
        representation_control(EMPTY, EMPTY, **arguments)
    assert str(raised.value) == message


def test_controls_status_not_judged_and_without_false_alarms():
    status = report_status_controls.controls_status
    Status = report_status.Status
    unusable = SimpleNamespace(status="not assessed", settings={"reason": "unusable"})
    assert status(unusable) == (Status.CAUTION, "a control was declared, but could not be judged")
    undeclared = SimpleNamespace(status="not assessed", settings={"reason": "none"})
    assert status(undeclared) == (Status.NOT_ASSESSED, "no positive control declared")
    pooled = pd.DataFrame({"candidate": ["reference"], "rate": [0.1]})
    alarms_only = ControlResult(
        pooled.assign(stratum="all FOVs"), pooled, "positive only", (), {"control": {}}
    )
    assert status(alarms_only) == (Status.CAUTION, "positive-control fields: not judged")
    pooled = pd.DataFrame({"candidate": ["positive"], "rate": [0.95]})
    settings = {"control": {"positive": {"condition": ["KD"]}}}
    positive = ControlResult(
        pooled.assign(stratum="all FOVs"), pooled, "positive only", (), settings
    )
    assert status(positive) == (
        Status.CAUTION,
        "KD fields recognized in 95% of checks; false alarms not assessed",
    )


def test_the_control_short_is_one_table_and_the_caveat_once(controls):
    text = controls.summary()
    assert "Can these features see the declared phenotype (KD against MOCK" in text
    assert "KD fields (positive control)" in text and "MOCK fields (false alarms)" in text
    rates = controls.pooled.set_index("candidate")
    assert f"{percent(rates.loc['positive', 'rate'])} of" in text
    assert _flat_text(text).lower().count("this holds for this definition of new") == 1


def test_the_control_full_keeps_every_count(controls):
    text = controls.summary(detail="full")
    for row in controls.pooled.to_dict("records"):
        assert f"{int(row['judged_new'])} of {int(row['candidates'])}" in text
    for row in controls.table.to_dict("records"):
        if row["candidates"]:
            assert f"({int(row['judged_new'])}/{int(row['candidates'])})" in text
