"""Regression tests of audit 4, group G: novelty and rare states.

One test per finding (P6-1..P6-6, P3-8, P3-9, P1-2, P1-5, P4-10, P9-7, and
the next-field wording of P3-4 and P4-9), on small deterministic tables; the
reproductions behind them are in the audit's P1, P3, P4, P6 and P9 folders.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables
from morphocoverage._wording import percent
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.bounds import lower_bound, upper_bound
from morphocoverage.objectives import ObjectiveConfig, RareTarget, SamplingObjective
from morphocoverage.retrospective import (
    RepresentationControl,
    microscopist_report,
    representation_control,
)
from morphocoverage.retrospective.controls import ControlResult
from morphocoverage.retrospective.report_profile import _detection_adjusted_bounds
from morphocoverage.sampling_unit import fov_clustering, unusual_objects
from morphocoverage.schema import fov_positions
from morphocoverage.statistics import first_detection_fovs, fovs_to_detect
from morphocoverage.strategies import SupportNoveltyEstimator, SupportNoveltyStrategy
from morphocoverage.strategies.support_novelty import UPPER_QUANTITY
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.sufficiency_text import _next_fov_answer, novelty_definition
from morphocoverage.transforms import RobustStandardize, Standardize
from morphocoverage.workflows import Setup, default_novelty
from morphocoverage.workflows._shared import control_checks, rare_clustering

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _fields(labels_per_fov, condition="drug", repeat="R1", well="A"):
    """Objects with a boolean label, one list of labels per field, and the FOV table."""
    objects, fovs = [], []
    for j, labels in enumerate(labels_per_fov):
        name = f"{condition}_{repeat}_{well}_f{j:02d}"
        identity = {"condition": condition, "biological_repeat": repeat, "technical_repeat": well}
        fovs.append({**identity, "position_id": name})
        for value in labels:
            objects.append(
                {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "technical_repeat": well,
                    "position_id": name,
                    "state": value,
                    "x": 0.0,
                }
            )
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _concat(*tables):
    return tuple(pd.concat(parts, ignore_index=True) for parts in zip(*tables, strict=True))


def test_p6_1_a_field_holds_an_unusual_state_only_with_m_objects():
    # A 15-object state in 1 of 30 drug fields, and one stray "unusual" object in
    # 6 other drug fields and in 5 of 30 control fields, as unusual_objects flags
    # a share of ordinary objects by construction.
    drug = [[False] * 20 for _ in range(30)]
    drug[11] = [True] * 15 + [False] * 20
    for j in (1, 4, 9, 17, 22, 28):
        drug[j] = [True] + [False] * 19
    control = [[False] * 20 for _ in range(30)]
    for j in (2, 7, 13, 19, 25):
        control[j] = [True] + [False] * 19
    objects, fovs = _concat(_fields(drug), _fields(control, condition="control"))
    # 99 shuffles: with 9, no p could reach 0.05 and the test is left out (audit 10).
    plain = fov_clustering(objects, fovs, label_col="state", states=[True], n_permutations=99)
    held = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=[True],
        n_permutations=99,
        min_objects_per_fov=3,
        reference={"condition": "control"},
    )
    before = plain.table.set_index("condition").loc["drug"]
    after = held.table.set_index("condition").loc["drug"]
    # Before: 7 of 30 fields "hold" it, so a handful of fields seem enough.
    assert before["fovs_needed"] == first_detection_fovs(7 / 30, 0.95) < 15
    # After: only the field with the state holds it, and the truth (89) is given.
    assert after["fovs_holding_state"] == 1
    assert after["fovs_needed"] == first_detection_fovs(1 / 30, 0.95) == 89
    # The clustering test still counts every field with one of the objects.
    assert after["fovs_with_state"] == 7 and after["p_clustered"] == before["p_clustered"]
    # The reference groups' own incidence is printed beside it.
    assert after["reference_fovs"] == 30 and after["reference_fovs_holding"] == 0
    text = held.summary()
    assert "holding ≥3" in text and "1 of 30" in text
    assert "reference holding" in held.summary(detail="full")
    assert "0 of 30 (0%)" in text
    assert "89 (up to" in text


def test_p6_1_the_report_counts_unusual_fields_with_the_novelty_rules_m():
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for condition in ("control", "drug"):
        for repeat in ("R1", "R2", "R3"):
            for j in range(6):
                name = f"{condition}{repeat}f{j}"
                fields.append(
                    {"condition": condition, "biological_repeat": repeat, "position_id": name}
                )
                for x, y in rng.normal(size=(20, 2)):
                    rows.append(
                        {"condition": condition, "biological_repeat": repeat, "position_id": name}
                        | {"a": x, "b": y}
                    )
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fields)
    setup = Setup("a", features=("a", "b"), control={"condition": "control"})
    results = control_checks(objects, fovs, setup, {})
    settings = results["clustering"].settings
    assert settings["min_objects_per_fov"] == default_novelty(["a", "b"]).min_novel_observations
    assert settings["reference"] == {"condition": "control"}


def _debris_table():
    """16 fields; a compact 8-object state at +6 SD in field 3, debris at +60 SD in 8, 11."""
    rng = np.random.default_rng(1)
    rows = []
    for j in range(16):
        xy = rng.normal(size=(30, 2))
        if j == 3:
            xy = np.r_[xy, np.array([6.0, 0.0]) + 0.3 * rng.normal(size=(8, 2))]
        if j in (8, 11):
            xy = np.r_[xy, np.array([60.0, 0.0]) + 3 * rng.normal(size=(20, 2))]
        rows.append(pd.DataFrame({"position_id": f"f{j:02d}", "a": xy[:, 0], "b": xy[:, 1]}))
    objects = pd.concat(rows, ignore_index=True)
    fovs = pd.DataFrame({"position_id": [f"f{j:02d}" for j in range(16)]})
    return validate_input_tables(objects, fovs=fovs, feature_cols=["a", "b"])


def test_p6_2_debris_in_two_fields_does_not_mask_a_new_state():
    data = _debris_table()
    positions = fov_positions(data, data.fovs)

    def new_fields(transform):
        estimator = SupportNoveltyEstimator(np.sqrt(2), 1, transform, 5, 3)
        members = estimator.novel_observation_flags(data, new_states_only=True)
        return set(positions[members].tolist())

    # The mean and SD are widened by the debris, and the state disappears.
    assert 3 not in new_fields(Standardize())
    assert 3 in new_fields(RobustStandardize())
    assert isinstance(default_novelty(["a", "b"]).transform, RobustStandardize)
    # The robust scale is the SD for normal data, and ignores a few extreme values.
    rng = np.random.default_rng(2)
    values = rng.normal(size=(2000, 1))
    state, _ = RobustStandardize().fit_matrix(values, ("a",))
    assert state["scale"][0] == pytest.approx(1.0, abs=0.05)
    spoiled = np.r_[values, np.full((20, 1), 500.0)]
    assert RobustStandardize().fit_matrix(spoiled, ("a",))[0]["scale"][0] < 1.1
    # A feature with a MAD of 0 that still varies (a count) is scaled by its SD.
    counts = np.r_[np.zeros((8, 1)), np.array([[1.0], [3.0]])]
    state, _ = RobustStandardize().fit_matrix(counts, ("n",))
    assert state["sd_scaled"] == ["n"]
    assert state["scale"][0] == pytest.approx(counts.std(ddof=1))


def test_p6_3_many_standardized_features_warn_and_the_definition_says_how_compact():
    features = tuple(f"f{i}" for i in range(8))
    with pytest.warns(UserWarning, match="small, ordinarily variable states are often missed"):
        Setup("f0", features=features, novelty=default_novelty(features))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Setup("f0", features=features[:4], novelty=default_novelty(features[:4]))
    text = novelty_definition(default_novelty(features[:4]), list(features[:4]))
    assert "each linked to another by a step within the distance" in text
    assert "sit about 2.8 apart" in text


def _controls(settings=None, **rates):
    """A ControlResult holding only pooled counts: kind -> (judged new, judged)."""
    pooled = pd.DataFrame(
        [
            {"candidate": k, "judged_new": n, "candidates": t, "not_judged": 0, "rate": n / t}
            for k, (n, t) in rates.items()
        ]
    )
    return ControlResult(
        pooled.assign(stratum="all FOVs"),
        pooled,
        "assessed",
        ("biological_repeat",),
        {"fovs_seen": 10, **(settings or {})},
    )


def _sufficiency(bound=0.1):
    frame = pd.DataFrame({"g": ["a"], "unseen_state_max_frequency": [bound], "confidence": [0.95]})
    return type("S", (), {"next_fov": frame, "group_cols": ("g",)})()


def test_p6_4_the_unseen_bound_is_adjusted_by_the_planted_rate_when_declared():
    planted = {"control": {"planted_objects": 5}}
    controls = _controls(planted, positive=(29, 30), positive_lofo=(29, 30), planted_lofo=(6, 30))
    text = _detection_adjusted_bounds(_sufficiency(), controls)
    low = lower_bound(6, 30, 0.95)
    assert "for 5 planted positive-control cells" in text
    assert "whole-field phenotypes" not in text
    assert f"up to {percent(min(1.0, 0.1 / low))} of FOVs" in text
    assert f"divided by {percent(low)}" in text
    # Without planted objects the whole-field rate is used, and its scope is said.
    whole = _detection_adjusted_bounds(_sufficiency(), _controls(positive_lofo=(29, 30)))
    assert "whole-field phenotypes like the control only" in whole
    # The weak-representation warning is gated on the planted rate when declared.
    controls = _controls(planted, positive=(29, 30), planted=(6, 30), positive_lofo=(29, 30))
    report = microscopist_report(controls=controls)
    note = next(n for n in report.notes if n.startswith("Weak representation"))
    assert "ordinary fields with 5 planted positive-control objects" in note
    no_control = microscopist_report()
    assert any("these features and this definition of new" in n for n in no_control.notes)


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


def test_p6_5_rare_state_fields_report_the_technical_repeats_holding_it():
    well_a = [[True] + [False] * 9 if j < 5 else [False] * 10 for j in range(10)]
    empty = [[False] * 10 for _ in range(10)]
    objects, fovs = _concat(_fields(well_a), _fields(empty, well="B"), _fields(empty, well="C"))
    result = fov_clustering(objects, fovs, label_col="state", states=[True], n_permutations=9)
    assert set(result.table["technical_repeats"]) == {1}
    text = result.summary()
    assert "drug · repeat R1" in text and "1 of 3 technical repeats" in text
    assert "The field counts assume fields from independent technical repeats" in text
    pooled = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=[True],
        n_permutations=9,
        group_cols=["condition", "biological_repeat"],
    )
    row = pooled.table.iloc[0]
    assert (row["technical_repeats"], row["technical_repeats_holding"]) == (3, 1)


def _row(**extra):
    row = {
        "chance_new": 0.02,
        "chance_new_at_most": 0.12,
        "confidence": 0.95,
        "fovs_with_new": 0,
        "fovs_imaged": 24,
        "unseen_state_max_frequency": upper_bound(0, 24, 0.95),
        "fovs_without_observations": 0,
        "extra_fovs_if_nothing_new": 34,
        "extra_fovs_at_current_rate": 40,
        "extra_fovs_to_rule_out_target": 34,
        "min_novel_observations": 3,
    }
    return pd.Series({**row, **extra})


def test_p6_6_states_below_m_objects_per_field_are_said_never_to_count():
    text = _next_fov_answer(_row(), None, None)
    assert (
        "A state carried by fewer than 3 similar observations per FOV is never counted, "
        "however many FOVs hold it." in text
    )
    assert "never counted" not in _next_fov_answer(_row(min_novel_observations=1), None, None)


def _wells_table():
    rng = np.random.default_rng(4)
    rows, fields = [], []
    for well in ("A", "B", "C"):
        for j in range(8):
            name = f"{well}{j}"
            fields.append(
                {
                    "condition": "c",
                    "biological_repeat": "R1",
                    "technical_repeat": well,
                    "position_id": name,
                }
            )
            for x, y in rng.normal(size=(15, 2)):
                rows.append(
                    {"condition": "c", "biological_repeat": "R1", "technical_repeat": well}
                    | {"position_id": name, "a": x, "b": y}
                )
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_p3_8_a_group_pooling_technical_repeats_bounds_their_share():
    objects, fovs = _wells_table()
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(3.0, 1, None, 5, 3),
        group_cols=["condition", "biological_repeat"],
    )
    row = report.next_fov.iloc[0]
    assert row["technical_repeats"] == 3
    assert row["unseen_state_max_technical_share"] == pytest.approx(upper_bound(0, 3, 0.95))
    assert "Fields of one technical repeat are not independent draws" in row["answer"]
    per_well = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(3.0, 1, None, 5, 3),
    )
    assert "technical repeat are not" not in " ".join(per_well.next_fov["answer"])


def test_p3_9_the_adjusted_bound_divides_by_the_low_end_of_the_detection_rate():
    text = _detection_adjusted_bounds(_sufficiency(0.004), _controls(positive_lofo=(19, 20)))
    low = lower_bound(19, 20, 0.95)
    assert f"divided by {percent(low)}" in text and "divided by 95%" not in text
    assert "at least 78% at 95% confidence, of 20 checks" in text


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


def test_p1_5_the_rare_field_plan_uses_min_fov_occurrences():
    labels = [[True, False] if j < 2 else [False, False] for j in range(8)]
    objects, fovs = _fields(labels)
    result = fov_clustering(
        objects,
        fovs,
        label_col="state",
        states=[True],
        detection_probability=0.9,
        min_fov_occurrences=2,
        n_permutations=9,
    )
    assert result.table.iloc[0]["fovs_needed"] == fovs_to_detect(0.25, 0.9, 2) == 15
    text = result.summary()
    assert "to see in ≥2, 90%" in text and " 15 (up to" in text
    rare = RareTarget(
        detection_probability=0.9, min_fov_occurrences=2, phenotype="state", fov_incidence=0.25
    )
    setup = Setup("x", rare=rare, rare_label="state")
    workflow = rare_clustering(objects, fovs, setup)
    assert workflow.table.iloc[0]["fovs_needed"] == 15


def test_p4_10_support_novelty_stops_on_the_fov_upper_bound_by_default():
    rng = np.random.default_rng(0)
    rows = []
    for j in range(5):
        xy = rng.normal(size=(800 if j < 2 else 20, 2))
        if j >= 2:  # each sparse field shows a state of its own
            xy = np.r_[xy, np.array([10.0 * j, 10.0]) + 0.05 * rng.normal(size=(4, 2))]
        rows.append(pd.DataFrame({"position_id": f"f{j}", "a": xy[:, 0], "b": xy[:, 1]}))
    data = validate_input_tables(
        pd.concat(rows, ignore_index=True),
        fovs=pd.DataFrame({"position_id": [f"f{j}" for j in range(5)]}),
        feature_cols=["a", "b"],
    )
    strategy = SupportNoveltyStrategy(1.0, 0.05, 1, None, 5, 3)
    assert strategy.quantity == UPPER_QUANTITY
    evidence = strategy.estimators[0].estimate(data, objective=POPULATION)
    # The object-weighted fraction reads settled; the default quantity does not.
    assert evidence.value("lofo_novel_fraction") < 0.05 < evidence.value(UPPER_QUANTITY)
    with pytest.raises(ValueError, match="tolerance 0"):
        SupportNoveltyStrategy(1.0, 0.0, 1, None, 5, 3)


def test_p9_7_near_constant_features_are_refused_and_named():
    rng = np.random.default_rng(6)
    noisy = np.c_[rng.normal(size=50), 7 + 1e-9 * rng.normal(size=50)]
    for transform in (Standardize(), RobustStandardize()):
        reason = transform.fit_matrix(noisy, ("area", "texture"))
        assert isinstance(reason, str) and "['texture']" in reason
    # The unusual-objects message names the feature, not its column index.
    rows, fields = [], []
    for repeat in ("R1", "R2", "R3"):
        for j in range(4):
            name = f"{repeat}f{j}"
            fields.append({"condition": "MOCK", "biological_repeat": repeat, "position_id": name})
            for value in rng.normal(size=10):
                rows.append(
                    {"condition": "MOCK", "biological_repeat": repeat, "position_id": name}
                    | {"area": value, "texture": 7.0}
                )
    result = unusual_objects(
        pd.DataFrame(rows),
        pd.DataFrame(fields),
        feature_cols=["area", "texture"],
        reference={"condition": "MOCK"},
    )
    statuses = " ".join(result.groups["status"])
    assert "['texture']" in statuses and "['1']" not in statuses


def test_p3_4_p4_9_the_next_field_answer_names_its_assumptions():
    calibrated = {"predicted": 0.04, "actual": 0.05, "bound_held": 0.95, "fovs_imaged": 10}
    text = _next_fov_answer(_row(), 0.05, {**calibrated, "draws": 20})
    assert "it treats FOVs as independent, representative draws" in text
    assert "neighbouring fields acquired in a raster" in text
    assert "The bound holds at a number of FOVs fixed in advance, not at the first FOV" in text
    assert "this check cannot reveal neighbour correlation" in text
