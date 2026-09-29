"""The readable style of the Foundation, question 1 and question 3 (restyle step R1).

Short summaries answer in a sentence and one table with groups by their
values; full summaries keep every number the sentences had; each caveat is
stated once; the status markers follow ``report_status``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from morphocoverage._style import CAUTION, Status, width
from morphocoverage._wording import percent
from morphocoverage.retrospective import (
    RepresentationControl,
    microscopist_report,
    representation_control,
)
from morphocoverage.retrospective.report_status import (
    controls_status,
    profile_status,
    unit_status,
)
from morphocoverage.sampling_unit import (
    descriptor_spread_check,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.strategies import SupportNoveltyEstimator


def _flat(text: str) -> str:
    return " ".join(text.split())


def _group(condition, repeat, *, n_fields=24, per_field=30, field_sd=0.0, seed=0, mean=10.0):
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, field_sd, n_fields)
    objects = pd.DataFrame(
        [
            {
                "condition": condition,
                "repeat": repeat,
                "image_file": f"{condition}{repeat}{j}",
                "x": v,
            }
            for j in range(n_fields)
            for v in mean + offsets[j] + rng.normal(0, 1.0, per_field)
        ]
    )
    fovs = objects[["condition", "repeat", "image_file"]].drop_duplicates().reset_index(drop=True)
    return objects, fovs


def _tables(field_sd=0.0):
    parts = [
        _group("MOCK", "r1", field_sd=field_sd, seed=1),
        _group("KD", "r1", field_sd=field_sd, seed=2, mean=12.0),
    ]
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


@pytest.fixture(scope="module")
def unit():
    objects, fovs = _tables(field_sd=1.5)
    return {
        "variability": fov_variability(objects, fovs, feature="x", observation_unit="cell"),
        "object_rules": object_rule_check(
            objects, fovs, feature="x", control={"condition": "MOCK"}, n_draws=200
        ),
        "descriptor_spread": descriptor_spread_check(
            objects, fovs, feature="x", threshold=100, n_draws=20
        ),
    }


def _plate(shift=6.0, plates=("P1", "P2")):
    rng = np.random.default_rng(0)
    fov_rows, object_rows = [], []
    for plate in plates:
        for condition, offset in (("MOCK", 0.0), ("KD", shift)):
            for i in range(12):
                name = f"{plate}_{condition}_{i}"
                fov_rows.append({"condition": condition, "repeat": plate, "image_file": name})
                values = rng.normal(0.0, 1.0, size=(25, 2))
                values[:, 0] += offset
                object_rows.append(
                    pd.DataFrame(values, columns=["a", "b"]).assign(
                        condition=condition, repeat=plate, image_file=name
                    )
                )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


CONTROL = RepresentationControl(
    positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_objects=5
)


def _control(objects, fovs):
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
    return representation_control(
        objects,
        fovs,
        feature_cols=["a", "b"],
        novelty=novelty,
        control=CONTROL,
        pair_cols=["repeat"],
        fovs_seen=6,
        draws=10,
    )


@pytest.fixture(scope="module")
def controls():
    return _control(*_plate())


@pytest.fixture(scope="module")
def states():
    objects, fovs = _tables()
    labels = pd.Series(False, index=objects.index)
    # MOCK: the state in 2 fields only (clustered); KD: one object in every field (spread).
    mock = objects["condition"].eq("MOCK") & objects["image_file"].isin(["MOCKr10", "MOCKr11"])
    labels[mock] = True
    kd = objects[objects["condition"].eq("KD")].groupby("image_file").head(1).index
    labels[kd] = True
    clustering = fov_clustering(
        objects.assign(state=labels), fovs, label_col="state", states=[True]
    )
    unusual = unusual_objects(
        objects, fovs, feature_cols=["x"], reference={"condition": "MOCK"}, seed_checks=2
    )
    return clustering, unusual


def test_every_summary_is_short_by_default_and_checks_its_detail(unit, controls, states):
    for result in [*unit.values(), controls, *states]:
        assert result.summary() == result.summary(detail="short")
        assert result.summary(detail="full") != ""
        with pytest.raises(ValueError, match="short"):
            result.summary(detail="brief")


def test_groups_print_by_their_values_not_as_column_equals_value(unit, controls, states):
    for result in [*unit.values(), controls, *states]:
        for detail in ("short", "full"):
            text = result.summary(detail=detail)
            assert "condition=" not in text and "repeat=" not in text, (result, detail)
    assert "MOCK · r1" in unit["variability"].summary()


def test_the_field_variability_short_is_one_answer_and_one_table(unit):
    text = unit["variability"].summary()
    assert text.startswith("How variable are the fields (x)? A mean from whole fields has")
    assert "design effect" in text and "KD · r1" in text
    assert all(width(line) <= 100 for line in text.splitlines())


def test_the_field_variability_full_keeps_every_number(unit):
    result = unit["variability"]
    text = result.summary(detail="full")
    for row in result.groups.to_dict("records"):
        for value in (
            str(int(row["fovs"])),
            str(int(row["observations"])),
            percent(row["icc"]),
            f"{row['design_effect']:.1f}",
            f"{row['error_ratio']:.1f}",
            percent(row["unusual_fov_share"]),
        ):
            assert value in text


def test_the_jcb_check_full_keeps_every_number_and_the_short_names_misses(unit):
    result = unit["object_rules"]
    full = _flat(result.summary(detail="full"))
    ok = result.table[result.table["status"] == "ok"]
    assert len(ok)
    for row in ok.to_dict("records"):
        for value in (
            str(int(row["rule_objects"])),
            f"{row['fovs_used']:.1f}",
            percent(row["predicted_error"]),
            percent(row["real_error"]),
            f"{row['error_ratio']:.1f}x",
            f"{int(row['fov_rule_fovs'])} ({int(row['fov_rule_fovs_upper95'])})",
        ):
            assert value in full
    short = _flat(result.summary())
    assert "the mean of all MOCK objects of the same repeat" in short
    missed = ok[~ok["meets_target"].astype(bool)]
    # A strong field effect: the rule's promise breaks with whole fields.
    assert len(missed) and "misses its target with whole fields" in short


def test_the_foundation_status_follows_the_rule_with_whole_fields(unit):
    status, headline = unit_status(unit["variability"], unit["object_rules"], None)
    assert status is Status.CAUTION
    assert "the JCB count rule missed with whole fields in" in headline
    held = dataclasses.replace(
        unit["object_rules"], table=unit["object_rules"].table.assign(meets_target=True)
    )
    status, headline = unit_status(unit["variability"], held, None)
    assert status is Status.OK
    assert "still held with whole fields in all 2 groups" in headline
    assert "a mean from whole fields varies" in headline


def test_the_control_short_is_one_table_and_the_caveat_once(controls):
    text = controls.summary()
    assert "Can these features see the declared phenotype (KD against MOCK" in text
    assert "KD fields (positive control)" in text and "MOCK fields (false alarms)" in text
    rates = controls.pooled.set_index("candidate")
    assert f"{percent(rates.loc['positive', 'rate'])} of" in text
    assert _flat(text).lower().count("this holds for this definition of new") == 1


def test_the_control_full_keeps_every_count(controls):
    text = controls.summary(detail="full")
    for row in controls.pooled.to_dict("records"):
        assert f"{int(row['judged_new'])} of {int(row['candidates'])}" in text
    for row in controls.table.to_dict("records"):
        if row["candidates"]:
            assert f"({int(row['judged_new'])}/{int(row['candidates'])})" in text


def test_a_stratum_not_judged_is_named_by_its_column_and_value():
    objects, fovs = _plate()
    keep = ~(fovs["repeat"].eq("P2") & fovs["condition"].eq("KD"))
    fovs = fovs[keep]
    objects = objects[objects["image_file"].isin(fovs["image_file"])]
    result = _control(objects, fovs)
    status, headline = controls_status(result, minimum_detection=0.01)
    assert status is Status.CAUTION
    assert "not judged in repeat P2 (no positive-control field with objects)" in headline
    assert "Not judged after 6 reference fields in repeat P2" in _flat(result.summary())


def test_clustering_short_answers_per_state_and_full_keeps_every_number(states):
    clustering, _ = states
    short = _flat(clustering.summary())
    assert "Are the state objects spread over the fields?" in short
    assert "in 1 of 2 groups where seen (MOCK · r1)" in short
    full = clustering.summary(detail="full")
    for row in clustering.table[clustering.table["state_objects"] > 0].to_dict("records"):
        for value in (
            percent(row["prevalence"]),
            f"{row['expected_fovs_if_random']:.1f}",
            f"{row['clustering_ratio']:.1f}",
            f"{row['p_clustered']:.2g}",
            f"{int(row['fovs_with_state'])} of {int(row['fovs'])}",
        ):
            assert value in full


def test_question_3_is_a_caution_when_a_state_clusters(states):
    clustering, unusual = states
    report = microscopist_report(clustering=clustering, unusual=unusual)
    assert report.statuses["profile"] is Status.CAUTION
    assert "state objects clustered in 1 of 2 groups" in report.headlines["profile"]
    spread = dataclasses.replace(clustering, table=clustering.table.assign(p_clustered=1.0))
    status, _ = profile_status(report.profile, [spread])
    assert status is Status.OK


def test_the_report_sections_are_short_with_each_caveat_once(unit, controls, states):
    clustering, unusual = states
    report = microscopist_report(**unit, controls=controls, clustering=clustering, unusual=unusual)
    for key in ("unit", "controls", "profile"):
        short = report.short[key]
        assert short != report.sections[key]
        cautions = [line for line in short.splitlines() if line.lstrip().startswith(CAUTION)]
        assert len(cautions) == len(set(cautions))
        assert all(width(line) <= 100 for line in short.splitlines()), key
        assert all(width(line) <= 100 for line in report.sections[key].splitlines()), key
        assert "condition=" not in short
    text = report.summary()
    assert f"{report.statuses['unit'].marker} Foundation: why count fields" in text
    assert report.sections["profile"] in report.summary(detail="full")
    # The full profile keeps every statement with its source.
    for row in report.profile.itertuples():
        assert _flat(row.statement) in _flat(report.sections["profile"])


def test_a_text_added_to_a_short_section_goes_before_its_caveats_each_once():
    from morphocoverage.workflows._document import _merged

    short = "  - a line\n  ⚠️ First caveat,\n     continued.\n  ⚠️ Second caveat."
    brief = "  - an added line\n  ⚠️ Second caveat.\n  ⚠️ Third caveat."
    assert _merged(short, brief).splitlines() == [
        "  - a line",
        "  - an added line",
        "  ⚠️ First caveat,",
        "     continued.",
        "  ⚠️ Second caveat.",
        "  ⚠️ Third caveat.",
    ]
