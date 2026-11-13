"""MC-2: the experiment block of MorphoFeature's coverage.json gives the default controls.

The golden exports (``tests/morphofeature_exports``) are MorphoFeature main's (MF-5b): each
``coverage.json`` is ``schema_version`` 3 and ends with the ``experiment`` block of MorphoFeature's
design note ``coverage-experiment-block-2026-11-12.md``. Each refusal of its section 4 and each
default is tested here: the controls default to the description's ``references`` (a chain, ``v10``,
compared as a chain), a user's control overrides them and the report says so, and the target
suggestions are used only once the user confirms them (``v27``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_morphofeature_exports import EFFECT, EXPORTS, NAMES, SPEC, with_feature

from morphocoverage._repeat_groups import FOLDER_CONTROL, PairingRefused
from morphocoverage.described import (
    COVERAGE_VERSION_UNKNOWN,
    DESCRIPTION_VERSION_UNKNOWN,
    DISAGREES,
    MALFORMED,
    MISSING,
    NOT_EXPORTED,
    TWO_DESCRIPTIONS,
    UNCONFIRMED,
    UNPAIRED,
    DescriptionRefused,
    parse_coverage,
    read_coverage,
    read_experiment,
)
from morphocoverage.workflows import Setup, described_setup, report, scan_design

DEFAULTS = {"v03": "CTRL", "v27": "CTRL", "v30": "CTRL", "v33": "ctrl"}
FAST = {"readout": "x", "features": ("x",)}


def _record(name: str) -> dict:
    return json.loads((EXPORTS / name / "coverage.json").read_text(encoding="utf-8"))


def _experiment(name: str, record: dict | None = None):
    objects, fovs = with_feature(name)
    coverage = EXPORTS / name if record is None else parse_coverage(record, f"{name}.json")
    return objects, fovs, read_experiment(coverage, fovs)


def _refused(code: str, call, *args, **kwargs) -> DescriptionRefused:
    with pytest.raises(DescriptionRefused) as refused:
        call(*args, **kwargs)
    assert refused.value.code == code, str(refused.value)
    return refused.value


# --- The goldens carry the block, as the design note's worked examples give it ----------------


@pytest.mark.parametrize("name", NAMES)
def test_each_golden_export_is_version_3_with_the_block_last(name) -> None:
    record = _record(name)
    assert record["schema_version"] == 3
    assert list(record)[-1] == "experiment"
    parsed = read_coverage(EXPORTS / name)
    assert parsed.schema_version == 3 and parsed.experiment is not None


def test_v03_and_v33_blocks_are_the_notes_worked_examples() -> None:
    # Design note section 5: the SHA-256 with the Cells mask, the digests and every ID.
    v03 = read_coverage(EXPORTS / "v03").experiment
    assert v03.description_sha256 == (
        "bd5099bfe78bb13369782891caf9c97f768210674441d03e56889e12bba28eae"
    )
    assert dict(v03.references) == {"CTRL": None, "DrugA": "CTRL"}
    assert dict(v03.repeat_groups["Day1"]) == {"CTRL": "rep1", "DrugA": "experiment_23"}
    v33 = read_coverage(EXPORTS / "v33").experiment
    assert v33.description_sha256 == (
        "20aded4ab7ca9b8730bdf55fedf482b7b9e8a4a18cb0e3b0450315c83fcf41bd"
    )
    assert dict(v33.conditions) == {"DrugA": "DrugA", "ctrl": "control"}
    assert v33.suggestions["DrugA"] == ("SRC", "saracatinib")


# --- Defaults: the controls are the description's references -----------------------------------


@pytest.mark.parametrize("name", sorted(DEFAULTS))  # one control each; v10 is a chain
def test_the_controls_default_to_the_descriptions_references(name) -> None:
    objects, fovs, experiment = _experiment(name)
    assert experiment.default_control() == {"control": {"condition": DEFAULTS[name]}}
    setup = described_setup(experiment, **FAST)
    assert setup.control == {"condition": DEFAULTS[name]}
    assert setup.described is not None and setup.described.chosen is None
    lines = [line for line in setup.declaration() if line.startswith("Controls:")]
    assert lines and "the experiment description's references" in lines[0]
    # The default runs exactly what declaring the same control runs.
    declared = Setup(**FAST, control={"condition": DEFAULTS[name]})
    pd.testing.assert_frame_equal(
        report(objects, fovs, setup).results["contrast"].table,
        report(objects, fovs, declared).results["contrast"].table,
    )


def test_v03_is_compared_by_default_and_the_report_says_where_the_control_comes_from() -> None:
    objects, fovs, experiment = _experiment("v03")
    result = report(objects, fovs, described_setup(experiment, **FAST))
    table = result.results["contrast"].table
    np.testing.assert_allclose(table["mean"] - table["control_mean"], EFFECT, atol=0.3)
    text = result.summary(detail="full")
    assert "Controls: the experiment description's references (description bd5099bfe78b)" in text
    assert "DrugA vs CTRL" in text


def test_a_user_control_overrides_the_reference_and_the_report_names_both() -> None:
    objects, fovs, experiment = _experiment("v27")
    setup = described_setup(experiment, **FAST, control={"condition": "DrugB"})
    assert setup.control == {"condition": "DrugB"}
    assert setup.described.chosen == "DrugB"
    text = report(objects, fovs, setup).summary(detail="full")
    assert "overriding the experiment description" in text
    assert "the description gives DrugA vs CTRL, DrugB vs CTRL and DrugC vs CTRL" in text
    assert "the user chose DrugB" in text
    # Declaring exactly the reference is no override.
    same = described_setup(experiment, **FAST, control={"condition": "CTRL"})
    assert same.described.chosen is None


def test_the_override_by_folder_label_is_refused_with_the_id() -> None:
    objects, fovs, experiment = _experiment("v33")
    setup = described_setup(experiment, **FAST, control={"condition": "control"})
    with pytest.raises(PairingRefused, match="declare it as 'ctrl'") as refused:
        report(objects, fovs, setup)
    assert refused.value.code == FOLDER_CONTROL


def test_the_ids_are_matched_exactly() -> None:
    # 'ctrl' is not 'CTRL': the default is the ID as written; a case variant names no condition.
    objects, fovs, experiment = _experiment("v33")
    assert described_setup(experiment, **FAST).control == {"condition": "ctrl"}
    setup = described_setup(experiment, **FAST, control={"condition": "CTRL"})
    with pytest.raises(ValueError, match="matched exactly"):
        report(objects, fovs, setup)


# --- v10: a chain is compared as a chain ------------------------------------------------------


def _described_tables(fixture: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tables of a described fixture, built from its description (two objects per field),
    with the block MorphoFeature would export for it (note section 2.3)."""
    raw = (SPEC / "fixtures" / "valid" / fixture).read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    conditions = data["conditions"]
    rows = []
    for rid, repeat in data["repeats"].items():
        for cid, member in repeat["members"].items():
            for field in range(member["image_count"]):
                folder, source = conditions[cid]["source_folder"], member["source_repeat"]
                key = (folder, source, "single", f"img{field}")
                rows.append((*key, rid, cid, "single"))
    columns = ["condition", "biological_repeat", "technical_repeat", "position_id"]
    fovs = pd.DataFrame(
        rows, columns=[*columns, "repeat_group", "condition_id", "technical_repeat_id"]
    )
    objects = fovs[columns].loc[fovs.index.repeat(2)].reset_index(drop=True)
    objects["label_id"] = np.tile([1, 2], len(fovs))
    depth = {cid: _depth(cid, conditions) for cid in conditions}
    ids = dict(zip(fovs["condition"], fovs["condition_id"], strict=False))
    objects["x"] = [
        10 + EFFECT * depth[ids[c]] + 0.05 * int(p[3:]) + 0.1 * label
        for c, p, label in zip(
            objects["condition"], objects["position_id"], objects["label_id"], strict=True
        )
    ]
    digest = hashlib.sha256(raw).hexdigest()
    block = {
        "description_sha256": digest,
        "description_schema_version": 1,
        "section_digests": {"conditions": digest, "repeats": digest},
        "conditions": {c: {"source_folder": v["source_folder"]} for c, v in conditions.items()},
        "references": {c: v.get("reference") for c, v in conditions.items()},
        "repeat_groups": {
            rid: {c: {"source_repeat": m["source_repeat"]} for c, m in r["members"].items()}
            for rid, r in data["repeats"].items()
        },
        "suggestions": {
            c: {"target": v.get("target"), "treatment_key": v.get("treatment_key")}
            for c, v in conditions.items()
        },
    }
    return objects, fovs, {"schema_version": 3, "exported": True, "experiment": block}


def _depth(cid: str, conditions: dict) -> int:
    reference = conditions[cid].get("reference")
    return 0 if reference is None else 1 + _depth(reference, conditions)


def test_v10_the_chain_is_compared_as_a_chain() -> None:
    objects, fovs, record = _described_tables("v10_reference_chain.toml")
    experiment = read_experiment(parse_coverage(record, "v10.json"), fovs)
    default = experiment.default_control()
    assert set(default) == {"contrasts"}
    pairs = set(
        zip(default["contrasts"]["condition"], default["contrasts"]["reference"], strict=True)
    )
    # Each condition against its own control, and each control against its parent.
    assert pairs == {
        ("VEC_SiNC", "VEC"),
        ("VEC_SiCol6a1", "VEC_SiNC"),
        ("THBS5_SiNC", "THBS5"),
        ("THBS5_SiCol6a1", "THBS5_SiNC"),
    }
    setup = described_setup(experiment, **FAST)
    assert setup.control is None
    table = report(objects, fovs, setup).results["contrast"].table
    assert set(table["condition"]) == {f"{c} vs {r}" for c, r in pairs}
    # Never every condition against one root: VEC_SiCol6a1 is compared with VEC_SiNC only.
    np.testing.assert_allclose(table["mean"] - table["control_mean"], EFFECT, atol=0.3)
    assert "the description gives" not in " ".join(setup.declaration())


def test_v10_suggests_the_shared_target_without_grouping_on_it() -> None:
    objects, fovs, record = _described_tables("v10_reference_chain.toml")
    experiment = read_experiment(parse_coverage(record, "v10.json"), fovs)
    assert dict(experiment.target_suggestions.suggested) == {
        "THBS5_SiCol6a1": "COL6A1",
        "VEC_SiCol6a1": "COL6A1",
    }
    result = report(objects, fovs, described_setup(experiment, **FAST))
    assert result.results["contrast"].targets.empty


# --- Not refusals: a control without fields, a condition and control that share no repeat ------


def test_a_reference_without_fields_is_not_compared_and_said() -> None:
    _, fovs = with_feature("v27")
    kept = fovs[fovs["condition_id"] != "CTRL"]
    experiment = read_experiment(EXPORTS / "v27", kept)
    assert experiment.pairs == ()
    assert experiment.default_control() == {}
    assert "DrugA: not compared: its control CTRL has no field in this export." in experiment.notes
    setup = described_setup(experiment, **FAST)
    assert setup.control is None and setup.contrasts is None  # never another control
    assert any("its control CTRL has no field" in line for line in setup.declaration())


def test_a_condition_sharing_no_repeat_with_its_control_is_not_paired_and_said() -> None:
    _, fovs = with_feature("v03")
    kept = fovs[
        ((fovs["condition_id"] == "CTRL") & (fovs["repeat_group"] == "Day1"))
        | ((fovs["condition_id"] == "DrugA") & (fovs["repeat_group"] == "Day2"))
    ]
    experiment = read_experiment(EXPORTS / "v03", kept)
    assert experiment.pairs == ()
    (note,) = experiment.notes
    assert "DrugA: not paired with its control CTRL" in note and UNPAIRED in note


def test_a_condition_without_a_reference_is_said_not_compared() -> None:
    _, fovs = with_feature("v03")
    record = _record("v03")
    record["experiment"]["references"]["DrugA"] = None
    experiment = read_experiment(parse_coverage(record, "v03.json"), fovs)
    assert experiment.default_control() == {}
    assert "DrugA: not compared: the description declares no reference for it." in experiment.notes


def test_an_override_never_says_a_compared_condition_is_not_compared() -> None:
    # The notes are about the description's references; a user who declares a control compares
    # DrugA with it, so the report must not state "DrugA: not compared" without saying whose.
    objects, fovs = with_feature("v03")
    record = _record("v03")
    record["experiment"]["references"]["DrugA"] = None
    experiment = read_experiment(parse_coverage(record, "v03.json"), fovs)
    setup = described_setup(experiment, **FAST, control={"condition": "CTRL"})
    table = report(objects, fovs, setup).results["contrast"].table
    assert set(table["condition"]) == {"DrugA"} and set(table["control_condition"]) == {"CTRL"}
    lines = setup.declaration()
    assert not any(line.startswith("DrugA: not compared") for line in lines)
    assert (
        "By the description's references alone, DrugA: not compared: the description declares "
        "no reference for it." in lines
    )
    # Without an override the notes are what the report does, and are stated as they are.
    assert "DrugA: not compared: the description declares no reference for it." in (
        described_setup(experiment, **FAST).declaration()
    )


# --- Refusal 1: unknown versions, and an unexported set --------------------------------------


# Versions 1 and 2 are no longer read (compatibility sweep MC-1): only 3 is.
@pytest.mark.parametrize("version", [4, 2, 1, 0, "3", 3.0, True, None])
def test_an_unknown_coverage_version_is_refused(version) -> None:
    record = _record("v03")
    record["schema_version"] = version
    _refused(COVERAGE_VERSION_UNKNOWN, parse_coverage, record)


@pytest.mark.parametrize("version", [0, 2])
def test_an_unknown_description_version_is_refused(version) -> None:
    record = _record("v03")
    record["experiment"]["description_schema_version"] = version
    record["experiment"]["new_key_of_version_2"] = {}  # read before the shape
    _refused(DESCRIPTION_VERSION_UNKNOWN, parse_coverage, record)


def test_an_unexported_set_is_refused() -> None:
    record = {"schema_version": 3, "object_set": "Cells", "exported": False}
    _refused(NOT_EXPORTED, parse_coverage, record)


# --- Refusal 2: a described FOV table without the block --------------------------------------


def _blockless(name: str) -> dict:
    record = _record(name)
    record.pop("experiment")
    return record


def test_without_the_block_the_default_controls_are_refused() -> None:
    _, _, experiment = _experiment("v27", _blockless("v27"))
    assert experiment.block is None and experiment.described
    error = _refused(MISSING, described_setup, experiment, **FAST)
    assert "never guessed from the folder names" in str(error)


def test_without_the_block_controls_declared_by_id_are_accepted_and_said() -> None:
    objects, fovs, experiment = _experiment("v33", _blockless("v33"))
    setup = described_setup(experiment, **FAST, control={"condition": "ctrl"})
    assert setup.control == {"condition": "ctrl"}
    text = report(objects, fovs, setup).summary(detail="full")
    assert "Controls: declared by the user (ctrl); none is defaulted" in text


def test_without_the_block_the_scan_guesses_no_control_and_no_reagents_by_name() -> None:
    objects, fovs, experiment = _experiment("v33", _blockless("v33"))
    # The tables carry condition_id: no control is read from a name, with or without the
    # description (#631).
    assert scan_design(objects, fovs, feature_cols=["x"]).design["control_guess"] is None
    scan = scan_design(objects, fovs, feature_cols=["x"], description=experiment)
    assert scan.design["control_guess"] is None
    assert scan.design["control_guesses"] == [] and scan.design["reagent_groups"] == {}
    assert any(f.about == "controls" and "No control is defaulted" in f.text for f in scan.findings)
    _refused(MISSING, scan.setup, "x", features=("x",))
    assert scan.setup("x", control={"condition": "ctrl"}).control == {"condition": "ctrl"}


def test_an_undescribed_export_without_the_block_is_read_as_before() -> None:
    _, fovs = with_feature("v03")
    plain = fovs.drop(columns=["repeat_group", "condition_id", "technical_repeat_id"])
    experiment = read_experiment(parse_coverage(_blockless("v03")), plain)
    assert not experiment.described
    setup = described_setup(experiment, **FAST, control={"condition": "CTRL"})
    assert setup.described is None
    assert setup.declaration() == Setup(**FAST, control={"condition": "CTRL"}).declaration()


# --- Refusal 3: a malformed block --------------------------------------------------------------


def _set(path: tuple, value):
    def change(block: dict) -> None:
        target = block
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return change


_DELETE = object()
MALFORMED_CASES = {
    "missing key": _set(("suggestions",), _DELETE),
    "extra key": _set(("controls",), ["CTRL"]),
    "block not an object": None,
    "sha upper case": _set(("description_sha256",), "BD" + "0" * 62),
    "sha short": _set(("description_sha256",), "0" * 63),
    "digest not hex": _set(("section_digests", "repeats"), "z" * 64),
    "digests extra key": _set(("section_digests", "masks"), "0" * 64),
    "version not integer": _set(("description_schema_version",), "1"),
    "source folder not text": _set(("conditions", "CTRL", "source_folder"), 1),
    "condition extra key": _set(("conditions", "CTRL", "display_name"), "Control"),
    "reference unknown": _set(("references", "DrugA"), "MOCK"),
    "reference by case": _set(("references", "DrugA"), "ctrl"),
    "reference to itself": _set(("references", "DrugA"), "DrugA"),
    "references missing a condition": _set(("references", "CTRL"), _DELETE),
    "references unknown condition": _set(("references", "DrugB"), None),
    "repeat member unknown": _set(("repeat_groups", "Day1", "DrugB"), {"source_repeat": "r"}),
    "source repeat not text": _set(("repeat_groups", "Day1", "CTRL", "source_repeat"), None),
    "suggestions missing a condition": _set(("suggestions", "DrugA"), _DELETE),
    "suggestions unknown condition": _set(
        ("suggestions", "DrugB"), {"target": None, "treatment_key": None}
    ),
    "suggestion not text": _set(("suggestions", "DrugA", "target"), 5),
    "suggestion extra key": _set(("suggestions", "DrugA", "display_name"), "Drug A"),
}


@pytest.mark.parametrize("case", sorted(MALFORMED_CASES))
def test_a_malformed_block_is_refused(case) -> None:
    record = _record("v03")
    change = MALFORMED_CASES[case]
    if change is None:
        record["experiment"] = ["not", "an", "object"]
    else:
        change(record["experiment"])
    _refused(MALFORMED, parse_coverage, copy.deepcopy(record))


def test_invalid_json_is_refused(tmp_path: Path) -> None:
    (tmp_path / "coverage.json").write_text("{", encoding="utf-8")
    _refused(MALFORMED, read_coverage, tmp_path)


# --- Refusal 4: a block that disagrees with the FOV table ----------------------------------


def _changed(fovs: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    changed = fovs.copy()
    changed.loc[changed.index[0], column] = value
    return changed


@pytest.mark.parametrize(
    ("column", "value", "says"),
    [
        ("condition", "Ctrl", "whose source folder is 'CTRL', not 'Ctrl'"),
        ("condition_id", "MOCK", "the condition_id 'MOCK', which the description does not declare"),
        ("repeat_group", "Day9", "the repeat_group 'Day9', which the description does not"),
        ("biological_repeat", "rep9", "whose source repeat of 'CTRL' is 'rep1', not 'rep9'"),
    ],
)
def test_a_block_that_disagrees_with_the_fov_table_is_refused(column, value, says) -> None:
    _, fovs = with_feature("v03")
    fovs = fovs.sort_values(["condition", "biological_repeat"]).reset_index(drop=True)
    error = _refused(DISAGREES, read_experiment, EXPORTS / "v03", _changed(fovs, column, value))
    assert says in str(error)


def test_a_block_beside_a_fov_table_without_ids_is_refused() -> None:
    _, fovs = with_feature("v03")
    _refused(DISAGREES, read_experiment, EXPORTS / "v03", fovs.drop(columns=["condition_id"]))


def test_a_workflow_reading_another_fov_table_than_the_description_is_refused() -> None:
    objects, fovs, experiment = _experiment("v30")
    setup = described_setup(experiment, **FAST)
    # v30 pairs CTRL/rep1 with DrugA/rep2: the same folders paired by label are not the export.
    by_label = fovs.assign(repeat_group=fovs["biological_repeat"])
    _refused(DISAGREES, report, objects, by_label, setup)


# --- Refusal 5: two descriptions in one analysis ---------------------------------------------


def test_exports_of_two_descriptions_are_refused() -> None:
    _, fovs = with_feature("v03")
    error = _refused(TWO_DESCRIPTIONS, read_experiment, [EXPORTS / "v03", EXPORTS / "v30"], fovs)
    assert "bd5099bfe78b" in str(error) and "a60ac877d6e9" in str(error)
    # Two sets of one description (an object set and a patch set) are one description.
    twice = read_experiment([EXPORTS / "v03", read_coverage(EXPORTS / "v03")], fovs)
    assert twice.default_control() == {"control": {"condition": "CTRL"}}


def test_one_export_without_the_block_beside_one_with_it_defaults_nothing() -> None:
    _, fovs = with_feature("v03")
    blockless = parse_coverage(_blockless("v03"), "patches.json")
    experiment = read_experiment([EXPORTS / "v03", blockless], fovs)
    _refused(MISSING, described_setup, experiment, **FAST)


# --- Refusal 6: a suggestion is used only once confirmed (v27) ------------------------------


def test_v27_unconfirmed_suggestions_are_refused_for_the_analysis() -> None:
    objects, fovs, experiment = _experiment("v27")
    suggestions = experiment.target_suggestions
    assert dict(suggestions.suggested) == {"DrugA": "SRC", "DrugB": "src", "DrugC": "SRC"}
    _refused(UNCONFIRMED, Setup, **FAST, targets=suggestions)
    _refused(UNCONFIRMED, described_setup, experiment, **FAST, targets=suggestions)
    # Unconfirmed, nothing is grouped: no target, no agreement.
    contrast = report(objects, fovs, described_setup(experiment, **FAST)).results["contrast"]
    assert contrast.targets.empty and contrast.settings["target_col"] is None


def test_v27_the_confirmed_grouping_is_used_exactly_as_written() -> None:
    objects, fovs, experiment = _experiment("v27")
    setup = described_setup(experiment, **FAST, targets=experiment.target_suggestions.confirm())
    targets = report(objects, fovs, setup).results["contrast"].targets
    # 'SRC' and 'src' are two targets: DrugA and DrugC are grouped, DrugB is alone.
    assert list(targets["target"]) == ["SRC"]
    assert sorted(targets["reagents"].iloc[0]) == ["DrugA", "DrugC"]
    assert "Targets: as confirmed by the user: SRC (DrugA and DrugC); src (DrugB)." in (
        setup.declaration()
    )
    edited = experiment.target_suggestions.confirm({"DrugB": "SRC"})
    targets = report(objects, fovs, replace_targets(setup, edited)).results["contrast"].targets
    assert sorted(targets["reagents"].iloc[0]) == ["DrugA", "DrugB", "DrugC"]


def replace_targets(setup: Setup, targets: dict) -> Setup:
    from dataclasses import replace

    return replace(setup, targets=targets)


def test_v27_editing_the_target_changes_no_confirmed_grouping() -> None:
    objects, fovs, experiment = _experiment("v27")
    confirmed = experiment.target_suggestions.confirm()
    setup = described_setup(experiment, **FAST, targets=confirmed)
    before = report(objects, fovs, setup).results["contrast"].targets
    # The description's target changes (a new export) and the dict given is edited after.
    record = _record("v27")
    record["experiment"]["suggestions"]["DrugC"]["target"] = "YES1"
    changed = read_experiment(parse_coverage(record, "v27.json"), fovs)
    assert changed.target_suggestions.suggested["DrugC"] == "YES1"
    confirmed["DrugA"] = "YES1"
    assert setup.targets == {"DrugA": "SRC", "DrugB": "src", "DrugC": "SRC"}
    after = report(objects, fovs, setup).results["contrast"].targets
    pd.testing.assert_frame_equal(before, after)


def test_confirmed_targets_name_conditions_by_id() -> None:
    objects, fovs, experiment = _experiment("v33")
    setup = described_setup(experiment, **FAST, targets={"DrugA": "SRC", "control": "none"})
    with pytest.raises(PairingRefused, match="declare it as 'ctrl'"):
        report(objects, fovs, setup)
    with_target = fovs.assign(target="SRC")
    setup = described_setup(experiment, **FAST, targets={"DrugA": "SRC"})
    with pytest.raises(ValueError, match="give the reagent grouping once"):
        report(objects, with_target, setup)


# --- The scan reads the description, never the names -----------------------------------------


def test_v33_scan_suggests_the_descriptions_control_and_its_setup_defaults_it() -> None:
    objects, fovs, experiment = _experiment("v33")
    scan = scan_design(objects, fovs, feature_cols=["x"], description=experiment)
    assert scan.design["control_guess"] == "ctrl"
    assert "ctrl (the description's control; not declared)" in scan.summary()
    assert not any(q.key == "control" for q in scan.questions)
    found = [f.text for f in scan.findings if f.about == "controls"]
    assert found and "Controls from the experiment description (20aded4ab7ca)" in found[0]
    suggested = [s for s in scan.suggestions if "suggests targets" in s]
    assert suggested and "SRC: DrugA" in suggested[0]
    setup = scan.setup("x", features=("x",))
    assert setup.control == {"condition": "ctrl"} and setup.described.chosen is None
    result = report(scan.objects, scan.fovs, setup)
    assert set(result.results["contrast"].table["control_condition"]) == {"ctrl"}
