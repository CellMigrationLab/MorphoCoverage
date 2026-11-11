"""MorphoCoverage reads MorphoFeature's exports of the Spec fixtures and pairs them as described.

The golden exports (``tests/morphofeature_exports``, MC-0 of the Stage 2 plan) were written by
MorphoFeature's MF-5 for the vendored fixtures ``v03``, ``v27``, ``v30`` and ``v33``. Each records
the MorphoFeature commit and the git blob hash of its fixture, which must be the vendored one
(``tests/spec_fixtures``, pinned by ``test_spec_pin``). The FOV table carries the description's
``repeat_group``, ``condition_id`` and ``technical_repeat_id`` beside the unchanged key, whose
``condition`` and ``biological_repeat`` stay the source folder labels (experiment-v1 section 3.4):

- ``v03``: ``CTRL/rep1`` and ``DrugA/experiment_23`` are one repeat (``Day1``), so they pair;
- ``v30``: ``CTRL/rep1`` pairs with ``DrugA/rep2`` (``Day1``), never ``rep1`` with ``rep1``;
- ``v33``: the condition folder ``control`` has the ID ``ctrl``; the repeats are their labels;
- ``v27``: ``target`` and ``treatment_key`` are descriptive only: they are not in any table.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _helpers_morphofeature_exports import (
    EFFECT,
    EXPORTS,
    FEATURE,
    IDS,
    KEY,
    NAMES,
    SPEC,
    described_conditions,
    described_groups,
    git_blob_id,
    provenance,
    read_export,
    with_feature,
)

from morphocoverage import validate_input_tables
from morphocoverage._repeat_groups import (
    FOLDER_CONTROL,
    FOLDER_PAIRING,
    PairingRefused,
    paired_tables,
)
from morphocoverage.normalize import normalize_to_control
from morphocoverage.sampling_unit import (
    condition_contrast,
    feature_discovery,
    feature_ranking,
    object_rule_check,
)
from morphocoverage.workflows import PositiveControl, Setup, default_novelty, html_report, report

FAST = {"n_permutations": 99, "n_draws": 2, "object_counts": (2,)}


def test_every_fixture_has_an_export() -> None:
    assert sorted(p.name for p in EXPORTS.iterdir() if p.is_dir()) == list(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_each_export_records_its_morphofeature_commit_and_the_vendored_fixture(name) -> None:
    record = provenance(name)
    assert re.fullmatch(r"[0-9a-f]{40}", record["morphofeature_commit"])
    fixture = SPEC / record["fixture"]
    assert fixture.is_file()
    assert record["fixture_blob"] == git_blob_id(fixture.read_bytes())
    pin = tomllib.loads((SPEC / "PIN.toml").read_text(encoding="utf-8"))
    assert record["spec_commit"] == pin["commit"]
    assert record["format"] == "csv" and record["object_set"] == "Cells"


@pytest.mark.parametrize("name", NAMES)
def test_the_fov_table_keeps_the_folder_labels_beside_the_described_ids(name) -> None:
    objects, fovs, shape = read_export(name)
    assert list(fovs.columns) == KEY + IDS
    folders = fovs[["condition", "biological_repeat", "repeat_group"]].drop_duplicates()
    found = {(c, r): g for c, r, g in folders.itertuples(index=False)}
    assert found == described_groups(name)
    conditions = dict(fovs[["condition", "condition_id"]].drop_duplicates().itertuples(False))
    assert conditions == described_conditions(name)
    assert set(fovs["technical_repeat_id"]) == {"single"}
    # Only the FOV table carries the IDs; the feature table is the provider's area.
    for table in (objects, shape):
        assert not set(IDS) & set(table.columns)
    assert list(shape.columns) == [*KEY, "label_id", FEATURE]
    assert not {"target", "treatment_key"} & {*fovs.columns, *objects.columns}


@pytest.mark.parametrize("name", NAMES)
def test_morphocoverage_validates_each_export(name) -> None:
    objects, fovs = with_feature(name, x=False)
    validated = validate_input_tables(objects, fovs=fovs, feature_cols=[FEATURE])
    assert len(validated.fovs) == len(fovs)
    assert validated.n_observations == len(objects)


@pytest.mark.parametrize("name", NAMES)
def test_the_exports_are_paired_by_the_described_repeat_groups(name) -> None:
    objects, fovs = with_feature(name)
    paired_objects, paired_fovs, _, pairing = paired_tables(objects, fovs)
    groups = described_groups(name)
    assert pairing.renamed == {key: g for key, g in groups.items() if key[1] != g}
    ids = described_conditions(name)
    assert pairing.conditions == {folder: cid for folder, cid in ids.items() if folder != cid}
    # Every repeat group holds one folder of every condition: each pair is the described one;
    # the working tables name the conditions by their IDs.
    by_group = paired_fovs.groupby("repeat_group")["condition"].unique().map(sorted).to_dict()
    conditions = sorted(ids.values())
    assert by_group == dict.fromkeys(sorted(set(groups.values())), conditions)
    # The observations take their folder's group too (they hold no repeat_group of their own).
    keyed = paired_objects.merge(paired_fovs, on=KEY, how="left", validate="many_to_one")
    assert (keyed["biological_repeat"] == keyed["repeat_group"]).all()
    assert (paired_fovs["biological_repeat"] == paired_fovs["repeat_group"]).all()
    # The export itself is not changed.
    assert read_export(name)[1].equals(fovs)


def _contrast(objects: pd.DataFrame, fovs: pd.DataFrame, control: str):
    return condition_contrast(objects, fovs, feature="x", control={"condition": control}, **FAST)


@pytest.mark.parametrize("name", ["v03", "v30"])
def test_each_paired_repeat_sees_the_effect_not_another_repeats_batch(name) -> None:
    objects, fovs = with_feature(name)
    contrast = _contrast(objects, fovs, "CTRL")
    difference = contrast.table["mean"] - contrast.table["control_mean"]
    assert len(difference) == 2
    np.testing.assert_allclose(difference, EFFECT, atol=0.3)
    # Pairing by the folder labels instead would compare the wrong batches (v30), or nothing.
    if name == "v30":
        by_label = fovs.assign(repeat_group=fovs["biological_repeat"])
        wrong = _contrast(objects, by_label, "CTRL")
        wrong_difference = wrong.table["mean"] - wrong.table["control_mean"]
        assert (np.abs(wrong_difference - EFFECT) > 4).all()


def test_v03_is_compared() -> None:
    objects, fovs = with_feature("v03")
    discovery = feature_discovery(
        objects, fovs, features=["x", FEATURE], control={"condition": "CTRL"}, n_permutations=99
    )
    assert set(discovery.ranking["feature"]) >= {"x"}
    assert "Not compared" not in discovery.summary()


# MC-1: the canonical IDs are shared columns: repeats pair by ``repeat_group`` by default, the
# conditions are named by ``condition_id``, and every output names the folder labels beside.


def test_v30_pairs_ctrl_rep1_with_druga_rep2_and_names_both_folders() -> None:
    objects, fovs = with_feature("v30")
    table = _contrast(objects, fovs, "CTRL").table.set_index("biological_repeat")
    assert table.loc["Day1", "folder"] == "DrugA/rep2"
    assert table.loc["Day1", "control_folder"] == "CTRL/rep1"
    assert table.loc["Day2", "folder"] == "DrugA/rep1"
    assert table.loc["Day2", "control_folder"] == "CTRL/rep2"


def test_v33_names_the_control_by_its_id_and_its_folder_beside() -> None:
    objects, fovs = with_feature("v33")
    contrast = _contrast(objects, fovs, "ctrl")
    assert set(contrast.table["control_condition"]) == {"ctrl"}
    assert set(contrast.table["control_folder"]) == {"control/rep1", "control/rep2"}
    np.testing.assert_allclose(
        contrast.table["mean"] - contrast.table["control_mean"], EFFECT, atol=0.3
    )
    # The caller's tables keep the folder label.
    assert set(fovs["condition"]) == {"control", "DrugA"}


@pytest.mark.parametrize(
    "analysis",
    [
        lambda o, f, c: _contrast(o, f, c),
        lambda o, f, c: feature_ranking(o, f, features=["x"], control={"condition": c}),
        lambda o, f, c: object_rule_check(o, f, feature="x", control={"condition": c}, n_draws=2),
        lambda o, f, c: normalize_to_control(o, f, feature_cols=["x"], control={"condition": c}),
    ],
    ids=["contrast", "ranking", "object_rule", "normalize"],
)
def test_v33_a_control_named_by_its_folder_label_is_refused_with_its_id(analysis) -> None:
    objects, fovs = with_feature("v33")
    with pytest.raises(PairingRefused, match="declare it as 'ctrl'") as refused:
        analysis(objects, fovs, "control")
    assert refused.value.code == FOLDER_CONTROL


@pytest.mark.parametrize("name", ["v03", "v33"])
def test_pairing_by_folder_name_is_refused_while_repeat_group_exists(name) -> None:
    objects, fovs = with_feature(name)
    control = {"condition": "ctrl" if name == "v33" else "CTRL"}
    with pytest.raises(PairingRefused, match="pairs by folder name") as refused:
        condition_contrast(
            objects, fovs, feature="x", control=control, repeat_col="biological_repeat", **FAST
        )
    assert refused.value.code == FOLDER_PAIRING
    # The default pairs by repeat_group; naming it gives the same result.
    named = condition_contrast(
        objects, fovs, feature="x", control=control, repeat_col="repeat_group", **FAST
    )
    default = condition_contrast(objects, fovs, feature="x", control=control, **FAST)
    pd.testing.assert_frame_equal(named.table, default.table)


def test_v27_compares_every_drug_with_ctrl_and_reads_no_target() -> None:
    objects, fovs = with_feature("v27")
    contrast = _contrast(objects, fovs, "CTRL")
    assert sorted(set(contrast.table["condition"])) == ["DrugA", "DrugB", "DrugC"]
    assert "folder" not in contrast.table.columns  # every label is its ID
    assert contrast.settings["target_col"] is None


def _report(name: str, control: str):
    objects, fovs = with_feature(name)
    setup = Setup(readout="x", features=("x",), control={"condition": control})
    return report(objects, fovs, setup), objects, fovs


def test_the_report_names_conditions_by_id_with_the_folder_beside(tmp_path) -> None:
    result, objects, fovs = _report("v33", "ctrl")
    assert "contrast" in result.results
    assert set(result.results["contrast"].table["control_condition"]) == {"ctrl"}
    note = result.settings["repeat_pairing"]
    assert "ctrl (folder control)" in note
    assert "ctrl (folder control)" in result.summary(detail="full")
    page = html_report(result, objects, fovs, tmp_path / "page").read_text(encoding="utf-8")
    assert "ctrl (folder control)" in page
    with pytest.raises(PairingRefused, match="declare it as 'ctrl'"):
        _report("v33", "control")


def test_the_report_pairs_v03_and_names_each_folder_beside_its_group() -> None:
    result, _, _ = _report("v03", "CTRL")
    table = result.results["contrast"].table
    assert sorted(table["biological_repeat"]) == ["Day1", "Day2"]
    assert sorted(table["folder"]) == ["DrugA/experiment_23", "DrugA/rep2_drug"]
    assert "experiment_23 (DrugA) = Day1" in result.settings["repeat_pairing"]
    assert "Not compared" not in result.summary(detail="full")


def test_a_comparison_row_names_the_folder_of_its_treated_condition() -> None:
    # The rows of declared comparisons are labelled "DrugA vs CTRL"; their folder is still
    # the treated condition's folder, never the label joined to the repeat group.
    objects, fovs = with_feature("v03")
    table = condition_contrast(
        objects, fovs, feature="x", comparisons=[("DrugA", "CTRL")], **FAST
    ).table.set_index("biological_repeat")
    assert table.loc["Day1", "folder"] == "DrugA/experiment_23"
    assert table.loc["Day2", "folder"] == "DrugA/rep2_drug"
    assert table.loc["Day1", "control_folder"] == "CTRL/rep1"


def _renamed_v27() -> tuple[pd.DataFrame, pd.DataFrame]:
    """v27 with the folder ``DrugA`` given the ID ``DA`` (the other IDs are their labels)."""
    objects, fovs = with_feature("v27")
    fovs = fovs.copy()
    fovs.loc[fovs["condition"] == "DrugA", "condition_id"] = "DA"
    return objects, fovs


def test_a_setup_naming_an_experiment_by_folder_label_is_refused_with_its_id() -> None:
    # Keyed by folder label, the mapping would no longer match the conditions (named by their
    # IDs), and DrugA would be corrected in a family of its own without a word.
    objects, fovs = _renamed_v27()
    setup = Setup(
        readout="x",
        features=("x",),
        control={"condition": "CTRL"},
        experiment={"DrugA": "e1", "DrugB": "e1", "DrugC": "e2"},
    )
    with pytest.raises(PairingRefused, match="declare it as 'DA'") as refused:
        report(objects, fovs, setup)
    assert refused.value.code == FOLDER_CONTROL
    by_id = replace(setup, experiment={"DA": "e1", "DrugB": "e1", "DrugC": "e2"})
    cells = report(objects, fovs, by_id).results["contrast"].cells.set_index("condition")
    assert cells.loc["DA", "experiment"] == cells.loc["DrugB", "experiment"]
    assert cells.loc["DA", "experiment"] != cells.loc["DrugC", "experiment"]


def test_a_positive_control_named_by_folder_label_is_refused_with_its_id() -> None:
    # Selected on tables whose conditions are named by their IDs, the folder label would select
    # no field, and the positive control would be judged on nothing without a word.
    objects, fovs = _renamed_v27()
    objects = objects.assign(y=objects["x"] * 0.5 + objects["label_id"])
    declared = {
        "readout": "x",
        "features": ("x", "y"),
        "control": {"condition": "CTRL"},
        "novelty": default_novelty(["x", "y"]),
    }
    setup = Setup(**declared, positive=PositiveControl({"condition": "DrugA"}, fovs_seen=5))
    with pytest.raises(PairingRefused, match="declare it as 'DA'") as refused:
        report(objects, fovs, setup)
    assert refused.value.code == FOLDER_CONTROL
    by_id = Setup(**declared, positive=PositiveControl({"condition": "DA"}, fovs_seen=5))
    table = report(objects, fovs, by_id).results["controls"].table
    assert "positive_lofo" in set(table["candidate"])
