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
from morphocoverage._repeat_groups import paired_tables
from morphocoverage.sampling_unit import condition_contrast, feature_discovery

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
    # Every repeat group holds one folder of every condition: each pair is the described one.
    by_group = paired_fovs.groupby("repeat_group")["condition"].unique().map(sorted).to_dict()
    conditions = sorted(set(fovs["condition"]))
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
