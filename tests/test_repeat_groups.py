"""Biological repeats are paired by their confirmed repeat group, never by folder label (#617).

MorphoBricks-Spec experiment-v1 section 3.4: the FOV key keeps the source folder label in
``biological_repeat`` and the canonical repeat ID is added beside it as ``repeat_group``;
pairing of conditions and every statement about biological N use ``repeat_group``, and a
consumer never falls back to the folder label. Fixture ``v03`` pairs repeats under different
folder names (``CTRL/rep1`` with ``DrugA/experiment_23``); ``v30`` gives one folder name to two
repeats (``rep1`` of ``CTRL`` in ``Day1``, ``rep1`` of ``DrugA`` in ``Day2``). Without the column,
or with an ambiguous one, a paired analysis is refused with a named reason, and the report runs
what needs no pairing and says why the conditions were not compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage import _repeat_groups as rg
from morphocoverage.normalize import normalize_to_control
from morphocoverage.retrospective import effect_preservation
from morphocoverage.sampling_unit import (
    condition_contrast,
    feature_discovery,
    feature_ranking,
    object_rule_check,
)
from morphocoverage.workflows import Setup, optimise, report

CONTROL = {"condition": "CTRL"}
FAST = {"n_permutations": 99}
# A large batch shift per repeat, so a pairing with the wrong repeat cannot go unseen.
BATCH = {"Day1": 0.0, "Day2": 6.0, "Day3": 12.0}
EFFECT = 1.5


def _tables(labels: dict[tuple[str, str], str], seed: int = 0):
    """CTRL and DrugA in three repeat groups; ``labels`` names each (condition, group)'s folder.

    Every repeat shifts both conditions by its batch; DrugA adds ``EFFECT`` in every repeat.
    """
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for group, batch in BATCH.items():
        for condition, effect in (("CTRL", 0.0), ("DrugA", EFFECT)):
            label = labels.get((condition, group), group)
            for well in ("w1", "w2"):
                for field in range(4):
                    name = f"{condition}_{label}_{well}_{field}.tif"
                    key = {
                        "condition": condition,
                        "biological_repeat": label,
                        "technical_repeat": well,
                        "image_file": name,
                    }
                    fovs.append({**key, "repeat_group": group})
                    offset = batch + effect + rng.normal(0, 0.2)
                    for _ in range(10):
                        objects.append({**key, "x": 20 + offset + rng.normal(), "y": rng.normal()})
    return pd.DataFrame(objects), pd.DataFrame(fovs)


# v03: each DrugA repeat sits in a folder named differently from its CTRL partner.
V03 = {("DrugA", "Day1"): "experiment_23", ("DrugA", "Day2"): "rep2_drug", ("DrugA", "Day3"): "x3"}
# v30: the folder names are crossed: DrugA's Day1 is in a folder called Day2, and so on.
V30 = {("DrugA", "Day1"): "Day2", ("DrugA", "Day2"): "Day3", ("DrugA", "Day3"): "Day1"}


def _contrast(objects, fovs):
    return condition_contrast(
        objects, fovs, feature="x", control=CONTROL, n_draws=2, object_counts=(10,), **FAST
    )


# The columns naming a row's repeat and folders, which differ when the folders are renamed.
_NAMES = ("biological_repeat", "folder", "control_folder")


def _numbers(contrast) -> pd.DataFrame:
    """The per-repeat rows and the test across repeats, without the repeat and folder labels."""
    columns = [c for c in contrast.table.columns if c not in _NAMES]
    return contrast.table[columns].reset_index(drop=True)


def _difference(contrast) -> np.ndarray:
    """Each paired repeat's treated mean minus its control mean."""
    return (contrast.table["mean"] - contrast.table["control_mean"]).to_numpy()


@pytest.fixture(scope="module")
def truth():
    """The contrast of the tables whose folder labels are their repeat groups."""
    return _contrast(*_tables({}))


@pytest.mark.parametrize("labels", [V03, V30], ids=["v03", "v30"])
def test_repeats_are_paired_by_repeat_group_whatever_the_folder_labels(labels, truth):
    contrast = _contrast(*_tables(labels))
    pd.testing.assert_frame_equal(_numbers(contrast), _numbers(truth))
    pd.testing.assert_frame_equal(contrast.repeats, truth.repeats)
    # Each paired row is named by its repeat group, its folders beside it.
    assert sorted(contrast.table["biological_repeat"]) == ["Day1", "Day2", "Day3"]
    for row in contrast.table.itertuples():
        assert row.folder == f"DrugA/{labels[('DrugA', row.biological_repeat)]}"
        assert row.control_folder == f"CTRL/{row.biological_repeat}"
    assert "folder" not in truth.table.columns  # nothing renamed: the table as before
    # Every repeat sees the effect: the batches were never mixed.
    assert _difference(contrast) == pytest.approx(EFFECT, abs=0.5)


def test_v30_is_not_paired_by_label():
    # Pairing the crossed folders by label compares DrugA of one repeat with CTRL of another:
    # the batch shifts (6 per repeat) then swamp the effect.
    objects, fovs = _tables(V30)
    by_label = fovs.assign(repeat_group=fovs["biological_repeat"])
    wrong = _contrast(objects, by_label)
    assert np.abs(_difference(wrong) - EFFECT).min() > 4
    right = _contrast(objects, fovs)
    assert _difference(right) == pytest.approx(EFFECT, abs=0.5)


def test_v03_is_compared_not_said_not_compared(truth):
    objects, fovs = _tables(V03)
    discovery = feature_discovery(objects, fovs, features=["x", "y"], control=CONTROL, **FAST)
    assert discovery.settings["uncompared_repeats"] == truth.settings.get(
        "uncompared_repeats", discovery.settings["uncompared_repeats"]
    )
    assert set(discovery.ranking["feature"]) == {"x", "y"}
    assert "Not compared" not in discovery.summary()


@pytest.mark.parametrize("labels", [V03, V30], ids=["v03", "v30"])
def test_the_ranking_and_the_normalisation_pair_by_repeat_group(labels):
    straight = _tables({})
    renamed = _tables(labels)
    ranked = [
        feature_ranking(*t, features=["x"], control=CONTROL, **FAST) for t in (straight, renamed)
    ]
    assert ranked[0].ranking.equals(ranked[1].ranking)
    normalised = [
        normalize_to_control(*t, feature_cols=["x"], control=CONTROL) for t in (straight, renamed)
    ]
    np.testing.assert_allclose(normalised[0]["x"], normalised[1]["x"])
    # The copy returned keeps the caller's folder labels.
    assert normalised[1]["biological_repeat"].tolist() == renamed[0]["biological_repeat"].tolist()


def test_the_callers_tables_are_not_changed():
    objects, fovs = _tables(V30)
    copies = objects.copy(), fovs.copy()
    _contrast(objects, fovs)
    pd.testing.assert_frame_equal(objects, copies[0])
    pd.testing.assert_frame_equal(fovs, copies[1])


def test_labels_that_are_their_groups_give_the_tables_unchanged():
    objects, fovs = _tables({})
    paired = rg.paired_tables(objects, fovs)
    assert paired[0] is objects and paired[1] is fovs and paired[3].renamed == {}
    assert rg.pairing_note(paired[3]) is None
    renamed = rg.paired_tables(*_tables(V03))[3]
    assert renamed.renamed[("DrugA", "experiment_23")] == "Day1"
    assert "experiment_23 (DrugA) = Day1" in rg.pairing_note(renamed)


# --- Refused, with the reason -----------------------------------------------------------------

PAIRED_ANALYSES = {
    "condition_contrast": lambda o, f: _contrast(o, f),
    "feature_ranking": lambda o, f: feature_ranking(o, f, features=["x"], control=CONTROL, **FAST),
    "feature_discovery": lambda o, f: feature_discovery(
        o, f, features=["x"], control=CONTROL, **FAST
    ),
    "normalize_to_control": lambda o, f: normalize_to_control(
        o, f, feature_cols=["x"], control=CONTROL
    ),
    "object_rule_check": lambda o, f: object_rule_check(
        o, f, feature="x", control=CONTROL, n_draws=50
    ),
    "effect_preservation": lambda o, f: effect_preservation(
        o, f, feature="x", control=CONTROL, fov_counts=[2], n_draws=5, n_splits=2
    ),
}


def _refused(code: str, objects: pd.DataFrame, fovs: pd.DataFrame, analysis=None) -> str:
    call = analysis or PAIRED_ANALYSES["condition_contrast"]
    with pytest.raises(rg.PairingRefused) as caught:
        call(objects, fovs)
    assert caught.value.code == code
    assert isinstance(caught.value, ValueError)
    return str(caught.value)


@pytest.mark.parametrize("name", sorted(PAIRED_ANALYSES))
def test_without_the_column_every_paired_analysis_is_refused(name):
    objects, fovs = _tables({})
    message = _refused(rg.ABSENT, objects, fovs.drop(columns="repeat_group"), PAIRED_ANALYSES[name])
    assert "'repeat_group'" in message and "never by the folder label" in message


def test_without_the_column_labels_that_happen_to_match_are_still_not_paired():
    # No fallback to the folder label, even when every label is shared.
    objects, fovs = _tables({})
    _refused(rg.ABSENT, objects, fovs.drop(columns="repeat_group"))


def test_a_folder_in_two_repeat_groups_is_refused():
    objects, fovs = _tables({})
    split = (fovs["condition"] == "CTRL") & (fovs["biological_repeat"] == "Day1")
    fovs.loc[split[split].index[:2], "repeat_group"] = "Day2"
    message = _refused(rg.SPLIT, objects, fovs)
    assert "'Day1' of condition 'CTRL' is in 2 repeat groups ('Day1', 'Day2')" in message


def test_a_repeat_group_holding_two_folders_of_one_condition_is_refused():
    objects, fovs = _tables({})
    merged = (fovs["condition"] == "CTRL") & (fovs["repeat_group"] == "Day3")
    fovs.loc[merged, "repeat_group"] = "Day2"
    message = _refused(rg.MERGED, objects, fovs)
    assert "'Day2' holds 2 folders of condition 'CTRL' ('Day2', 'Day3')" in message


def test_a_field_without_a_repeat_group_is_refused():
    objects, fovs = _tables({})
    fovs.loc[fovs.index[5], "repeat_group"] = ""
    assert "1 of the 48 fields have no 'repeat_group' value" in _refused(rg.EMPTY, objects, fovs)
    fovs.loc[fovs.index[5], "repeat_group"] = np.nan
    _refused(rg.EMPTY, objects, fovs)


def test_an_observation_table_that_disagrees_is_refused():
    objects, fovs = _tables({})
    groups = fovs.set_index("image_file")["repeat_group"]
    objects["repeat_group"] = objects["image_file"].map(groups)
    _contrast(objects, fovs)  # agreeing copies are fine
    objects.loc[objects["biological_repeat"] == "Day1", "repeat_group"] = "Day3"
    _refused(rg.DISAGREES, objects, fovs)


def test_a_repeat_group_without_the_folder_label_is_refused():
    objects, fovs = _tables({})
    _refused(
        rg.NO_SOURCE,
        objects.drop(columns="biological_repeat"),
        fovs.drop(columns="biological_repeat"),
    )


def test_an_explicit_pooling_or_another_declared_column_is_the_callers_choice():
    objects, fovs = _tables(V30)
    fovs = fovs.drop(columns="repeat_group")
    pooled = condition_contrast(
        objects, fovs, feature="x", control=CONTROL, repeat_col=None, n_draws=2, **FAST
    )
    assert len(pooled.table) == 1
    # A column of the caller's own, named explicitly, is its declaration of the repeats.
    groups = _tables({})[1]["repeat_group"].to_numpy()
    own = condition_contrast(
        objects.merge(fovs.assign(day=groups)[["image_file", "day"]]),
        fovs.assign(day=groups),
        feature="x",
        control=CONTROL,
        repeat_col="day",
        group_cols=["condition", "biological_repeat", "technical_repeat", "day"],
        n_draws=2,
        **FAST,
    )
    assert _difference(own) == pytest.approx(EFFECT, abs=0.5)


def test_tables_without_any_repeat_column_are_one_repeat_as_before():
    objects, fovs = _tables({})
    one = (fovs["repeat_group"] == "Day1").to_numpy()
    kept = fovs.loc[one, "image_file"]
    result = _contrast(
        objects[objects["image_file"].isin(kept)].drop(columns="biological_repeat"),
        fovs[one].drop(columns=["biological_repeat", "repeat_group"]),
    )
    assert len(result.table) == 1


# --- The workflows ----------------------------------------------------------------------------


def _setup(**changes) -> Setup:
    return Setup(**{"readout": "x", "features": ("x", "y"), "control": CONTROL, **changes})


def test_the_report_pairs_by_repeat_group():
    straight = report(*_tables({}), _setup())
    crossed = report(*_tables(V30), _setup())
    pd.testing.assert_frame_equal(
        _numbers(crossed.results["contrast"]), _numbers(straight.results["contrast"])
    )
    assert "repeat_pairing" not in straight.settings
    assert "Day2 (DrugA) = Day1" in crossed.settings["repeat_pairing"]


def test_the_report_without_the_column_runs_what_needs_no_pairing_and_says_why():
    objects, fovs = _tables({})
    result = report(objects, fovs.drop(columns="repeat_group"), _setup())
    assert "contrast" not in result.results and "discovery" not in result.results
    assert {"overview", "variability", "inspection"} <= set(result.results)
    for section in ("unit", "profile", "conditions"):
        reason = result.not_assessed[section]
        assert reason.startswith(f"not compared ({rg.ABSENT}): the FOV table has no")
        assert "cannot be paired" in reason
    text = result.summary(detail="full")
    assert "cannot be paired" in text and "never by the folder label" in text
    # The declaration printed is the one given.
    assert result.setup.control == CONTROL


def test_the_report_with_an_ambiguous_column_says_which():
    objects, fovs = _tables({})
    merged = (fovs["condition"] == "CTRL") & (fovs["repeat_group"] == "Day3")
    fovs.loc[merged, "repeat_group"] = "Day2"
    result = report(objects, fovs, _setup())
    assert rg.MERGED in result.not_assessed["conditions"]


def test_a_setup_that_compares_nothing_reads_the_tables_as_given():
    objects, fovs = _tables({})
    result = report(objects, fovs.drop(columns="repeat_group"), Setup(readout="x"))
    assert "cannot be paired" not in result.summary(detail="full")


def test_the_planning_workflow_refuses_a_target_it_cannot_pair():
    objects, fovs = _tables({})
    with pytest.raises(rg.PairingRefused, match="repeat_group"):
        optimise(
            objects,
            fovs.drop(columns="repeat_group"),
            _setup(normalize="control_mean"),
            fov_counts=[2],
            reservoir_fraction=0.5,
        )


def test_a_table_without_the_condition_column_keeps_each_folder_in_its_own_group():
    # DrugA's Day1 and Day2 repeats are in swapped folders; CTRL's folders are their groups.
    # A table without the condition column cannot tell CTRL's "Day1" (group Day1) from
    # DrugA's "Day1" (group Day2): refused, never relabelled by DrugA's renaming.
    objects, fovs = _tables({("DrugA", "Day1"): "Day2", ("DrugA", "Day2"): "Day1"})
    with pytest.raises(rg.PairingRefused) as caught:
        rg.paired_tables(objects, fovs, others=[objects.drop(columns="condition")])
    assert caught.value.code == rg.SPLIT
    assert "in different repeat groups" in str(caught.value)


def test_the_report_names_each_folder_beside_its_repeat_group():
    # experiment-v1 section 3.4: the source label is never replaced in what the report says.
    result = report(*_tables(V03), _setup())
    text = result.summary(detail="full")
    assert "experiment_23 (DrugA) = Day1" in " ".join(text.split())
    assert "not by folder label" not in report(*_tables({}), _setup()).summary(detail="full")


def test_the_benchmark_and_the_distributions_refuse_without_the_column():
    from morphocoverage.sampling_unit.distribution import distribution_contrast
    from morphocoverage.workflows import benchmark

    objects, fovs = _tables({})
    absent = fovs.drop(columns="repeat_group")
    _refused(
        rg.ABSENT,
        objects,
        absent,
        lambda o, f: distribution_contrast(o, f, features=["x"], control=CONTROL),
    )
    with pytest.raises(rg.PairingRefused, match="repeat_group"):
        benchmark(objects, absent, _setup(normalize="control_mean"), reservoir_fraction=0.5)


def test_a_feature_table_carrying_the_repeat_group_is_refused_as_metadata():
    from _helpers_feature_tables import _tables as _feature_tables

    from morphocoverage.feature_tables import attach_feature_tables

    objects, fovs, embedding = _feature_tables()
    embedding = embedding.assign(repeat_group=embedding["biological_repeat"])
    with pytest.raises(ValueError, match=r"metadata columns \['repeat_group'\]"):
        attach_feature_tables(objects, fovs, {"DINO": embedding})


# Conditions named by their condition_id (MC-1): the folder label stays in `condition`.
IDS = {"CTRL": "ctrl", "DrugA": "drug_a"}


def _with_ids(table: pd.DataFrame, ids: dict = IDS) -> pd.DataFrame:
    return table.assign(condition_id=table["condition"].map(ids))


def test_the_canonical_ids_are_shared_columns_and_their_near_misses_are_refused():
    from morphocoverage.metadata import MORPHOBRICKS_SHARED_COLUMNS, canonical_spelling

    assert {"repeat_group", "condition_id", "technical_repeat_id"} <= set(
        MORPHOBRICKS_SHARED_COLUMNS
    )
    assert canonical_spelling("Condition_ID") == "condition_id"
    assert canonical_spelling("repeat group") == "repeat_group"
    assert canonical_spelling("technical_repeat_ID") == "technical_repeat_id"
    assert canonical_spelling("condition_id") is None
    assert canonical_spelling("technical_repeat_id") is None


def test_conditions_are_named_by_their_id_and_the_contrast_matches_the_folder_labels():
    objects, fovs = _tables(V03)
    by_id = condition_contrast(
        _with_ids(objects),
        _with_ids(fovs),
        feature="x",
        control={"condition": "ctrl"},
        n_draws=2,
        object_counts=(10,),
        **FAST,
    )
    by_label = _contrast(objects, fovs)
    assert set(by_id.table["condition"]) == {"drug_a"}
    assert set(by_id.table["control_condition"]) == {"ctrl"}
    assert sorted(by_id.table["folder"]) == ["DrugA/experiment_23", "DrugA/rep2_drug", "DrugA/x3"]
    # The same pairs and values (the random draws are seeded by the condition's name).
    columns = ["fields", "observations", "control_fields", "control_observations", "mean"]
    columns += ["control_mean", "d_objects"]
    pd.testing.assert_frame_equal(
        by_id.table[columns].reset_index(drop=True), by_label.table[columns].reset_index(drop=True)
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda f: f.assign(condition_id=f["condition_id"].where(f.index > 0, "")), rg.ID_EMPTY),
        (
            lambda f: f.assign(condition_id=f["condition_id"].where(f.index > 0, "other")),
            rg.ID_SPLIT,
        ),
        (lambda f: f.assign(condition_id="one"), rg.ID_MERGED),
        (lambda f: f.drop(columns="condition"), rg.ID_NO_SOURCE),
    ],
    ids=["empty", "split", "merged", "no_condition"],
)
def test_an_ambiguous_condition_id_is_refused(change, code):
    objects, fovs = _tables({})
    with pytest.raises(rg.PairingRefused) as caught:
        rg.paired_tables(objects, change(_with_ids(fovs)))
    assert caught.value.code == code


def test_an_observation_table_disagreeing_on_the_condition_id_is_refused():
    objects, fovs = _tables({})
    wrong = _with_ids(objects, {"CTRL": "drug_a", "DrugA": "ctrl"})
    with pytest.raises(rg.PairingRefused) as caught:
        rg.paired_tables(wrong, _with_ids(fovs))
    assert caught.value.code == rg.ID_DISAGREES


def test_a_folder_label_that_is_another_conditions_id_is_that_id():
    # The IDs are crossed: the folder CTRL is condition "DrugA" and the reverse.
    objects, fovs = _tables({})
    crossed = {"CTRL": "DrugA", "DrugA": "CTRL"}
    paired = rg.paired_tables(_with_ids(objects, crossed), _with_ids(fovs, crossed), names=["CTRL"])
    assert paired[3].conditions == crossed
    assert set(paired[1].loc[fovs["condition"] == "DrugA", "condition"]) == {"CTRL"}


def test_tables_without_the_ids_are_unchanged_and_need_no_id():
    objects, fovs = _tables({})
    paired = rg.paired_tables(objects, fovs, names=["CTRL"])
    assert paired[0] is objects and paired[1] is fovs
    assert paired[3] == rg.Pairing({})
    assert rg.pairing_note(paired[3]) is None


def test_a_feature_table_carrying_a_condition_id_is_refused_as_metadata():
    from _helpers_feature_tables import _tables as _feature_tables

    from morphocoverage.feature_tables import attach_feature_tables

    objects, fovs, embedding = _feature_tables()
    embedding = embedding.assign(condition_id=embedding["condition"])
    with pytest.raises(ValueError, match=r"metadata columns \['condition_id'\]"):
        attach_feature_tables(objects, fovs, {"DINO": embedding})


def test_a_setup_naming_the_control_by_its_folder_is_refused_by_the_report():
    objects, fovs = _tables({})
    with pytest.raises(rg.PairingRefused, match="declare it as 'ctrl'"):
        report(_with_ids(objects), _with_ids(fovs), _setup())
    result = report(_with_ids(objects), _with_ids(fovs), _setup(control={"condition": "ctrl"}))
    assert set(result.results["contrast"].table["control_condition"]) == {"ctrl"}
    assert "ctrl (folder CTRL)" in result.settings["repeat_pairing"]
