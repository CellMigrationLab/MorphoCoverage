"""Regressions for the fourth audit, group B (input traps).

One test per finding, ``test_<id>_...``. Where a finding is about what the
tables accept, the test also checks that ``check_tables`` and
``validate_input_tables`` agree (``_agree``).
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage import check_tables, planes_as_features, validate_input_tables
from morphocoverage.benchmark import permuted_fovs
from morphocoverage.metadata import canonical_spelling, sorted_by_key
from morphocoverage.plots import _layout
from morphocoverage.reservoir import ReservoirSplit
from morphocoverage.retrospective import dataset_overview
from morphocoverage.workflows import Setup
from morphocoverage.workflows._shared import check_workflow_inputs


def _tables(n_repeats=2, wells=("B02", "B03"), fields=3, cells=4, seed=0):
    """Cells in fields of two conditions: condition > repeat > well > field > cell."""
    rng = np.random.default_rng(seed)
    fov_rows, cell_rows = [], []
    for condition, row in (("control", "B"), ("drug", "C")):
        for r in range(n_repeats):
            repeat = f"2026_01_0{r + 1}"
            for well in wells:
                well = row + well[1:]
                for s in range(fields):
                    image = f"{repeat}_{well}_s{s + 1:02d}.tif"
                    ids = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "technical_repeat": well,
                        "image_file": image,
                    }
                    fov_rows.append(ids)
                    for k in range(cells):
                        cell_rows.append({**ids, "label_id": k + 1, "area": rng.normal(100, 10)})
    return pd.DataFrame(cell_rows), pd.DataFrame(fov_rows)


def _agree(objects, fovs, **arguments):
    """Run both; ``check_tables`` never raises and accepts exactly what the validator accepts."""
    arguments = {"feature_cols": ["area"], "observation_unit": "cell", **arguments}
    check = check_tables(objects, fovs, **arguments)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_input_tables(objects, fovs=fovs, **arguments)
        accepted = True
    except (ValueError, TypeError):
        accepted = False
    assert check.ok == accepted, check.summary()
    return accepted, check


def _titles(check, level="warning"):
    return [p.title for p in check.problems if p.level == level]


def test_p5_1_rows_of_several_planes_in_one_field_are_refused():
    cells, fovs = _tables()
    stack = pd.concat([cells.assign(z=z) for z in range(5)], ignore_index=True)
    accepted, check = _agree(stack, fovs)
    assert not accepted
    message = _titles(check, "error")[0]
    assert "several 'z' values" in message and "planes_as_features" in message
    assert "observation_unit='slice'" in message
    # Undeclared (assumed objects) is refused too; a plane-level unit is what the rows are.
    assert not _agree(stack, fovs, observation_unit=None)[0]
    data = validate_input_tables(stack, fovs=fovs, feature_cols=["area"], observation_unit="slice")
    assert len(data.objects) == 5 * len(cells)
    # CellPatcher's per-slice export: z0/z1 per plane, one image_file.
    patches = pd.concat([cells.assign(z0=z, z1=z + 1) for z in range(3)], ignore_index=True)
    accepted, check = _agree(patches, fovs, observation_unit="patch")
    assert not accepted and "several 'z0' values" in _titles(check, "error")[0]
    # One plane per field is fine, and so is the pivot's one row per object.
    assert _agree(cells.assign(z=0), fovs)[0]
    wide, names = planes_as_features(stack, fovs, features=["area"])
    assert _agree(wide, fovs, feature_cols=names)[0]


def test_p5_2_plane_synonyms_and_plane_tokens_in_file_names():
    for name, meant in (
        ("Metadata_Z", "z"),
        ("Metadata_T", "t"),
        ("plane", "z"),
        ("Slice", "z"),
        ("frame", "t"),
        ("Timepoint", "t"),
        ("channel", "c"),
        ("Metadata_Channel", "c"),
    ):
        assert canonical_spelling(name) == meant, name
    for name in ("Metadata_Well", "Metadata_Plate", "time", "position", "planes_z"):
        assert canonical_spelling(name) is None, name
    cells, fovs = _tables()
    per_plane = pd.concat([cells.assign(Metadata_Z=z) for z in range(3)], ignore_index=True)
    per_plane["image_file"] = (
        per_plane["image_file"].str.replace(".tif", "") + "_z" + per_plane["Metadata_Z"].astype(str)
    ) + ".tif"
    files = per_plane[
        ["condition", "biological_repeat", "technical_repeat", "image_file"]
    ].drop_duplicates()
    accepted, check = _agree(per_plane, files)
    assert not accepted and "'z'" in _titles(check, "error")[0]
    # The plane only in the file name: 3 files per position counted as fields, with a warning.
    only_names = per_plane.drop(columns="Metadata_Z")
    accepted, check = _agree(only_names, files)
    assert accepted and any("plane, channel or frame token" in t for t in _titles(check))
    with pytest.warns(UserWarning, match="plane, channel or frame token"):
        validate_input_tables(only_names, fovs=files, feature_cols=["area"])
    # Well C02 in a file name is not a channel token.
    assert not any("token" in t for t in _titles(check_tables(cells, fovs, feature_cols=["area"])))


def test_p5_5_check_warnings_reach_the_workflows_and_the_overview():
    cells, fovs = _tables(n_repeats=1)
    copied = pd.concat([cells, cells], ignore_index=True)  # the cell file read twice
    # One drug well typed 'drug ' in both tables.
    for table in (copied, fovs):
        table.loc[table["technical_repeat"] == "C03", "condition"] = "drug "
    setup = Setup(readout="area", control={"condition": "control"}, observation_unit="cell")
    with pytest.warns(UserWarning, match="check_tables found"):
        _, _, check = check_workflow_inputs(copied, fovs, setup)
    titles = _titles(check)
    assert any("exact copy" in t for t in titles)
    assert any("differ only in spaces" in t for t in titles)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, quiet = check_workflow_inputs(copied, fovs, setup, warn=False)
    result = dataset_overview(
        copied, fovs, feature_cols=["area"], observation_unit="cell", data_check=quiet
    )
    overview = result.summary(detail="full")
    assert "The data check found" in overview and "exact copy" in overview
    # Labels that print alike are quoted, in the sentence and in the (aligned) table.
    assert "'drug', 'drug '" in overview and re.search(r"'drug ' +2026_01_01", overview)
    short = " ".join(result.summary().split())
    assert "⚠️ Data checks: 2 warnings (" in short and "exact copy" in short
    assert "'drug', 'drug '" in short


def test_p5_7_undeclared_group_columns_are_warned_about():
    cells, fovs = _tables()
    assert canonical_spelling("Metadata_condition") == "condition"
    assert canonical_spelling("Metadata_Repeat") == "biological_repeat"
    renamed = {"biological_repeat": "replicate"}
    accepted, check = _agree(cells.rename(columns=renamed), fovs.rename(columns=renamed))
    assert accepted
    assert any("'replicate' looks like a group column" in t for t in _titles(check))
    # Declared, or a column that only restates a group (a plate per repeat), says nothing.
    accepted, check = _agree(
        cells.rename(columns=renamed),
        fovs.rename(columns=renamed),
        group_cols=["condition", "replicate", "technical_repeat"],
    )
    assert not any("looks like a group" in t for t in _titles(check))
    plate = fovs.assign(
        plate=fovs["biological_repeat"].map({"2026_01_01": "P1", "2026_01_02": "P2"})
    )
    assert not any("looks like a group" in t for t in _titles(_agree(cells, plate)[1]))
    # A file name, one per field, is not a group.
    masks = fovs.assign(mask_name="m_" + fovs["image_file"])
    assert not any("looks like a group" in t for t in _titles(_agree(cells, masks)[1]))


def test_p5_8_several_stage_positions_in_one_field_are_refused():
    cells, fovs = _tables()
    for table in (cells, fovs):
        table["series"] = table["image_file"]
        table["image_file"] = table["biological_repeat"] + "_" + table["technical_repeat"] + ".nd2"
    fovs = fovs.drop_duplicates(
        ["condition", "biological_repeat", "technical_repeat", "image_file"]
    )
    fovs = fovs.drop(columns="series")
    accepted, check = _agree(cells, fovs)
    assert not accepted
    assert "several 'series' values" in _titles(check, "error")[0]
    assert any("multi-position files" in t for t in _titles(check))
    # With position_id the files hold their positions apart.
    cells["position_id"] = cells["image_file"] + "#" + cells["series"]
    listed = cells[
        ["condition", "biological_repeat", "technical_repeat", "position_id"]
    ].drop_duplicates()
    assert _agree(cells, listed)[0]


def test_p5_9_overview_names_the_declared_groups():
    cells, fovs = _tables()
    renamed = {"condition": "Treatment", "biological_repeat": "Plate", "technical_repeat": "Well"}
    summary = dataset_overview(
        cells.rename(columns=renamed),
        fovs.rename(columns=renamed),
        feature_cols=["area"],
        group_cols=["Treatment", "Plate", "Well"],
        observation_unit="cell",
    )
    full = summary.summary(detail="full")
    assert "all fields pooled" not in full
    assert "the analysis groups are Treatment, Plate, Well (8 groups), each analysed apart" in full
    short = " ".join(summary.summary().split())
    assert "8 analysis groups (Treatment · Plate · Well), each analysed apart" in short


def test_p5_10_planes_as_features_missing_planes_and_cellpatcher_z0():
    rows = []
    for position in ("p0", "p1"):
        depth = 3 if position == "p0" else 4  # stacks of different depth
        for label in (1, 2):
            planes = range(depth) if label == 1 else range(1, 3)  # label 2 spans two planes
            rows += [
                {
                    "condition": "A",
                    "biological_repeat": "r1",
                    "position_id": position,
                    "label_id": label,
                    "z": z,
                    "area": 10.0 * label + z,
                }
                for z in planes
            ]
    stack = pd.DataFrame(rows)
    fovs = stack[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    with pytest.raises(ValueError, match="allow_missing_planes=True"):
        planes_as_features(stack, fovs, features=["area"])
    wide, names = planes_as_features(stack, fovs, features=["area"], allow_missing_planes=True)
    assert names == ["area_z0", "area_z1", "area_z2", "area_z3"]
    assert len(wide) == 4 and int(wide[names].isna().sum().sum()) == 16 - 11
    assert wide.loc[(wide.position_id == "p0") & (wide.label_id == 1), "area_z3"].isna().all()
    # CellPatcher's z0 is the z axis; a patch spanning planes is refused.
    patches = stack.drop(columns="z").assign(z0=stack["z"], z1=stack["z"] + 1)
    wide_z0, names_z0 = planes_as_features(
        patches, fovs, features=["area"], allow_missing_planes=True
    )
    assert names_z0 == names and wide_z0[names].equals(wide[names])
    with pytest.raises(ValueError, match="span several planes"):
        planes_as_features(patches.assign(z1=patches["z0"] + 2), fovs, features=["area"])
    # Text against numbers is named as such; a label such as 'max' stays a plane.
    mixed = stack.astype({"z": object})
    mixed.loc[mixed.position_id == "p0", "z"] = mixed.loc[mixed.position_id == "p0", "z"].map(str)
    with pytest.raises(ValueError, match="text in some rows and a number in others"):
        planes_as_features(mixed, fovs, features=["area"], allow_missing_planes=True)
    with pytest.raises(TypeError, match="True or False"):
        planes_as_features(stack, fovs, features=["area"], allow_missing_planes="yes")


def test_p5_11_wells_without_any_observation_are_warned_about():
    cells, fovs = _tables()
    plate_map = pd.concat(
        [fovs, fovs[fovs.technical_repeat == "C02"].assign(technical_repeat="C09")],
        ignore_index=True,
    )
    plate_map.loc[plate_map.technical_repeat == "C09", "image_file"] = "never_" + plate_map.loc[
        plate_map.technical_repeat == "C09", "image_file"
    ].astype(str)
    accepted, check = _agree(cells, plate_map)
    assert accepted
    assert "2 technical repeats of the FOV table hold no observation in any field." in _titles(
        check
    )
    # One empty field in an imaged well is only an empty field.
    one_empty = cells[cells.image_file != fovs.image_file.iloc[0]]
    assert not any("technical repeat" in t for t in _titles(_agree(one_empty, fovs)[1]))


def test_p5_12_zero_padding_and_one_well_in_two_conditions():
    cells, fovs = _tables()
    for table in (cells, fovs):
        second = table["biological_repeat"] == "2026_01_02"
        table.loc[second, "technical_repeat"] = table.loc[second, "technical_repeat"].str.replace(
            "0", ""
        )
    accepted, check = _agree(cells, fovs)
    assert accepted and any("zero padding" in t for t in _titles(check))
    cells, fovs = _tables()
    for table in (cells, fovs):
        table.loc[table.technical_repeat == "C02", "technical_repeat"] = "B02"
    accepted, check = _agree(cells, fovs)
    assert accepted
    assert any("plate well is named in several conditions" in t for t in _titles(check))
    # Dishes that restart in each condition are not plate wells.
    cells, fovs = _tables(wells=("d1", "d2"))
    for table in (cells, fovs):
        table["technical_repeat"] = "d" + table["technical_repeat"].str[-1]
    assert set(fovs["technical_repeat"]) == {"d1", "d2"}
    assert not any("plate well" in t for t in _titles(_agree(cells, fovs)[1]))


def test_p2_13_invalid_cellpatcher_rows_are_warned_about():
    cells, fovs = _tables()
    cells["valid"] = True
    cells.loc[:4, "valid"] = False
    cells["reason"] = np.where(cells["valid"], "", "patch outside image")
    with pytest.warns(UserWarning, match="5 observations have valid=False"):
        validate_input_tables(cells, fovs=fovs, feature_cols=["area"], observation_unit="patch")
    accepted, check = _agree(cells, fovs, observation_unit="patch")
    assert accepted and any("valid=False" in t for t in _titles(check))
    assert any("patch outside image" in t for t in _titles(check))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_input_tables(
            cells.assign(valid=True), fovs=fovs, feature_cols=["area"], observation_unit="patch"
        )


def test_p9_3_category_order_does_not_change_orders_or_reservoirs():
    cells, fovs = _tables(n_repeats=1, fields=8)
    one = fovs[fovs.technical_repeat == "B02"].reset_index(drop=True)
    key = ["condition", "biological_repeat", "technical_repeat", "image_file"]
    group = {"condition": "control", "biological_repeat": "2026_01_01", "technical_repeat": "B02"}
    reordered = one.assign(
        image_file=pd.Categorical(one["image_file"], categories=sorted(one["image_file"])[::-1])
    )
    plain = permuted_fovs(one, key, 0, group, 1)["image_file"].tolist()
    categorical = permuted_fovs(reordered, key, 0, group, 1)["image_file"].astype(str).tolist()
    assert plain == categorical
    assert sorted_by_key(reordered, key)["image_file"].astype(str).tolist() == sorted(
        one["image_file"]
    )
    data = validate_input_tables(
        cells[cells.technical_repeat == "B02"], fovs=one, feature_cols=["area"]
    )
    data_cat = validate_input_tables(
        cells[cells.technical_repeat == "B02"].assign(
            image_file=lambda t: pd.Categorical(
                t["image_file"], categories=reordered["image_file"].cat.categories
            )
        ),
        fovs=reordered,
        feature_cols=["area"],
    )
    split = ReservoirSplit(0.25, seed=3, block_col=None)
    held = split.split(group, data)[1].fovs["image_file"].tolist()
    held_cat = split.split(group, data_cat)[1].fovs["image_file"].astype(str).tolist()
    assert held == held_cat


def test_p9_9_index_levels_named_like_columns_are_accepted():
    cells, fovs = _tables()
    indexed = cells.set_index(["condition", "biological_repeat"], drop=False)
    listed = fovs.set_index(["condition", "biological_repeat"], drop=False)
    data = validate_input_tables(indexed, fovs=listed, feature_cols=["area"])
    # The caller's labels are kept; only the clashing level names are dropped.
    assert data.objects.index.tolist() == indexed.index.tolist()
    summary = dataset_overview(indexed, listed, feature_cols=["area"]).summary()
    assert "2 conditions" in summary
    assert list(indexed.index.names) == [
        "condition",
        "biological_repeat",
    ]  # the input is not changed


def test_p9_10_control_first_then_sorted_whatever_the_row_order():
    cells, fovs = _tables()
    cells = cells.assign(condition=cells["condition"].map({"control": "mock", "drug": "a_drug"}))
    fovs = fovs.assign(condition=fovs["condition"].map({"control": "mock", "drug": "a_drug"}))
    shuffled = fovs.sample(frac=1, random_state=7)
    shuffled_cells = cells.sample(frac=1, random_state=8)
    overview = dataset_overview(
        shuffled_cells, shuffled, feature_cols=["area"], control={"condition": "mock"}
    )
    assert overview.layout["condition"].tolist()[:2] == ["mock", "mock"]
    assert "(condition: mock, a_drug)" in overview.summary(detail="full")
    assert "(biological_repeat: 2026_01_01, 2026_01_02)" in overview.summary(detail="full")
    short = " ".join(overview.summary().split())
    assert "2 conditions (mock, a_drug), 2 biological repeats (2026_01_01, 2026_01_02)" in short
    data = validate_input_tables(shuffled_cells, fovs=shuffled, feature_cols=["area"])
    layout = _layout(data, "condition", "biological_repeat", None, 6, "mock")
    assert layout.conditions == ["mock", "a_drug"]
    assert layout.repeats == ["2026_01_01", "2026_01_02"]
    assert _layout(data, "condition", "biological_repeat", None, 6).conditions == ["a_drug", "mock"]
