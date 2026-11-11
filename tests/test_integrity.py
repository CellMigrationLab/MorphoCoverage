"""Data-integrity checks (issue #49): each finds the copy it is for, and nothing on clean data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_feature_tables import _cells, _table
from _helpers_integrity import FEATURES, _check, _field, _replace_field
from _helpers_tables import (
    CONTINUOUS_FEATURES,
    KEY,
    _checks,
    _continuous_tables,
    _found,
    _replace,
    _rows,
)
from _helpers_workflows import _flat

from morphocoverage import integrity_checks
from morphocoverage._integrity_tables import integrity_by_table
from morphocoverage._style import NOT_ASSESSED, OK, PROBLEM, Status
from morphocoverage.feature_tables import attach_feature_tables

DATA = Path(__file__).resolve().parents[1] / "data" / "crick_idr"
sys.path.insert(0, str(DATA))
from datasets import load


@pytest.fixture(scope="module")
def idr0028():
    objects, fovs = load("idr0028_screenB").tables
    return objects, fovs


def test_clean_datasets_have_no_findings(idr0028):
    result = _check(*idr0028)
    assert result.findings.empty
    assert (result.checks["status"] == "checked").all()
    assert "✅ Data integrity — nothing found by the exact copy checks" in result.summary()


def test_a_field_exported_twice_under_another_name_is_a_copied_field(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)]
    copied = _replace_field(objects, fovs, 40, source.copy())
    result = _check(copied, fovs)
    assert result.findings["check"].tolist() == ["copied field"]
    assert result.findings["observations"].iloc[0] == len(source)
    assert "structure to inspect" in result.summary()


def test_the_same_observations_in_another_order_are_a_reordered_copy(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)]
    shuffled = source.sample(frac=1, random_state=0)
    result = _check(_replace_field(objects, fovs, 40, shuffled), fovs)
    assert result.findings["check"].tolist() == ["reordered copy"]


def test_a_change_of_units_is_a_scaled_copy(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)].copy()
    source["Cytoplasm_Intensity_MeanIntensity_YAPTAZ"] *= 0.65**2  # another intensity scale
    source["Nuclei_Intensity_MeanIntensity_YAPTAZ"] += 100.0  # an offset
    result = _check(_replace_field(objects, fovs, 40, source), fovs)
    assert result.findings["check"].tolist() == ["scaled copy"]


def test_a_block_pasted_into_another_field_is_a_repeated_block(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)]
    target = objects[_field(objects, fovs, 40)].copy()
    pasted = pd.concat([target.iloc[:5], source.iloc[3:15], target.iloc[5:]])
    result = _check(_replace_field(objects, fovs, 40, pasted), fovs)
    [finding] = result.findings.itertuples()
    # The whole run of 12 pasted observations, not the 10 of block_length (P1-11).
    assert finding.check == "repeated block" and finding.observations == 12
    assert "12 consecutive observations" in finding.detail
    assert "from observation 4 of the first field and 6 of the second" in finding.detail


def test_same_means_from_different_observations_are_a_repeated_summary(idr0028):
    objects, fovs = idr0028
    source = objects[_field(objects, fovs, 0)].copy()
    # Shift every nucleus's value by a zero-sum amount: every row differs, every mean is the
    # same. (Swapping two values kept the other rows, which are now shared observations.)
    column = "Cytoplasm_Intensity_MeanIntensity_YAPTAZ"
    shift = np.linspace(-1, 1, len(source)) * source[column].std() / 10
    source[column] = source[column].to_numpy() + (shift - shift.mean())
    result = _check(_replace_field(objects, fovs, 40, source), fovs)
    assert result.findings["check"].tolist() == ["repeated field summary"]


def test_a_small_field_inside_another_is_a_copied_small_field(idr0028):
    """Fields of 2-4 observations (tracks per position) are below min_objects; a copy is still
    found when every observation of the smaller field is in the other."""
    objects, fovs = idr0028
    rows = objects[_field(objects, fovs, 0)].iloc[:4].copy()
    rows[FEATURES] *= 1.5  # values no other field holds, so only the copies share them
    extra = objects[_field(objects, fovs, 41)].iloc[:1]
    copied = _replace_field(objects, fovs, 40, rows)
    copied = _replace_field(copied, fovs, 41, pd.concat([rows, extra]))
    result = _check(copied, fovs)
    [finding] = result.findings.itertuples()
    assert finding.check == "copied small field" and finding.observations == 4
    assert "all 4 observations of the first field are also in the second" in finding.detail


def _one_cell_fields(n_features, copies):
    """Twelve fields of one cell each, ``n_features`` continuous features; the first cell's
    row is written again in ``copies`` other fields."""
    rng = np.random.default_rng(1)
    names = [f"f{i}.tif" for i in range(12)]
    fovs = pd.DataFrame(
        {"condition": "a", "biological_repeat": "r", "repeat_group": "r", "image_file": names}
    )
    values = rng.normal(100, 10, (12, n_features)).round(3)
    values[1 : 1 + copies] = values[0]
    objects = pd.DataFrame(values, columns=[f"x{j}" for j in range(n_features)]).assign(
        condition="a", biological_repeat="r", image_file=names
    )
    features = [f"x{j}" for j in range(n_features)]
    return integrity_checks(objects, fovs, feature_cols=features, observation_unit="cell")


def test_a_cell_row_pasted_into_another_one_cell_field_is_found():
    result = _one_cell_fields(n_features=3, copies=1)
    [finding] = result.findings.itertuples()
    assert finding.check == "copied small field" and finding.observations == 1
    assert "the two fields hold the same single observation" in finding.detail
    # Pasted into two other fields, the row recurs in three: set aside, as documented.
    assert _one_cell_fields(n_features=3, copies=2).findings.empty


def test_one_shared_observation_needs_three_continuous_features():
    result = _one_cell_fields(n_features=2, copies=1)
    assert result.findings.empty
    status = result.checks.set_index("check")
    assert status.loc["copied small field", "status"] == "not applicable"
    assert (
        "one shared observation can be a coincidence" in status.loc["copied small field", "reason"]
    )


def test_quantized_features_make_the_copy_checks_not_applicable():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "repeat_group": "r",
            "image_file": [f"f{i}.tif" for i in range(6)],
        }
    )
    objects = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "repeat_group": "r",
            "image_file": np.repeat(fovs["image_file"], 20).to_numpy(),
            "count": rng.integers(0, 3, 120).astype(float),
        }
    )
    result = integrity_checks(objects, fovs, feature_cols=["count"])
    status = dict(zip(result.checks["check"], result.checks["status"], strict=True))
    assert status["copied field"] == "not applicable"
    assert status["repeated block"] == "not applicable"
    assert status["repeated field summary"] == "not applicable"
    assert result.findings.empty
    assert "cannot be told from a coincidence" in " ".join(result.summary().split())


def test_the_synthetic_acquisition_has_no_findings():
    objects = pd.read_csv(DATA.parent / "synthetic" / "synthetic_cells.csv")
    fovs = pd.read_csv(DATA.parent / "synthetic" / "synthetic_fovs.csv")
    # Each folder label is confirmed as its own repeat group, by which repeats pair (#617).
    fovs = fovs.assign(repeat_group=fovs["biological_repeat"])
    result = integrity_checks(
        objects, fovs, feature_cols=["area", "intensity"], observation_unit="cell"
    )
    assert result.findings.empty, result.summary()


def test_arguments_are_checked():
    fovs = pd.DataFrame(
        {"condition": "a", "biological_repeat": "r", "repeat_group": "r", "image_file": ["f.tif"]}
    )
    objects = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "repeat_group": "r",
            "image_file": "f.tif",
            "x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="block_length"):
        integrity_checks(objects, fovs, feature_cols=["x"], block_length=1)
    with pytest.raises(ValueError, match="min_distinct_share"):
        integrity_checks(objects, fovs, feature_cols=["x"], min_distinct_share=2)


def test_the_report_carries_the_integrity_checks_only_when_asked(idr0028):
    from morphocoverage.workflows import Setup, report

    objects, fovs = idr0028
    wells = fovs["technical_repeat"].isin(fovs["technical_repeat"].unique()[:2])
    objects = objects[objects["technical_repeat"].isin(fovs.loc[wells, "technical_repeat"])]
    setup = Setup(readout=FEATURES[1], features=tuple(FEATURES), observation_unit="nucleus")
    plain = report(objects, fovs[wells], setup)
    checked = report(objects, fovs[wells], setup, integrity=True)
    assert "integrity" not in plain.results
    assert "Data integrity" not in plain.summary(detail="full")
    assert "➖ Copy checks not run (report(..., integrity=True) runs them)" in plain.summary()
    full = checked.summary(detail="full")
    assert "Data integrity" in full and checked.results["integrity"].findings.empty
    assert "✅ Integrity: nothing found." in checked.summary()


def test_integrity_findings_carry_the_typed_key_of_each_field_they_name():
    import numpy as np

    from morphocoverage import integrity_checks

    rng = np.random.default_rng(0)
    block = rng.normal(size=(12, 4))
    rows = []
    for fov in (1, "1", 2, 3):
        values = block if fov in (1, "1") else rng.normal(size=(12, 4))
        for row in values:
            rows.append(
                {
                    "condition": "a",
                    "position_id": fov,
                    **dict(zip(["a1", "a2", "a3", "a4"], row, strict=True)),
                }
            )
    objects = pd.DataFrame(rows)
    result = integrity_checks(
        objects,
        objects[["condition", "position_id"]].drop_duplicates(),
        feature_cols=["a1", "a2", "a3", "a4"],
        observation_unit="cell",
        fov_col="position_id",
    )
    keys = result.settings["field_keys"]
    assert len(result.findings) >= 1
    assert {("a", 1), ("a", "1")} <= set(keys.values()) or set(keys.values()) <= {
        ("a", 1),
        ("a", "1"),
    }
    named = {*result.findings["field"], *result.findings["other_field"]}
    assert named <= set(keys)


def test_p7_1_fixed_size_fields_are_compared_without_a_loop_over_pairs():
    # 2000 fields of 16 patches each (CellPatcher grid): every field has every other's size,
    # 2e6 pairs; the fingerprint is sorted once, so this takes seconds, not hours.
    objects, fovs = _continuous_tables(n_fields=2000, size=16)
    source = _rows(objects, fovs, 3)
    source[CONTINUOUS_FEATURES] = source[CONTINUOUS_FEATURES].to_numpy() * [0.65**2, 1.0, -1.0]
    result = _checks(_replace(objects, fovs, 1500, source.sample(frac=1, random_state=0)), fovs)
    assert _found(result) == ["scaled copy"]
    assert (result.checks["status"] == "checked").all()


def test_p7_2_a_copy_through_float32_or_six_digits_is_a_copied_field():
    objects, fovs = _continuous_tables()
    for convert in (
        lambda v: v.astype(np.float32).astype(float),
        np.vectorize(lambda v: float(f"{v:.6g}")),
    ):
        source = _rows(objects, fovs, 0)
        source[CONTINUOUS_FEATURES] = convert(source[CONTINUOUS_FEATURES].to_numpy())
        result = _checks(_replace(objects, fovs, 7, source), fovs)
        assert _found(result) == ["copied field"]
        assert "to 6 significant digits" in result.findings["detail"].iloc[0]
    clean = " ".join(_checks(objects, fovs).summary().split())
    assert "no field repeats another" not in clean
    assert clean.startswith("✅ Data integrity — nothing found by the exact copy checks")
    assert "re-rounded to fewer than 6 significant digits" in clean


def test_p7_3_a_copy_with_a_row_dropped_or_a_block_pasted_in_another_order_is_found():
    objects, fovs = _continuous_tables(size=40)
    source = _rows(objects, fovs, 0)
    dropped = source.iloc[1:].sample(frac=1, random_state=1)
    assert _found(_checks(_replace(objects, fovs, 9, dropped), fovs)) == ["shared observations"]
    target = _rows(objects, fovs, 9)
    pasted = pd.concat([target.iloc[:5], source.iloc[:20].iloc[::-1], target.iloc[5:]])
    result = _checks(_replace(objects, fovs, 9, pasted), fovs)
    [finding] = result.findings.itertuples()
    assert finding.check == "shared observations" and finding.observations == 20


def test_p7_4_sign_flips_ties_and_missing_values_do_not_hide_a_scaled_copy():
    objects, fovs = _continuous_tables(size=40)
    objects["area"] = np.round(objects["area"] / 20)  # integer pixel areas: many ties
    objects.loc[objects.index[3], "texture"] = np.nan  # a missing value in field 0
    source = _rows(objects, fovs, 0)
    cases = {
        "sign flip on one feature, with an offset": [1.0, -3.0, 1.0],
        "raw grey levels (x65535), ties in the first feature": [65535.0, 65535.0, 1.0],
    }
    for scale in cases.values():
        copy = source.copy()
        copy[CONTINUOUS_FEATURES] = copy[CONTINUOUS_FEATURES].to_numpy() * scale + [0.0, 7.0, 0.0]
        result = _checks(_replace(objects, fovs, 11, copy.sample(frac=1, random_state=2)), fovs)
        assert _found(result) == ["scaled copy"]


def test_p7_5_fields_with_a_missing_value_are_compared_and_counted():
    objects, fovs = _continuous_tables(n_fields=40, size=30)
    objects.loc[objects.index[::7], "texture"] = np.nan  # a missing value in every field
    source = _rows(objects, fovs, 0)
    source["area"] *= 0.65**2
    result = _checks(_replace(objects, fovs, 5, source), fovs)
    assert _found(result) == ["scaled copy"]
    compared = dict(zip(result.checks["check"], result.checks["compared"], strict=True))
    assert compared["scaled copy"] == 40 and compared["repeated field summary"] == 40
    # Fields that could not be compared are counted, not silently skipped.
    small, fovs_small = _continuous_tables(n_fields=10, size=[3] * 4 + [30] * 6)
    assert "Whole fields compared: 6 of 10; 4 fields held fewer than 5 observations" in (
        _checks(small, fovs_small).summary()
    )


def test_p7_6_identical_hot_pixel_objects_are_not_copied_fields():
    objects, fovs = _continuous_tables(n_fields=100, size=30)
    hot = pd.DataFrame({"area": [0.1075**2] * 6, "intensity": [4095.0] * 5 + [1873.0]})
    hot["texture"] = 0.0
    empty = objects
    for i in range(10):  # ten empty fields, each with the same six hot pixels
        empty = _replace(empty, fovs, i, hot.copy())
    assert _checks(empty, fovs).findings.empty
    # Every field starts with the same twelve hot-pixel objects.
    prefix = pd.concat([hot, hot.assign(intensity=2011.0)], ignore_index=True)
    parts = [
        pd.concat([prefix.assign(**fovs.iloc[i][KEY]), _rows(objects, fovs, i)]) for i in range(100)
    ]
    result = _checks(pd.concat(parts, ignore_index=True), fovs)
    assert result.findings.empty
    assert "recur in 3 or more fields" in result.summary()


def test_p7_7_one_row_fields_make_no_repeated_summary():
    rng = np.random.default_rng(1)
    objects, fovs = _continuous_tables(n_fields=1000, size=1)
    n = rng.poisson(20, 1000) + 1
    objects = objects.assign(n_cells=n.astype(float), positive=rng.binomial(n, 0.3) / n)
    result = _checks(objects, fovs, features=["n_cells", "positive"])
    assert result.findings.empty


def test_p7_11_wide_range_integer_features_are_checked_for_repeated_summaries():
    rng = np.random.default_rng(2)
    objects, fovs = _continuous_tables(n_fields=60, size=30)
    objects["area_px"] = rng.integers(200, 5000, len(objects))
    objects["intensity_sum"] = rng.integers(10**5, 10**7, len(objects))
    features = ["area_px", "intensity_sum"]
    source = _rows(objects, fovs, 0)
    shift = np.arange(len(source)) - (len(source) - 1) / 2  # every row differs, same means
    source["area_px"] = source["area_px"] + 2 * shift
    result = _checks(_replace(objects, fovs, 8, source), fovs, features=features)
    assert _found(result) == ["repeated field summary"]
    assert "integer values" not in result.summary()


def test_p7_12_a_repeated_block_reports_its_run_and_tiny_copies_are_not_called_different():
    objects, fovs = _continuous_tables(n_fields=40, size=40)
    source, target = _rows(objects, fovs, 0), _rows(objects, fovs, 20)
    pasted = pd.concat([target.iloc[:3], source.iloc[:25], target.iloc[3:]])
    [finding] = _checks(_replace(objects, fovs, 20, pasted), fovs).findings.itertuples()
    assert finding.check == "repeated block" and finding.observations == 25
    tiny, tiny_fovs = _continuous_tables(n_fields=40, size=3)
    copy = _replace(tiny, tiny_fovs, 1, _rows(tiny, tiny_fovs, 0))
    result = _checks(copy, tiny_fovs)
    assert "although the observations differ" not in result.summary(detail="full")
    assert "40 fields held fewer than 5 observations" in result.summary()


def test_p7_13_background_patches_do_not_hide_a_copied_field():
    objects, fovs = _continuous_tables(n_fields=100, size=50)
    rng = np.random.default_rng(3)
    objects.loc[rng.random(len(objects)) < 0.6, CONTINUOUS_FEATURES] = (
        0.0  # empty background patches
    )
    result = _checks(_replace(objects, fovs, 1, _rows(objects, fovs, 0)), fovs)
    assert _found(result) == ["copied field"]


def test_p7_14_the_summary_states_what_the_checks_cannot_see():
    objects, fovs = _continuous_tables()
    source = _rows(objects, fovs, 0)
    source[["area", "intensity"]] = source[["intensity", "area"]].to_numpy()  # columns swapped
    result = _checks(_replace(objects, fovs, 3, source), fovs)
    assert result.findings.empty
    for text in (result.summary(), result.summary(detail="full")):
        text = " ".join(text.split())
        assert "a block of fewer than 10 observations" in text
        assert "feature columns swapped" in text


def test_p1_11_a_pasted_block_reports_its_whole_run():
    objects, fovs = _continuous_tables(n_fields=20, size=40)
    source, target = _rows(objects, fovs, 2), _rows(objects, fovs, 12)
    pasted = pd.concat([target.iloc[:10], source.iloc[5:20], target.iloc[10:]])
    [finding] = _checks(_replace(objects, fovs, 12, pasted), fovs).findings.itertuples()
    assert finding.observations == 15
    assert "15 consecutive observations" in finding.detail
    assert "from observation 6 of the first field and 11 of the second" in finding.detail


def test_a_copy_confined_to_one_table_is_found_and_named():
    # Audit B, finding 2: pooled, the copy in the embedding alone was missed.
    objects, fovs, rng = _cells(fields=3, cells=12)
    n = len(objects)
    for j in range(3):
        objects[f"x{j}"] = rng.normal(size=n)
    values = rng.normal(size=(n, 6))
    first = (objects["image_file"] == "a_R1_0.tif").to_numpy()
    second = (objects["image_file"] == "a_R1_1.tif").to_numpy()
    values[second] = values[first]  # the embedding of one image computed twice
    merged = attach_feature_tables(objects, fovs, {"DINO": _table(objects, values, "d")}).objects
    columns = {"explainable": ["x0", "x1", "x2"], "DINO": [f"d{j}" for j in range(6)]}
    pooled = integrity_checks(merged, fovs, feature_cols=[c for v in columns.values() for c in v])
    assert pooled.findings.empty
    found = integrity_by_table(merged, fovs, tables=columns)
    assert list(found.findings["check"]) == ["copied field"]
    assert list(found.findings["table"]) == ["DINO"]
    assert "copied field in DINO" in found.summary()
    assert "each feature table apart: explainable (3 columns) and DINO (6 columns)" in " ".join(
        found.summary().split()
    )


OUT_OF_REACH = "Out of reach of these checks"


@pytest.fixture(scope="module")
def integrity():
    objects, fovs = load("idr0028_screenB").tables
    clean = _check(objects, fovs)
    source = objects[_field(objects, fovs, 0)]
    copied = _replace_field(objects, fovs, 40, source.copy())
    copied = _replace_field(copied, fovs, 41, source.sample(frac=1, random_state=0))
    twice = objects[_field(objects, fovs, 60)]
    copied = _replace_field(copied, fovs, 60, pd.concat([twice, twice]))
    return clean, _check(copied, fovs)


def test_integrity_marker_follows_the_findings(integrity):
    clean, dirty = integrity
    assert clean.status is Status.OK and clean.summary().startswith(f"{OK} Data integrity")
    assert dirty.status is Status.PROBLEM
    assert dirty.summary().startswith(f"{PROBLEM} Data integrity — 3 findings in 4 fields")
    with pytest.raises(ValueError, match="short"):
        clean.summary(detail="brief")


def test_integrity_short_groups_the_findings_in_a_table_and_states_caveats_once(integrity):
    _, dirty = integrity
    text = dirty.summary()
    assert "structure to inspect, not a judgment" in text
    assert "Fields are named condition · biological repeat · technical repeat · image file." in text
    # One row per check and pair of groups, the groups named as group labels.
    assert "copied field             between MOCK · repeat 1A · B6 and" in text
    assert "repeated within a field  within " in text
    assert "condition=" not in text
    assert text.count(OUT_OF_REACH) == 1
    assert "Check where these fields came from" in text
    assert "on 3 features" in text


def test_integrity_full_keeps_every_detail_of_the_findings(integrity):
    _, dirty = integrity
    full = _flat(dirty.summary(detail="full"))
    for finding in dirty.findings.itertuples():
        assert _flat(finding.detail) in full
        assert f"{finding.observations} observations" in full
    settings = dirty.settings
    assert f"{settings['n_fovs']} fields and {settings['n_observations']} nuclei" in full
    for feature in FEATURES:
        assert feature in full
    assert full.count(OUT_OF_REACH) == 1


def test_integrity_with_no_applicable_check_is_not_assessed():
    files = [f"f{i}" for i in range(4)]
    fovs = pd.DataFrame(
        {"condition": "a", "biological_repeat": "r", "repeat_group": "r", "image_file": files}
    )
    objects = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "r",
            "repeat_group": "r",
            "image_file": [f"f{i // 10}" for i in range(40)],
            "count": [float(i % 3) for i in range(40)],
        }
    )
    result = integrity_checks(objects, fovs, feature_cols=["count"])
    assert result.status is Status.NOT_ASSESSED
    text = result.summary()
    assert text.startswith(f"{NOT_ASSESSED} Data integrity")
    assert "cannot be told from a coincidence" in _flat(text)
