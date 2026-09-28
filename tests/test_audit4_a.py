"""Adversarial audit 4, group A: the data-integrity checks and check_tables' copy warnings.

One test per finding (P7-1 to P7-15, P5-4, P1-11), each on small deterministic tables.
"""

import numpy as np
import pandas as pd

from morphocoverage import check_tables, integrity_checks
from morphocoverage.retrospective import microscopist_report
from morphocoverage.workflows._document import _with_integrity

KEY = ["condition", "repeat", "technical_repeat", "image_file"]
FEATURES = ["area", "intensity", "texture"]


def _tables(n_fields=40, size=30, seed=0, wells=4):
    """``n_fields`` fields of ``size`` (an int or one per field) continuous observations."""
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        {
            "condition": "ctrl",
            "repeat": "r1",
            "technical_repeat": [f"W{i % wells}" for i in range(n_fields)],
            "image_file": [f"f{i:05d}.tif" for i in range(n_fields)],
        }
    )
    sizes = np.broadcast_to(size, n_fields)
    objects = fovs.loc[np.repeat(fovs.index, sizes)].reset_index(drop=True)
    objects["area"] = rng.lognormal(5, 0.3, len(objects))
    objects["intensity"] = rng.gamma(4, 50, len(objects))
    objects["texture"] = rng.normal(0, 1, len(objects))
    return objects, fovs


def _rows(objects, fovs, i):
    return objects[objects["image_file"].eq(fovs["image_file"].iloc[i])].copy()


def _replace(objects, fovs, target, rows):
    """``objects`` with field ``target``'s observations replaced by ``rows``."""
    keep = objects[~objects["image_file"].eq(fovs["image_file"].iloc[target])]
    identity = fovs.iloc[target][KEY].to_dict()
    return pd.concat([keep, rows.assign(**identity)], ignore_index=True)


def _checks(objects, fovs, features=FEATURES, **options):
    return integrity_checks(
        objects, fovs, feature_cols=features, observation_unit="cell", **options
    )


def _found(result):
    return result.findings["check"].tolist()


def test_p7_1_fixed_size_fields_are_compared_without_a_loop_over_pairs():
    # 2000 fields of 16 patches each (CellPatcher grid): every field has every other's size,
    # 2e6 pairs; the fingerprint is sorted once, so this takes seconds, not hours.
    objects, fovs = _tables(n_fields=2000, size=16)
    source = _rows(objects, fovs, 3)
    source[FEATURES] = source[FEATURES].to_numpy() * [0.65**2, 1.0, -1.0]
    result = _checks(_replace(objects, fovs, 1500, source.sample(frac=1, random_state=0)), fovs)
    assert _found(result) == ["scaled copy"]
    assert (result.checks["status"] == "checked").all()


def test_p7_2_a_copy_through_float32_or_six_digits_is_a_copied_field():
    objects, fovs = _tables()
    for convert in (
        lambda v: v.astype(np.float32).astype(float),
        np.vectorize(lambda v: float(f"{v:.6g}")),
    ):
        source = _rows(objects, fovs, 0)
        source[FEATURES] = convert(source[FEATURES].to_numpy())
        result = _checks(_replace(objects, fovs, 7, source), fovs)
        assert _found(result) == ["copied field"]
        assert "to 6 significant digits" in result.findings["detail"].iloc[0]
    clean = _checks(objects, fovs).summary()
    assert "no field repeats another" not in clean
    assert "Nothing was found by these checks." in clean
    assert "re-rounded to fewer than 6 significant digits" in clean


def test_p7_3_a_copy_with_a_row_dropped_or_a_block_pasted_in_another_order_is_found():
    objects, fovs = _tables(size=40)
    source = _rows(objects, fovs, 0)
    dropped = source.iloc[1:].sample(frac=1, random_state=1)
    assert _found(_checks(_replace(objects, fovs, 9, dropped), fovs)) == ["shared observations"]
    target = _rows(objects, fovs, 9)
    pasted = pd.concat([target.iloc[:5], source.iloc[:20].iloc[::-1], target.iloc[5:]])
    result = _checks(_replace(objects, fovs, 9, pasted), fovs)
    [finding] = result.findings.itertuples()
    assert finding.check == "shared observations" and finding.observations == 20


def test_p7_4_sign_flips_ties_and_missing_values_do_not_hide_a_scaled_copy():
    objects, fovs = _tables(size=40)
    objects["area"] = np.round(objects["area"] / 20)  # integer pixel areas: many ties
    objects.loc[objects.index[3], "texture"] = np.nan  # a missing value in field 0
    source = _rows(objects, fovs, 0)
    cases = {
        "sign flip on one feature, with an offset": [1.0, -3.0, 1.0],
        "raw grey levels (x65535), ties in the first feature": [65535.0, 65535.0, 1.0],
    }
    for scale in cases.values():
        copy = source.copy()
        copy[FEATURES] = copy[FEATURES].to_numpy() * scale + [0.0, 7.0, 0.0]
        result = _checks(_replace(objects, fovs, 11, copy.sample(frac=1, random_state=2)), fovs)
        assert _found(result) == ["scaled copy"]


def test_p7_5_fields_with_a_missing_value_are_compared_and_counted():
    objects, fovs = _tables(n_fields=40, size=30)
    objects.loc[objects.index[::7], "texture"] = np.nan  # a missing value in every field
    source = _rows(objects, fovs, 0)
    source["area"] *= 0.65**2
    result = _checks(_replace(objects, fovs, 5, source), fovs)
    assert _found(result) == ["scaled copy"]
    compared = dict(zip(result.checks["check"], result.checks["compared"], strict=True))
    assert compared["scaled copy"] == 40 and compared["repeated field summary"] == 40
    # Fields that could not be compared are counted, not silently skipped.
    small, fovs_small = _tables(n_fields=10, size=[3] * 4 + [30] * 6)
    assert "Whole fields compared: 6 of 10; 4 fields held fewer than 5 observations" in (
        _checks(small, fovs_small).summary()
    )


def test_p7_6_identical_hot_pixel_objects_are_not_copied_fields():
    objects, fovs = _tables(n_fields=100, size=30)
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
    objects, fovs = _tables(n_fields=1000, size=1)
    n = rng.poisson(20, 1000) + 1
    objects = objects.assign(n_cells=n.astype(float), positive=rng.binomial(n, 0.3) / n)
    result = _checks(objects, fovs, features=["n_cells", "positive"])
    assert result.findings.empty


def test_p7_8_findings_are_grouped_capped_and_named_among_the_fields_to_look_at():
    objects, fovs = _tables(n_fields=40, size=20)
    copied = objects
    for i in range(10):  # every field of well W1 among the first 40 is a copy of one of W0
        if fovs["technical_repeat"].iloc[i] == "W1":
            copied = _replace(copied, fovs, i, _rows(objects, fovs, i - 1))
    result = _checks(copied, fovs)
    lines = [line for line in result.summary().splitlines() if line.startswith("  - ")]
    assert len(lines) == 1 and "3 pairs of fields between" in lines[0]
    document = _with_integrity(microscopist_report(), result)
    assert "name 6 fields in 3 findings" in document.sections["inspect"]
    assert any("Data integrity" in note for note in document.notes)
    assert "Data integrity" in document.limits()


def test_p7_9_an_appended_table_with_a_fresh_index_and_object_numbers_is_reported():
    objects, fovs = _tables(n_fields=12, size=20)
    objects["ObjectNumber"] = objects.groupby("image_file").cumcount() + 1
    twice = pd.concat([objects, objects], ignore_index=True).reset_index()
    twice = twice.rename(columns={"index": "Unnamed: 0"})
    report = check_tables(twice, fovs, feature_cols=FEATURES, observation_unit="cell")
    assert any("repeat another observation of the same field" in p.title for p in report.warnings)
    result = _checks(twice.sort_values(["image_file", "Unnamed: 0"]), fovs)
    assert set(_found(result)) == {"repeated within a field"}
    assert "holds its observations 1-20 again from observation 21" in result.findings["detail"][0]


def test_p7_10_an_exact_copy_does_not_hide_the_label_warning():
    objects, fovs = _tables(n_fields=6, size=10)
    objects["label_id"] = 1 + np.arange(len(objects)) % 10
    # Two fields of well W0 exported under one file name: their labels collide.
    objects.loc[objects["image_file"].eq("f00004.tif"), "image_file"] = "f00000.tif"
    fovs = fovs[fovs["image_file"] != "f00004.tif"]
    doubled = pd.concat([objects, objects.iloc[[30]]], ignore_index=True)
    titles = [p.title for p in check_tables(doubled, fovs, feature_cols=FEATURES).warnings]
    assert any("exact copy" in t for t in titles)
    assert any("label_ids appear twice in the same field" in t for t in titles)


def test_p7_11_wide_range_integer_features_are_checked_for_repeated_summaries():
    rng = np.random.default_rng(2)
    objects, fovs = _tables(n_fields=60, size=30)
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
    objects, fovs = _tables(n_fields=40, size=40)
    source, target = _rows(objects, fovs, 0), _rows(objects, fovs, 20)
    pasted = pd.concat([target.iloc[:3], source.iloc[:25], target.iloc[3:]])
    [finding] = _checks(_replace(objects, fovs, 20, pasted), fovs).findings.itertuples()
    assert finding.check == "repeated block" and finding.observations == 25
    tiny, tiny_fovs = _tables(n_fields=40, size=3)
    copy = _replace(tiny, tiny_fovs, 1, _rows(tiny, tiny_fovs, 0))
    result = _checks(copy, tiny_fovs)
    assert "although the observations differ" not in result.summary()
    assert "40 fields held fewer than 5 observations" in result.summary()


def test_p7_13_background_patches_do_not_hide_a_copied_field():
    objects, fovs = _tables(n_fields=100, size=50)
    rng = np.random.default_rng(3)
    objects.loc[rng.random(len(objects)) < 0.6, FEATURES] = 0.0  # empty background patches
    result = _checks(_replace(objects, fovs, 1, _rows(objects, fovs, 0)), fovs)
    assert _found(result) == ["copied field"]


def test_p7_14_the_summary_states_what_the_checks_cannot_see():
    objects, fovs = _tables()
    source = _rows(objects, fovs, 0)
    source[["area", "intensity"]] = source[["intensity", "area"]].to_numpy()  # columns swapped
    result = _checks(_replace(objects, fovs, 3, source), fovs)
    assert result.findings.empty
    text = result.summary()
    assert "a block of fewer than 10 observations" in text
    assert "feature columns swapped" in text


def test_p7_15_frame_columns_and_failed_segmentations_are_worded_honestly():
    objects, fovs = _tables(n_fields=6, size=10)
    objects["label_id"] = 1 + np.arange(len(objects)) % 10
    frames = pd.concat(
        [objects.assign(timepoint=0), objects.assign(timepoint=1, area=objects["area"] * 1.01)],
        ignore_index=True,
    )
    warnings = check_tables(frames, fovs, feature_cols=FEATURES, observation_unit="cell").warnings
    [warning] = [p for p in warnings if "label_id" in p.title]
    assert "a plane or frame column not called z, c or t" in warning.fix
    failed = objects.drop(columns="label_id").iloc[:4].copy()
    failed[FEATURES] = np.nan  # failed segmentations: no value, no label
    rows = pd.concat([objects.drop(columns="label_id"), failed, failed], ignore_index=True)
    titles = [p.title for p in check_tables(rows, fovs, feature_cols=FEATURES).warnings]
    assert not any("exact copy" in t for t in titles)


def test_p5_4_a_merge_with_one_row_per_channel_is_reported():
    objects, fovs = _tables(n_fields=12, size=15)
    objects["ObjectNumber"] = objects.groupby("image_file").cumcount() + 1
    merged = pd.concat([objects.assign(Channel="DAPI"), objects.assign(Channel="GFP")])
    merged = merged.sort_values(["image_file", "Channel"], kind="stable")
    titles = [p.title for p in check_tables(merged, fovs, feature_cols=FEATURES).warnings]
    assert any("repeat another observation of the same field in every feature" in t for t in titles)
    result = _checks(merged, fovs)
    assert set(_found(result)) == {"repeated within a field"} and len(result.findings) == 12
    assert "Nothing was found" not in result.summary()


def test_p1_11_a_pasted_block_reports_its_whole_run():
    objects, fovs = _tables(n_fields=20, size=40)
    source, target = _rows(objects, fovs, 2), _rows(objects, fovs, 12)
    pasted = pd.concat([target.iloc[:10], source.iloc[5:20], target.iloc[10:]])
    [finding] = _checks(_replace(objects, fovs, 12, pasted), fovs).findings.itertuples()
    assert finding.observations == 15
    assert "15 consecutive observations" in finding.detail
    assert "from observation 6 of the first field and 11 of the second" in finding.detail
