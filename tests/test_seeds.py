"""Tests of ``_seeds``: stable per-group seeds and group keys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import MEAN
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage._seeds import group_identity, group_key, group_rng, group_seed, split_seed
from morphocoverage.retrospective import sampling_reliability
from morphocoverage.sampling_unit import condition_contrast, object_rule_check, unusual_objects


def test_a10_b11_group_identity_does_not_depend_on_the_numpy_type():
    assert group_identity({"biological_repeat": np.int64(1)}) == group_identity(
        {"biological_repeat": 1}
    )
    # Audit 4 (owner decision on P9-2): a retyped label is the same group for seeding.
    assert group_identity({"biological_repeat": 1}) == group_identity({"biological_repeat": "1"})
    assert group_identity({"biological_repeat": 1}) != group_identity({"biological_repeat": 2})


def test_j1_group_key_tells_every_group_apart():
    a = {"condition": "plate24_wellI03_KO", "biological_repeat": "2024_01_15"}
    b = {"condition": "plate15_wellP15_siGENE", "biological_repeat": "2024_01_15"}
    assert group_identity(a) == group_identity(b)  # a real crc32 collision
    assert group_key(a) != group_key(b)
    assert group_key({"biological_repeat": np.int64(1)}) == group_key({"biological_repeat": 1})


def test_f6_split_seeds_of_neighbouring_random_states_do_not_overlap():
    first = {split_seed(0, s) for s in range(20)}
    second = {split_seed(1, s) for s in range(20)}
    assert not first & second and len(first) == 20
    assert split_seed(np.int64(3), np.int64(2)) == split_seed(3, 2)


def test_a10_group_identity_is_the_same_for_numpy_and_python_values():
    assert group_identity({"biological_repeat": np.int64(1)}) == group_identity(
        {"biological_repeat": 1}
    )
    assert group_identity({"flag": np.True_}) == group_identity({"flag": True})


def test_j1_groups_whose_seeds_collide_are_still_told_apart():
    first, second, repeat = "plate24_wellI03_KO", "plate15_wellP15_siGENE", "2024_01_15"
    assert group_identity({"condition": first, "biological_repeat": repeat}) == group_identity(
        {"condition": second, "biological_repeat": repeat}
    )
    rng = np.random.default_rng(0)
    rows, fov_rows = [], []
    for condition, shift in ((first, 0.0), (second, 5.0)):
        for f in range(8):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "position_id": f"{condition}_{f}",
            }
            fov_rows.append(key)
            rows += [{**key, "area": 10 + shift + v} for v in rng.normal(size=30)]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fov_rows)
    control = {"condition": first}
    contrast = condition_contrast(objects, fovs, feature="area", control=control, n_draws=5)
    assert contrast.table["d_objects"].iloc[0] > 3  # not the control against itself
    rules = object_rule_check(objects, fovs, feature="area", control=control, n_draws=20)
    assert rules.table["status"].eq("ok").all()
    unusual = unusual_objects(objects, fovs, feature_cols=["area"], reference=control)
    treated = unusual.groups.set_index("condition").loc[second]
    assert treated["reference_groups"] == [{"condition": first, "biological_repeat": repeat}]
    assert treated["unusual_share"] > 0.5


def test_p9_2_retyped_labels_keep_their_draws_and_shares_carry_their_monte_carlo_error():
    # Owner decision: seed by the labels written canonically, so a retyped label is the
    # same group; other groups keep independent draws and never enter a group's seed.
    assert group_seed(0, {"biological_repeat": 1}) == group_seed(0, {"biological_repeat": "1"})
    assert group_seed(0, {"biological_repeat": 1}) == group_seed(0, {"biological_repeat": 1.0})
    assert group_seed(3, {"well": "A1"}, 2)[::2] == [3, 2]
    first = group_rng(0, {"condition": "ctrl"}).permutation(12)
    assert not (group_rng(0, {"condition": "drug"}).permutation(12) == first).all()
    built = SCENARIOS["homogeneous"]()
    settings = {
        "feature_cols": list(FEATURE_COLS),
        "criteria": [MEAN],
        "fov_counts": [2, 5],
        "reservoir_fraction": 0.3,
        "n_splits": 4,
        "draws": 10,
    }
    numbered = {"R1": 1, "R2": 2, "R3": 3}

    def retyped(table: pd.DataFrame, as_text: bool) -> pd.DataFrame:
        labels = table["biological_repeat"].map(lambda r: numbered.get(r, r))
        return table.assign(biological_repeat=labels.astype(str) if as_text else labels.astype(int))

    as_numbers = sampling_reliability(
        retyped(built.objects, False), retyped(built.fovs, False), **settings
    )
    as_text = sampling_reliability(
        retyped(built.objects, True), retyped(built.fovs, True), **settings
    )
    columns = ["fovs", "preserved", "preserved_split_sd", "preserved_mc_se"]
    pd.testing.assert_frame_equal(as_numbers.curve[columns], as_text.curve[columns])
    row = as_numbers.curve.iloc[0]
    assert row["preserved_mc_se"] == pytest.approx(row["preserved_split_sd"] / 2)
    assert "Monte Carlo SE" in as_numbers.summary(detail="full")
