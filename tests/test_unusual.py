"""Tests of ``sampling_unit.unusual``: unusual objects and their thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from _helpers_sampling_unit import CONTROL, _error, _field_rows, _join, _screen

from morphocoverage.sampling_unit import fov_clustering, unusual_objects


def test_j12_a_group_without_complete_features_is_not_assessed():
    rng = np.random.default_rng(0)
    rows = []
    for condition in ("MOCK", "MOCK2", "KD"):
        for j in range(4):
            for v, w in rng.normal(0, 1, size=(20, 2)):
                y = np.nan if condition == "KD" else w
                rows.append(
                    {"condition": condition, "position_id": f"{condition}{j}", "x": v, "y": y}
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "position_id"]].drop_duplicates()
    result = unusual_objects(
        objects,
        fovs,
        feature_cols=["x", "y"],
        reference={"condition": "MOCK"},
        group_cols=["condition"],
    )
    kd = result.groups.set_index("condition").loc["KD"]
    assert kd["status"].startswith("not assessed") and kd["status"] != "ok"
    text = " ".join(result.summary().split())
    assert (
        "Unusual cells not scored in KD: not assessed: none of its 80 cells has a value of "
        "every feature." in text
    )
    assert "nan%" not in text and "0 of 0" not in text


def _repeat_shifted(shift, seed, n_repeats):
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for repeat in range(n_repeats):
        offset = rng.normal(0, shift, 2)
        for condition in ("ctrl", "drug"):
            for j in range(10):
                key = {
                    "condition": condition,
                    "biological_repeat": f"r{repeat}",
                    "position_id": f"{condition}{repeat}_{j}",
                }
                fovs.append(key)
                values = offset + rng.normal(0, 0.3, 2) + rng.normal(0, 1, (40, 2))
                rows += [{**key, "a": a, "b": b} for a, b in values]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def test_h9_unusual_thresholds_leave_one_group_out_for_every_group():
    shares = {"ctrl": [], "drug": []}
    for seed in range(3):
        objects, fovs = _repeat_shifted(0.5, seed, n_repeats=6)
        result = unusual_objects(
            objects, fovs, feature_cols=["a", "b"], reference=CONTROL, seed_checks=0
        )
        groups = result.groups
        assert (groups["calibration"] == "leave-one-group-out").all()
        for condition in shares:
            shares[condition] += list(groups[groups["condition"] == condition]["unusual_share"])
    # A repeat shift of half an SD flagged 2.9% of control objects for a nominal 1%
    # when thresholds came from fields of the same repeat.
    assert np.mean(shares["ctrl"]) < 0.02 and np.mean(shares["drug"]) < 0.02
    # A single reference group cannot leave one out: the fields are, and it is said.
    single, single_fovs = _repeat_shifted(0.5, 0, n_repeats=1)
    one = unusual_objects(
        single, single_fovs, feature_cols=["a", "b"], reference=CONTROL, seed_checks=0
    )
    row = one.groups.set_index("condition").loc["drug"]
    assert row["calibration"] == "leave-one-field-out (a single reference group)"
    # The reference named as the reference (audit 10, A24).
    said = " ".join(one.summary().split())
    assert "Against a single reference group, ctrl · repeat r0, the threshold comes from" in said


def test_g8_a_group_without_labels_is_not_said_to_lack_the_state():
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for condition in ("MOCK", "KD"):
        for j in range(8):
            name = f"{condition}_{j}"
            fields.append({"condition": condition, "image_file": name})
            rows += [
                {"condition": condition, "image_file": name, "a": rng.normal(), "b": rng.normal()}
                for _ in range(30)
            ]
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fields)
    unusual = unusual_objects(
        objects,
        fovs,
        feature_cols=["a", "b"],
        reference={"condition": "MOCK"},
        group_cols=["condition"],
    )
    text = fov_clustering(
        objects.assign(unusual=unusual.labels),
        fovs,
        label_col="unusual",
        states=[True],
        group_cols=["condition"],
    ).summary()
    text = " ".join(text.split())
    assert "Not looked for in MOCK (none of the cells of its 8 fields scored)" in text
    assert "Not seen in KD (240 cells scored in 8 fields, none unusual)" in text


def test_unusual_cells_are_judged_against_their_own_sets_control():
    objects, fovs = _screen()
    result = unusual_objects(
        objects, fovs, feature_cols=["x"], reference={"condition": "ctrl"}, seed_checks=0
    )
    for row in result.groups.to_dict("records"):
        own = row["biological_repeat"][0]
        assert {g["biological_repeat"][0] for g in row["reference_groups"]} == {own}
    pooled = unusual_objects(
        objects,
        fovs,
        feature_cols=["x"],
        reference={"condition": "ctrl"},
        seed_checks=0,
        within_experiment=False,
    )
    assert result.settings["within_experiment"] and not pooled.settings["within_experiment"]
    sets = {g["biological_repeat"][0] for g in pooled.groups["reference_groups"].iloc[0]}
    assert sets == {"a", "b"}


def _unusual_edges():
    from morphocoverage.sampling_unit import unusual_objects

    parts = [_field_rows("ctrl", f"r{r}", 4, 6, 1.0, r) for r in range(2)]
    objects, fovs = _join(*parts, _field_rows("A", "r0", 4, 6, 1.0, 7))
    objects["y"] = np.random.default_rng(3).normal(0, 1, len(objects))
    error = _error(
        lambda: unusual_objects(
            objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, quantile=1.0
        )
    )
    missing = objects.copy()
    missing.loc[missing["condition"] == "A", "y"] = np.nan
    result = unusual_objects(
        missing, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, seed_checks=1
    )
    return error, result.groups["status"].tolist()


EXPECTED_UNUSUAL_EDGES = (
    ("ValueError", "quantile must be between 0 and 1"),
    ["ok", "ok", "not assessed: none of its 24 cells has a value of every feature"],
)


def test_unusual_edges():
    assert _unusual_edges() == EXPECTED_UNUSUAL_EDGES
