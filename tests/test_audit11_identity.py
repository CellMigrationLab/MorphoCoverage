"""Identity is typed, never display text: 1 and "1" are two values (issues 283, 319)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables
from morphocoverage._identity import (
    canonical_codes,
    display,
    identity_key,
)
from morphocoverage.sampling_unit._groups import contrasts_of
from morphocoverage.sampling_unit._units import resolve_unit
from morphocoverage.schema import canonical_fields
from morphocoverage.strategies.fov_bootstrap import FOVBootstrapEstimator


def _mixed(order: list[int]) -> pd.DataFrame:
    """Four FOVs whose identifiers print alike in pairs: 1 / "1" and 2 / "2"."""
    fovs = pd.DataFrame(
        {
            "position_id": [1, "1", 2, "2"],
            "condition": "c",
            "block": [1, "1", 2, "2"],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return fovs.iloc[order].reset_index(drop=True)


def _data(order: list[int]):
    fovs = _mixed(order)
    return validate_input_tables(
        objects=fovs, fovs=fovs[["position_id", "condition", "block"]], feature_cols=["v"]
    )


def test_identity_keys_tell_equal_text_apart_and_ignore_numpy_wrappers():
    assert identity_key(1) != identity_key("1") != identity_key(1.0)
    assert identity_key(np.int64(1)) == identity_key(1)
    with pytest.raises(ValueError, match="never missing"):
        identity_key(float("nan"))
    with pytest.raises(TypeError, match="hashable"):
        identity_key([1])


def test_display_labels_add_the_type_only_where_text_collides():
    shown = display([identity_key(1), identity_key("1"), identity_key("a")])
    assert shown[identity_key(1)] == "1 (int)" and shown[identity_key("1")] == "1 (str)"
    assert shown[identity_key("a")] == "a"


def test_the_canonical_fov_order_does_not_depend_on_row_order_when_ids_print_alike():
    forward, backward = _data([0, 1, 2, 3]), _data([3, 2, 1, 0])
    first = [forward.fovs["position_id"].iloc[i] for i in canonical_fields(forward)]
    second = [backward.fovs["position_id"].iloc[i] for i in canonical_fields(backward)]
    assert first == second


def test_blocks_need_not_be_orderable_and_are_numbered_by_identity():
    frame = _mixed([0, 1, 2, 3])["block"]
    codes, keys = canonical_codes(frame)
    shuffled, same_keys = canonical_codes(frame.iloc[[3, 1, 0, 2]].reset_index(drop=True))
    assert keys == same_keys and len(set(codes)) == 4
    assert list(shuffled) == [codes[3], codes[1], codes[0], codes[2]]


def test_the_bootstrap_draws_the_same_blocks_whatever_the_row_order():
    def weights(order: list[int]) -> np.ndarray:
        data = _data(order)
        estimator = FOVBootstrapEstimator(
            summary="centroid",
            transform=None,
            n_resamples=20,
            random_state=3,
            min_fovs=2,
            block_col="block",
        )
        return estimator._weights(data)

    forward, backward = weights([0, 1, 2, 3]), weights([3, 2, 1, 0])
    assert np.array_equal(forward, backward[:, ::-1])


def test_contrasts_are_ordered_by_identity_not_by_set_iteration():
    described = [{"repeat": r, "condition": "drug", "is_control": False} for r in (1, "1", 2, "2")]
    expected = contrasts_of(described)
    for order in ([3, 2, 1, 0], [1, 3, 0, 2]):
        assert contrasts_of([described[i] for i in order]) == expected
    assert len(expected) == 4


def test_a_technical_unit_is_not_claimed_because_two_repeats_print_alike():
    # Repeat 1 and repeat "1" are two sides, each with one well: no side holds two wells.
    fovs = pd.DataFrame(
        {
            "position_id": ["a", "b"],
            "condition": ["c", "c"],
            "biological_repeat": [1, "1"],
            "well": ["w1", "w2"],
        }
    )
    objects = fovs.assign(v=[1.0, 2.0])
    data = validate_input_tables(
        objects=objects,
        fovs=fovs,
        feature_cols=["v"],
        group_cols=["condition", "biological_repeat", "well"],
    )
    described = [
        {"repeat": 1, "condition": "c"},
        {"repeat": "1", "condition": "c"},
    ]
    unit, why = resolve_unit(data, described, "condition", "biological_repeat", None)
    assert unit is None and "one value per condition and repeat" in why


def test_a_composite_fov_key_is_written_out_whole_in_messages():
    from morphocoverage._identity import key_labels

    fovs = pd.DataFrame({"condition": ["a", "a"], "biological_repeat": ["R1", "R2"], "id": [1, 1]})
    assert key_labels(fovs, ["condition", "biological_repeat", "id"]) == [
        "condition=a, biological_repeat=R1, id=1",
        "condition=a, biological_repeat=R2, id=1",
    ]


def test_the_missing_value_message_names_the_repeat_when_field_names_restart():
    from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
    from morphocoverage.strategies import ScalarPrecisionEstimator

    rows, fovs = [], []
    for repeat in ("R1", "R2"):
        for f in range(4):
            fovs.append({"condition": "c", "biological_repeat": repeat, "position_id": f"f{f}"})
            for k in range(3):
                value = np.nan if (repeat == "R2" and f == 1 and k == 0) else 10.0 + f + k
                rows.append({**fovs[-1], "x": value})
    data = validate_input_tables(
        objects=pd.DataFrame(rows), fovs=pd.DataFrame(fovs), feature_cols=["x"]
    )
    estimator = ScalarPrecisionEstimator("x", "fov", "upper_95")
    quantity = estimator.estimate(
        data, objective=ObjectiveConfig(SamplingObjective.POPULATION)
    ).quantities["averaging_cv"]
    assert "condition=c, biological_repeat=R2, position_id=f1" in quantity.reason
