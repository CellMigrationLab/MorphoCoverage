"""Tests for the shuffle check of the next-FOV new-phenotype prediction."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage._seeds import group_rng
from morphocoverage.calibration import (
    CALIBRATION_COLUMNS,
    calibrate_next_fov_novelty,
    pool_calibration,
)
from morphocoverage.strategies import SupportNoveltyEstimator

NOVELTY = SupportNoveltyEstimator(0.5, 1, None, 2, 2)


def _acquisition(p_new, n_fovs=30, n_empty=0, seed=0, condition="c"):
    """FOVs of common cells; each independently holds its own unseen state with ``p_new``."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_fovs):
        cells = rng.normal(0, 0.1, size=(15, 2))
        if rng.random() < p_new:
            cells = np.vstack([cells, rng.normal(10 + 5 * j, 0.05, size=(3, 2))])
        rows += [{"condition": condition, "image_file": f"f{j}", "a": a, "b": b} for a, b in cells]
    names = [f"f{j}" for j in range(n_fovs + n_empty)]
    return pd.DataFrame(rows), pd.DataFrame({"condition": condition, "image_file": names})


def _calibrate(objects, fovs, novelty=NOVELTY, fov_counts=(5, 10), draws=30, seed=0):
    return calibrate_next_fov_novelty(
        objects,
        fovs,
        feature_cols=["a", "b"],
        novelty=novelty,
        fov_counts=fov_counts,
        draws=draws,
        random_state=seed,
    )


def test_no_new_states_are_predicted_or_found_in_homogeneous_fovs():
    table = _calibrate(*_acquisition(0.0))
    assert list(table.columns) == ["condition", *CALIBRATION_COLUMNS]
    assert (table["predicted"] == 0).all() and (table["actual"] == 0).all()
    assert (table["bound_held"] == 1).all()
    # The bound still reflects how few FOVs were seen.
    assert table["bound"].iloc[0] == pytest.approx(1 - 0.05 ** (1 / 5))


def test_every_fov_new_is_predicted_and_found():
    table = _calibrate(*_acquisition(1.0))
    assert (table["predicted"] == 1).all() and (table["actual"] == 1).all()


def test_the_actual_rate_is_the_remaining_fovs_judged_against_the_drawn_ones():
    objects, fovs = _acquisition(0.3, n_fovs=12, seed=3)
    table = _calibrate(objects, fovs, fov_counts=(4,), draws=1, seed=7)
    # Replay the one draw: FOVs with their own state are new after any subset
    # that does not contain them, so the actual rate is known by construction.
    order = group_rng(7, {"condition": "c"}, 4).permutation(12)
    has_state = objects.groupby("image_file")["a"].max() > 5
    # The draws index the FOVs in FOV-key order (f0, f1, f10, f11, f2, ...).
    names = sorted(f"f{j}" for j in range(12))
    remaining = [names[j] for j in order[4:]]
    assert table["actual"].iloc[0] == pytest.approx(has_state.reindex(remaining).mean())
    drawn = [names[j] for j in order[:4]]
    assert table["predicted"].iloc[0] == pytest.approx(has_state.reindex(drawn).mean())


def test_prediction_is_calibrated_when_fovs_are_exchangeable():
    # Each FOV independently holds a new state with probability 0.2: on average
    # the prediction matches the actual rate and the 95% bound mostly holds.
    objects, fovs = _acquisition(0.2, n_fovs=60, seed=4)
    table = _calibrate(objects, fovs, fov_counts=(10, 30), draws=200, seed=1)
    assert np.allclose(table["predicted"], table["actual"], atol=0.03)
    # With 30 of 60 FOVs drawn, the actual rate rests on 30 remaining FOVs and
    # is noisy, which lowers how often the bound appears to hold (0.8 to 0.92
    # across seeds); with 10 drawn it holds in about 95% of draws or more.
    assert table["bound_held"].iloc[0] >= 0.9 and table["bound_held"].iloc[1] >= 0.8


def test_zero_observation_fovs_can_be_drawn_and_are_never_new():
    objects, fovs = _acquisition(1.0, n_fovs=10, n_empty=10, seed=5)
    table = _calibrate(objects, fovs, fov_counts=(8,), draws=50)
    assert table["remaining_fovs"].iloc[0] == 12
    # Half the FOVs are empty, so about half of each side is new.
    assert 0.35 < table["actual"].iloc[0] < 0.65
    assert 0.35 < table["predicted"].iloc[0] < 0.65


def test_undefined_predictions_are_counted_not_averaged():
    # Only 3 FOVs with cells out of 20: small subsets often have fewer than 2.
    objects, fovs = _acquisition(0.0, n_fovs=3, n_empty=17, seed=6)
    table = _calibrate(objects, fovs, fov_counts=(2,), draws=40)
    row = table.iloc[0]
    assert row["unavailable_draws"] > 0 and row["draws"] + row["unavailable_draws"] == 40


def test_groups_are_calibrated_separately_and_large_counts_skipped():
    built = SCENARIOS["condition_repeat_groups"]()
    novelty = SupportNoveltyEstimator(0.5, 1, None, 2, 1)
    table = calibrate_next_fov_novelty(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        novelty=novelty,
        fov_counts=(3, 8, 50),
        draws=5,
        random_state=0,
    )
    # Four groups of 8 FOVs: only k = 3 is smaller than the group; the larger
    # counts keep a row with no draws and the reason (second audit, B-12).
    assert (
        len(table) == 12 and len(table[["condition", "biological_repeat"]].drop_duplicates()) == 4
    )
    drawn = table[table["fovs_imaged"] == 3]
    assert len(drawn) == 4 and set(drawn["remaining_fovs"]) == {5}
    assert drawn["reason"].isna().all()
    skipped = table[table["fovs_imaged"] > 3]
    assert (skipped["draws"] == 0).all() and skipped["reason"].str.contains("has 8 FOVs").all()


def test_deterministic_under_a_seed():
    objects, fovs = _acquisition(0.3, seed=8)
    first, again = _calibrate(objects, fovs, seed=3), _calibrate(objects, fovs, seed=3)
    pd.testing.assert_frame_equal(first, again)
    assert not _calibrate(objects, fovs, seed=4).equals(first)


def test_pooling_weights_groups_by_usable_draws():
    table = pd.DataFrame(
        {
            "fovs_imaged": [5, 5],
            "draws": [30, 10],
            "predicted": [0.2, 0.6],
            "actual": [0.1, 0.5],
            "bound": [0.5, 0.9],
            "bound_held": [1.0, 0.5],
        }
    )
    pooled = pool_calibration(table).iloc[0]
    assert pooled["draws"] == 40 and pooled["groups"] == 2
    assert pooled["predicted"] == pytest.approx(0.3)
    assert pooled["bound_held"] == pytest.approx(0.875)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fov_counts": ()},
        {"fov_counts": (1,)},
        {"fov_counts": (5,), "draws": 0},
        {"novelty": "support_novelty"},
    ],
)
def test_invalid_settings_are_refused(kwargs):
    objects, fovs = _acquisition(0.1, n_fovs=10)
    with pytest.raises((TypeError, ValueError)):
        _calibrate(objects, fovs, **kwargs)
