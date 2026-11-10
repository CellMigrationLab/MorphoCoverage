"""Tests of ``retrospective.effect_preservation``: the difference from the control at k fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import EMPTY

from morphocoverage.retrospective import EffectPreservation, effect_preservation
from morphocoverage.retrospective.effect_preservation import (
    BEYOND,
    NO_CONTROL,
    OK,
    SHALLOW,
    UNCLEAR,
)

CONTROL = {"condition": "ctrl"}


def _tables(
    effect: float,
    *,
    n_fields: int = 30,
    per_field: int = 20,
    field_sd: float = 0.1,
    repeats: tuple[str, ...] = ("r1",),
    conditions: tuple[str, ...] = ("drug",),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One control and the given conditions per repeat, each shifted by ``effect``."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for repeat in repeats:
        for condition in ("ctrl", *conditions):
            shift = 0.0 if condition == "ctrl" else effect
            for i in range(n_fields):
                name = f"{repeat}_{condition}_{i}"
                labels = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "position_id": name,
                }
                fovs.append(labels)
                mean = shift + rng.normal(0, field_sd)
                objects += [{**labels, "x": v} for v in rng.normal(mean, 1.0, per_field)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _run(objects: pd.DataFrame, fovs: pd.DataFrame, **kwargs: object) -> EffectPreservation:
    settings = {"feature": "x", "control": CONTROL, "n_draws": 60, "n_splits": 3, **kwargs}
    return effect_preservation(objects, fovs, **settings)


def test_a_planted_effect_is_preserved_more_often_with_more_fields():
    objects, fovs = _tables(0.6)
    result = _run(objects, fovs, fov_counts=(1, 3, 6, 12))
    rows = result.table.set_index("fovs")
    assert (rows["status"] == OK).all()
    shares = rows["share"].to_numpy()
    assert np.all(np.diff(shares) > 0), shares
    assert shares[0] < 0.6 and shares[-1] > 0.8
    assert (rows["draws"] == 60 * 3).all()
    assert rows["reference_d"].between(0.4, 0.8).all()
    assert (rows["stream_fovs"] == 15).all() and (rows["reference_fovs"] == 15).all()


def test_no_effect_gives_not_evaluable_or_low_agreement():
    objects, fovs = _tables(0.0, repeats=("r1", "r2", "r3"))
    table = _run(objects, fovs, fov_counts=(2, 5, 10)).table
    assert set(table["status"]) <= {OK, UNCLEAR}
    assert (table["status"] == UNCLEAR).any()
    judged = table[table["status"] == OK]
    assert (judged["share"] < 0.5).all()
    # a held-back d that is a fluke is matched less often as more fields approach 0
    for _, rows in judged.groupby("biological_repeat"):
        assert rows["share"].iloc[-1] <= rows["share"].iloc[0]
    assert table.loc[table["status"] == UNCLEAR, "share"].isna().all()


def test_the_reference_is_held_back_and_never_shrinks_with_k():
    objects, fovs = _tables(0.6, n_fields=10)
    result = _run(objects, fovs, fov_counts=(1, 2, 5, 6))
    rows = result.table.set_index("fovs")
    # 5 of 10 fields a side are held back: 5 remain to draw from, so 6 is beyond them
    assert rows.loc[6, "status"] == BEYOND and np.isnan(rows.loc[6, "share"])
    assert rows["reference_fovs"].eq(5).all()
    assert rows.loc[[1, 2, 5], "reference_d"].nunique() == 1
    assert len(result.references) == 3
    assert set(result.references["status"]) == {OK}


def test_too_few_held_back_fields_is_not_evaluable():
    objects, fovs = _tables(2.0, n_fields=2)
    rows = _run(objects, fovs, fov_counts=(1,)).table
    assert rows["status"].tolist() == [SHALLOW]


def test_a_repeat_without_its_control_is_named():
    objects, fovs = _tables(0.6, repeats=("r1", "r2"))
    keep = ~((fovs["biological_repeat"] == "r2") & (fovs["condition"] == "ctrl"))
    names = set(fovs.loc[keep, "position_id"])
    objects = objects[objects["position_id"].isin(names)]
    table = _run(objects, fovs[keep], fov_counts=(2,)).table
    assert table.set_index("biological_repeat").loc["r2", "status"] == NO_CONTROL


def test_the_same_seed_gives_the_same_result_and_another_seed_other_draws():
    objects, fovs = _tables(0.4, field_sd=0.3)
    first = _run(objects, fovs, random_state=3).table
    again = _run(objects, fovs, random_state=3).table
    other = _run(objects, fovs, random_state=4).table
    pd.testing.assert_frame_equal(first, again)
    assert not first["share"].equals(other["share"])


def test_draws_depend_on_neither_row_order_other_conditions_nor_other_counts():
    objects, fovs = _tables(0.4, field_sd=0.3, conditions=("drug", "other"))
    alone_objects = objects[objects["condition"] != "other"]
    alone_fovs = fovs[fovs["condition"] != "other"]
    full = _run(objects, fovs, fov_counts=(2, 5)).table
    shuffled = _run(
        objects.sample(frac=1, random_state=1),
        fovs.sample(frac=1, random_state=2),
        fov_counts=(2, 5),
    ).table
    pd.testing.assert_frame_equal(full, shuffled)
    alone = _run(alone_objects, alone_fovs, fov_counts=(5,)).table
    drug = full[(full["condition"] == "drug") & (full["fovs"] == 5)].reset_index(drop=True)
    pd.testing.assert_frame_equal(drug, alone)


def test_the_tolerance_is_a_parameter():
    objects, fovs = _tables(0.6, field_sd=0.3)
    strict = _run(objects, fovs, fov_counts=(3,), tolerance=0.05).table["share"].iloc[0]
    loose = _run(objects, fovs, fov_counts=(3,), tolerance=1.0).table["share"].iloc[0]
    assert strict < loose
    with pytest.raises(ValueError, match="tolerance"):
        _run(objects, fovs, tolerance=-0.1)


def test_a_setup_fills_what_is_left_out():
    from morphocoverage.workflows import Setup

    objects, fovs = _tables(0.6)
    setup = Setup(readout="x", control=CONTROL, random_state=2)
    from_setup = effect_preservation(
        objects, fovs, setup=setup, fov_counts=(3,), n_draws=60, n_splits=3
    )
    explicit = _run(objects, fovs, fov_counts=(3,), random_state=2)
    pd.testing.assert_frame_equal(from_setup.table, explicit.table)


def test_the_summary_states_the_shares_and_what_each_status_means():
    objects, fovs = _tables(0.6, n_fields=10)
    text = _run(objects, fovs, fov_counts=(2, 6)).summary()
    assert "Feature 'x'" in text and "within 0.2" in text
    assert "more fields a side than remain outside the held-back ones" in text
    assert "question 9" in text
    assert all(len(line) <= 100 for line in text.splitlines())
    null_objects, null_fovs = _tables(0.0)
    null = _run(null_objects, null_fovs, fov_counts=(2,))
    assert null.table["status"].tolist() == [UNCLEAR]
    assert "unclear: the held-back fields do not establish the difference" in null.summary()


def test_invalid_settings_raise():
    objects, fovs = _tables(0.6, n_fields=4)
    with pytest.raises(ValueError, match="fov_counts"):
        _run(objects, fovs, fov_counts=(0,))
    with pytest.raises(ValueError, match="reservoir_fraction"):
        _run(objects, fovs, reservoir_fraction=1.0)
    with pytest.raises(TypeError, match="feature"):
        effect_preservation(objects, fovs, control=CONTROL)
    with pytest.raises(ValueError, match="control"):
        effect_preservation(objects, fovs, feature="x")


def test_effect_preservation_checks_the_control_before_the_tables():
    with pytest.raises(ValueError) as raised:
        effect_preservation(EMPTY, EMPTY, feature="x", control={"well": "A1"})
    assert str(raised.value) == "control must name the condition column 'condition'"
    with pytest.raises(TypeError) as raised:
        effect_preservation(EMPTY, EMPTY, feature="x", comparisons="not pairs")
    assert str(raised.value) == "comparisons must be a sequence of (condition, reference) pairs"
    # Valid comparisons replace the control: the tables are read next.
    with pytest.raises(ValueError, match="^Could not resolve a spatial FOV/position identifier"):
        effect_preservation(EMPTY, EMPTY, feature="x", comparisons=[("a", "b")])
