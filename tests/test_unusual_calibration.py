"""Unusual objects: one leave-out for calibration and scoring (#99, #286; owner, 2026-10-05).

Every left-out reference group (with a single reference group, every field) weighs the same in
the threshold, as in the uniform draw a tested observation leaves out. The rates are those of
``data/simulations/unusual_calibration.py`` on fewer datasets.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from morphocoverage.sampling_unit._outliers import (
    LEAVE_ONE_FIELD_OUT,
    LEAVE_ONE_GROUP_OUT,
    _weighted_quantile,
    score,
)

K = 5


def _reference(rng: np.random.Generator, fields_per_group: tuple[int, ...]) -> tuple:
    """Reference observations (40 a field, 2 features), their field and group ids."""
    rows, fov_ids, group_ids, field = [], [], [], 0
    for group, fields in enumerate(fields_per_group):
        for _ in range(fields):
            rows.append(rng.normal(0, 0.3, 2) + rng.normal(0, 1, (40, 2)))
            fov_ids.append(np.full(40, field))
            group_ids.append(np.full(40, group))
            field += 1
    return np.concatenate(rows), np.concatenate(fov_ids), np.concatenate(group_ids)


def _tested(rng: np.random.Generator) -> np.ndarray:
    """A tested group of 10 fields drawn like the reference."""
    return np.concatenate([rng.normal(0, 0.3, 2) + rng.normal(0, 1, (40, 2)) for _ in range(10)])


def _null_share(fields_per_group: tuple[int, ...], replicates: int) -> float:
    """The mean share of a null tested group called unusual at a nominal 5%."""
    rng = np.random.default_rng(99)
    shares = []
    for seed in range(replicates):
        reference, fov_ids, group_ids = _reference(rng, fields_per_group)
        scored = score(
            reference, fov_ids, group_ids, _tested(rng), K, 0.05, np.random.default_rng(seed)
        )
        shares.append(np.mean(scored[0] > scored[1]))
    return float(np.mean(shares))


def test_unequal_reference_groups_hold_the_share_asked_for():
    # Groups of 3, 10 and 30 fields: the threshold weighed the big group's scores (against the
    # sparse rest) by its size while tested objects left a group out uniformly, so 3.5% of a
    # null group was called unusual for a nominal 5%.
    assert 0.046 < _null_share((3, 10, 30), 120) < 0.054


def test_a_single_reference_group_scores_tested_objects_at_its_calibration_density():
    # Four fields: tested objects scored against every field (a denser pool than the
    # calibration's three) were called unusual 4.0% of the time for a nominal 5%. Left one
    # field out, the share is 5.5%, the excess np.quantile's at 160 reference observations.
    assert 0.051 < _null_share((4,), 200) < 0.06


def _own_group_left_out(reference: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Each standardized reference observation's score with its own group left out."""
    ref = (reference - reference.mean(axis=0)) / reference.std(axis=0, ddof=1)
    scores = np.empty(len(ref))
    for group in np.unique(group_ids):
        inside = group_ids == group
        scores[inside] = cKDTree(ref[~inside]).query(ref[inside], k=K)[0].mean(axis=1)
    return scores


def test_the_threshold_weighs_every_left_out_group_the_same():
    rng = np.random.default_rng(1)
    reference, fov_ids, group_ids = _reference(rng, (2, 6))
    _, threshold, _, calibration = score(
        reference, fov_ids, group_ids, _tested(rng), K, 0.05, np.random.default_rng(0)
    )
    assert calibration == LEAVE_ONE_GROUP_OUT
    own = _own_group_left_out(reference, group_ids)
    weights = np.where(group_ids == 0, 1 / 80, 1 / 240)
    assert threshold == pytest.approx(_weighted_quantile(own, weights, 0.95))
    assert threshold != pytest.approx(np.quantile(own, 0.95))
    # Equal groups weigh their observations equally: np.quantile, as before.
    reference, fov_ids, group_ids = _reference(rng, (4, 4))
    _, threshold, _, _ = score(
        reference, fov_ids, group_ids, _tested(rng), K, 0.05, np.random.default_rng(0)
    )
    assert threshold == np.quantile(_own_group_left_out(reference, group_ids), 0.95)


def test_weighted_quantile_is_numpys_with_equal_weights():
    values = np.random.default_rng(0).normal(size=37)
    for q in (0.0, 0.5, 0.95, 1.0):
        assert _weighted_quantile(values, np.full(37, 0.25), q) == np.quantile(values, q)
    # A value of weight 3 counts as three values of weight 1.
    values = np.array([1.0, 2.0, 3.0])
    tripled = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    assert _weighted_quantile(values, np.array([1.0, 3.0, 1.0]), 0.3) == pytest.approx(
        np.quantile(tripled, 0.3), abs=0.25
    )


def test_single_group_tested_objects_leave_one_field_out_reproducibly():
    rng = np.random.default_rng(2)
    reference, fov_ids, group_ids = _reference(rng, (5,))
    tested = _tested(rng)
    scores, _, _, calibration = score(
        reference, fov_ids, group_ids, tested, K, 0.05, np.random.default_rng(0)
    )
    assert calibration == LEAVE_ONE_FIELD_OUT
    ref = (reference - reference.mean(axis=0)) / reference.std(axis=0, ddof=1)
    points = (tested - reference.mean(axis=0)) / reference.std(axis=0, ddof=1)
    per_field = np.array(
        [cKDTree(ref[fov_ids != f]).query(points, k=K)[0].mean(axis=1) for f in np.unique(fov_ids)]
    )
    # Each tested object is scored against the reference without one field, never all of it.
    matches = np.isclose(per_field, scores)
    assert matches.any(axis=0).all()
    whole = cKDTree(ref).query(points, k=K)[0].mean(axis=1)
    assert np.mean(scores > whole + 1e-12) > 0.5
    # The fields are left out about equally often.
    left_out = np.bincount(matches.argmax(axis=0), minlength=5)
    assert left_out.min() > 40
    # The row order of the tested table does not change any object's score.
    order = np.random.default_rng(3).permutation(len(tested))
    again = score(reference, fov_ids, group_ids, tested[order], K, 0.05, np.random.default_rng(0))
    assert np.array_equal(again[0], scores[order])
