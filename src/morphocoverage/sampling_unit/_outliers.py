"""The distances behind ``clustering.unusual_objects`` (private).

A reference sample of up to ``per_fov`` complete observations per reference
field (drawn in a canonical order, so the tables' row order never changes
it), standardized on that sample; the score of an observation is its mean
Euclidean distance to the ``k`` nearest reference observations. The
threshold is calibrated leaving out one reference group at a time (each
reference observation scored against the other reference groups, each tested
observation against all reference groups but one, drawn at random, so both
see an equally dense reference), or, with a single reference group, leaving
out one reference field at a time.

Shared-private: ``reference_sample``, ``score`` and the calibration labels
``LEAVE_ONE_GROUP_OUT`` and ``LEAVE_ONE_FIELD_OUT`` are used by ``clustering``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..schema import ValidatedInput, fov_positions
from ..transforms import Standardize
from ._common import canonical_fields

LEAVE_ONE_GROUP_OUT = "leave-one-group-out"
LEAVE_ONE_FIELD_OUT = "leave-one-field-out (a single reference group)"


def _by_values(matrix: np.ndarray) -> np.ndarray:
    """Row positions of ``matrix`` sorted by their values (lexicographic, first column first)."""
    if not len(matrix):
        return np.empty(0, dtype=np.int64)
    return np.lexsort(matrix.T[::-1])


def reference_sample(
    group_datas: Sequence[ValidatedInput], per_fov: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Up to ``per_fov`` complete observations of every reference FOV, their FOV and group ids.

    The draw walks the groups and their fields in a canonical order (groups
    by their sorted FOV keys, fields by FOV key) and each field's observations
    sorted by value, so the row order of the tables never changes which
    observations are drawn.
    """
    rows, fov_ids, group_ids, offset = [], [], [], 0
    orders = [canonical_fields(data) for data in group_datas]
    keys = [
        [repr(k) for k in data.fovs[list(data.fov_key)].iloc[order].itertuples(index=False)]
        for data, order in zip(group_datas, orders, strict=True)
    ]
    # Group ids are canonical ranks, so the left-out group each tested object gets is too.
    for rank, g in enumerate(sorted(range(len(group_datas)), key=keys.__getitem__)):
        data, order = group_datas[g], orders[g]
        matrix = data.feature_matrix()
        positions = fov_positions(data, data.fovs)
        complete = ~np.isnan(matrix).any(axis=1)
        for j in order:
            members = np.flatnonzero((positions == j) & complete)
            members = members[_by_values(matrix[members])]
            if len(members) > per_fov:
                members = members[np.sort(rng.choice(len(members), size=per_fov, replace=False))]
            rows.append(matrix[members])
            fov_ids.append(np.full(len(members), offset + j))
            group_ids.append(np.full(len(members), rank))
        offset += len(data.fovs)
    return np.concatenate(rows), np.concatenate(fov_ids), np.concatenate(group_ids)


def _mean_distance(tree: cKDTree, points: np.ndarray, k: int) -> np.ndarray:
    """The mean distance of each point to its ``k`` nearest objects of ``tree``."""
    if not len(points):
        return np.empty(0)
    distance, _ = tree.query(points, k=k)
    return np.asarray(distance).reshape(len(points), k).mean(axis=1)


def _leave_one_group_out(
    ref: np.ndarray, group_ids: np.ndarray, tested: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | str:
    """Reference and tested scores, each against the reference with one group left out.

    A reference object is scored against the other reference groups; a tested
    object against the reference without one group, drawn at random, so both
    see as many reference groups (as dense a reference) and their scores are
    alike when the tested group is like a reference group.
    """
    groups = np.unique(group_ids)
    trees = []
    reference_scores = np.empty(len(ref))
    for group in groups:
        inside = group_ids == group
        if int((~inside).sum()) < k:
            return f"fewer than {k} reference observations outside some reference group"
        trees.append(cKDTree(ref[~inside]))
        reference_scores[inside] = _mean_distance(trees[-1], ref[inside], k)
    # Assigned in the order of the tested values, so the row order never changes them.
    left_out = np.empty(len(tested), dtype=np.int64)
    left_out[_by_values(tested)] = rng.integers(0, len(groups), size=len(tested))
    scores = np.empty(len(tested))
    for index, tree in enumerate(trees):
        chosen = left_out == index
        if chosen.any():
            scores[chosen] = _mean_distance(tree, tested[chosen], k)
    return reference_scores, scores


def _leave_one_field_out(
    ref: np.ndarray, fov_ids: np.ndarray, tested: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | str:
    """Reference objects scored against other reference fields; tested ones against all."""
    tree = cKDTree(ref)
    largest = int(np.bincount(fov_ids).max())
    if len(ref) - largest < k:
        return f"fewer than {k} reference observations outside some reference FOV"
    distance, index = tree.query(ref, k=min(len(ref), k + largest))
    other = fov_ids[index] != fov_ids[:, None]
    reference_scores = np.array([d[o][:k].mean() for d, o in zip(distance, other, strict=True)])
    return reference_scores, _mean_distance(tree, tested, k)


def score(
    reference: np.ndarray,
    fov_ids: np.ndarray,
    group_ids: np.ndarray,
    tested: np.ndarray,
    k: int,
    quantile: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict[str, Any], str] | str:
    """Scores of ``tested``, the threshold, the fitted state and how it was calibrated.

    The threshold is the ``1 - quantile`` quantile of reference objects scored
    against the other reference groups, the tested objects being scored
    against all reference groups but one (``_leave_one_group_out``); with a
    single reference group, reference objects are scored against the other
    fields of that group and tested objects against all of it.
    """
    fitted = Standardize().fit_matrix(reference, tuple(str(i) for i in range(reference.shape[1])))
    if isinstance(fitted, str):
        return fitted
    state, _ = fitted
    ref = Standardize().apply_matrix(state, reference)
    complete = ~np.isnan(tested).any(axis=1)
    points = Standardize().apply_matrix(state, tested[complete])
    several = len(np.unique(group_ids)) > 1
    scored = (
        _leave_one_group_out(ref, group_ids, points, k, rng)
        if several
        else _leave_one_field_out(ref, fov_ids, points, k)
    )
    if isinstance(scored, str):
        return scored
    reference_scores, tested_scores = scored
    threshold = float(np.quantile(reference_scores, 1 - quantile))
    scores = np.full(len(tested), np.nan)
    scores[complete] = tested_scores
    calibration = LEAVE_ONE_GROUP_OUT if several else LEAVE_ONE_FIELD_OUT
    return scores, threshold, state, calibration
