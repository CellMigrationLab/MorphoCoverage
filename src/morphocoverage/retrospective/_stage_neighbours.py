"""The stage-neighbour reason of "Which fields should I look at?" (private): a field whose
mean of a feature differs from the means of its nearest fields on the stage.

Split from ``inspection`` (owner, 2026-10-06, audit 10) to keep that module under 600
lines; the reason and its thresholds are described there. Shared-private: used by
``retrospective._field_reasons``, which holds inspection's reasons.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .._identity import canonical_order
from .._stage_coordinates import stage_matrix
from .._wording import count
from ..schema import ValidatedInput, fov_positions

_TIED = "; fields tied in distance taken by field name"


def stage_skipped(fovs: pd.DataFrame, x_col: str, y_col: str, k: int) -> str | None:
    """Why stage neighbours cannot be compared in this group, or None."""
    if len(fovs) <= k:
        return f"{len(fovs)} fields, not more than the {k} neighbours compared"
    if fovs[[x_col, y_col]].isna().any().any():
        return "stage coordinates are missing"
    return None


def _neighbour_words(valued: np.ndarray, k: int) -> str:
    """``its 4 nearest fields``, or ``3 of its 4 nearest fields (1 without a value)``."""
    used = int(valued.sum())
    if used == k:
        return f"its {k} nearest fields"
    return f"{used} of its {k} nearest fields ({count(k - used, 'field')} without a value)"


def _nearest(
    fovs: pd.DataFrame, xy: np.ndarray, fov_key: Sequence[str], k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each field's ``k`` nearest other fields, ties in distance broken by field identity.

    On a regular grid several fields sit at the same distance; taking them in
    row order made which were compared depend on the order of the FOV table.
    Distances equal to 1e-9 of the largest are ties, broken by the canonical
    order of the FOV key (its typed values, so ``1`` and ``"1"`` never tie).
    Returns the distances and positions of the neighbours, and whether a tie
    at the k-th distance was broken.
    """
    n = len(fovs)
    rank = np.empty(n, dtype=np.int64)
    rank[canonical_order(fovs, list(fov_key))] = np.arange(n)  # typed: 1 is not "1" (#237)
    distance = cdist(xy, xy)
    scale = float(distance.max()) or 1.0
    rounded = np.round(distance / scale, 9)
    near = np.empty((n, k), dtype=np.int64)
    tied = np.zeros(n, dtype=bool)
    for fov in range(n):
        # Never the FOV itself, even when another field shares its stage position.
        others = np.delete(np.arange(n), fov)
        order = others[np.lexsort((rank[others], rounded[fov, others]))]
        near[fov] = order[:k]
        tied[fov] = len(order) > k and rounded[fov, order[k]] == rounded[fov, order[k - 1]]
    return np.take_along_axis(distance, near, axis=1), near, tied


def neighbour_rows(
    data: ValidatedInput, x_col: str, y_col: str, k: int, threshold: float, unit: str
) -> list[dict]:
    fovs = data.fovs
    xy = stage_matrix(fovs, (x_col, y_col), missing="keep")
    if stage_skipped(fovs, x_col, y_col, k) is not None:
        return []
    positions = fov_positions(data, fovs)
    frame = pd.DataFrame(data.feature_matrix(), columns=list(data.feature_cols))
    means = frame.groupby(positions).mean().reindex(range(len(fovs)))
    distances, near, tied = _nearest(fovs, xy, data.fov_key, k)
    rows = []
    for feature in data.feature_cols:
        column = means[feature].to_numpy(dtype=float)
        for fov in range(len(fovs)):
            others = column[near[fov]]
            if np.isnan(column[fov]) or np.isnan(others).all():
                continue
            # An empty neighbour has no mean: the comparison is with the others only.
            valued = ~np.isnan(others)
            reach = distances[fov][valued].max()
            # Spread of the other FOVs' means: including the FOV itself would cap
            # its own z at sqrt(n) and hide a lone deviant field in a small group.
            spread = np.nanstd(np.delete(column, fov), ddof=1)
            if not np.isfinite(spread) or spread == 0:
                continue
            z = (column[fov] - np.nanmean(others)) / spread
            if abs(z) >= threshold:
                rows.append(
                    {
                        "position": fov,
                        "reason": "neighbours",
                        "value": float(z),
                        "detail": (
                            f"its mean {feature} is {abs(z):.1f} SD "
                            f"{'above' if z > 0 else 'below'} {_neighbour_words(valued, k)} "
                            f"(within {reach:.0f} {unit}{_TIED if tied[fov] else ''})"
                        ),
                    }
                )
    return rows


__all__ = ["neighbour_rows", "stage_skipped"]
