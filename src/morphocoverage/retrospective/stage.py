"""Stage geometry of an acquisition order: dispersed orders and stage travel."""

from __future__ import annotations

import numpy as np


def dispersed_order(xy: np.ndarray, start: int) -> np.ndarray:
    """Indices of ``xy`` in greedy farthest-point order from ``start`` (ties: lower index)."""
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    if not 0 <= start < n:
        raise ValueError("start must index a row of xy")
    if np.isnan(xy).any():
        raise ValueError("stage coordinates must not be missing")
    order = [int(start)]
    nearest = np.linalg.norm(xy - xy[start], axis=1)
    nearest[start] = -np.inf
    for _ in range(n - 1):
        chosen = int(np.argmax(nearest))
        order.append(chosen)
        nearest = np.minimum(nearest, np.linalg.norm(xy - xy[chosen], axis=1))
        nearest[order] = -np.inf
    return np.array(order, dtype=int)


def stage_travel(xy_in_order: np.ndarray) -> np.ndarray:
    """Cumulative stage path after each FOV, in the units of the coordinates (0 at the first)."""
    xy = np.asarray(xy_in_order, dtype=float)
    if len(xy) == 0:
        return np.array([])
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


__all__ = ["dispersed_order", "stage_travel"]
