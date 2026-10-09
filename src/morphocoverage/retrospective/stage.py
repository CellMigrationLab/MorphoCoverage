"""Stage geometry of an acquisition order: dispersed orders and stage travel."""

from __future__ import annotations

import numpy as np

from .._checks import is_integer
from .._stage_coordinates import check_xy


def dispersed_order(xy: np.ndarray, start: int) -> np.ndarray:
    """Indices of ``xy`` in greedy farthest-point order from ``start`` (ties: lower index)."""
    if not is_integer(start):
        raise TypeError("start must be a whole number (a row of xy)")
    xy = check_xy(xy, "xy") if len(xy) else np.empty((0, 2))
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    if not 0 <= start < n:
        raise ValueError("start must index a row of xy")
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
    xy = check_xy(xy_in_order, "xy_in_order") if len(xy_in_order) else np.empty((0, 2))
    if len(xy) == 0:
        return np.array([])
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


__all__ = ["dispersed_order", "stage_travel"]
