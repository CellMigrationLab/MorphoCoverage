"""The clusters of a ``plots.plots_of_data`` figure (private): one cluster drawn (its
points by technical repeat, its summary and resampled interval), and each cluster's
top-level units and technical repeats.

``layout`` is the figure's ``plots._Layout``. Shared-private: ``_mask``, ``_draw_cluster``
and ``_cluster_units``, used by ``plots``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ._palette import INK, MARKERS, OTHER_MARKER
from .statistics import bootstrap_nested


def _mask(table: pd.DataFrame, column: str | None, value: Any) -> np.ndarray:
    """The rows holding ``value`` in ``column``; every row when either is absent."""
    if column is None or value is None:
        return np.ones(len(table), dtype=bool)
    return table[column].to_numpy() == value


def _unit_keys(frame: pd.DataFrame, mask: np.ndarray, layout: Any) -> list[tuple[Any, Any]]:
    """``(top-level unit, technical repeat)`` of each row of ``frame`` in ``mask``.

    The top-level unit is the tuple of the layout's pooled columns (empty
    when nothing is pooled), so a technical repeat is identified within it:
    well ``B02`` of two pooled repeats is two technical repeats.
    """
    rows = frame.loc[mask]
    tops = list(zip(*(rows[c] for c in layout.pooled_cols), strict=True))
    tops = tops if layout.pooled_cols else [()] * len(rows)
    technical = rows[layout.technical_col] if layout.technical_col else [None] * len(rows)
    return list(zip(tops, technical, strict=True))


def _unit_mask(frame: pd.DataFrame, layout: Any, top: tuple, replicate: Any) -> np.ndarray:
    """The rows of ``frame`` in one ``(top-level unit, technical repeat)``."""
    mask = _mask(frame, layout.technical_col, replicate)
    for column, value in zip(layout.pooled_cols, top, strict=True):
        mask &= frame[column].to_numpy() == value
    return mask


def _draw_cluster(
    ax: Axes,
    x: float,
    width: float,
    rows: np.ndarray,
    values: np.ndarray,
    fields: tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray], np.ndarray],
    layout: Any,
    colour: str,
    statistic: Callable[[np.ndarray], float],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """One cluster (a condition and biological repeat): its points in the condition's
    ``colour``, each technical repeat by shape, then the summary and interval. Returns the
    summary and its interval.

    ``rows`` is the boolean mask of the cluster's observations; ``fields``
    holds every acquired field's metadata, every observation's (its field's),
    every field's values, and the boolean mask of the cluster's fields, empty
    ones included.
    """
    meta, observed, per_field, in_cluster = fields
    # Units in the order their observations come, then units whose fields are all empty.
    units = dict.fromkeys(
        [*_unit_keys(observed, rows, layout), *_unit_keys(meta, in_cluster, layout)]
    )
    half = 0.4 * width
    by_top: dict[tuple, list[list[np.ndarray]]] = {}
    for top, replicate in units:
        # Shapes by rank within the top-level unit (the biological repeat), as the legend says.
        rank = len(by_top.setdefault(top, []))
        unit_fields = in_cluster & _unit_mask(meta, layout, top, replicate)
        by_top[top].append([per_field[i] for i in np.flatnonzero(unit_fields)])
        y = values[rows & _unit_mask(observed, layout, top, replicate)]
        y = y[~np.isnan(y)]
        if not len(y):
            continue
        marker = MARKERS[rank] if rank < len(MARKERS) else OTHER_MARKER
        jitter = rng.uniform(-half, half, size=len(y))
        ax.scatter(
            x + jitter,
            y,
            s=16,
            alpha=0.5,
            color=colour,
            marker=marker,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    tops = [t for t in by_top.values() if any(len(f) for unit in t for f in unit)]
    if len(tops) > 1:
        # Pooled repeats: the repeats alone resampled, all their fields as acquired (Ren 2010).
        units_resampled = [[f for unit in top for f in unit] for top in tops]
    else:
        units_resampled = tops[0] if tops else [[]]
    point, low, high = bootstrap_nested(
        units_resampled, statistic, n_boot=n_boot, rng=rng, confidence=confidence
    )
    if np.isfinite(low) and np.isfinite(high):
        # The interval as an error bar with caps, so a narrow one stays visible.
        ax.vlines(x, low, high, color=INK, lw=1.8, zorder=3)
        ax.hlines([low, high], x - 0.13 * width, x + 0.13 * width, color=INK, lw=1.8, zorder=3)
    if np.isfinite(point):
        ax.hlines(point, x - 0.34 * width, x + 0.34 * width, color=INK, lw=3, zorder=4)
    return float(point), float(low), float(high)


def _cluster_units(observed: pd.DataFrame, layout: Any) -> list[tuple[int, list[int]]]:
    """Per cluster: its top-level units with observations, and each one's technical repeats."""
    shown = [layout.condition_col] + ([] if layout.pooled_repeats else [layout.repeat_col])
    cluster_cols = [c for c in shown if c is not None]
    top_cols = [*cluster_cols, *layout.pooled_cols]
    clusters = observed.groupby(cluster_cols, sort=False) if cluster_cols else [((), observed)]
    result = []
    for _, cluster in clusters:
        tops = cluster.groupby(top_cols, sort=False) if top_cols else [((), cluster)]
        technical = [
            int(top[layout.technical_col].nunique()) if layout.technical_col else 0
            for _, top in tops
        ]
        result.append((len(technical), technical))
    return result
