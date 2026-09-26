"""Spatial diagnostics: how similar are FOVs as a function of stage distance?

Roadmap 8.2 (docs/proposals/candidate_estimators_roadmap.md). The FOV is the
acquisition unit, not automatically an independent one. The empirical
semivariogram

    gamma(h) = 1/2 * mean over FOV pairs at distance ~h of (Z_i - Z_j)^2

of an FOV-level summary ``Z`` shows the distance over which neighbouring FOVs
stay alike: where it levels off (the range) is a scale for spatial blocks in
``reservoir.spatial_blocks`` and the block bootstrap. A flat semivariogram
means no detectable spatial structure at the sampled lags; one that keeps
rising means a trend across the whole sample.

These are diagnostics, not stopping rules, and use every FOV passed in: run
them retrospectively or on the visible prefix, never to tune a prospective
estimator on future FOVs. Pairs are binned by distance; bins without pairs
are omitted rather than reported as NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import distance

from .schema import ValidatedInput, fov_positions


def empirical_semivariogram(
    values: np.ndarray, coords: np.ndarray, bin_edges: Sequence[float]
) -> pd.DataFrame:
    """Matheron's semivariogram estimator in distance bins ``[edge_i, edge_{i+1})``.

    ``values`` is ``(n,)`` or ``(n, d)``; with several columns the squared
    difference is averaged over columns. ``coords`` is ``(n, 2)`` (or any
    number of dimensions). Returns one row per non-empty bin: its edges, the
    mean pair distance, ``gamma`` and the number of pairs.
    """
    values = np.asarray(values, dtype=float)
    values = values[:, None] if values.ndim == 1 else values
    coords = np.asarray(coords, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    if len(values) != len(coords) or len(values) < 2:
        raise ValueError("values and coords need the same number (at least 2) of rows")
    if np.isnan(values).any() or np.isnan(coords).any():
        raise ValueError("values and coords must not contain missing values")
    # An infinite coordinate or value would drop its pairs from every bin without a word.
    if np.isinf(values).any() or np.isinf(coords).any():
        raise ValueError(
            "values and coords must be finite: an infinite stage coordinate or value has no "
            "distance to the other FOVs; correct or drop that FOV"
        )
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("bin_edges must be at least two increasing non-negative distances")
    lags = distance.pdist(coords)
    half_squared = 0.5 * distance.pdist(values, "sqeuclidean") / values.shape[1]
    bins = np.searchsorted(edges, lags, side="right") - 1
    inside = (bins >= 0) & (bins < len(edges) - 1)
    rows = []
    for b in np.unique(bins[inside]):
        members = inside & (bins == b)
        rows.append(
            {
                "lag_low": float(edges[b]),
                "lag_high": float(edges[b + 1]),
                "mean_lag": float(lags[members].mean()),
                "gamma": float(half_squared[members].mean()),
                "n_pairs": int(members.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["lag_low", "lag_high", "mean_lag", "gamma", "n_pairs"])


def _require_one_group(data: ValidatedInput) -> None:
    """Raise when ``data`` spans several analysis groups (conditions, repeats, wells)."""
    groups = list(data.group_cols)
    if groups and len(data.fovs[groups].drop_duplicates()) > 1:
        raise ValueError(
            f"the tables hold several analysis groups ({', '.join(groups)}); pairing fields "
            "across them by stage distance would mix samples: pass one group at a time "
            "(evaluator.iter_analysis_groups)"
        )


def fov_means(data: ValidatedInput) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation mean of every selected feature in each FOV with observations.

    Returns ``(means, fov_rows)``: ``means`` is ``(n_fovs_with_observations, d)``
    and ``fov_rows`` gives each mean's row in ``data.fovs``. Missing feature
    values raise, since an FOV mean over them would be undefined. ``data``
    must hold one analysis group (``evaluator.iter_analysis_groups``), since
    fields of different samples are not neighbours on one stage.
    """
    _require_one_group(data)
    matrix = data.feature_matrix()
    if np.isnan(matrix).any():
        raise ValueError("missing feature values; FOV means are undefined")
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions, minlength=len(data.fovs))
    sums = np.zeros((len(data.fovs), matrix.shape[1]))
    np.add.at(sums, positions, matrix)
    rows = np.flatnonzero(counts)
    return sums[rows] / counts[rows, None], rows


def fov_semivariogram(
    data: ValidatedInput, x_col: str, y_col: str, bin_edges: Sequence[float]
) -> pd.DataFrame:
    """Semivariogram of the FOV means of the selected features against stage distance.

    FOVs without observations have no mean and are left out. Standardize or
    select features beforehand if they are on different scales, since the
    squared differences are averaged over features. ``data`` must hold one
    analysis group (see ``fov_means``).
    """
    for column in (x_col, y_col):
        if column not in data.fovs.columns:
            raise ValueError(f"{column!r} is not a column of the FOV table")
    means, rows = fov_means(data)
    coords = data.fovs[[x_col, y_col]].to_numpy(dtype=float)[rows]
    return empirical_semivariogram(means, coords, bin_edges)


__all__ = ["empirical_semivariogram", "fov_means", "fov_semivariogram"]
