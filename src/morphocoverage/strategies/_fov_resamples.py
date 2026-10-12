"""The resample arithmetic of the FOV bootstrap (``fov_bootstrap``).

Pure functions of plain arrays: each summary of every resample at once, from
per-FOV sums and a matrix of how often each FOV was drawn. ``fov_bootstrap``
re-exports the public names.

Shared-private: ``_per_fov_sums`` is used by ``fov_bootstrap``; it is private
to the package and may change with it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import linalg


def _per_fov_sums(values: np.ndarray, positions: np.ndarray, k: int) -> np.ndarray:
    """Sum of each row of ``values`` over the observations of each of ``k`` FOVs."""
    sums = np.zeros((k, values.shape[1]))
    np.add.at(sums, positions, values)
    return sums


def centroid_resamples(counts: np.ndarray, sums: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pooled mean of each resample; ``weights[b, j]`` is how often FOV ``j`` was drawn.

    Rows of resamples without observations are NaN.
    """
    n = weights @ counts
    # fallback: a resample without observations: NaN (documented)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ sums) / n[:, None]


def covariance_resamples(
    counts: np.ndarray, sums: np.ndarray, outer: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Pooled covariance (``ddof=1``) of each resample, flattened; NaN below 2 observations."""
    n = weights @ counts
    total, cross = weights @ sums, weights @ outer
    mean_outer = np.einsum("bi,bj->bij", total, total).reshape(len(weights), -1)
    # fallback: below 2 observations: NaN (documented)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cross - mean_outer / n[:, None]) / (n - 1)[:, None]


def quantile_resamples(
    values: np.ndarray, positions: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Per-feature quantiles of the pooled observations of each resample, flattened.

    ``weights[b, j]`` is how often FOV ``j`` was drawn in resample ``b``.
    Resamples without observations are NaN. One ``np.quantile`` call per
    resample.
    """
    k = weights.shape[1]
    by_fov = np.argsort(positions, kind="stable")
    rows_of = np.split(by_fov, np.cumsum(np.bincount(positions, minlength=k))[:-1])
    out = np.full((len(weights), len(quantiles) * values.shape[1]), np.nan)
    for b, counts in enumerate(weights.astype(int)):
        rows = np.concatenate([rows_of[j] for j in np.repeat(np.arange(k), counts)])
        if len(rows):
            out[b] = np.quantile(values[rows], quantiles, axis=0).ravel()
    return out


def eigenspectrum_resamples(covariances: np.ndarray, d: int) -> np.ndarray:
    """Explained-variance fractions, largest first, of each flattened ``d x d`` covariance."""
    eigenvalues = np.clip(np.linalg.eigvalsh(covariances.reshape(-1, d, d))[:, ::-1], 0, None)
    # A resample with no spread at all has undefined fractions (NaN, left out later).
    # fallback: a resample without spread: NaN, left out later
    with np.errstate(invalid="ignore", divide="ignore"):
        return eigenvalues / eigenvalues.sum(axis=1, keepdims=True)


def subspace_angles_to(covariances: np.ndarray, reference: np.ndarray, r: int) -> np.ndarray:
    """Largest principal angle (radians) between leading ``r``-dimensional subspaces.

    One angle per flattened ``d x d`` matrix in ``covariances``, measured
    against the leading subspace of ``reference``.
    """
    d = len(reference)
    target = np.linalg.eigh(reference)[1][:, ::-1][:, :r]
    vectors = np.linalg.eigh(covariances.reshape(-1, d, d))[1][:, :, ::-1][:, :, :r]
    return np.array([linalg.subspace_angles(v, target).max() for v in vectors])


__all__ = [
    "centroid_resamples",
    "covariance_resamples",
    "eigenspectrum_resamples",
    "quantile_resamples",
    "subspace_angles_to",
]
