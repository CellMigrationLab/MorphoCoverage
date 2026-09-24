"""Distances between two samples of phenotype vectors: pure functions, no framework types.

Each function takes two 2-D arrays ``(n_observations, n_features)`` with the
same number of features and no missing values; callers own validation,
scaling and subsampling. They are statistical primitives shared by outcome
metrics and, later, by estimators (AGENTS.md section 6).

Small-sample bias matters when a distance is compared across stops with
different numbers of observations: a biased estimate shrinks as observations
accumulate even when no information is gained. The energy distance and MMD
here are therefore the unbiased U-statistic estimates, which can be slightly
negative when the two distributions are equal. The Fréchet distance uses
plug-in means and covariances and remains biased upward for small samples.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from scipy.spatial import distance


def _pair(x: np.ndarray, y: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    """Two float matrices with matching feature counts and at least ``minimum`` rows each."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2-D arrays with the same number of features")
    if len(x) < minimum or len(y) < minimum:
        raise ValueError(f"each sample needs at least {minimum} observations")
    return x, y


def _off_diagonal_mean(pairwise: np.ndarray) -> float:
    """Mean over i != j of a square pairwise matrix."""
    n = len(pairwise)
    return float((pairwise.sum() - np.trace(pairwise)) / (n * (n - 1)))


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Fréchet (2-Wasserstein between Gaussians) distance of the samples' means and covariances.

    ``sqrt(|mu_x - mu_y|^2 + Tr(S_x + S_y - 2 (S_x^1/2 S_y S_x^1/2)^1/2))``, with
    sample covariances (``ddof=1``). It ignores structure beyond the second
    moment.
    """
    x, y = _pair(x, y, 2)
    cov_x = np.atleast_2d(np.cov(x, rowvar=False))
    cov_y = np.atleast_2d(np.cov(y, rowvar=False))
    root_x = linalg.sqrtm(cov_x)
    cross = linalg.sqrtm(root_x @ cov_y @ root_x)
    trace = np.trace(cov_x + cov_y - 2 * np.real(cross))
    squared = float(np.sum((x.mean(axis=0) - y.mean(axis=0)) ** 2) + trace)
    # sqrtm rounding leaves a residual of order 1e-15 * scale for identical
    # samples, which the square root would magnify; treat it as zero.
    scale = float(np.trace(cov_x) + np.trace(cov_y)) + 1.0
    return float(np.sqrt(squared)) if squared > 1e-12 * scale else 0.0


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased U-statistic estimate of the energy distance ``2E|X-Y| - E|X-X'| - E|Y-Y'|``.

    Euclidean distances; within-sample terms average over distinct pairs only.
    Quadratic in the number of observations.
    """
    x, y = _pair(x, y, 2)
    between = distance.cdist(x, y).mean()
    within_x = _off_diagonal_mean(distance.squareform(distance.pdist(x)))
    within_y = _off_diagonal_mean(distance.squareform(distance.pdist(y)))
    return float(2 * between - within_x - within_y)


def mmd_squared(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Unbiased estimate of squared MMD with the Gaussian kernel ``exp(-|a-b|^2 / (2 bw^2))``.

    Gretton et al. (JMLR 2012), equation 3. Quadratic in the number of
    observations. ``bandwidth`` is in the units of the supplied features.
    """
    x, y = _pair(x, y, 2)
    if not (bandwidth > 0 and np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be positive and finite")
    scale = -1.0 / (2.0 * bandwidth**2)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(scale * distance.cdist(a, b, "sqeuclidean"))

    within = _off_diagonal_mean(kernel(x, x)) + _off_diagonal_mean(kernel(y, y))
    return float(within - 2 * kernel(x, y).mean())


def random_directions(n_features: int, n_projections: int, seed: int) -> np.ndarray:
    """``n_projections`` unit vectors drawn uniformly on the sphere, reproducible from ``seed``."""
    if n_projections < 1:
        raise ValueError("n_projections must be at least 1")
    directions = np.random.default_rng(seed).normal(size=(n_projections, n_features))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, directions: np.ndarray) -> float:
    """Mean 1-Wasserstein distance between the samples projected onto each direction.

    Uses ``scipy.stats.wasserstein_distance`` for each one-dimensional
    projection; ``directions`` is ``(n_projections, n_features)``.
    """
    x, y = _pair(x, y, 1)
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != x.shape[1]:
        raise ValueError("directions must be (n_projections, n_features)")
    px, py = x @ directions.T, y @ directions.T
    return float(
        np.mean([stats.wasserstein_distance(px[:, j], py[:, j]) for j in range(len(directions))])
    )
