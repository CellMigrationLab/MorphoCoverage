"""Stopping rules for a mean, P4: at each look, an interval for the sample's mean from the
fields so far; stop at the first look whose relative half-width is at most the tolerance.

Intervals (95% unless the rule spends its error over the looks):

- ``nested_bootstrap``: the package's (``statistics.bootstrap_nested``: wells resampled,
  their fields kept; ``bootstrap_fields`` with one well or none recorded), widened for the
  units resampled;
- ``cluster_se``: the ratio-estimator SE over fields (``statistics.object_weighted_mean_se``)
  with a t quantile on the fields with cells less one;
- ``two_stage_bootstrap``: wells, then fields within each drawn well, with replacement
  (percentile, no widening);
- ``block_bootstrap``: moving blocks of consecutive fields in acquisition order, length
  ``ceil(k^(1/3))`` (Künsch 1989), for spatially or temporally correlated fields;
- ``bonferroni_looks``: ``cluster_se`` at level ``1 - alpha / K`` for ``K`` planned looks;
- ``spending``: ``cluster_se`` at the Lan-DeMets O'Brien-Fleming increment of each look
  (``alpha(t) = 2 - 2 Phi(z_{alpha/2} / sqrt(t))``, ``t`` the share of the planned fields;
  each increment used as its own two-sided level, a conservative simplification of the
  recursive boundaries);
- ``confidence_sequence``: the normal-mixture boundary (Howard et al. 2021) for the running
  mean, its variance from ``cluster_se`` (a plug-in: approximate), tuned to the planned
  stop of ``PLANNED`` fields.

Each returns ``(estimate, low, high)`` or NaN bounds when it cannot (fewer than two fields
with cells).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from morphocoverage.statistics import (
    bootstrap_fields,
    bootstrap_nested,
    object_weighted_mean_se,
)

PLANNED = 20
_NAN = (math.nan, math.nan, math.nan)


def _pooled(fields: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.array([f.sum() for f in fields], dtype=float)
    counts = np.array([len(f) for f in fields], dtype=float)
    return sums, counts


def cluster_se(fields: Sequence[np.ndarray], level: float = 0.95) -> tuple[float, float, float]:
    """The ratio-estimator interval at ``level`` (see the module)."""
    sums, counts = _pooled(fields)
    if (counts > 0).sum() < 2:
        return _NAN
    mean, se, k = object_weighted_mean_se(sums, counts)
    half = stats.t.ppf(1 - (1 - level) / 2, k - 1) * se
    return mean, mean - half, mean + half


def nested_bootstrap(
    fields: Sequence[np.ndarray], wells: Sequence[str] | None, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    """The package's interval: wells resampled when recorded, else fields."""
    if sum(len(f) > 0 for f in fields) < 2:
        return _NAN
    if wells is None or len(set(wells)) < 2:
        return bootstrap_fields(fields, np.mean, n_boot=n_boot, rng=rng)
    units: dict[str, list[np.ndarray]] = {}
    for well, values in zip(wells, fields, strict=True):
        units.setdefault(well, []).append(values)
    return bootstrap_nested(list(units.values()), np.mean, n_boot=n_boot, rng=rng)


def two_stage_bootstrap(
    fields: Sequence[np.ndarray], wells: Sequence[str] | None, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    """Wells, then fields within each drawn well, with replacement (percentile)."""
    sums, counts = _pooled(fields)
    if (counts > 0).sum() < 2:
        return _NAN
    labels = (
        np.zeros(len(fields), dtype=int)
        if wells is None
        else np.unique(wells, return_inverse=True)[1]
    )
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        picked = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in chosen])
        total = counts[picked].sum()
        draws[b] = sums[picked].sum() / total if total > 0 else np.nan
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def block_bootstrap(
    fields: Sequence[np.ndarray], rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    """Moving blocks of consecutive fields in acquisition order (percentile)."""
    sums, counts = _pooled(fields)
    k = len(fields)
    if (counts > 0).sum() < 2:
        return _NAN
    length = max(1, math.ceil(k ** (1 / 3)))
    starts = np.arange(k - length + 1)
    blocks = math.ceil(k / length)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        first = rng.choice(starts, size=blocks)
        picked = (first[:, None] + np.arange(length)).reshape(-1)[:k]
        total = counts[picked].sum()
        draws[b] = sums[picked].sum() / total if total > 0 else np.nan
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def spending_levels(looks: Sequence[int], planned: int, alpha: float = 0.05) -> list[float]:
    """The confidence level of each look under the Lan-DeMets O'Brien-Fleming spending."""
    z = stats.norm.ppf(1 - alpha / 2)
    spent, levels = 0.0, []
    for look in looks:
        t = min(1.0, look / planned)
        cumulative = 2 - 2 * stats.norm.cdf(z / math.sqrt(t))
        increment = max(cumulative - spent, 1e-12)
        spent = cumulative
        levels.append(1 - increment)
    return levels


def confidence_sequence(
    fields: Sequence[np.ndarray], alpha: float = 0.05, planned: int = PLANNED
) -> tuple[float, float, float]:
    """The normal-mixture confidence sequence for the running mean (see the module)."""
    sums, counts = _pooled(fields)
    if (counts > 0).sum() < 2:
        return _NAN
    mean, se, k = object_weighted_mean_se(sums, counts)
    variance = se**2 * k  # one field's share
    log_term = -2 * math.log(alpha) + math.log(-2 * math.log(alpha) + 1)
    rho = variance * planned / log_term
    v = k * variance
    half = math.sqrt((v + rho) * math.log((v + rho) / (rho * alpha**2))) / k
    return mean, mean - half, mean + half


__all__ = [
    "PLANNED",
    "block_bootstrap",
    "cluster_se",
    "confidence_sequence",
    "nested_bootstrap",
    "spending_levels",
    "two_stage_bootstrap",
]
