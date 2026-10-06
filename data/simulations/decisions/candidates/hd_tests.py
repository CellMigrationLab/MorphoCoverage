"""Whole-embedding tests of P3: does a condition's centre move away from its control's, the
same way in every biological repeat (mean shift), or does its distribution change
(distributional)? All read field means (a field's cells averaged first, as the package's
whole-table test), and every test but ``min_p`` depends on the geometry only: Euclidean
inner products, distances or rotation-equivariant covariance estimates.

Mean shift, on the repeat shifts ``Δ_r`` (condition centre minus control centre in repeat
``r``):

- ``u_statistic``: the paired high-dimensional inner-product U-statistic of the Chen and
  Qin (2010) family, ``U = mean_{r != s} Δ_r · Δ_s``, standardised by the ratio-consistent
  estimate of its null variance ``2 tr(Σ²) / (n (n - 1))`` (``tr(Σ²)`` by
  ``mean_{r != s} (Δ_r · Δ_s)²``), one-sided normal p; and its exact sign-flip version
  (``sign_flip_u``: the ``2^n`` sign assignments, smallest p ``2 / 2^n``);
- ``pc_hotelling``: one-sample Hotelling T² of the ``Δ_r`` on the first ``n - 2`` principal
  components of the within-group field residuals (F test);
- ``shrinkage_hotelling``: ``n mean(Δ)' S^-1 mean(Δ)``, ``S`` the Ledoit-Wolf covariance of
  the within-group field residuals (shrunk to a multiple of the identity, so equivariant),
  p by the exact sign flips of the ``Δ_r``;
- ``min_p``: per coordinate, the paired t-test of the ``Δ_r``, the smallest p times the
  number of coordinates (Bonferroni): the non-invariant negative comparator.

Distributional, with the labels of the fields permuted within each repeat
(``permutation_test``): PERMANOVA's pseudo-F (Anderson 2001) with restricted permutations,
the energy distance (Székely and Rizzo 2004) and the MMD with a Gaussian kernel at the
median distance (Gretton et al. 2012), summed over repeats. The field is the unit permuted,
as in the package's exploratory permutation: anti-conservative under a well batch.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy import stats

PERMUTATIONS = 199


def field_means(
    values: np.ndarray, condition: np.ndarray, repeat: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each field's mean vector, and its condition and repeat."""
    codes = np.unique(field, return_inverse=True)[1].reshape(-1)
    first = np.unique(codes, return_index=True)[1]
    sums = np.zeros((codes.max() + 1, values.shape[1]))
    np.add.at(sums, codes, values)
    means = sums / np.bincount(codes)[:, None]
    return means, condition[first], repeat[first]


def deltas(
    means: np.ndarray, condition: np.ndarray, repeat: np.ndarray, treated: str, control: str
) -> tuple[np.ndarray, list[str]]:
    """Per repeat holding both sides, the condition's centre (mean of its field means) minus
    the control's."""
    rows, kept = [], []
    for r in dict.fromkeys(repeat):
        a = means[(repeat == r) & (condition == treated)]
        b = means[(repeat == r) & (condition == control)]
        if len(a) and len(b):
            rows.append(a.mean(axis=0) - b.mean(axis=0))
            kept.append(r)
    return np.array(rows), kept


def residuals(means: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """The field means minus the centre of their condition in their repeat."""
    out = means.copy()
    keys = np.char.add(condition.astype(str), "\x1f" + repeat.astype(str))
    for key in np.unique(keys):
        rows = keys == key
        out[rows] -= means[rows].mean(axis=0)
    return out


def u_statistic(delta: np.ndarray) -> dict[str, float]:
    """The paired U-statistic, its z and one-sided normal p (see the module)."""
    n = len(delta)
    if n < 2:
        return {"u": np.nan, "z": np.nan, "p": np.nan}
    gram = delta @ delta.T
    off = ~np.eye(n, dtype=bool)
    u = float(gram[off].mean())
    trace = float(np.square(gram[off]).mean())
    if trace <= 0:
        return {"u": u, "z": np.nan, "p": np.nan}
    z = u / np.sqrt(2 * trace / (n * (n - 1)))
    return {"u": u, "z": float(z), "p": float(stats.norm.sf(z))}


def _signs(n: int) -> np.ndarray:
    return np.array(list(product((1.0, -1.0), repeat=n)))


def sign_flip_u(delta: np.ndarray) -> float:
    """The exact one-sided sign-flip p of the U-statistic (``2^n`` assignments)."""
    n = len(delta)
    if n < 2:
        return np.nan
    gram = delta @ delta.T
    np.fill_diagonal(gram, 0.0)
    signs = _signs(n)
    stat = np.einsum("bi,ij,bj->b", signs, gram, signs)
    observed = stat[0]
    return float(np.mean(stat >= observed - 1e-12 * max(abs(observed), 1.0)))


def pc_hotelling(delta: np.ndarray, resid: np.ndarray) -> float:
    """Hotelling's one-sample T² on the first ``n - 2`` principal components of ``resid``."""
    n = len(delta)
    k = n - 2
    if k < 1 or len(resid) < 2:
        return np.nan
    _, _, vt = np.linalg.svd(resid - resid.mean(axis=0), full_matrices=False)
    scores = delta @ vt[:k].T
    mean = scores.mean(axis=0)
    cov = np.atleast_2d(np.cov(scores, rowvar=False))
    try:
        t2 = n * float(mean @ np.linalg.solve(cov, mean))
    except np.linalg.LinAlgError:
        return np.nan
    f = (n - k) / (k * (n - 1)) * t2
    return float(stats.f.sf(f, k, n - k))


def shrinkage_hotelling(delta: np.ndarray, resid: np.ndarray) -> float:
    """``n mean(Δ)' S^-1 mean(Δ)`` with the Ledoit-Wolf ``S``, exact sign-flip p."""
    from sklearn.covariance import ledoit_wolf

    n = len(delta)
    if n < 2 or len(resid) < 2:
        return np.nan
    shrunk = ledoit_wolf(resid, assume_centered=True)[0]
    inverse = np.linalg.pinv(shrunk)
    signs = _signs(n)
    means = signs @ delta / n
    stat = np.einsum("bi,ij,bj->b", means, inverse, means)
    return float(np.mean(stat >= stat[0] - 1e-12 * max(abs(stat[0]), 1.0)))


def min_p(delta: np.ndarray) -> float:
    """The smallest per-coordinate paired-t p, times the coordinates (Bonferroni)."""
    if len(delta) < 2:
        return np.nan
    p = stats.ttest_1samp(delta, 0.0, axis=0).pvalue
    p = p[np.isfinite(p)]
    return float(min(1.0, p.min() * delta.shape[1])) if len(p) else np.nan


def _labels(
    side: np.ndarray, strata: np.ndarray, permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """The observed labels and ``permutations`` shuffles within each stratum (rows)."""
    labels = np.tile(side.astype(float), (permutations + 1, 1))
    for value in np.unique(strata):
        mine = np.flatnonzero(strata == value)
        labels[1:, mine] = rng.permuted(labels[1:, mine], axis=1)
    return labels


def _between(labels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Per label row, ``mean M(x, y)`` and the two within means of a symmetric ``matrix``."""
    ones = labels.sum(axis=1)
    zeros = labels.shape[1] - ones
    lm = labels @ matrix
    within1 = (lm * labels).sum(axis=1) / np.maximum(ones, 1) ** 2
    cross = (lm * (1 - labels)).sum(axis=1) / np.maximum(ones * zeros, 1)
    within0 = (((1 - labels) @ matrix) * (1 - labels)).sum(axis=1) / np.maximum(zeros, 1) ** 2
    return np.stack([cross, within1, within0])


def permutation_test(
    means: np.ndarray,
    side: np.ndarray,
    strata: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    permutations: int = PERMUTATIONS,
) -> float:
    """PERMANOVA (``permanova``), energy (``energy``) or MMD (``mmd``) summed over strata,
    p from the labels permuted within each stratum, ``(extreme + 1) / (B + 1)``."""
    total = np.zeros(permutations + 1)
    for value in np.unique(strata):
        rows = np.flatnonzero(strata == value)
        here = side[rows]
        if here.min() == here.max():
            continue
        x = means[rows]
        squared = np.square(x[:, None, :] - x[None, :, :]).sum(axis=2)
        labels = _labels(here, np.zeros(len(rows)), permutations, rng)
        if kind == "permanova":
            # Between-group sum of squares within the stratum (the pseudo-F's numerator; the
            # stratum's total is fixed under its permutations).
            parts = _between(labels, squared)
            ones = labels.sum(axis=1)
            total += ones * (len(rows) - ones) / len(rows) * (parts[0] - (parts[1] + parts[2]) / 2)
        elif kind == "energy":
            parts = _between(labels, np.sqrt(squared))
            total += 2 * parts[0] - parts[1] - parts[2]
        elif kind == "mmd":
            off = squared[~np.eye(len(rows), dtype=bool)]
            bandwidth = np.median(off[off > 0]) if np.any(off > 0) else 1.0
            parts = _between(labels, np.exp(-squared / (2 * bandwidth)))
            total += parts[1] + parts[2] - 2 * parts[0]
        else:
            raise ValueError("kind must be permanova, energy or mmd")
    if not np.any(total):
        return np.nan
    extreme = int((total[1:] >= total[0] - 1e-12 * max(abs(total[0]), 1.0)).sum())
    return (extreme + 1) / (permutations + 1)


__all__ = [
    "deltas",
    "field_means",
    "min_p",
    "pc_hotelling",
    "permutation_test",
    "residuals",
    "shrinkage_hotelling",
    "sign_flip_u",
    "u_statistic",
]
