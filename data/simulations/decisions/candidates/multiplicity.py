"""Multiple-testing procedures of P5. Each returns which hypotheses are rejected at false
discovery rate ``q``:

- ``bh``: Benjamini-Hochberg over one family (the package's ``statistics.benjamini_hochberg``;
  today: one family over every explainable feature of every table, per sub-experiment);
- ``by``: Benjamini-Yekutieli, valid under any dependence (a reference);
- ``per_family``: BH within each family separately (each table its own family);
- ``weighted_bh``: weighted BH (Genovese, Roeder and Wasserman 2006) with weights fixed in
  advance and averaging 1: ``p / w`` into BH;
- ``hierarchical``: Benjamini and Bogomolov (2014): each family's Simes p, BH over the
  families at ``q``, then BH within each of the ``R`` selected families at ``q R / F``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from morphocoverage.statistics import benjamini_hochberg


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejections (NaN never rejected)."""
    adjusted = benjamini_hochberg(np.asarray(p, dtype=float))
    return np.nan_to_num(adjusted, nan=np.inf) <= q


def by(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Yekutieli rejections."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = stats.false_discovery_control(p[finite], method="by") <= q
    return out


def per_family(p: np.ndarray, families: np.ndarray, q: float = 0.05) -> np.ndarray:
    """BH within each family."""
    out = np.zeros(len(p), dtype=bool)
    for family in np.unique(families):
        rows = families == family
        out[rows] = bh(np.asarray(p)[rows], q)
    return out


def weighted_bh(p: np.ndarray, weights: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Weighted BH: weights rescaled to mean 1, ``p / w`` into BH."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    return bh(np.minimum(1.0, np.asarray(p, dtype=float) / weights), q)


def simes(p: np.ndarray) -> float:
    """Simes' combined p of one family."""
    p = np.sort(np.asarray(p, dtype=float)[np.isfinite(p)])
    if not len(p):
        return np.nan
    return float(np.min(p * len(p) / np.arange(1, len(p) + 1)))


def hierarchical(p: np.ndarray, families: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Bogomolov: families selected by BH on their Simes p, then BH within each
    selected family at ``q R / F``."""
    p = np.asarray(p, dtype=float)
    names = np.unique(families)
    combined = np.array([simes(p[families == f]) for f in names])
    selected = bh(combined, q)
    out = np.zeros(len(p), dtype=bool)
    level = q * selected.sum() / len(names)
    for name, chosen in zip(names, selected, strict=True):
        if chosen:
            rows = families == name
            out[rows] = bh(p[rows], level)
    return out


__all__ = ["bh", "by", "hierarchical", "per_family", "simes", "weighted_bh"]
