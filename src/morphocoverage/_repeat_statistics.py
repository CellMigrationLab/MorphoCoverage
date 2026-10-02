"""The statistics of a comparison across biological repeats, and the shape of a sample
(private; re-exported by ``statistics``, where every shared statistic is looked up).

- ``paired_t``: the paired t-test on one value per biological repeat;
- ``repeat_values``: the values that test compares, the one place the rule is written: the
  log ratios of the repeat means to the control's when every mean is positive (a fold
  change), else their differences;
- ``bimodality_coefficient``: Sarle's coefficient, above ``BIMODAL`` (5/9) when the values
  may form two groups; ``excess_kurtosis`` tells a heavy tail apart from two groups.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t(differences: np.ndarray) -> float:
    """The two-sided paired t-test of ``differences`` (one per biological repeat); NaN when a
    value is missing or they do not vary."""
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or np.isnan(differences).any() or np.ptp(differences) == 0:
        return float("nan")
    return float(stats.ttest_1samp(differences, 0.0).pvalue)


def repeat_values(means: np.ndarray, controls: np.ndarray) -> tuple[str, np.ndarray, float]:
    """The values the paired test across repeats compares, from each repeat's mean of the
    condition and of its control: ``("normalised", log ratios, fold change)`` when every
    mean is positive (the fold change the geometric mean of the ratios), else ``("raw",
    differences, mean difference)``. The one place this rule is written (owner,
    2026-10-01: the paired t-test on the repeat means, normalised to the control)."""
    means, controls = np.asarray(means, dtype=float), np.asarray(controls, dtype=float)
    if len(means) and (means > 0).all() and (controls > 0).all():
        ratios = np.log(means / controls)
        return "normalised", ratios, float(np.exp(ratios.mean()))
    differences = means - controls
    return "raw", differences, float(differences.mean()) if len(differences) else float("nan")


# Sarle's bimodality coefficient above this suggests two groups of values (5/9, its value
# for a uniform distribution).
BIMODAL = 5 / 9


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient ``(g1**2 + 1) / (G2 + 3 (n-1)**2 / ((n-2)(n-3)))`` with
    the bias-corrected skewness and excess kurtosis (``scipy.stats``, ``bias=False``); NaN
    below 4 values or without spread. Above ``BIMODAL`` the values may form two groups (a
    strongly skewed distribution can also reach it)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 4 or np.ptp(values) == 0:
        return float("nan")
    skew = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, bias=False))
    return (skew**2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def excess_kurtosis(values: np.ndarray) -> float:
    """The bias-corrected excess kurtosis (``scipy.stats``, ``bias=False``); NaN below 4
    values or without spread."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4 or np.ptp(values) == 0:
        return float("nan")
    return float(stats.kurtosis(values, bias=False))


__all__ = ["BIMODAL", "bimodality_coefficient", "excess_kurtosis", "paired_t", "repeat_values"]
