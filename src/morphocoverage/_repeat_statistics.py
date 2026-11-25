"""The statistics of a comparison across biological repeats, and the shape of a sample
(private; re-exported by ``statistics``, where every shared statistic is looked up).

- ``paired_t``: the paired t-test on one value per biological repeat;
- ``log_scale`` and ``repeat_values``: the values that test compares, the one place the
  rule is written: the log ratios of the repeat means to the control's (a fold change) when
  every observation of the feature is positive, else their differences, chosen once per
  feature so every condition is tested on the same scale (owner, 2026-10-06, audit 8);
- ``repeat_effect`` and ``contrasts_text``: the per-repeat contrasts on that scale and the
  t-based 95% interval of their mean (plan v2, owner, 2026-10-06);
- ``bimodality_coefficient``: Sarle's coefficient, above ``BIMODAL`` (5/9) when the values
  may form two groups; ``excess_kurtosis`` tells a heavy tail apart from two groups.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._checks import is_share


def paired_t(differences: np.ndarray) -> float:
    """The two-sided paired t-test of ``differences`` (one per biological repeat); NaN when a
    value is missing or they do not vary."""
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or np.isnan(differences).any() or np.ptp(differences) == 0:
        return float("nan")
    return float(stats.ttest_1samp(differences, 0.0).pvalue)


# The scales a feature can be declared on (``Setup.scales``; plan v2, owner, 2026-10-06).
SCALES = ("ratio", "difference")


def log_scale(values: np.ndarray, declared: str | None = None, feature: str = "") -> bool:
    """Whether the paired test across repeats reads a feature as log ratios. ``declared``,
    the feature's scale in the Setup (``scales``; plan v2, owner, 2026-10-06): ``"ratio"``
    (refused, ``ValueError``, when a value is not positive) or ``"difference"``. Undeclared,
    the automatic rule: every one of its observations ``values`` positive (missing values
    aside), else differences. Chosen once per feature from all its observations, the same
    for every condition (owner, 2026-10-06, audit 8: chosen per condition from its means, an
    offset of the feature changed the test's power)."""
    if declared not in (None, *SCALES):
        raise ValueError(f"a scale is one of {SCALES}, not {declared!r}")
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    positive = bool(len(values)) and bool((values > 0).all())
    if declared == "ratio" and not positive:
        lowest = f"{values.min():g}" if len(values) else "no value"
        raise ValueError(
            f"{feature or 'a feature'} is declared on the ratio scale (scales), but holds a "
            f"value that is not positive ({lowest}): a ratio to the control needs positive "
            "values; declare it 'difference'"
        )
    return positive if declared is None else declared == "ratio"


def repeat_values(
    means: np.ndarray, controls: np.ndarray, log: bool | None = None
) -> tuple[str, np.ndarray, float]:
    """The values the paired test across repeats compares, from each repeat's mean of the
    condition and of its control: ``("normalised", log ratios, fold change)`` on the log
    scale (the fold change the geometric mean of the ratios), else ``("raw", differences,
    mean difference)``. ``log`` is the feature's scale (``log_scale`` of all its
    observations); ``None``, where they are not at hand, reads it from these means: the log
    scale when every mean is positive. The one place this rule is written (owner,
    2026-10-01: the paired t-test on the repeat means, normalised to the control)."""
    means, controls = np.asarray(means, dtype=float), np.asarray(controls, dtype=float)
    if log is None:
        log = bool(len(means)) and bool((means > 0).all() and (controls > 0).all())
    if log:
        ratios = np.log(means / controls)
        return "normalised", ratios, float(np.exp(ratios.mean()))
    differences = means - controls
    return "raw", differences, float(differences.mean()) if len(differences) else float("nan")


def repeat_effect(values: str, compared: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """The per-repeat contrasts in the feature's own terms (fold changes for ``normalised``
    log ratios, else differences) and the t-based ``level`` interval of their mean effect
    (``n - 1`` degrees of freedom, back-transformed for fold changes): ``ci_low`` and
    ``ci_high``, NaN below 2 repeats or with a missing value (plan v2, owner, 2026-10-06: with
    3 repeats each verdict shows them). ``level`` must be in (0, 1), and an infinite
    contrast is refused: it has no mean to give an interval of (#528)."""
    if not is_share(level):
        raise ValueError(f"level must be a number in (0, 1), not {level!r}")
    compared = np.asarray(compared, dtype=float)
    if np.isinf(compared).any():
        raise ValueError("the per-repeat contrasts must be finite or missing, not infinite")
    shown = np.exp(compared) if values == "normalised" else compared
    low = high = float("nan")
    if len(compared) >= 2 and not np.isnan(compared).any():
        half = stats.t.ppf(0.5 + level / 2, len(compared) - 1) * stats.sem(compared)
        low, high = compared.mean() - half, compared.mean() + half
        if values == "normalised":
            low, high = float(np.exp(low)), float(np.exp(high))
    return {"contrasts": shown, "ci_low": float(low), "ci_high": float(high)}


def contrasts_text(repeats: list, contrasts: np.ndarray) -> str:
    """``R1: 0.82; R2: 0.75; R3: 0.91``, each repeat's contrast for a CSV (4 significant
    digits)."""
    return "; ".join(f"{r}: {float(v):.4g}" for r, v in zip(repeats, contrasts, strict=True))


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
    return sarle(skew, kurtosis, n)


def sarle(skew: float, kurtosis: float, n: int) -> float:
    """Sarle's bimodality coefficient from the bias-corrected skewness and excess kurtosis
    of ``n`` values (the one formula, shared with the averaging CV's incremental path)."""
    return (skew**2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def excess_kurtosis(values: np.ndarray) -> float:
    """The bias-corrected excess kurtosis (``scipy.stats``, ``bias=False``); NaN below 4
    values or without spread."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4 or np.ptp(values) == 0:
        return float("nan")
    return float(stats.kurtosis(values, bias=False))


__all__ = [
    "BIMODAL",
    "SCALES",
    "bimodality_coefficient",
    "contrasts_text",
    "excess_kurtosis",
    "log_scale",
    "paired_t",
    "repeat_effect",
    "repeat_values",
    "sarle",
]
