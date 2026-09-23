"""JCB/Zweifach-style scalar precision baseline.

Reproduces the averaging-noise criterion of Zweifach, "Determining how many
cells to average for statistical testing of microscopy experiments",
J Cell Biol 2024;223(8):e202401074, doi:10.1083/jcb.202401074.

Scope stated by the paper: one scalar property of whole cells at a single time
point, in spatially uniform samples, with each sample's cell average treated
as n = 1. With the between-cell SD expressed as a fraction of the mean
(the paper's % CV), the number of cells to average is

    n = ceil((SD / target) ** 2),

equivalently, the averaging noise of a mean of ``n`` cells is ``SD / sqrt(n)``.
The paper recommends a target of about 5% CV when samples are independent
(Student's t test) and 2.5% CV for matched/paired designs whose samples share
between-repeat variability. It reports both a point estimate and one based on
the upper bound of the 95% confidence interval for SD, recommended when only a
few cells are available; the upper bound is the chi-square bound with
``n - 1`` degrees of freedom, which reproduces the paper's worked example.

This is the paper's **single-sample / pilot variant**: the requirement is
estimated from the cells of one analysis group. The paper's full multi-sample
procedure is not implemented here, because it needs data from other analysis
groups, which a strategy never sees (see below). It belongs in a separate,
leakage-safe planning step.

``resampling_unit="object"`` is the published criterion. ``resampling_unit="fov"``
is a MorphoCoverage variant, not part of the paper: the same averaging noise
estimated with FOVs as the sampling unit (ratio-estimator variance for cluster
sampling), so spatially clustered phenotypes inflate it. The gap between the
two is the comparison the regime map needs; it does not make either wrong.

Not reproduced here, both parts of the multi-sample procedure:

- normalizing values to the average of all control samples;
- when several samples are used, subtracting each sample's mean before
  estimating the pooled within-sample SD.

Within one group the sample's own mean is used, as in the paper's
single-sample case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..objectives import ObjectiveConfig
from ..results import Decision, StrategyResult
from ..schema import ValidatedInput, fov_positions
from .base import SamplingStrategy

_UNITS = ("object", "fov")
_SD_ESTIMATES = ("point", "upper_95")
# The paper's advice: a pilot SD estimate above 50% CV calls for more data.
_MAX_TRUSTED_CV = 0.5


def _regularized_lower_gamma(a: float, x: float) -> float:
    """P(a, x), by series for x < a + 1 and continued fraction otherwise."""
    if x <= 0:
        return 0.0
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        denominator = a
        for _ in range(10_000):
            denominator += 1
            term *= x / denominator
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(log_prefactor)
    # Lentz's method for the continued fraction of Q(a, x).
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1.0 - h * math.exp(log_prefactor)


def chi2_quantile(probability: float, df: int) -> float:
    """Lower-tail quantile of the chi-square distribution, by bisection on its CDF."""
    if not 0 < probability < 1:
        raise ValueError("probability must be between 0 and 1")
    if df < 1:
        raise ValueError("df must be at least 1")
    low, high = 0.0, max(1.0, df + 20 * math.sqrt(2 * df))
    for _ in range(100):
        middle = (low + high) / 2
        if _regularized_lower_gamma(df / 2, middle / 2) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@lru_cache(maxsize=4096)
def sd_upper_bound_factor(df: int, confidence: float = 0.95) -> float:
    """Factor turning a sample SD into the upper bound of its confidence interval."""
    return math.sqrt(df / chi2_quantile((1 - confidence) / 2, df))


def cell_averaging_cv(values: np.ndarray) -> dict[str, float]:
    """Between-cell CV and averaging CV of the mean, treating cells as independent."""
    n = len(values)
    mean = float(np.mean(values))
    cv_cells = float(np.std(values, ddof=1)) / mean
    cv_cells_upper = cv_cells * sd_upper_bound_factor(n - 1)
    return {
        "mean": mean,
        "cv_cells": cv_cells,
        "cv_cells_upper_95": cv_cells_upper,
        "cv_mean": cv_cells / math.sqrt(n),
        "cv_mean_upper_95": cv_cells_upper / math.sqrt(n),
    }


def fov_averaging_cv(sums: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    """Averaging CV of the object-weighted mean with FOVs as the sampling unit.

    Ratio-estimator variance for cluster sampling: with per-FOV sums ``T_j``,
    object counts ``N_j``, ``k`` acquired FOVs and mean ``m = sum(T) / sum(N)``,

        Var(m) = k / (k - 1) * sum((T_j - m * N_j) ** 2) / sum(N) ** 2.

    Estimand: the per-object mean over the sampled area, with FOVs as the
    sampling units (cluster sampling). Acquired FOVs with zero objects are
    sampled units: they contribute ``T_j = N_j = 0`` and count towards ``k``,
    as in the standard ratio-estimator variance. Their only effect on the
    estimated variance is through the ``k / (k - 1)`` degrees-of-freedom
    factor, which shrinks slightly as empty FOVs are added. They do matter for
    ``required_fovs``, since future FOVs may be empty too. Excluding them would
    give a slightly more conservative bound. The upper bound applies the same
    chi-square bound as the object criterion with ``k - 1`` degrees of freedom;
    it is an approximation that has not been validated empirically for this
    estimator.
    """
    k = len(sums)
    total = float(np.sum(counts))
    mean = float(np.sum(sums)) / total
    residual = float(np.sum((sums - mean * counts) ** 2))
    cv_mean = math.sqrt(k / (k - 1) * residual / total**2) / mean
    return {
        "mean": mean,
        "cv_mean": cv_mean,
        "cv_mean_upper_95": cv_mean * sd_upper_bound_factor(k - 1),
    }


@dataclass(frozen=True)
class ScalarPrecisionStrategy(SamplingStrategy):
    """Stop when the averaging CV of one scalar feature's mean reaches ``target_cv``.

    Parameters
    ----------
    feature:
        The single scalar measurement whose mean is estimated. It must be one of
        the explicitly selected feature columns, with non-negative values and a
        positive mean, as for the whole-cell properties (sizes, counts,
        intensities) the criterion was designed for. Negative values make the
        criterion undefined (``decision=None``).
    target_cv:
        Target averaging CV as a fraction: the paper suggests 0.05 for
        independent samples and 0.025 for matched/paired designs. No default.
    resampling_unit:
        ``"object"`` (the published criterion) or ``"fov"`` (MorphoCoverage variant).
    sd_estimate:
        ``"point"`` or ``"upper_95"`` (upper bound of the 95% confidence interval,
        which the paper recommends when few cells are available). No default.
    pilot_fovs, order_col:
        ``None`` (default) re-checks the criterion on all visible data after every
        FOV, an adaptation for sequential stopping. With ``pilot_fovs=k`` the
        requirement is planned once from the first ``k`` FOVs by ``order_col`` and
        then held fixed, as in the paper's pilot procedure: a number of cells
        (object unit) or of FOVs (fov unit).
    """

    feature: str
    target_cv: float
    resampling_unit: str
    sd_estimate: str
    pilot_fovs: int | None = None
    order_col: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("feature must be one column name")
        if not 0 < self.target_cv < 1:
            raise ValueError("target_cv must be between 0 and 1")
        if self.resampling_unit not in _UNITS:
            raise ValueError(f"resampling_unit must be one of {_UNITS}")
        if self.sd_estimate not in _SD_ESTIMATES:
            raise ValueError(f"sd_estimate must be one of {_SD_ESTIMATES}")
        if (self.pilot_fovs is None) != (self.order_col is None):
            raise ValueError("pilot_fovs and order_col must be given together")
        if self.pilot_fovs is not None and self.pilot_fovs < 2:
            raise ValueError("pilot_fovs must be at least 2")

    @property
    def name(self) -> str:
        """Variant-specific name, e.g. ``scalar_precision_object_upper95_pilot``."""
        parts = ["scalar_precision", self.resampling_unit]
        if self.sd_estimate == "upper_95":
            parts.append("upper95")
        if self.pilot_fovs is not None:
            parts.append("pilot")
        return "_".join(parts)

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the criterion, for result provenance."""
        return {
            "feature": self.feature,
            "target_cv": self.target_cv,
            "resampling_unit": self.resampling_unit,
            "sd_estimate": self.sd_estimate,
            "pilot_fovs": self.pilot_fovs,
            "order_col": self.order_col,
        }

    def _suffix(self) -> str:
        return "_upper_95" if self.sd_estimate == "upper_95" else ""

    def _precision(self, data: ValidatedInput, fovs) -> dict[str, Any] | str:
        """Averaging-precision estimate on the given FOVs, or why it is undefined."""
        # Per-FOV sums and counts in the order of ``fovs``; empty FOVs give zeros.
        positions = fov_positions(data, fovs)
        inside = positions >= 0
        values = data.feature_values(self.feature)[inside]
        sums = np.bincount(positions[inside], weights=values, minlength=len(fovs))
        counts = np.bincount(positions[inside], minlength=len(fovs)).astype(float)
        n_objects = int(counts.sum())
        if self.resampling_unit == "object":
            if n_objects < 2:
                return "fewer than 2 objects"
        elif len(fovs) < 2 or n_objects == 0:
            return "fewer than 2 acquired FOVs or no objects"
        if (values < 0).any():
            return (
                "negative values; the CV criterion applies to non-negative "
                "whole-cell measurements"
            )
        if sums.sum() <= 0:
            return "non-positive mean; CV is only defined for positive-valued measurements"

        cells = cell_averaging_cv(values) if n_objects >= 2 else None
        if self.resampling_unit == "object":
            estimate: dict[str, Any] = dict(cells)
            cv_cells = estimate["cv_cells" + self._suffix()]
            estimate["required_objects"] = (cv_cells / self.target_cv) ** 2
        else:
            estimate = fov_averaging_cv(sums, counts)
            # Var(mean) scales as 1/k under FOV resampling.
            cv_mean = estimate["cv_mean" + self._suffix()]
            estimate["required_fovs"] = len(fovs) * (cv_mean / self.target_cv) ** 2
            if cells is not None and cells["cv_mean"] > 0:
                # Variance inflation from treating FOVs, not cells, as the sampling unit.
                estimate["design_effect"] = (estimate["cv_mean"] / cells["cv_mean"]) ** 2
        if cells is not None and cells["cv_cells"] > _MAX_TRUSTED_CV:
            estimate["warning"] = (
                "between-cell SD exceeds 50% CV; the paper advises acquiring more data "
                "and repeating the estimate"
            )
        estimate["n_objects"] = n_objects
        estimate["n_fovs"] = len(fovs)
        return estimate

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        """Averaging CV of the visible data as ``score``; stop once it meets the target.

        With a pilot, the decision instead compares visible effort with the
        requirement planned from the pilot FOVs. ``decision`` is ``None`` when
        the criterion is undefined (too little data or a non-positive mean),
        with the reason in ``diagnostics["undefined_reason"]``.
        """
        data.feature_values(self.feature)  # fails early on a non-feature column
        diagnostics: dict[str, Any] = {
            "scalar_parameter": f"mean of {self.feature!r} over objects",
            "assumes_spatial_uniformity": self.resampling_unit == "object",
        }

        current = self._precision(data, data.fovs)
        if isinstance(current, str):
            diagnostics["undefined_reason"] = current
            return self._result(objective, None, None, None, diagnostics)
        diagnostics.update(current)
        score = current["cv_mean" + self._suffix()]

        if self.pilot_fovs is None:
            stop = score <= self.target_cv
            return self._result(objective, score, stop, self._remaining(current, data), diagnostics)

        if self.order_col not in data.fovs.columns:
            raise ValueError(f"order_col {self.order_col!r} is not a column of the FOV table")
        if len(data.fovs) < self.pilot_fovs:
            diagnostics["undefined_reason"] = "pilot not yet complete"
            return self._result(objective, score, False, None, diagnostics)
        pilot_fovs = data.fovs.sort_values(self.order_col, kind="stable").iloc[: self.pilot_fovs]
        pilot = self._precision(data, pilot_fovs)
        if isinstance(pilot, str):
            diagnostics["undefined_reason"] = f"pilot: {pilot}"
            return self._result(objective, score, None, None, diagnostics)
        diagnostics["pilot"] = pilot
        remaining = self._remaining(pilot, data)
        return self._result(objective, score, remaining == 0, remaining, diagnostics)

    def _remaining(self, estimate: dict[str, Any], data: ValidatedInput) -> float:
        """Further objects (object unit) or FOVs (fov unit) the estimate says are needed."""
        if self.resampling_unit == "object":
            return float(max(0, math.ceil(estimate["required_objects"]) - len(data.objects)))
        return float(max(0, math.ceil(estimate["required_fovs"]) - len(data.fovs)))

    def _result(self, objective, score, stop, remaining, diagnostics) -> StrategyResult:
        decision = None if stop is None else (Decision.STOP if stop else Decision.CONTINUE)
        return StrategyResult(
            strategy=self.name,
            objective=objective,
            score=score,
            decision=decision,
            predicted_remaining=remaining,
            diagnostics=diagnostics,
            parameters=self.parameters,
        )
