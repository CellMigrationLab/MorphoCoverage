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

This is a **single-group JCB-style adaptation**: the requirement is estimated
from the cells of one analysis group and the group's own mean is used as the
normalizing mean. It exactly matches the paper's single-sample calculation
when that group is the normalization control, or when the supplied feature has
already been normalized to the control mean. For an arbitrary treated group,
the published procedure instead expresses its within-sample SD relative to the
control mean. The faithful multi-sample/control-normalized planning procedure
is therefore ``plan_cells_to_average`` plus ``PlannedCellCountStrategy``
in ``jcb_planning``.

``resampling_unit="object"`` is the published criterion. ``resampling_unit="fov"``
is a MorphoCoverage variant, not part of the paper: the same averaging noise
estimated with FOVs as the sampling unit (ratio-estimator variance for cluster
sampling), so spatially clustered phenotypes inflate it. The gap between the
two is the comparison the regime map needs; it does not make either wrong.

Not reproduced here, but in ``jcb_planning``, both parts of the multi-sample
procedure:

- normalizing values to the average of all control samples;
- when several samples are used, subtracting each sample's mean before
  estimating the pooled within-sample SD.

Within one group the sample's own mean is used by this adaptation. That is
not the general published normalization for treated samples; use the planning
procedure when a separate control normalization is required.

Estimation and decision are separate. ``ScalarPrecisionEstimator`` reports the
averaging CV (and, with a pilot, the averaging CV the pilot projects for the
visible effort); ``ScalarPrecisionStrategy`` applies the paper's target to it
with a ``ThresholdPolicy``. The projected CV reaches the target exactly when
the visible effort reaches the pilot's required number of cells
(``required_objects``) or FOVs (``required_fovs``), so the stopping rule is the
paper's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import chi2

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput, fov_positions
from .base import Estimator, EstimatorState, SamplingStrategy, unsupported_observation_unit

_UNITS = ("object", "fov")
_SD_ESTIMATES = ("point", "upper_95")
# The paper's advice: a pilot SD estimate above 50% CV calls for more data.
_MAX_TRUSTED_CV = 0.5


@lru_cache(maxsize=4096)
def sd_upper_bound_factor(df: int, confidence: float = 0.95) -> float:
    """Factor turning a sample SD into the upper bound of its confidence interval.

    The chi-square bound ``sqrt(df / chi2.ppf((1 - confidence) / 2, df))``.
    """
    if df < 1:
        raise ValueError("df must be at least 1")
    return math.sqrt(df / chi2.ppf((1 - confidence) / 2, df))


def cell_averaging_cv(values: np.ndarray) -> dict[str, float]:
    """Between-cell CV and averaging CV of the mean, treating cells as independent."""
    return _cell_cv_from_moments(len(values), float(np.mean(values)), float(np.std(values, ddof=1)))


def _cell_cv_from_moments(n: int, mean: float, sd: float) -> dict[str, float]:
    """Cell-level and averaging CVs, point and upper-95, from count, mean and SD."""
    cv_cells = sd / mean
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


def required_objects(cv_cells: float, target_cv: float) -> int:
    """The paper's number of cells to average: ``ceil((cv_cells / target_cv) ** 2)``."""
    return math.ceil((cv_cells / target_cv) ** 2)


def required_fovs(cv_mean: float, n_fovs: int, target_cv: float) -> int:
    """FOVs needed for ``target_cv``, since Var(mean) scales as 1/k under FOV resampling."""
    return math.ceil(n_fovs * (cv_mean / target_cv) ** 2)


_CV_UNITS = "fraction of the mean (CV)"


# Effort counts of the visible data belong to the framework (``StrategyResult``),
# not to an estimator's diagnostics; only the pilot's own counts are reported.
_EFFORT_KEYS = ("n_objects", "n_fovs")


def _cv_quantity(value: float | None, availability: str = "available", reason=None) -> Quantity:
    """An averaging-CV quantity: population evidence, lower is better, fraction of the mean."""
    return Quantity(
        value, Direction.LOWER_IS_BETTER, _CV_UNITS, Component.POPULATION, availability, reason
    )


def _check_configuration(resampling_unit, sd_estimate, pilot_fovs, order_col) -> None:
    """Validate the settings the estimator and the strategy share."""
    if resampling_unit not in _UNITS:
        raise ValueError(f"resampling_unit must be one of {_UNITS}")
    if sd_estimate not in _SD_ESTIMATES:
        raise ValueError(f"sd_estimate must be one of {_SD_ESTIMATES}")
    if (pilot_fovs is None) != (order_col is None):
        raise ValueError("pilot_fovs and order_col must be given together")
    if pilot_fovs is not None and pilot_fovs < 2:
        raise ValueError("pilot_fovs must be at least 2")


@dataclass(frozen=True)
class ScalarPrecisionEstimator(Estimator):
    """Averaging CV of one scalar feature's mean, as evidence for population precision.

    Quantities (lower is better, as a fraction of the mean, component
    ``population``):

    - ``averaging_cv``: the averaging CV of the visible data (``SD / sqrt(n)``
      relative to the mean for the object unit; the cluster ratio-estimator CV
      for the FOV unit), from the point or upper-95 SD estimate.
    - ``pilot_projected_averaging_cv`` (only with ``pilot_fovs``): the averaging
      CV that the pilot's estimate projects for the visible effort,
      ``cv_cells_pilot / sqrt(n_objects)`` (object unit) or
      ``cv_mean_pilot * sqrt(k_pilot / k)`` (FOV unit). The requirement is
      planned once from the first ``pilot_fovs`` FOVs by ``order_col`` and then
      held fixed, as in the paper's pilot procedure.

    Too few objects or FOVs give ``insufficient_data``. Negative values or a
    non-positive mean make the CV undefined (``not_applicable``), as does a
    completed pilot that cannot define the requirement. See
    ``ScalarPrecisionStrategy`` for the parameters.
    """

    feature: str
    resampling_unit: str
    sd_estimate: str
    pilot_fovs: int | None = None
    order_col: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("feature must be one column name")
        _check_configuration(
            self.resampling_unit, self.sd_estimate, self.pilot_fovs, self.order_col
        )

    @property
    def name(self) -> str:
        """Variant-specific name, e.g. ``averaging_cv_object_upper95_pilot``."""
        parts = ["averaging_cv", self.resampling_unit]
        if self.sd_estimate == "upper_95":
            parts.append("upper95")
        if self.pilot_fovs is not None:
            parts.append("pilot")
        return "_".join(parts)

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the estimate, for result provenance."""
        return {
            "feature": self.feature,
            "resampling_unit": self.resampling_unit,
            "sd_estimate": self.sd_estimate,
            "pilot_fovs": self.pilot_fovs,
            "order_col": self.order_col,
        }

    def _suffix(self) -> str:
        """Diagnostic-key suffix selecting the point or upper-95 SD estimate."""
        return "_upper_95" if self.sd_estimate == "upper_95" else ""

    def _precision(self, data: ValidatedInput, fovs) -> dict[str, Any] | tuple[str, str]:
        """Averaging-precision estimate on the given FOVs, or (availability, reason)."""
        # Per-FOV sums and counts in the order of ``fovs``; empty FOVs give zeros.
        positions = fov_positions(data, fovs)
        inside = positions >= 0
        values = data.feature_values(self.feature)[inside]
        sums = np.bincount(positions[inside], weights=values, minlength=len(fovs))
        counts = np.bincount(positions[inside], minlength=len(fovs)).astype(float)
        return self._precision_from(
            sums,
            counts,
            has_negative=bool((values < 0).any()),
            cells=lambda: cell_averaging_cv(values),
        )

    def _precision_from(
        self, sums: np.ndarray, counts: np.ndarray, *, has_negative: bool, cells
    ) -> dict[str, Any] | tuple[str, str]:
        """The estimate from per-FOV sums and counts; ``cells()`` gives the cell-level CVs."""
        if np.isnan(sums).any():
            return "not_applicable", "missing values in the selected scalar feature"
        n_objects = int(counts.sum())
        if self.resampling_unit == "object":
            if n_objects < 2:
                return "insufficient_data", "fewer than 2 objects"
        elif len(sums) < 2 or n_objects == 0:
            return "insufficient_data", "fewer than 2 acquired FOVs or no objects"
        if has_negative:
            return "not_applicable", (
                "negative values; the CV criterion applies to non-negative whole-cell measurements"
            )
        if sums.sum() <= 0:
            return "not_applicable", (
                "non-positive mean; CV is only defined for positive-valued measurements"
            )

        cell_cvs = cells() if n_objects >= 2 else None
        if self.resampling_unit == "object":
            estimate: dict[str, Any] = dict(cell_cvs)
        else:
            estimate = fov_averaging_cv(sums, counts)
            if cell_cvs is not None and cell_cvs["cv_mean"] > 0:
                # Variance inflation from treating FOVs, not cells, as the sampling unit.
                estimate["design_effect"] = (estimate["cv_mean"] / cell_cvs["cv_mean"]) ** 2
        if cell_cvs is not None and cell_cvs["cv_cells"] > _MAX_TRUSTED_CV:
            estimate["warning"] = (
                "between-cell SD exceeds 50% CV; the paper advises acquiring more data "
                "and repeating the estimate"
            )
        # The data this estimate came from: kept for the pilot's provenance, and
        # dropped from the visible estimate's diagnostics (the framework owns effort).
        estimate["n_objects"] = n_objects
        estimate["n_fovs"] = len(sums)
        return estimate

    def _check_order_col(self, fovs) -> None:
        """With a pilot, the order column must be in the FOV table."""
        if self.pilot_fovs is not None and self.order_col not in fovs.columns:
            raise ValueError(f"order_col {self.order_col!r} is not a column of the FOV table")

    def _unit_reason(self, data: ValidatedInput) -> str | None:
        """The published criterion averages cells; the FOV-unit variant any observation."""
        if self.resampling_unit != "object":
            return None
        return unsupported_observation_unit(
            data, OBJECT_LIKE_UNITS, method="the object-unit JCB criterion"
        )

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """Averaging CV of the visible data, and the pilot's projection when configured."""
        data.feature_values(self.feature)  # fails early on a non-feature column
        self._check_order_col(data.fovs)
        reason = self._unit_reason(data)
        if reason is not None:
            return self._evidence(
                ("not_applicable", reason), None, len(data.objects), len(data.fovs)
            )
        current = self._precision(data, data.fovs)
        pilot = None
        if self.pilot_fovs is not None and len(data.fovs) >= self.pilot_fovs:
            ordered = data.fovs.sort_values(self.order_col, kind="stable")
            pilot = self._precision(data, ordered.iloc[: self.pilot_fovs])
        return self._evidence(current, pilot, len(data.objects), len(data.fovs))

    def _evidence(self, current, pilot, n_objects: int, n_fovs: int) -> Evidence:
        """Quantities and diagnostics from the current and (if complete) pilot estimates."""
        diagnostics: dict[str, Any] = {
            "scalar_parameter": f"mean of {self.feature!r} over objects",
            "assumes_spatial_uniformity": self.resampling_unit == "object",
        }
        quantities: dict[str, Quantity] = {}
        if isinstance(current, tuple):
            availability, reason = current
            diagnostics["undefined_reason"] = reason
            quantities["averaging_cv"] = _cv_quantity(None, availability, reason)
        else:
            diagnostics.update({k: v for k, v in current.items() if k not in _EFFORT_KEYS})
            quantities["averaging_cv"] = _cv_quantity(current["cv_mean" + self._suffix()])
        if self.pilot_fovs is not None:
            quantities["pilot_projected_averaging_cv"] = self._projection(
                current, pilot, n_objects, n_fovs, diagnostics
            )
        return Evidence(quantities, diagnostics)

    def _projection(self, current, pilot, n_objects, n_fovs, diagnostics) -> Quantity:
        """Averaging CV the fixed pilot estimate projects for the visible effort."""
        if isinstance(current, tuple) and current[0] == "not_applicable":
            return _cv_quantity(None, *current)
        if n_fovs < self.pilot_fovs:
            diagnostics["pilot_status"] = "pilot not yet complete"
            return _cv_quantity(None, "insufficient_data", "pilot not yet complete")
        if isinstance(pilot, tuple):
            # The pilot is fixed once complete, so more acquisition cannot fix it.
            reason = f"pilot: {pilot[1]}"
            diagnostics["pilot_status"] = reason
            return _cv_quantity(None, "not_applicable", reason)
        diagnostics["pilot"] = pilot
        if self.resampling_unit == "object":
            projected = pilot["cv_cells" + self._suffix()] / math.sqrt(n_objects)
        else:
            projected = pilot["cv_mean" + self._suffix()] * math.sqrt(self.pilot_fovs / n_fovs)
        return _cv_quantity(projected)

    def start(
        self, template: ValidatedInput, *, objective: ObjectiveConfig
    ) -> EstimatorState | None:
        """Incremental version: keeps sufficient statistics per acquired FOV.

        Returns ``None`` (full recomputation, which reports not applicable)
        when the observation unit does not fit the criterion.
        """
        template.feature_values(self.feature)
        self._check_order_col(template.fovs)
        if self._unit_reason(template) is not None:
            return None
        return _ScalarPrecisionState(self)


@dataclass(frozen=True)
class _FovSummary:
    """Sufficient statistics of one acquired FOV for the averaging-CV criterion."""

    order: Any
    count: int
    total: float
    mean: float
    m2: float
    has_negative: bool


def _merged_moments(fovs: list[_FovSummary]) -> tuple[int, float, float]:
    """Object count, mean and sum of squared deviations over FOVs (Chan et al. merge)."""
    n, mean, m2 = 0, 0.0, 0.0
    for fov in fovs:
        if fov.count == 0:
            continue
        combined = n + fov.count
        delta = fov.mean - mean
        mean += delta * fov.count / combined
        m2 += fov.m2 + delta**2 * n * fov.count / combined
        n = combined
    return n, mean, m2


class _ScalarPrecisionState(EstimatorState):
    """Per-FOV sufficient statistics; each update costs O(new objects + acquired FOVs).

    The estimate from these statistics equals full recomputation up to
    floating-point rounding: the cell mean and variance are merged per FOV
    instead of computed over all values at once.
    """

    def __init__(self, estimator: ScalarPrecisionEstimator) -> None:
        self.estimator = estimator
        self.fovs: list[_FovSummary] = []

    def _precision(self, fovs: list[_FovSummary]):
        """The estimate from per-FOV summaries, as the full path computes it from values."""
        sums = np.array([fov.total for fov in fovs], dtype=float)
        counts = np.array([fov.count for fov in fovs], dtype=float)

        def cells():
            """Cell-level CVs from the merged moments."""
            n, mean, m2 = _merged_moments(fovs)
            return _cell_cv_from_moments(n, mean, math.sqrt(m2 / (n - 1)))

        return self.estimator._precision_from(
            sums, counts, has_negative=any(fov.has_negative for fov in fovs), cells=cells
        )

    def update(self, new: ValidatedInput) -> Evidence:
        """Summarize the new FOV(s) and recompute the estimate from the summaries."""
        estimator = self.estimator
        values = new.feature_values(estimator.feature)
        positions = fov_positions(new, new.fovs)
        for j in range(len(new.fovs)):
            members = values[positions == j]
            mean = float(members.mean()) if len(members) else 0.0
            self.fovs.append(
                _FovSummary(
                    order=None
                    if estimator.order_col is None
                    else new.fovs[estimator.order_col].iloc[j],
                    count=len(members),
                    total=float(members.sum()),
                    mean=mean,
                    m2=float(((members - mean) ** 2).sum()),
                    has_negative=bool((members < 0).any()),
                )
            )
        current = self._precision(self.fovs)
        pilot = None
        if estimator.pilot_fovs is not None and len(self.fovs) >= estimator.pilot_fovs:
            # Same rule as full recomputation: stable sort by order, ties by arrival.
            ordered = sorted(range(len(self.fovs)), key=lambda i: self.fovs[i].order)
            pilot = self._precision([self.fovs[i] for i in ordered[: estimator.pilot_fovs]])
        n_objects = sum(fov.count for fov in self.fovs)
        return estimator._evidence(current, pilot, n_objects, len(self.fovs))


@dataclass(frozen=True)
class ScalarPrecisionStrategy(SamplingStrategy):
    """Stop when the averaging CV of one scalar feature's mean reaches ``target_cv``.

    ``ScalarPrecisionEstimator`` plus ``ThresholdPolicy(target_cv)`` on
    ``averaging_cv``, or on ``pilot_projected_averaging_cv`` with a pilot. It
    answers the population component only.

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
        _check_configuration(
            self.resampling_unit, self.sd_estimate, self.pilot_fovs, self.order_col
        )

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

    @property
    def estimators(self) -> tuple[ScalarPrecisionEstimator]:
        """The averaging-CV estimator with this strategy's settings."""
        return (
            ScalarPrecisionEstimator(
                self.feature,
                self.resampling_unit,
                self.sd_estimate,
                self.pilot_fovs,
                self.order_col,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """The paper's target applied to the running or pilot-projected averaging CV."""
        quantity = "averaging_cv" if self.pilot_fovs is None else "pilot_projected_averaging_cv"
        return (ThresholdPolicy(quantity, self.target_cv),)
