"""Multivariate outcome metrics: how far the stop's phenotype distribution is from the reference.

Retrospective, strategy-independent outcomes in the sense of ``outcomes.py``:
``evaluate(full, stopped)`` compares the observations of the stop prefix with
those of the reference (the full group, or a held-out reservoir), never with
a strategy's evidence. They are outcomes, not stopping rules: a two-sample
distance between early and late FOVs measures stationarity, not
completeness (docs/proposals/candidate_estimators_roadmap.md, section 6).

Shared settings, explicit on every metric:

- ``features``: the phenotype columns compared; each must be one of the
  validated feature columns.
- ``standardize``: when true, both samples are centred and scaled by the
  reference's mean and SD (``ddof=1``) per feature, so features in different
  units contribute comparably. The reference is fixed for a group, so every
  stop is measured in the same units. A reference feature with zero spread
  makes the outcome undefined.

Metric-specific settings (quantiles, kernel bandwidth, subsample size, number
of projections, seeds) have no defaults: they define the geometry of the
outcome and belong in its provenance.

Every metric returns ``None`` (undefined, never NaN) when either sample has
too few observations or any selected value is missing: missing values are
neither imputed nor dropped.

Weighting: every metric pools observations, so each FOV counts in proportion
to its number of observations. The estimand is the observation-level
(cell-weighted) phenotype distribution, not that of a typical FOV: a stop
that misses one sparse FOV with a different phenotype scores almost as well
as one that includes it (1,000 cells at 0 in one FOV and 1 cell at 10 in
another give a centroid error of 10/1,001 for a stop without the second FOV,
although the FOV-weighted centroids differ by 5). This favours cell-count
rules when sparse FOVs hold different biology; read these outcomes alongside
FOV-level ones and check sensitivity to observations per FOV.

Bias: the energy distance and MMD are unbiased U-statistics only when the
stop and the reference are independent samples of independent observations.
Against the default reference, the full acquisition, the stop is contained
in the reference: identical observations enter the cross term and bias both
downward (about -0.0004 for the energy distance and -0.0013 for MMD with
bandwidth 1, for a 100-observation stop inside 1,000 i.i.d. normal
observations). Values against the full acquisition are therefore descriptive,
not unbiased; a held-out reservoir (``morphocoverage.reservoir``) keeps the
two samples disjoint. Cells within one FOV are usually correlated as well,
so even then the property holds only approximately; compare strategies at
matched FOV and observation counts. The centroid,
covariance, quantile and Fréchet errors use plug-in estimates and are biased
upward for small stops; compare them at matched observation counts or read
them alongside ``RandomCellRelativeError``-style references. The quadratic
metrics take ``max_observations``: each sample larger than that is replaced
by a seeded random subsample of observations (a Monte Carlo approximation of
the same statistic, not a resampling of FOVs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import distances
from .outcomes import OutcomeMetric
from .schema import ValidatedInput


def _feature_tuple(features: Sequence[str]) -> tuple[str, ...]:
    """A non-empty tuple of distinct column names."""
    if isinstance(features, str) or not features:
        raise TypeError("features must be a non-empty sequence of column names")
    features = tuple(features)
    if not all(isinstance(f, str) for f in features) or len(set(features)) != len(features):
        raise ValueError("features must be distinct column names")
    return features


def _matrix(data: ValidatedInput, features: tuple[str, ...]) -> np.ndarray:
    """The selected features of every observation, in the given order."""
    return np.column_stack([data.feature_values(f) for f in features])


def _samples(
    metric: _MultivariateOutcome, full: ValidatedInput, stopped: ValidatedInput, minimum: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """(stop, reference) matrices on the metric's scale, or ``None`` if undefined."""
    reference = _matrix(full, metric.features)
    sample = _matrix(stopped, metric.features)
    if len(reference) < max(minimum, 2) or len(sample) < minimum:
        return None
    if np.isnan(reference).any() or np.isnan(sample).any():
        return None
    if metric.standardize:
        centre, scale = reference.mean(axis=0), reference.std(axis=0, ddof=1)
        if not (scale > 0).all():
            return None
        reference, sample = (reference - centre) / scale, (sample - centre) / scale
    return sample, reference


def _subsample(values: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    """At most ``limit`` rows, chosen at random without replacement from ``seed``."""
    if limit is None or len(values) <= limit:
        return values
    rows = np.random.default_rng(seed).choice(len(values), size=limit, replace=False)
    return values[np.sort(rows)]


@dataclass(frozen=True)
class _MultivariateOutcome(OutcomeMetric):
    """Shared validation and provenance for the metrics in this module."""

    features: tuple[str, ...]
    standardize: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _feature_tuple(self.features))
        if not isinstance(self.standardize, bool):
            raise TypeError("standardize must be True or False")

    @property
    def label(self) -> str:
        """Short name of the metric family; subclasses override."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """``<label>:<feature,...>``."""
        return f"{self.label}:{','.join(self.features)}"

    @property
    def parameters(self) -> dict[str, Any]:
        """Features and scaling; subclasses add their own settings."""
        return {"features": list(self.features), "standardize": self.standardize}


@dataclass(frozen=True)
class CentroidError(_MultivariateOutcome):
    """Root-mean-square difference of the feature means, ``sqrt(mean_j (m_stop - m_ref)^2)``."""

    @property
    def label(self) -> str:
        """``centroid_error``."""
        return "centroid_error"

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """RMS difference of the means; needs one observation at the stop."""
        pair = _samples(self, full, stopped, 1)
        if pair is None:
            return None
        sample, reference = pair
        return float(np.sqrt(np.mean((sample.mean(axis=0) - reference.mean(axis=0)) ** 2)))


@dataclass(frozen=True)
class CovarianceError(_MultivariateOutcome):
    """Relative Frobenius error of the covariance matrix, ``|S_stop - S_ref|_F / |S_ref|_F``."""

    @property
    def label(self) -> str:
        """``covariance_error``."""
        return "covariance_error"

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Relative covariance error; needs two observations at the stop."""
        pair = _samples(self, full, stopped, 2)
        if pair is None:
            return None
        sample, reference = pair
        cov_ref = np.atleast_2d(np.cov(reference, rowvar=False))
        norm = np.linalg.norm(cov_ref)
        if norm == 0:
            return None
        cov_stop = np.atleast_2d(np.cov(sample, rowvar=False))
        return float(np.linalg.norm(cov_stop - cov_ref) / norm)


@dataclass(frozen=True)
class QuantileError(_MultivariateOutcome):
    """Root-mean-square difference of per-feature quantiles, over features and ``quantiles``.

    Quantiles use numpy's default linear interpolation. Tail quantiles of a
    small stop are poorly determined; that imprecision is part of the outcome.
    """

    quantiles: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        quantiles = tuple(float(q) for q in self.quantiles)
        if not quantiles or not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("quantiles must be a non-empty sequence of values in [0, 1]")
        object.__setattr__(self, "quantiles", quantiles)

    @property
    def label(self) -> str:
        """``quantile_error``."""
        return "quantile_error"

    @property
    def parameters(self) -> dict[str, Any]:
        """Features, scaling and the quantiles."""
        return {**super().parameters, "quantiles": list(self.quantiles)}

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """RMS quantile difference; needs one observation at the stop."""
        pair = _samples(self, full, stopped, 1)
        if pair is None:
            return None
        sample, reference = pair
        difference = np.quantile(sample, self.quantiles, axis=0) - np.quantile(
            reference, self.quantiles, axis=0
        )
        return float(np.sqrt(np.mean(difference**2)))


@dataclass(frozen=True)
class FrechetError(_MultivariateOutcome):
    """Fréchet distance between the stop and the reference (``distances.frechet_distance``)."""

    @property
    def label(self) -> str:
        """``frechet_error``."""
        return "frechet_error"

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Fréchet distance; needs two observations at the stop."""
        pair = _samples(self, full, stopped, 2)
        return None if pair is None else distances.frechet_distance(*pair)


@dataclass(frozen=True)
class _SubsampledOutcome(_MultivariateOutcome):
    """A quadratic-cost metric: samples above ``max_observations`` are subsampled with ``seed``."""

    max_observations: int | None
    seed: int

    def __post_init__(self) -> None:
        super().__post_init__()
        limit = self.max_observations
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
            raise TypeError("max_observations must be an integer or None")
        if limit is not None and limit < 2:
            raise ValueError("max_observations must be at least 2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    @property
    def parameters(self) -> dict[str, Any]:
        """Features, scaling, the subsample limit and its seed."""
        return {**super().parameters, "max_observations": self.max_observations, "seed": self.seed}

    def _pair(
        self, full: ValidatedInput, stopped: ValidatedInput
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Scaled (stop, reference) samples, each subsampled to ``max_observations``."""
        pair = _samples(self, full, stopped, 2)
        if pair is None:
            return None
        sample, reference = pair
        # Distinct seeds for the two samples, so equal sizes do not pick equal rows.
        return (
            _subsample(sample, self.max_observations, self.seed),
            _subsample(reference, self.max_observations, self.seed + 1),
        )


@dataclass(frozen=True)
class EnergyError(_SubsampledOutcome):
    """Energy-distance U-statistic between the stop and the reference (``distances``).

    Unbiased only for a reference disjoint from the stop, such as a held-out
    reservoir; against the full acquisition, which contains the stop, it is
    biased downward (module docstring). Can be slightly negative when the two
    distributions are equal.
    """

    @property
    def label(self) -> str:
        """``energy_error``."""
        return "energy_error"

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Energy distance; needs two observations at the stop."""
        pair = self._pair(full, stopped)
        return None if pair is None else distances.energy_distance(*pair)


@dataclass(frozen=True)
class MMDError(_SubsampledOutcome):
    """Squared-MMD U-statistic with a Gaussian kernel of explicit ``bandwidth`` (``distances``).

    ``bandwidth`` is in the metric's feature units (standard deviations when
    ``standardize`` is true). Unbiased only for a reference disjoint from the
    stop (module docstring). Can be slightly negative when the two
    distributions are equal.
    """

    bandwidth: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (float(self.bandwidth) > 0 and np.isfinite(self.bandwidth)):
            raise ValueError("bandwidth must be positive and finite")
        object.__setattr__(self, "bandwidth", float(self.bandwidth))

    @property
    def label(self) -> str:
        """``mmd_error``."""
        return "mmd_error"

    @property
    def parameters(self) -> dict[str, Any]:
        """Shared settings and the kernel bandwidth."""
        return {**super().parameters, "bandwidth": self.bandwidth}

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Squared MMD; needs two observations at the stop."""
        pair = self._pair(full, stopped)
        return None if pair is None else distances.mmd_squared(*pair, self.bandwidth)


@dataclass(frozen=True)
class SlicedWassersteinError(_MultivariateOutcome):
    """Mean 1-Wasserstein distance over ``n_projections`` random directions drawn from ``seed``.

    The directions depend only on the number of features and the seed, so
    every stop of every strategy is projected identically.
    """

    n_projections: int
    seed: int

    def __post_init__(self) -> None:
        super().__post_init__()
        for value, label in ((self.n_projections, "n_projections"), (self.seed, "seed")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
        if self.n_projections < 1:
            raise ValueError("n_projections must be at least 1")

    @property
    def label(self) -> str:
        """``sliced_wasserstein_error``."""
        return "sliced_wasserstein_error"

    @property
    def parameters(self) -> dict[str, Any]:
        """Shared settings, the number of projections and their seed."""
        return {**super().parameters, "n_projections": self.n_projections, "seed": self.seed}

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Sliced Wasserstein distance; needs one observation at the stop."""
        pair = _samples(self, full, stopped, 1)
        if pair is None:
            return None
        directions = distances.random_directions(len(self.features), self.n_projections, self.seed)
        return distances.sliced_wasserstein(*pair, directions)


__all__ = [
    "CentroidError",
    "CovarianceError",
    "EnergyError",
    "FrechetError",
    "MMDError",
    "QuantileError",
    "SlicedWassersteinError",
]
