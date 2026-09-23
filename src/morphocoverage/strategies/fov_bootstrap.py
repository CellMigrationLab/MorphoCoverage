"""FOV-bootstrap uncertainty of multivariate population summaries.

Roadmap Phase 1 (docs/proposals/candidate_estimators_roadmap.md): how
uncertain is the current estimate of simple population properties when the
FOV, not the cell, is the sampling unit? The visible FOVs are resampled with
replacement (whole FOVs, including acquired FOVs with no observations), the
chosen summary is recomputed on the pooled observations of each resample,
and the estimator reports the root-mean-square bootstrap standard error of
the summary's components.

Summaries:

- ``"centroid"``: the per-observation mean of each feature (a ratio estimator
  under cluster sampling). For one feature without a transform this extends
  the FOV-unit JCB rule (``ScalarPrecisionStrategy(..., "fov", ...)``) from
  one mean to a vector of means; the bootstrap SE approaches the analytic
  ratio-estimator SE times ``sqrt((k - 1) / k)``.
- ``"covariance"``: every entry of the feature covariance matrix.
- ``"quantiles"``: the given per-feature quantiles.

This is a MorphoCoverage method, not a published stopping rule. Its limits,
stated in the roadmap and not removed by any setting here:

- it resamples only what has been seen, so an entirely unseen rare component
  contributes zero variance: low uncertainty is not evidence that no unseen
  phenotype remains;
- with few FOVs the bootstrap is discrete and understates variance (by about
  ``(k - 1) / k``), hence the explicit ``min_fovs``;
- tail quantiles are poorly supported by few FOVs or observations;
- neighbouring FOVs that are spatially correlated make independent FOV
  resampling optimistic;
- the value depends on feature scale, so ``transform`` is explicit (``None``
  means raw feature units). The transform is fitted once on the visible data
  and held fixed across resamples.

Missing feature values make the quantity ``not_applicable`` (more acquisition
does not remove them). Fewer than ``min_fovs`` visible FOVs with observations,
or a transform that cannot be fitted, give ``insufficient_data``. Resamples
that draw only FOVs without observations (or, for the covariance, fewer than 2
observations) have no summary; they are left out and counted in the
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from ..transforms import FeatureTransform
from .base import Estimator, SamplingStrategy

SUMMARIES = ("centroid", "covariance", "quantiles")


def _check(summary, n_resamples, random_state, min_fovs, quantiles, transform) -> None:
    """Validate the settings shared by the estimator and the strategy."""
    if summary not in SUMMARIES:
        raise ValueError(f"summary must be one of {SUMMARIES}")
    for value, label, minimum in ((n_resamples, "n_resamples", 2), (min_fovs, "min_fovs", 2)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    if (summary == "quantiles") != (quantiles is not None):
        raise ValueError("quantiles are required for, and only for, summary='quantiles'")
    if quantiles is not None and (not quantiles or not all(0 <= q <= 1 for q in quantiles)):
        raise ValueError("quantiles must be a non-empty sequence of values in [0, 1]")
    if transform is not None and not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform or None (raw feature units)")


def _per_fov_sums(values: np.ndarray, positions: np.ndarray, k: int) -> np.ndarray:
    """Sum of each row of ``values`` over the observations of each of ``k`` FOVs."""
    sums = np.zeros((k, values.shape[1]))
    np.add.at(sums, positions, values)
    return sums


def centroid_resamples(counts, sums, weights) -> np.ndarray:
    """Pooled mean of each resample; ``weights[b, j]`` is how often FOV ``j`` was drawn.

    Rows of resamples without observations are NaN.
    """
    n = weights @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ sums) / n[:, None]


def covariance_resamples(counts, sums, outer, weights) -> np.ndarray:
    """Pooled covariance (``ddof=1``) of each resample, flattened; NaN below 2 observations."""
    n = weights @ counts
    total, cross = weights @ sums, weights @ outer
    mean_outer = np.einsum("bi,bj->bij", total, total).reshape(len(weights), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cross - mean_outer / n[:, None]) / (n - 1)[:, None]


def quantile_resamples(values, positions, k, draws, quantiles) -> np.ndarray:
    """Per-feature quantiles of the pooled observations of each resample, flattened.

    ``draws[b]`` lists the FOV indices (of ``k``) in resample ``b``. Resamples
    without observations are NaN. One ``np.quantile`` call per resample.
    """
    by_fov = np.argsort(positions, kind="stable")
    rows_of = np.split(by_fov, np.cumsum(np.bincount(positions, minlength=k))[:-1])
    out = np.full((len(draws), len(quantiles) * values.shape[1]), np.nan)
    for b, fovs in enumerate(draws):
        rows = np.concatenate([rows_of[j] for j in fovs])
        if len(rows):
            out[b] = np.quantile(values[rows], quantiles, axis=0).ravel()
    return out


def _meta(units: str) -> dict[str, Any]:
    """Direction, units and component of the reported standard error."""
    return {
        "direction": Direction.LOWER_IS_BETTER,
        "units": units,
        "component": Component.POPULATION,
    }


@dataclass(frozen=True)
class FOVBootstrapEstimator(Estimator):
    """Root-mean-square FOV-bootstrap standard error of a multivariate summary.

    Reports ``fov_bootstrap_<summary>_se`` (population, lower is better), in
    the units of the transformed features. See the module docstring for the
    summaries, their limits and missing-data behaviour.
    """

    summary: str
    transform: FeatureTransform | None
    n_resamples: int
    random_state: int
    min_fovs: int
    quantiles: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.quantiles is not None:
            object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        _check(
            self.summary,
            self.n_resamples,
            self.random_state,
            self.min_fovs,
            self.quantiles,
            self.transform,
        )

    @property
    def name(self) -> str:
        """``fov_bootstrap_<summary>``."""
        return f"fov_bootstrap_{self.summary}"

    @property
    def quantity(self) -> str:
        """Name of the reported quantity."""
        return f"fov_bootstrap_{self.summary}_se"

    @property
    def parameters(self) -> dict[str, Any]:
        """Every setting that defines the estimate, including the transform and seed."""
        return {
            "summary": self.summary,
            "transform": None if self.transform is None else self.transform.describe(),
            "n_resamples": self.n_resamples,
            "random_state": self.random_state,
            "min_fovs": self.min_fovs,
            "quantiles": None if self.quantiles is None else list(self.quantiles),
        }

    def _values(self, data: ValidatedInput) -> tuple[np.ndarray | None, dict[str, Any], str]:
        """Feature values in the estimator's units, provenance, and those units' name."""
        if self.transform is None:
            return data.feature_matrix(), {"transform": None}, "raw feature units"
        fitted = self.transform.fit(data)
        units = f"units of the {self.transform.name!r} transform"
        if not fitted.fitted:
            return None, {"transform": fitted.provenance()}, units
        return fitted.apply(data), {"transform": fitted.provenance()}, units

    def _resampled(self, values: np.ndarray, positions: np.ndarray, k: int) -> np.ndarray:
        """The summary's components for each bootstrap resample of whole FOVs."""
        draws = np.random.default_rng(self.random_state).integers(0, k, (self.n_resamples, k))
        if self.summary == "quantiles":
            return quantile_resamples(values, positions, k, draws, self.quantiles)
        weights = np.zeros((self.n_resamples, k))
        np.add.at(weights, (np.arange(self.n_resamples)[:, None], draws), 1.0)
        counts = np.bincount(positions, minlength=k).astype(float)
        sums = _per_fov_sums(values, positions, k)
        if self.summary == "centroid":
            return centroid_resamples(counts, sums, weights)
        # Outer products only for the covariance: memory grows with features squared.
        products = np.einsum("ni,nj->nij", values, values).reshape(len(values), -1)
        outer = _per_fov_sums(products, positions, k)
        return covariance_resamples(counts, sums, outer, weights)

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """Bootstrap standard error of the summary under whole-FOV resampling."""
        name = self.quantity
        if np.isnan(data.feature_matrix()).any():
            reason = "missing values in the selected features"
            quantity = Quantity.not_applicable(reason, **_meta("units of the selected features"))
            return Evidence({name: quantity}, {"undefined_reason": reason})
        positions = fov_positions(data, data.fovs)
        k = len(data.fovs)
        with_observations = len(np.unique(positions))
        values, diagnostics, units = self._values(data)
        diagnostics["fovs_with_observations"] = with_observations
        if with_observations < self.min_fovs:
            reason = f"fewer than {self.min_fovs} visible FOVs with observations"
        elif values is None:
            reason = f"the transform could not be fitted: {diagnostics['transform']['reason']}"
        else:
            reason = None
        if reason is not None:
            diagnostics["undefined_reason"] = reason
            return Evidence({name: Quantity.insufficient(reason, **_meta(units))}, diagnostics)

        components = self._resampled(values, positions, k)
        defined = ~np.isnan(components).any(axis=1)
        diagnostics["undefined_resamples"] = int((~defined).sum())
        if defined.sum() < 2:
            reason = "fewer than 2 resamples with a defined summary"
            diagnostics["undefined_reason"] = reason
            return Evidence({name: Quantity.insufficient(reason, **_meta(units))}, diagnostics)
        se = np.std(components[defined], axis=0, ddof=1)
        value = float(np.sqrt(np.mean(se**2)))
        diagnostics["max_component_se"] = float(se.max())
        return Evidence({name: Quantity.available(value, **_meta(units))}, diagnostics)


@dataclass(frozen=True)
class FOVBootstrapStrategy(SamplingStrategy):
    """Stop when the FOV-bootstrap standard error of a summary is at most ``tolerance``.

    ``FOVBootstrapEstimator`` plus ``ThresholdPolicy`` on its quantity. The
    tolerance is in the estimator's units (for ``Standardize``, standard
    deviations of the visible data). No setting has a default: the summary,
    scale, resampling and minimum information all define the rule.
    """

    summary: str
    tolerance: float
    transform: FeatureTransform | None
    n_resamples: int
    random_state: int
    min_fovs: int
    quantiles: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not (self.tolerance > 0 and np.isfinite(self.tolerance)):
            raise ValueError("tolerance must be positive and finite")
        if self.quantiles is not None:
            object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        _check(
            self.summary,
            self.n_resamples,
            self.random_state,
            self.min_fovs,
            self.quantiles,
            self.transform,
        )

    @property
    def name(self) -> str:
        """``fov_bootstrap_<summary>``."""
        return f"fov_bootstrap_{self.summary}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The estimator's settings and the tolerance."""
        return {**self.estimators[0].parameters, "tolerance": self.tolerance}

    @property
    def estimators(self) -> tuple[FOVBootstrapEstimator]:
        """The bootstrap estimator with this strategy's settings."""
        return (
            FOVBootstrapEstimator(
                self.summary,
                self.transform,
                self.n_resamples,
                self.random_state,
                self.min_fovs,
                self.quantiles,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``fov_bootstrap_<summary>_se <= tolerance``."""
        return (ThresholdPolicy(self.estimators[0].quantity, self.tolerance),)
