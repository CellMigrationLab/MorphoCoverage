"""FOV-bootstrap uncertainty of multivariate population summaries.

Roadmap Phase 1 (docs/proposals/candidate_estimators_roadmap.md): how
uncertain is the current estimate of simple population properties when the
FOV, not the object, is the sampling unit? The visible FOVs are resampled with
replacement (whole FOVs, including acquired FOVs with no observations), the
chosen summary is recomputed on the pooled observations of each resample,
and the estimator reports the bootstrap standard error of the summary's
components two ways: ``fov_bootstrap_<summary>_se``, their root-mean-square,
and ``fov_bootstrap_<summary>_max_se``, the largest. The RMS is an average:
adding perfectly stable features lowers it although the uncertainty of an
unstable feature is unchanged (one unresolved feature among 100 constant ones
has a tenth of its own SE). The maximum is not diluted, so it is the quantity
for "is every selected feature characterized well enough?", and the strategy
stops on it by default. Components share one scale only after a transform
such as ``Standardize``; in raw units the largest SE is that of the feature
with the largest units.

Summaries:

- ``"centroid"``: the per-observation mean of each feature (a ratio estimator
  under cluster sampling). For one feature without a transform this extends
  the FOV-unit JCB rule (``ScalarPrecisionStrategy(..., "fov", ...)``) from
  one mean to a vector of means; the bootstrap SE approaches the analytic
  ratio-estimator SE times ``sqrt((k - 1) / k)``.
- ``"covariance"``: every entry of the feature covariance matrix.
- ``"quantiles"``: the given per-feature quantiles.
- ``"eigenspectrum"``: the explained-variance fractions of the covariance
  eigenvalues, largest first (roadmap 1.6).
- ``"subspace"``: the leading ``n_components`` principal directions (roadmap
  1.7). Directions have no sign or order within the subspace, so instead of a
  standard error the estimator reports the root-mean-square, over resamples,
  of the largest principal angle (``scipy.linalg.subspace_angles``, radians)
  between each resample's subspace and the subspace of all visible data.
  Near-equal eigenvalues at the cut make this large, as they should: the
  subspace is then genuinely undetermined.

This is a MorphoCoverage method, not a published stopping rule. Its limits,
stated in the roadmap and not removed by any setting here:

- it resamples only what has been seen, so an entirely unseen rare component
  contributes zero variance: low uncertainty is not evidence that no unseen
  phenotype remains;
- with few FOVs the bootstrap is discrete and understates variance (by about
  ``(k - 1) / k``), hence the explicit ``min_fovs``;
- tail quantiles are poorly supported by few FOVs or observations;
- neighbouring FOVs that are spatially correlated make independent FOV
  resampling optimistic. ``block_col`` names an FOV-table column of spatial
  blocks (for example from ``reservoir.spatial_blocks``); whole blocks are then
  resampled, so correlation within a block is kept (a spatial block bootstrap,
  roadmap 8.4). Blocks should be at least as large as the correlation length;
  correlation between blocks is still ignored, and ``min_fovs`` then also
  requires that many blocks;
- the value depends on feature scale, so ``transform`` is explicit (``None``
  means raw feature units). The transform is fitted once on the visible data
  and held fixed across resamples.

Missing feature values make the quantity ``not_applicable`` (more acquisition
does not remove them). Fewer than ``min_fovs`` visible FOVs with observations,
or a transform that cannot be fitted, give ``insufficient_data``. Resamples
that draw only FOVs without observations (or, for the covariance, fewer than 2
observations) have no summary; they are left out and counted in the
diagnostics. PCA resamples whose covariance has observations but too little
spread to define the summary (no spread for the eigenspectrum; a zero
eigenvalue among the leading ``n_components`` for the subspace, whose basis
would be arbitrary) are genuinely undetermined. When they are more than
``DEGENERATE_TOLERANCE`` (5%) of the resamples the quantity is
``insufficient_data``, since dropping them would report certainty that the
data do not support (two one-object FOVs: half the resamples, and an SE of 0
if dropped). Below that share they are left out and counted
(``degenerate_resamples``, ``degenerate_fraction``); this slightly understates
the uncertainty. The rule depends on the share, a property of the data,
not on whether one rare degenerate resample happens to be drawn, which with
enough resamples always happens: five one-object FOVs repeat one FOV in 0.16%
of resamples (5 * 0.2 ** 5), so 2,000 resamples almost surely draw one.

Blocks: with ``block_col`` the bootstrap resamples blocks, each with equal
probability, and takes every FOV of a drawn block. This is the cluster
bootstrap for blocks as the sampling units, so the random total size of a
resample is part of its uncertainty. With blocks of very different
occupancy (for example grid objects cut by the sample edge) that size
variation can dominate the standard error even when the design sampled
FOVs, not blocks: 9 FOVs at 0 in one block and 1 FOV at 10 in another give a
centroid SE of about 4.1, against 0.95 for the FOV bootstrap. Use many
blocks of similar occupancy; ``block_sizes`` in the diagnostics reports the
smallest and largest visible block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from ..transforms import FeatureTransform
from .base import Estimator, SamplingStrategy

SUMMARIES = ("centroid", "covariance", "quantiles", "eigenspectrum", "subspace")
AGGREGATES = ("max", "rms")
# Largest share of PCA resamples without enough spread that is left out.
DEGENERATE_TOLERANCE = 0.05


def _check(
    summary: str,
    n_resamples: int,
    random_state: int,
    min_fovs: int,
    quantiles: Sequence[float] | None,
    transform: FeatureTransform | None,
    n_components: int | None,
) -> None:
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
    if (summary == "subspace") != (n_components is not None):
        raise ValueError("n_components is required for, and only for, summary='subspace'")
    if n_components is not None and (
        isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1
    ):
        raise ValueError("n_components must be a positive integer")
    if transform is not None and not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform or None (raw feature units)")


def _per_fov_sums(values: np.ndarray, positions: np.ndarray, k: int) -> np.ndarray:
    """Sum of each row of ``values`` over the observations of each of ``k`` FOVs."""
    sums = np.zeros((k, values.shape[1]))
    np.add.at(sums, positions, values)
    return sums


def centroid_resamples(counts: np.ndarray, sums: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pooled mean of each resample; ``weights[b, j]`` is how often FOV ``j`` was drawn.

    Rows of resamples without observations are NaN.
    """
    n = weights @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ sums) / n[:, None]


def covariance_resamples(
    counts: np.ndarray, sums: np.ndarray, outer: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Pooled covariance (``ddof=1``) of each resample, flattened; NaN below 2 observations."""
    n = weights @ counts
    total, cross = weights @ sums, weights @ outer
    mean_outer = np.einsum("bi,bj->bij", total, total).reshape(len(weights), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cross - mean_outer / n[:, None]) / (n - 1)[:, None]


def quantile_resamples(
    values: np.ndarray, positions: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Per-feature quantiles of the pooled observations of each resample, flattened.

    ``weights[b, j]`` is how often FOV ``j`` was drawn in resample ``b``.
    Resamples without observations are NaN. One ``np.quantile`` call per
    resample.
    """
    k = weights.shape[1]
    by_fov = np.argsort(positions, kind="stable")
    rows_of = np.split(by_fov, np.cumsum(np.bincount(positions, minlength=k))[:-1])
    out = np.full((len(weights), len(quantiles) * values.shape[1]), np.nan)
    for b, counts in enumerate(weights.astype(int)):
        rows = np.concatenate([rows_of[j] for j in np.repeat(np.arange(k), counts)])
        if len(rows):
            out[b] = np.quantile(values[rows], quantiles, axis=0).ravel()
    return out


def eigenspectrum_resamples(covariances: np.ndarray, d: int) -> np.ndarray:
    """Explained-variance fractions, largest first, of each flattened ``d x d`` covariance."""
    eigenvalues = np.clip(np.linalg.eigvalsh(covariances.reshape(-1, d, d))[:, ::-1], 0, None)
    # A resample with no spread at all has undefined fractions (NaN, left out later).
    with np.errstate(invalid="ignore", divide="ignore"):
        return eigenvalues / eigenvalues.sum(axis=1, keepdims=True)


def subspace_angles_to(covariances: np.ndarray, reference: np.ndarray, r: int) -> np.ndarray:
    """Largest principal angle (radians) between leading ``r``-dimensional subspaces.

    One angle per flattened ``d x d`` matrix in ``covariances``, measured
    against the leading subspace of ``reference``.
    """
    d = len(reference)
    target = np.linalg.eigh(reference)[1][:, ::-1][:, :r]
    vectors = np.linalg.eigh(covariances.reshape(-1, d, d))[1][:, :, ::-1][:, :, :r]
    return np.array([linalg.subspace_angles(v, target).max() for v in vectors])


def _meta(units: str, summary: str) -> dict[str, Any]:
    """Direction, units and component of the reported standard error.

    The centroid is a population summary; the covariance, the quantiles and
    the PCA summaries describe the distribution of phenotypes, so their
    uncertainty is distribution evidence (the phenotypic fingerprint).
    """
    return {
        "direction": Direction.LOWER_IS_BETTER,
        "units": units,
        "component": Component.POPULATION if summary == "centroid" else Component.DISTRIBUTION,
    }


def _visible_blocks(data: ValidatedInput, block_col: str | None) -> int | None:
    """Number of visible spatial blocks, or ``None`` without blocks; checks the column."""
    if block_col is None:
        return None
    if block_col not in data.fovs.columns:
        raise ValueError(f"block_col {block_col!r} is not a column of the FOV table")
    if data.fovs[block_col].isna().any():
        raise ValueError(f"block_col {block_col!r} contains missing values")
    return int(data.fovs[block_col].nunique())


def _pca_reason(estimator: FOVBootstrapEstimator, values: np.ndarray) -> str | None:
    """Why a PCA summary is undefined for these features, if it is."""
    d = values.shape[1]
    if d < 2:
        return "PCA summaries need at least 2 features"
    if estimator.summary == "subspace" and estimator.n_components >= d:
        return f"n_components={estimator.n_components} must be below the {d} features"
    return None


@dataclass(frozen=True)
class FOVBootstrapEstimator(Estimator):
    """Root-mean-square FOV-bootstrap standard error of a multivariate summary.

    Reports ``fov_bootstrap_<summary>_se`` (lower is better; population
    evidence for the centroid, distribution evidence for the other summaries),
    in the units of the transformed features. See the module docstring for the
    summaries, their limits and missing-data behaviour.
    """

    summary: str
    transform: FeatureTransform | None
    n_resamples: int
    random_state: int
    min_fovs: int
    quantiles: tuple[float, ...] | None = None
    n_components: int | None = None
    block_col: str | None = None

    def __post_init__(self) -> None:
        if self.quantiles is not None:
            object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        if self.block_col is not None and not isinstance(self.block_col, str):
            raise TypeError("block_col must be an FOV-table column name or None")
        _check(
            self.summary,
            self.n_resamples,
            self.random_state,
            self.min_fovs,
            self.quantiles,
            self.transform,
            self.n_components,
        )

    @property
    def name(self) -> str:
        """``fov_bootstrap_<summary>``."""
        return f"fov_bootstrap_{self.summary}"

    @property
    def quantity(self) -> str:
        """``fov_bootstrap_<summary>_se`` (RMS over components), or the subspace angle."""
        if self.summary == "subspace":
            return "fov_bootstrap_subspace_angle"
        return f"fov_bootstrap_{self.summary}_se"

    @property
    def max_quantity(self) -> str | None:
        """``fov_bootstrap_<summary>_max_se``, the largest component SE; ``None`` for subspace."""
        return None if self.summary == "subspace" else f"fov_bootstrap_{self.summary}_max_se"

    def _every(self, quantity: Quantity) -> dict[str, Quantity]:
        """The same quantity under every name this estimator reports."""
        names = [self.quantity] + ([self.max_quantity] if self.max_quantity else [])
        return dict.fromkeys(names, quantity)

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
            "n_components": self.n_components,
            "block_col": self.block_col,
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

    def _weights(self, data: ValidatedInput) -> np.ndarray:
        """How often each visible FOV enters each resample: FOVs, or whole blocks, drawn."""
        if self.block_col is None:
            block_of_fov = np.arange(len(data.fovs))
        else:
            block_of_fov = np.unique(data.fovs[self.block_col].to_numpy(), return_inverse=True)[1]
        n_blocks = int(block_of_fov.max()) + 1
        rng = np.random.default_rng(self.random_state)
        draws = rng.integers(0, n_blocks, (self.n_resamples, n_blocks))
        block_weights = np.zeros((self.n_resamples, n_blocks))
        np.add.at(block_weights, (np.arange(self.n_resamples)[:, None], draws), 1.0)
        return block_weights[:, block_of_fov]

    def _resampled(
        self, values: np.ndarray, positions: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """The summary's components for each resample, and how many PCA resamples degenerate.

        A PCA resample is degenerate when its covariance has observations but
        does not determine the summary: no spread at all (eigenspectrum), or a
        leading subspace with a zero eigenvalue inside it, whose basis
        ``eigh`` would pick arbitrarily (subspace).
        """
        k = weights.shape[1]
        if self.summary == "quantiles":
            return quantile_resamples(values, positions, weights, self.quantiles), 0
        counts = np.bincount(positions, minlength=k).astype(float)
        sums = _per_fov_sums(values, positions, k)
        if self.summary == "centroid":
            return centroid_resamples(counts, sums, weights), 0
        # Covariance is shift-invariant; centring first keeps the one-pass formula
        # accurate when a feature's mean is large relative to its spread.
        centred = values - values.mean(axis=0)
        centred_sums = _per_fov_sums(centred, positions, k)
        # Outer products only for covariance summaries: memory grows with features squared.
        products = np.einsum("ni,nj->nij", centred, centred).reshape(len(values), -1)
        outer = _per_fov_sums(products, positions, k)
        covariances = covariance_resamples(counts, centred_sums, outer, weights)
        if self.summary == "covariance":
            return covariances, 0
        d = values.shape[1]
        defined = ~np.isnan(covariances).any(axis=1)
        eigenvalues = np.zeros((len(covariances), d))
        eigenvalues[defined] = np.linalg.eigvalsh(covariances[defined].reshape(-1, d, d))[:, ::-1]
        # The summary needs spread in the first r directions (r = 1 for the spectrum).
        r = 1 if self.summary == "eigenspectrum" else self.n_components
        scale = np.maximum(np.abs(eigenvalues).max(axis=1), np.finfo(float).tiny)
        degenerate = defined & (eigenvalues[:, r - 1] <= 1e-12 * scale)
        usable = defined & ~degenerate
        out = np.full((len(covariances), d if self.summary == "eigenspectrum" else 1), np.nan)
        if self.summary == "eigenspectrum":
            out[usable] = eigenspectrum_resamples(covariances[usable], d)
        else:
            reference = np.cov(values, rowvar=False)
            out[usable, 0] = subspace_angles_to(covariances[usable], reference, self.n_components)
        return out, int(degenerate.sum())

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """Bootstrap standard error of the summary under whole-FOV resampling."""
        name = self.quantity
        if np.isnan(data.feature_matrix()).any():
            reason = "missing values in the selected features"
            quantity = Quantity.not_applicable(
                reason, **_meta("units of the selected features", self.summary)
            )
            return Evidence(self._every(quantity), {"undefined_reason": reason})
        positions = fov_positions(data, data.fovs)
        with_observations = len(np.unique(positions))
        values, diagnostics, units = self._values(data)
        diagnostics["fovs_with_observations"] = with_observations
        blocks = _visible_blocks(data, self.block_col)
        if blocks is not None:
            diagnostics["visible_blocks"] = blocks
            sizes = data.fovs[self.block_col].value_counts()
            diagnostics["block_sizes"] = {"min": int(sizes.min()), "max": int(sizes.max())}
        if with_observations < self.min_fovs:
            reason = f"fewer than {self.min_fovs} visible FOVs with observations"
        elif blocks is not None and blocks < self.min_fovs:
            reason = f"fewer than {self.min_fovs} visible spatial blocks"
        elif values is None:
            reason = f"the transform could not be fitted: {diagnostics['transform']['reason']}"
        elif self.summary in ("eigenspectrum", "subspace") and _pca_reason(self, values):
            reason = _pca_reason(self, values)
            quantity = Quantity.not_applicable(reason, **_meta(units, self.summary))
            return Evidence(self._every(quantity), {**diagnostics, "undefined_reason": reason})
        else:
            reason = None
        if reason is not None:
            diagnostics["undefined_reason"] = reason
            return Evidence(
                self._every(Quantity.insufficient(reason, **_meta(units, self.summary))),
                diagnostics,
            )

        components, degenerate = self._resampled(values, positions, self._weights(data))
        defined = ~np.isnan(components).any(axis=1)
        diagnostics["undefined_resamples"] = int((~defined).sum())
        if degenerate:
            diagnostics["degenerate_resamples"] = degenerate
            diagnostics["degenerate_fraction"] = degenerate / self.n_resamples
        if degenerate > DEGENERATE_TOLERANCE * self.n_resamples:
            # Leaving these out would condition the bootstrap on the resamples that
            # happen to show spread and can report near-zero uncertainty; when they
            # are a real share of the bootstrap distribution the summary is not yet
            # determined. The share, not the count, decides, so that more
            # resamples estimate the same property instead of eventually drawing
            # one rare degenerate resample.
            reason = (
                f"{degenerate} of {self.n_resamples} resamples (more than "
                f"{DEGENERATE_TOLERANCE:.0%}) have too little spread to define the PCA summary"
            )
            diagnostics["undefined_reason"] = reason
            return Evidence(
                self._every(Quantity.insufficient(reason, **_meta(units, self.summary))),
                diagnostics,
            )
        if defined.sum() < 2:
            reason = "fewer than 2 resamples with a defined summary"
            diagnostics["undefined_reason"] = reason
            return Evidence(
                self._every(Quantity.insufficient(reason, **_meta(units, self.summary))),
                diagnostics,
            )
        if self.summary == "subspace":
            angles = components[defined, 0]
            diagnostics["max_angle"] = float(angles.max())
            value = float(np.sqrt(np.mean(angles**2)))
            return Evidence(
                {name: Quantity.available(value, **_meta("radians", self.summary))}, diagnostics
            )
        if self.summary == "eigenspectrum":
            units = "fraction of total variance"
        se = np.std(components[defined], axis=0, ddof=1)
        value = float(np.sqrt(np.mean(se**2)))
        diagnostics["max_component_se"] = float(se.max())
        diagnostics["max_component"] = int(se.argmax())
        quantities = {
            name: Quantity.available(value, **_meta(units, self.summary)),
            self.max_quantity: Quantity.available(float(se.max()), **_meta(units, self.summary)),
        }
        return Evidence(quantities, diagnostics)


@dataclass(frozen=True)
class FOVBootstrapStrategy(SamplingStrategy):
    """Stop when the FOV-bootstrap standard error of a summary is at most ``tolerance``.

    ``FOVBootstrapEstimator`` plus ``ThresholdPolicy`` on its quantity:
    with ``aggregate="max"`` (the default) the largest component SE, so every
    selected feature (or covariance entry, quantile, eigenvalue fraction) must
    be within tolerance; with ``"rms"`` the root-mean-square over components,
    an average that adding stable features dilutes. The subspace summary has a
    single angle and ignores ``aggregate``. The
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
    n_components: int | None = None
    block_col: str | None = None
    aggregate: str = "max"

    def __post_init__(self) -> None:
        if not (self.tolerance > 0 and np.isfinite(self.tolerance)):
            raise ValueError("tolerance must be positive and finite")
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}")
        if self.quantiles is not None:
            object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        _check(
            self.summary,
            self.n_resamples,
            self.random_state,
            self.min_fovs,
            self.quantiles,
            self.transform,
            self.n_components,
        )

    @property
    def name(self) -> str:
        """``fov_bootstrap_<summary>``."""
        return f"fov_bootstrap_{self.summary}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The estimator's settings and the tolerance."""
        return {
            **self.estimators[0].parameters,
            "tolerance": self.tolerance,
            "aggregate": self.aggregate,
        }

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
                self.n_components,
                self.block_col,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """The largest (or RMS) component SE, or the subspace angle, ``<= tolerance``."""
        estimator = self.estimators[0]
        use_max = self.aggregate == "max" and estimator.max_quantity is not None
        quantity = estimator.max_quantity if use_max else estimator.quantity
        return (ThresholdPolicy(quantity, self.tolerance),)
