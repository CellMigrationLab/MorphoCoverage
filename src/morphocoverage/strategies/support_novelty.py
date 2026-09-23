"""Leave-one-FOV-out support novelty at a declared phenotype resolution.

Roadmap 4.1-4.2 (docs/proposals/candidate_estimators_roadmap.md): are
observations still arriving in regions of phenotype space that the other
FOVs do not cover? At resolution ``epsilon`` an observation is covered when
at least ``n_neighbors`` observations of *other* FOVs lie within
``epsilon`` (Euclidean, in the transformed feature space). Holding out each
visible FOV in turn estimates how much of a new representative FOV would be
novel:

    lofo_novel_fraction = (novel observations) / (observations), pooled over
    the visible FOVs, each judged against all the others.

This is the leave-one-FOV-out estimate of ``1 - C_epsilon``, the chance that
an observation from another FOV falls outside the current support. It counts
FOVs, not cells: a state seen only in one FOV stays novel however many cells
it has there, and becomes covered once a second FOV shows it.

Choices and limits:

- The estimate uses ``k - 1`` FOVs for each held-out FOV, so it slightly
  overstates the novelty of the full visible set.
- It is resolution-dependent by construction: ``epsilon`` and
  ``n_neighbors`` define the estimand and have no defaults. Density varies
  across phenotype space, so one global ``epsilon`` is coarse where
  phenotypes are dense and fine where they are sparse.
- In high dimensions nearest-neighbour distances concentrate, and a single
  ``epsilon`` either covers everything or nothing; use a transform that
  reduces the dimension.
- ``transform`` is fitted once on all visible data (including the held-out
  FOV), which uses no future FOVs but is not a fully nested holdout.
- It only sees states that have appeared: a phenotype absent from every
  visible FOV contributes nothing.
- Zero-observation FOVs add no observations to judge; they still count as
  acquisition effort.

Cost: one k-d tree over all visible observations and one per FOV
(``scipy.spatial.cKDTree``), with counting ball queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from ..transforms import FeatureTransform
from .base import Estimator, SamplingStrategy

QUANTITY = "lofo_novel_fraction"


def _check(epsilon, n_neighbors, min_fovs, transform) -> None:
    """Validate the settings shared by the estimator and the strategy."""
    if isinstance(epsilon, bool) or not (float(epsilon) > 0 and np.isfinite(epsilon)):
        raise ValueError("epsilon must be a positive, finite distance")
    for value, label, minimum in ((n_neighbors, "n_neighbors", 1), (min_fovs, "min_fovs", 2)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if transform is not None and not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform or None (raw feature units)")


def novel_observations(
    values: np.ndarray, positions: np.ndarray, epsilon: float, n_neighbors: int
) -> np.ndarray:
    """Whether each observation has too few other-FOV observations within ``epsilon``.

    Novel means fewer than ``n_neighbors`` observations of other FOVs at
    distance ``<= epsilon``. ``positions`` gives each row's FOV. Counts over all
    rows, minus those within its own FOV, are the other-FOV counts.
    """
    total = cKDTree(values).query_ball_point(values, epsilon, return_length=True)
    same = np.empty(len(values), dtype=np.int64)
    for fov in np.unique(positions):
        rows = np.flatnonzero(positions == fov)
        same[rows] = cKDTree(values[rows]).query_ball_point(
            values[rows], epsilon, return_length=True
        )
    return (total - same) < n_neighbors


def _meta() -> dict[str, Any]:
    """Direction, units and component of the novelty fraction."""
    return {
        "direction": Direction.LOWER_IS_BETTER,
        "units": "fraction of observations",
        "component": Component.POPULATION,
    }


@dataclass(frozen=True)
class SupportNoveltyEstimator(Estimator):
    """Leave-one-FOV-out fraction of observations outside the other FOVs' ``epsilon``-support.

    Reports ``lofo_novel_fraction`` (population, lower is better). See the
    module docstring for the estimand and its limits.
    """

    epsilon: float
    n_neighbors: int
    transform: FeatureTransform | None
    min_fovs: int
    name: str = "support_novelty"

    def __post_init__(self) -> None:
        _check(self.epsilon, self.n_neighbors, self.min_fovs, self.transform)
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @property
    def parameters(self) -> dict[str, Any]:
        """The resolution, neighbour count, transform and minimum FOVs."""
        return {
            "epsilon": self.epsilon,
            "n_neighbors": self.n_neighbors,
            "transform": None if self.transform is None else self.transform.describe(),
            "min_fovs": self.min_fovs,
        }

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The pooled leave-one-FOV-out novel fraction."""
        matrix = data.feature_matrix()
        if np.isnan(matrix).any():
            reason = "missing values in the selected features"
            quantity = Quantity.not_applicable(reason, **_meta())
            return Evidence({QUANTITY: quantity}, {"undefined_reason": reason})
        positions = fov_positions(data, data.fovs)
        diagnostics: dict[str, Any] = {"fovs_with_observations": len(np.unique(positions))}
        values, reason = matrix, None
        if self.transform is not None:
            fitted = self.transform.fit(data)
            diagnostics["transform"] = fitted.provenance()
            if fitted.fitted:
                values = fitted.apply(data)
            else:
                reason = f"the transform could not be fitted: {fitted.reason}"
        if diagnostics["fovs_with_observations"] < self.min_fovs:
            reason = f"fewer than {self.min_fovs} visible FOVs with observations"
        if reason is not None:
            diagnostics["undefined_reason"] = reason
            return Evidence({QUANTITY: Quantity.insufficient(reason, **_meta())}, diagnostics)

        novel = novel_observations(values, positions, self.epsilon, self.n_neighbors)
        per_fov = np.bincount(positions, weights=novel) / np.maximum(np.bincount(positions), 1)
        observed = np.bincount(positions) > 0
        diagnostics.update(
            {
                "novel_observations": int(novel.sum()),
                "fovs_with_novel_observations": int(np.unique(positions[novel]).size),
                "mean_fov_novel_fraction": float(per_fov[observed].mean()),
            }
        )
        return Evidence({QUANTITY: Quantity.available(float(novel.mean()), **_meta())}, diagnostics)


@dataclass(frozen=True)
class SupportNoveltyStrategy(SamplingStrategy):
    """Stop when the leave-one-FOV-out novel fraction is at most ``tolerance``.

    ``SupportNoveltyEstimator`` plus ``ThresholdPolicy`` on
    ``lofo_novel_fraction``. Every setting defines the rule and has no default.
    """

    epsilon: float
    tolerance: float
    n_neighbors: int
    transform: FeatureTransform | None
    min_fovs: int
    name: str = "support_novelty"

    def __post_init__(self) -> None:
        _check(self.epsilon, self.n_neighbors, self.min_fovs, self.transform)
        if not 0 <= self.tolerance < 1:
            raise ValueError("tolerance must be a fraction in [0, 1)")

    @property
    def parameters(self) -> dict[str, Any]:
        """The estimator's settings and the tolerance."""
        return {**self.estimators[0].parameters, "tolerance": self.tolerance}

    @property
    def estimators(self) -> tuple[SupportNoveltyEstimator]:
        """The support-novelty estimator with this strategy's settings."""
        return (
            SupportNoveltyEstimator(self.epsilon, self.n_neighbors, self.transform, self.min_fovs),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``lofo_novel_fraction <= tolerance``."""
        return (ThresholdPolicy(QUANTITY, self.tolerance),)
