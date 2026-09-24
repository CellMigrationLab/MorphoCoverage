"""Leave-one-FOV-out support novelty at a declared phenotype resolution.

Roadmap 4.1-4.2 (docs/proposals/candidate_estimators_roadmap.md): are
observations still arriving in regions of phenotype space that the other
FOVs do not cover? At resolution ``epsilon`` an observation is covered when
observations from at least ``min_other_fovs`` *other* FOVs lie within
``epsilon`` (Euclidean, in the transformed feature space). Holding out each
visible FOV in turn estimates how much of a new representative FOV would be
novel:

    lofo_novel_fraction = (novel observations) / (observations), pooled over
    the visible FOVs, each judged against all the others.

This is the leave-one-FOV-out estimate of ``1 - C_epsilon``, the chance that
an observation from another FOV falls outside the current support. At the FOV
level,

    lofo_novel_fov_fraction = (FOVs with at least ``min_novel_observations``
    novel observations) / (visible FOVs),

estimates the probability that one more representative FOV shows something
outside the current support at this resolution. Acquired FOVs without
observations count in the denominator, since the next FOV may be empty too.
Requiring several novel observations keeps a single outlier (for example a
segmentation error) from counting as a new phenotype.

With few FOVs that fraction is uninformative: two homogeneous FOVs give 0,
yet say little about a third. ``lofo_novel_fov_upper`` is the one-sided
Clopper-Pearson upper bound at ``confidence`` on the same probability
(``morphocoverage.bounds``), 0.78 for 0 novel FOVs of 2 at 95%, and is the
quantity to stop on when few FOVs have been seen. It counts
FOVs, not cells, on both sides: a state seen only in one FOV stays novel
however many cells it has there, and support from another FOV counts once
however many of its cells are nearby, so a dense FOV cannot cover what a
sparse FOV showing the same state would not.

Choices and limits:

- The estimate uses ``k - 1`` FOVs for each held-out FOV, so it slightly
  overstates the novelty of the full visible set.
- It is resolution-dependent by construction: ``epsilon`` and
  ``min_other_fovs`` define the estimand and have no defaults. Density varies
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

Cost: ``scipy.spatial.cKDTree`` queries. With ``min_other_fovs=1``, counting
ball queries (about 10 ms for a 1,000-cell well); otherwise one bounded
nearest-neighbour query of every observation per FOV (about 0.5 s for 6,700
cells in 168 FOVs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..bounds import check_confidence, upper_bound
from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from ..transforms import FeatureTransform
from .base import Estimator, SamplingStrategy

QUANTITY = "lofo_novel_fraction"
FOV_QUANTITY = "lofo_novel_fov_fraction"
UPPER_QUANTITY = "lofo_novel_fov_upper"
QUANTITIES = (QUANTITY, FOV_QUANTITY, UPPER_QUANTITY)
_UNITS = {
    QUANTITY: "fraction of observations",
    FOV_QUANTITY: "fraction of FOVs (probability)",
    UPPER_QUANTITY: "probability (upper confidence bound)",
}


def _check(epsilon, min_other_fovs, min_fovs, transform) -> None:
    """Validate the settings shared by the estimator and the strategy."""
    if isinstance(epsilon, bool) or not (float(epsilon) > 0 and np.isfinite(epsilon)):
        raise ValueError("epsilon must be a positive, finite distance")
    for value, label, minimum in (
        (min_other_fovs, "min_other_fovs", 1),
        (min_fovs, "min_fovs", 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if transform is not None and not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform or None (raw feature units)")


def _other_fov_observations(values: np.ndarray, positions: np.ndarray, epsilon: float):
    """Observations of other FOVs within ``epsilon`` of each row, by counting queries."""
    total = cKDTree(values).query_ball_point(values, epsilon, return_length=True)
    same = np.empty(len(values), dtype=np.int64)
    for fov in np.unique(positions):
        rows = np.flatnonzero(positions == fov)
        same[rows] = cKDTree(values[rows]).query_ball_point(
            values[rows], epsilon, return_length=True
        )
    return total - same


def _check_novel(min_novel_observations) -> None:
    """A positive integer number of novel observations per FOV."""
    value = min_novel_observations
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("min_novel_observations must be a positive integer")


def novel_observations(
    values: np.ndarray, positions: np.ndarray, epsilon: float, min_other_fovs: int
) -> np.ndarray:
    """Whether each observation has too few other FOVs within ``epsilon``.

    Novel means that fewer than ``min_other_fovs`` distinct other FOVs have an
    observation at distance ``<= epsilon``. ``positions`` gives each row's FOV.
    """
    if min_other_fovs == 1:
        # One other FOV is one other observation: count ball members without
        # listing them (all rows minus the row's own FOV), which is much faster.
        return _other_fov_observations(values, positions, epsilon) < 1
    # One nearest-neighbour query per FOV: does that FOV reach each row within epsilon?
    supporting = np.zeros(len(values), dtype=np.int64)
    for fov in np.unique(positions):
        own = positions == fov
        # The bound is exclusive in cKDTree.query, so widen it and compare exactly.
        nearest, _ = cKDTree(values[own]).query(
            values, k=1, distance_upper_bound=epsilon * (1 + 1e-12)
        )
        supporting += (nearest <= epsilon) & ~own
    return supporting < min_other_fovs


def _meta(name: str) -> dict[str, Any]:
    """Direction, units and component of a novelty quantity."""
    return {
        "direction": Direction.LOWER_IS_BETTER,
        "units": _UNITS[name],
        "component": Component.POPULATION,
    }


def _unavailable(kind: str, reason: str, diagnostics: dict[str, Any]) -> Evidence:
    """Both quantities insufficient or not applicable, with the reason."""
    make = Quantity.insufficient if kind == "insufficient" else Quantity.not_applicable
    quantities = {name: make(reason, **_meta(name)) for name in QUANTITIES}
    return Evidence(quantities, {**diagnostics, "undefined_reason": reason})


@dataclass(frozen=True)
class SupportNoveltyEstimator(Estimator):
    """Leave-one-FOV-out fraction of observations outside the other FOVs' ``epsilon``-support.

    Reports ``lofo_novel_fraction`` and ``lofo_novel_fov_fraction``
    (population, lower is better). See the module docstring for the estimands
    and their limits.
    """

    epsilon: float
    min_other_fovs: int
    transform: FeatureTransform | None
    min_fovs: int
    min_novel_observations: int = 1
    confidence: float = 0.95
    name: str = "support_novelty"

    def __post_init__(self) -> None:
        _check(self.epsilon, self.min_other_fovs, self.min_fovs, self.transform)
        _check_novel(self.min_novel_observations)
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "confidence", check_confidence(self.confidence))

    @property
    def parameters(self) -> dict[str, Any]:
        """The resolution, neighbour count, transform and minimum FOVs."""
        return {
            "epsilon": self.epsilon,
            "min_other_fovs": self.min_other_fovs,
            "transform": None if self.transform is None else self.transform.describe(),
            "min_fovs": self.min_fovs,
            "min_novel_observations": self.min_novel_observations,
            "confidence": self.confidence,
        }

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The leave-one-FOV-out novel fractions of observations and of FOVs."""
        matrix = data.feature_matrix()
        if np.isnan(matrix).any():
            return _unavailable("not_applicable", "missing values in the selected features", {})
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
            return _unavailable("insufficient", reason, diagnostics)

        novel = novel_observations(values, positions, self.epsilon, self.min_other_fovs)
        k = len(data.fovs)
        novel_per_fov = np.bincount(positions, weights=novel, minlength=k)
        counts = np.bincount(positions, minlength=k)
        per_fov = novel_per_fov / np.maximum(counts, 1)
        observed = counts > 0
        novel_fov_count = int(np.sum(novel_per_fov >= self.min_novel_observations))
        upper = upper_bound(novel_fov_count, k, self.confidence)
        diagnostics.update(
            {
                "novel_observations": int(novel.sum()),
                "novel_fovs": novel_fov_count,
                "fovs_with_novel_observations": int(np.unique(positions[novel]).size),
                "mean_fov_novel_fraction": float(per_fov[observed].mean()),
            }
        )
        quantities = {
            QUANTITY: Quantity.available(float(novel.mean()), **_meta(QUANTITY)),
            FOV_QUANTITY: Quantity.available(novel_fov_count / k, **_meta(FOV_QUANTITY)),
            UPPER_QUANTITY: Quantity.available(upper, **_meta(UPPER_QUANTITY)),
        }
        return Evidence(quantities, diagnostics)


@dataclass(frozen=True)
class SupportNoveltyStrategy(SamplingStrategy):
    """Stop when a leave-one-FOV-out novel fraction is at most ``tolerance``.

    ``SupportNoveltyEstimator`` plus ``ThresholdPolicy`` on ``quantity``: the
    fraction of observations (default) or of FOVs that are novel. The
    resolution, support and minimum FOVs have no defaults.
    """

    epsilon: float
    tolerance: float
    min_other_fovs: int
    transform: FeatureTransform | None
    min_fovs: int
    min_novel_observations: int = 1
    quantity: str = QUANTITY
    confidence: float = 0.95
    name: str = "support_novelty"

    def __post_init__(self) -> None:
        _check(self.epsilon, self.min_other_fovs, self.min_fovs, self.transform)
        _check_novel(self.min_novel_observations)
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        check_confidence(self.confidence)
        if not 0 <= self.tolerance < 1:
            raise ValueError("tolerance must be a fraction in [0, 1)")

    @property
    def parameters(self) -> dict[str, Any]:
        """The estimator's settings and the tolerance."""
        return {
            **self.estimators[0].parameters,
            "quantity": self.quantity,
            "tolerance": self.tolerance,
        }

    @property
    def estimators(self) -> tuple[SupportNoveltyEstimator]:
        """The support-novelty estimator with this strategy's settings."""
        return (
            SupportNoveltyEstimator(
                self.epsilon,
                self.min_other_fovs,
                self.transform,
                self.min_fovs,
                self.min_novel_observations,
                self.confidence,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``quantity <= tolerance``."""
        return (ThresholdPolicy(self.quantity, self.tolerance),)
