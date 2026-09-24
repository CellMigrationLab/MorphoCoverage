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

    lofo_novel_fov_fraction = (FOVs showing a new state) / (visible FOVs),

estimates the probability that one more representative FOV shows something
outside the current support at this resolution. A FOV shows a **new state**
when at least ``min_novel_observations`` of its novel observations form one
cluster: they are linked by chains of steps of at most ``epsilon`` between
novel observations of that FOV (single linkage at the same resolution). So
``m`` new cells must resemble each other, and scattered outliers (debris,
segmentation errors) do not add up to a phenotype however many there are.
With ``min_novel_observations=1`` any novel observation counts. Acquired FOVs
without observations count in the denominator, since the next FOV may be
empty too.

With few FOVs that fraction is uninformative: two homogeneous FOVs give 0,
yet say little about a third. ``lofo_novel_fov_upper`` is the one-sided
Clopper-Pearson upper bound at ``confidence`` on the same probability
(``morphocoverage.bounds``), 0.78 for 0 novel FOVs of 2 at 95%, and is the
quantity to stop on when few FOVs have been seen. Support is counted in
FOVs, not cells: a state seen only in one FOV stays novel however many cells
it has there, and support from another FOV counts once however many of its
cells are nearby, so a dense FOV cannot cover what a sparse FOV showing the
same state would not.

Support is symmetric with the new-state rule: another FOV supports an
observation only with at least ``min_novel_observations`` of its own
observations within ``epsilon``. A state must recur as a group of similar
cells to be covered, so a single stray cell (debris, a mis-segmented
object) in another FOV cannot cancel a real new state, just as scattered
outliers cannot make one. With ``min_novel_observations=1`` one nearby
observation is enough, as before.

Single linkage can chain: unsupported cells spaced at most ``epsilon``
apart form one state however far the chain extends. A continuous run of
unseen cells is treated as one new state.

Weighting of the held-out side differs between the quantities:

- ``lofo_novel_fraction`` pools observations, so each FOV counts in
  proportion to its cells: two dense, mutually supporting FOVs of 1,000
  cells and eight one-cell FOVs each in its own unsupported state give
  8 / 2,008 = 0.004, although 8 of 10 FOVs are wholly novel. It estimates
  the chance that a random *cell* of a new FOV is novel, not a
  representative-FOV quantity, and a dense homogeneous FOV can make it
  look settled.
- ``lofo_mean_fov_novel_fraction`` averages the novel fraction per FOV over
  the FOVs with observations (0.8 above): each FOV counts once.
- ``lofo_novel_fov_fraction`` and its bound count FOVs showing a new state
  (each FOV once, empty FOVs in the denominator), the quantity for "would
  one more FOV show something new?".

Recurrence across FOVs is evidence that a state is common only when the FOVs
are representative, exchangeable draws. Adjacent, spatially correlated FOVs
showing the same local patch support each other with ``min_other_fovs=1``
although they may be one occurrence; raise ``min_other_fovs`` or use FOVs far
enough apart when neighbours are correlated.

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
from scipy import sparse
from scipy.sparse import csgraph
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
MEAN_FOV_QUANTITY = "lofo_mean_fov_novel_fraction"
QUANTITIES = (QUANTITY, MEAN_FOV_QUANTITY, FOV_QUANTITY, UPPER_QUANTITY)
_UNITS = {
    QUANTITY: "fraction of observations",
    MEAN_FOV_QUANTITY: "mean fraction of observations per FOV",
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


def _supporting_fovs(
    reference: np.ndarray,
    reference_positions: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
    min_observations: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Reference FOVs with at least ``min_observations`` observations within ``epsilon``.

    One count per query row. ``exclude[i]`` is a reference FOV that must not
    count for query row ``i`` (its own FOV), or ``None``.
    """
    supporting = np.zeros(len(queries), dtype=np.int64)
    for fov in np.unique(reference_positions):
        members = reference[reference_positions == fov]
        if len(members) < min_observations or not len(queries):
            continue
        if min_observations == 1:
            # The bound is exclusive in cKDTree.query, so widen it and compare exactly.
            nearest, _ = cKDTree(members).query(
                queries, k=1, distance_upper_bound=epsilon * (1 + 1e-12)
            )
            near = nearest <= epsilon
        else:
            counts = cKDTree(members).query_ball_point(queries, epsilon, return_length=True)
            near = counts >= min_observations
        if exclude is not None:
            near &= exclude != fov
        supporting += near
    return supporting


def novel_observations(
    values: np.ndarray,
    positions: np.ndarray,
    epsilon: float,
    min_other_fovs: int,
    min_support_observations: int = 1,
) -> np.ndarray:
    """Whether each observation has too few other FOVs supporting it within ``epsilon``.

    Novel means that fewer than ``min_other_fovs`` distinct other FOVs each
    have at least ``min_support_observations`` observations at distance
    ``<= epsilon``. ``positions`` gives each row's FOV.
    """
    if min_other_fovs == 1 and min_support_observations == 1:
        # One other FOV is one other observation: count ball members without
        # listing them (all rows minus the row's own FOV), which is much faster.
        return _other_fov_observations(values, positions, epsilon) < 1
    supporting = _supporting_fovs(
        values, positions, values, epsilon, min_support_observations, exclude=positions
    )
    return supporting < min_other_fovs


def novel_against(
    reference: np.ndarray,
    reference_positions: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
    min_other_fovs: int,
    min_support_observations: int = 1,
) -> np.ndarray:
    """Whether each query row has too few reference FOVs supporting it within ``epsilon``.

    The rule of ``novel_observations`` applied to observations from FOVs that
    are not in the reference set, such as a newly acquired FOV judged against
    all the FOVs acquired before it.
    """
    supporting = _supporting_fovs(
        reference, reference_positions, queries, epsilon, min_support_observations
    )
    return supporting < min_other_fovs


def new_state_members(
    values: np.ndarray,
    positions: np.ndarray,
    novel: np.ndarray,
    epsilon: float,
    min_size: int,
) -> np.ndarray:
    """Novel observations that belong to a cluster of at least ``min_size`` in their FOV.

    Clusters are the connected components of novel observations of one FOV,
    linked when within ``epsilon`` of each other (single linkage).
    """
    members = np.zeros(len(values), dtype=bool)
    if min_size == 1:
        return novel.copy()
    for fov in np.unique(positions[novel]):
        rows = np.flatnonzero(novel & (positions == fov))
        if len(rows) < min_size:
            continue
        pairs = cKDTree(values[rows]).query_pairs(epsilon * (1 + 1e-12), output_type="ndarray")
        # query_pairs uses <=, with the bound widened for rounding; recheck exactly.
        close = np.linalg.norm(values[rows[pairs[:, 0]]] - values[rows[pairs[:, 1]]], axis=1)
        pairs = pairs[close <= epsilon]
        graph = sparse.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(rows), len(rows))
        )
        _, labels = csgraph.connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        members[rows] = sizes[labels] >= min_size
    return members


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

    Reports ``lofo_novel_fraction`` (cell-weighted),
    ``lofo_mean_fov_novel_fraction`` (FOV-weighted), ``lofo_novel_fov_fraction``
    and its upper bound (population, lower is better). See the module
    docstring for the estimands, their weighting and their limits.
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

        novel = novel_observations(
            values, positions, self.epsilon, self.min_other_fovs, self.min_novel_observations
        )
        k = len(data.fovs)
        novel_per_fov = np.bincount(positions, weights=novel, minlength=k)
        counts = np.bincount(positions, minlength=k)
        per_fov = novel_per_fov / np.maximum(counts, 1)
        observed = counts > 0
        members = new_state_members(
            values, positions, novel, self.epsilon, self.min_novel_observations
        )
        novel_fov_count = int(np.unique(positions[members]).size)
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
            MEAN_FOV_QUANTITY: Quantity.available(
                float(per_fov[observed].mean()), **_meta(MEAN_FOV_QUANTITY)
            ),
            FOV_QUANTITY: Quantity.available(novel_fov_count / k, **_meta(FOV_QUANTITY)),
            UPPER_QUANTITY: Quantity.available(upper, **_meta(UPPER_QUANTITY)),
        }
        return Evidence(quantities, diagnostics)

    def novel_fovs_against(self, visible: ValidatedInput, new: ValidatedInput) -> np.ndarray | None:
        """Whether each FOV of ``new`` would count as novel after the FOVs of ``visible``.

        The same definition the estimate uses, applied to FOVs not yet seen: a
        FOV of ``new`` is novel when at least ``min_novel_observations`` of its
        observations have fewer than ``min_other_fovs`` FOVs of ``visible``
        within ``epsilon`` and form one cluster at ``epsilon`` (a new state).
        The transform is fitted on ``visible`` only. FOVs
        without observations are not novel. Returns ``None`` when the estimate
        on ``visible`` would be unavailable (missing values, too few FOVs, a
        transform that cannot be fitted) or ``new`` has missing values.
        """
        reference = visible.feature_matrix()
        queries = new.feature_matrix()
        if np.isnan(reference).any() or np.isnan(queries).any():
            return None
        positions = fov_positions(visible, visible.fovs)
        if len(np.unique(positions)) < self.min_fovs:
            return None
        if self.transform is not None:
            fitted = self.transform.fit(visible)
            if not fitted.fitted:
                return None
            reference, queries = fitted.apply(visible), fitted.apply(new)
        flags = novel_against(
            reference,
            positions,
            queries,
            self.epsilon,
            self.min_other_fovs,
            self.min_novel_observations,
        )
        new_positions = fov_positions(new, new.fovs)
        members = new_state_members(
            queries, new_positions, flags, self.epsilon, self.min_novel_observations
        )
        return np.isin(np.arange(len(new.fovs)), new_positions[members])

    def novel_observation_flags(
        self, data: ValidatedInput, *, new_states_only: bool = False
    ) -> np.ndarray | None:
        """Leave-one-FOV-out novelty of each observation of ``data``, or ``None``.

        The per-observation flags behind the estimate, for inspecting which
        cells were counted as new; ``None`` when the estimate is unavailable.
        With ``new_states_only``, only the novel observations that form a new
        state (a cluster of at least ``min_novel_observations``) are flagged.
        """
        matrix = data.feature_matrix()
        positions = fov_positions(data, data.fovs)
        if np.isnan(matrix).any() or len(np.unique(positions)) < self.min_fovs:
            return None
        if self.transform is not None:
            fitted = self.transform.fit(data)
            if not fitted.fitted:
                return None
            matrix = fitted.apply(data)
        novel = novel_observations(
            matrix, positions, self.epsilon, self.min_other_fovs, self.min_novel_observations
        )
        if not new_states_only:
            return novel
        return new_state_members(
            matrix, positions, novel, self.epsilon, self.min_novel_observations
        )


@dataclass(frozen=True)
class SupportNoveltyStrategy(SamplingStrategy):
    """Stop when a leave-one-FOV-out novel fraction is at most ``tolerance``.

    ``SupportNoveltyEstimator`` plus ``ThresholdPolicy`` on ``quantity``: the
    cell-weighted fraction of novel observations (default), the FOV-weighted
    mean fraction, the fraction of FOVs showing a new state or its upper
    bound. The
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
