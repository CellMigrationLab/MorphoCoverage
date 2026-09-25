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
``m`` new objects must resemble each other, and scattered outliers (debris,
segmentation errors) do not add up to a phenotype however many there are.
With ``min_novel_observations=1`` any novel observation counts. Acquired FOVs
without observations count in the denominator, since the next FOV may be
empty too.

With few FOVs that fraction is uninformative: two homogeneous FOVs give 0,
yet say little about a third. ``lofo_novel_fov_upper`` is the one-sided
Clopper-Pearson upper bound at nominal level ``confidence`` on the same
probability. The leave-one-FOV-out outcomes are not independent, so its
coverage is approximate and is checked empirically by
``morphocoverage.calibration`` (``morphocoverage.bounds``). It is 0.78 for 0
novel FOVs of 2 at a nominal 95%, and is the quantity to stop on when few
FOVs have been seen. Support is counted in
FOVs, not objects: a state seen only in one FOV stays novel however many objects
it has there, and support from another FOV counts once however many of its
objects are nearby, so a dense FOV cannot cover what a sparse FOV showing the
same state would not.

Support is symmetric with the new-state rule: another FOV supports an
observation only with at least ``min_novel_observations`` of its own
observations within ``epsilon``. A state must recur as a group of similar
objects to be covered, so a single stray object (debris, a mis-segmented
object) in another FOV cannot cancel a real new state, just as scattered
outliers cannot make one. With ``min_novel_observations=1`` one nearby
observation is enough, as before.

Single linkage can chain: unsupported objects spaced at most ``epsilon``
apart form one state however far the chain extends. A continuous run of
unseen objects is treated as one new state.

Weighting of the held-out side differs between the quantities:

- ``lofo_novel_fraction`` pools observations, so each FOV counts in
  proportion to its objects: two dense, mutually supporting FOVs of 1,000
  objects and eight one-object FOVs each in its own unsupported state give
  8 / 2,008 = 0.004, although 8 of 10 FOVs are wholly novel. It estimates
  the chance that a random *object* of a new FOV is novel, not a
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
- With a ``transform``, each held-out FOV is judged in a geometry fitted on
  the other FOVs only (a nested holdout), as a genuinely new FOV is by
  ``novel_fovs_against``. Fitting once on all visible FOVs would let an
  extreme FOV widen the standardization that judges it: three FOVs near 0
  and one at 100 put it about 2 SD away instead of about 1,000. The
  diagnostics record the fit on all visible FOVs and the number of folds.
- It only sees states that have appeared: a phenotype absent from every
  visible FOV contributes nothing.
- Zero-observation FOVs add no observations to judge; they still count as
  acquisition effort.

Cost: ``scipy.spatial.cKDTree`` queries. With ``min_other_fovs=1``, counting
ball queries (about 10 ms for a 1,000-object well); otherwise one bounded
nearest-neighbour query of every observation per FOV (about 0.5 s for 6,700
objects in 168 FOVs).
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree, distance

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


# Relative margin around ``epsilon`` inside which a tree's distance is rechecked
# with ``_distances``, so that every counting path applies one rule, ``<= epsilon``.
_TIE = 1e-12


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distances: the one definition every tie is decided by."""
    return np.linalg.norm(a - b, axis=1)


def _ball_counts(tree: cKDTree, data: np.ndarray, queries: np.ndarray, epsilon: float):
    """Rows of ``data`` (indexed by ``tree``) within ``epsilon`` of each query, exactly.

    Counting inside a slightly smaller and a slightly larger radius needs no
    listing; only queries whose two counts differ have a neighbour near
    ``epsilon``, and only those are listed and rechecked with ``_distances``.
    """
    inner = tree.query_ball_point(queries, epsilon * (1 - _TIE), return_length=True)
    outer = tree.query_ball_point(queries, epsilon * (1 + _TIE), return_length=True)
    counts = np.asarray(inner, dtype=np.int64).copy()
    for row in np.flatnonzero(inner != outer):
        hits = np.asarray(tree.query_ball_point(queries[row], epsilon * (1 + _TIE)), dtype=int)
        counts[row] = int(np.sum(_distances(data[hits], queries[row]) <= epsilon))
    return counts


def _other_fov_observations(values: np.ndarray, positions: np.ndarray, epsilon: float):
    """Observations of other FOVs within ``epsilon`` of each row, by counting queries."""
    total = _ball_counts(cKDTree(values), values, values, epsilon)
    same = np.empty(len(values), dtype=np.int64)
    for fov in np.unique(positions):
        rows = np.flatnonzero(positions == fov)
        same[rows] = _ball_counts(cKDTree(values[rows]), values[rows], values[rows], epsilon)
    return total - same


def _check_novel(min_novel_observations) -> None:
    """A positive integer number of novel observations per FOV."""
    value = min_novel_observations
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("min_novel_observations must be a positive integer")


_QUERY_CHUNK = 512
_LIST_NEIGHBOURS_UP_TO = 64


def _supporting_fovs_per_fov(
    reference, reference_positions, queries, epsilon, min_observations, exclude
):
    """``_supporting_fovs`` with one tree per reference FOV (fast in dense neighbourhoods)."""
    supporting = np.zeros(len(queries), dtype=np.int64)
    for fov in np.unique(reference_positions):
        members = reference[reference_positions == fov]
        if len(members) < min_observations:
            continue
        tree = cKDTree(members)
        if min_observations == 1:
            nearest, _ = tree.query(queries, k=1, distance_upper_bound=epsilon * (1 + _TIE))
            near = nearest <= epsilon * (1 - _TIE)
            # A nearest neighbour close to epsilon is decided by the exact count.
            close = ~near & np.isfinite(nearest)
            near[close] = _ball_counts(tree, members, queries[close], epsilon) >= 1
        else:
            near = _ball_counts(tree, members, queries, epsilon) >= min_observations
        if exclude is not None:
            near &= exclude != fov
        supporting += near
    return supporting


# Trees stop pruning in high dimensions: from here on, exact distances to every
# reference observation, counted per FOV with one matrix product, are faster.
_BRUTE_FORCE_DIMENSIONS = 8
_DENSE_CELLS = 4_000_000
# Up to this many query-reference pairs, exact pairwise distances are faster
# than any tree, whatever the dimension.
_DENSE_PAIRS = 20_000_000


# Pair indices are cached only for FOVs up to this many novel observations
# (FOV sizes repeat); larger arrays are built when needed rather than kept.
_CACHED_PAIRS_UP_TO = 2_000


@functools.lru_cache(maxsize=8)
def _cached_upper_pairs(n: int) -> np.ndarray:
    pairs = np.column_stack(np.triu_indices(n, k=1))
    pairs.flags.writeable = False
    return pairs


def _upper_pairs(n: int) -> np.ndarray:
    """All index pairs ``i < j`` of ``n`` rows, in ``pdist`` order."""
    if n <= _CACHED_PAIRS_UP_TO:
        return _cached_upper_pairs(n)
    return np.column_stack(np.triu_indices(n, k=1))


def _supporting_fovs_dense(
    reference, reference_positions, queries, epsilon, min_observations, exclude, n_fovs
):
    """``_supporting_fovs`` by exact pairwise distances, in memory-bounded chunks.

    Reference observations are sorted by FOV, so the observations within
    ``epsilon`` of a query are counted per FOV by summing contiguous columns;
    memory is bounded by the chunk, whatever the number of FOVs.
    """
    order = np.argsort(reference_positions, kind="stable")
    sorted_reference, sorted_positions = reference[order], reference_positions[order]
    starts = np.flatnonzero(np.r_[True, sorted_positions[1:] != sorted_positions[:-1]])
    # Column of each FOV among the FOVs present, or -1.
    column = np.full(n_fovs, -1)
    column[sorted_positions[starts]] = np.arange(len(starts))
    supporting = np.zeros(len(queries), dtype=np.int64)
    chunk = max(1, _DENSE_CELLS // len(reference))
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk]
        within = (distance.cdist(block, sorted_reference) <= epsilon).astype(np.int32)
        counts = np.add.reduceat(within, starts, axis=1)  # per query and FOV
        near = counts >= min_observations
        if exclude is not None:
            own = exclude[start : start + len(block)]
            inside = (own >= 0) & (own < n_fovs)
            rows, columns = np.flatnonzero(inside), column[own[inside]]
            present = columns >= 0
            near[rows[present], columns[present]] = False
        supporting[start : start + len(block)] = near.sum(axis=1)
    return supporting


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

    The fastest exact method is chosen: pairwise distances for small problems
    or eight or more dimensions (trees do not prune there); otherwise one tree
    over all reference observations with its neighbours counted per FOV; or a
    tree per reference FOV when neighbourhoods are dense. Distances are
    ``<= epsilon``; tree searches are widened for rounding and candidates near
    ``epsilon`` are rechecked with one distance definition, so the paths agree
    except, by floating-point rounding, for points at exactly ``epsilon``.
    """
    supporting = np.zeros(len(queries), dtype=np.int64)
    if not len(queries) or not len(reference):
        return supporting
    if not (np.isfinite(reference).all() and np.isfinite(queries).all()):
        # Inputs are finite after validation; only a transform can produce this.
        raise ValueError("support novelty needs finite feature values after the transform")
    n_fovs = int(reference_positions.max()) + 1
    small = len(queries) * len(reference) <= _DENSE_PAIRS
    if small or reference.shape[1] >= _BRUTE_FORCE_DIMENSIONS:
        return _supporting_fovs_dense(
            reference, reference_positions, queries, epsilon, min_observations, exclude, n_fovs
        )
    tree = cKDTree(reference)
    radius = epsilon * (1 + _TIE)
    # Listing neighbours pays off when they are few; in dense neighbourhoods a
    # nearest-neighbour search per reference FOV stops earlier. Counting them
    # first (in C, without listing) chooses.
    sample = queries[:: max(1, len(queries) // 64)]
    if tree.query_ball_point(sample, radius, return_length=True).mean() > _LIST_NEIGHBOURS_UP_TO:
        return _supporting_fovs_per_fov(
            reference, reference_positions, queries, epsilon, min_observations, exclude
        )
    for start in range(0, len(queries), _QUERY_CHUNK):
        block = queries[start : start + _QUERY_CHUNK]
        hits = tree.query_ball_point(block, radius, return_sorted=False)
        lengths = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        total = int(lengths.sum())
        if total == 0:
            continue
        rows = np.repeat(np.arange(len(block)), lengths)
        cols = np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=total)
        close = _distances(reference[cols], block[rows]) <= epsilon
        rows, fovs = rows[close], reference_positions[cols[close]]
        if exclude is not None:
            other = fovs != exclude[start + rows]
            rows, fovs = rows[other], fovs[other]
        pairs, counts = np.unique(rows * n_fovs + fovs, return_counts=True)
        supported = pairs[counts >= min_observations] // n_fovs
        supporting[start : start + len(block)] += np.bincount(supported, minlength=len(block))
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
    # Linked pairs of every FOV, in one graph: pairs never join two FOVs, so its
    # connected components are the clusters of each FOV, and one graph costs far
    # less than one per FOV.
    firsts, seconds = [], []
    for fov in np.unique(positions[novel]):
        rows = np.flatnonzero(novel & (positions == fov))
        if len(rows) < min_size:
            continue
        if values.shape[1] >= _BRUTE_FORCE_DIMENSIONS:
            # Trees do not prune in high dimensions; all pairs, exactly.
            pairs = _upper_pairs(len(rows))
            close = distance.pdist(values[rows]) <= epsilon
        else:
            pairs = cKDTree(values[rows]).query_pairs(epsilon * (1 + _TIE), output_type="ndarray")
            # query_pairs uses <=, with the bound widened for rounding; recheck exactly.
            close = _distances(values[rows[pairs[:, 0]]], values[rows[pairs[:, 1]]]) <= epsilon
        pairs = pairs[close]
        firsts.append(rows[pairs[:, 0]])
        seconds.append(rows[pairs[:, 1]])
        # A FOV's rows are all candidates, linked or not (single rows are clusters of 1).
        members[rows] = True
    candidates = np.flatnonzero(members)
    if not len(candidates):
        return members
    local = np.full(len(values), -1, dtype=np.int64)
    local[candidates] = np.arange(len(candidates))
    first = local[np.concatenate(firsts)] if firsts else np.array([], dtype=np.int64)
    second = local[np.concatenate(seconds)] if seconds else np.array([], dtype=np.int64)
    graph = sparse.coo_matrix(
        (np.ones(len(first)), (first, second)), shape=(len(candidates), len(candidates))
    )
    _, labels = csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    members[candidates] = sizes[labels] >= min_size
    return members


def _meta(name: str) -> dict[str, Any]:
    """Direction, units and component of a novelty quantity."""
    return {
        "direction": Direction.LOWER_IS_BETTER,
        "units": _UNITS[name],
        "component": Component.POPULATION,
    }


def _unavailable(kind: str, reason: str, diagnostics: dict[str, Any]) -> Evidence:
    """Every quantity insufficient or not applicable, with the reason."""
    make = Quantity.insufficient if kind == "insufficient" else Quantity.not_applicable
    quantities = {name: make(reason, **_meta(name)) for name in QUANTITIES}
    return Evidence(quantities, {**diagnostics, "undefined_reason": reason})


@dataclass(frozen=True)
class SupportNoveltyEstimator(Estimator):
    """Leave-one-FOV-out fraction of observations outside the other FOVs' ``epsilon``-support.

    Reports ``lofo_novel_fraction`` (object-weighted),
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
        reason = None
        if self.transform is not None:
            # Fitted on all visible FOVs for provenance and the fit check only:
            # each leave-one-FOV-out fold refits it on the other FOVs.
            fitted = self.transform.fit(data)
            diagnostics["transform"] = fitted.provenance()
            if not fitted.fitted:
                reason = f"the transform could not be fitted: {fitted.reason}"
        if diagnostics["fovs_with_observations"] < self.min_fovs:
            reason = f"fewer than {self.min_fovs} visible FOVs with observations"
        if reason is not None:
            return _unavailable("insufficient", reason, diagnostics)

        lofo = self._lofo(matrix, positions, tuple(data.feature_cols))
        if isinstance(lofo, str):
            return _unavailable("insufficient", lofo, diagnostics)
        novel, members = lofo
        if self.transform is not None:
            diagnostics["transform_folds"] = len(np.unique(positions))
        k = len(data.fovs)
        novel_per_fov = np.bincount(positions, weights=novel, minlength=k)
        counts = np.bincount(positions, minlength=k)
        per_fov = novel_per_fov / np.maximum(counts, 1)
        observed = counts > 0
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
        objects were counted as new; ``None`` when the estimate is unavailable.
        With ``new_states_only``, only the novel observations that form a new
        state (a cluster of at least ``min_novel_observations``) are flagged.
        """
        matrix = data.feature_matrix()
        positions = fov_positions(data, data.fovs)
        if np.isnan(matrix).any() or len(np.unique(positions)) < self.min_fovs:
            return None
        if self.transform is not None and not self.transform.fit(data).fitted:
            return None
        lofo = self._lofo(matrix, positions, tuple(data.feature_cols))
        if isinstance(lofo, str):
            return None
        novel, members = lofo
        return members if new_states_only else novel

    def _lofo(
        self,
        matrix: np.ndarray,
        positions: np.ndarray,
        names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray] | str:
        """Leave-one-FOV-out novel observations and new-state members, or why not.

        Without a transform, the raw features serve every fold. With
        one, each held-out FOV is judged in a geometry fitted on the other FOVs
        only, as a genuinely new FOV would be (``novel_fovs_against``): an extreme
        FOV must not widen the standardization or rotate the PCA that decides
        whether it is extreme.
        """
        m = self.min_novel_observations
        if self.transform is None:
            novel = novel_observations(matrix, positions, self.epsilon, self.min_other_fovs, m)
            return novel, new_state_members(matrix, positions, novel, self.epsilon, m)
        novel = np.zeros(len(matrix), dtype=bool)
        members = np.zeros(len(matrix), dtype=bool)
        for fov in np.unique(positions):
            own = positions == fov
            fitted = self.transform.fit_matrix(matrix[~own], names)
            if isinstance(fitted, str):
                return f"the transform could not be fitted without one FOV: {fitted}"
            state = fitted[0]
            reference = self.transform.apply_matrix(state, matrix[~own])
            queries = self.transform.apply_matrix(state, matrix[own])
            flags = novel_against(
                reference, positions[~own], queries, self.epsilon, self.min_other_fovs, m
            )
            novel[own] = flags
            members[own] = new_state_members(
                queries, np.zeros(len(queries), dtype=np.int64), flags, self.epsilon, m
            )
        return novel, members


@dataclass(frozen=True)
class SupportNoveltyStrategy(SamplingStrategy):
    """Stop when a leave-one-FOV-out novel fraction is at most ``tolerance``.

    ``SupportNoveltyEstimator`` plus ``ThresholdPolicy`` on ``quantity``: the
    object-weighted fraction of novel observations (default), the FOV-weighted
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
