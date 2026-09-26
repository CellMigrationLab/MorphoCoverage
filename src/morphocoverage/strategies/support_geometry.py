"""The geometry of support novelty: which observations other FOVs support at resolution epsilon.

``support_novelty`` decides and reports; this module only counts. Every
count applies one rule, distance ``<= epsilon`` in the transformed feature
space, whether it comes from a ``cKDTree`` query or from a dense distance
matrix, and ties at the boundary are rechecked with the same Euclidean
distance. The public functions are ``novel_observations`` (leave-one-FOV-out,
within one acquisition), ``novel_against`` (new observations against a
reference acquisition) and ``new_state_members`` (the novel observations that
form clusters of at least a declared size, linked within epsilon).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree, distance

# Relative margin around ``epsilon`` inside which a tree's distance is rechecked
# with ``_distances``, so that every counting path applies one rule, ``<= epsilon``.
_TIE = 1e-12


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distances: the one definition every tie is decided by."""
    return np.linalg.norm(a - b, axis=1)


def _ball_counts(
    tree: cKDTree, data: np.ndarray, queries: np.ndarray, epsilon: float
) -> np.ndarray:
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


def _other_fov_observations(
    values: np.ndarray, positions: np.ndarray, epsilon: float
) -> np.ndarray:
    """Observations of other FOVs within ``epsilon`` of each row, by counting queries."""
    total = _ball_counts(cKDTree(values), values, values, epsilon)
    same = np.empty(len(values), dtype=np.int64)
    for fov in np.unique(positions):
        rows = np.flatnonzero(positions == fov)
        same[rows] = _ball_counts(cKDTree(values[rows]), values[rows], values[rows], epsilon)
    return total - same


_QUERY_CHUNK = 512


_LIST_NEIGHBOURS_UP_TO = 64


def _supporting_fovs_per_fov(
    reference: np.ndarray,
    reference_positions: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
    min_observations: int,
    exclude: np.ndarray | None,
) -> np.ndarray:
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
    reference: np.ndarray,
    reference_positions: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
    min_observations: int,
    exclude: np.ndarray | None,
    n_fovs: int,
) -> np.ndarray:
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


__all__ = ["new_state_members", "novel_against", "novel_observations"]
