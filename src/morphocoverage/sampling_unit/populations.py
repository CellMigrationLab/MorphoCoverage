"""Are there distinct populations? Groups of observations that cluster on all the features.

A rare phenotype is a group of cells that sit together when every declared feature is
considered, whichever condition they come from (owner, 2026-10-02): the populations are
found first, on the cells of every condition pooled, and whether a condition is enriched in
one is asked after (``workflows/_populations``, with the paired test across repeats).

``find_populations`` standardises the features robustly (median and MAD, as ``embed``
does), embeds them with UMAP in ``MAP_COMPONENTS`` dimensions (``min_dist`` 0, as
umap-learn advises for clustering) and clusters the embedding with HDBSCAN (scikit-learn):
dense groups of any shape, the observations between them left unassigned. A cluster needs
at least ``min_cells`` observations (default ``MIN_SHARE`` of them, and never fewer than
``MIN_CELLS``). Above ``max_points`` observations, both run on a seeded sample of the same
share of every condition, and every other observation takes the cluster of its nearest
sampled neighbour (on the principal components holding ``VARIANCE`` of the variance).

UMAP evens out density, so a cluster on it is a candidate only, judged on all the
standardised features (owner, 2026-10-02, after HDBSCAN on the principal components split
a Gaussian and a straight line into populations and missed two separate blobs):

- the **main** population, when one cluster holds more than ``MAIN_SHARE`` of the
  observations: the rest of the cells, listed when another kind is found;
- a **distinct** population (``P1``, ``P2``, ..., numbered after the main population when
  there is one, which is then ``P1``): the density along
  the line from the rest of the observations to the cluster dips between them to at most
  ``VALLEY`` of the lower of the two peaks (a gap of low density);
- a **group on the map** (``G1``, ``G2``, ...): no such gap, but ``PURITY`` or more of its
  observations' ``NEIGHBOURS`` nearest neighbours on all the features are in it too; it
  may be the end of a continuum (the most elongated cells of a gradient, say);
- otherwise the cluster is a stretch of one continuum and is not reported.

Each is reported with its share of the observations, the fields and biological repeats it
is seen in, the features that set it apart from the other observations (Cohen's d), its
``valley`` and its ``purity``. Observations with a missing value are left out, counted.

A description, not a test: the clusters depend on the features declared and on the
standardisation, and a population seen in one field only may be one image's artefact (its
fields are listed so they can be opened).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count, check_seed
from ..embedding import _sample, _standardised
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..schema import fov_positions, validate_input_tables
from ..statistics import cohen_d

VARIANCE = 0.9
MAX_COMPONENTS = 10
MIN_SHARE = 0.005
MIN_CELLS = 15
MAX_POINTS = 20000
# The features named as setting a population apart, the largest |d| first.
DEFINING = 3
# The embedding clustered: this many UMAP dimensions.
MAP_COMPONENTS = 5
# A cluster holding more than this share is the main population, the rest of the cells.
MAIN_SHARE = 0.5
# Distinct: the density between a cluster and the rest dips to this share of the lower peak.
VALLEY = 0.5
# A group on the map: this share of its observations' nearest neighbours is inside it.
NEIGHBOURS = 10
PURITY = 0.94
KINDS = ("main", "distinct", "group")
_COLUMNS = [
    "population",
    "kind",
    "observations",
    "share",
    "fields",
    "repeats",
    "defined_by",
    "defining",
    "valley",
    "purity",
]


@dataclass(frozen=True)
class Populations:
    """``labels``: each observation's population or group (``P1``, ``P2``, ...; ``G1``,
    ``G2``, ...; ``None`` unassigned or incomplete), indexed like the observation table;
    ``table``: one row per population and group, with its ``kind``; ``settings``."""

    labels: pd.Series
    table: pd.DataFrame
    settings: dict[str, Any]

    @property
    def names(self) -> list[str]:
        """The populations, then the groups on the map, each the largest first."""
        return list(self.table["population"])

    def of_kind(self, *kinds: str) -> list[str]:
        """The names of the populations of these ``kinds`` (``KINDS``)."""
        return list(self.table.loc[self.table["kind"].isin(kinds), "population"])


def _components(z: np.ndarray) -> tuple[np.ndarray, int]:
    """The principal components holding ``VARIANCE`` of the variance (at most
    ``MAX_COMPONENTS``), and how many."""
    centred = z - z.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    if variance.sum() == 0:
        return centred[:, :1], 1
    share = np.cumsum(variance) / variance.sum()
    k = int(min(MAX_COMPONENTS, np.searchsorted(share, VARIANCE) + 1, len(vt)))
    return centred @ vt[:k].T, k


def _cluster(points: np.ndarray, min_cells: int) -> np.ndarray:
    """HDBSCAN labels (-1 unassigned); imported here, as umap-learn brings scikit-learn."""
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=min_cells, copy=True).fit_predict(points)


def _embedding(z: np.ndarray, random_state: int) -> np.ndarray:
    """The UMAP clustered (imported here: loading umap-learn takes seconds)."""
    import warnings

    import umap

    reducer = umap.UMAP(
        n_neighbors=min(15, max(2, len(z) - 1)),
        min_dist=0.0,
        n_components=min(MAP_COMPONENTS, max(2, z.shape[1])),
        random_state=random_state,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        # umap-learn warns that a seed disables its parallelism: that is the point here.
        warnings.simplefilter("ignore", UserWarning)
        return reducer.fit_transform(z)


def _valley(z: np.ndarray, member: np.ndarray) -> float:
    """The lowest density between the rest of the observations and ``member``, along the
    line joining their means, over the lower of their two peaks (0: a gap; 1: none)."""
    from scipy.stats import gaussian_kde

    direction = z[member].mean(axis=0) - z[~member].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if not norm or member.sum() < 2 or (~member).sum() < 2:
        return 1.0
    scores = z @ (direction / norm)
    if np.ptp(scores) == 0:
        return 1.0
    density = gaussian_kde(scores)
    grid = np.linspace(scores.min(), scores.max(), 512)
    values = density(grid)

    def peak(inside: np.ndarray) -> float:
        low, high = np.quantile(scores[inside], [0.02, 0.98])
        window = (grid >= low) & (grid <= high)
        return float(grid[window][np.argmax(values[window])]) if window.any() else np.nan

    own, rest = peak(member), peak(~member)
    if not np.isfinite(own) or not np.isfinite(rest):
        return 1.0
    low, high = sorted((own, rest))
    between = values[(grid >= low) & (grid <= high)]
    lower_peak = float(min(density(own)[0], density(rest)[0]))
    return float(between.min() / lower_peak) if lower_peak > 0 and len(between) else 1.0


def _purity(z: np.ndarray, member: np.ndarray) -> float:
    """The share of the ``NEIGHBOURS`` nearest neighbours of ``member``'s observations, on
    all the standardised features, that are in ``member`` too."""
    from scipy.spatial import cKDTree

    k = min(NEIGHBOURS, len(z) - 1)
    if k < 1 or not member.any():
        return 0.0
    rows = np.flatnonzero(member)
    _, nearest = cKDTree(z).query(z[rows], k=k + 1)
    # Each observation's own row out, wherever the tree put it (an exact duplicate can come
    # first); without it among the k + 1, the farthest.
    own = nearest == rows[:, None]
    drop = np.where(own.any(axis=1), own.argmax(axis=1), k)
    keep = np.ones_like(nearest, dtype=bool)
    keep[np.arange(len(rows)), drop] = False
    return float(member[nearest[keep].reshape(len(rows), k)].mean())


def _defining(
    z: np.ndarray, member: np.ndarray, features: Sequence[str]
) -> list[tuple[str, float]]:
    """The features that set ``member`` apart, by Cohen's d against the other observations."""
    inside, outside = z[member], z[~member]
    if len(inside) < 2 or len(outside) < 2:
        return []
    d = np.array([cohen_d(inside[:, j], outside[:, j]) for j in range(z.shape[1])])
    order = np.argsort(-np.abs(np.nan_to_num(d)))
    return [(features[i], float(d[i])) for i in order[:DEFINING] if np.isfinite(d[i])]


def defined_by(pairs: Sequence[tuple[str, float]]) -> str:
    """``high dna_intensity (d +4.1), low area (d -2.3)``."""
    return ", ".join(f"{traits([(f, d)])} (d {d:+.1f})" for f, d in pairs)


def traits(pairs: Sequence[tuple[str, float]]) -> str:
    """``high dna_intensity, low area``: the defining features without their d."""
    return ", ".join(f"{'high' if d > 0 else 'low'} {f}" for f, d in pairs)


def _kinds(raw: np.ndarray, z: np.ndarray) -> dict[int, tuple[str | None, float, float]]:
    """Each cluster's kind (``KINDS``, or ``None``: a stretch of a continuum), valley and
    purity (see the module)."""
    sizes = pd.Series(raw[raw >= 0]).value_counts()
    judged: dict[int, tuple[str | None, float, float]] = {}
    for cluster, size in sizes.items():
        member = raw == cluster
        if size / len(raw) > MAIN_SHARE:
            judged[int(cluster)] = ("main", np.nan, np.nan)
            continue
        valley, purity = _valley(z, member), _purity(z, member)
        # Both need their observations' neighbours inside them; the valley says which.
        kind = None if purity < PURITY else "distinct" if valley <= VALLEY else "group"
        judged[int(cluster)] = (kind, valley, purity)
    return judged


def _names(raw: np.ndarray, judged: dict) -> dict[int, str]:
    """``P1``, ``P2``, ... for the main and distinct populations, ``G1``, ... for the groups,
    the largest first; none when nothing but a main population and continua is found."""
    found = [c for c, (kind, _, _) in judged.items() if kind is not None]
    if not any(judged[c][0] in ("distinct", "group") for c in found):
        return {}
    order = {"main": 0, "distinct": 1, "group": 2}
    found.sort(key=lambda c: (order[judged[c][0]], -int((raw == c).sum())))
    names, counts = {}, {"P": 0, "G": 0}
    for cluster in found:
        prefix = "G" if judged[cluster][0] == "group" else "P"
        counts[prefix] += 1
        names[cluster] = f"{prefix}{counts[prefix]}"
    return names


def find_populations(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    min_cells: int | None = None,
    max_points: int = MAX_POINTS,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> Populations:
    """The populations and the groups on the map of ``features`` over every observation
    (see the module)."""
    features = list(dict.fromkeys(features))
    max_points = as_count(max_points, "max_points", 100)
    random_state = check_seed(random_state)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=features,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    matrix = data.feature_matrix()
    complete = ~np.isnan(matrix).any(axis=1)
    z, varies = _standardised(matrix[complete])
    kept = [f for f, v in zip(features, varies, strict=True) if v]
    n = int(complete.sum())
    size = as_count(min_cells, "min_cells", 2) if min_cells is not None else None
    size = size or max(MIN_CELLS, round(MIN_SHARE * n))
    points, k = _components(z) if len(kept) else (np.zeros((n, 1)), 0)
    meta = data.fovs.iloc[fov_positions(data, data.fovs)[complete]].reset_index(drop=True)
    conditions = meta[CONDITION_COLUMN].to_numpy() if CONDITION_COLUMN in meta else np.zeros(n)
    sample = _sample(conditions, max_points, np.random.default_rng(random_state))
    raw = np.full(n, -1)
    # Too few observations (or no feature that varies) for two clusters of ``size``: not
    # clustered, which is not a continuum.
    clustered = bool(len(kept)) and len(sample) >= 2 * size + 2
    if clustered:
        raw[sample] = _cluster(_embedding(z[sample], random_state), size)
    if len(sample) < n:
        from scipy.spatial import cKDTree

        rest = np.setdiff1d(np.arange(n), sample)
        _, nearest = cKDTree(points[sample]).query(points[rest])
        raw[rest] = raw[sample][nearest]
    judged = _kinds(raw, z)
    names = _names(raw, judged)
    labels = pd.Series([None] * len(data.objects), index=data.objects.index, dtype=object)
    labels.iloc[np.flatnonzero(complete)] = [names.get(c) for c in raw]
    # One key per field: the FOV identity (its groups and its identifier).
    keys = meta[list(data.fov_key)].astype(str).agg(" · ".join, axis=1).to_numpy()
    field_keys = pd.Series(keys)
    rows = []
    for cluster, name in names.items():
        member = raw == cluster
        kind, valley, purity = judged[cluster]
        pairs = _defining(z, member, kept)
        rows.append(
            {
                "population": name,
                "kind": kind,
                "observations": int(member.sum()),
                "share": float(member.mean()),
                "fields": int(field_keys[member].nunique()),
                "repeats": int(meta.loc[member, REPEAT_COLUMN].nunique())
                if REPEAT_COLUMN in meta
                else 1,
                "defined_by": defined_by(pairs),
                "defining": pairs,
                "valley": valley,
                "purity": purity,
            }
        )
    table = pd.DataFrame(rows, columns=_COLUMNS)
    settings = {
        "features": kept,
        "components": k,
        "variance": VARIANCE,
        "map_components": min(MAP_COMPONENTS, max(2, len(kept))),
        "min_cells": int(size),
        "valley": VALLEY,
        "purity": PURITY,
        "neighbours": NEIGHBOURS,
        "observations": n,
        "incomplete_observations": int((~complete).sum()),
        "assigned": float(np.isin(raw, list(names)).mean()) if n else 0.0,
        "sampled": len(sample),
        "clustered": clustered,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    return Populations(labels, table, settings)


__all__ = [
    "KINDS",
    "MIN_CELLS",
    "MIN_SHARE",
    "PURITY",
    "VALLEY",
    "Populations",
    "defined_by",
    "find_populations",
]
