"""Are there distinct populations? Groups of observations that cluster on all the features.

A rare phenotype is a group of cells that sit together when every declared feature is
considered, whichever condition they come from (owner, 2026-10-02): the populations are
found first, on the cells of every condition pooled, and whether a condition is enriched in
one is asked after (``workflows/_populations``, with the paired test across repeats).

``find_populations`` standardises the features robustly (median and MAD, as ``embed``
does), keeps the principal components that hold ``VARIANCE`` of their variance (at most
``MAX_COMPONENTS``), and clusters them with HDBSCAN (scikit-learn): dense groups of any
shape and size, the observations between them left unassigned. A population needs at least
``min_cells`` observations (default ``MIN_SHARE`` of them, and never fewer than
``MIN_CELLS``). Above ``max_points`` observations, HDBSCAN runs on a seeded sample of the
same share of every condition and every other observation takes the population of its
nearest sampled neighbour. Every population is reported, the largest first, with its share
of the observations, the fields and biological repeats it is seen in, and the features
that set it apart from the other observations (Cohen's d). Observations with a missing
value are left out, counted.

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

VARIANCE = 0.9
MAX_COMPONENTS = 10
MIN_SHARE = 0.005
MIN_CELLS = 15
MAX_POINTS = 20000
# The features named as setting a population apart, the largest |d| first.
DEFINING = 3


@dataclass(frozen=True)
class Populations:
    """``labels``: each observation's population (``P1``, ``P2``, ... by size; ``None``
    unassigned or incomplete), indexed like the observation table; ``table``: one row per
    population; ``settings``."""

    labels: pd.Series
    table: pd.DataFrame
    settings: dict[str, Any]

    @property
    def names(self) -> list[str]:
        """The populations, the largest first."""
        return list(self.table["population"])


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

    if len(points) < 2 * min_cells:
        return np.full(len(points), -1)
    return HDBSCAN(min_cluster_size=min_cells, copy=True).fit_predict(points)


def _defining(
    z: np.ndarray, member: np.ndarray, features: Sequence[str]
) -> list[tuple[str, float]]:
    """The features that set ``member`` apart, by Cohen's d against the other observations."""
    inside, outside = z[member], z[~member]
    if len(inside) < 2 or len(outside) < 2:
        return []
    pooled = np.sqrt((inside.var(axis=0, ddof=1) + outside.var(axis=0, ddof=1)) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (inside.mean(axis=0) - outside.mean(axis=0)) / pooled
    order = np.argsort(-np.abs(np.nan_to_num(d)))
    return [(features[i], float(d[i])) for i in order[:DEFINING] if np.isfinite(d[i])]


def defined_by(pairs: Sequence[tuple[str, float]]) -> str:
    """``high dna_intensity (d +4.1), low area (d -2.3)``."""
    return ", ".join(f"{'high' if d > 0 else 'low'} {f} (d {d:+.1f})" for f, d in pairs)


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
    """The populations of ``features`` over every observation (see the module)."""
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
    raw[sample] = _cluster(points[sample], size) if len(kept) else -1
    if len(sample) < n:
        from scipy.spatial import cKDTree

        rest = np.setdiff1d(np.arange(n), sample)
        _, nearest = cKDTree(points[sample]).query(points[rest])
        raw[rest] = raw[sample][nearest]
    # Named by size: P1 the largest.
    found = [c for c in pd.Series(raw[raw >= 0]).value_counts().index]
    names = {c: f"P{i + 1}" for i, c in enumerate(found)}
    labels = pd.Series([None] * len(data.objects), index=data.objects.index, dtype=object)
    labels.iloc[np.flatnonzero(complete)] = [names.get(c) for c in raw]
    rows = []
    # One key per field: the FOV identity (its groups and its identifier).
    field_keys = meta[list(data.fov_key)].astype(str).agg(" · ".join, axis=1).to_numpy()
    field_keys = pd.Series(field_keys)
    for cluster, name in names.items():
        member = raw == cluster
        pairs = _defining(z, member, kept)
        rows.append(
            {
                "population": name,
                "observations": int(member.sum()),
                "share": float(member.mean()),
                "fields": int(field_keys[member].nunique()),
                "repeats": int(meta.loc[member, REPEAT_COLUMN].nunique())
                if REPEAT_COLUMN in meta
                else 1,
                "defined_by": defined_by(pairs),
                "defining": pairs,
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "population",
            "observations",
            "share",
            "fields",
            "repeats",
            "defined_by",
            "defining",
        ],
    )
    settings = {
        "features": kept,
        "components": k,
        "variance": VARIANCE,
        "min_cells": int(size),
        "observations": n,
        "incomplete_observations": int((~complete).sum()),
        "assigned": float((raw >= 0).mean()) if n else 0.0,
        "sampled": len(sample),
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    return Populations(labels, table, settings)


__all__ = ["MIN_CELLS", "MIN_SHARE", "Populations", "defined_by", "find_populations"]
