"""PCA and UMAP of the declared features: how the conditions sit in the space of all of them.

With many features, a plot per feature cannot show whether the conditions separate on a
combination of them. ``embed`` places every observation in two dimensions, two ways:

- **PCA** of the robustly standardised features (each centred on its median and divided
  by its MAD as an SD, or by its SD where the MAD is 0; a feature with one value
  everywhere is left out), with the share of the variance each axis holds and the
  ``loadings``: how much each feature weighs on each axis, the ``top`` features (the
  discovery's best-ranked ones) marked, so the axes can be read in features;
- **UMAP** of the same standardised features (``umap-learn``: ``n_neighbors``,
  ``min_dist``, seeded by ``random_state``), which keeps neighbourhoods rather than
  variance and so can show groups PCA mixes.

Both are descriptions, not tests: a separation in either is a pattern to read with the
discovery's tests, and its absence does not show the conditions alike. Observations with a
missing value in any feature are left out (counted). Above ``max_points`` observations,
UMAP runs on a seeded sample of the same share of every condition (PCA uses every
observation); the settings say how many. Values are used as given: pass the normalised
table (``Setup.prepared``) to embed values relative to each repeat's control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._checks import as_count, check_seed
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN
from .schema import fov_positions, validate_input_tables

MIN_FEATURES = 3
MAX_POINTS = 5000
N_NEIGHBORS = 15
MIN_DIST = 0.1


@dataclass(frozen=True)
class Embedding:
    """``points``: one row per observation kept (its condition and biological repeat when
    the tables have them, its ``field``, ``pc1``, ``pc2``, ``umap1``, ``umap2``; UMAP NaN for
    observations outside its sample); ``loadings``: one row per feature (``pc1``, ``pc2``, ``top``);
    ``explained``: the share of the variance on each PCA axis; ``settings``."""

    points: pd.DataFrame
    loadings: pd.DataFrame
    explained: tuple[float, float]
    settings: dict[str, Any]


def _standardised(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD-scaled columns (SD where the MAD is 0), and which columns vary."""
    centre = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - centre), axis=0)
    sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    scale = np.where(scale > 0, scale, sd)
    varies = scale > 0
    return (matrix[:, varies] - centre[varies]) / scale[varies], varies


def _pca(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """The first two principal components, the loadings and the shares of variance."""
    centred = z - z.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    share = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    # A sign convention, so the same data always gives the same picture: the largest
    # loading of each axis is positive.
    for axis in range(min(2, len(vt))):
        if vt[axis, np.argmax(np.abs(vt[axis]))] < 0:
            vt[axis] = -vt[axis]
    scores = centred @ vt[:2].T
    if scores.shape[1] < 2:
        scores = np.column_stack([scores, np.zeros(len(scores))])
    loadings = vt[:2].T if len(vt) >= 2 else np.column_stack([vt[:1].T, np.zeros(len(vt[0]))])
    shares = (float(share[0]), float(share[1]) if len(share) > 1 else 0.0)
    return scores, loadings, shares


def _sample(conditions: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a sample of the same share of every condition, ``max_points`` in all."""
    if len(conditions) <= max_points:
        return np.arange(len(conditions))
    share = max_points / len(conditions)
    chosen = []
    for value in pd.unique(conditions):
        members = np.flatnonzero(conditions == value)
        take = max(1, round(share * len(members)))
        chosen.append(rng.choice(members, take, replace=False))
    return np.sort(np.concatenate(chosen))


def embed(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    top: Sequence[str] = (),
    carry: Sequence[str] = (),
    umap: bool = True,
    max_points: int = MAX_POINTS,
    n_neighbors: int = N_NEIGHBORS,
    min_dist: float = MIN_DIST,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> Embedding:
    """PCA (and, with ``umap``, UMAP) of ``features`` (see the module); needs at least
    ``MIN_FEATURES`` features that vary. ``carry`` are observation columns copied onto the
    points (a population, a label); the points also hold each feature's standardised value
    (``z:<feature>``), for colouring the maps and describing a selection."""
    features = list(dict.fromkeys(features))
    max_points = as_count(max_points, "max_points", 10)
    n_neighbors = as_count(n_neighbors, "n_neighbors", 2)
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
    positions = fov_positions(data, data.fovs)
    complete = ~np.isnan(matrix).any(axis=1)
    z, varies = _standardised(matrix[complete])
    kept_features = [f for f, v in zip(features, varies, strict=True) if v]
    if len(kept_features) < MIN_FEATURES:
        raise ValueError(
            f"an embedding needs at least {MIN_FEATURES} features that vary; "
            f"{len(kept_features)} of {len(features)} do"
        )
    scores, loadings, explained = _pca(z)
    meta = data.fovs.iloc[positions[complete]].reset_index(drop=True)
    columns = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN) if c in meta]
    points = pd.DataFrame({c: meta[c].to_numpy() for c in columns})
    if data.fov_col in meta and data.fov_col not in columns:
        points["field"] = meta[data.fov_col].to_numpy()
    kept_rows = data.objects.loc[complete]
    for column in carry:
        if column in kept_rows:
            points[column] = kept_rows[column].to_numpy()
    for i, feature in enumerate(kept_features):
        points[f"z:{feature}"] = z[:, i]
    points["pc1"], points["pc2"] = scores[:, 0], scores[:, 1]
    points["umap1"] = points["umap2"] = np.nan
    sampled = 0
    if umap:
        conditions = (
            points[CONDITION_COLUMN].to_numpy()
            if CONDITION_COLUMN in points
            else np.zeros(len(points))
        )
        rng = np.random.default_rng(random_state)
        index = _sample(conditions, max_points, rng)
        sampled = len(index)
        points.loc[index, ["umap1", "umap2"]] = _umap(z[index], n_neighbors, min_dist, random_state)
    table = pd.DataFrame(
        {
            "feature": kept_features,
            "pc1": loadings[:, 0],
            "pc2": loadings[:, 1],
            "top": [f in set(top) for f in kept_features],
        }
    )
    settings = {
        "features": kept_features,
        "constant_features": [f for f, v in zip(features, varies, strict=True) if not v],
        "observations": int(complete.sum()),
        "incomplete_observations": int((~complete).sum()),
        "umap_points": sampled,
        "n_neighbors": n_neighbors,
        "min_dist": float(min_dist),
        "random_state": random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
    }
    return Embedding(points, table, explained, settings)


def _umap(z: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    """UMAP coordinates of ``z`` (imported here: loading umap-learn takes seconds)."""
    import warnings

    import umap

    neighbours = min(n_neighbors, max(2, len(z) - 1))
    reducer = umap.UMAP(
        n_neighbors=neighbours, min_dist=min_dist, random_state=random_state, n_jobs=1
    )
    with warnings.catch_warnings():
        # umap-learn warns that a seed disables its parallelism: that is the point here.
        warnings.simplefilter("ignore", UserWarning)
        return reducer.fit_transform(z)


__all__ = ["MIN_FEATURES", "Embedding", "embed"]
