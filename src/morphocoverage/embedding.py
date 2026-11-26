"""PCA and UMAP of the declared features: how the conditions sit in the space of all of them.

With many features, a plot per feature cannot show whether the conditions separate on a
combination of them. ``embed`` places every observation in two dimensions, two ways:

- **PCA** of the robustly standardised features (each centred on its median and divided
  by its MAD as an SD, or by its SD where the MAD is under a quarter of the SD; a feature
  constant up to rounding is left out), with the share of the variance each axis
  holds and the ``loadings``: how much each feature weighs on each axis, the ``top`` features (the
  discovery's best-ranked ones) marked, so the axes can be read in features;
- **UMAP** of the same standardised features (``umap-learn``: ``n_neighbors``,
  ``min_dist``, seeded by ``random_state``), which keeps neighbourhoods rather than
  variance and so can show groups PCA mixes.

With ``blocks`` (each feature table's columns, the explainable features first), each
standardised table is weighted by its first singular value before both: the weighted
combined space of several feature tables (``feature_tables.mfa_weights``, owner 2026-10-06),
and the settings give each table's share of each axis (``table_shares``). ``weights`` (a
fixed weight per feature, the others 1) multiply the standardised features as given.

The floor on the MAD (owner, 2026-10-06): a column with just under half its values at one
value, as an embedding's ReLU outputs, has a MAD far below its spread, and divided by it
alone would carry thousands of times the variance of the others. Below a quarter of the SD
(``statistics.robust_sd``'s geometry rule), the SD scales it, so no column carries more than
16 times the variance of another's middle; a column whose MAD is at least a quarter of its
SD, as in normal, skewed or moderately heavy-tailed data, is scaled by its MAD as before. A
column constant up to rounding (single-precision noise around one value) is left out, as
the novelty transform refuses it (audit 12).

Both are descriptions, not tests: a separation in either is a pattern to read with the
discovery's tests, and its absence does not show the conditions alike. Observations with a
missing value in any feature are left out (counted). Above ``max_points`` observations,
UMAP runs on a seeded sample of the same share of every condition (PCA uses every
observation); the settings say how many. Values are used as given: pass the normalised
table (``Setup.prepared``) to embed values relative to each repeat's control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._checks import as_count, check_seed
from ._frozen import owned
from .feature_tables import mfa_weights, table_shares
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN
from .schema import ValidatedInput, fov_positions, validate_input_tables
from .statistics import MAD_TO_SD_ROUNDED, robust_sd

MIN_FEATURES = 3
MAX_POINTS = 5000
N_NEIGHBORS = 15
MIN_DIST = 0.1
# umap-learn 0.6's ``compatibility_layout`` (owner, 2026-11-26, #644): True is the 0.5 layout
# and today's default; it is passed explicitly so no installed version decides it, and the
# embeddings and populations are as before. ``False`` (the layout umap-learn plans as its
# default) split one Gaussian cloud into 41 HDBSCAN clusters and named no main population, so
# it waits for the populations to be retuned with it.
COMPATIBILITY_LAYOUT = True
# The MorphoBricks column naming an object within its image (CellPatcher's patches).
LABEL_COLUMN = "label_id"


@owned("points", "loadings", "settings")
@dataclass(frozen=True)
class Embedding:
    """``points``: one row per observation kept (its condition and biological repeat when
    the tables have them, its ``field``, its ``observation``, which is its index in the
    observation table, its ``label_id`` when the table has one, ``pc1``, ``pc2``, ``umap1``
    and ``umap2``; UMAP NaN for observations outside its sample); ``loadings``: one row per
    feature (``pc1``, ``pc2``, ``top``); ``explained``: the share of the variance on each PCA
    axis; ``settings``."""

    points: pd.DataFrame
    loadings: pd.DataFrame
    explained: tuple[float, float]
    settings: dict[str, Any]


def robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column's median and robust scale, ``statistics.robust_sd``'s geometry rule: its
    MAD as an SD, or its SD where the MAD is under a quarter of it (see the module); 0 for a
    column constant up to rounding. The one robust standardisation of the maps, the
    fingerprint of a wide table and the whole-table test (owner, 2026-10-06)."""
    scale = robust_sd(matrix, "geometry", MAD_TO_SD_ROUNDED)
    return np.median(matrix, axis=0), scale


def robust_components(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, scale and the first ``k`` loadings of the robustly standardised ``matrix``
    (complete rows), the centre folding in the standardised columns' means; a column with
    one value gets scale 1 and no weight. The fit of a wide table's fingerprint
    (``multivariate_outcomes.ComponentsError``); ``ValueError`` when ``matrix`` cannot give
    ``k`` components."""
    centre, scale = robust_scale(matrix)
    varies = scale > 0
    scale = np.where(varies, scale, 1.0)
    z = (matrix - centre) / scale
    means = z.mean(axis=0)
    _, _, vt = np.linalg.svd((z - means)[:, varies], full_matrices=False)
    if len(vt) < k:
        raise ValueError(f"{len(matrix)} rows and {int(varies.sum())} varying columns")
    loadings = np.zeros((k, matrix.shape[1]))
    loadings[:, varies] = vt[:k]
    # One sign for one data set: the largest loading of each component is positive.
    for row in loadings:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return centre + means * scale, scale, loadings


def _standardised(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD-scaled columns (``robust_scale``), and which columns vary."""
    centre, scale = robust_scale(matrix)
    varies = scale > 0
    return (matrix[:, varies] - centre[varies]) / scale[varies], varies


def combined(
    z: np.ndarray,
    features: Sequence[str],
    weights: Mapping[str, float] | None = None,
    blocks: Mapping[str, Sequence[str]] | None = None,
) -> np.ndarray:
    """``z`` with each table weighted as the combined space weighs it (``blocks``,
    ``feature_tables.mfa_weights``), then each column times its fixed weight (``weights``);
    ``z`` itself without either."""
    if blocks:
        z = z * mfa_weights(z, features, blocks)
    return weighted(z, features, weights)


def weighted(
    z: np.ndarray, features: Sequence[str], weights: Mapping[str, float] | None
) -> np.ndarray:
    """``z`` with each column times its feature's weight (1 when not given); ``z`` itself
    without ``weights``."""
    if not weights:
        return z
    return z * np.array([float(weights.get(f, 1.0)) for f in features])


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
    weights: Mapping[str, float] | None = None,
    standardised: Sequence[str] | None = None,
    blocks: Mapping[str, Sequence[str]] | None = None,
) -> Embedding:
    """PCA (and, with ``umap``, UMAP) of ``features`` (see the module); needs at least
    ``MIN_FEATURES`` features that vary. ``carry`` are observation columns copied onto the
    points (a population, a label); the points also hold each feature's standardised value
    (``z:<feature>``), for colouring the maps and describing a selection (of the features in
    ``standardised`` only, when given); ``weights`` and ``blocks`` weigh the standardised
    features (see the module)."""
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
    z = combined(z, kept_features, weights, blocks)
    if len(kept_features) < MIN_FEATURES:
        raise ValueError(
            f"an embedding needs at least {MIN_FEATURES} features that vary; "
            f"{len(kept_features)} of {len(features)} do"
        )
    scores, loadings, explained = _pca(z)
    points = _points(data, positions, complete, kept_features, z, scores, carry, standardised)
    sampled = 0
    if umap:
        sampled = _add_umap(points, z, max_points, n_neighbors, min_dist, random_state)
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
        "compatibility_layout": COMPATIBILITY_LAYOUT,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "weights": dict(weights) if weights else None,
        # The combined space: each table's share of PC1 and PC2 (owner, 2026-10-06).
        "table_shares": table_shares(loadings, kept_features, blocks) if blocks else None,
        "table_columns": (
            {n: sum(c in set(kept_features) for c in cols) for n, cols in blocks.items()}
            if blocks
            else None
        ),
    }
    return Embedding(points, table, explained, settings)


def _points(
    data: ValidatedInput,
    positions: np.ndarray,
    complete: np.ndarray,
    kept_features: list[str],
    z: np.ndarray,
    scores: np.ndarray,
    carry: Sequence[str],
    standardised: Sequence[str] | None,
) -> pd.DataFrame:
    """One row per complete observation: its condition, repeat and field, where it came
    from, the ``carry`` columns, the standardised features, the PCA scores and UMAP columns
    still empty."""
    meta = data.fovs.iloc[positions[complete]].reset_index(drop=True)
    columns = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN) if c in meta]
    # Every column at once: inserted one by one, hundreds of features fragment the frame.
    built: dict[str, Any] = {c: meta[c].to_numpy() for c in columns}
    if data.fov_col in meta and data.fov_col not in columns:
        built["field"] = meta[data.fov_col].to_numpy()
    kept_rows = data.objects.loc[complete]
    # Where each point came from, to find the observation again (#74).
    built["observation"] = kept_rows.index.to_numpy()
    if LABEL_COLUMN in kept_rows:
        built[LABEL_COLUMN] = kept_rows[LABEL_COLUMN].to_numpy()
    for column in carry:
        if column in kept_rows:
            built[column] = kept_rows[column].to_numpy()
    carried = set(kept_features if standardised is None else standardised)
    for i, feature in enumerate(kept_features):
        if feature in carried:
            built[f"z:{feature}"] = z[:, i]
    built["pc1"], built["pc2"] = scores[:, 0], scores[:, 1]
    built["umap1"], built["umap2"] = np.full(len(z), np.nan), np.full(len(z), np.nan)
    return pd.DataFrame(built, index=pd.RangeIndex(len(z)))


def _add_umap(
    points: pd.DataFrame,
    z: np.ndarray,
    max_points: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> int:
    """Fill the UMAP columns of a sample of ``points`` of the same share of every
    condition; the number of points sampled."""
    conditions = (
        points[CONDITION_COLUMN].to_numpy() if CONDITION_COLUMN in points else np.zeros(len(points))
    )
    rng = np.random.default_rng(random_state)
    index = _sample(conditions, max_points, rng)
    points.loc[index, ["umap1", "umap2"]] = _umap(z[index], n_neighbors, min_dist, random_state)
    return len(index)


def _umap(
    z: np.ndarray, n_neighbors: int, min_dist: float, random_state: int, n_components: int = 2
) -> np.ndarray:
    """UMAP coordinates of ``z`` in ``n_components`` dimensions, on one thread for the seed
    to hold (imported here: loading umap-learn takes seconds)."""
    import warnings

    import umap

    neighbours = min(n_neighbors, max(2, len(z) - 1))
    reducer = umap.UMAP(
        n_neighbors=neighbours,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        n_jobs=1,
        compatibility_layout=COMPATIBILITY_LAYOUT,
    )
    with warnings.catch_warnings():
        # fallback: umap-learn warns that a seed disables its parallelism: the point here
        warnings.simplefilter("ignore", UserWarning)
        return reducer.fit_transform(z)


__all__ = ["MIN_FEATURES", "Embedding", "embed", "robust_components", "robust_scale"]
