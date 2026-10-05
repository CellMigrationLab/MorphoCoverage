"""Are there populations or groups of cells? Groups of observations that cluster on all the
features.

A rare phenotype is a group of cells that sit together when every declared feature is
considered, whichever condition they come from (owner, 2026-10-02): the groups are found
first, on the cells of every condition pooled, and whether a condition is enriched in one is
asked after (``workflows/_populations``, with the paired test across repeats).

``find_populations`` standardises the features robustly (median and MAD, as ``embed``
does), embeds them with UMAP in ``MAP_COMPONENTS`` dimensions (``min_dist`` 0, as
umap-learn advises for clustering) and clusters the embedding with HDBSCAN (scikit-learn):
dense groups of any shape, the observations between them left unassigned. A cluster needs
at least ``min_cells`` of the observations clustered (default ``MIN_SHARE`` of them, and
never fewer than ``MIN_CELLS``). Above ``max_points`` observations, both run on a seeded
sample of the same share of every condition, and every other observation takes the cluster
of its nearest sampled neighbour (on the principal components holding ``VARIANCE`` of the
variance).

UMAP evens out density, so a cluster on it is a candidate, judged on all the standardised
features:

- the **main** population (``P1``), when one cluster holds more than ``MAIN_SHARE`` of the
  observations: the rest of the cells, listed when a group is found;
- a **group** (``G1``, ``G2``, ...): ``PURITY`` or more of its observations' ``NEIGHBOURS``
  nearest neighbours on all the features are in it too (HDBSCAN on a Gaussian or a line
  cuts clusters that fail this);
- otherwise the cluster is not a group and is not reported; finding no group says
  nothing of a continuum or of rare states (owner, 2026-10-05).

Whether a group is a separate state or the end of a continuum is not decided here: no test
of a gap of low density told the two apart reliably at the sizes of an experiment (owner,
2026-10-02, after one along the line between the centres and one along a held-out
discriminant both missed groups a UMAP shows apart). It is for the user to validate on the
images; each group comes with the simplest rule on the features that picks out its cells
(``_population_rule``: at most two thresholds fitted on every cell, and how well the procedure
that fits them does refitted with whole fields held out), to label it and study it as a rare
phenotype.

Each is reported with its share of the observations, the fields and biological repeats it
is seen in, the share of its observations in its largest field (``largest_field``: a group
living mostly in one field may be that image's artefact) and, with a ``technical_repeat``
column, the technical repeats it is seen in (owner, 2026-10-05), the features that set it
apart from the other observations (Cohen's d), its ``purity`` and its rule. Observations
with a missing value are left out, counted.

A description, not a test: the clusters depend on the features declared and on the
standardisation, and a group seen in one field only may be one image's artefact (its
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
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..schema import fov_positions, validate_input_tables
from ..statistics import cohen_d
from ._population_rule import Rule, population_rule

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
# A group: this share of its observations' nearest neighbours is inside it.
NEIGHBOURS = 10
PURITY = 0.94
KINDS = ("main", "group")
_COLUMNS = [
    "population",
    "kind",
    "observations",
    "share",
    "fields",
    "repeats",
    "largest_field",
    "technical_repeats",
    "defined_by",
    "defining",
    "purity",
    "rule",
    "rule_code",
    "rule_terms",
    "rule_recall",
    "rule_precision",
    "rule_held_out",
]


@dataclass(frozen=True)
class Populations:
    """``labels``: each observation's population or group (``P1`` the main one; ``G1``,
    ``G2``, ...; ``None`` unassigned or incomplete), indexed like the observation table;
    ``table``: one row per population and group, with its ``kind``; ``settings``."""

    labels: pd.Series
    table: pd.DataFrame
    settings: dict[str, Any]

    @property
    def names(self) -> list[str]:
        """The main population, then the groups, the largest first."""
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


def _kinds(raw: np.ndarray, z: np.ndarray) -> dict[int, tuple[str | None, float]]:
    """Each cluster's kind (``KINDS``, or ``None``: not a group) and purity
    (see the module)."""
    sizes = pd.Series(raw[raw >= 0]).value_counts()
    judged: dict[int, tuple[str | None, float]] = {}
    for cluster, size in sizes.items():
        member = raw == cluster
        if size / len(raw) > MAIN_SHARE:
            judged[int(cluster)] = ("main", np.nan)
            continue
        purity = _purity(z, member)
        judged[int(cluster)] = ("group" if purity >= PURITY else None, purity)
    return judged


def _names(raw: np.ndarray, judged: dict) -> dict[int, str]:
    """``P1`` for the main population, ``G1``, ``G2``, ... for the groups, the largest first;
    none when nothing but a main population and continua is found."""
    found = [c for c, (kind, _) in judged.items() if kind is not None]
    if not any(judged[c][0] == "group" for c in found):
        return {}
    found.sort(key=lambda c: (judged[c][0] != "main", -int((raw == c).sum())))
    names, groups = {}, 0
    for cluster in found:
        if judged[cluster][0] == "main":
            names[cluster] = "P1"
        else:
            groups += 1
            names[cluster] = f"G{groups}"
    return names


def _support(member: np.ndarray, field_keys: pd.Series, meta: pd.DataFrame) -> dict[str, Any]:
    """The share of ``member``'s observations in its largest field, and the technical
    repeats holding it (``None`` without a ``technical_repeat`` column)."""
    largest = float(field_keys[member].value_counts().iloc[0] / member.sum())
    if TECHNICAL_REPEAT_COLUMN not in meta:
        return {"largest_field": largest, "technical_repeats": None}
    columns = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN) if c in meta]
    wells = meta.loc[member, columns].astype(str).drop_duplicates()
    return {"largest_field": largest, "technical_repeats": len(wells)}


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
    """The main population and the groups of ``features`` over every observation (see the
    module)."""
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
    values = matrix[complete][:, np.flatnonzero(varies)]
    n = int(complete.sum())
    size = as_count(min_cells, "min_cells", 2) if min_cells is not None else None
    points, k = _components(z) if len(kept) else (np.zeros((n, 1)), 0)
    meta = data.fovs.iloc[fov_positions(data, data.fovs)[complete]].reset_index(drop=True)
    conditions = meta[CONDITION_COLUMN].to_numpy() if CONDITION_COLUMN in meta else np.zeros(n)
    sample = _sample(conditions, max_points, np.random.default_rng(random_state))
    # A share of the observations clustered, not of all of them: above ``max_points`` a
    # share of the whole table would grow past what the sample can hold.
    size = size or max(MIN_CELLS, round(MIN_SHARE * len(sample)))
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
        kind, purity = judged[cluster]
        pairs = _defining(z, member, kept)
        # The rule on the features in their own units, for a group (the main one is the rest).
        rule = (
            population_rule(values, member, kept, random_state, groups=keys)
            if kind == "group"
            else Rule((), np.nan, np.nan)
        )
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
                **_support(member, field_keys, meta),
                "defined_by": defined_by(pairs),
                "defining": pairs,
                "purity": purity,
                "rule": rule.words(),
                "rule_code": rule.code(),
                "rule_terms": rule.terms,
                "rule_recall": rule.recall,
                "rule_precision": rule.precision,
                "rule_held_out": rule.held_out,
            }
        )
    table = pd.DataFrame(rows, columns=_COLUMNS)
    settings = {
        "features": kept,
        "components": k,
        "variance": VARIANCE,
        "map_components": min(MAP_COMPONENTS, max(2, len(kept))),
        "min_cells": int(size),
        "purity": PURITY,
        "neighbours": NEIGHBOURS,
        "observations": n,
        "incomplete_observations": int((~complete).sum()),
        "assigned": float(np.isin(raw, list(names)).mean()) if n else 0.0,
        "sampled": len(sample),
        "clustered": clustered,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
    }
    return Populations(labels, table, settings)


__all__ = [
    "KINDS",
    "MIN_CELLS",
    "MIN_SHARE",
    "PURITY",
    "Populations",
    "defined_by",
    "find_populations",
]
