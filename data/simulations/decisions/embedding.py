"""Embeddings on a ``design.Layout``: the outcome generator of module P3.

Each cell is a ``dims``-dimensional vector, ``latent @ diag(scales) @ basis``: the latent
part has, in each of ``rank`` dimensions, a repeat, well, field and cell term with variance
shares ``icc_repeat``, ``icc_well``, ``icc_field`` and the rest; ``scales`` fall as
``j^-decay`` (an embedding's spectrum) and ``basis`` is a fixed random orthogonal matrix
(``basis_seed``), so no coordinate is special. The within-group SD (the root of the summed
variances around a condition's centre in a repeat, as ``data/simulations/whole_table.py``
defines it) is the unit of the effects.

The condition effect (``shift``, in within-group SDs; every non-control condition):

- ``dense``: a mean shift along a random direction of the data's span (weighted by the
  scales); ``sparse``: a mean shift on ``sparse_dims`` coordinates of the output (a
  coordinate-sparse shift, which favours per-coordinate tests);
- ``aligned``: a mean shift along the batch direction (biology aligned with batch);
- ``covariance``: no mean shift; the treated cells' spread along the first ``sparse_dims``
  latent axes multiplied by ``1 + shift`` (a null for a mean test, an alternative for a
  distributional one).

The batch between repeats beyond the repeat effect (``batch``, of size ``batch_size``):
``additive`` (each repeat shifted along one batch direction), ``scale`` (each repeat
scaled around the centre), ``rotation`` (a batch-specific rotation of each repeat: a real
distortion, unlike a global change of basis, which must change nothing: ``rotate``) and
``nonlinear``. ``truth`` holds the shift vector of the biology, before any batch.
``semi_synthetic`` injects a shift into real control observations instead.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import expm

from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN

from .design import Design, Layout, layout, tables

SHIFTS = ("dense", "sparse", "aligned", "covariance")
BATCHES = ("none", "additive", "scale", "rotation", "nonlinear")


@dataclass(frozen=True)
class EmbeddingSpec:
    """The outcome of one embedding table (see the module)."""

    dims: int = 32
    rank: int = 0
    decay: float = 0.7
    icc_repeat: float = 0.2
    icc_well: float = 0.05
    icc_field: float = 0.1
    shift: float = 0.0
    shift_kind: str = "dense"
    sparse_dims: int = 3
    batch: str = "none"
    batch_size: float = 1.0
    basis_seed: int = 0

    def __post_init__(self) -> None:
        if self.shift_kind not in SHIFTS:
            raise ValueError(f"shift_kind must be one of {SHIFTS}")
        if self.batch not in BATCHES:
            raise ValueError(f"batch must be one of {BATCHES}")
        if self.icc_repeat + self.icc_well + self.icc_field >= 1:
            raise ValueError("the variance shares of the levels must sum to less than 1")

    @property
    def latent_rank(self) -> int:
        """The latent dimensions: ``rank``, or every dimension when it is 0."""
        return self.dims if self.rank <= 0 else min(self.rank, self.dims)

    @property
    def scales(self) -> np.ndarray:
        """The SD of each latent axis, ``j^-decay``."""
        return np.arange(1, self.latent_rank + 1) ** -self.decay

    @property
    def within_sd(self) -> float:
        """The within-group SD: the root of the summed variances around a group's centre."""
        return float(np.sqrt((1 - self.icc_repeat) * np.square(self.scales).sum()))


def embedding_from(params: dict[str, Any]) -> EmbeddingSpec:
    """An ``EmbeddingSpec`` from the outcome keys of ``params`` (the others ignored)."""
    names = {f.name for f in fields(EmbeddingSpec)}
    return EmbeddingSpec(**{k: v for k, v in params.items() if k in names})


def random_rotation(dims: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthogonal matrix (Haar: QR of a Gaussian matrix, signs fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((dims, dims)))
    return q * np.sign(np.diag(r))


@lru_cache(maxsize=32)
def _basis(dims: int, seed: int) -> np.ndarray:
    return random_rotation(dims, np.random.default_rng(seed))


def rotate(objects: pd.DataFrame, columns: list[str], rotation: np.ndarray) -> pd.DataFrame:
    """``objects`` with ``columns`` replaced by their values times ``rotation``: a global
    orthogonal change of basis, which a geometry-only analysis must not notice."""
    out = objects.copy()
    out[columns] = objects[columns].to_numpy(dtype=float) @ rotation
    return out


@dataclass(frozen=True)
class EmbeddingData:
    """One simulated dataset: the tables, the embedding's columns, the truth (``shift``: the
    biology's mean shift vector of a treated condition; ``mean_null``, ``distribution_null``)
    and the layout."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    columns: tuple[str, ...]
    truth: dict[str, Any]
    layout: Layout | None


def _levels(frame: pd.DataFrame, rank: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per field, standard normal repeat, well and field vectors (``rank`` dimensions)."""
    out = {}
    for level, key in (("repeat", REPEAT_COLUMN), ("well", "well"), ("field", None)):
        codes = (
            np.arange(len(frame))
            if key is None
            else np.unique(frame[key].to_numpy(), return_inverse=True)[1]
        )
        size = codes.max() + 1 if len(codes) else 0
        out[level] = rng.standard_normal((size, rank))[codes]
    return out


def _direction(spec: EmbeddingSpec, basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The unit direction of the mean shift in output coordinates (see the module)."""
    if spec.shift_kind == "sparse":
        chosen = rng.choice(spec.dims, size=min(spec.sparse_dims, spec.dims), replace=False)
        vector = np.zeros(spec.dims)
        vector[chosen] = rng.standard_normal(len(chosen))
    elif spec.shift_kind == "aligned":
        vector = _batch_direction(spec, basis)
    else:
        vector = (spec.scales * rng.standard_normal(spec.latent_rank)) @ basis[: spec.latent_rank]
    return vector / np.linalg.norm(vector)


def _batch_direction(spec: EmbeddingSpec, basis: np.ndarray) -> np.ndarray:
    """The one direction of the additive batch: the second latent axis (fixed)."""
    axis = min(1, spec.latent_rank - 1)
    return basis[axis] / np.linalg.norm(basis[axis])


def _batch(
    spec: EmbeddingSpec,
    values: np.ndarray,
    repeats: np.ndarray,
    basis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``values`` with the batch of each repeat applied (see the module)."""
    if spec.batch == "none" or spec.batch_size == 0:
        return values
    out = values.copy()
    centre = values.mean(axis=0)
    size = spec.batch_size
    for r in pd.unique(repeats):
        rows = repeats == r
        if spec.batch == "additive":
            out[rows] += (
                size * spec.within_sd * rng.standard_normal() * _batch_direction(spec, basis)
            )
        elif spec.batch == "scale":
            out[rows] = centre + (values[rows] - centre) * np.exp(
                0.3 * size * rng.standard_normal()
            )
        elif spec.batch == "rotation":
            skew = rng.standard_normal((spec.dims, spec.dims))
            skew = (skew - skew.T) / np.sqrt(2 * spec.dims)
            out[rows] = centre + (values[rows] - centre) @ expm(0.3 * size * skew)
        else:  # nonlinear: a coordinatewise bend of each repeat, around the centre
            scale = spec.within_sd / np.sqrt(spec.dims)
            bend = 0.3 * size * rng.standard_normal(spec.dims)
            out[rows] = values[rows] + bend * scale * np.tanh((values[rows] - centre) / scale)
    return out


def generate(design: Design, spec: EmbeddingSpec, rng: np.random.Generator) -> EmbeddingData:
    """One dataset of ``spec`` on a fresh layout of ``design`` (see the module)."""
    lay = layout(design, rng)
    frame = lay.fields
    rows = lay.cells()
    rank, basis = spec.latent_rank, _basis(spec.dims, spec.basis_seed)
    z = _levels(frame, rank, rng)
    hier = (
        np.sqrt(spec.icc_repeat) * z["repeat"]
        + np.sqrt(spec.icc_well) * z["well"]
        + np.sqrt(spec.icc_field) * z["field"]
    )
    rest = 1 - spec.icc_repeat - spec.icc_well - spec.icc_field
    residual = np.sqrt(rest) * rng.standard_normal((len(rows), rank))
    treated = frame["treated"].to_numpy()[rows]
    if spec.shift_kind == "covariance" and spec.shift:
        k = min(spec.sparse_dims, rank)
        residual[treated, :k] *= 1 + spec.shift
    latent = (hier[rows] + residual) * spec.scales
    values = latent @ basis[:rank]
    vector = np.zeros(spec.dims)
    if spec.shift and spec.shift_kind != "covariance":
        vector = spec.shift * spec.within_sd * _direction(spec, basis, rng)
        values[treated] += vector
    repeats = frame[REPEAT_COLUMN].to_numpy()[rows]
    values = _batch(spec, values, repeats, basis, rng)
    columns = tuple(f"e{j}" for j in range(spec.dims))
    objects, fovs = tables(lay, pd.DataFrame(values, columns=list(columns)))
    truth = {
        "shift": vector,
        "shift_norm": float(np.linalg.norm(vector)),
        "mean_null": not np.any(vector),
        "distribution_null": spec.shift == 0,
    }
    return EmbeddingData(objects, fovs, columns, truth, lay)


def semi_synthetic(
    values: np.ndarray,
    meta: pd.DataFrame,
    shift: float,
    rng: np.random.Generator,
    field_col: str = "position_id",
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Real control observations made into a two-condition experiment: within each
    biological repeat, half the fields (at random) relabelled ``treated`` and shifted by
    ``shift`` within-group SDs along a random direction of the data's own span. Returns the
    values, the metadata with the new ``condition`` and the shift vector."""
    meta = meta.reset_index(drop=True).copy()
    centred = values - values.mean(axis=0)
    within = float(np.sqrt(np.square(centred).sum(axis=1).mean()))
    direction = centred[rng.integers(len(centred))] + centred[rng.integers(len(centred))]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    vector = shift * within * direction
    condition = np.full(len(meta), "control", dtype=object)
    for rows in meta.groupby(REPEAT_COLUMN).groups.values():
        fields_here = pd.unique(meta.loc[rows, field_col])
        chosen = rng.permutation(fields_here)[: len(fields_here) // 2]
        condition[meta.index.isin(rows) & meta[field_col].isin(chosen).to_numpy()] = "treated"
    meta[CONDITION_COLUMN] = condition
    out = values.copy()
    out[condition == "treated"] += vector
    return out, meta, vector


__all__ = [
    "BATCHES",
    "SHIFTS",
    "EmbeddingData",
    "EmbeddingSpec",
    "embedding_from",
    "generate",
    "random_rotation",
    "rotate",
    "semi_synthetic",
]
