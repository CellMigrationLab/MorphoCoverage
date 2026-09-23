"""Fixed evaluation reservoirs: score stops against FOVs the replay never reveals.

Scoring a stop prefix against the full acquisition is self-referential: the
reference contains the prefix, so every error goes to zero as the prefix
grows, whatever a strategy does. A reservoir split instead sets aside part of
each analysis group before replay. Strategies see only the stream; outcome
metrics compare the stop with the reservoir, which does not grow with the
prefix (docs/proposals/candidate_estimators_roadmap.md, section 4.2).

Splitting rules:

- The split is made per analysis group, so every group keeps its own
  reservoir and groups are never pooled.
- Whole **blocks** go to one side. With ``block_col`` (an FOV-table column,
  for example from ``spatial_blocks``), neighbouring FOVs in one block cannot
  sit on opposite sides, which would leak spatially correlated information
  into the reference. Without it every FOV is its own block, which assumes
  FOVs are exchangeable; the record states which was used.
- Blocks are drawn at random from ``seed`` and the group identity, until the
  reservoir holds at least ``fraction`` of the group's FOVs. Repeat a
  benchmark over several seeds so that conclusions do not rest on one
  partition.

A finite reservoir has its own sampling error. ``reference_error_floor``
estimates its scale from split halves; adequacy tolerances should sit well
above it.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import objects_in_fovs, restrict_input
from .outcomes import OutcomeMetric
from .schema import ValidatedInput


def spatial_blocks(fovs: pd.DataFrame, x_col: str, y_col: str, block_size: float) -> pd.Series:
    """Square-grid block of each FOV from its stage coordinates, e.g. ``"3_-1"``.

    Blocks are ``block_size`` wide in the coordinates' units; choose a size
    at least as large as the spatial correlation length, if known.
    """
    if not block_size > 0:
        raise ValueError("block_size must be positive")
    for column in (x_col, y_col):
        if column not in fovs.columns:
            raise ValueError(f"{column!r} is not a column of the FOV table")
        if fovs[column].isna().any():
            raise ValueError(f"FOV coordinates in {column!r} contain missing values")
    x = np.floor(fovs[x_col].to_numpy(dtype=float) / block_size).astype(int)
    y = np.floor(fovs[y_col].to_numpy(dtype=float) / block_size).astype(int)
    return pd.Series([f"{i}_{j}" for i, j in zip(x, y, strict=True)], index=fovs.index)


def _group_seed(seed: int, group: Mapping[str, Any]) -> list[int]:
    """Seed material for one group: the split seed and a stable hash of the group identity."""
    identity = json.dumps(sorted(group.items()), default=str).encode()
    return [seed, zlib.crc32(identity)]


def _subset(data: ValidatedInput, fovs: pd.DataFrame) -> ValidatedInput:
    """The given FOV rows of ``data`` with their observations."""
    return restrict_input(data, objects_in_fovs(data, fovs), fovs)


@dataclass(frozen=True)
class ReservoirSplit:
    """Split each analysis group into a replay stream and a fixed evaluation reservoir.

    ``fraction`` is the minimum share of each group's FOVs held out, ``seed``
    makes the split reproducible, and ``block_col`` names an FOV-table column
    of spatial blocks (``None``: each FOV is its own block).
    """

    fraction: float
    seed: int
    block_col: str | None

    def __post_init__(self) -> None:
        if isinstance(self.fraction, bool) or not 0 < float(self.fraction) < 1:
            raise ValueError("fraction must be between 0 and 1")
        object.__setattr__(self, "fraction", float(self.fraction))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.block_col is not None and not isinstance(self.block_col, str):
            raise TypeError("block_col must be a column name or None")

    @property
    def parameters(self) -> dict[str, Any]:
        """The split settings, for provenance."""
        return {"fraction": self.fraction, "seed": self.seed, "block_col": self.block_col}

    def _blocks(self, fovs: pd.DataFrame) -> np.ndarray:
        """Block label of each FOV row."""
        if self.block_col is None:
            return np.arange(len(fovs))
        if self.block_col not in fovs.columns:
            raise ValueError(f"block_col {self.block_col!r} is not a column of the FOV table")
        if fovs[self.block_col].isna().any():
            raise ValueError(f"block_col {self.block_col!r} contains missing values")
        return fovs[self.block_col].to_numpy()

    def split(
        self, group: Mapping[str, Any], data: ValidatedInput
    ) -> tuple[ValidatedInput, ValidatedInput]:
        """(stream, reservoir) for one analysis group; both keep whole blocks."""
        blocks = self._blocks(data.fovs)
        labels, block_of_fov = np.unique(blocks, return_inverse=True)
        order = np.random.default_rng(_group_seed(self.seed, group)).permutation(len(labels))
        sizes = np.bincount(block_of_fov, minlength=len(labels))[order]
        target = math.ceil(self.fraction * len(data.fovs))
        # Blocks enter the reservoir in random order until it reaches the target.
        n_blocks = int(np.searchsorted(np.cumsum(sizes), target) + 1)
        if n_blocks >= len(labels):
            raise ValueError(
                f"analysis group {dict(group)} has too few FOVs or blocks ({len(labels)}) to "
                f"hold out {self.fraction:.0%} and keep a stream"
            )
        held_out = np.isin(block_of_fov, order[:n_blocks])
        return _subset(data, data.fovs.loc[~held_out]), _subset(data, data.fovs.loc[held_out])


def reference_error_floor(
    metric: OutcomeMetric,
    reference: ValidatedInput,
    *,
    n_splits: int,
    seed: int,
    block_col: str | None,
) -> list[float | None]:
    """The metric between random halves of a reservoir, once per split.

    Each split puts whole blocks (or FOVs) into two halves and scores one
    half against the other. This is the error of a sample half the
    reservoir's size against an independent reference of the same size, so
    for mean-type errors it is about twice the reservoir's own standard
    error: a conservative scale for the floor below which a stop's error
    cannot be distinguished from reference noise. Values are ``None`` where
    the metric is undefined.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    floors: list[float | None] = []
    for i in range(n_splits):
        half, other = ReservoirSplit(0.5, seed + i, block_col).split({}, reference)
        floors.append(metric.evaluate(other, half))
    return floors


__all__ = ["ReservoirSplit", "reference_error_floor", "spatial_blocks"]
