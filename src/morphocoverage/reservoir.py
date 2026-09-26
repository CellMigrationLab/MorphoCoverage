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
  for example from ``spatial_blocks``), FOVs of one block never sit on
  opposite sides. That limits spatial leakage into the reference only as
  far as the blocks encode the correlation: FOVs on either side of a block
  edge can be close neighbours, and a square grid does not prevent this, so
  blocks should be larger than the correlation length (and ideally follow
  the sample's own structure, such as wells or tiles). Without ``block_col``
  every FOV is its own block, which assumes FOVs are exchangeable; the record
  states which was used.
- The reservoir is a simple random sample of ``ceil(fraction * n_blocks)``
  blocks (at most all but one; the product is rounded to 9 decimals first, so
  ``0.07 * 100`` gives 7, not 8), drawn from ``seed`` and the group identity.
  Every block, and so every FOV, has the same chance ``m / n_blocks`` of
  being held out, whatever the block's size, so the reservoir represents the
  same FOV population as the stream on average. ``fraction`` is therefore a
  share of blocks: with unequal blocks the share of FOVs held out varies
  between seeds, and one large block can take most of a group. The achieved
  size is recorded (``reservoir_fovs``). Use many blocks of similar size,
  and repeat a benchmark over several seeds so that conclusions do not rest
  on one partition.

The stream is a thinned acquisition. Holding FOVs out removes them from the
recorded order too, so a recorded-order replay of the stream is not the
recorded acquisition: with single-FOV blocks and ``fraction=0.5``, successive
stream FOVs of a raster scan lie about twice as far apart as successive
acquired FOVs, which weakens neighbour correlation and can make stopping look
easier than it would be at the microscope. Contiguous blocks keep runs of
neighbours together in the stream. Report reservoir runs as thinned-stream
validation, and read recorded-order results alongside a replay without a
reservoir.

A finite reservoir has its own sampling error. ``reference_error_floor``
estimates its scale from split halves. It is a diagnostic noise scale, not a
decision threshold: a stop whose error is below it is not thereby
"sufficient", since the floor depends on the metric, the reservoir size and
the blocks. Adequacy tolerances should sit well above it, and conclusions
should rest on several splits, never one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._checks import is_integer
from ._seeds import group_seed
from .evaluator import objects_in_fovs, restrict_input
from .outcomes import OutcomeMetric
from .schema import ValidatedInput


def spatial_blocks(fovs: pd.DataFrame, x_col: str, y_col: str, block_size: float) -> pd.Series:
    """Square-grid block of each FOV from its stage coordinates, e.g. ``"3_-1"``.

    Blocks are ``block_size`` wide in the coordinates' units; choose a size
    at least as large as the spatial correlation length, if known. Missing or
    infinite coordinates raise: they belong to no block.
    """
    if not (block_size > 0 and math.isfinite(block_size)):
        raise ValueError("block_size must be positive and finite")
    for column in (x_col, y_col):
        if column not in fovs.columns:
            raise ValueError(f"{column!r} is not a column of the FOV table")
        if fovs[column].isna().any():
            raise ValueError(f"FOV coordinates in {column!r} contain missing values")
        if not np.isfinite(fovs[column].to_numpy(dtype=float)).all():
            raise ValueError(f"FOV coordinates in {column!r} contain infinite values")
    x = np.floor(fovs[x_col].to_numpy(dtype=float) / block_size).astype(int)
    y = np.floor(fovs[y_col].to_numpy(dtype=float) / block_size).astype(int)
    return pd.Series([f"{i}_{j}" for i, j in zip(x, y, strict=True)], index=fovs.index)


def _check_one_group(data: ValidatedInput) -> None:
    """Refuse an input holding several analysis groups: a split never pools groups."""
    if not data.group_cols:
        return
    groups = data.fovs.loc[:, list(data.group_cols)].drop_duplicates()
    if len(groups) > 1:
        raise ValueError(
            f"the input holds {len(groups)} analysis groups "
            f"(e.g. {groups.head(2).to_dict('records')}); a reservoir is split within one "
            "group, never across pooled groups. Split each group of "
            "morphocoverage.evaluator.iter_analysis_groups apart."
        )


def _subset(data: ValidatedInput, fovs: pd.DataFrame) -> ValidatedInput:
    """The given FOV rows of ``data`` with their observations."""
    return restrict_input(data, objects_in_fovs(data, fovs), fovs)


@dataclass(frozen=True)
class ReservoirSplit:
    """Split each analysis group into a replay stream and a fixed evaluation reservoir.

    ``fraction`` is the share of each group's blocks held out (rounded up,
    and at most all but one block), ``seed`` makes the split reproducible, and
    ``block_col`` names an FOV-table column of spatial blocks (``None``: each
    FOV is its own block). Blocks are sampled with equal probability; see the
    module docstring.
    """

    fraction: float
    seed: int
    block_col: str | None

    def __post_init__(self) -> None:
        if isinstance(self.fraction, bool) or not 0 < float(self.fraction) < 1:
            raise ValueError("fraction must be between 0 and 1")
        object.__setattr__(self, "fraction", float(self.fraction))
        if not is_integer(self.seed):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
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

    def n_blocks(self, data: ValidatedInput) -> int:
        """The number of blocks (or FOVs, without ``block_col``) in ``data``."""
        return len(np.unique(self._blocks(data.fovs)))

    def split(
        self, group: Mapping[str, Any], data: ValidatedInput
    ) -> tuple[ValidatedInput, ValidatedInput]:
        """(stream, reservoir) for one analysis group; both keep whole blocks.

        FOVs are put in FOV-key order first, so the split depends on which FOVs
        were acquired and never on the row order of the FOV table. ``data`` must
        hold one analysis group: an input with several groups raises rather than
        pooling them into one stream and one reservoir; split each group of
        ``iter_analysis_groups`` apart.
        """
        _check_one_group(data)
        fovs = data.fovs.sort_values(list(data.fov_key), kind="stable")
        blocks = self._blocks(fovs)
        labels, block_of_fov = np.unique(blocks, return_inverse=True)
        if len(labels) < 2:
            raise ValueError(
                f"analysis group {dict(group)} has too few FOVs or blocks ({len(labels)}) to "
                "hold out a reservoir and keep a stream"
            )
        # A simple random sample of blocks: equal inclusion probability for every
        # block, so large blocks are not under-represented in the reservoir.
        # A size cap would skip large blocks and bias the reference.
        # Rounded first, so floating-point error (0.07 * 100 = 7.000000000000001)
        # cannot add a block.
        n_held = min(math.ceil(round(self.fraction * len(labels), 9)), len(labels) - 1)
        rng = np.random.default_rng(group_seed(self.seed, group))
        chosen = rng.choice(len(labels), size=n_held, replace=False)
        held_out = np.isin(block_of_fov, chosen)
        return _subset(data, fovs.loc[~held_out]), _subset(data, fovs.loc[held_out])


def reference_error_floor(
    metric: OutcomeMetric,
    reference: ValidatedInput,
    *,
    n_splits: int,
    seed: int,
    block_col: str | None,
) -> list[float | None]:
    """The metric between random halves of a reservoir, once per split.

    Each split puts half the blocks (or FOVs), chosen at random, on each side
    and scores one side against the other; with an odd number of blocks the
    scored side has one block fewer than the side it is scored against
    (``ReservoirSplit(0.5, ...)`` rounds the held-out side up). ``reference``
    must hold one analysis group, as for ``ReservoirSplit.split``. With equal,
    independent halves
    this is the error of a sample half the reservoir's size against an
    independent reference of the same size, so for mean-type errors it is
    about twice the reservoir's own standard error: a conservative scale for
    the floor below which a stop's error cannot be distinguished from
    reference noise. With unequal blocks the halves differ in size, and with
    correlated blocks they are not independent, so the scale is then only
    approximate. It is a noise scale for reading errors, not a threshold for
    deciding that imaging was sufficient. Values are ``None`` where the metric
    is undefined.
    """
    if not is_integer(n_splits) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    if not is_integer(seed):
        raise TypeError("seed must be an integer")
    floors: list[float | None] = []
    for i in range(int(n_splits)):
        half, other = ReservoirSplit(0.5, int(seed) + i, block_col).split({}, reference)
        floors.append(metric.evaluate(other, half))
    return floors


__all__ = ["ReservoirSplit", "reference_error_floor", "spatial_blocks"]
