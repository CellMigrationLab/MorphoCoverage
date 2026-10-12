"""The balanced permutation test's arithmetic (private), every labelling at once.

``contrast_cells`` describes the test: the median Cohen's d over fixed balanced draws, its
null from labellings shuffled within each repeat. Here each labelling's statistic comes
from matrix products instead of a copy of every draw: per repeat, the values a labelling
puts at the treated and control positions times a 0/1 matrix of the positions each draw
takes give every draw's sum, and the squared values its sum of squares, so the means and
pooled SDs of all draws and labellings follow without a loop over either. The values are
first shifted by a constant shared by both sides (on raw values a centre of the pooled
values, normalised the value 1 after dividing by each labelling's control mean), which
leaves Cohen's d as it is and keeps the sums of squares from cancelling. Given the same
generator, the labellings are those of the earlier loop (the random keys drawn in the same
order, block by block), so the d and p are the same up to rounding (d within 1e-12,
``tests/test_cells_vectorised.py``).

Precomputed once per test (``BalancedDesign``) and reused by its observed statistic, its
first pass and the rerun with more labellings: the shifted values and their squares, and
the draws' matrices (built again from the generator for the rerun, so a test kept for it
holds a few numbers per observation).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

# Labellings handled at once: about this many values drawn, gathered and multiplied a block.
_VALUES_PER_BLOCK = 2_000_000
# The block of labellings the earlier loop drew its keys in, from this many values a block.
_FORMER_BLOCK_VALUES = 4_000_000
# A pooled variance this small a share of the mean square of the shifted values is zero
# (rounding of the sums of squares), as the earlier ``pooled > 0`` read an exact zero.
_ZERO_VARIANCE = 1e-13


def picks(sizes: list[int], keep: int, draws: int, rng: np.random.Generator) -> list[Any]:
    """Per repeat, which ``keep`` of its ``sizes`` observations each of ``draws`` draws takes
    (``(draws, keep)`` positions; every one, once, when ``keep`` is all of them)."""
    if all(n == keep for n in sizes):
        return [np.arange(n)[None, :] for n in sizes]
    return [np.argsort(rng.random((draws, n)), axis=1)[:, :keep] for n in sizes]


def _taken(positions: np.ndarray, size: int) -> np.ndarray:
    """The ``(size, draws)`` 0/1 matrix of the positions each draw takes."""
    matrix = np.zeros((size, len(positions)))
    matrix[positions.T, np.arange(len(positions))[None, :]] = 1.0
    return matrix


# Per side, each repeat's matrix of the positions its draws take (``_taken``).
Taken = tuple[list[np.ndarray], list[np.ndarray]]


@dataclass(frozen=True)
class BalancedDesign:
    """One balanced test's fixed parts: per repeat its shifted pooled values (treated first)
    and their squares, its treated count and its observed control mean; the observations a
    draw keeps a repeat on each side, the draws and the generator they were drawn from (so a
    test kept for its rerun holds its values only, not its draws), whether the values are
    normalised, and the block of labellings the earlier loop drew its keys in."""

    shifted: list[np.ndarray]
    squared: list[np.ndarray]
    treated_sizes: list[int]
    control_means: list[float]
    keep: tuple[int, int]
    n_draws: int
    draws_from: np.random.Generator
    normalise: bool
    former_block: int

    @classmethod
    def of(
        cls,
        pools: list[np.ndarray],
        treated_sizes: list[int],
        n_draws: int,
        rng: np.random.Generator,
        normalise: bool,
    ) -> tuple[BalancedDesign, Taken]:
        """The design of ``pools`` (each repeat's treated values, then its control's) and the
        matrices of its draws, ``n_draws`` taking the fewest any repeat holds on each side,
        drawn from ``rng``."""
        control_sizes = [len(pool) - n_t for pool, n_t in zip(pools, treated_sizes, strict=True)]
        means = [float(pool[n_t:].mean()) for pool, n_t in zip(pools, treated_sizes, strict=True)]
        # Raw, one centre for every value; normalised, each repeat's own control mean, then
        # 1 after the division (see the module).
        centres = means if normalise else [float(np.concatenate(pools).mean())] * len(pools)
        shifted = [pool - centre for pool, centre in zip(pools, centres, strict=True)]
        keep = (min(treated_sizes), min(control_sizes))
        # A side whose every repeat holds the fewest is one draw of every observation.
        width = sum(len(pool) for pool in pools) * max(
            1 if all(n == k for n in sizes) else n_draws
            for sizes, k in zip((treated_sizes, control_sizes), keep, strict=True)
        )
        design = cls(
            shifted=shifted,
            squared=[values**2 for values in shifted],
            treated_sizes=list(treated_sizes),
            control_means=means,
            keep=keep,
            n_draws=n_draws,
            draws_from=copy.deepcopy(rng),
            normalise=normalise,
            former_block=max(1, _FORMER_BLOCK_VALUES // width),
        )
        return design, design.matrices(design.draws(rng))

    def draws(self, rng: np.random.Generator | None = None) -> tuple[list[Any], list[Any]]:
        """Each side's ``picks``, from ``rng`` or again from the generator they came from."""
        rng = copy.deepcopy(self.draws_from) if rng is None else rng
        sizes = [len(v) for v in self.shifted]
        control_sizes = [n - n_t for n, n_t in zip(sizes, self.treated_sizes, strict=True)]
        return (
            picks(self.treated_sizes, self.keep[0], self.n_draws, rng),
            picks(control_sizes, self.keep[1], self.n_draws, rng),
        )

    def matrices(self, draws: tuple[list[Any], list[Any]]) -> Taken:
        """Per side, each repeat's matrix of the positions the ``draws`` take."""
        sizes = [len(v) for v in self.shifted]
        return (
            [_taken(p, n_t) for p, n_t in zip(draws[0], self.treated_sizes, strict=True)],
            [
                _taken(p, n - n_t)
                for p, n, n_t in zip(draws[1], sizes, self.treated_sizes, strict=True)
            ],
        )

    def statistic(self, orders: list[np.ndarray], taken: Taken) -> np.ndarray:
        """The median Cohen's d over the draws (``taken``) for each labelling: ``orders``
        holds, per repeat, a ``(labellings, n)`` permutation of its pooled values (treated
        positions first); a labelling whose every draw is undefined gives NaN."""
        sums = [0.0, 0.0]
        squares = [0.0, 0.0]
        for index, order in enumerate(orders):
            n_t = self.treated_sizes[index]
            values, squared = self.shifted[index][order], self.squared[index][order]
            scale = offset = None
            if self.normalise:
                control = values[:, n_t:].mean(axis=1, keepdims=True)
                mean = self.control_means[index] + control
                # x / m - 1 = y / m + (m0 / m - 1), y = x - m0: the shift stays exact.
                scale, offset = 1.0 / mean, -control / mean
            for side, columns in enumerate((slice(None, n_t), slice(n_t, None))):
                matrix = taken[side][index]
                total = values[:, columns] @ matrix
                square = squared[:, columns] @ matrix
                if scale is not None:
                    count = self.keep[side]
                    square = scale**2 * square + 2 * scale * offset * total + count * offset**2
                    total = scale * total + count * offset
                sums[side] = sums[side] + total
                squares[side] = squares[side] + square
        n_a, n_b = (count * len(orders) for count in self.keep)
        # A side drawn whole holds one draw, broadcast against the other side's.
        deviation_a = squares[0] - sums[0] ** 2 / n_a
        deviation_b = squares[1] - sums[1] ** 2 / n_b
        pooled = (deviation_a + deviation_b) / (n_a + n_b - 2)
        floor = _ZERO_VARIANCE * (squares[0] + squares[1]) / (n_a + n_b)
        # fallback: a draw without spread: NaN, left out of the median
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(pooled > floor, (sums[0] / n_a - sums[1] / n_b) / np.sqrt(pooled), np.nan)
        finite = np.isfinite(d).any(axis=1)
        median = np.full(len(d), np.nan)
        median[finite] = np.nanmedian(d[finite], axis=1)
        return median

    def observed(self, taken: Taken) -> float:
        """The statistic of the labelling as observed."""
        orders = [np.arange(len(v))[None, :] for v in self.shifted]
        return float(self.statistic(orders, taken)[0])

    def labellings(self, count: int, rng: np.random.Generator) -> list[np.ndarray]:
        """``count`` labellings shuffled within each repeat, one ``(count, n)`` order per
        repeat, from keys drawn as the earlier loop drew them: per block of
        ``former_block`` labellings, each repeat's keys in turn."""
        sizes = [len(v) for v in self.shifted]
        block = self.former_block
        parts: list[list[np.ndarray]] = []
        full, rest = divmod(count, block)
        for rows, blocks in ((block, full), (rest, int(rest > 0))):
            if not blocks:
                continue
            keys = rng.random((blocks, rows * sum(sizes)))
            start, part = 0, []
            for n in sizes:
                part.append(keys[:, start : start + rows * n].reshape(blocks * rows, n))
                start += rows * n
            parts.append(part)
        return [
            np.argsort(np.concatenate([part[index] for part in parts]), axis=1)
            for index in range(len(sizes))
        ]

    def p_value(
        self,
        observed: float,
        n_permutations: int,
        rng: np.random.Generator,
        taken: Taken | None = None,
    ) -> float:
        """The permutation p, ``(extreme + 1) / (B + 1)`` over ``B`` labellings; a labelling
        whose statistic is undefined counts as extreme. ``taken``: the draws' matrices,
        drawn again when not given."""
        if not np.isfinite(observed):
            return np.nan
        taken = self.matrices(self.draws()) if taken is None else taken
        # Whole blocks of the earlier loop at a time, so its keys come in its order.
        total = sum(len(v) for v in self.shifted)
        step = self.former_block * max(1, _VALUES_PER_BLOCK // (total * self.former_block))
        extreme = 0
        for start in range(0, n_permutations, step):
            orders = self.labellings(min(step, n_permutations - start), rng)
            null = self.statistic(orders, taken)
            extreme += int(np.sum(~(np.abs(null) < abs(observed) - 1e-12)))
        return (extreme + 1) / (n_permutations + 1)


@dataclass(frozen=True)
class BalancedRerun:
    """A balanced test's p over ``count`` new labellings, its observed statistic kept
    (``contrast_cells.refined``); a plain object, so worker processes can return it."""

    design: BalancedDesign | None
    observed: float

    def __call__(self, count: int, rng: np.random.Generator) -> float:
        if self.design is None:
            return np.nan
        return self.design.p_value(self.observed, count, rng)


__all__ = ["BalancedDesign", "BalancedRerun", "picks"]
