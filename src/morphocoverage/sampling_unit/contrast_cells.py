"""The cell-level randomization test (Plot&Stats), balanced when the repeats are not.

The lab's Plot&Stats test compares every observation of a condition with every
observation of the control, pooled as provided (all repeats together): Cohen's
d on the observations, with a randomization test that shuffles the condition
labels over the observations (``n_permutations``, ``(extreme + 1) / (n + 1)``),
Benjamini-Hochberg corrected within each sub-experiment (``_experiments``; the ``q_``
columns). It asks whether these wells differ,
and it is the only test that responds with one or two biological repeats. A test as provided
whose control pools repeats the condition was not imaged in is withheld from the report, so
it has no q and leaves the family: the q-values are corrected over the tests reported
(owner, 2026-10-05).

Every number is given on the raw values and on the **normalised** values: each
value divided by the mean of the control of its own repeat (the control's
values by their own repeat's mean too), so a scale that both wells of a repeat
share (a staining or imaging batch) is removed. Cohen's d within one repeat is
unchanged by it; pooled over repeats it is not. The normalised version needs a
positive control mean in every repeat (NaN otherwise).

**The balance check.** A condition is *unbalanced* when, on its side or the
control's, the repeat with the most observations holds more than
``IMBALANCE`` (25%) more than the one with the fewest, or when the control is
pooled from repeats the condition was not imaged in. Then one repeat weighs
more than the others in the pooled test and its d, and the report gives, next
to the test as provided, a **balanced** version: each condition against the
control of its own repeats only, as one permutation test (owner, 2026-10-05; before, the
median p over the draws, which is not the p of one test). Its statistic, ``d_balanced``,
is the median Cohen's d over ``BALANCED_DRAWS`` draws, each taking the same number of
observations from every one of those repeats (the fewest any repeat holds, the
condition's and the control's separately), so every repeat weighs the same; across the
draws every observation is used. Its null shuffles the condition and control labels
within each repeat, ``BALANCED_PERMUTATIONS`` times, and computes the same statistic, on
the same draws (the same positions in each repeat), for each labelling:
``p = (extreme + 1) / (B + 1)``, corrected the same way. Given the draws, the statistic is a
fixed function of the labels, so the p is calibrated when the observations of the condition
and the control of a repeat are exchangeable (``tests/test_contrast_cells.py`` simulates
it). Normalised, each labelling divides a repeat by the mean of its control as labelled.
When every repeat holds the fewest on both sides, one draw holds every observation and the
test is a permutation test stratified by repeat. SciPy's ``permutation_test`` is not used:
it does not shuffle within strata.

Balancing down to a repeat that holds very few observations would leave little
of the others in each draw, so a repeat holding fewer than ``SPARSE`` (a
quarter) of the median repeat on its side, or fewer than ``MIN_BALANCED``
observations, is *sparse*: it is named in ``sparse_repeats`` (a repeat that
small is worth looking at before any test) and left out of the balanced test
only, which balances the other repeats (``balanced_repeats``). The test as
provided keeps every observation. A sparse repeat on either side is left out of
both sides, so each condition stays paired with its own repeats' control.

**More permutations where the correction needs them** (owner, 2026-10-02). A
randomization p from ``B`` shuffles cannot fall below ``1 / (B + 1)``, so with ``K`` tests
in a family Benjamini-Hochberg cannot give a q below ``K / (B + 1)``: with 210 tests and the
balanced test's 999 shuffles, three features at that floor got q 0.07, never 0.05,
whatever their effect. Every test whose p is at most ``ALPHA`` after the first pass is
therefore rerun with ``B = max(default, ceil(K / ALPHA))`` shuffles (``permutations_for``;
``K`` the tests of its sub-experiment), so that ``K / (B + 1) < ALPHA``. Only those: the
threshold of rank ``r`` is ``ALPHA r / K``, at most ``ALPHA``, so a test above it cannot
pass at any rank, and a smaller p moving never changes the q of a larger one (a q depends
only on the p-values at or above it). The rerun is seeded from the test's own identity, so
it does not depend on the order of the tests. The balanced test is rerun with ``B`` new
labellings, its observed statistic kept (``balanced_draws``, ``refined``).

It treats observations as independent draws. When the difference varies
between repeats by ``s`` observation SDs (the variation between repeats that
``repeat_plan`` estimates, ``between_sd``), the variance of the mean difference
is ``k = 1 + n s^2 / 2`` times what the test assumes (``n`` observations per
well; with repeats holding different numbers, the effective
``n = sum(n_r^2) / sum(n_r)``, each repeat weighted by its own share, the mean
of the two sides: ``effective_per_well``), so without an effect it gives
p <= alpha in about
``2 * P(Z > z_alpha/2 / sqrt(k))`` of experiments (``chance_without_effect``).
In ``data/simulations/test_choice.py`` this matched the simulated rate of
the test on values normalised to each repeat's control within a few points
(16% and 37% at s = 0.25 and 0.5 with 30 cells, against 15-17% and 32-36%),
and in ``data/simulations/unbalanced.py`` within 1-2 points with one repeat
holding 2 to 8 times the cells of the others (it rises from 16% to 46% at
s = 0.25 while the effective ``n`` grows). The balanced test is given the
chance at the observations per repeat it keeps (``chance_balanced_*``): with
``unbalanced.py``'s generator at s = 0.25 it gave 15-19% against 16% (normalised, 500
experiments a setting, 2026-10-05).
A control pooled from repeats the condition was not imaged in adds to it
(22% against 16%, normalised), which the balanced test, against the control
of the condition's own repeats, removes;
on raw values the rate is at or below it (shifts that both wells of a repeat
share add noise that holds it down), so there it is an upper bound. With one
repeat, ``s`` is unknown and the chance is given for ``TYPICAL_BETWEEN``
(0.25-0.5, the range in the lab datasets). Fields holding
several observations add to it (their own design effect), which this does not
count.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..statistics import cohen_d
from ._experiments import corrected, experiments, family_sizes

ALPHA = 0.05
# Balanced resamples per condition, and the shuffles of the randomization test in each.
BALANCED_DRAWS = 100
BALANCED_PERMUTATIONS = 999
# Most over fewest observations per repeat beyond which a side is unbalanced (25% more).
IMBALANCE = 1.25
# A repeat holding less than this share of the median repeat on its side, or fewer than
# MIN_BALANCED observations, is too small to balance to.
SPARSE = 0.25
MIN_BALANCED = 5
# The variation between repeats assumed when one repeat cannot estimate it (observation SDs).
TYPICAL_BETWEEN = (0.25, 0.5)
CELL_COLUMNS = [
    "condition",
    "control_condition",
    "observations",
    "control_observations",
    "d_cells",
    "p_cells",
    "q_cells",
    "d_cells_normalised",
    "p_cells_normalised",
    "q_cells_normalised",
    "repeats",
    "spread",
    "other_control_repeats",
    "control_shift",
    "unbalanced",
    "sparse_repeats",
    "balanced_repeats",
    "balanced_cells",
    "balanced_control_cells",
    "d_balanced",
    "p_balanced",
    "q_balanced",
    "d_balanced_normalised",
    "p_balanced_normalised",
    "q_balanced_normalised",
    "per_well",
    "between_sd",
    "chance_low",
    "chance_high",
    "chance_balanced_low",
    "chance_balanced_high",
    "experiment",
    "permutations",
    "balanced_permutations",
]
# A treated and a control sample a randomization p can be rerun on; a test is rerun on its
# samples (the mean p over them) or, balanced, by its own rerun(B, rng).
Pair = tuple[np.ndarray, np.ndarray]
Retest = list[Pair] | Callable[[int, np.random.Generator], float]
_P_COLUMNS = ("p_cells", "p_cells_normalised", "p_balanced", "p_balanced_normalised")


def effective_per_well(sizes: list[int]) -> float:
    """The observations per well a pooled test weighs the repeats by: ``sum(n^2) / sum(n)``
    (``n`` when every repeat holds ``n``; near the largest when one repeat dominates)."""
    counts = np.asarray(sizes, float)
    return float((counts**2).sum() / counts.sum())


def _chances(per_well: float, between_sd: float) -> tuple[float, float]:
    """The chance at the estimated variation between repeats, or the ``TYPICAL_BETWEEN``
    range when it is unknown."""
    if np.isfinite(between_sd):
        value = chance_without_effect(per_well, between_sd)
        return value, value
    low, high = (chance_without_effect(per_well, value) for value in TYPICAL_BETWEEN)
    return low, high


def chance_without_effect(per_well: float, between_sd: float, alpha: float = ALPHA) -> float:
    """How often the cell-level test gives p <= ``alpha`` without an effect (see the module)."""
    inflation = 1 + per_well * between_sd**2 / 2
    return float(2 * stats.norm.sf(stats.norm.isf(alpha / 2) / np.sqrt(inflation)))


def cell_randomization(
    treated: np.ndarray, control: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """``(d, p)``: Cohen's d on the observations and the randomization p over them."""
    d = cohen_d(treated, control)
    if not np.isfinite(d):
        return float(d), np.nan
    pooled = np.concatenate([treated, control])
    n = len(treated)
    extreme = 0
    for start in range(0, n_permutations, 256):
        block = min(256, n_permutations - start)
        shuffled = pooled[np.argsort(rng.random((block, len(pooled))), axis=1)]
        a, b = shuffled[:, :n], shuffled[:, n:]
        pooled_var = ((n - 1) * a.var(axis=1, ddof=1) + (len(b[0]) - 1) * b.var(axis=1, ddof=1)) / (
            len(pooled) - 2
        )
        null = np.abs(a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(pooled_var)
        extreme += int(np.sum(null >= abs(d) - 1e-12))
    return float(d), (extreme + 1) / (n_permutations + 1)


def _picks(sizes: list[int], keep: int, draws: int, rng: np.random.Generator) -> list[Any]:
    """Per repeat, which ``keep`` of its ``sizes`` observations each of ``draws`` draws takes
    (``(draws, keep)`` positions; every one, once, when ``keep`` is all of them)."""
    if all(n == keep for n in sizes):
        return [np.arange(n)[None, :] for n in sizes]
    return [np.argsort(rng.random((draws, n)), axis=1)[:, :keep] for n in sizes]


def _median_d(
    pools: list[np.ndarray],
    treated_sizes: list[int],
    picks: tuple[list[Any], list[Any]],
    normalise: bool,
    orders: list[np.ndarray] | None,
) -> np.ndarray:
    """The balanced statistic, the median Cohen's d over the balanced draws ``picks`` (the
    treated's and the control's positions in each repeat), per labelling of each repeat's
    ``pools`` (its treated values first, then its control's): the ``orders`` of a block of
    shuffles within each repeat, or as observed without them. Normalised, each repeat is
    divided by the mean of its control as labelled."""
    treated, control = [], []
    for index, (pool, n_t) in enumerate(zip(pools, treated_sizes, strict=True)):
        labelled = pool[None, :] if orders is None else pool[orders[index]]
        mine, theirs = labelled[:, :n_t], labelled[:, n_t:]
        if normalise:
            mean = theirs.mean(axis=1, keepdims=True)
            mine, theirs = mine / mean, theirs / mean
        treated.append(mine[:, picks[0][index]])
        control.append(theirs[:, picks[1][index]])
    # A side drawn whole holds one draw, broadcast against the other side's.
    a, b = np.concatenate(treated, axis=2), np.concatenate(control, axis=2)
    n_a, n_b = a.shape[2], b.shape[2]
    pooled = ((n_a - 1) * a.var(axis=2, ddof=1) + (n_b - 1) * b.var(axis=2, ddof=1)) / (
        n_a + n_b - 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled > 0, (a.mean(axis=2) - b.mean(axis=2)) / np.sqrt(pooled), np.nan)
    finite = np.isfinite(d).any(axis=1)
    median = np.full(len(d), np.nan)
    median[finite] = np.nanmedian(d[finite], axis=1)
    return median


def _balanced_p(
    pools: list[np.ndarray],
    treated_sizes: list[int],
    picks: tuple[list[Any], list[Any]],
    normalise: bool,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """The balanced permutation p, ``(extreme + 1) / (B + 1)`` over ``B`` labellings shuffled
    within each repeat; a labelling whose statistic is undefined counts as extreme."""
    if not np.isfinite(observed):
        return np.nan
    width = sum(len(pool) for pool in pools) * max(len(picks[0][0]), len(picks[1][0]))
    block = max(1, min(n_permutations, 4_000_000 // width))
    extreme = 0
    for start in range(0, n_permutations, block):
        count = min(block, n_permutations - start)
        orders = [np.argsort(rng.random((count, len(pool))), axis=1) for pool in pools]
        null = _median_d(pools, treated_sizes, picks, normalise, orders)
        extreme += int(np.sum(~(np.abs(null) < abs(observed) - 1e-12)))
    return (extreme + 1) / (n_permutations + 1)


def balanced_draws(
    treated: list[np.ndarray],
    control: list[np.ndarray],
    rng: np.random.Generator,
    normalise: bool = False,
) -> tuple[float, float, int, int, Callable[[int, np.random.Generator], float]]:
    """``balanced_test`` and its rerun: ``rerun(B, rng)`` is its p over ``B`` new labellings,
    the observed d and the draws kept (``refined``). ``normalise`` divides each repeat by the
    mean of its control, as labelled in each labelling (see the module)."""
    per_t, per_c = min(len(t) for t in treated), min(len(c) for c in control)
    if per_t < 2 or per_c < 2:
        return np.nan, np.nan, per_t, per_c, lambda count, generator: np.nan
    pools = [np.concatenate([t, c]) for t, c in zip(treated, control, strict=True)]
    sizes = [len(t) for t in treated]
    # One set of draws for the observed labelling and every shuffled one: given the draws,
    # the statistic is a fixed function of the labels, so the shuffles test it exactly.
    picks = (
        _picks(sizes, per_t, BALANCED_DRAWS, rng),
        _picks([len(c) for c in control], per_c, BALANCED_DRAWS, rng),
    )
    observed = float(_median_d(pools, sizes, picks, normalise, None)[0])

    def rerun(count: int, generator: np.random.Generator) -> float:
        return _balanced_p(pools, sizes, picks, normalise, observed, count, generator)

    return observed, rerun(BALANCED_PERMUTATIONS, rng), per_t, per_c, rerun


def balanced_test(
    treated: list[np.ndarray], control: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, float, int, int]:
    """``(d, p, cells, control cells)`` of the balanced permutation test: d the median over
    ``BALANCED_DRAWS`` draws taking the same number of observations from every repeat (the
    fewest any repeat holds), p over ``BALANCED_PERMUTATIONS`` labellings shuffled within
    each repeat (see the module); NaN when a repeat holds fewer than 2 on a side."""
    return balanced_draws(treated, control, rng)[:4]


def permutations_for(family_size: int, default: int, alpha: float = ALPHA) -> int:
    """The shuffles a test of a family of ``family_size`` is rerun with when its p is at most
    ``alpha``: enough that the smallest q, ``K / (B + 1)``, is below ``alpha``, and never
    fewer than ``default`` (see the module)."""
    return max(int(default), int(np.ceil(family_size / alpha)))


def refined(
    p_values: pd.Series,
    family: pd.Series,
    used: pd.Series,
    retests: Mapping[Any, Retest],
    rng_for: Callable[[Any], np.random.Generator],
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """The p-values once every test at ``p <= alpha`` whose family needs more shuffles than
    it ``used`` is rerun on its ``retests`` (samples: the mean p over them; a balanced
    test's ``rerun(B, rng)``: its p over ``B`` labellings), and the shuffles behind each p
    (NaN without one). ``rng_for(index)`` seeds the rerun of the test at ``index`` from its
    identity (see the module)."""
    sizes = family_sizes(p_values, family)
    values = p_values.to_numpy(dtype=float).copy()
    counts = np.where(np.isfinite(values), used.to_numpy(dtype=float), np.nan)
    for position, index in enumerate(p_values.index):
        pairs = retests.get(index)
        if not pairs or not values[position] <= alpha:
            continue
        needed = permutations_for(int(sizes[position]), int(counts[position]), alpha)
        if needed <= counts[position]:
            continue
        rng = rng_for(index)
        values[position] = float(
            pairs(needed, rng)
            if callable(pairs)
            else np.mean([cell_randomization(t, c, needed, rng)[1] for t, c in pairs])
        )
        counts[position] = needed
    return values, counts


def sparse(sizes: dict[Any, int]) -> list[Any]:
    """The repeats too small to balance to (see the module): fewer than ``SPARSE`` of the
    median repeat's observations, or fewer than ``MIN_BALANCED``. Against the median, not the
    largest: one repeat holding many more observations than the others leaves them all
    ordinary (``data/simulations/unbalanced.py``, 240/30/30 cells)."""
    typical = float(np.median(list(sizes.values())))
    return [r for r, n in sizes.items() if n < SPARSE * typical or n < MIN_BALANCED]


def normalised(
    values: dict[Any, np.ndarray], controls: dict[Any, np.ndarray]
) -> dict[Any, np.ndarray] | None:
    """Each repeat's values divided by the mean of that repeat's control; ``None`` when a
    control mean is not positive (no fold change)."""
    means = {repeat: float(np.mean(controls[repeat])) for repeat in values}
    if any(not mean > 0 for mean in means.values()):
        return None
    return {repeat: x / means[repeat] for repeat, x in values.items()}


def _tests(
    by_t: dict[Any, np.ndarray],
    by_c: dict[Any, np.ndarray],
    every_control: dict[Any, np.ndarray],
    kept: list[Any],
    scale: bool,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int, int, dict[str, Retest]]:
    """``(d, p, d balanced, p balanced, cells, control cells, retests)`` on raw values or,
    with ``scale``, normalised ones: every observation pooled, and the balanced permutation
    test over the ``kept`` repeats (none when empty), which normalises each labelling
    itself; ``retests`` holds what ``refined`` reruns, as provided (``"cells"``) and
    balanced (its ``rerun``)."""
    pooled_sides = (by_t, every_control)
    if scale:
        pooled_sides = normalised(by_t, every_control), normalised(every_control, every_control)
    treated = np.concatenate(list(pooled_sides[0].values()))
    control = np.concatenate(list(pooled_sides[1].values()))
    d, p = cell_randomization(treated, control, n_permutations, rng)
    per_t = min(len(x) for x in by_t.values())
    per_c = min(len(x) for x in by_c.values())
    d_bal = p_bal = np.nan
    rerun: Retest = []
    if kept:
        d_bal, p_bal, per_t, per_c, rerun = balanced_draws(
            [by_t[r] for r in kept], [by_c[r] for r in kept], rng, scale
        )
    return d, p, d_bal, p_bal, per_t, per_c, {"cells": [(treated, control)], "balanced": rerun}


def trusted_test(
    by_t: dict[Any, np.ndarray],
    by_c: dict[Any, np.ndarray],
    every: dict[Any, np.ndarray],
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The cell-level test in the version the readings trust most (``reading_text``): on
    values normalised to each repeat's control when every control mean is positive, and
    balanced over the repeats that are not sparse when the condition is unbalanced (as
    ``cell_table`` decides it); otherwise pooled as provided, unless that pools the control
    of repeats the condition was not imaged in: then no test (``d`` and ``p`` NaN). ``d``,
    ``p``, whether the values were ``normalised`` and the test ``balanced``, and the
    observations it used."""
    sizes = {r: len(x) for r, x in by_t.items()}
    control_sizes = {r: len(x) for r, x in by_c.items()}
    spread = max(max(n.values()) / max(min(n.values()), 1) for n in (sizes, control_sizes))
    unbalanced = bool(spread > IMBALANCE or len(every) > len(control_sizes))
    small = set(sparse(sizes)) | set(sparse(control_sizes))
    kept = [r for r in by_t if r not in small] if unbalanced else []
    scaled = normalised(by_t, every), normalised(by_c, every), normalised(every, every)
    scale = all(x is not None for x in scaled)
    d, p, d_bal, p_bal, per_t, per_c, retests = _tests(
        by_t, by_c, every, kept, scale, n_permutations, rng
    )
    balanced = bool(kept) and bool(np.isfinite(p_bal))
    if not balanced and len(every) > len(control_sizes):
        d = p = np.nan
    return {
        "d": d_bal if balanced else d,
        "p": p_bal if balanced else p,
        "normalised": scale,
        "balanced": balanced,
        "observations": per_t * len(kept) if balanced else sum(sizes.values()),
        "control_observations": (
            per_c * len(kept) if balanced else sum(len(x) for x in every.values())
        ),
        # What refined reruns, and the shuffles behind p (the labellings, when balanced).
        "retest": retests["balanced" if balanced else "cells"],
        "permutations": BALANCED_PERMUTATIONS if balanced else n_permutations,
    }


def cell_table(
    pooled: dict[Any, dict[str, Any]],
    plan: pd.DataFrame,
    n_permutations: int,
    rng_for: Any,
) -> pd.DataFrame:
    """One row per condition: the cell-level test pooled as provided, the balance check and,
    when unbalanced, the balanced test over the repeats that are not sparse, each on raw and
    normalised
    values; and the chance of p <= 0.05 without an effect. ``pooled`` maps a condition to
    its observations and the control's per repeat it was imaged in (``treated_by_repeat``,
    ``control_by_repeat``: repeat -> values), every control observation of the dataset by
    repeat (``control_all``) and the control label; ``plan`` gives the variation between
    repeats. ``rng_for(condition, values)`` gives the generator of each set of values
    (``"raw"`` or ``"normalised"``; with `` refined`` and the version, the rerun of a small
    p, ``refined``)."""
    between = plan.set_index("condition")["between_sd"] if len(plan) else pd.Series(dtype=float)
    rows = []
    # What refined may rerun, per p column: only the tests at p <= ALPHA keep theirs.
    retests: dict[str, dict[Any, Retest]] = {column: {} for column in _P_COLUMNS}
    for condition, side in pooled.items():
        by_t, by_c, every = (
            side["treated_by_repeat"],
            side["control_by_repeat"],
            side["control_all"],
        )
        sizes = {r: len(x) for r, x in by_t.items()}
        control_sizes = {r: len(x) for r, x in by_c.items()}
        spread = max(max(n.values()) / max(min(n.values()), 1) for n in (sizes, control_sizes))
        others = len(every) - len(control_sizes)
        # How far apart the repeats' control means are: pooled, they mix those levels.
        control_means = [float(np.mean(x)) for x in every.values() if len(x)]
        shift = (
            max(control_means) / min(control_means)
            if control_means and min(control_means) > 0
            else np.nan
        )
        unbalanced = bool(spread > IMBALANCE or others > 0)
        small = sorted(set(sparse(sizes)) | set(sparse(control_sizes)), key=str)
        kept = [r for r in by_t if r not in small] if unbalanced else []
        d, p, d_bal, p_bal, per_t, per_c, raw = _tests(
            by_t, by_c, every, kept, False, n_permutations, rng_for(condition, "raw")
        )
        scaled = normalised(by_t, every), normalised(by_c, every), normalised(every, every)
        d_n = p_n = d_bal_n = p_bal_n = np.nan
        scaled_retests: dict[str, Retest] = {}
        if all(x is not None for x in scaled):
            d_n, p_n, d_bal_n, p_bal_n, _, _, scaled_retests = _tests(
                by_t, by_c, every, kept, True, n_permutations, rng_for(condition, "normalised")
            )
        for column, value, pairs in (
            ("p_cells", p, raw["cells"]),
            ("p_balanced", p_bal, raw["balanced"]),
            ("p_cells_normalised", p_n, scaled_retests.get("cells", [])),
            ("p_balanced_normalised", p_bal_n, scaled_retests.get("balanced", [])),
        ):
            if value <= ALPHA:
                retests[column][condition] = pairs
        s = float(between.get(condition, np.nan)) if condition in between.index else np.nan
        per_well = float(
            np.mean(
                [
                    effective_per_well(list(sizes.values())),
                    effective_per_well(list(control_sizes.values())),
                ]
            )
        )
        low, high = _chances(per_well, s)
        low_bal = high_bal = np.nan
        if np.isfinite(p_bal):
            low_bal, high_bal = _chances(float(np.mean([per_t, per_c])), s)
        rows.append(
            {
                "condition": condition,
                "control_condition": side["control_condition"],
                "observations": sum(sizes.values()),
                "control_observations": sum(len(x) for x in every.values()),
                "d_cells": d,
                "p_cells": p,
                "d_cells_normalised": d_n,
                "p_cells_normalised": p_n,
                "repeats": len(sizes),
                "spread": float(spread),
                "other_control_repeats": others,
                "control_shift": shift,
                "unbalanced": unbalanced,
                "sparse_repeats": small,
                "balanced_repeats": len(kept),
                "balanced_cells": per_t,
                "balanced_control_cells": per_c,
                "d_balanced": d_bal,
                "p_balanced": p_bal,
                "d_balanced_normalised": d_bal_n,
                "p_balanced_normalised": p_bal_n,
                "per_well": per_well,
                "between_sd": s,
                "chance_low": low,
                "chance_high": high,
                "chance_balanced_low": low_bal,
                "chance_balanced_high": high_bal,
            }
        )
    table = pd.DataFrame(rows, columns=CELL_COLUMNS)
    family = experiments(
        {condition: list(side["treated_by_repeat"]) for condition, side in pooled.items()}
    )
    table["experiment"] = table["condition"].map(family)
    table = table.set_index("condition", drop=False)
    # A test as provided whose control pools repeats the condition lacks is withheld, so it
    # leaves the correction family: q only over the tests reported (owner, 2026-10-05).
    withheld = table["other_control_repeats"] > 0
    for column in _P_COLUMNS:
        balanced = "balanced" in column
        used = BALANCED_PERMUTATIONS if balanced else n_permutations
        values = "normalised" if column.endswith("normalised") else "raw"
        version = "balanced" if balanced else "as provided"
        out = withheld if not balanced else pd.Series(False, index=table.index)
        p_values, shuffles = refined(
            table[column].where(~out),
            table["experiment"],
            pd.Series(used, index=table.index),
            retests[column],
            lambda condition, v=values, w=version: rng_for(condition, f"{v} refined {w}"),
        )
        name = "balanced_permutations" if balanced else "permutations"
        table[name] = np.fmax(table[name].to_numpy(dtype=float), shuffles)
        table[column.replace("p_", "q_", 1)] = corrected(
            pd.Series(p_values, index=table.index), table["experiment"]
        )
        # A withheld test keeps its p, for inspection only: it has no q.
        table[column] = np.where(out, table[column], p_values)
    return table.reset_index(drop=True)


__all__ = [
    "ALPHA",
    "BALANCED_DRAWS",
    "BALANCED_PERMUTATIONS",
    "CELL_COLUMNS",
    "IMBALANCE",
    "MIN_BALANCED",
    "SPARSE",
    "TYPICAL_BETWEEN",
    "balanced_draws",
    "balanced_test",
    "cell_randomization",
    "cell_table",
    "chance_without_effect",
    "effective_per_well",
    "normalised",
    "permutations_for",
    "refined",
    "sparse",
    "trusted_test",
]
