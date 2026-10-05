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
control of its own repeats only, each draw taking the same number of
observations from every one of those repeats (the fewest any repeat holds, the
condition's and the control's separately), so every repeat weighs the same;
``BALANCED_DRAWS`` draws, the randomization test on each, and the median d and
median p over the draws (corrected the same way). Across the
draws every observation is used.

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
balanced test's 999 shuffles per draw, three features at that floor got q 0.07, never 0.05,
whatever their effect. Every test whose p is at most ``ALPHA`` after the first pass is
therefore rerun with ``B = max(default, ceil(K / ALPHA))`` shuffles (``permutations_for``;
``K`` the tests of its sub-experiment), so that ``K / (B + 1) < ALPHA``. Only those: the
threshold of rank ``r`` is ``ALPHA r / K``, at most ``ALPHA``, so a test above it cannot
pass at any rank, and a smaller p moving never changes the q of a larger one (a q depends
only on the p-values at or above it). The rerun is seeded from the test's own identity, so
it does not depend on the order of the tests. The balanced test is rerun on its median
draws only (``balanced_draws``): the draws at the median ``|d|`` (two for an even number of
draws), whose mean p replaces the median p. Every draw holds the same numbers of
observations, so the randomization p falls with ``|d|`` and the median-``|d|`` draws carry
the median p, up to the Monte Carlo error of the first pass; rerunning all
``BALANCED_DRAWS`` draws would cost a hundred times as much (``refined``).

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
chance at the observations per repeat it keeps (``chance_balanced_*``); the
median p over its draws is at or below it (12-14% against 16% at s = 0.25).
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
# A treated and a control sample a randomization p can be rerun on.
Pair = tuple[np.ndarray, np.ndarray]
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


def balanced_draws(
    treated: list[np.ndarray], control: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, float, int, int, list[Pair]]:
    """``balanced_test`` and its draws at the median ``|d|`` (two for an even number of
    draws), which ``refined`` reruns with more shuffles (see the module)."""
    per_t, per_c = min(len(t) for t in treated), min(len(c) for c in control)
    if per_t < 2 or per_c < 2:
        return np.nan, np.nan, per_t, per_c, []
    ds, ps, draws = [], [], []
    for _ in range(BALANCED_DRAWS):
        t = np.concatenate([rng.choice(x, per_t, replace=False) for x in treated])
        c = np.concatenate([rng.choice(x, per_c, replace=False) for x in control])
        d, p = cell_randomization(t, c, BALANCED_PERMUTATIONS, rng)
        ds.append(d)
        ps.append(p)
        draws.append((t, c))
    sizes = np.abs(np.asarray(ds, dtype=float))
    order = [int(i) for i in np.argsort(sizes, kind="stable") if np.isfinite(sizes[i])]
    middle = sorted({order[(len(order) - 1) // 2], order[len(order) // 2]}) if order else []
    return (
        float(np.nanmedian(ds)),
        float(np.nanmedian(ps)),
        per_t,
        per_c,
        [draws[i] for i in middle],
    )


def balanced_test(
    treated: list[np.ndarray], control: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, float, int, int]:
    """``(median d, median p, cells, control cells)`` over ``BALANCED_DRAWS`` draws taking the
    same number of observations from every repeat (the fewest any repeat holds); NaN when a
    repeat holds fewer than 2 on a side."""
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
    retests: Mapping[Any, list[Pair]],
    rng_for: Callable[[Any], np.random.Generator],
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """The p-values once every test at ``p <= alpha`` whose family needs more shuffles than
    it ``used`` is rerun on its ``retests`` (the mean p over them), and the shuffles behind
    each p (NaN without one). ``rng_for(index)`` seeds the rerun of the test at ``index``
    from its identity (see the module)."""
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
            np.mean([cell_randomization(t, c, needed, rng)[1] for t, c in pairs])
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
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int, int, dict[str, list[Pair]]]:
    """``(d, p, d balanced, p balanced, cells, control cells, retests)`` on one set of
    values: every observation pooled, and the balanced draws over the ``kept`` repeats (none
    when empty); ``retests`` holds what ``refined`` reruns, as provided (``"cells"``) and
    balanced (its median draws)."""
    treated = np.concatenate(list(by_t.values()))
    control = np.concatenate(list(every_control.values()))
    d, p = cell_randomization(treated, control, n_permutations, rng)
    per_t = min(len(x) for x in by_t.values())
    per_c = min(len(x) for x in by_c.values())
    d_bal = p_bal = np.nan
    middle: list[Pair] = []
    if kept:
        d_bal, p_bal, per_t, per_c, middle = balanced_draws(
            [by_t[r] for r in kept], [by_c[r] for r in kept], rng
        )
    return d, p, d_bal, p_bal, per_t, per_c, {"cells": [(treated, control)], "balanced": middle}


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
    values = scaled if all(x is not None for x in scaled) else (by_t, by_c, every)
    d, p, d_bal, p_bal, per_t, per_c, retests = _tests(*values, kept, n_permutations, rng)
    balanced = bool(kept) and bool(np.isfinite(p_bal))
    if not balanced and len(every) > len(control_sizes):
        d = p = np.nan
    return {
        "d": d_bal if balanced else d,
        "p": p_bal if balanced else p,
        "normalised": values is scaled,
        "balanced": balanced,
        "observations": per_t * len(kept) if balanced else sum(sizes.values()),
        "control_observations": (
            per_c * len(kept) if balanced else sum(len(x) for x in every.values())
        ),
        # What refined reruns, and the shuffles behind p (each draw's, when balanced).
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
    retests: dict[str, dict[Any, list[Pair]]] = {column: {} for column in _P_COLUMNS}
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
            by_t, by_c, every, kept, n_permutations, rng_for(condition, "raw")
        )
        scaled = normalised(by_t, every), normalised(by_c, every), normalised(every, every)
        d_n = p_n = d_bal_n = p_bal_n = np.nan
        scaled_retests: dict[str, list[Pair]] = {}
        if all(x is not None for x in scaled):
            d_n, p_n, d_bal_n, p_bal_n, _, _, scaled_retests = _tests(
                *scaled, kept, n_permutations, rng_for(condition, "normalised")
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
