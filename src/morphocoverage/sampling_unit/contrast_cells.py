"""The cell-level randomization test (Plot&Stats), balanced when the repeats are not.

The lab's Plot&Stats test compares every observation of a condition with every
observation of the control, pooled as provided (all repeats together): Cohen's
d on the observations, with a randomization test that shuffles the condition
labels over the observations (``n_permutations``, ``(extreme + 1) / (n + 1)``),
Bonferroni-corrected over the conditions. It asks whether these wells differ,
and it is the only test that responds with one or two biological repeats.

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
median p over the draws (Bonferroni-corrected over the conditions). Across the
draws every observation is used.

Balancing down to a repeat that holds very few observations would leave little
of the others in each draw, so a repeat holding fewer than ``SPARSE`` (a
quarter) of the largest repeat on its side, or fewer than ``MIN_BALANCED``
observations, is *sparse*: it is named in ``sparse_repeats`` and the condition
is not balanced (the test as provided only), since a repeat that small is
worth looking at before any test.

It treats observations as independent draws. When the difference varies
between repeats by ``s`` observation SDs (the variation between repeats that
``repeat_plan`` estimates, ``between_sd``), the variance of the mean difference
is ``k = 1 + n s^2 / 2`` times what the test assumes (``n`` observations per
well), so without an effect it gives p <= alpha in about
``2 * P(Z > z_alpha/2 / sqrt(k))`` of experiments (``chance_without_effect``).
In ``data/simulations/test_choice.py`` this matched the simulated rate of
the test on values normalised to each repeat's control within a few points
(16% and 37% at s = 0.25 and 0.5 with 30 cells, against 15-17% and 32-36%);
on raw values the rate is at or below it (shifts that both wells of a repeat
share add noise that holds it down), so there it is an upper bound. With one
repeat, ``s`` is unknown and the chance is given for ``TYPICAL_BETWEEN``
(0.25-0.5, the range in the lab datasets). Fields holding
several observations add to it (their own design effect), which this does not
count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..statistics import cohen_d

ALPHA = 0.05
# Balanced resamples per condition, and the shuffles of the randomization test in each.
BALANCED_DRAWS = 100
BALANCED_PERMUTATIONS = 999
# Most over fewest observations per repeat beyond which a side is unbalanced (25% more).
IMBALANCE = 1.25
# A repeat holding less than this share of the largest repeat on its side, or fewer than
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
    "p_cells_bonferroni",
    "repeats",
    "spread",
    "other_control_repeats",
    "unbalanced",
    "sparse_repeats",
    "balanced_cells",
    "balanced_control_cells",
    "d_balanced",
    "p_balanced",
    "p_balanced_bonferroni",
    "per_well",
    "between_sd",
    "chance_low",
    "chance_high",
]


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


def balanced_test(
    treated: list[np.ndarray], control: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, float, int, int]:
    """``(median d, median p, cells, control cells)`` over ``BALANCED_DRAWS`` draws taking the
    same number of observations from every repeat (the fewest any repeat holds); NaN when a
    repeat holds fewer than 2 on a side."""
    per_t, per_c = min(len(t) for t in treated), min(len(c) for c in control)
    if per_t < 2 or per_c < 2:
        return np.nan, np.nan, per_t, per_c
    ds, ps = [], []
    for _ in range(BALANCED_DRAWS):
        t = np.concatenate([rng.choice(x, per_t, replace=False) for x in treated])
        c = np.concatenate([rng.choice(x, per_c, replace=False) for x in control])
        d, p = cell_randomization(t, c, BALANCED_PERMUTATIONS, rng)
        ds.append(d)
        ps.append(p)
    return float(np.nanmedian(ds)), float(np.nanmedian(ps)), per_t, per_c


def sparse(sizes: dict[Any, int]) -> list[Any]:
    """The repeats too small to balance to (see the module): fewer than ``SPARSE`` of the
    largest repeat's observations, or fewer than ``MIN_BALANCED``."""
    largest = max(sizes.values())
    return [r for r, n in sizes.items() if n < SPARSE * largest or n < MIN_BALANCED]


def cell_table(
    pooled: dict[Any, dict[str, Any]],
    plan: pd.DataFrame,
    n_permutations: int,
    rng_for: Any,
) -> pd.DataFrame:
    """One row per condition: the cell-level test pooled as provided, the balance check and,
    when unbalanced without a sparse repeat, the balanced test; and the chance of p <= 0.05
    without an effect. ``pooled`` maps a condition to its observations and the control's per
    repeat it was imaged in (``treated_by_repeat``, ``control_by_repeat``: repeat -> values),
    every control observation of the dataset (``control``), the number of repeats holding it
    (``control_repeats``) and the control label; ``plan`` gives the variation between
    repeats."""
    between = plan.set_index("condition")["between_sd"] if len(plan) else pd.Series(dtype=float)
    rows = []
    for condition, side in pooled.items():
        by_t, by_c = side["treated_by_repeat"], side["control_by_repeat"]
        treated, control = np.concatenate(list(by_t.values())), side["control"]
        rng = rng_for(condition)
        d, p = cell_randomization(treated, control, n_permutations, rng)
        sizes = {r: len(x) for r, x in by_t.items()}
        control_sizes = {r: len(x) for r, x in by_c.items()}
        spread = max(max(n.values()) / max(min(n.values()), 1) for n in (sizes, control_sizes))
        others = int(side["control_repeats"]) - len(control_sizes)
        unbalanced = bool(spread > IMBALANCE or others > 0)
        small = sorted(set(sparse(sizes)) | set(sparse(control_sizes)), key=str)
        d_bal = p_bal = np.nan
        per_t, per_c = min(sizes.values()), min(control_sizes.values())
        if unbalanced and not small:
            d_bal, p_bal, per_t, per_c = balanced_test(
                list(by_t.values()), list(by_c.values()), rng
            )
        s = float(between.get(condition, np.nan)) if condition in between.index else np.nan
        per_well = float(
            min(np.median(list(sizes.values())), np.median(list(control_sizes.values())))
        )
        if np.isfinite(s):
            low = high = chance_without_effect(per_well, s)
        else:
            low, high = (chance_without_effect(per_well, value) for value in TYPICAL_BETWEEN)
        rows.append(
            {
                "condition": condition,
                "control_condition": side["control_condition"],
                "observations": len(treated),
                "control_observations": len(control),
                "d_cells": d,
                "p_cells": p,
                "p_cells_bonferroni": np.nan,
                "repeats": len(sizes),
                "spread": float(spread),
                "other_control_repeats": others,
                "unbalanced": unbalanced,
                "sparse_repeats": small,
                "balanced_cells": per_t,
                "balanced_control_cells": per_c,
                "d_balanced": d_bal,
                "p_balanced": p_bal,
                "p_balanced_bonferroni": np.nan,
                "per_well": per_well,
                "between_sd": s,
                "chance_low": low,
                "chance_high": high,
            }
        )
    table = pd.DataFrame(rows, columns=CELL_COLUMNS)
    for column in ("p_cells", "p_balanced"):
        tested = table[column].notna()
        table[f"{column}_bonferroni"] = np.where(
            tested, np.minimum(1.0, table[column] * int(tested.sum())), np.nan
        )
    return table


__all__ = [
    "ALPHA",
    "BALANCED_DRAWS",
    "BALANCED_PERMUTATIONS",
    "CELL_COLUMNS",
    "IMBALANCE",
    "MIN_BALANCED",
    "SPARSE",
    "TYPICAL_BETWEEN",
    "balanced_test",
    "cell_randomization",
    "cell_table",
    "chance_without_effect",
    "sparse",
]
