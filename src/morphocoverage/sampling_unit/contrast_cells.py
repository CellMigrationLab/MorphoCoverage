"""The cell-level randomization test (Plot&Stats), with the chance it is positive without an effect.

The lab's Plot&Stats test compares every observation of a condition with every
observation of the control, pooled as provided (all repeats together): Cohen's
d on the observations, with a randomization test that shuffles the condition
labels over the observations (``n_permutations``, ``(extreme + 1) / (n + 1)``),
Bonferroni-corrected over the conditions. It asks whether these wells differ,
and it is the only test that responds with one or two biological repeats.

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


def cell_table(
    pooled: dict[Any, dict[str, Any]],
    plan: pd.DataFrame,
    n_permutations: int,
    rng_for: Any,
) -> pd.DataFrame:
    """One row per condition: the cell-level test and the chance of p <= 0.05 without an
    effect. ``pooled`` maps a condition to its observations, the control's, the control
    label and its observations per well; ``plan`` gives the variation between repeats."""
    between = plan.set_index("condition")["between_sd"] if len(plan) else pd.Series(dtype=float)
    rows = []
    for condition, side in pooled.items():
        treated, control = side["treated"], side["control"]
        d, p = cell_randomization(treated, control, n_permutations, rng_for(condition))
        s = float(between.get(condition, np.nan)) if condition in between.index else np.nan
        per_well = float(side["per_well"])
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
                "per_well": per_well,
                "between_sd": s,
                "chance_low": low,
                "chance_high": high,
            }
        )
    table = pd.DataFrame(rows, columns=CELL_COLUMNS)
    tested = table["p_cells"].notna()
    table["p_cells_bonferroni"] = np.where(
        tested, np.minimum(1.0, table["p_cells"] * int(tested.sum())), np.nan
    )
    return table


__all__ = [
    "ALPHA",
    "CELL_COLUMNS",
    "TYPICAL_BETWEEN",
    "cell_randomization",
    "cell_table",
    "chance_without_effect",
]
