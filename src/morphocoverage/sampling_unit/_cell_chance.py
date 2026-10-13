"""The chance the cell-level test (``contrast_cells``) gives p <= alpha without an effect.

When the difference varies between repeats by ``s`` observation SDs, the variance of the
mean difference is ``k = 1 + n s^2 / 2`` times what the test assumes (``n`` observations per
well, the effective ``n`` when the repeats hold different numbers), so without an effect it
gives p <= alpha in about ``2 * P(Z > z_alpha/2 / sqrt(k))`` of experiments (the module
``contrast_cells`` gives the derivation and the simulations behind it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from ..statistics import ALPHA

# The variation between repeats assumed when one repeat cannot estimate it (observation SDs).
TYPICAL_BETWEEN = (0.25, 0.5)


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


def _chance_columns(
    sizes: dict[Any, int],
    control_sizes: dict[Any, int],
    between_sd: float,
    balanced_p: float,
    per_t: int,
    per_c: int,
) -> dict[str, float]:
    """The chance of p <= 0.05 without an effect, as provided and balanced (NaN without a
    balanced p), with the observations per well and the variation between repeats behind
    it: one condition's columns of ``cell_table``."""
    per_well = float(
        np.mean(
            [
                effective_per_well(list(sizes.values())),
                effective_per_well(list(control_sizes.values())),
            ]
        )
    )
    low, high = _chances(per_well, between_sd)
    low_bal = high_bal = np.nan
    if np.isfinite(balanced_p):
        low_bal, high_bal = _chances(float(np.mean([per_t, per_c])), between_sd)
    return {
        "per_well": per_well,
        "between_sd": between_sd,
        "chance_low": low,
        "chance_high": high,
        "chance_balanced_low": low_bal,
        "chance_balanced_high": high_bal,
    }
