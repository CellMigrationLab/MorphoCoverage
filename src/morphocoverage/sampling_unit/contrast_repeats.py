"""Each condition against the control across biological repeats, the repeat as the replicate.

``condition_contrast`` compares a condition with the control within each
biological repeat, and gives a test there only when each side holds at least
two technical repeats. Most experiments have one well or dish per condition
and repeat: within a repeat there is then no test, but the design still
answers the question it was built for, with the biological repeat as the
replicate (as SuperPlots do, Lord et al. 2020). ``across_repeats`` gives, per
condition and control, from the per-repeat contrasts:

- **the direction in each repeat**: how many repeats have the condition's
  mean below the control's of the same repeat, and how many above;
- **the per-repeat Cohen's d** on the observations: their mean and range;
- **a paired t-test** on the per-repeat differences of means (treated minus
  control, paired by repeat), the usual SuperPlots test; it assumes the
  differences are normally distributed;
- **an exact sign-flip test** on the same differences, which assumes only that
  their sign is exchangeable without an effect: every one of the ``2 ** n``
  sign assignments is enumerated, so p is exact and never below ``2 / 2 ** n``.
  It is run only when that floor, after the Bonferroni correction, can reach
  0.05 (from 6 repeats for one condition); below, it could not help and is
  left out (``p_sign_flip`` NaN, its floor kept).

Both p-values are Bonferroni-corrected over the conditions tested; a
condition needs at least ``MIN_REPEATS`` repeats holding both sides for either
test. A repeat's value is the mean of all its observations of the condition
(the SuperPlots replicate mean), since within a repeat the fields are not the
replicate. Like the per-repeat contrast, this compares means.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._seeds import group_rng
from ._reach import reachable

# The fewest biological repeats holding both sides for a test across repeats.
MIN_REPEATS = 3
# Sign assignments enumerated at most (2 ** 16): beyond, the repeats are many and the
# assignments are drawn at random.
_EXACT_REPEATS = 16
REPEAT_COLUMNS = [
    "condition",
    "control_condition",
    "repeats",
    "lower",
    "higher",
    "mean_difference",
    "d_mean",
    "d_min",
    "d_max",
    "p_sign_flip",
    "p_sign_flip_bonferroni",
    "p_sign_flip_floor",
    "p_sign_flip_exact",
    "p_paired_t",
    "p_paired_t_bonferroni",
    "status",
]


def sign_flip_test(
    differences: np.ndarray, rng: np.random.Generator, n_draws: int = 20000
) -> tuple[float, float, bool]:
    """``(p, floor, exact)``: the two-sided sign-flip test of paired differences.

    Every sign assignment is enumerated up to ``2 ** 16`` of them (p exact, never below
    ``floor = 2 / 2 ** n`` when no difference is 0); beyond, ``n_draws`` random
    assignments, p as ``(extreme + 1) / (n_draws + 1)``.
    """
    n = len(differences)
    observed = abs(differences.mean())
    tolerance = 1e-12 * max(1.0, float(np.abs(differences).max(initial=0.0)))
    if n <= _EXACT_REPEATS:
        signs = np.array(list(product((1.0, -1.0), repeat=n)))
        flipped = np.abs((signs * differences).mean(axis=1))
        p = float(np.mean(flipped >= observed - tolerance))
        return p, 2.0 / 2**n, True
    signs = rng.choice((1.0, -1.0), size=(n_draws, n))
    extreme = int((np.abs((signs * differences).mean(axis=1)) >= observed - tolerance).sum())
    return (extreme + 1) / (n_draws + 1), 1.0 / (n_draws + 1), False


def _paired_t(differences: np.ndarray) -> float:
    """The two-sided paired t-test of the differences; NaN when they do not vary."""
    if np.ptp(differences) == 0:
        return np.nan
    return float(stats.ttest_1samp(differences, 0.0).pvalue)


def across_repeats(table: pd.DataFrame, repeat_col: str | None, random_state: int) -> pd.DataFrame:
    """One row per condition and control across the repeats of the per-repeat ``table``
    (``condition_contrast``'s, with ``mean`` and ``control_mean``); see the module.

    Empty when the repeats are not kept apart or no condition is compared in two repeats.
    """
    if repeat_col is None or table.empty or "mean" not in table:
        return pd.DataFrame(columns=REPEAT_COLUMNS)
    paired = table[table["mean"].notna() & table["control_mean"].notna()]
    rows = []
    for (condition, control), block in paired.groupby(
        ["condition", "control_condition"], sort=False
    ):
        differences = (block["mean"] - block["control_mean"]).to_numpy(float)
        d = block["d_objects"].to_numpy(float)
        d = d[np.isfinite(d)]
        row: dict[str, Any] = {
            "condition": condition,
            "control_condition": control,
            "repeats": len(differences),
            "lower": int((differences < 0).sum()),
            "higher": int((differences > 0).sum()),
            "mean_difference": float(differences.mean()),
            "d_mean": float(d.mean()) if len(d) else np.nan,
            "d_min": float(d.min()) if len(d) else np.nan,
            "d_max": float(d.max()) if len(d) else np.nan,
            "p_sign_flip": np.nan,
            "p_sign_flip_bonferroni": np.nan,
            "p_sign_flip_floor": np.nan,
            "p_sign_flip_exact": False,
            "p_paired_t": np.nan,
            "p_paired_t_bonferroni": np.nan,
            "status": "ok",
        }
        if len(differences) < MIN_REPEATS:
            row["status"] = (
                f"fewer than {MIN_REPEATS} biological repeats hold both the condition and the "
                "control"
            )
        else:
            rng = group_rng(random_state, {"condition": condition, "control": control})
            p, floor, exact = sign_flip_test(differences, rng)
            row.update(
                p_sign_flip=p,
                p_sign_flip_floor=floor,
                p_sign_flip_exact=exact,
                p_paired_t=_paired_t(differences),
            )
        rows.append(row)
    result = pd.DataFrame(rows, columns=REPEAT_COLUMNS)
    if result.empty or result["repeats"].max() < 2:
        return pd.DataFrame(columns=REPEAT_COLUMNS)
    unhelpful = result["p_sign_flip"].notna() & ~reachable(
        result["p_sign_flip_floor"].where(result["p_sign_flip"].notna())
    )
    result.loc[unhelpful, "p_sign_flip"] = np.nan
    for column in ("p_sign_flip", "p_paired_t"):
        tested = result[column].notna()
        result[f"{column}_bonferroni"] = np.where(
            tested, np.minimum(1.0, result[column] * int(tested.sum())), np.nan
        )
    return result


__all__ = ["MIN_REPEATS", "REPEAT_COLUMNS", "across_repeats", "sign_flip_test"]
