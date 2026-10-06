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
- **the fold change** the normalised test compares: the geometric mean of the repeats'
  ratios of the condition's mean to the control's (``fold_change``; NaN when the feature
  has a value that is not positive, the raw change being ``mean_difference``);
- **a paired t-test** on the per-repeat differences of means (treated minus
  control, paired by repeat), the usual SuperPlots test; it assumes the
  differences are normally distributed;
- **the same paired t-test on the normalised values**: the log of each
  repeat's ratio of the condition's mean to the control's (each value divided
  by the mean of its repeat's control, as "normalised to the control"), which
  tests a fold change and removes a scale both wells of a repeat share; NaN
  when a value of the feature is not positive (the scale is chosen once per
  feature from all its observations, owner 2026-10-06);
- **an exact sign-flip test** on the same differences, which assumes only that
  their sign is exchangeable without an effect: every one of the ``2 ** n``
  sign assignments is enumerated, so p is exact and never below ``2 / 2 ** n``.
  It is run only when that floor, corrected over the tests of its
  sub-experiment, can reach 0.05 (from 6 repeats for one condition); below, it
  could not help and is left out (``p_sign_flip`` NaN, its floor kept).

The p-values are corrected (Benjamini-Hochberg, the ``q_`` columns) within each
sub-experiment (``_experiments``: the conditions linked by a biological repeat
they were both compared in, numbered in ``experiment``), not over every
condition of the report: a screen's separate sub-experiments do not dilute
each other. A
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

from .._seeds import group_rng
from ..metadata import REPEAT_COLUMN as REPEAT
from ..statistics import paired_t, repeat_values
from ._experiments import corrected, experiments
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
    "fold_change",
    "d_mean",
    "d_min",
    "d_max",
    "p_sign_flip",
    "q_sign_flip",
    "p_sign_flip_floor",
    "p_sign_flip_exact",
    "p_paired_t",
    "q_paired_t",
    "p_paired_t_normalised",
    "q_paired_t_normalised",
    "experiment",
    "status",
]


def sign_flip_test(
    differences: np.ndarray, rng: np.random.Generator, n_draws: int = 20000
) -> tuple[float, float, bool]:
    """``(p, floor, exact)``: the two-sided sign-flip test of paired differences.

    Every sign assignment is enumerated up to ``2 ** 16`` of them (p exact, never below
    ``floor = 2 / 2 ** m``, ``m`` the differences that are not 0, since a 0 flips to itself);
    beyond, ``n_draws`` random assignments, p as ``(extreme + 1) / (n_draws + 1)``.
    """
    n = len(differences)
    observed = abs(differences.mean())
    # Ties in floating point, relative to the differences' own scale (an absolute floor
    # made every flip a tie for differences near 1e-12, areas in square metres).
    tolerance = 1e-12 * float(np.abs(differences).max(initial=0.0))
    if n <= _EXACT_REPEATS:
        signs = np.array(list(product((1.0, -1.0), repeat=n)))
        flipped = np.abs((signs * differences).mean(axis=1))
        p = float(np.mean(flipped >= observed - tolerance))
        return p, min(1.0, 2.0 / 2 ** int(np.count_nonzero(differences))), True
    signs = rng.choice((1.0, -1.0), size=(n_draws, n))
    extreme = int((np.abs((signs * differences).mean(axis=1)) >= observed - tolerance).sum())
    return (extreme + 1) / (n_draws + 1), 1.0 / (n_draws + 1), False


def _log_ratios(block: pd.DataFrame, log: bool | None = None) -> np.ndarray:
    """Per repeat, log(condition mean / control mean); NaN (no test) unless the feature is
    read on the log scale (``log``, ``statistics.log_scale``; ``None``: every mean
    positive), since a ratio to a value at or below 0 has no meaning as a fold change."""
    values, ratios, _ = repeat_values(block["mean"], block["control_mean"], log)
    return ratios if values == "normalised" else np.full(len(block), np.nan)


def across_repeats(
    table: pd.DataFrame, repeat_col: str | None, random_state: int, log: bool | None = None
) -> pd.DataFrame:
    """One row per condition and control across the repeats of the per-repeat ``table``
    (``condition_contrast``'s, with ``mean`` and ``control_mean``); see the module. ``log``:
    the feature's scale, from all its observations (``statistics.log_scale``), the same for
    every condition (owner, 2026-10-06); ``None`` reads it from each condition's means.

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
            "fold_change": float(np.exp(np.mean(_log_ratios(block, log)))),
            "d_mean": float(d.mean()) if len(d) else np.nan,
            "d_min": float(d.min()) if len(d) else np.nan,
            "d_max": float(d.max()) if len(d) else np.nan,
            "p_sign_flip": np.nan,
            "q_sign_flip": np.nan,
            "p_sign_flip_floor": np.nan,
            "p_sign_flip_exact": False,
            "p_paired_t": np.nan,
            "q_paired_t": np.nan,
            "p_paired_t_normalised": np.nan,
            "q_paired_t_normalised": np.nan,
            "experiment": 0,
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
                p_paired_t=paired_t(differences),
                p_paired_t_normalised=paired_t(_log_ratios(block, log)),
            )
        rows.append(row)
    result = pd.DataFrame(rows, columns=REPEAT_COLUMNS)
    if result.empty or result["repeats"].max() < 2:
        return pd.DataFrame(columns=REPEAT_COLUMNS)
    # The repeats each condition is compared in; a table without them is one experiment.
    named = next((c for c in (REPEAT, repeat_col) if c in paired), None)
    held = {
        condition: list(block[named]) if named else [0]
        for condition, block in paired.groupby("condition", sort=False)
    }
    family = experiments(held)
    result["experiment"] = result["condition"].map(family)
    floors = result["p_sign_flip_floor"].where(result["p_sign_flip"].notna())
    helpful = pd.Series(False, index=result.index)
    for members in result.groupby("experiment").groups.values():
        helpful.loc[members] = reachable(floors.loc[members])
    result.loc[result["p_sign_flip"].notna() & ~helpful, "p_sign_flip"] = np.nan
    for column in ("p_sign_flip", "p_paired_t", "p_paired_t_normalised"):
        result[column.replace("p_", "q_", 1)] = corrected(result[column], result["experiment"])
    return result


TARGET_COLUMNS = ["target", "reagents", "repeats", "agree", "direction"]


def reagent_agreement(table: pd.DataFrame, targets: dict[Any, Any]) -> pd.DataFrame:
    """Per target with two or more reagents (conditions), whether they agree: in how many of
    the repeats holding them all every reagent is on the same side of the control.

    ``targets`` maps a condition to its target (the FOV table's ``target`` column). Two
    independent reagents going the same way is evidence against an off-target effect of one;
    it is described, not tested (the reagents share the control, so they are not independent
    tests of the target).
    """
    if not targets or table.empty or "mean" not in table:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    paired = table[table["mean"].notna() & table["control_mean"].notna()]
    paired = paired.assign(target=paired["condition"].map(targets))
    rows = []
    for target, block in paired.dropna(subset=["target"]).groupby("target", sort=False):
        reagents = list(dict.fromkeys(block["condition"]))
        if len(reagents) < 2:
            continue
        signs = np.sign(block["mean"] - block["control_mean"]).groupby(block[REPEAT]).agg(list)
        complete = signs[signs.map(len) == len(reagents)]
        same = complete.map(lambda s: len(set(s)) == 1 and s[0] != 0)
        directions = {s[0] for s in complete[same]}
        direction = (
            ("lower" if directions == {-1.0} else "higher") if len(directions) == 1 else "mixed"
        )
        rows.append(
            {
                "target": target,
                "reagents": reagents,
                "repeats": len(complete),
                "agree": int(same.sum()),
                "direction": direction if int(same.sum()) else "none",
            }
        )
    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


__all__ = [
    "MIN_REPEATS",
    "REPEAT_COLUMNS",
    "TARGET_COLUMNS",
    "across_repeats",
    "reagent_agreement",
    "sign_flip_test",
]
