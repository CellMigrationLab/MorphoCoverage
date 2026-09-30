"""Planning the next experiment from a pilot: how many biological repeats, and do more fields help?

A pilot of one or two biological repeats cannot test a difference across
repeats, but it can plan the experiment that will. Per condition against the
control, from the per-repeat contrasts of ``condition_contrast`` (their
Cohen's d and its bootstrap interval from the fields resampled as acquired,
``contrast_sampling``):

- **the effect**: the mean of the per-repeat d;
- **the sampling error within a repeat**: the standard error of each repeat's
  d, from its interval (half its width over 1.96), which shrinks as the fields
  a side grow (``1 / sqrt(fields)``);
- **the variation between repeats**: the variance of the per-repeat d beyond
  that sampling error (0 when the repeats agree within it), which more fields
  cannot reduce. It needs at least two repeats.

From these, **the repeats needed**: the fewest biological repeats (at most
``MAX_REPEATS``) for which a paired t-test on the repeat values reaches
``power`` at ``alpha`` Bonferroni-corrected over the conditions planned,
with the fields a side acquired and with twice as many. When twice the fields
barely change the repeats needed, the variation between repeats dominates.
With one repeat the variation between repeats is unknown: the plan says so,
and question 2 still gives the fields that would resolve the difference within
a repeat. The variation between repeats estimated from two or three repeats
is itself uncertain, so the numbers are a guide, and the summary says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..metadata import REPEAT_COLUMN

ALPHA = 0.05
POWER = 0.8
MAX_REPEATS = 30
_Z = float(stats.norm.ppf(0.975))
PLAN_COLUMNS = [
    "condition",
    "control_condition",
    "repeats",
    "d_mean",
    "between_sd",
    "within_se",
    "repeats_needed",
    "repeats_needed_twice_fields",
    "status",
]


def paired_power(delta: float, sd: float, repeats: int, alpha: float) -> float:
    """The power of a two-sided paired t-test on ``repeats`` differences of mean ``delta`` and
    standard deviation ``sd``."""
    if repeats < 2 or sd <= 0:
        return 0.0
    df = repeats - 1
    critical = float(stats.t.ppf(1 - alpha / 2, df))
    shift = delta * np.sqrt(repeats) / sd
    power = float(stats.nct.sf(critical, df, shift) + stats.nct.cdf(-critical, df, shift))
    if np.isnan(power):
        # scipy's noncentral t can fail far in its tail: the shifted central t is close there.
        power = float(stats.t.sf(critical - shift, df) + stats.t.cdf(-critical - shift, df))
    return power


def repeats_needed(delta: float, sd: float, alpha: float, power: float = POWER) -> float:
    """The fewest repeats (2 to ``MAX_REPEATS``) reaching ``power``; ``inf`` beyond."""
    for repeats in range(2, MAX_REPEATS + 1):
        if paired_power(abs(delta), sd, repeats, alpha) >= power:
            return float(repeats)
    return np.inf


def repeat_plan(table: pd.DataFrame, alpha: float = ALPHA, power: float = POWER) -> pd.DataFrame:
    """One row per condition and control: the plan for the next experiment (see the module).

    ``table`` is ``condition_contrast``'s per-repeat table (with ``d_low``/``d_high``).
    """
    if table.empty or "d_low" not in table:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    usable = table[table["d_low"].notna() & table["d_objects"].notna()]
    groups = list(usable.groupby(["condition", "control_condition"], sort=False))
    planned = sum(1 for _, block in groups if len(block) >= 2)
    corrected = alpha / max(planned, 1)
    rows = []
    for (condition, control), block in groups:
        d = block["d_objects"].to_numpy(float)
        se = ((block["d_high"] - block["d_low"]) / (2 * _Z)).to_numpy(float)
        within = float(np.mean(se**2))
        row: dict[str, Any] = {
            "condition": condition,
            "control_condition": control,
            "repeats": len(d),
            "d_mean": float(d.mean()),
            "between_sd": np.nan,
            "within_se": float(np.sqrt(within)),
            "repeats_needed": np.nan,
            "repeats_needed_twice_fields": np.nan,
            "status": "ok",
        }
        if len(d) < 2:
            row["status"] = "one biological repeat: the variation between repeats is unknown"
        else:
            between = max(0.0, float(np.var(d, ddof=1)) - within)
            row["between_sd"] = float(np.sqrt(between))
            for column, factor in (("repeats_needed", 1), ("repeats_needed_twice_fields", 2)):
                sd = float(np.sqrt(between + within / factor))
                row[column] = repeats_needed(row["d_mean"], sd, corrected, power)
        rows.append(row)
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def plan_settings(table: pd.DataFrame, alpha: float = ALPHA, power: float = POWER) -> dict:
    """The plan's settings, for its sentence: power, alpha and the conditions it corrects over."""
    planned = int((table["status"] == "ok").sum()) if len(table) else 0
    return {"alpha": alpha, "power": power, "planned": planned, "repeat_col": REPEAT_COLUMN}


__all__ = [
    "ALPHA",
    "MAX_REPEATS",
    "PLAN_COLUMNS",
    "POWER",
    "paired_power",
    "plan_settings",
    "repeat_plan",
    "repeats_needed",
]
