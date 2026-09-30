"""Planning the next experiment at the design the user ran: what to change within a repeat.

A dataset is one experiment, pooled as provided, and its number of biological
repeats is the design the user runs. The plan says what can be changed within
that design, per condition against the control, from the per-repeat contrasts
of ``condition_contrast`` (their Cohen's d and its bootstrap interval from the
fields resampled as acquired, ``contrast_sampling``):

- **the fields a side per repeat**: the most any repeat needs to resolve the
  difference within it, projected from its interval (fewer than acquired
  where it is resolved, more where it is not); ``inf`` when some repeat's
  difference is too small to settle;
- **what limits it**: the sampling error within a repeat (the standard error
  of each repeat's d, from its interval, which shrinks as ``1 / sqrt(fields)``)
  against the variation between repeats (the per-repeat d's spread beyond that
  sampling error, which more fields cannot reduce; it needs two repeats).
  When the repeats vary more than their sampling error, more fields per repeat
  help little;
- in the full report only, **the repeats needed**: the fewest biological
  repeats (at most ``MAX_REPEATS``) for which a paired t-test on the repeat
  values reaches ``power`` at ``alpha`` for this condition (not corrected
  for the number of conditions: planning is per knockdown), with the fields
  acquired and with twice as many.

With one repeat the variation between repeats is unknown: the plan gives the
fields a side and says so. The variation between repeats estimated from two
or three repeats is itself uncertain, so the numbers are a guide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .contrast_sampling import PRACTICAL

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
    "fields_per_repeat",
    "limited_by",
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


def _fields_per_repeat(block: pd.DataFrame) -> float:
    """The most fields a side any repeat needs to resolve its difference: the fields at which
    its interval's half-width, shrinking as ``1 / sqrt(fields)``, falls below |d| (fewer than
    acquired where it is resolved); ``inf`` when a repeat's difference is too small to settle
    with ``PRACTICAL`` times its fields."""
    needed = []
    for row in block.to_dict("records"):
        fields = float(min(row["fields"], row["control_fields"]))
        half, d = (row["d_high"] - row["d_low"]) / 2, abs(row["d_objects"])
        projected = fields * (half / d) ** 2 if d else np.inf
        needed.append(np.inf if projected > PRACTICAL * fields else max(2.0, np.ceil(projected)))
    return max(needed) if needed else np.nan


def repeat_plan(table: pd.DataFrame, alpha: float = ALPHA, power: float = POWER) -> pd.DataFrame:
    """One row per condition and control: the plan for the next experiment (see the module).

    ``table`` is ``condition_contrast``'s per-repeat table (with ``d_low``/``d_high``).
    """
    if table.empty or "d_low" not in table:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    usable = table[table["d_low"].notna() & table["d_objects"].notna()]
    groups = list(usable.groupby(["condition", "control_condition"], sort=False))
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
            "fields_per_repeat": _fields_per_repeat(block),
            "limited_by": "",
            "repeats_needed": np.nan,
            "repeats_needed_twice_fields": np.nan,
            "status": "ok",
        }
        if len(d) < 2:
            row["status"] = "one biological repeat: the variation between repeats is unknown"
        else:
            between = max(0.0, float(np.var(d, ddof=1)) - within)
            row["between_sd"] = float(np.sqrt(between))
            row["limited_by"] = "between repeats" if between > within else "within repeats"
            for column, factor in (("repeats_needed", 1), ("repeats_needed_twice_fields", 2)):
                sd = float(np.sqrt(between + within / factor))
                row[column] = repeats_needed(row["d_mean"], sd, alpha, power)
        rows.append(row)
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


__all__ = [
    "ALPHA",
    "MAX_REPEATS",
    "PLAN_COLUMNS",
    "POWER",
    "paired_power",
    "repeat_plan",
    "repeats_needed",
]
