"""Planning the next experiment at the design the user ran: what to change within a repeat.

A dataset is one experiment, pooled as provided, and its number of biological
repeats is the design the user runs. The plan says what can be changed within
that design, per condition against the control, from the per-repeat contrasts
of ``condition_contrast`` (their Cohen's d and its bootstrap interval from the
fields resampled as acquired, ``contrast_sampling``):

- **the fields a side per repeat**: the most any repeat needs to resolve the
  difference within it, projected from the part of its interval within wells
  (``contrast_sampling.projected_fields``, owner 2026-10-05: fewer than acquired
  where it is resolved, more where it is not); ``inf`` when some repeat's
  difference is too small to settle, or its wells differ too much for more fields
  to resolve it;
- **what limits it**: the sampling error within a repeat against the variation between
  repeats (the repeat values' spread beyond that sampling error, which more fields cannot
  reduce; it needs two repeats), on the values the paired t-test compares (below). When the
  repeats vary more than their sampling error, more fields per repeat help little. The
  report no longer prints it as a verdict (owner, 2026-10-02: from 3 repeats it
  contradicted the readings, and docs/optimise/optimise.md, P10, needs 5 or more units); the
  column stays for the advice of "Did I image enough?" (``_html_sections``);
- **the repeats needed**: the fewest biological repeats (from ``MIN_REPEATS``, the paired
  test's minimum, to ``MAX_REPEATS``)
  for which a paired t-test on the repeat values reaches ``power`` at the
  correction the report uses, with the fields acquired and with twice as many (which
  halve only the part of the sampling error within wells, ``between_share``; owner,
  2026-10-05).
  The values are the test's (``contrast_repeats``): the log of each repeat's ratio of the
  condition's mean to the control's when the feature is read on the log scale (every value
  positive, ``log``; ``planned_on``
  ``"log ratio"``, the change a fold change ``change``), else the difference of the means
  (``"difference"``). Their sampling error within a repeat is the bootstrap's
  (``contrast_sampling``), which shrinks as ``1 / sqrt(fields)``; a table without it (an
  older one) is planned on the per-repeat Cohen's d (``"d"``).
  The report corrects with Benjamini-Hochberg within each sub-experiment
  (``_experiments``); the plan holds at ``alpha / m`` (``alpha_planned``), ``m`` the
  conditions of the sub-experiment, which is what Benjamini-Hochberg asks at most of
  one test, so the plan never promises too few.

``d_mean``, ``between_sd`` and ``within_se`` stay on Cohen's d (the cell-level screen's
chance of p <= 0.05 without an effect uses ``between_sd``, ``contrast_cells``).

With one repeat the variation between repeats is unknown: the plan gives the
fields a side and says so. Every repeat compared counts in ``repeats`` and ``d_mean``; the
rest rests on the repeats with an interval of d, and with fewer than two of them the
variation between repeats is not estimated (``status``). The variation between repeats
estimated from two or three repeats is itself uncertain, so the numbers are a guide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..metadata import REPEAT_COLUMN as REPEAT
from ..statistics import repeat_values
from ._experiments import experiments
from .contrast_repeats import MIN_REPEATS
from .contrast_sampling import PRACTICAL, projected_fields

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
    "planned_on",
    "change",
    "fields_per_repeat",
    "limited_by",
    "repeats_needed",
    "repeats_needed_twice_fields",
    "alpha_planned",
    "experiment",
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
    # The lower tail as the upper tail of the mirrored shift: scipy's noncentral t cdf
    # returns NaN far in its tail (scipy 1.17), while its sf stays exact there.
    upper = float(stats.nct.sf(critical, df, shift))
    lower = float(stats.nct.sf(critical, df, -shift))
    return upper + (lower if np.isfinite(lower) else 0.0)


def repeats_needed(delta: float, sd: float, alpha: float, power: float = POWER) -> float:
    """The fewest repeats (``MIN_REPEATS`` to ``MAX_REPEATS``) reaching ``power``; ``inf``
    beyond. It starts at the paired test's minimum: two repeats are never tested, whatever
    their power would be (audit 10, owner, 2026-10-06)."""
    for repeats in range(MIN_REPEATS, MAX_REPEATS + 1):
        if paired_power(abs(delta), sd, repeats, alpha) >= power:
            return float(repeats)
    return np.inf


def _fields_per_repeat(block: pd.DataFrame) -> float:
    """The most fields a side any repeat needs to resolve its difference: the fields at which
    its interval's half-width falls below |d|, from its part within wells
    (``projected_fields``; fewer than acquired where it is resolved); ``inf`` when a repeat's
    difference is too small to settle with ``PRACTICAL`` times its fields, or the part
    between wells alone is as wide as |d|."""
    needed = []
    for row in block.to_dict("records"):
        fields = float(min(row["fields"], row["control_fields"]))
        share = row.get("between_share", 0.0)
        projected, between = projected_fields(
            fields, abs(row["d_objects"]), row["d_low"], row["d_high"], share
        )
        too_many = between or projected > PRACTICAL * fields
        needed.append(np.inf if too_many else max(2.0, np.ceil(projected)))
    return max(needed) if needed else np.nan


def _test_values(
    block: pd.DataFrame, log: bool | None = None
) -> tuple[str, np.ndarray, np.ndarray]:
    """The values the paired t-test compares, per repeat, and their sampling errors: the log
    ratios on the feature's log scale (``log``), else the differences
    (``statistics.repeat_values``, as the test chooses); Cohen's d when the table has no
    sampling error of them."""
    d_errors = ((block["d_high"] - block["d_low"]) / (2 * _Z)).to_numpy(float)
    if not {"mean", "control_mean", "log_ratio_se", "difference_se"} <= set(block.columns):
        return "d", block["d_objects"].to_numpy(float), d_errors
    values, compared, _ = repeat_values(block["mean"], block["control_mean"], log)
    column = "log_ratio_se" if values == "normalised" else "difference_se"
    errors = block[column].to_numpy(float)
    if not (np.isfinite(compared).all() and np.isfinite(errors).all()):
        return "d", block["d_objects"].to_numpy(float), d_errors
    return ("log ratio" if values == "normalised" else "difference"), compared, errors


def _change(table: pd.DataFrame, condition: Any, control: Any, log: bool | None) -> float:
    """The change against the control over every repeat holding both means, as the verdict
    gives it (``statistics.repeat_values``), whether or not a repeat has an interval of d."""
    rows = table[(table["condition"] == condition) & (table["control_condition"] == control)]
    if "mean" not in rows:
        return np.nan
    rows = rows[rows["mean"].notna() & rows["control_mean"].notna()]
    return repeat_values(rows["mean"], rows["control_mean"], log)[2] if len(rows) else np.nan


def repeat_plan(
    table: pd.DataFrame, alpha: float = ALPHA, power: float = POWER, log: bool | None = None
) -> pd.DataFrame:
    """One row per condition and control: the plan for the next experiment (see the module).

    ``table`` is ``condition_contrast``'s per-repeat table (with ``d_low``/``d_high``, and
    ``mean``, ``control_mean``, ``difference_se`` and ``log_ratio_se``); ``log`` the
    feature's scale (``statistics.log_scale``; ``None``: read from each condition's means).
    """
    if table.empty or "d_low" not in table:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    compared = table[table["d_objects"].notna()]
    groups = list(compared.groupby(["condition", "control_condition"], sort=False))
    named = REPEAT if REPEAT in compared else None
    family = experiments(
        {condition: list(block[named]) if named else [0] for (condition, _), block in groups}
    )
    sizes = pd.Series(family).value_counts()
    rows = []
    for (condition, control), whole in groups:
        # Every repeat compared counts; the plan rests on those with an interval of d.
        block = whole[whole["d_low"].notna()]
        d = whole["d_objects"].to_numpy(float)
        row: dict[str, Any] = {
            "condition": condition,
            "control_condition": control,
            "repeats": len(d),
            "d_mean": float(d.mean()),
            "between_sd": np.nan,
            "within_se": np.nan,
            "planned_on": "d",
            "change": np.nan,
            "fields_per_repeat": _fields_per_repeat(block),
            "limited_by": "",
            "repeats_needed": np.nan,
            "repeats_needed_twice_fields": np.nan,
            "alpha_planned": alpha / int(sizes[family[condition]]),
            "experiment": family[condition],
            "status": "ok",
        }
        if len(d) < 2:
            row["status"] = "one biological repeat: the variation between repeats is unknown"
        elif len(block) < 2:
            row["status"] = (
                f"{len(block)} of {len(d)} repeats with an interval of d (two fields a side): "
                "the variation between repeats is not estimated"
            )
        if not len(block):
            rows.append(row)
            continue
        d_within = float(np.mean(((block["d_high"] - block["d_low"]) / (2 * _Z)) ** 2))
        planned_on, values, errors = _test_values(block, log)
        within = float(np.mean(errors**2))
        # Twice the fields halve only the part of each repeat's sampling error within wells
        # (``between_share``, owner 2026-10-05, #96); the part between wells stays.
        shares = (
            block["between_share"].fillna(0.0).to_numpy(float)
            if "between_share" in block
            else np.zeros(len(block))
        )
        within_twice = float(np.mean(errors**2 * (shares + (1 - shares) / 2)))
        mean = float(values.mean())
        row["within_se"] = float(np.sqrt(d_within))
        row["planned_on"] = planned_on
        row["change"] = _change(table, condition, control, log) if planned_on != "d" else np.nan
        if len(block) >= 2:
            d = block["d_objects"].to_numpy(float)
            row["between_sd"] = float(np.sqrt(max(0.0, float(np.var(d, ddof=1)) - d_within)))
            between = max(0.0, float(np.var(values, ddof=1)) - within)
            row["limited_by"] = "between repeats" if between > within else "within repeats"
            for column, error in (
                ("repeats_needed", within),
                ("repeats_needed_twice_fields", within_twice),
            ):
                sd = float(np.sqrt(between + error))
                row[column] = repeats_needed(mean, sd, row["alpha_planned"], power)
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
