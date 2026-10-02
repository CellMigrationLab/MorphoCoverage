"""Did I image enough to resolve the difference from the control? (question 2, with a control).

When a control is declared, the question a microscopist asks of "enough" is
whether the fields acquired pin down the difference the experiment was built
to see, not whether each group's mean reached a fixed precision. Per repeat
and condition, ``sampling_row`` resamples what was acquired, as it was
acquired: technical repeats with replacement, then each one's fields with
replacement (with one technical repeat a side, its fields alone), both sides
independently, ``n_boot`` times, and takes Cohen's d on the observations of
each draw. From the draws:

- **the interval**: the central ``level`` share of the draws' d (a
  percentile interval);
- **resolved**: the interval excludes 0, so the fields acquired give the
  difference's sign, whatever field-to-field variation they carry;
- **the sampling error of the repeat's value**: the draws' standard deviation of the
  difference of means and of the log of their ratio, the values the paired t-test across
  repeats compares (``repeat_plan`` plans on them);
- **fields to resolve it**: when not resolved, the fields per side at which
  the interval's half-width, shrinking as ``1 / sqrt(fields)``, would fall
  below |d| (a projection from these fields, not a promise). Beyond
  ``PRACTICAL`` times the fields acquired, the difference is called too small
  to settle with more fields of these wells: it is described, not flagged.

This is a sampling answer about these wells and repeats: whether the
difference between conditions generalizes is question 9, across biological
repeats. The draws are seeded from each contrast's labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..statistics import cohen_d

# The share of the bootstrap draws the interval covers.
LEVEL = 0.95
# Draws per contrast.
N_BOOT = 400
# Fields to resolve a difference beyond this many times the fields acquired: too small to
# settle with more fields of these wells.
PRACTICAL = 4
SAMPLING_COLUMNS = [
    "d_low",
    "d_high",
    "resolved",
    "fields_to_resolve",
    "too_small",
    "difference_se",
    "log_ratio_se",
]


def _draw(units: list[list[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    """One resample of a side: units with replacement, then their fields with replacement."""
    filled = [[f for f in u if len(f)] for u in units]
    filled = [u for u in filled if u]
    taken = []
    for i in rng.integers(len(filled), size=len(filled)):
        unit = filled[int(i)]
        taken += [unit[int(j)] for j in rng.integers(len(unit), size=len(unit))]
    return np.concatenate(taken)


def sampling_row(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    d: float,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    level: float = LEVEL,
) -> dict[str, Any]:
    """Whether the fields acquired resolve the difference ``d`` (see the module)."""
    empty = {
        "d_low": np.nan,
        "d_high": np.nan,
        "resolved": False,
        "fields_to_resolve": np.nan,
        "too_small": False,
        "difference_se": np.nan,
        "log_ratio_se": np.nan,
    }
    n_fields = min(
        sum(1 for u in t_units for f in u if len(f)), sum(1 for u in c_units for f in u if len(f))
    )
    if not np.isfinite(d) or n_fields < 2:
        return empty
    sampled = []
    for _ in range(n_boot):
        treated, control = _draw(t_units, rng), _draw(c_units, rng)
        sampled.append((cohen_d(treated, control), treated.mean(), control.mean()))
    values = np.array(sampled, dtype=float)
    values = values[np.isfinite(values[:, 0])]
    draws = values[:, 0]
    if len(draws) < n_boot / 2:
        return empty
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    resolved = bool(low > 0 or high < 0)
    row = {
        **empty,
        "d_low": float(low),
        "d_high": float(high),
        "resolved": resolved,
        "difference_se": float(np.std(values[:, 1] - values[:, 2], ddof=1)),
    }
    # The log ratio from the draws whose means are both positive (a draw of only the lowest
    # fields can reach 0), when most are: the test takes the log ratios when the acquired
    # means are positive, whatever a few draws give.
    positive = (values[:, 1] > 0) & (values[:, 2] > 0)
    if positive.sum() >= n_boot / 2:
        ratios = np.log(values[positive, 1] / values[positive, 2])
        row["log_ratio_se"] = float(np.std(ratios, ddof=1))
    if not resolved:
        half = (high - low) / 2
        needed = n_fields * (half / abs(d)) ** 2 if d else np.inf
        row["fields_to_resolve"] = float(np.ceil(needed)) if np.isfinite(needed) else np.inf
        row["too_small"] = bool(needed > PRACTICAL * n_fields)
    return row


__all__ = ["LEVEL", "N_BOOT", "PRACTICAL", "SAMPLING_COLUMNS", "sampling_row"]
