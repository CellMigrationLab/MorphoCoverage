"""Did I image enough to resolve the difference from the control? (question 2, with a control).

When a control is declared, the question a microscopist asks of "enough" is
whether the fields acquired pin down the difference the experiment was built
to see, not whether each group's mean reached a fixed precision. Per repeat
and condition, ``sampling_row`` resamples what was acquired, as it was
acquired: technical repeats with replacement, then each one's fields with
replacement (with one technical repeat a side, its fields alone), fields
without observations included (a draw of one adds a field and no cell, #289),
both sides independently, ``n_boot`` times, and takes Cohen's d on the
observations of each draw. From the draws:

- **the interval**: the central ``level`` share of the draws' d (a
  percentile interval);
- **resolved**: the interval excludes 0, so the fields acquired give the
  difference's sign, whatever field-to-field variation they carry;
- **the sampling error of the repeat's value**: the draws' standard deviation of the
  difference of means and of the log of their ratio, the values the paired t-test across
  repeats compares (``repeat_plan`` plans on them);
- **the part between wells** (``between_share``, owner 2026-10-05, #96): the share of the
  draws' variance of d that more fields per well do not reduce, from a nested bootstrap
  (the wells resampled with their fields as acquired, less the fields resampled within
  the wells as acquired); 0 with one well a side, whose interval holds no variation
  between wells;
- **fields to resolve it**: when not resolved, the fields per side at which
  the interval's half-width would fall to |d|, its part within wells shrinking
  as ``1 / fields`` in variance and its part between wells staying (a projection
  from these fields, not a promise), counted as acquired, fields without
  observations included. When the part between wells alone is as wide as |d|
  (``between_wells``), more fields will not resolve it: more wells or biological
  repeats would. Beyond ``PRACTICAL`` times the fields acquired, the difference
  is called too small to settle with more fields of these wells: it is
  described, not flagged.

**Fewer than ``MIN_FIELDS`` fields a side: not assessable** (audit 10, owner,
2026-10-06). A percentile bootstrap of 3 fields a side called a difference "shown" in 22%
of null comparisons, 11-15% from 5 vs 4 to 10 vs 3 (audit 10B). Below ``MIN_FIELDS`` fields
with observations on either side the row is ``assessable`` False and never ``resolved``;
its interval and sampling errors are kept for the plan of the next experiment
(``repeat_plan``), not for a verdict. **One observation per field and no wells**: the
observations are then independent draws, and the interval of d is the exact one, from the
noncentral t distribution of ``d sqrt(n1 n2 / (n1 + n2))`` (``noncentral_d_interval``),
instead of the bootstrap's; the bootstrap still gives the sampling errors.

This is a sampling answer about these wells and repeats: whether the
difference between conditions generalizes is question 9, across biological
repeats. The draws are seeded from each contrast's labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

from ..statistics import cohen_d

# The share of the bootstrap draws the interval covers.
LEVEL = 0.95
# Draws per contrast.
N_BOOT = 400
# Fields to resolve a difference beyond this many times the fields acquired: too small to
# settle with more fields of these wells.
PRACTICAL = 4
# What a difference that more fields will not resolve needs (owner, 2026-10-05).
MORE_WELLS = "more wells or biological repeats"
# Below this many fields with observations on a side, "shown" is not assessable (audit 10,
# owner, 2026-10-06).
MIN_FIELDS = 5
SAMPLING_COLUMNS = [
    "d_low",
    "d_high",
    "assessable",
    "interval",
    "resolved",
    "fields_to_resolve",
    "too_small",
    "between_wells",
    "between_share",
    "difference_se",
    "log_ratio_se",
]
# A contrast with no interval: its sampling columns.
NOT_SAMPLED: dict[str, Any] = {
    "d_low": np.nan,
    "d_high": np.nan,
    "assessable": False,
    "interval": "",
    "resolved": False,
    "fields_to_resolve": np.nan,
    "too_small": False,
    "between_wells": False,
    "between_share": np.nan,
    "difference_se": np.nan,
    "log_ratio_se": np.nan,
}


def _draw(
    units: list[list[np.ndarray]],
    rng: np.random.Generator,
    by_unit: bool = True,
    by_field: bool = True,
) -> np.ndarray:
    """One resample of a side: units with replacement, then their fields with replacement,
    fields without observations included (#289). ``by_unit=False`` keeps the units as
    acquired, ``by_field=False`` each unit's fields as acquired."""
    units = [u for u in units if u]
    picks = rng.integers(len(units), size=len(units)) if by_unit else range(len(units))
    taken = []
    for i in picks:
        unit = units[int(i)]
        if by_field:
            taken += [unit[int(j)] for j in rng.integers(len(unit), size=len(unit))]
        else:
            taken += unit
    return np.concatenate(taken) if taken else np.empty(0)


def _draws(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    rng: np.random.Generator,
    n_boot: int,
    by_unit: bool = True,
    by_field: bool = True,
) -> np.ndarray:
    """``n_boot`` resamples of both sides: d and each side's mean (NaN when a side drew fewer
    than two observations, which can happen once empty fields are drawn)."""
    sampled = []
    for _ in range(n_boot):
        treated = _draw(t_units, rng, by_unit, by_field)
        control = _draw(c_units, rng, by_unit, by_field)
        if len(treated) < 2 or len(control) < 2:
            sampled.append((np.nan, np.nan, np.nan))
            continue
        sampled.append((cohen_d(treated, control), treated.mean(), control.mean()))
    return np.array(sampled, dtype=float)


def noncentral_d_interval(
    d: float, n1: int, n2: int, level: float = LEVEL
) -> tuple[float, float]:
    """The exact ``level`` interval of Cohen's d of two independent normal samples of ``n1``
    and ``n2`` values (Hedges and Olkin 1985; Steiger and Fouladi 1997): ``t = d sqrt(n1 n2 /
    (n1 + n2))`` follows the noncentral t with ``n1 + n2 - 2`` degrees of freedom, and the
    interval inverts it for its noncentrality, scaled back to d. NaN without a d or below
    two values a side."""
    if not np.isfinite(d) or n1 < 2 or n2 < 2:
        return np.nan, np.nan
    scale = np.sqrt(n1 * n2 / (n1 + n2))
    t, df, tail = d * scale, n1 + n2 - 2, (1 - level) / 2

    def gap(ncp: float, target: float) -> float:
        # The cdf as the upper tail of the mirrored t: scipy's noncentral t cdf returns NaN
        # far in its tail (scipy 1.17), its sf stays exact there (as in repeat_plan).
        value = float(stats.nct.sf(-t, df, -ncp))
        return (value if np.isfinite(value) else float(stats.nct.cdf(t, df, ncp))) - target

    def bound(target: float) -> float:
        # The cdf at t falls as the noncentrality grows: widen a bracket around t to the root.
        width = 2.0
        while gap(t - width, target) * gap(t + width, target) > 0 and width < 1e3:
            width *= 2
        return float(optimize.brentq(gap, t - width, t + width, args=(target,)))

    return float(bound(1 - tail) / scale), float(bound(tail) / scale)


def _one_per_field(units: list[list[np.ndarray]]) -> bool:
    """Whether a side is one unit (no wells) whose fields hold at most one observation each:
    its observations are then independent draws."""
    units = [u for u in units if u]
    return len(units) == 1 and all(len(field) <= 1 for field in units[0])


def _variance(values: np.ndarray) -> float:
    """The variance of the finite draws of d."""
    d = values[:, 0]
    d = d[np.isfinite(d)]
    return float(np.var(d, ddof=1)) if len(d) > 1 else np.nan


def _between_share(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    total: float,
    rng: np.random.Generator,
    n_boot: int,
) -> float:
    """The share of the variance of d that is between wells, which more fields per well do
    not reduce (#96): the wells resampled with their fields as acquired carry the variation
    between wells and their field error once, the fields resampled within wells held as
    acquired carry the field error alone, and the difference is the part between wells
    (a nested bootstrap, as a one-way analysis of variance splits the variance of well
    means). 0 with one well a side: the interval then holds no variation between wells."""
    if max(len(t_units), len(c_units)) < 2 or not total > 0:
        return 0.0
    wells = _variance(_draws(t_units, c_units, rng, n_boot, by_field=False))
    fields = _variance(_draws(t_units, c_units, rng, n_boot, by_unit=False))
    if not (np.isfinite(wells) and np.isfinite(fields)):
        return 0.0
    return float(min(max(wells - fields, 0.0) / total, 1.0))


def projected_fields(
    fields: float, d: float, low: float, high: float, between_share: float = 0.0
) -> tuple[float, bool]:
    """The fields a side at which the interval's half-width would fall to |d|, projected from
    its field part only (``between_share`` of its variance does not shrink with more fields,
    the rest as ``1 / fields``), and whether the part between wells alone is already as wide
    as |d|: then more fields will not resolve it (``inf``, ``True``)."""
    half = (high - low) / 2
    share = float(between_share) if np.isfinite(between_share) else 0.0
    if not d:
        return np.inf, False
    between, within = half**2 * share, half**2 * (1 - share)
    if between >= d**2:
        return np.inf, True
    return fields * within / (d**2 - between), False


def sampling_row(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    d: float,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    level: float = LEVEL,
    min_fields: int = MIN_FIELDS,
) -> dict[str, Any]:
    """Whether the fields acquired resolve the difference ``d`` (see the module); below
    ``min_fields`` fields a side, not ``assessable``."""
    empty = dict(NOT_SAMPLED)
    n_fields = min(
        sum(1 for u in t_units for f in u if len(f)), sum(1 for u in c_units for f in u if len(f))
    )
    if not np.isfinite(d) or n_fields < 2:
        return empty
    values = _draws(t_units, c_units, rng, n_boot)
    values = values[np.isfinite(values[:, 0])]
    draws = values[:, 0]
    if len(draws) < n_boot / 2:
        return empty
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    interval = "bootstrap"
    if _one_per_field(t_units) and _one_per_field(c_units):
        # Independent observations: the exact interval (see the module).
        counts = [sum(len(f) for u in units for f in u) for units in (t_units, c_units)]
        low, high = noncentral_d_interval(d, *counts, level)
        interval = "noncentral t"
    assessable = n_fields >= min_fields
    resolved = bool(assessable and (low > 0 or high < 0))
    row = {
        **empty,
        "d_low": float(low),
        "d_high": float(high),
        "assessable": assessable,
        "interval": interval,
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
    # After the draws above, so the interval does not depend on it.
    row["between_share"] = _between_share(t_units, c_units, _variance(values), rng, n_boot)
    if assessable and not resolved:
        # In fields acquired, empty ones included: the draws hold them too (#289).
        acquired = min(sum(len(u) for u in t_units), sum(len(u) for u in c_units))
        needed, between = projected_fields(acquired, d, low, high, row["between_share"])
        row["between_wells"] = between
        if not between:
            row["fields_to_resolve"] = float(np.ceil(needed)) if np.isfinite(needed) else np.inf
            row["too_small"] = bool(needed > PRACTICAL * acquired)
    return row


__all__ = [
    "LEVEL",
    "MIN_FIELDS",
    "MORE_WELLS",
    "NOT_SAMPLED",
    "N_BOOT",
    "PRACTICAL",
    "SAMPLING_COLUMNS",
    "noncentral_d_interval",
    "projected_fields",
    "sampling_row",
]
