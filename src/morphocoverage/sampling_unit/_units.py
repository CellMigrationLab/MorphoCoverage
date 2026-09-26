"""Shared helpers of the condition contrast and the feature ranking (private).

Both compare each condition with the declared control per repeat, with the
technical repeat (an analysis group split by the ``technical_repeat`` column)
as the unit when each side holds at least two, else, without that column,
the whole field. A side with a single technical repeat gets no test (its
fields cannot stand in for technical repeats). The helpers here resolve the
columns, describe the analysis groups, gather each side's fields by unit,
and run the randomization test over those units, exactly when the distinct
assignments are few enough to enumerate.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from .._seeds import group_identity
from ..metadata import REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..schema import ValidatedInput
from ..statistics import RELATIVE_TOLERANCE, cohen_d, mean_is_zero
from ._common import fov_values, no_match_message

FIELD = "field"


def resolve_columns(
    data: ValidatedInput,
    condition_col: str,
    repeat_col: str | None,
    technical_col: str | None,
    control: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """The repeat and technical-repeat columns to use, after checking every named column.

    Only a default may be absent: without the ``repeat`` column the whole
    table is one implicit repeat, and without a ``technical_repeat`` analysis
    group the fields are the units. A column named explicitly that the tables
    do not hold (a misspelling, say) raises, since dropping it would pool the
    repeats or compare fields where technical repeats were meant.
    """
    columns = data.fovs.columns
    if condition_col not in columns:
        raise ValueError(f"condition_col {condition_col!r} is not a column of the FOV table")
    missing = [c for c in control if c not in columns]
    if missing:
        raise ValueError(
            f"control names columns absent from the FOV table: {missing}; name the control "
            f"by FOV-table columns, such as {{{condition_col!r}: ...}}"
        )
    if repeat_col is not None and repeat_col not in columns:
        if repeat_col != REPEAT_COLUMN:
            raise ValueError(
                f"repeat_col {repeat_col!r} is not a column of the FOV table; name the "
                "biological-repeat column, or pass repeat_col=None to compare within the "
                "whole table"
            )
        repeat_col = None  # the default is absent: one implicit repeat
    if technical_col is not None and technical_col not in data.group_cols:
        if technical_col != TECHNICAL_REPEAT_COLUMN:
            where = "an analysis-group column" if technical_col in columns else "a column"
            raise ValueError(
                f"technical_col {technical_col!r} is not {where} of the FOV table; name the "
                "technical-repeat column and include it in group_cols, or pass "
                "technical_col=None to compare fields"
            )
        technical_col = None  # the groups are not split by technical repeat
    return repeat_col, technical_col


def describe_groups(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    condition_col: str,
    repeat_col: str | None,
    control: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Each analysis group with its condition, repeat and whether it is the control.

    Raises when a group spans several conditions or repeats (the column is
    then missing from ``group_cols``) or when no group matches ``control``.
    """

    def value_of(group_data: ValidatedInput, column: str) -> Any:
        values = group_data.fovs[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(
                f"an analysis group spans several values of {column!r}; add it to group_cols"
            )
        return values.iloc[0]

    described = []
    for group, group_data in all_groups:
        described.append(
            {
                "group": group,
                "condition": value_of(group_data, condition_col),
                "repeat": value_of(group_data, repeat_col) if repeat_col else None,
                "is_control": all(
                    value_of(group_data, column) == value for column, value in control.items()
                ),
            }
        )
    if not any(d["is_control"] for d in described):
        raise ValueError(no_match_message(all_groups, control, "control"))
    return described


def contrasts_of(described: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    """The ``(repeat, condition)`` pairs to compare with the control, in a stable order."""
    return sorted(
        {(d["repeat"], d["condition"]) for d in described if not d["is_control"]},
        key=lambda pair: (str(pair[0]), str(pair[1])),
    )


def sides(
    described: list[dict[str, Any]], repeat: Any, condition: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The treated groups of ``condition`` and the control groups, within ``repeat``."""
    controls = [d["group"] for d in described if d["is_control"] and d["repeat"] == repeat]
    treated = [
        d["group"]
        for d in described
        if not d["is_control"] and d["repeat"] == repeat and d["condition"] == condition
    ]
    return treated, controls


def side_units(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    feature: str,
    chosen: list[dict[str, Any]],
) -> tuple[list[list[np.ndarray]], int]:
    """The fields (arrays of the feature, NaN dropped) of each chosen group, and the field count.

    One inner list per analysis group: the technical repeats of the side when
    the groups are split by the technical-repeat column.
    """
    by_identity = {group_identity(g): d for g, d in all_groups}
    units, acquired = [], 0
    for group in chosen:
        data = by_identity[group_identity(group)]
        units.append(fov_values(data, data.feature_values(feature)))
        acquired += len(data.fovs)
    return units, acquired


def unit_means(units: list[list[np.ndarray]]) -> np.ndarray:
    """One mean per unit with observations."""
    pooled = [np.concatenate(u) for u in units if sum(len(f) for f in u)]
    return np.array([p.mean() for p in pooled])


def field_means(units: list[list[np.ndarray]]) -> np.ndarray:
    """One mean per field with observations, over every unit."""
    return np.array([f.mean() for u in units for f in u if len(f)])


def separation_d(first: np.ndarray, second: np.ndarray) -> float:
    """Cohen's d, with a perfect separation as an infinite d instead of undefined.

    When each side holds at least two values, none varies and the means
    differ, the pooled variance is zero and ``cohen_d`` is NaN, yet the sides
    are as separated as they can be; this returns ``+inf`` or ``-inf`` then.
    Equal means with no variation stay NaN, and so does a difference of means
    that is rounding noise (a constant feature such as 0.1 leaves field means
    that differ by about 1e-17): a constant feature is never "separated".
    """
    d = cohen_d(first, second)
    if np.isnan(d) and len(first) >= 2 and len(second) >= 2:
        difference = float(first.mean() - second.mean())
        if not mean_is_zero(difference, np.concatenate([first, second])):
            return float(np.copysign(np.inf, difference))
    return d


def technical_counts(
    t_units: list[list[np.ndarray]], c_units: list[list[np.ndarray]], technical_col: str | None
) -> tuple[int, int]:
    """Technical repeats with observations on each side; ``(0, 0)`` without the column."""
    if not technical_col:
        return 0, 0
    return len(unit_means(t_units)), len(unit_means(c_units))


def untested_reason(
    n_fields: tuple[int, int], n_technical: tuple[int, int], technical_col: str | None
) -> str | None:
    """Why a contrast gets no randomization test, or ``None`` when it gets one.

    With the technical-repeat column, a side holding a single technical
    repeat gets no test: shuffling that side's fields would treat them as
    independent wells, and a well-to-well shift then passes for a condition
    effect (in a check with no condition effect, 60% of such tests gave
    p <= 0.05). One well cannot be compared with another condition's wells.
    """
    if min(n_fields) < 2:
        return "fewer than 2 fields with observations on one side"
    if technical_col and min(n_technical) == 1:
        return (
            f"only one technical repeat ({technical_col}) with observations on one side: its "
            "fields cannot stand in for technical repeats, so there is no test"
        )
    return None


def smallest_p(n_treated: int, n_control: int) -> float:
    """The smallest p-value the randomization test can give with these unit counts.

    Each of the ``comb(n, n_treated)`` assignments is equally likely, and the
    observed one is among those at least as extreme; with equal sides its
    mirror (the sides swapped) gives the same |d|, so the floor is twice as high.
    """
    total = comb(n_treated + n_control, n_treated)
    return (2 if n_treated == n_control else 1) / total


def _at_least(statistic: float, observed: float) -> bool:
    """Whether a shuffled statistic is at least as extreme as the observed one.

    Ties count, also when rounding put them just below (the observed |d| and
    a shuffle's are computed from differently ordered arrays), and an
    infinite statistic equals an infinite observed one; only NaN never counts.
    """
    if np.isnan(statistic):
        return False
    if statistic == observed:
        return True
    return bool(np.isfinite(observed) and statistic >= observed - RELATIVE_TOLERANCE * observed)


def permutation_p(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    by_unit: bool,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """The randomization p-value of the absolute Cohen's d on field means.

    The statistic is ``separation_d``, so a shuffle that separates the sides
    perfectly counts as at least as extreme as a perfectly separated observation.

    ``by_unit`` shuffles whole technical repeats (each keeps its fields; those
    without observations carry no label that matters and are left out);
    otherwise the fields themselves are shuffled between the sides. When no
    more distinct assignments exist than ``n_permutations``, every assignment
    is enumerated and the p-value is exact: the share of assignments at least
    as extreme as the observed one, which is among them. Otherwise it is
    ``(extreme + 1) / (n + 1)`` over ``n_permutations`` random shuffles, as in
    Plot&Stats.
    """
    if by_unit:
        items: list[Any] = [u for u in [*t_units, *c_units] if sum(len(f) for f in u)]
        n_treated = sum(1 for u in t_units if sum(len(f) for f in u))

        def statistic(treated: np.ndarray) -> float:
            inside = np.zeros(len(items), dtype=bool)
            inside[treated] = True
            first = [items[i] for i in np.flatnonzero(inside)]
            second = [items[i] for i in np.flatnonzero(~inside)]
            return abs(separation_d(field_means(first), field_means(second)))

    else:
        means = np.concatenate([field_means(t_units), field_means(c_units)])
        items = list(means)
        n_treated = len(field_means(t_units))

        def statistic(treated: np.ndarray) -> float:
            inside = np.zeros(len(means), dtype=bool)
            inside[treated] = True
            return abs(separation_d(means[inside], means[~inside]))

    n = len(items)
    total = comb(n, n_treated)
    if total <= n_permutations:
        extreme = sum(
            _at_least(statistic(np.array(chosen, dtype=int)), observed)
            for chosen in combinations(range(n), n_treated)
        )
        return extreme / total
    extreme = sum(
        _at_least(statistic(rng.permutation(n)[:n_treated]), observed)
        for _ in range(n_permutations)
    )
    return (extreme + 1) / (n_permutations + 1)


def relabellings(
    t_units: list[list[np.ndarray]], c_units: list[list[np.ndarray]], by_unit: bool
) -> int:
    """The distinct assignments of the condition labels ``permutation_p`` draws from."""
    if by_unit:
        n_treated, n_control = len(unit_means(t_units)), len(unit_means(c_units))
    else:
        n_treated, n_control = len(field_means(t_units)), len(field_means(c_units))
    return comb(n_treated + n_control, n_treated)
