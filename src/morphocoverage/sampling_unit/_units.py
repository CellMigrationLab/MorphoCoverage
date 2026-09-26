"""Shared helpers of the condition contrast and the feature ranking (private).

Both compare each condition with the declared control per repeat, with the
technical repeat (an analysis group split by the ``technical_repeat`` column)
as the unit when each side holds at least two, else the whole field. The
helpers here describe the analysis groups, gather each side's fields by
unit, and run the randomization test over those units.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..retrospective._text import group_identity
from ..schema import ValidatedInput
from ..statistics import cohen_d
from ._common import fov_values

FIELD = "field"


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
        raise ValueError(f"no analysis group matches control {dict(control)}")
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
    Equal means with no variation stay NaN.
    """
    d = cohen_d(first, second)
    if np.isnan(d) and len(first) >= 2 and len(second) >= 2:
        difference = float(first.mean() - second.mean())
        if difference != 0:
            return float(np.copysign(np.inf, difference))
    return d


def permutation_p(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    by_unit: bool,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """``(extreme + 1) / (n + 1)`` for the absolute Cohen's d on field means under shuffled labels.

    The statistic is ``separation_d``, so a shuffle that separates the sides
    perfectly counts as at least as extreme as a perfectly separated observation.

    ``by_unit`` shuffles whole technical repeats (each keeps its fields);
    otherwise the fields themselves are shuffled between the sides.
    """
    extreme = 0
    if by_unit:
        units = [*t_units, *c_units]
        n_treated = len(t_units)
        for _ in range(n_permutations):
            order = rng.permutation(len(units))
            treated = [units[i] for i in order[:n_treated]]
            control = [units[i] for i in order[n_treated:]]
            statistic = abs(separation_d(field_means(treated), field_means(control)))
            if np.isfinite(statistic) and statistic >= observed:
                extreme += 1
        return (extreme + 1) / (n_permutations + 1)
    pooled_means = np.concatenate([field_means(t_units), field_means(c_units)])
    n_treated = len(field_means(t_units))
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled_means)
        statistic = abs(separation_d(shuffled[:n_treated], shuffled[n_treated:]))
        if np.isfinite(statistic) and statistic >= observed:
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)
