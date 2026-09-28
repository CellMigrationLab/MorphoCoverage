"""Shared helpers of the condition contrast and the feature ranking (private).

Both compare each condition with the declared control per repeat, with the
technical repeat (the FOV table's ``technical_repeat`` column, or the column
``technical_col`` names) as the unit when each side holds at least two, else,
without that column, the whole field. The technical repeat is the unit
whatever ``group_cols`` says: an analysis group that pools several technical
repeats is split by the column, since shuffling its fields as if they were
independent lets a well-to-well shift pass for a condition effect (with three
wells a side and no condition effect, 42% of such tests gave p <= 0.05). A
side with a single technical repeat gets no test (its fields cannot stand in
for technical repeats). The helpers here check the arguments, resolve the
columns, describe the analysis groups, gather each side's fields by unit, and
run the randomization test over those units, exactly when the distinct
assignments are few enough to enumerate.

Shared-private: every public-looking name here is used by ``contrast`` and
``ranking`` (and ``FIELD`` by their summaries).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from .._checks import as_count, check_seed
from .._seeds import group_key
from ..metadata import REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN, empty_values
from ..schema import ValidatedInput, fov_values
from ..statistics import RELATIVE_TOLERANCE, cohen_d, mean_is_zero
from ._common import no_match_message

FIELD = "field"
# Assignments enumerated exactly, whatever n_permutations asks for: below this many the
# exact p-value costs less than a few thousand random shuffles would.
ENUMERATION_BUDGET = 20000
_CHUNK = 4096  # assignments scored at once


def check_arguments(
    control: object, condition_col: str, n_permutations: object, random_state: object
) -> tuple[int, int]:
    """The checks ``condition_contrast`` and ``feature_ranking`` share; the counts as ``int``."""
    if not isinstance(control, Mapping) or not control:
        raise ValueError("control must be a non-empty mapping of FOV-table column to value")
    if condition_col not in control:
        raise ValueError(f"control must name the condition column {condition_col!r}")
    return as_count(n_permutations, "n_permutations", 1), check_seed(random_state)


def resolve_columns(
    data: ValidatedInput,
    condition_col: str,
    repeat_col: str | None,
    technical_col: str | None,
    control: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """The repeat and technical-repeat columns to use, after checking every named column.

    Only a default may be absent: without the ``repeat`` column the whole
    table is one implicit repeat, and without a ``technical_repeat`` column
    the fields are the units. A column named explicitly that the FOV table
    does not hold (a misspelling, say) raises, since dropping it would pool
    the repeats or compare fields where technical repeats were meant. The
    technical-repeat column need not be an analysis-group column: groups that
    pool technical repeats are split by it. A field without a technical-repeat
    value raises, since its unit is unknown.
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
    if technical_col is not None and technical_col not in columns:
        if technical_col != TECHNICAL_REPEAT_COLUMN:
            raise ValueError(
                f"technical_col {technical_col!r} is not a column of the FOV table; name the "
                "technical-repeat column, or pass technical_col=None to compare fields"
            )
        technical_col = None  # the default is absent: the fields are the units
    if technical_col is not None:
        unplaced = int(empty_values(data.fovs[technical_col]).sum())
        if unplaced:
            raise ValueError(
                f"{unplaced} of the {len(data.fovs)} fields have no {technical_col!r} value, "
                "so their technical repeat is unknown; fill it in, or pass technical_col=None "
                "to compare fields"
            )
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


def control_label(described: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    """The control conditions of ``controls``, joined by ``/`` (empty without controls)."""
    return "/".join(sorted({str(d["condition"]) for d in described if d["group"] in controls}))


def side_units(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    feature: str,
    chosen: list[dict[str, Any]],
    technical_col: str | None,
) -> tuple[list[list[np.ndarray]], int]:
    """The fields (arrays of the feature, NaN dropped) of each unit of a side, and the field count.

    One inner list per unit: per analysis group, one per technical repeat
    when there is a technical-repeat column (a technical repeat is identified
    within its analysis group, so well ``A1`` of two groups is two units),
    else one per group.
    """
    by_key = {group_key(g): d for g, d in all_groups}
    units, acquired = [], 0
    for group in chosen:
        data = by_key[group_key(group)]
        per_field = fov_values(data, data.feature_values(feature))
        acquired += len(data.fovs)
        if technical_col is None:
            units.append(per_field)
            continue
        labels = data.fovs[technical_col].to_numpy()
        for label in dict.fromkeys(labels.tolist()):
            units.append([per_field[i] for i in np.flatnonzero(labels == label)])
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


def is_exact(total: int, n_permutations: int) -> bool:
    """Whether ``total`` assignments are enumerated: at most ``n_permutations`` or the budget."""
    return total <= max(n_permutations, ENUMERATION_BUDGET)


def _separations(items: list[np.ndarray], inside: np.ndarray) -> np.ndarray:
    """``|separation_d|`` of the field means for each assignment, all at once.

    ``items`` hold the field means of each shuffled unit (one each when fields
    are shuffled); row ``a`` of ``inside`` marks the treated units of
    assignment ``a``. Sums are taken around the grand mean, so the pooled
    variance does not lose digits, and a side's sums are computed the same
    way as its mirror's, so an assignment and its mirror give the same |d|.
    Rounding-noise spreads and differences are judged as in ``separation_d``.
    """
    means = np.concatenate(items)
    tolerance = RELATIVE_TOLERANCE * float(np.mean(np.abs(means)))
    centre = float(means.mean())
    counts = np.array([len(i) for i in items], dtype=float)
    sums = np.array([np.sum(i - centre) for i in items])
    squares = np.array([np.sum((i - centre) ** 2) for i in items])

    def side(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weight = mask.astype(float)
        return (weight * counts).sum(1), (weight * sums).sum(1), (weight * squares).sum(1)

    n1, s1, q1 = side(inside)
    n2, s2, q2 = side(~inside)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1, m2 = s1 / n1, s2 / n2
        pooled = ((q1 - s1 * m1) + (q2 - s2 * m2)) / (n1 + n2 - 2)
        sd = np.sqrt(np.maximum(pooled, 0.0))
        difference = m1 - m2
        d = np.abs(difference / sd)
    flat = ~(sd > tolerance)
    same = (np.abs(difference) <= tolerance) | (difference == 0)
    d = np.where(flat, np.where(same, np.nan, np.inf), d)
    return np.where((n1 < 2) | (n2 < 2), np.nan, d)


def _extreme(statistic: np.ndarray, observed: float) -> int:
    """How many statistics are at least as extreme as the observed one.

    Ties count, also when rounding put them just below (the observed |d| and
    a shuffle's are computed differently), and an infinite statistic equals
    an infinite observed one; NaN never counts.
    """
    hit = statistic == observed
    if np.isfinite(observed):
        hit |= statistic >= observed - RELATIVE_TOLERANCE * observed
    return int(np.count_nonzero(hit))


def _assignments(chosen: Iterator[tuple[int, ...]] | np.ndarray, n: int) -> Iterator[np.ndarray]:
    """Boolean ``(rows, n)`` blocks of treated units, from rows of chosen unit indices."""
    block: list[Any] = []
    for row in chosen:
        block.append(row)
        if len(block) == _CHUNK:
            yield _membership(block, n)
            block = []
    if block:
        yield _membership(block, n)


def _membership(rows: list[Any], n: int) -> np.ndarray:
    inside = np.zeros((len(rows), n), dtype=bool)
    index = np.asarray(rows, dtype=int).reshape(len(rows), -1)
    np.put_along_axis(inside, index, True, axis=1)
    return inside


def _shuffled_items(
    t_units: list[list[np.ndarray]], c_units: list[list[np.ndarray]], by_unit: bool
) -> tuple[list[np.ndarray], int]:
    """The shuffled items (their field means) and how many are treated."""
    if by_unit:
        filled_t = [u for u in t_units if sum(len(f) for f in u)]
        filled_c = [u for u in c_units if sum(len(f) for f in u)]
        items = [field_means([u]) for u in [*filled_t, *filled_c]]
        return items, len(filled_t)
    treated, control = field_means(t_units), field_means(c_units)
    return [np.array([m]) for m in np.concatenate([treated, control])], len(treated)


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
    otherwise the fields themselves are shuffled between the sides. When the
    distinct assignments are no more than ``n_permutations`` or
    ``ENUMERATION_BUDGET``, every one is enumerated and the p-value is exact:
    the share of assignments at least as extreme as the observed one, which
    is among them. Otherwise it is ``(extreme + 1) / (n + 1)`` over
    ``n_permutations`` random shuffles, as in Plot&Stats, never below the
    smallest p-value the exact test can give (``smallest_p``).
    """
    items, n_treated = _shuffled_items(t_units, c_units, by_unit)
    n = len(items)
    total = comb(n, n_treated)
    if is_exact(total, n_permutations):
        extreme = sum(
            _extreme(_separations(items, inside), observed)
            for inside in _assignments(combinations(range(n), n_treated), n)
        )
        return extreme / total
    draws = np.array([rng.permutation(n)[:n_treated] for _ in range(n_permutations)])
    extreme = sum(
        _extreme(_separations(items, inside), observed) for inside in _assignments(draws, n)
    )
    return max((extreme + 1) / (n_permutations + 1), smallest_p(n_treated, n - n_treated))


def relabellings(
    t_units: list[list[np.ndarray]], c_units: list[list[np.ndarray]], by_unit: bool
) -> int:
    """The distinct assignments of the condition labels ``permutation_p`` draws from."""
    items, n_treated = _shuffled_items(t_units, c_units, by_unit)
    return comb(len(items), n_treated)


def randomization_test(
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    technical_col: str | None,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The field-level d and the randomization test of one contrast, as table columns.

    ``technical``/``control_technical`` count the technical repeats with
    observations, ``d_fields`` is ``separation_d`` on field means, and the
    test shuffles whole technical repeats when each side holds at least two
    (``permutation_unit`` is then the column, with its
    ``distinct_assignments``), else whole fields. ``p_floor`` is the smallest
    p-value the test can give with these units (``smallest_p``); ``status``
    is ``"ok"`` or why there is no test (``p_value`` NaN).
    """
    n_technical = technical_counts(t_units, c_units, technical_col)
    by_unit = min(n_technical) >= 2
    t_means, c_means = field_means(t_units), field_means(c_units)
    row: dict[str, Any] = {
        "technical": n_technical[0],
        "control_technical": n_technical[1],
        # The measure feature_ranking scores: a perfect separation is an infinite d.
        "d_fields": separation_d(t_means, c_means),
        "permutation_unit": technical_col if by_unit else FIELD,
        "distinct_assignments": comb(sum(n_technical), n_technical[0]) if by_unit else np.nan,
        "p_value": np.nan,
        "p_exact": False,
        "p_floor": np.nan,
        "status": "ok",
    }
    reason = untested_reason((len(t_means), len(c_means)), n_technical, technical_col)
    if reason is not None:
        row["status"] = reason
        return row
    if np.isnan(row["d_fields"]):
        # Every field mean equal on both sides (up to rounding): nothing to test.
        row["status"] = "the field means do not vary, so there is nothing to test"
        return row
    shuffled = n_technical if by_unit else (len(t_means), len(c_means))
    row["p_floor"] = smallest_p(*shuffled)
    row["p_exact"] = is_exact(relabellings(t_units, c_units, by_unit), n_permutations)
    row["p_value"] = permutation_p(
        t_units, c_units, by_unit, abs(row["d_fields"]), n_permutations, rng
    )
    return row
