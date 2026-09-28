"""The matching behind the data-integrity checks (private), with no loop over pairs of fields.

Observations are compared at a stated precision (``DIGITS`` significant
digits, so a copy that went through float32 or a six-digit export still
matches). Every step is one pass over the table or a sort:

- ``Rows``: the observations sorted by field, each rounded row's identity
  (``ids``), the fields that hold exactly the same rows (``copy_of``), and in
  how many such fields each row occurs (``recurrence``);
- ``field_moments``: each field's order-, sign- and unit-free fingerprint per
  feature (the absolute skewness and the kurtosis of its standardised values),
  whose near-equal values, found by sorting, name the candidate copies;
- ``unit_change``: whether one field is the other under a change of units per
  feature, rows aligned by sorting on every standardised feature, so ties,
  sign flips and missing values do not misalign them;
- ``longest_run`` and ``repeated_run``: the longest run of observations shared
  in the same order by two fields, or repeated within one.

Shared-private: every name in ``__all__`` is used by ``integrity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

DIGITS = 6  # significant digits compared: float32 keeps about seven
TOLERANCE = 1e-5  # a change of units must fit every value to this share of its largest
RECURRING = 3  # a row found in this many (non-identical) fields is low-information
_CANDIDATE = 1e-3  # how close two fields' fingerprints must be to be verified
_AMBIGUOUS = 3  # symmetric features whose sign is tried both ways when aligning rows


def rounded(values: np.ndarray, digits: int = DIGITS) -> np.ndarray:
    """``values`` rounded to ``digits`` significant digits; NaN as +inf (never an input value)."""
    out = np.array(values, dtype=float, copy=True)
    finite = np.isfinite(out) & (out != 0)
    magnitude = np.clip(np.floor(np.log10(np.abs(out[finite]))), -300, 300)
    scale = 10.0 ** (digits - 1 - magnitude)
    out[finite] = np.round(out[finite] * scale) / scale
    out[np.isnan(out)] = np.inf
    return out + 0.0  # -0.0 as 0.0


def _row_ids(matrix: np.ndarray) -> np.ndarray:
    """One integer per distinct row of ``matrix`` (equal rows, equal integers)."""
    if not matrix.size:
        return np.zeros(len(matrix), dtype=np.int64)
    contiguous = np.ascontiguousarray(matrix)
    width = contiguous.dtype.itemsize * contiguous.shape[1]
    view = contiguous.view(np.dtype((np.void, width))).ravel()
    return np.unique(view, return_inverse=True)[1].ravel().astype(np.int64)


@dataclass(frozen=True)
class Rows:
    """The observations sorted by field, and the identity of each rounded row (see the module).

    ``values`` and ``exact`` (rounded, NaN as +inf) are the feature rows in
    field order, table order kept within a field; ``field`` is each row's
    field; ``starts`` and ``sizes`` delimit each field's rows. ``copy_of`` is
    each field's first field with exactly the same rounded rows (itself when
    none), and ``recurrence`` the number of such distinct fields each row id
    occurs in.
    """

    values: np.ndarray
    exact: np.ndarray
    field: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    ids: np.ndarray
    copy_of: np.ndarray
    recurrence: np.ndarray

    def of(self, i: int) -> slice:
        return slice(int(self.starts[i]), int(self.starts[i] + self.sizes[i]))

    @classmethod
    def build(cls, matrix: np.ndarray, positions: np.ndarray, n_fields: int) -> Rows:
        order = np.argsort(positions, kind="stable")
        values, field = matrix[order], positions[order]
        sizes = np.bincount(field, minlength=n_fields)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        exact = rounded(values)
        ids = _row_ids(exact)
        # Fields with the same rounded rows, in any order: one sorted id sequence each.
        by_field = ids[np.lexsort((ids, field))]
        copy_of = np.arange(n_fields)
        first: dict[bytes, int] = {}
        for i in np.flatnonzero(sizes):
            key = by_field[starts[i] : starts[i] + sizes[i]].tobytes()
            copy_of[i] = first.setdefault(key, int(i))
        n_ids = int(ids.max()) + 1 if len(ids) else 0
        code = np.unique(ids * n_fields + copy_of[field])
        recurrence = np.bincount(code // n_fields, minlength=n_ids)
        return cls(values, exact, field, starts, sizes, ids, copy_of, recurrence)


def distinct_per_field(values: np.ndarray, field: np.ndarray, n_fields: int) -> np.ndarray:
    """How many distinct finite values each field holds (``values`` already rounded)."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(n_fields, dtype=np.int64)
    ranks = np.unique(values[finite], return_inverse=True)[1].ravel()
    code = np.unique(field[finite].astype(np.int64) * (int(ranks.max()) + 1) + ranks)
    return np.bincount(code // (int(ranks.max()) + 1), minlength=n_fields)


def field_moments(rows: Rows) -> tuple[np.ndarray, ...]:
    """Per field and feature: missing values, whether constant, ``|skewness|``, kurtosis.

    The skewness and kurtosis of standardised values are unchanged by a
    change of units (``|skewness|`` also by a sign flip) and by the order of
    the rows, so a copy has its original's fingerprint.
    """
    n_fields = len(rows.sizes)
    shape = (n_fields, rows.values.shape[1])
    missing, constant = np.zeros(shape, dtype=np.int64), np.ones(shape, dtype=bool)
    skew, kurt = np.zeros(shape), np.zeros(shape)
    nonempty = np.flatnonzero(rows.sizes)
    for j in range(shape[1]):
        x = rows.values[:, j]
        finite = ~np.isnan(x)
        n = np.bincount(rows.field, weights=finite, minlength=n_fields)
        missing[:, j] = rows.sizes - n.astype(np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            high = np.maximum.reduceat(np.where(finite, x, -np.inf), rows.starts[nonempty])
            low = np.minimum.reduceat(np.where(finite, x, np.inf), rows.starts[nonempty])
            constant[nonempty, j] = ~(high > low)
            mean = np.bincount(rows.field, weights=np.where(finite, x, 0), minlength=n_fields) / n
            d = np.where(finite, x - mean[rows.field], 0.0)
            sd = np.sqrt(np.bincount(rows.field, weights=d**2, minlength=n_fields) / n)
            z = d / sd[rows.field]
            m3 = np.bincount(rows.field, weights=np.where(finite, z**3, 0), minlength=n_fields)
            m4 = np.bincount(rows.field, weights=np.where(finite, z**4, 0), minlength=n_fields)
            skew[:, j] = np.where(constant[:, j], 0.0, np.abs(m3 / n))
            kurt[:, j] = np.where(constant[:, j], 0.0, m4 / n)
    return missing, constant, skew, kurt


def candidates(
    fields: list[int], rows: Rows, moments: tuple[np.ndarray, ...]
) -> list[tuple[int, int]]:
    """Pairs of ``fields`` whose fingerprints agree, found by sorting (no loop over all pairs)."""
    missing, constant, skew, kurt = moments
    buckets: dict[tuple, list[int]] = {}
    for i in fields:
        key = (int(rows.sizes[i]), missing[i].tobytes(), constant[i].tobytes())
        buckets.setdefault(key, []).append(i)
    pairs = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        signature = np.column_stack([skew[members], kurt[members]])
        sweep = signature.sum(axis=1)
        order = np.argsort(sweep, kind="stable")
        for a, first in enumerate(order):
            for second in order[a + 1 :]:
                if sweep[second] - sweep[first] > _CANDIDATE * (1 + abs(sweep[first])):
                    break
                close = np.abs(signature[first] - signature[second])
                if np.all(close <= _CANDIDATE * (1 + np.abs(signature[first]))):
                    i, j = members[first], members[second]
                    pairs.append((min(i, j), max(i, j)))
    return sorted(pairs)


def _standardised(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column as (x - mean) / sd (constant or empty columns as 0), and which vary."""
    finite = ~np.isnan(rows)
    n = finite.sum(axis=0)
    filled = np.where(finite, rows, 0.0)
    safe = np.maximum(n, 1)
    mean = filled.sum(axis=0) / safe
    high = np.where(finite, rows, -np.inf).max(axis=0)
    low = np.where(finite, rows, np.inf).min(axis=0)
    varying = high > low
    sd = np.sqrt((np.where(finite, rows - mean, 0.0) ** 2).sum(axis=0) / safe)
    z = np.where(varying, (rows - mean) / np.where(varying, sd, 1.0), 0.0)
    return np.where(finite, z, np.nan), varying


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[bool, bool]:
    """Whether ``y = a * x + b`` fits every value, and whether it changes them."""
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False, False
    finite = ~np.isnan(x)
    x, y = x[finite], y[finite]
    if not len(x):
        return True, False
    scale = max(float(np.abs(y).max()), float(np.abs(x).max()))
    if np.ptp(x) == 0:
        same = np.ptp(y) <= TOLERANCE * scale
        return bool(same), bool(abs(y[0] - x[0]) > TOLERANCE * scale)
    dx = x - x.mean()
    slope = float((dx * (y - y.mean())).sum() / (dx**2).sum())
    intercept = float(y.mean() - slope * x.mean())
    residual = float(np.abs(y - (slope * x + intercept)).max())
    ok = residual <= TOLERANCE * float(np.abs(y).max()) and residual <= 1e-3 * np.ptp(y)
    changed = abs(slope - 1) > TOLERANCE or abs(intercept) > TOLERANCE * scale
    return bool(ok), bool(changed)


def _maps(first: np.ndarray, second: np.ndarray) -> bool | None:
    """Whether a change of units per column maps ``first`` onto ``second`` row for row
    (``None`` when not), and whether it changes any value."""
    changed = False
    for j in range(first.shape[1]):
        fits, differs = _fit(first[:, j], second[:, j])
        if not fits:
            return None
        changed |= differs
    return changed


def unit_change(first: np.ndarray, second: np.ndarray) -> tuple[bool, bool] | None:
    """Whether ``second`` holds ``first``'s rows under a change of units, in any order.

    ``None`` when it does not; otherwise ``(changed, same_order)``: whether a
    feature's values changed (``a`` not 1 or ``b`` not 0), and whether the
    rows are in the same order. Rows are aligned by sorting both fields on
    every standardised feature, the most varied first, each sign chosen by
    comparing the sorted values, so a feature with ties, a sign flip or a
    missing value does not misalign them.
    """
    if first.shape != second.shape:
        return None
    in_order = _maps(first, second)
    if in_order is not None:
        return in_order, True
    z_first, varying = _standardised(first)
    z_second, varying_second = _standardised(second)
    if not np.array_equal(varying, varying_second):
        return None
    signs, ambiguous = np.ones(first.shape[1]), []
    for j in np.flatnonzero(varying):
        a = np.sort(z_first[:, j][~np.isnan(z_first[:, j])])
        b = z_second[:, j][~np.isnan(z_second[:, j])]
        width = 1e-3 * max(1.0, float(np.abs(a).max()))
        plus = bool(np.abs(a - np.sort(b)).max() <= width)
        minus = bool(np.abs(a - np.sort(-b)).max() <= width)
        if not plus and not minus:
            return None
        signs[j] = 1.0 if plus else -1.0
        if plus and minus:
            ambiguous.append(j)
    distinct = [len(np.unique(first[:, j][~np.isnan(first[:, j])])) for j in range(len(signs))]
    priority = sorted(np.flatnonzero(varying), key=lambda j: -distinct[j])
    keys = priority[::-1]  # np.lexsort sorts by its last key first
    for flips in product((1.0, -1.0), repeat=min(len(ambiguous), _AMBIGUOUS)):
        trial = signs.copy()
        trial[ambiguous[: len(flips)]] = flips
        order_first = np.lexsort([z_first[:, j] for j in keys])
        order_second = np.lexsort([trial[j] * z_second[:, j] for j in keys])
        mapped = _maps(first[order_first], second[order_second])
        if mapped is not None:
            return mapped, False
    return None


def longest_run(first: np.ndarray, second: np.ndarray) -> tuple[int, int, int]:
    """The longest run of consecutive ids of ``second`` found in the same order in ``first``.

    ``(length, start in first, start in second)``; an id seen twice in
    ``first`` is matched at its first place.
    """
    at: dict[int, int] = {}
    for i, value in enumerate(first.tolist()):
        at.setdefault(value, i)
    best, run, previous = (0, 0, 0), 0, -2
    for j, value in enumerate(second.tolist()):
        i = at.get(value)
        if i is None:
            run, previous = 0, -2
            continue
        run = run + 1 if run and i == previous + 1 else 1
        previous = i
        if run > best[0]:
            best = (run, i - run + 1, j - run + 1)
    return best


def repeated_run(ids: np.ndarray) -> tuple[int, int, int]:
    """The longest run of consecutive ids that repeats an earlier run of the same sequence.

    ``(length, start of the earlier run, start of the repeat)``.
    """
    last: dict[int, int] = {}
    best, run, offset = (0, 0, 0), 0, None
    for j, value in enumerate(ids.tolist()):
        i = last.get(value)
        last[value] = j
        if i is None:
            run, offset = 0, None
            continue
        run = run + 1 if run and j - i == offset else 1
        offset = j - i
        if run > best[0]:
            best = (run, i - run + 1, j - run + 1)
    return best


__all__ = [
    "DIGITS",
    "RECURRING",
    "Rows",
    "candidates",
    "distinct_per_field",
    "field_moments",
    "longest_run",
    "repeated_run",
    "rounded",
    "unit_change",
]
