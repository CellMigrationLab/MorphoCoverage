"""Statistics shared by the analyses, isolated so they are written once and tested once.

Every function takes plain arrays and returns plain numbers, so it can be
read and checked on its own. The field-aware versions take the objects of an
acquisition as a list of arrays, one per field, because that is how a
microscope produced them (the Foundation of the report): whole fields are the
unit that is resampled, drawn or shuffled, never individual objects.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np
from scipy import stats

from ._checks import is_integer

# Relative size below which a mean, a spread or a variance is rounding noise
# of a zero one: a constant such as 0.1 leaves an SD of about 1e-17, not 0.
RELATIVE_TOLERANCE = 1e-12


def _scale(values: np.ndarray) -> float:
    """The mean absolute value, the scale against which rounding noise is judged."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(np.abs(finite))) if len(finite) else 0.0


def mean_is_zero(mean: float, values: np.ndarray) -> bool:
    """Whether ``mean`` is zero up to rounding, judged against the values it came from."""
    return abs(mean) <= RELATIVE_TOLERANCE * _scale(values) or mean == 0


def spread_is_zero(sd: float, values: np.ndarray) -> bool:
    """Whether a standard deviation is zero up to rounding (a constant feature)."""
    return not sd > RELATIVE_TOLERANCE * _scale(values)


def cohen_d(first: np.ndarray, second: np.ndarray) -> float:
    """Cohen's d with the pooled sample variance, as the lab's Plot&Stats notebook computes it.

    ``(mean(first) - mean(second)) / sqrt(pooled variance)``; NaN when a side
    has fewer than two values or the pooled variance is zero (up to rounding:
    a constant feature is not given a d from its rounding noise).
    """
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = ((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)) / (n1 + n2 - 2)
    if spread_is_zero(math.sqrt(max(pooled, 0.0)), np.concatenate([first, second])):
        return float("nan")
    return float((first.mean() - second.mean()) / np.sqrt(pooled))


def jcs_effect_size(treated: np.ndarray, control: np.ndarray) -> float:
    """The JCS effect size (Marcotti et al. 2026, Eqn 2): treated mean minus control mean,
    over the control standard deviation; NaN when the control SD is undefined or zero
    (up to rounding)."""
    if len(control) < 2 or len(treated) == 0:
        return float("nan")
    sd = control.std(ddof=1)
    if spread_is_zero(sd, np.concatenate([treated, control])):
        return float("nan")
    return float((treated.mean() - control.mean()) / sd)


def field_sample(fields: Sequence[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` objects as a microscope stopping at ``n`` has them: whole fields in random order,
    with a random part of the last field. Needs at least ``n`` objects in total."""
    total = sum(len(f) for f in fields)
    if total < n:
        raise ValueError(f"asked for {n} objects from fields holding {total}")
    taken, count = [], 0
    for i in rng.permutation(len(fields)):
        field = fields[i]
        need = n - count
        if len(field) >= need:
            taken.append(rng.choice(field, size=need, replace=False))
            break
        taken.append(field)
        count += len(field)
    return np.concatenate(taken)


def fields_holding(
    fields: Sequence[np.ndarray], n: int, rng: np.random.Generator, draws: int = 50
) -> float:
    """Mean number of fields, in random order, whose objects reach ``n`` objects."""
    sizes = np.array([len(f) for f in fields])
    if sizes.sum() < n:
        raise ValueError(f"asked for {n} objects from fields holding {int(sizes.sum())}")
    counts = []
    for _ in range(draws):
        cumulative = np.cumsum(sizes[rng.permutation(len(sizes))])
        reached = np.flatnonzero(cumulative >= n)
        counts.append(int(reached[0]) + 1 if len(reached) else len(sizes))
    return float(np.mean(counts))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaN entries are ignored and kept NaN.

    ``q_i = min over j >= i of (m p_(j) / j)`` over the ``m`` finite p-values in
    increasing order, capped at 1, as fiberfeatures corrects its per-feature
    randomization tests before gating them.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return q
    order = finite[np.argsort(p[finite], kind="stable")]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def bootstrap_fields(
    fields: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """``statistic`` of all objects, and its percentile interval from resampling whole fields.

    Each resample draws as many fields as were acquired, with replacement, and
    pools their objects; the interval is the ``confidence`` percentile range
    of the statistic over the ``n_boot`` resamples on which it is defined. A
    resample that drew only empty fields has no objects and is left out, so
    with many empty fields the interval is conditional on a resample holding
    at least one object. Fields without objects take
    part in the resampling (they were acquired) and contribute no value. NaN
    bounds when fewer than two fields hold objects.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    values = [np.asarray(f, dtype=float) for f in fields]
    pooled = np.concatenate(values) if values else np.array([])
    point = float(statistic(pooled)) if len(pooled) else float("nan")
    if sum(len(f) > 0 for f in values) < 2:
        return point, float("nan"), float("nan")
    draws = np.empty(n_boot)
    k = len(values)
    for b in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        sample = np.concatenate([values[i] for i in chosen])
        draws[b] = statistic(sample) if len(sample) else np.nan
    defined = draws[np.isfinite(draws)]
    if len(defined) < 2:
        return point, float("nan"), float("nan")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(defined, [tail, 100 - tail])
    return point, float(low), float(high)


def bootstrap_nested(
    units: Sequence[Sequence[np.ndarray]],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """``statistic`` of all objects, and its interval from resampling whole units.

    ``units`` are the technical repeats (a well, a dish) of one cluster, each a
    list of its fields' values. Each resample draws as many units as there
    are, with replacement, and keeps every drawn unit's fields as they are
    (Ren et al. 2010, "Nonparametric bootstrapping for hierarchical data":
    resampling the top level with replacement and the lower level without).
    Resampling the fields again inside each unit would count the
    field-to-field variance twice: in a check with 10 or 20 wells that
    interval covered the mean 98% of the time for a nominal 95%. A
    well-to-well shift widens this interval instead of passing for
    field-to-field noise. With a single unit this is ``bootstrap_fields`` on
    its fields. With two or three units no resampling scheme gives a reliable
    interval (a units-only resample of 2 wells covered 49% of the time), so a
    caller should say how many units there are. As there, resamples holding
    no object are left out of the percentiles. NaN bounds when fewer than two
    fields hold objects.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    groups = [[np.asarray(f, dtype=float) for f in unit] for unit in units if len(unit)]
    if len(groups) == 1:
        return bootstrap_fields(groups[0], statistic, n_boot=n_boot, rng=rng, confidence=confidence)
    fields = [f for unit in groups for f in unit]
    pooled = np.concatenate(fields) if fields else np.array([])
    point = float(statistic(pooled)) if len(pooled) else float("nan")
    if sum(len(f) > 0 for f in fields) < 2:
        return point, float("nan"), float("nan")
    pooled_units = [np.concatenate(unit) for unit in groups]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([pooled_units[i] for i in chosen])
        draws[b] = statistic(sample) if len(sample) else np.nan
    defined = draws[np.isfinite(draws)]
    if len(defined) < 2:
        return point, float("nan"), float("nan")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(defined, [tail, 100 - tail])
    return point, float(low), float(high)


def _dense_codes(code: np.ndarray) -> np.ndarray:
    """Codes renumbered 0..K-1 by value; returned as they are when they already are."""
    # bincount allocates max + 1 entries, so only codes that could be dense qualify.
    if code.dtype.kind in "iu" and len(code) and code.min() == 0 and code.max() < len(code):
        present = np.bincount(code)
        if present.all():
            return code
    return np.unique(code, return_inverse=True)[1]


def nested_variance_components(values: np.ndarray, codes: Sequence[np.ndarray]) -> np.ndarray:
    """Method-of-moments variance components of a nested random design (Henderson's method I).

    ``codes`` gives, coarsest level first, the integer unit of each value at
    each level (units must nest: every unit of a level lies inside one unit
    of the level before it). Returns one component per level plus the
    residual (observation) component, in the same order.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or np.isnan(values).any():
        raise ValueError("need at least two non-missing values")
    levels = [np.zeros(n, dtype=int), *[np.asarray(c) for c in codes], np.arange(n)]
    sizes, totals = [], []
    for code in levels:
        inverse = _dense_codes(code)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=values)
        sizes.append((inverse, counts))
        totals.append(float(np.sum(sums**2 / counts)))
    depth = len(levels)

    def coefficient(level: int, component: int) -> float:
        # E[T_level] = N mu^2 + sum_k c[level, k] v_k, with T_level = sum_u n_u ybar_u^2.
        if component <= level:
            return float(n)
        inverse, counts = sizes[level]
        finer_inverse, finer_counts = sizes[component]
        parent_of_finer = np.zeros(len(finer_counts), dtype=int)
        parent_of_finer[finer_inverse] = inverse
        squared = np.bincount(parent_of_finer, weights=finer_counts.astype(float) ** 2)
        return float(np.sum(squared / counts))

    rows, rhs = [], []
    for level in range(1, depth):
        rows.append([coefficient(level, k) - coefficient(level - 1, k) for k in range(1, depth)])
        rhs.append(totals[level] - totals[level - 1])
    return np.linalg.solve(np.array(rows), np.array(rhs))


def cluster_size(counts: np.ndarray) -> float:
    """``sum n_j^2 / N`` over the clusters holding observations: the size that sets the design
    effect of unequal clusters (the mean size when all are equal)."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    return float(np.sum(counts**2) / np.sum(counts)) if len(counts) else float("nan")


def design_effect(v_fov: float, v_obs: float, size: float) -> float | None:
    """``(m v_fov + v_obs) / (v_fov + v_obs)``: the variance of a mean of whole fields over that of
    as many random objects, for cluster size ``m``; negative components count as zero, ``None``
    when both are zero."""
    v_fov, v_obs = max(float(v_fov), 0.0), max(float(v_obs), 0.0)
    total = v_fov + v_obs
    return None if total <= 0 else (size * v_fov + v_obs) / total


@lru_cache(maxsize=4096)
def sd_upper_bound_factor(df: int, confidence: float = 0.95) -> float:
    """Factor turning a sample SD into the upper bound of its confidence interval.

    The chi-square bound ``sqrt(df / chi2.ppf((1 - confidence) / 2, df))``.
    """
    if not is_integer(df) or df < 1:
        raise ValueError("df must be an integer of at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return math.sqrt(df / stats.chi2.ppf((1 - confidence) / 2, df))


def jcb_required_objects(cv_objects: float, target_cv: float) -> int:
    """The JCB number of objects to average: ``ceil((cv_objects / target_cv) ** 2)``.

    The square is rounded to nine decimals first, so floating-point error never
    adds one object (0.07 / 0.01 squares to 49.000000000000014, which is 49).
    """
    return math.ceil(round((cv_objects / target_cv) ** 2, 9))


def fovs_to_detect(fov_incidence: float, detection_probability: float, occurrences: int) -> int:
    """Smallest n with ``P(Binomial(n, fov_incidence) >= occurrences) >= detection_probability``.

    Uses ``scipy.stats.binom.sf``, which is monotone increasing in ``n``;
    the answer is found by doubling then bisection.
    """
    for value, label in ((fov_incidence, "fov_incidence"), (detection_probability, "detection")):
        if not 0 < value < 1:
            raise ValueError(f"{label} must be between 0 and 1")
    if not is_integer(occurrences) or occurrences < 1:
        raise ValueError("occurrences must be a positive integer")
    occurrences = int(occurrences)

    def detected(n: int) -> bool:
        return stats.binom.sf(occurrences - 1, n, fov_incidence) >= detection_probability

    low, high = occurrences - 1, occurrences
    while not detected(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if detected(middle) else (middle, high)
    return high


def first_detection_fovs(fov_incidence: float, detection_probability: float) -> int:
    """FOVs to see one occurrence: ``ceil(log(1 - beta) / log(1 - p))``, as ``fovs_to_detect``.

    The closed form is computed from ``fovs_to_detect`` itself, since the
    logarithms can round across an integer at an exact boundary (``p = 0.2``,
    ``beta = 0.36`` gives 2.0000000000000004, which would ceil to 3).
    """
    return fovs_to_detect(fov_incidence, detection_probability, 1)


def srs_relative_rmse(variance: float, mean: float, n: float, total: float) -> float:
    """Expected relative RMSE of the mean of ``n`` objects drawn at random from ``total``.

    Simple random sampling without replacement, with finite-population
    correction: ``sqrt(variance / n * (1 - n / total)) / |mean|``. This is the
    error object-number rules assume.
    """
    if not 0 < n <= total:
        raise ValueError(f"n must be between 1 and total ({total}), not {n}")
    if variance < 0:
        raise ValueError("variance must not be negative")
    if mean == 0:
        raise ValueError("the relative error of a zero mean is undefined")
    return math.sqrt(variance / n * (1 - n / total)) / abs(mean)


__all__ = [
    "RELATIVE_TOLERANCE",
    "benjamini_hochberg",
    "bootstrap_fields",
    "bootstrap_nested",
    "cluster_size",
    "cohen_d",
    "design_effect",
    "field_sample",
    "fields_holding",
    "first_detection_fovs",
    "fovs_to_detect",
    "jcb_required_objects",
    "jcs_effect_size",
    "mean_is_zero",
    "nested_variance_components",
    "sd_upper_bound_factor",
    "spread_is_zero",
    "srs_relative_rmse",
]
