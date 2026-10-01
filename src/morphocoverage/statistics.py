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

from ._checks import is_integer, is_number

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


PAPER_DESCRIPTORS = ("mean", "median", "sd", "iqr")


def describe(
    values: np.ndarray, descriptors: tuple[str, ...] = PAPER_DESCRIPTORS
) -> dict[str, float]:
    """Descriptors of one sample, as in Marcotti et al.'s Table 1; NaN with fewer than 2 values.

    Quantiles use linear interpolation, matching the companion code's
    ``np.percentile`` and ``pandas.quantile``; SD uses ``ddof=1``.
    """
    if len(values) < 2:
        return {name: float("nan") for name in descriptors}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    available = {
        "mean": float(np.mean(values)),
        "median": float(median),
        "sd": float(np.std(values, ddof=1)),
        "iqr": float(q3 - q1),
    }
    return {name: available[name] for name in descriptors}


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
    with a random part of the last field. Needs at least ``n`` objects in total; ``n = 0``
    gives an empty sample."""
    if not is_integer(n) or n < 0:
        raise ValueError(f"n must be a non-negative integer, not {n!r}")
    if n == 0:
        return np.array([], dtype=float)
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
    """Mean number of fields, in random order, whose objects reach ``n`` objects (0 for 0)."""
    if not is_integer(n) or n < 0:
        raise ValueError(f"n must be a non-negative integer, not {n!r}")
    if n == 0:
        return 0.0
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


def _interval(
    point: float, draws: np.ndarray, units: int, confidence: float
) -> tuple[float, float, float]:
    """The percentile interval of ``draws``, widened for the number of units resampled.

    A percentile bootstrap over ``k`` units under-covers: its variance is too
    small by ``(k - 1) / k`` and it uses normal rather than t quantiles (a
    nominal 95% covered 81-85% of the time with 5 wells and 87-89% with 10).
    Each half-width, from the point to the percentile, is multiplied by
    ``sqrt(k / (k - 1)) * t_{k-1} / z`` at the two-sided ``confidence`` level,
    which keeps the interval's asymmetry and brings it close to nominal.
    """
    defined = draws[np.isfinite(draws)]
    if len(defined) < 2 or units < 2 or not math.isfinite(point):
        return point, float("nan"), float("nan")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(defined, [tail, 100 - tail])
    widen = small_sample_widening(units, confidence)
    return point, float(point - (point - low) * widen), float(point + (high - point) * widen)


def bootstrap_fields(
    fields: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """``statistic`` of all objects, and its interval from resampling whole fields.

    Each resample draws as many fields as were acquired, with replacement, and
    pools their objects. The interval is the ``confidence`` percentile range
    of the statistic over the ``n_boot`` resamples on which it is defined,
    widened for the small number of fields holding objects (``_interval``). A
    resample that drew only empty fields has no objects and is left out, so
    with many empty fields the interval is conditional on a resample holding
    at least one object. Fields without objects take part in the resampling
    (they were acquired) and contribute no value. NaN bounds when fewer than
    two fields hold objects.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    values = [np.asarray(f, dtype=float) for f in fields]
    pooled = np.concatenate(values) if values else np.array([])
    point = float(statistic(pooled)) if len(pooled) else float("nan")
    filled = sum(len(f) > 0 for f in values)
    if filled < 2:
        return point, float("nan"), float("nan")
    draws = np.empty(n_boot)
    k = len(values)
    for b in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        sample = np.concatenate([values[i] for i in chosen])
        draws[b] = statistic(sample) if len(sample) else np.nan
    return _interval(point, draws, filled, confidence)


def bootstrap_nested(
    units: Sequence[Sequence[np.ndarray]],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """``statistic`` of all objects, and its interval from resampling whole units.

    ``units`` are the top-level units of one cluster (technical repeats such
    as wells, or biological repeats), each a list of its fields' values (all
    the fields of all its lower levels). Each resample draws as many units as
    hold objects, with replacement, and keeps every drawn unit's fields as
    they are (Ren et al. 2010, "Nonparametric bootstrapping for hierarchical
    data": resample the top level with replacement and the lower levels
    without). Resampling the lower levels again would count their variance
    twice. A unit whose fields are all empty holds no information about the
    statistic and is left out; with a single unit holding objects this is
    ``bootstrap_fields`` on that unit's fields. The interval is widened for the
    number of units (``_interval``); with two or three units no resampling
    scheme gives a reliable interval, so a caller should say how many units
    there are. NaN bounds when fewer than two fields hold objects.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    groups = [[np.asarray(f, dtype=float) for f in unit] for unit in units]
    groups = [unit for unit in groups if any(len(f) for f in unit)]
    if len(groups) <= 1:
        fields = groups[0] if groups else []
        return bootstrap_fields(fields, statistic, n_boot=n_boot, rng=rng, confidence=confidence)
    pooled_units = [np.concatenate(unit) for unit in groups]
    point = float(statistic(np.concatenate(pooled_units)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        draws[b] = statistic(np.concatenate([pooled_units[i] for i in chosen]))
    return _interval(point, draws, len(groups), confidence)


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
    of the level before it; a unit number reused under two parents is
    refused). Returns one component per level plus the residual (observation)
    component, in the same order. Method-of-moments components are unbiased
    but can be negative when a level's true variance is small; callers decide
    how to report them.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite, non-missing values")
    levels = [np.zeros(n, dtype=int), *[np.asarray(c) for c in codes], np.arange(n)]
    for depth in range(1, len(levels) - 1):
        parent, child = _dense_codes(levels[depth - 1]), _dense_codes(levels[depth])
        parents_per_child = np.bincount(
            np.unique(np.stack([child, parent]), axis=1)[0], minlength=child.max() + 1
        )
        if (parents_per_child > 1).any():
            raise ValueError(
                f"the units of level {depth} do not nest in those of level {depth - 1}: "
                "number each unit uniquely (not within its parent)"
            )
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
    try:
        return np.linalg.solve(np.array(rows), np.array(rhs))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "the levels cannot be told apart (two levels have the same units, or a level "
            "has one observation per unit)"
        ) from error


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
    if not (math.isfinite(cv_objects) and cv_objects >= 0):
        raise ValueError(f"cv_objects must be a finite number of at least 0, not {cv_objects}")
    if not (math.isfinite(target_cv) and target_cv > 0):
        raise ValueError(f"target_cv must be a finite positive number, not {target_cv}")
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


# The mean-precision rule with the field as the unit can estimate two different means. Both are
# kept, under distinct names, because they answer different questions; they agree
# when every field holds as many objects, and differ most when field size and the
# phenotype are related (a few crowded fields of one state, many sparse ones of another).


def object_weighted_mean_se(sums: np.ndarray, counts: np.ndarray) -> tuple[float, float, int]:
    """The object-weighted mean of the acquired fields and its standard error, fields as the unit.

    Estimand: the mean over the objects of the sampled area, every object
    weighing the same (a crowded field weighs more), which is the mean a
    pooled analysis of the objects reports. Estimate: the ratio
    ``m = sum(T_j) / sum(N_j)`` of the per-field sums ``T_j`` and object
    counts ``N_j``, with the ratio-estimator (cluster sampling) variance

        Var(m) = k' / (k' - 1) * sum((T_j - m N_j) ** 2) / sum(N) ** 2

    over the ``k'`` fields holding objects. A field without objects has a
    residual of exactly 0 and no between-field information, so it enters
    neither the ``k' / (k' - 1)`` factor nor the ``k' - 1`` degrees of freedom
    of a bound on this SE (counting empty fields there made a nominal 95%
    upper bound cover 37% of the time with 90% of fields empty). It is still
    acquisition effort: the SE of this mean over ``K`` acquired fields like
    these is about ``se * sqrt(k / K)`` with ``k`` every acquired field.

    The variance is a first-order (linearization) approximation. When field
    size and field mean are related it can be several times smaller than the
    spread of ``m`` over resampled whole fields (relative to the mean, 0.013
    against 0.044 on 8 fields of 2 to 150 objects whose crowded fields hold a
    different state).

    Returns ``(m, se, k')``. Needs two fields with objects.
    """
    sums, counts = np.asarray(sums, dtype=float), np.asarray(counts, dtype=float)
    filled = counts > 0
    k = int(filled.sum())
    if k < 2:
        raise ValueError("the SE of a mean over fields needs at least 2 fields with objects")
    total = float(counts.sum())
    mean = float(sums.sum()) / total
    residual = float(np.sum((sums[filled] - mean * counts[filled]) ** 2))
    return mean, math.sqrt(k / (k - 1) * residual) / total, k


def field_weighted_mean_se(field_means: np.ndarray) -> tuple[float, float, int]:
    """The unweighted mean of the field means and its standard error, fields as the unit.

    Estimand: the mean of a field's average, every field with objects weighing
    the same whatever its object count, which is what an analysis with the
    field as the unit (one value per field) compares. Estimate: the mean of
    the ``k'`` field means and its SE, ``SD(field means, ddof=1) / sqrt(k')``,
    with ``k' - 1`` degrees of freedom. Fields without objects have no mean
    and are left out, so the number of fields this SE projects to counts
    fields with objects: the SD of one field's mean is ``se * sqrt(k')``.

    Returns ``(mean, se, k')``. Needs two field means.
    """
    means = np.asarray(field_means, dtype=float)
    k = len(means)
    if k < 2:
        raise ValueError("the SE of a mean over fields needs at least 2 fields with objects")
    return float(means.mean()), float(means.std(ddof=1)) / math.sqrt(k), k


def small_sample_widening(units: int, confidence: float = 0.95) -> float:
    """``sqrt(k / (k - 1)) * t_{k-1} / z``: the widening of a bootstrap over ``k`` units.

    A bootstrap over ``k`` resampled units understates the variance by
    ``(k - 1) / k`` and uses normal rather than t quantiles, so its percentile
    half-widths (``_interval``) and its standard errors (the FOV bootstrap's
    stopping SE) are multiplied by this factor at the two-sided ``confidence``
    level (1.70 for 5 units, 1.20 for 10, 1.08 for 20 at 95%). Needs 2 units.
    """
    if not is_integer(units) or units < 2:
        raise ValueError("the widening needs an integer number of units, at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    level = (1 + confidence) / 2
    return float(
        math.sqrt(units / (units - 1)) * stats.t.ppf(level, units - 1) / stats.norm.ppf(level)
    )


def kish_effective_fields(counts: np.ndarray) -> float:
    """Kish's effective number of fields, ``(sum N_j) ** 2 / sum N_j ** 2``.

    ``counts`` are the object counts ``N_j`` of the fields (empty fields add
    nothing). Equal counts give the number of fields holding objects; one
    crowded field among sparse ones gives about 1 (1000 and 5 objects: 1.01),
    since the object-weighted mean is then that field's mean. It is the degrees
    of freedom basis of the FOV-unit JCB bound (effective fields minus one);
    0 when no field holds objects.
    """
    counts = np.asarray(counts, dtype=float)
    squares = float(np.sum(counts**2))
    # Rounded, so equal counts give their exact number of fields.
    return round(float(np.sum(counts)) ** 2 / squares, 9) if squares > 0 else 0.0


@lru_cache(maxsize=4096)
def effective_sd_upper_bound_factor(df: float, confidence: float = 0.95) -> float:
    """``sd_upper_bound_factor`` for a real number of degrees of freedom of at least 1.

    The same chi-square bound, ``sqrt(df / chi2.ppf((1 - confidence) / 2, df))``,
    for a degrees-of-freedom basis that need not be whole, such as Kish's
    effective number of fields minus one (``kish_effective_fields``); at an
    integer it equals ``sd_upper_bound_factor``.
    """
    if not (is_number(df) and df >= 1):
        raise ValueError("df must be a number of at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return math.sqrt(df / stats.chi2.ppf((1 - confidence) / 2, df))


def median_standard_error(values: np.ndarray) -> float | None:
    """The Monte Carlo standard error of the median of ``values``, or None below two values.

    ``sqrt(pi / 2) * SD / sqrt(n)``, the large-sample standard error of the
    median of a normal sample: an approximation, used to say how much a
    median over a few random orders or draws would move under another seed
    (a median stop of 8 fields over 5 orders with SD 3 moves by about 1.7).
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return None
    return float(math.sqrt(math.pi / 2) * finite.std(ddof=1) / math.sqrt(len(finite)))


__all__ = [
    "PAPER_DESCRIPTORS",
    "RELATIVE_TOLERANCE",
    "benjamini_hochberg",
    "bootstrap_fields",
    "bootstrap_nested",
    "cluster_size",
    "cohen_d",
    "describe",
    "design_effect",
    "effective_sd_upper_bound_factor",
    "field_sample",
    "field_weighted_mean_se",
    "fields_holding",
    "first_detection_fovs",
    "fovs_to_detect",
    "jcb_required_objects",
    "jcs_effect_size",
    "kish_effective_fields",
    "mean_is_zero",
    "median_standard_error",
    "nested_variance_components",
    "object_weighted_mean_se",
    "sd_upper_bound_factor",
    "small_sample_widening",
    "spread_is_zero",
    "srs_relative_rmse",
]
