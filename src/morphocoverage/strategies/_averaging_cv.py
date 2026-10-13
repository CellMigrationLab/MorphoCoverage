"""The averaging-CV arithmetic of the JCB/Zweifach criterion (``scalar_precision``).

Pure functions of plain arrays and numbers: the object-level CVs, the FOV-unit
ratio-estimator CV, the paper's required counts, the check of whether FOV
size and FOV mean are related, and the check of whether the values look
normal enough for the chi-square SD bound. ``scalar_precision`` re-exports the
public names.

Shared-private: ``_field_size_check``, ``_object_cv_from_moments``,
``_no_spread``, ``_FovSummary``, ``_merged_moments``, ``_central_sums``,
``_missing_reason`` and ``_shape_warning`` are used by ``scalar_precision``
(its full and incremental paths), ``_shape_warning`` also by ``jcb_planning``;
they are private to the package and may change with it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import stats

from .._repeat_statistics import sarle
from ..statistics import (
    RELATIVE_TOLERANCE,
    jcb_required_objects,
    kish_effective_fields,
    object_weighted_mean_se,
    sd_upper_bound_factor,
    small_sample_widening,
    spread_is_zero,
)


def object_averaging_cv(values: np.ndarray) -> dict[str, float]:
    """Between-object CV and averaging CV of the mean, treating objects as independent."""
    return _object_cv_from_moments(
        len(values), float(np.mean(values)), float(np.std(values, ddof=1))
    )


def _object_cv_from_moments(n: int, mean: float, sd: float) -> dict[str, float]:
    """Cell-level and averaging CVs, point and upper-95, from count, mean and SD."""
    cv_objects = sd / mean
    cv_objects_upper = cv_objects * sd_upper_bound_factor(n - 1)
    return {
        "mean": mean,
        "cv_objects": cv_objects,
        "cv_objects_upper_95": cv_objects_upper,
        "cv_mean": cv_objects / math.sqrt(n),
        "cv_mean_upper_95": cv_objects_upper / math.sqrt(n),
    }


# The whole-field bootstrap of the FOV unit's upper-95 bound: resamples (owner, 2026-10-05).
BOUND_RESAMPLES = 400


def fov_averaging_cv(sums: np.ndarray, counts: np.ndarray, random_state: int = 0) -> dict[str, Any]:
    """Averaging CV of the object-weighted mean with FOVs as the sampling unit.

    ``statistics.object_weighted_mean_se`` (the ratio estimator for cluster
    sampling, with per-FOV sums ``T_j`` and object counts ``N_j``) as a
    fraction of the mean. Estimand: the per-object mean over the sampled
    area. Only the ``k'`` FOVs holding objects enter the ``k' / (k' - 1)``
    factor. ``effective_fovs`` is Kish's effective number of FOVs,
    ``(sum N_j) ** 2 / sum N_j ** 2`` (``statistics.kish_effective_fields``):
    ``k'`` when every FOV holds as many objects, about 1 when one crowded FOV
    holds nearly all of them; below 2 the estimator reports
    ``insufficient_data``.

    The upper bound is a bootstrap of whole fields (owner, 2026-10-05; it
    replaces the chi-square bound on Kish's effective FOVs minus one, which
    covered 70-85% with over-dispersed field sizes): ``_bootstrap_upper``,
    seeded with ``random_state``. Acquired FOVs without objects are still
    effort: they count in ``required_fovs`` and in the pilot projection,
    since future FOVs may be empty too. The unweighted mean of FOV means
    (``field_weighted_mean_se``) is a different estimand;
    ``sampling_unit.object_rule_check`` uses it.
    """
    mean, se, filled = object_weighted_mean_se(sums, counts)
    cv_mean = se / mean
    effective = kish_effective_fields(counts)
    upper = _bootstrap_upper(sums, counts, cv_mean, random_state) if effective >= 2 else math.nan
    return {
        "mean": mean,
        "cv_mean": cv_mean,
        "cv_mean_upper_95": upper,
        "fovs_with_objects": filled,
        "effective_fovs": effective,
    }


def _bootstrap_upper(
    sums: np.ndarray, counts: np.ndarray, cv_mean: float, random_state: int
) -> float:
    """The upper-95 bound of the FOV-unit averaging CV from a bootstrap of whole fields.

    The ``k'`` FOVs holding objects are resampled with replacement
    (``BOUND_RESAMPLES`` times; empty FOVs add nothing to the CV, so padding
    with them leaves the bound as it is) and the averaging CV ``c*`` is
    recomputed on each resample. The bound is ``c`` times the 95th percentile
    of ``c / c*``, as the chi-square bound divides by a low quantile of the SD
    estimate's spread, with the log half-width widened for ``k'`` units
    (``statistics.small_sample_widening(k', 0.90)``, the one-sided 95% level).
    The 95th percentile of ``c*`` itself covered only 23-77% with over-dispersed field
    sizes. A resample without spread between its FOVs gives an infinite ratio: when the 95th
    percentile falls on one, the bound is NaN (the estimator reports ``insufficient_data``),
    as with 2 or 3 FOVs (one FOV drawn every time in 50% and 11% of the resamples).

    Coverage of the true averaging CV at a nominal 95%, simulated with 1000 wells per cell
    (5-40 FOVs; field sizes Poisson or negative binomial with mean 20 objects and dispersion 1
    or 0.5; lognormal field effects of SD 0.1 or 0.3): with Poisson sizes 93-99%; with
    over-dispersed sizes 89-98% up to 10 FOVs, but 83-89% with 20-40 FOVs, where the
    ratio-estimator CV itself is below the truth in most wells. The chi-square bound on
    Kish's effective FOVs at the same level covered 79-96%. Resamples are drawn over the FOVs
    sorted by count and sum, so the bound depends on the FOVs' values, never on their order,
    and the full and incremental paths agree. Between seeds the bound moves by about 7% of
    its value with 10 FOVs, 4% with 20 and 2% with 40; one bound costs well under a
    millisecond up to 40 FOVs.
    """
    filled = counts > 0
    order = np.lexsort((sums[filled], counts[filled]))
    field_sums, field_counts = sums[filled][order], counts[filled][order]
    units = len(field_sums)
    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, units, (BOUND_RESAMPLES, units))
    drawn_sums, drawn_counts = field_sums[draws], field_counts[draws]
    totals = drawn_counts.sum(axis=1)
    # fallback: a resample without observations: NaN, left out
    with np.errstate(divide="ignore", invalid="ignore"):
        means = drawn_sums.sum(axis=1) / totals
        residuals = np.sum((drawn_sums - means[:, None] * drawn_counts) ** 2, axis=1)
        resampled = np.sqrt(units / (units - 1) * residuals) / totals / means
        log_ratio = np.log(cv_mean) - np.log(resampled)
    # No spread (one FOV drawn every time, or FOVs of one mean, up to rounding) or no positive
    # mean: this resample cannot bound the CV.
    log_ratio[~(resampled > RELATIVE_TOLERANCE) | ~np.isfinite(log_ratio)] = np.inf
    high = float(np.quantile(log_ratio, 0.95, method="inverted_cdf"))
    if not math.isfinite(high):
        return math.nan
    return cv_mean * math.exp(_one_sided_widening(units) * high)


@lru_cache(maxsize=1024)
def _one_sided_widening(units: int) -> float:
    """``statistics.small_sample_widening`` at the one-sided 95% level, cached."""
    return small_sample_widening(units, 0.90)


def required_objects(cv_objects: float, target_cv: float) -> int:
    """The paper's number of objects to average (``statistics.jcb_required_objects``)."""
    return jcb_required_objects(cv_objects, target_cv)


def required_fovs(cv_mean: float, n_fovs: int, target_cv: float) -> int:
    """FOVs needed for ``target_cv``, since Var(mean) scales as 1/k under FOV resampling.

    Rounded as ``jcb_required_objects``: floating-point error never adds a FOV.
    The estimator does not use it (its stopping rule compares the projected CV
    with the target), so at a rounding boundary the projection can first reach
    the target one FOV later than this count (a pilot CV of 0.05 from 2 FOVs
    and a target of 0.01: 50 here, 51 by the projection).
    """
    return math.ceil(round(n_fovs * (cv_mean / target_cv) ** 2, 9))


# Up to this many FOVs the Spearman p-value is exact (every pairing enumerated,
# 8! = 40,320); above it scipy's t approximation, which is close from about 9 on.
_EXACT_SPEARMAN_MAX = 8


@lru_cache(maxsize=_EXACT_SPEARMAN_MAX)
def _pairings(n: int) -> np.ndarray:
    """Every permutation of ``range(n)``, one per row."""
    return np.array(list(itertools.permutations(range(n))))


def _spearman(sizes: np.ndarray, means: np.ndarray) -> tuple[float, float]:
    """Spearman's rho and its two-sided p-value; exact for few FOVs.

    With few FOVs scipy's t approximation is far off: a perfect rank order of 4
    FOVs gives p = 0, where the exact permutation p-value is 2/24 = 0.083, and
    under independence it warned in 7-8% of 4-5 FOV acquisitions at 5%. The
    exact p-value permutes the ranks of the means against those of the sizes
    (ties keep their mid-ranks), so it holds for any tie pattern.
    """
    result = stats.spearmanr(sizes, means)
    rho = float(result.statistic)
    if len(sizes) > _EXACT_SPEARMAN_MAX:
        return rho, float(result.pvalue)
    x, y = stats.rankdata(sizes), stats.rankdata(means)
    x, y = x - x.mean(), y - y.mean()
    permuted = (y[_pairings(len(y))] @ x) / math.sqrt(float((x**2).sum() * (y**2).sum()))
    # A small tolerance, so rounding cannot leave out pairings as extreme as the observed one.
    return rho, float(np.mean(np.abs(permuted) >= abs(rho) - 1e-9))


def _field_size_check(
    sums: np.ndarray, counts: np.ndarray, rows: str = "objects"
) -> dict[str, Any]:
    """Whether FOV size and FOV mean are related, which the ratio-estimator CV does not see.

    Spearman correlation between the object count and the mean of the FOVs
    holding objects (from 4 such FOVs, when both vary), with an exact
    permutation p-value up to 8 such FOVs (``_spearman``). When it is
    significant at 5%, a warning: the ratio estimator's linearized variance can
    then be several times smaller than the spread of the mean over resampled
    whole FOVs (0.013 against 0.051 in the check of ``object_weighted_mean_se``).
    With 4 FOVs the smallest exact p-value is 1/12, so a warning needs at least
    5 FOVs with objects. The check has little power (a slope of -0.1 was
    flagged in 19% of 12-FOV and 31% of 20-FOV acquisitions), and with many
    small FOVs the t approximation warns slightly too often (6-8% at a nominal
    5% with 20-40 FOVs of about 3 objects): no warning is not evidence that
    FOV size and FOV mean are unrelated.
    """
    filled = counts > 0
    sizes, means = counts[filled], sums[filled] / counts[filled]
    if len(sizes) < 4 or np.ptp(sizes) == 0 or np.ptp(means) == 0:
        return {}
    rho, p_value = _spearman(sizes, means)
    found: dict[str, Any] = {"field_size_correlation": rho, "field_size_correlation_p": p_value}
    if p_value < 0.05:
        found["field_size_warning"] = (
            f"field size and field mean are related (Spearman rho {rho:.2f}, p = "
            f"{p_value:.2g}, over {len(sizes)} fields with {rows}): the per-field averaging CV "
            "can then be several times too small; check it against a bootstrap of whole fields "
            "(FOVBootstrapEstimator, centroid summary) or the mean of field means "
            "(statistics.field_weighted_mean_se)"
        )
    return found


def _no_spread(sd: float, mean: float) -> bool:
    """Whether an SD (or SE) is zero up to rounding for non-negative values of this mean.

    ``statistics.spread_is_zero`` judges rounding against the values' mean
    absolute value, which is their mean here, since negative values are
    refused before this check.
    """
    return spread_is_zero(sd, np.array([mean]))


# The shape check of the chi-square SD bound: from this many objects, and flagged above
# these values (excess kurtosis, Sarle's bimodality coefficient). Normal values of 10, 20 and
# 100 objects are flagged 9%, 5% and 0.4% of the time; a 90/10 two-state readout 67%, 88%
# and 100%.
_SHAPE_MIN_OBJECTS = 8
_MAX_EXCESS_KURTOSIS = 2.0
_MAX_BIMODALITY = 5 / 9


def _central_sums(values: np.ndarray) -> tuple[int, float, float, float]:
    """Count and the sums of the 2nd, 3rd and 4th powers of the deviations from the mean."""
    if not len(values):
        return 0, 0.0, 0.0, 0.0
    deviations = values - values.mean()
    return (
        len(values),
        float(np.sum(deviations**2)),
        float(np.sum(deviations**3)),
        float(np.sum(deviations**4)),
    )


def _shape_warning(n: int, m2: float, m3: float, m4: float, rows: str) -> str | None:
    """A caution when values are far from normal for the chi-square SD bound, else ``None``.

    The upper bound of the SD (the paper's, ``sd_upper_bound_factor``) holds
    for normal values. Its coverage depends on the kurtosis: a readout with a
    minority state (90% of cells at one level, 10% at four times it) has a
    large excess kurtosis, and the bound then covered the true SD 68-88% of the
    time for a nominal 97.5% (audit 4, P3-7 and P4-7; a lognormal readout,
    sigma 1, left the plan short in 45% of pilots). The sample excess kurtosis
    and Sarle's bimodality coefficient (with the usual small-sample
    corrections, as ``scipy.stats`` with ``bias=False``) are checked from
    ``_SHAPE_MIN_OBJECTS`` values on; ``m2``, ``m3`` and ``m4`` are the sums
    of the powers of the deviations (``_central_sums``). ``rows`` names the
    values in the caution.
    """
    if n < _SHAPE_MIN_OBJECTS or not m2 > 0:
        return None
    skew = math.sqrt(n * (n - 1)) / (n - 2) * math.sqrt(n) * m3 / m2**1.5
    kurtosis = ((n + 1) * (n * m4 / m2**2 - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
    bimodality = sarle(skew, kurtosis, n)
    if kurtosis <= _MAX_EXCESS_KURTOSIS and bimodality <= _MAX_BIMODALITY:
        return None
    # Only the value that crosses its threshold is named, to three decimals near it.
    tripped = []
    if kurtosis > _MAX_EXCESS_KURTOSIS:
        limit = f"{_MAX_EXCESS_KURTOSIS:g}"
        tripped.append(f"excess kurtosis {kurtosis:.1f} (above {limit}: a heavy tail)")
    if bimodality > _MAX_BIMODALITY:
        tripped.append(
            f"bimodality coefficient {bimodality:.3f} (above 5/9 = 0.556: two groups of values)"
        )
    return (
        f"the upper-95 SD bound assumes normal values, and these {rows} have "
        f"{' and '.join(tripped)}, so the bound can be optimistic (with 10% of "
        f"{rows} at four times the level of the rest it covered the true SD 68 to 88% of "
        "the time, not 97.5%)"
    )


@dataclass(frozen=True)
class _FovSummary:
    """Sufficient statistics of one acquired FOV for the averaging-CV criterion.

    ``m2``, ``m3`` and ``m4`` sum the 2nd to 4th powers of the deviations from
    the FOV's mean (for the shape check); ``name`` is the FOV's identifier and
    ``first_missing`` the 1-based position, in table order, of its first object
    without a value (``None`` when every object has one).
    """

    count: int
    total: float
    mean: float
    m2: float
    has_negative: bool
    m3: float = 0.0
    m4: float = 0.0
    name: Any = ""
    first_missing: int | None = None


def _merged_moments(fovs: list[_FovSummary]) -> tuple[int, float, float, float, float]:
    """Object count, mean and the sums of the 2nd to 4th powers of the deviations over FOVs.

    The pairwise merge of Chan et al. for the mean and ``m2``, extended to
    ``m3`` and ``m4`` by Pebay (2008), so the incremental path gives the full
    path's moments up to rounding.
    """
    n, mean, m2, m3, m4 = 0, 0.0, 0.0, 0.0, 0.0
    for fov in fovs:
        if fov.count == 0:
            continue
        a, b = n, fov.count
        combined = a + b
        delta = fov.mean - mean
        m4 += (
            fov.m4
            + delta**4 * a * b * (a * a - a * b + b * b) / combined**3
            + 6 * delta**2 * (a * a * fov.m2 + b * b * m2) / combined**2
            + 4 * delta * (a * fov.m3 - b * m3) / combined
        )
        m3 += (
            fov.m3
            + delta**3 * a * b * (a - b) / combined**2
            + 3 * delta * (a * fov.m2 - b * m2) / combined
        )
        m2 += fov.m2 + delta**2 * a * b / combined
        mean += delta * b / combined
        n = combined
    return n, mean, m2, m3, m4


def _missing_reason(fov: Any, position: int, n_fovs: int, unit: str) -> str:
    """Why the rule is undefined: where the first missing value is, and what to do.

    ``fov`` is the first FOV (in acquisition order) holding an observation
    without a value, ``position`` that observation's 1-based place in the FOV's
    rows and ``n_fovs`` how many FOVs hold one.
    """
    others = f" ({n_fovs} FOVs hold one)" if n_fovs > 1 else ""
    return (
        f"missing values in the selected scalar feature: {unit} {position} of FOV {fov!r} has "
        f"none{others}; the rule stays undefined for this group until the value is filled in, "
        "since values are never imputed or dropped"
    )


__all__ = ["fov_averaging_cv", "object_averaging_cv", "required_fovs", "required_objects"]
