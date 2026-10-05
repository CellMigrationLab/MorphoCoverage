"""The owner's decisions of 2026-10-05 on the strategies' precision estimates.

#97: the FOV unit's upper-95 averaging CV is a bootstrap of whole fields, not the chi-square
bound on Kish's effective fields; its coverage is simulated here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from morphocoverage.statistics import kish_effective_fields
from morphocoverage.strategies._averaging_cv import fov_averaging_cv


def _wells(
    rng: np.random.Generator, n: int, k: int, dispersion: float | None, sd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-field sums and counts of ``n`` wells of ``k`` fields of mean 20 objects.

    Field sizes are Poisson, or negative binomial (over-dispersed) with ``dispersion``; field
    means are lognormal around 100 (SD ``sd`` on the log scale) and objects normal around their
    field's mean (SD 30), so a field's sum is drawn from its size and mean directly.
    """
    if dispersion is None:
        sizes = rng.poisson(20, (n, k)).astype(float)
    else:
        sizes = rng.negative_binomial(dispersion, dispersion / (dispersion + 20), (n, k))
        sizes = sizes.astype(float)
    means = 100 * np.exp(sd * rng.standard_normal((n, k)))
    return sizes * means + np.sqrt(sizes) * 30 * rng.standard_normal((n, k)), sizes


def _coverage(k: int, dispersion: float | None, sd: float, wells: int) -> tuple[float, float]:
    """How often the bootstrap bound and the former Kish bound (at 95%) cover the truth."""
    rng = np.random.default_rng(97)
    sums, counts = _wells(rng, 20000, k, dispersion, sd)
    usable = (counts > 0).sum(axis=1) >= 2
    pooled = sums[usable].sum(axis=1) / counts[usable].sum(axis=1)
    truth = pooled.std() / pooled.mean()  # the averaging CV of k fields
    sums, counts = _wells(rng, wells, k, dispersion, sd)
    bootstrap, kish = [], []
    for well in range(wells):
        if (counts[well] > 0).sum() < 2 or kish_effective_fields(counts[well]) < 2:
            continue  # insufficient_data for both bounds
        found = fov_averaging_cv(sums[well], counts[well], random_state=well)
        upper = found["cv_mean_upper_95"]
        bootstrap.append(not math.isfinite(upper) or truth <= upper)
        df = found["effective_fovs"] - 1
        kish.append(truth <= found["cv_mean"] * math.sqrt(df / stats.chi2.ppf(0.05, df)))
    return float(np.mean(bootstrap)), float(np.mean(kish))


def test_the_fov_upper_bound_covers_near_95_with_over_dispersed_field_sizes():
    # Regression (#97): the chi-square bound on Kish's effective fields covered 70-85% with
    # over-dispersed field sizes. Negative binomial sizes (dispersion 1), 8 fields.
    bootstrap, kish = _coverage(8, 1.0, 0.3, 600)
    assert 0.90 <= bootstrap <= 0.97
    assert bootstrap > kish
    # Poisson sizes, 10 fields: close to nominal.
    bootstrap, _ = _coverage(10, None, 0.3, 600)
    assert 0.93 <= bootstrap <= 0.99


def test_the_fov_upper_bound_needs_spread_in_95_percent_of_its_resamples():
    # Two fields of one mean among four: 1/16 of the resamples draw only those two (over 5%),
    # so the bound is not defined yet, while the point CV is.
    sums, counts = np.array([200.0, 200.0, 150.0, 260.0]), np.full(4, 2.0)
    found = fov_averaging_cv(sums, counts)
    assert found["cv_mean"] > 0 and math.isnan(found["cv_mean_upper_95"])
    five = fov_averaging_cv(np.r_[sums, 230.0], np.full(5, 2.0))
    assert five["cv_mean_upper_95"] > five["cv_mean"]
