"""The owner's decisions of 2026-10-05 on the strategies' precision estimates.

#97: the FOV unit's upper-95 averaging CV is a bootstrap of whole fields, not the chi-square
bound on Kish's effective fields; its coverage is simulated here.
#315: the FOV bootstrap's SE is not available when more than 10% of its resamples are
undefined, and the share is said.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from morphocoverage import ObjectiveConfig, SamplingObjective, validate_input_tables
from morphocoverage.statistics import kish_effective_fields
from morphocoverage.strategies import FOVBootstrapEstimator
from morphocoverage.strategies._averaging_cv import fov_averaging_cv
from morphocoverage.strategies.fov_bootstrap import UNDEFINED_TOLERANCE


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


def _fields(filled: int, empty: int) -> object:
    """``filled`` fields of 5 objects each, then ``empty`` acquired fields without objects."""
    rng = np.random.default_rng(315)
    objects = pd.DataFrame(
        {
            "fov": np.repeat([f"f{j:02d}" for j in range(filled)], 5),
            "x": rng.normal(10, 2, 5 * filled) + np.repeat(rng.normal(0, 1, filled), 5),
        }
    )
    fovs = pd.DataFrame({"fov": [f"f{j:02d}" for j in range(filled + empty)]})
    return validate_input_tables(
        objects, fovs=fovs, feature_cols=["x"], fov_col="fov", group_cols=[]
    )


def _bootstrap(data: object, n: int = 2000) -> object:
    estimator = FOVBootstrapEstimator("centroid", None, n, 0, 2)
    return estimator.estimate(data, objective=ObjectiveConfig(SamplingObjective.POPULATION))


def test_the_fov_bootstrap_is_not_available_above_10_percent_undefined_resamples():
    # Regression (#315, #316): 2 fields with objects among 15 acquired: (13/15) ** 15 = 12% of
    # the resamples draw no observation. Their SE was taken from the rest, which is
    # conditional on a resample holding observations.
    assert UNDEFINED_TOLERANCE == 0.10
    evidence = _bootstrap(_fields(2, 13))
    quantity = evidence.quantities["fov_bootstrap_centroid_max_se"]
    assert quantity.availability == "insufficient_data"
    share = evidence.diagnostics["undefined_fraction"]
    assert 0.10 < share < 0.14
    assert evidence.diagnostics["undefined_resamples"] == round(share * 2000)
    assert quantity.reason == (
        f"{round(share * 2000)} of 2000 resamples ({round(100 * share)}%; more than 10%) draw "
        "too few observations to define the summary"
    )


def test_below_10_percent_undefined_the_fov_bootstrap_se_is_from_the_defined_resamples():
    # 3 fields with objects among 12: (9/12) ** 12 = 3% undefined. The SE is computed as before
    # from the defined resamples, and the share is in the diagnostics beside it.
    evidence = _bootstrap(_fields(3, 9))
    assert evidence.quantities["fov_bootstrap_centroid_max_se"].availability == "available"
    assert 0.01 < evidence.diagnostics["undefined_fraction"] < 0.06
    assert _bootstrap(_fields(4, 0)).diagnostics["undefined_fraction"] == 0
