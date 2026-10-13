"""The steps of ``ScalarPrecisionEstimator._precision_from`` (private): why there is no
estimate yet, the FOV unit's estimate from per-FOV sums and counts, and the cautions on an
estimate.

Shared-private: ``_not_yet``, ``_fov_estimate`` and ``_cautions``, used by
``scalar_precision``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .._wording import plural
from ..statistics import RELATIVE_TOLERANCE
from ._averaging_cv import _field_size_check, _no_spread, _shape_warning, fov_averaging_cv

# The paper's advice: a pilot SD estimate above 50% CV calls for more data.
_MAX_TRUSTED_CV = 0.5
# Below this many FOVs with objects the FOV unit's point estimate carries a caution.
_FEW_FOVS_FOR_POINT = 5


def _not_yet(
    sums: np.ndarray,
    counts: np.ndarray,
    has_negative: bool,
    missing: str | None,
    resampling_unit: str,
) -> tuple[str, str] | None:
    """``(status, reason)`` when the values can give no estimate: missing or negative
    values (never), too few objects or FOVs with objects, or only zeros (not yet);
    ``None`` otherwise."""
    if np.isnan(sums).any():
        return "not_applicable", missing or "missing values in the selected scalar feature"
    if has_negative:
        return "not_applicable", (
            "negative values; the CV criterion applies to non-negative whole-cell measurements"
        )
    n_objects = int(counts.sum())
    if resampling_unit == "object":
        if n_objects < 2:
            return "insufficient_data", "fewer than 2 objects"
    elif int((counts > 0).sum()) < 2:
        return "insufficient_data", "fewer than 2 acquired FOVs with objects"
    if sums.sum() <= 0:
        # Only zeros so far (negatives are refused above): a later FOV can give a
        # positive mean, so this is not yet enough data rather than undefined.
        return "insufficient_data", "zero mean so far; the CV needs a positive mean"
    return None


def _fov_estimate(
    sums: np.ndarray,
    counts: np.ndarray,
    object_cvs: dict[str, float] | None,
    unit: str,
    sd_estimate: str,
) -> dict[str, Any] | tuple[str, str]:
    """The FOV unit's estimate (with the design effect when the objects vary), or
    ``("insufficient_data", reason)``: no spread between the FOV means, fewer than 2
    effective FOVs, no bootstrap bound for the upper-95 variant."""
    estimate = fov_averaging_cv(sums, counts)
    if _no_spread(estimate["cv_mean"] * estimate["mean"], estimate["mean"]):
        # Every FOV has the same mean: the ratio-estimator SE is 0 whatever the
        # objects within the FOVs, and a later FOV can differ.
        return "insufficient_data", "no spread among the FOV means so far"
    if estimate["effective_fovs"] < 2:
        # The ratio-estimator residuals of one crowded and one sparse FOV are
        # tiny whatever the between-FOV spread (audit 4, P4-3).
        return "insufficient_data", (
            f"fewer than 2 effective FOVs (Kish: {estimate['effective_fovs']:.2f}); "
            f"one FOV holds most of the {plural(unit)}"
        )
    if math.isnan(estimate["cv_mean_upper_95"]):
        estimate["cv_mean_upper_95"] = None  # no bound yet: the point variant goes on
    if sd_estimate == "upper_95" and estimate["cv_mean_upper_95"] is None:
        return "insufficient_data", (
            "too few FOVs for the whole-field bootstrap bound: over 5% of its resamples "
            "draw no spread between FOVs"
        )
    # Values are non-negative here, so a between-object CV at rounding level is a
    # constant feature: its design effect would be rounding noise over noise.
    if object_cvs is not None and object_cvs["cv_objects"] > RELATIVE_TOLERANCE:
        # Variance inflation from treating FOVs, not objects, as the sampling unit.
        ratio = (estimate["cv_mean"] / object_cvs["cv_mean"]) ** 2
        estimate["ratio_design_effect"] = ratio
        estimate.update(_field_size_check(sums, counts, plural(unit)))
    return estimate


def _cautions(
    object_cvs: dict[str, float] | None,
    moments: Callable[[], tuple[int, float, float, float]] | None,
    counts: np.ndarray,
    unit: str,
    resampling_unit: str,
    sd_estimate: str,
) -> list[str]:
    """The cautions on an estimate: a large SD between objects, a skewed or heavy-tailed
    distribution for the upper-95 bound, a point estimate from few FOVs."""
    cautions = []
    if object_cvs is not None and object_cvs["cv_objects"] > _MAX_TRUSTED_CV:
        cautions.append(
            f"the SD between {plural(unit)} exceeds 50% of their mean; Zweifach (JCB 2024) "
            "advises acquiring more data and repeating the estimate"
        )
    if moments and resampling_unit == "object" and sd_estimate == "upper_95":
        cautions.append(_shape_warning(*moments(), plural(unit)))
    filled = int((counts > 0).sum())
    if resampling_unit == "fov" and sd_estimate == "point" and filled < _FEW_FOVS_FOR_POINT:
        cautions.append(
            f"the point estimate from {filled} FOVs with {plural(unit)} is optimistic; the "
            "paper advises the upper-95 bound for small samples (the point rule stopped at "
            "2-3 FOVs in 39% of simulated homogeneous wells, and 18% of its stops were "
            "more than 9.8% off the well mean)"
        )
    return [text for text in cautions if text]
