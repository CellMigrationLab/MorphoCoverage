"""The averaging-CV arithmetic of the JCB/Zweifach criterion (``scalar_precision``).

Pure functions of plain arrays and numbers: the object-level CVs, the FOV-unit
ratio-estimator CV, the paper's required counts, and the check of whether FOV
size and FOV mean are related. ``scalar_precision`` re-exports the public names.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..statistics import jcb_required_objects, object_weighted_mean_se, sd_upper_bound_factor


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


def fov_averaging_cv(sums: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """Averaging CV of the object-weighted mean with FOVs as the sampling unit.

    ``statistics.object_weighted_mean_se`` (the ratio estimator for cluster
    sampling, with per-FOV sums ``T_j`` and object counts ``N_j``) as a
    fraction of the mean. Estimand: the per-object mean over the sampled
    area. Only the ``k'`` FOVs holding objects enter the ``k' / (k' - 1)``
    factor and the ``k' - 1`` degrees of freedom of the upper bound, which is
    the object criterion's chi-square bound (in a check with up to 90% of FOVs
    empty it covered the true averaging CV 94-97% of the time for a nominal
    95%). Acquired FOVs without objects are still effort: they count in
    ``required_fovs`` and in the pilot projection, since future FOVs may be
    empty too. The unweighted mean of FOV means (``field_weighted_mean_se``)
    is a different estimand; ``sampling_unit.object_rule_check`` uses it.
    """
    mean, se, filled = object_weighted_mean_se(sums, counts)
    cv_mean = se / mean
    return {
        "mean": mean,
        "cv_mean": cv_mean,
        "cv_mean_upper_95": cv_mean * sd_upper_bound_factor(filled - 1),
        "fovs_with_objects": filled,
    }


def required_objects(cv_objects: float, target_cv: float) -> int:
    """The paper's number of objects to average (``statistics.jcb_required_objects``)."""
    return jcb_required_objects(cv_objects, target_cv)


def required_fovs(cv_mean: float, n_fovs: int, target_cv: float) -> int:
    """FOVs needed for ``target_cv``, since Var(mean) scales as 1/k under FOV resampling.

    Rounded as ``jcb_required_objects``: floating-point error never adds a FOV.
    """
    return math.ceil(round(n_fovs * (cv_mean / target_cv) ** 2, 9))


def _field_size_check(sums: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """Whether FOV size and FOV mean are related, which the ratio-estimator CV does not see.

    Spearman correlation between the object count and the mean of the FOVs
    holding objects (from 4 such FOVs, when both vary). When it is significant
    at 5%, a warning: the ratio estimator's linearized variance can then be
    several times smaller than the spread of the mean over resampled whole
    FOVs (0.013 against 0.051 in the check of ``object_weighted_mean_se``).
    """
    filled = counts > 0
    sizes, means = counts[filled], sums[filled] / counts[filled]
    if len(sizes) < 4 or np.ptp(sizes) == 0 or np.ptp(means) == 0:
        return {}
    rho, p_value = (float(v) for v in stats.spearmanr(sizes, means))
    found: dict[str, Any] = {"field_size_correlation": rho, "field_size_correlation_p": p_value}
    if p_value < 0.05:
        found["field_size_warning"] = (
            f"FOV size and FOV mean are related (Spearman rho {rho:.2f}, p = {p_value:.2g}, "
            f"over {len(sizes)} FOVs with objects): the FOV-unit averaging CV can then be "
            "several times too small; check it against a whole-FOV bootstrap of the mean "
            "(FOVBootstrapEstimator, centroid summary) or the mean of FOV means "
            "(statistics.field_weighted_mean_se)"
        )
    return found


__all__ = ["fov_averaging_cv", "object_averaging_cv", "required_fovs", "required_objects"]
