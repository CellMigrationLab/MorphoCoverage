"""The checks of the FOV bootstrap (``fov_bootstrap``): its settings and its data.

Shared-private: every name here is used by ``fov_bootstrap`` (``SUMMARIES`` is
re-exported there); they are private to the package and may change with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._checks import check_seed, is_integer
from .._wording import listing
from ..metadata import column_names
from ..schema import ValidatedInput
from ..statistics import spread_is_zero
from ..transforms import Chain, FeatureTransform

SUMMARIES = ("centroid", "covariance", "quantiles", "eigenspectrum", "subspace")


def _check(
    summary: str,
    n_resamples: int,
    random_state: int,
    min_fovs: int,
    quantiles: Sequence[float] | None,
    transform: FeatureTransform | None,
    n_components: int | None,
) -> None:
    """Validate the settings shared by the estimator and the strategy."""
    if summary not in SUMMARIES:
        raise ValueError(f"summary must be one of {SUMMARIES}")
    _check_counts(n_resamples, min_fovs, random_state)
    _check_summary_settings(summary, quantiles, n_components)
    if transform is not None and not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform or None (raw feature units)")


def _check_counts(n_resamples: int, min_fovs: int, random_state: int) -> None:
    """The resamples and the fewest fields: integers of at least 2; the seed an integer."""
    for value, label, minimum in ((n_resamples, "n_resamples", 2), (min_fovs, "min_fovs", 2)):
        if not is_integer(value):  # numpy integers too
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if not is_integer(random_state):
        raise TypeError("random_state must be an integer seed")
    check_seed(random_state)


def _check_summary_settings(
    summary: str, quantiles: Sequence[float] | None, n_components: int | None
) -> None:
    """``quantiles`` given for, and only for, the quantiles; ``n_components`` for, and only
    for, the subspace."""
    if (summary == "quantiles") != (quantiles is not None):
        raise ValueError("quantiles are required for, and only for, summary='quantiles'")
    if quantiles is not None and (not quantiles or not all(0 <= q <= 1 for q in quantiles)):
        raise ValueError("quantiles must be a non-empty sequence of values in [0, 1]")
    if (summary == "subspace") != (n_components is not None):
        raise ValueError("n_components is required for, and only for, summary='subspace'")
    if n_components is not None and (not is_integer(n_components) or n_components < 1):
        raise ValueError("n_components must be a positive integer")


def _visible_blocks(data: ValidatedInput, block_col: str | None) -> int | None:
    """Number of visible spatial blocks, or ``None`` without blocks; checks the column."""
    if block_col is None:
        return None
    if block_col not in data.fovs.columns:
        raise ValueError(f"block_col {block_col!r} is not a column of the FOV table")
    if data.fovs[block_col].isna().any():
        raise ValueError(f"block_col {block_col!r} contains missing values")
    return int(data.fovs[block_col].nunique())


def _never_fits(transform: FeatureTransform, n_features: int) -> bool:
    """Whether the transform's declared output width says no fit can ever succeed.

    ``output_width`` is ``None`` both when a width is unknown (the default of
    ``FeatureTransform``) and when no fit can succeed, so only a transform
    that declares its width is read: ``PCA(3)`` of 2 features never fits,
    whatever is acquired, while an unknown width could still fit later. A
    ``Chain`` is read step by step.
    """
    steps = transform.transforms if isinstance(transform, Chain) else (transform,)
    width: int | None = n_features
    for step in steps:
        if type(step).output_width is FeatureTransform.output_width:
            return False  # an undeclared width: a later fit may succeed
        width = step.output_width(width)
        if width is None:
            return True
    return False


def _pca_reason(summary: str, n_components: int | None, d: int) -> str | None:
    """Why a PCA summary is undefined for ``d`` (transformed) features, if it is."""
    if d < 2:
        return "PCA summaries need at least 2 features"
    if summary == "subspace" and n_components >= d:
        return f"n_components={n_components} must be below the {d} features"
    return None


def _constant_features(data: ValidatedInput, positions: np.ndarray, summary: str) -> list[str]:
    """The selected features without spread so far: between FOV means (centroid) or at all.

    Up to rounding (``statistics.spread_is_zero``). A centroid feature whose
    FOV means are all alike has a bootstrap SE of 0 whatever the well holds,
    as do the covariance entries and quantiles of a feature whose values are
    all alike. The PCA summaries are left out: a direction without spread
    either lies beyond the summary or makes its resamples degenerate, which
    ``fov_bootstrap`` refuses on its own (``DEGENERATE_TOLERANCE``).
    """
    if summary in ("eigenspectrum", "subspace"):
        return []
    values = data.feature_matrix()
    found = []
    for j, name in enumerate(data.feature_cols):
        column = values[:, j]
        if summary == "centroid":
            counts = np.bincount(positions)
            filled = counts > 0
            spread = np.std(np.bincount(positions, weights=column)[filled] / counts[filled])
        else:
            spread = np.std(column)
        if spread_is_zero(float(spread), column):
            found.append(name)
    return found


def _never_reason(
    estimator: Any, n_features: int, values: np.ndarray | None, diagnostics: dict[str, Any]
) -> str | None:
    """Why the estimator's summary can never be defined on these features, whatever the
    fields: a transform that can never be fitted, a PCA summary wider than its space.

    A PCA summary these features can never define is not_applicable from the first FOV
    on, not insufficient until min_fovs and then not_applicable. The transform's output
    width is known before it can be fitted.
    """
    width = values.shape[1] if values is not None else estimator.transform.output_width(n_features)
    never = None
    if values is None and _never_fits(estimator.transform, n_features):
        never = (
            f"the {estimator.transform.name!r} transform can never be fitted on "
            f"{n_features} features: {diagnostics['transform']['reason']}"
        )
    elif estimator.summary in ("eigenspectrum", "subspace") and width is not None:
        never = _pca_reason(estimator.summary, estimator.n_components, width)
    return never


def _too_few_reason(
    estimator: Any,
    data: ValidatedInput,
    positions: np.ndarray,
    values: np.ndarray | None,
    diagnostics: dict[str, Any],
    counts: tuple[int, int | None, int],
) -> str | None:
    """Why the fields so far cannot define the summary yet: too few fields or blocks with
    observations, a transform not fitted, no spread; ``None`` when they can. ``counts``:
    the fields with observations, the visible blocks (``None`` without) and the units
    resampled."""
    with_observations, blocks, resampled = counts
    if with_observations < estimator.min_fovs:
        return f"fewer than {estimator.min_fovs} visible FOVs with observations"
    if blocks is not None and resampled < estimator.min_fovs:
        return f"fewer than {estimator.min_fovs} spatial blocks with observations"
    if values is None:
        return f"the transform could not be fitted: {diagnostics['transform']['reason']}"
    if constant := _constant_features(data, positions, estimator.summary):
        where = "between the FOV means" if estimator.summary == "centroid" else "among the values"
        return f"no spread {where} of {listing(constant, limit=5)} so far"
    return None


def _feature_selection(features: Sequence[str] | None) -> tuple[str, ...] | None:
    """``features`` as a tuple of distinct column names, or ``None`` for every feature column."""
    if features is None:
        return None
    selected = column_names(features, "features")
    if not selected or any(not isinstance(f, str) or not f for f in selected):
        raise ValueError("features must name at least one column")
    if len(set(selected)) != len(selected):
        raise ValueError("features must be distinct")
    return selected
