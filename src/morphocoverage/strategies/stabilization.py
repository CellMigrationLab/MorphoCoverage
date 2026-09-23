"""Marcotti et al.-style descriptor-stabilization baselines.

Operationalizes the sample-size advice of Marcotti, Gerontogianni, Kelly and
Barry, "Practical statistics for bioimage analysis - a guide to experimental
design and data interpretation", J Cell Sci 2026;139(10):jcs264367,
doi:10.1242/jcs.264367, with companion code at
https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility.

The paper advises increasing the number of cells "until statistical
descriptors stabilise" (Box 1) - the mean, median, standard deviation and
interquartile range of one metric - and stopping once they "stabilise around a
constant value and range". It gives no numeric criterion and calls the
threshold "somewhat arbitrary", so every tolerance here is an explicit user
choice, and both strategies are operationalizations rather than published
rules. The paper assumes cells are sampled randomly from the well; it
describes this as reasonable for cells on coverslips and less so for tissue.

- ``DescriptorPlateauStrategy`` follows Fig. 2B: descriptors of the cumulative
  sample, tracked as data arrive. Here data arrive by FOV in acquisition
  order, as on a microscope. The paper compares against the full well, which a
  prospective strategy cannot see, so stability is judged over a trailing
  window of acquired FOVs instead.
- ``DescriptorSpreadStrategy`` follows Fig. 2A: the difference between the
  maximum and minimum of a descriptor across repeated random samples. The
  paper draws subsamples without replacement from the whole well; drawn from
  the visible cells, that spread collapses to zero at the full visible size,
  so resamples are drawn with replacement at the current size.
  ``resampling_unit="object"`` resamples cells, as in the paper;
  ``resampling_unit="fov"`` resamples whole FOVs (a MorphoCoverage variant).

The paper's exponential fit to the spread curve is a visual aid and is not used.

Each strategy is an estimator (``DescriptorPlateauEstimator``,
``DescriptorSpreadEstimator``) reporting the largest descriptor range or
spread, in the feature's units, plus a ``ThresholdPolicy`` at ``tolerance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from .base import Estimator, SamplingStrategy, require_observation_unit

PAPER_DESCRIPTORS = ("mean", "median", "sd", "iqr")
_UNITS = ("object", "fov")


def describe(
    values: np.ndarray, descriptors: tuple[str, ...] = PAPER_DESCRIPTORS
) -> dict[str, float]:
    """Descriptors of one sample, as in the paper's Table 1; NaN with fewer than 2 values.

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


def _reported(descriptors: dict[str, float]) -> dict[str, float | None]:
    """Descriptors for result diagnostics, with undefined values as None rather than NaN."""
    return {name: None if np.isnan(value) else value for name, value in descriptors.items()}


def _describe_many(samples: np.ndarray, descriptors) -> dict[str, np.ndarray]:
    """Descriptors for each row of a 2-D array of equally sized samples."""
    q1, median, q3 = np.percentile(samples, [25, 50, 75], axis=1)
    available = {
        "mean": samples.mean(axis=1),
        "median": median,
        "sd": samples.std(axis=1, ddof=1),
        "iqr": q3 - q1,
    }
    return {name: available[name] for name in descriptors}


def _check_tolerance(tolerance) -> None:
    if not (tolerance > 0 and np.isfinite(tolerance)):
        raise ValueError("tolerance must be positive and finite (in the feature's units)")


def _check_common(feature, descriptors) -> None:
    if not isinstance(feature, str):
        raise TypeError("feature must be one column name")
    if not descriptors or any(name not in PAPER_DESCRIPTORS for name in descriptors):
        raise ValueError(f"descriptors must be a non-empty subset of {PAPER_DESCRIPTORS}")
    if len(set(descriptors)) != len(descriptors):
        raise ValueError("descriptors contains duplicates")


def _check_plateau(window, order_col) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer number of FOVs")
    if not isinstance(order_col, str):
        raise TypeError("order_col must be the FOV table's acquisition-order column")


def _check_spread(resampling_unit, n_resamples, random_state) -> None:
    if resampling_unit not in _UNITS:
        raise ValueError(f"resampling_unit must be one of {_UNITS}")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")


def _feature_quantity(feature: str, value: float | None, reason: str | None = None) -> Quantity:
    units = f"units of {feature!r}"
    if reason is not None:
        # More FOVs with objects make the window or resamples computable.
        return Quantity.insufficient(
            reason, direction=Direction.LOWER_IS_BETTER, units=units, component=Component.POPULATION
        )
    return Quantity.available(
        value, direction=Direction.LOWER_IS_BETTER, units=units, component=Component.POPULATION
    )


@dataclass(frozen=True)
class DescriptorPlateauEstimator(Estimator):
    """Largest range of cumulative descriptors over the last ``window`` growth steps.

    The descriptors of all visible objects are recomputed after each acquired
    FOV that added objects. Over the last ``window + 1`` such cumulative
    states (the current one and the ``window`` before it), it reports the
    largest descriptor range (max - min) as ``max_descriptor_range``, in the
    feature's units. Until ``window + 1`` states with at least 2 objects exist,
    the quantity is ``insufficient_data``.

    FOVs without objects add no state: the paper's curve is indexed by number
    of cells, so empty FOVs must not make descriptors look stable.
    """

    feature: str
    window: int
    order_col: str
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS
    name: str = "descriptor_range"

    def __post_init__(self) -> None:
        _check_common(self.feature, self.descriptors)
        _check_plateau(self.window, self.order_col)

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the estimate, for result provenance."""
        return {
            "feature": self.feature,
            "window": self.window,
            "order_col": self.order_col,
            "descriptors": list(self.descriptors),
        }

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """``max_descriptor_range`` over the trailing window of growth steps."""
        require_observation_unit(
            data, ("object", "cell", "nucleus"), method="DescriptorPlateauEstimator"
        )
        values = data.feature_values(self.feature)
        if np.isnan(values).any():
            reason = "missing values in the selected feature"
            return Evidence(
                {"max_descriptor_range": Quantity.not_applicable(
                    reason, direction=Direction.LOWER_IS_BETTER,
                    units=f"units of {self.feature!r}", component=Component.POPULATION
                )},
                {"undefined_reason": reason},
            )
        if self.order_col not in data.fovs.columns:
            raise ValueError(f"order_col {self.order_col!r} is not a column of the FOV table")

        ordered = data.fovs.sort_values(self.order_col, kind="stable")
        positions = fov_positions(data, ordered)
        # Prefix sizes (in FOVs) after which the cumulative sample gained objects.
        per_fov = np.bincount(positions[positions >= 0], minlength=len(ordered))
        growth_points = np.flatnonzero(per_fov) + 1
        states = [
            describe(values[(positions >= 0) & (positions < n_fovs)], self.descriptors)
            for n_fovs in growth_points[-(self.window + 1):]
        ]

        diagnostics: dict[str, Any] = {
            "descriptors": _reported(describe(values, self.descriptors)),
            "fovs_in_window": [int(n) for n in growth_points[-(self.window + 1):]],
        }
        reason = None
        if len(states) < self.window + 1:
            reason = f"fewer than {self.window + 1} acquired FOVs with objects"
        else:
            ranges = {
                name: float(np.ptp([state[name] for state in states]))
                for name in self.descriptors
            }
            diagnostics["window_ranges"] = _reported(ranges)
            if any(np.isnan(value) for value in ranges.values()):
                reason = "fewer than 2 objects in the window"
        if reason is not None:
            diagnostics["undefined_reason"] = reason
            quantity = _feature_quantity(self.feature, None, reason)
        else:
            quantity = _feature_quantity(self.feature, max(ranges.values()))
        return Evidence({"max_descriptor_range": quantity}, diagnostics)


@dataclass(frozen=True)
class DescriptorSpreadEstimator(Estimator):
    """Largest descriptor spread across resamples of the visible data.

    Draws ``n_resamples`` resamples, with replacement, of the currently visible
    data at its current size: cells for ``resampling_unit="object"``, whole
    FOVs (including zero-object FOVs) for ``"fov"``. For each descriptor, the
    spread is max - min across resamples, as in the paper's Fig. 2A, which used
    100 random samples. It reports the largest as ``max_descriptor_spread``, in
    the feature's units (``inf`` when a resample has fewer than 2 objects).
    Because a range grows with the number of resamples, ``n_resamples`` is part
    of the estimate and has no default.
    """

    feature: str
    resampling_unit: str
    n_resamples: int
    random_state: int
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS

    def __post_init__(self) -> None:
        _check_common(self.feature, self.descriptors)
        _check_spread(self.resampling_unit, self.n_resamples, self.random_state)

    @property
    def name(self) -> str:
        """``descriptor_spread_object`` or ``descriptor_spread_fov``."""
        return f"descriptor_spread_{self.resampling_unit}"

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the estimate, including the seed, for provenance."""
        return {
            "feature": self.feature,
            "resampling_unit": self.resampling_unit,
            "n_resamples": self.n_resamples,
            "random_state": self.random_state,
            "descriptors": list(self.descriptors),
        }

    def _resampled_descriptors(self, data: ValidatedInput) -> dict[str, np.ndarray] | str:
        rng = np.random.default_rng(self.random_state)
        values = data.feature_values(self.feature)
        if self.resampling_unit == "object":
            if len(values) < 2:
                return "fewer than 2 objects"
            draws = rng.integers(0, len(values), size=(self.n_resamples, len(values)))
            return _describe_many(values[draws], self.descriptors)

        k = len(data.fovs)
        if k < 2 or len(values) == 0:
            return "fewer than 2 acquired FOVs or no objects"
        positions = fov_positions(data, data.fovs)
        members = [values[positions == j] for j in range(k)]
        results: dict[str, list[float]] = {name: [] for name in self.descriptors}
        for chosen in rng.integers(0, k, size=(self.n_resamples, k)):
            sample = np.concatenate([members[j] for j in chosen])
            for name, value in describe(sample, self.descriptors).items():
                results[name].append(value)
        return {name: np.asarray(series) for name, series in results.items()}

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """``max_descriptor_spread`` across resamples of the visible data."""
        require_observation_unit(
            data, ("object", "cell", "nucleus"), method="DescriptorSpreadEstimator"
        )
        values = data.feature_values(self.feature)
        if np.isnan(values).any():
            reason = "missing values in the selected feature"
            return Evidence(
                {"max_descriptor_spread": Quantity.not_applicable(
                    reason, direction=Direction.LOWER_IS_BETTER,
                    units=f"units of {self.feature!r}", component=Component.POPULATION
                )},
                {"undefined_reason": reason},
            )
        diagnostics: dict[str, Any] = {
            "descriptors": _reported(describe(values, self.descriptors)),
            "assumes_random_cell_sampling": self.resampling_unit == "object",
        }
        resampled = self._resampled_descriptors(data)
        if isinstance(resampled, str):
            diagnostics["undefined_reason"] = resampled
            quantity = _feature_quantity(self.feature, None, resampled)
            return Evidence({"max_descriptor_spread": quantity}, diagnostics)
        # A resample with fewer than 2 objects has undefined descriptors: not stable.
        spreads = {
            name: float(np.ptp(series)) if np.isfinite(series).all() else float("inf")
            for name, series in resampled.items()
        }
        diagnostics["resample_spreads"] = spreads
        quantity = _feature_quantity(self.feature, max(spreads.values()))
        return Evidence({"max_descriptor_spread": quantity}, diagnostics)


@dataclass(frozen=True)
class DescriptorPlateauStrategy(SamplingStrategy):
    """Stop when cumulative descriptors have stayed within ``tolerance`` for ``window`` FOVs.

    ``DescriptorPlateauEstimator`` plus ``ThresholdPolicy(tolerance)`` on
    ``max_descriptor_range``: it stops once, over the last ``window + 1``
    cumulative states, every descriptor's range is at most ``tolerance``, in
    the feature's units, and continues until enough states exist.
    """

    feature: str
    tolerance: float
    window: int
    order_col: str
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS
    name: str = "descriptor_plateau"

    def __post_init__(self) -> None:
        _check_common(self.feature, self.descriptors)
        _check_tolerance(self.tolerance)
        _check_plateau(self.window, self.order_col)

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the criterion, for result provenance."""
        return {
            "feature": self.feature,
            "tolerance": self.tolerance,
            "window": self.window,
            "order_col": self.order_col,
            "descriptors": list(self.descriptors),
        }

    @property
    def estimators(self) -> tuple[DescriptorPlateauEstimator]:
        """The descriptor-range estimator with this strategy's settings."""
        return (
            DescriptorPlateauEstimator(self.feature, self.window, self.order_col, self.descriptors),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``max_descriptor_range <= tolerance``."""
        return (ThresholdPolicy("max_descriptor_range", self.tolerance),)


@dataclass(frozen=True)
class DescriptorSpreadStrategy(SamplingStrategy):
    """Stop when descriptor spread across resamples of the visible data is within ``tolerance``.

    ``DescriptorSpreadEstimator`` plus ``ThresholdPolicy(tolerance)`` on
    ``max_descriptor_spread``: it stops when every descriptor's spread across
    resamples is at most ``tolerance``, in the feature's units.
    """

    feature: str
    tolerance: float
    resampling_unit: str
    n_resamples: int
    random_state: int
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS

    def __post_init__(self) -> None:
        _check_common(self.feature, self.descriptors)
        _check_tolerance(self.tolerance)
        _check_spread(self.resampling_unit, self.n_resamples, self.random_state)

    @property
    def name(self) -> str:
        """``descriptor_spread_object`` or ``descriptor_spread_fov``."""
        return f"descriptor_spread_{self.resampling_unit}"

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the criterion, including the seed, for provenance."""
        return {
            "feature": self.feature,
            "tolerance": self.tolerance,
            "resampling_unit": self.resampling_unit,
            "n_resamples": self.n_resamples,
            "random_state": self.random_state,
            "descriptors": list(self.descriptors),
        }

    @property
    def estimators(self) -> tuple[DescriptorSpreadEstimator]:
        """The resampled-spread estimator with this strategy's settings."""
        return (
            DescriptorSpreadEstimator(
                self.feature,
                self.resampling_unit,
                self.n_resamples,
                self.random_state,
                self.descriptors,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``max_descriptor_spread <= tolerance``."""
        return (ThresholdPolicy("max_descriptor_spread", self.tolerance),)
