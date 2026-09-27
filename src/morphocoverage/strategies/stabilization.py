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
  the visible objects, that spread collapses to zero at the full visible size,
  so resamples are drawn with replacement at the current size.
  ``resampling_unit="object"`` resamples objects, as in the paper;
  ``resampling_unit="fov"`` resamples whole FOVs (a MorphoCoverage variant).

The paper's exponential fit to the spread curve is a visual aid and is not used.

Each strategy is an estimator (``DescriptorPlateauEstimator``,
``DescriptorSpreadEstimator``) reporting the largest descriptor range or
spread, in the feature's units, plus a ``ThresholdPolicy`` at ``tolerance``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput, fov_positions
from ..statistics import PAPER_DESCRIPTORS, describe
from .base import Estimator, SamplingStrategy, check_order_col, unsupported_observation_unit

_UNITS = ("object", "fov")


def _reported(descriptors: dict[str, float]) -> dict[str, float | None]:
    """Descriptors for result diagnostics, with undefined values as None rather than NaN."""
    return {name: None if np.isnan(value) else value for name, value in descriptors.items()}


def _describe_many(samples: np.ndarray, descriptors: Sequence[str]) -> dict[str, np.ndarray]:
    """Descriptors for each row of a 2-D array of equally sized samples."""
    q1, median, q3 = np.percentile(samples, [25, 50, 75], axis=1)
    available = {
        "mean": samples.mean(axis=1),
        "median": median,
        "sd": samples.std(axis=1, ddof=1),
        "iqr": q3 - q1,
    }
    return {name: available[name] for name in descriptors}


# Upper bound on values held at once when resampling whole FOVs.
_MAX_RESAMPLED_VALUES = 5_000_000


def _describe_fov_resamples(
    grouped: np.ndarray, counts: np.ndarray, draws: np.ndarray, descriptors: Sequence[str]
) -> dict[str, np.ndarray]:
    """Descriptors of each resample of whole FOVs, without a Python loop over resamples.

    ``grouped`` holds the values ordered by FOV, ``counts`` the values per FOV
    and ``draws`` one row of FOV indices per resample. Each resample is the
    concatenation of its FOVs' values, as in ``describe``; quantiles use the
    same linear interpolation as ``np.percentile``. Resamples with fewer than
    2 values give NaN.
    """
    n_resamples = len(draws)
    starts = np.cumsum(counts) - counts
    chosen = draws.ravel()
    lengths = counts[chosen]
    # Index of every value of every chosen FOV, and the resample it belongs to.
    offsets = np.repeat(starts[chosen] - (np.cumsum(lengths) - lengths), lengths)
    members = offsets + np.arange(lengths.sum())
    values = grouped[members]
    resample = np.repeat(np.repeat(np.arange(n_resamples), draws.shape[1]), lengths)
    sizes = np.bincount(resample, minlength=n_resamples)
    defined = sizes >= 2
    safe = np.maximum(sizes, 1)

    mean = np.bincount(resample, weights=values, minlength=n_resamples) / safe
    deviations = np.bincount(
        resample, weights=(values - mean[resample]) ** 2, minlength=n_resamples
    )
    sd = np.sqrt(deviations / np.maximum(sizes - 1, 1))

    # Values sorted within each resample: one integer sort on (resample, value rank).
    by_value = np.argsort(grouped, kind="stable")
    rank = np.empty_like(by_value)
    rank[by_value] = np.arange(len(grouped))
    keys = np.sort(resample * len(grouped) + rank[members])
    # A trailing NaN keeps indexing valid for resamples made only of empty FOVs.
    ordered = np.append(grouped[by_value][keys % max(1, len(grouped))], np.nan)
    first = np.cumsum(sizes) - sizes

    def quantile(q: float) -> np.ndarray:
        """Linearly interpolated quantile of every resample (``np.percentile``'s default)."""
        position = q * (safe - 1)
        low = np.floor(position).astype(np.int64)
        high = np.minimum(low + 1, safe - 1)
        a, b = ordered[first + low], ordered[first + high]
        return a + (b - a) * (position - low)

    available = {
        "mean": lambda: mean,
        "sd": lambda: sd,
        "median": lambda: quantile(0.5),
        "iqr": lambda: quantile(0.75) - quantile(0.25),
    }
    return {name: np.where(defined, available[name](), np.nan) for name in descriptors}


def _chunk_sizes(total: int, chunk: int) -> list[int]:
    """``total`` split into consecutive parts of at most ``chunk``."""
    return [min(chunk, total - start) for start in range(0, total, chunk)]


def _check_tolerance(tolerance: float) -> None:
    """The tolerance is a positive, finite amount in the feature's units."""
    if not (tolerance > 0 and np.isfinite(tolerance)):
        raise ValueError("tolerance must be positive and finite (in the feature's units)")


def _check_common(feature: str, descriptors: Sequence[str]) -> None:
    """One feature, and a non-empty duplicate-free subset of the paper's descriptors."""
    if not isinstance(feature, str):
        raise TypeError("feature must be one column name")
    if not descriptors or any(name not in PAPER_DESCRIPTORS for name in descriptors):
        raise ValueError(f"descriptors must be a non-empty subset of {PAPER_DESCRIPTORS}")
    if len(set(descriptors)) != len(descriptors):
        raise ValueError("descriptors contains duplicates")


def _check_plateau(window: int, order_col: str) -> None:
    """A positive integer window and an order column name."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer number of FOVs")
    if not isinstance(order_col, str):
        raise TypeError("order_col must be the FOV table's acquisition-order column")


def _check_spread(resampling_unit: str, n_resamples: int, random_state: int) -> None:
    """A known resampling unit, at least 2 resamples and an integer seed."""
    if resampling_unit not in _UNITS:
        raise ValueError(f"resampling_unit must be one of {_UNITS}")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")


SPREAD_DESCRIPTORS = ("sd", "iqr")


def component_of(descriptors: Sequence[str]) -> Component:
    """Which objective component the descriptors are evidence about.

    The paper's procedure watches the mean, median, SD and IQR settle. Location
    descriptors alone summarise the population; once a spread descriptor (SD or
    IQR) is among them the rule describes the distribution of phenotypes, so
    it is distribution evidence (the phenotypic fingerprint). The choice is a
    parameter, so it is recorded in the estimator's provenance.
    """
    if any(name in SPREAD_DESCRIPTORS for name in descriptors):
        return Component.DISTRIBUTION
    return Component.POPULATION


def _feature_quantity(
    feature: str, descriptors: Sequence[str], value: float | None, reason: str | None = None
) -> Quantity:
    """A quantity in the feature's units: a value, or insufficient data with a reason."""
    units = f"units of {feature!r}"
    component = component_of(descriptors)
    if reason is not None:
        # More FOVs with objects make the window or resamples computable.
        return Quantity.insufficient(
            reason, direction=Direction.LOWER_IS_BETTER, units=units, component=component
        )
    return Quantity.available(
        value, direction=Direction.LOWER_IS_BETTER, units=units, component=component
    )


def _not_applicable(
    quantity: str, feature: str, descriptors: Sequence[str], reason: str
) -> Evidence:
    """Evidence for data the rule does not apply to; more acquisition will not fix it."""
    value = Quantity.not_applicable(
        reason,
        direction=Direction.LOWER_IS_BETTER,
        units=f"units of {feature!r}",
        component=component_of(descriptors),
    )
    return Evidence({quantity: value}, {"undefined_reason": reason})


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
    of objects, so empty FOVs must not make descriptors look stable.
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
        values = data.feature_values(self.feature)
        # The paper's curve is indexed by number of objects, so other rows do not fit.
        reason = unsupported_observation_unit(
            data, OBJECT_LIKE_UNITS, method="the descriptor-plateau rule"
        )
        if reason is None and np.isnan(values).any():
            reason = "missing values in the selected feature"
        if reason is not None:
            return _not_applicable("max_descriptor_range", self.feature, self.descriptors, reason)
        check_order_col(data.fovs, self.order_col, "the descriptor-plateau rule")

        ordered = data.fovs.sort_values(self.order_col, kind="stable")
        positions = fov_positions(data, ordered)
        # Prefix sizes (in FOVs) after which the cumulative sample gained objects.
        per_fov = np.bincount(positions[positions >= 0], minlength=len(ordered))
        growth_points = np.flatnonzero(per_fov) + 1
        states = [
            describe(values[(positions >= 0) & (positions < n_fovs)], self.descriptors)
            for n_fovs in growth_points[-(self.window + 1) :]
        ]

        diagnostics: dict[str, Any] = {
            "descriptors": _reported(describe(values, self.descriptors)),
            "fovs_in_window": [int(n) for n in growth_points[-(self.window + 1) :]],
        }
        reason = None
        if len(states) < self.window + 1:
            reason = f"fewer than {self.window + 1} acquired FOVs with objects"
        else:
            ranges = {
                name: float(np.ptp([state[name] for state in states])) for name in self.descriptors
            }
            diagnostics["window_ranges"] = _reported(ranges)
            if any(np.isnan(value) for value in ranges.values()):
                reason = "fewer than 2 objects in the window"
        if reason is not None:
            diagnostics["undefined_reason"] = reason
            quantity = _feature_quantity(self.feature, self.descriptors, None, reason)
        else:
            quantity = _feature_quantity(self.feature, self.descriptors, max(ranges.values()))
        return Evidence({"max_descriptor_range": quantity}, diagnostics)


@dataclass(frozen=True)
class DescriptorSpreadEstimator(Estimator):
    """Largest descriptor spread across resamples of the visible data.

    Draws ``n_resamples`` resamples, with replacement, of the currently visible
    data at its current size: objects for ``resampling_unit="object"``, whole
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
        """Descriptors of each resample, or the reason resampling is not yet possible."""
        rng = np.random.default_rng(self.random_state)
        values = data.feature_values(self.feature)
        if self.resampling_unit == "object":
            if len(values) < 2:
                return "fewer than 2 objects"
            # Bound memory as for whole FOVs: each chunk holds its resamples' values at once.
            # Drawing chunk by chunk from one generator gives the same draws as all at once.
            chunk = max(1, int(_MAX_RESAMPLED_VALUES // len(values)))
            parts = [
                _describe_many(
                    values[rng.integers(0, len(values), size=(size, len(values)))],
                    self.descriptors,
                )
                for size in _chunk_sizes(self.n_resamples, chunk)
            ]
            return {name: np.concatenate([p[name] for p in parts]) for name in self.descriptors}

        k = len(data.fovs)
        if k < 2 or len(values) == 0:
            return "fewer than 2 acquired FOVs or no objects"
        positions = fov_positions(data, data.fovs)
        by_fov = np.argsort(positions, kind="stable")  # values grouped by FOV
        counts = np.bincount(positions, minlength=k)
        draws = rng.integers(0, k, size=(self.n_resamples, k))
        # Bound memory: each chunk materializes its resamples' values at once.
        chunk = max(1, int(_MAX_RESAMPLED_VALUES // max(1, len(values))))
        parts = [
            _describe_fov_resamples(values[by_fov], counts, draws[i : i + chunk], self.descriptors)
            for i in range(0, len(draws), chunk)
        ]
        return {name: np.concatenate([part[name] for part in parts]) for name in self.descriptors}

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """``max_descriptor_spread`` across resamples of the visible data."""
        values = data.feature_values(self.feature)
        # Resampling objects assumes object rows; the FOV variant resamples whole
        # FOVs and applies to any observation unit.
        reason = None
        if self.resampling_unit == "object":
            reason = unsupported_observation_unit(
                data, OBJECT_LIKE_UNITS, method="the object-unit descriptor-spread rule"
            )
        if reason is None and np.isnan(values).any():
            reason = "missing values in the selected feature"
        if reason is not None:
            return _not_applicable("max_descriptor_spread", self.feature, self.descriptors, reason)
        diagnostics: dict[str, Any] = {
            "descriptors": _reported(describe(values, self.descriptors)),
            "assumes_random_object_sampling": self.resampling_unit == "object",
        }
        resampled = self._resampled_descriptors(data)
        if isinstance(resampled, str):
            diagnostics["undefined_reason"] = resampled
            quantity = _feature_quantity(self.feature, self.descriptors, None, resampled)
            return Evidence({"max_descriptor_spread": quantity}, diagnostics)
        # A resample with fewer than 2 objects has undefined descriptors: not stable.
        spreads = {
            name: float(np.ptp(series)) if np.isfinite(series).all() else float("inf")
            for name, series in resampled.items()
        }
        diagnostics["resample_spreads"] = spreads
        quantity = _feature_quantity(self.feature, self.descriptors, max(spreads.values()))
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
