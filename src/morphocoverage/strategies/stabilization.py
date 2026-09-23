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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..objectives import ObjectiveConfig
from ..results import Decision, StrategyResult
from ..schema import ValidatedInput, fov_positions
from .base import SamplingStrategy

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


def _check_common(feature, tolerance, descriptors) -> None:
    if not isinstance(feature, str):
        raise TypeError("feature must be one column name")
    if not tolerance > 0:
        raise ValueError("tolerance must be positive (in the feature's units)")
    if not descriptors or any(name not in PAPER_DESCRIPTORS for name in descriptors):
        raise ValueError(f"descriptors must be a non-empty subset of {PAPER_DESCRIPTORS}")
    if len(set(descriptors)) != len(descriptors):
        raise ValueError("descriptors contains duplicates")


def _result(strategy, objective, score, stop, diagnostics) -> StrategyResult:
    return StrategyResult(
        strategy=strategy.name,
        objective=objective,
        score=score,
        decision=Decision.STOP if stop else Decision.CONTINUE,
        diagnostics=diagnostics,
        parameters=strategy.parameters,
    )


@dataclass(frozen=True)
class DescriptorPlateauStrategy(SamplingStrategy):
    """Stop when cumulative descriptors have stayed within ``tolerance`` for ``window`` FOVs.

    The descriptors of all visible objects are recomputed after each acquired
    FOV that added objects. The strategy stops once, over the last
    ``window + 1`` such cumulative states (the current one and the ``window``
    before it), every descriptor's range (max - min) is at most ``tolerance``,
    in the feature's units. Until ``window + 1`` states with at least 2
    objects exist, it continues.

    FOVs without objects add no state: the paper's curve is indexed by number
    of cells, so empty FOVs must not make descriptors look stable.
    """

    feature: str
    tolerance: float
    window: int
    order_col: str
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS
    name: str = "descriptor_plateau"

    def __post_init__(self) -> None:
        _check_common(self.feature, self.tolerance, self.descriptors)
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError("window must be a positive integer number of FOVs")
        if not isinstance(self.order_col, str):
            raise TypeError("order_col must be the FOV table's acquisition-order column")

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

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        """Largest descriptor range over the trailing window as ``score``; stop within tolerance."""
        values = data.feature_values(self.feature)
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
        if len(states) < self.window + 1:
            diagnostics["undefined_reason"] = (
                f"fewer than {self.window + 1} acquired FOVs with objects"
            )
            return _result(self, objective, None, False, diagnostics)
        ranges = {
            name: float(np.ptp([state[name] for state in states])) for name in self.descriptors
        }
        diagnostics["window_ranges"] = _reported(ranges)
        if any(np.isnan(value) for value in ranges.values()):
            diagnostics["undefined_reason"] = "fewer than 2 objects in the window"
            return _result(self, objective, None, False, diagnostics)
        score = max(ranges.values())
        return _result(self, objective, score, score <= self.tolerance, diagnostics)


@dataclass(frozen=True)
class DescriptorSpreadStrategy(SamplingStrategy):
    """Stop when descriptor spread across resamples of the visible data is within ``tolerance``.

    Draws ``n_resamples`` resamples, with replacement, of the currently visible
    data at its current size: cells for ``resampling_unit="object"``, whole
    FOVs (including zero-object FOVs) for ``"fov"``. For each descriptor, the
    spread is max - min across resamples, as in the paper's Fig. 2A, which used
    100 random samples. Because a range grows with the number of resamples,
    ``n_resamples`` is part of the criterion and has no default. The strategy
    stops when every spread is at most ``tolerance``, in the feature's units.
    """

    feature: str
    tolerance: float
    resampling_unit: str
    n_resamples: int
    random_state: int
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS

    def __post_init__(self) -> None:
        _check_common(self.feature, self.tolerance, self.descriptors)
        if self.resampling_unit not in _UNITS:
            raise ValueError(f"resampling_unit must be one of {_UNITS}")
        if isinstance(self.n_resamples, bool) or not isinstance(self.n_resamples, int):
            raise TypeError("n_resamples must be an integer")
        if self.n_resamples < 2:
            raise ValueError("n_resamples must be at least 2")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer seed")

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

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        """Largest descriptor spread across resamples as ``score``; stop within tolerance."""
        values = data.feature_values(self.feature)
        diagnostics: dict[str, Any] = {
            "descriptors": _reported(describe(values, self.descriptors)),
            "assumes_random_cell_sampling": self.resampling_unit == "object",
        }
        resampled = self._resampled_descriptors(data)
        if isinstance(resampled, str):
            diagnostics["undefined_reason"] = resampled
            return _result(self, objective, None, False, diagnostics)
        # A resample with fewer than 2 objects has undefined descriptors: not stable.
        spreads = {
            name: float(np.ptp(series)) if np.isfinite(series).all() else float("inf")
            for name, series in resampled.items()
        }
        diagnostics["resample_spreads"] = spreads
        score = max(spreads.values())
        return _result(self, objective, score, score <= self.tolerance, diagnostics)
