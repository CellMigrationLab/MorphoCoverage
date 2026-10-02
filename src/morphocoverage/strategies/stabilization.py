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
``DescriptorSpreadEstimator``) plus ``ThresholdPolicy`` at ``tolerance``. The
paper watches two kinds of descriptor settle: where the values lie (the mean
and the median, ``LOCATION_DESCRIPTORS``) and how widely they spread (the SD
and the IQR, ``SPREAD_DESCRIPTORS``). So each estimator reports, in the
feature's units, the largest range (plateau) or spread across resamples
(spread) of each kind among the chosen descriptors:

- ``max_location_range`` / ``max_location_spread`` over the mean and median,
  population evidence (the population summary);
- ``max_scale_range`` / ``max_scale_spread`` over the SD and IQR, distribution
  evidence (the width of the phenotype distribution, part of its fingerprint);
- ``max_descriptor_range`` / ``max_descriptor_spread``, the largest over all
  chosen descriptors, kept with its earlier meaning for threshold sweeps
  (distribution evidence when an SD or IQR is among them, else population).

A kind is reported only when the chosen descriptors include it, and the
strategy applies ``tolerance`` to each kind reported. With the paper's four
descriptors the rule therefore answers the population objective (from the
location descriptors), the distribution objective (from the spread
descriptors) and the phenotype objective (both), and its own verdict (stop
when every descriptor is within tolerance) is unchanged.

Missing values in the feature make every quantity ``not_applicable``: the
framework does not drop them, and more acquisition does not remove them. No
spread so far (every visible object alike: a zero count, a saturated
intensity) makes them ``insufficient_data`` in both rules, as in the mean-precision rule:
a range or spread of 0 says nothing, and a later FOV can differ (audit 4,
P4-1: the rules stopped at 1-3 FOVs on counts of 0, well mean 0.137).

What a stop means:

- The plateau's ``tolerance`` bounds how much the cumulative descriptors
  changed over the last ``window`` FOVs with objects, not their error
  against the well. The range over the window shrinks roughly as
  ``window / k`` whatever the data: at a tolerance of 5% of the mean and a
  window of 3 the rule stopped at a median of 7-10 FOVs, and the location
  error exceeded the tolerance at 40% of stops in homogeneous wells, 65% with
  a subpopulation in 10% of FOVs, and at every stop in a raster over a
  gradient or under drift (P4-5). Its evidence says so in ``warning``.
- The per-FOV spread needs ``MIN_SPREAD_FOVS`` (5) FOVs with objects: with 2
  the resamples can only repeat them, so the spread is smallest there (a
  median of 27% of the mean, against 39-43% at 3-5 FOVs; P4-6).

Resamples are drawn in a canonical order (objects by value; FOVs by their
key, compared as text, their objects by value), so row order, which carries
no information, never changes a spread (it moved a per-object stop by up to
12 FOVs, P4-11).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._caveats import assumption
from .._checks import check_seed, is_integer, is_number
from .._wording import plural
from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput, fov_positions
from ..statistics import PAPER_DESCRIPTORS, describe, spread_is_zero
from ._descriptor_resamples import _fov_resamples, _object_resamples
from .base import (
    Estimator,
    SamplingStrategy,
    _as_ints,
    _canonical_fovs,
    check_order_col,
    unsupported_observation_unit,
)

_UNITS = ("object", "fov")
# Fewest FOVs with objects for the per-FOV spread (the meaning of the bootstrap's min_fovs).
MIN_SPREAD_FOVS = 5


def _no_spread_reason(values: np.ndarray, unit: str) -> str | None:
    """``insufficient_data``'s reason when the visible values are all alike, else ``None``.

    Up to rounding (``statistics.spread_is_zero``): a constant so far says
    nothing about how settled the descriptors are.
    """
    if len(values) >= 2 and spread_is_zero(float(np.std(values)), values):
        return f"no spread among the {plural(unit)} so far"
    return None


# The plateau's caution beside its quantities (audit 4, P4-5).
_PLATEAU_CAUTION = (
    "the tolerance bounds how much the descriptors changed over the last {window} FOVs with "
    "objects, not their error against the well: at a tolerance of 5% of the mean and a window "
    "of 3, the location error exceeded the tolerance at 40% of stops in homogeneous wells and "
    "at every stop in a raster over a spatial gradient"
)


def _reported(descriptors: dict[str, float]) -> dict[str, float | None]:
    """Descriptors for result diagnostics, with undefined values as None rather than NaN."""
    return {name: None if np.isnan(value) else value for name, value in descriptors.items()}


def _check_tolerance(tolerance: float) -> None:
    """The tolerance is a positive, finite amount in the feature's units."""
    if not (is_number(tolerance) and tolerance > 0 and np.isfinite(tolerance)):
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
    """A positive integer window (numpy integers too) and an order column name."""
    if not is_integer(window) or window < 1:
        raise ValueError("window must be a positive integer number of FOVs with objects")
    if not isinstance(order_col, str):
        raise TypeError("order_col must be the FOV table's acquisition-order column")


def _check_spread(resampling_unit: str, n_resamples: int, random_state: int) -> None:
    """A known resampling unit, at least 2 resamples and a non-negative integer seed."""
    if resampling_unit not in _UNITS:
        raise ValueError(f"resampling_unit must be one of {_UNITS}")
    if not is_integer(n_resamples):
        raise TypeError("n_resamples must be an integer")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not is_integer(random_state):
        raise TypeError("random_state must be an integer seed")
    check_seed(random_state)


LOCATION_DESCRIPTORS = ("mean", "median")
SPREAD_DESCRIPTORS = ("sd", "iqr")


def component_of(descriptors: Sequence[str]) -> Component:
    """Which objective component the descriptors, taken together, are evidence about.

    The paper's procedure watches the mean, median, SD and IQR settle. Location
    descriptors alone summarise the population; once a spread descriptor (SD or
    IQR) is among them the combined quantity (``max_descriptor_range`` or
    ``max_descriptor_spread``) describes the distribution of phenotypes, so it
    is distribution evidence (the phenotypic fingerprint). The location and
    scale quantities each have their own component (``_kinds``).
    """
    if any(name in SPREAD_DESCRIPTORS for name in descriptors):
        return Component.DISTRIBUTION
    return Component.POPULATION


def _kinds(
    measure: str, descriptors: Sequence[str]
) -> dict[str, tuple[tuple[str, ...], Component]]:
    """The quantities a rule reports, by name: their descriptors and their component.

    ``measure`` is ``"range"`` (plateau) or ``"spread"`` (resampled spread).
    First ``max_descriptor_<measure>`` over every chosen descriptor, then
    ``max_location_<measure>`` over the mean and median (population evidence)
    and ``max_scale_<measure>`` over the SD and IQR (distribution evidence),
    each only when the chosen descriptors include one of its kind.
    """
    location = tuple(name for name in descriptors if name in LOCATION_DESCRIPTORS)
    scale = tuple(name for name in descriptors if name in SPREAD_DESCRIPTORS)
    found = {f"max_descriptor_{measure}": (tuple(descriptors), component_of(descriptors))}
    if location:
        found[f"max_location_{measure}"] = (location, Component.POPULATION)
    if scale:
        found[f"max_scale_{measure}"] = (scale, Component.DISTRIBUTION)
    return found


def _quantities(
    measure: str,
    feature: str,
    descriptors: Sequence[str],
    by_descriptor: dict[str, float] | None = None,
    *,
    reason: str | None = None,
    applicable: bool = True,
) -> dict[str, Quantity]:
    """Every quantity of ``_kinds``, in the feature's units, lower is better.

    With ``by_descriptor`` (each descriptor's range or spread) each quantity is
    the largest over its descriptors. Otherwise every quantity carries
    ``reason``: ``insufficient_data`` (more FOVs with objects make the window or
    the resamples computable) or, with ``applicable=False``, ``not_applicable``.
    """
    found = {}
    for name, (names, component) in _kinds(measure, descriptors).items():
        meta = {
            "direction": Direction.LOWER_IS_BETTER,
            "units": f"units of {feature!r}",
            "component": component,
        }
        if by_descriptor is not None:
            found[name] = Quantity.available(max(by_descriptor[n] for n in names), **meta)
        elif applicable:
            found[name] = Quantity.insufficient(reason, **meta)
        else:
            found[name] = Quantity.not_applicable(reason, **meta)
    return found


def _not_applicable(
    measure: str, feature: str, descriptors: Sequence[str], reason: str
) -> Evidence:
    """Evidence for data the rule does not apply to; more acquisition will not fix it."""
    quantities = _quantities(measure, feature, descriptors, reason=reason, applicable=False)
    return Evidence(quantities, {"undefined_reason": reason})


def _policies(
    measure: str, descriptors: Sequence[str], tolerance: float
) -> tuple[ThresholdPolicy, ...]:
    """``tolerance`` on the location and on the scale quantity, each when reported."""
    names = [name for name in _kinds(measure, descriptors) if not name.startswith("max_descriptor")]
    return tuple(ThresholdPolicy(name, tolerance) for name in names)


@dataclass(frozen=True)
class DescriptorPlateauEstimator(Estimator):
    """Largest range of cumulative descriptors over the last ``window`` growth steps.

    The descriptors of all visible objects are recomputed after each acquired
    FOV that added objects. Over the last ``window + 1`` such cumulative
    states (the current one and the ``window`` before it), it reports the
    descriptor ranges (max - min), in the feature's units: the largest over
    the mean and median as ``max_location_range`` (population), over the SD
    and IQR as ``max_scale_range`` (distribution), and over all chosen
    descriptors as ``max_descriptor_range`` (see the module docstring). Until
    ``window + 1`` states with at least 2 objects exist, the quantities are
    ``insufficient_data``.

    ``window`` counts FOVs that added objects, not acquired FOVs: FOVs without
    objects add no state, since the paper's curve is indexed by number of
    objects and empty FOVs must not make descriptors look stable. With every
    other FOV empty, a window of 2 spans 5 acquired FOVs (``fovs_in_window``
    lists the prefix sizes of its states). ``order_col`` must pass
    ``base.check_order_col`` (complete numbers or times). Missing values in
    the feature, or rows that are not objects, make the quantities
    ``not_applicable``; no spread among the visible objects so far makes them
    ``insufficient_data``. The ranges bound the change over the window, not
    the error against the well (the module docstring; ``warning`` says so).
    """

    feature: str
    window: int
    order_col: str
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS
    name: str = "descriptor_range"

    def __post_init__(self) -> None:
        _check_common(self.feature, self.descriptors)
        _check_plateau(self.window, self.order_col)
        _as_ints(self, "window")

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
        """The descriptor ranges over the trailing window of growth steps."""
        values = data.feature_values(self.feature)
        # The paper's curve is indexed by number of objects, so other rows do not fit.
        reason = unsupported_observation_unit(
            data, OBJECT_LIKE_UNITS, method="the descriptor-plateau rule"
        )
        if reason is None and np.isnan(values).any():
            reason = f"missing values in {self.feature}"
        if reason is not None:
            return _not_applicable("range", self.feature, self.descriptors, reason)
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
        elif (constant := _no_spread_reason(values, data.observation_unit)) is not None:
            reason = constant
        else:
            ranges = {
                name: float(np.ptp([state[name] for state in states])) for name in self.descriptors
            }
            diagnostics["window_ranges"] = _reported(ranges)
            if any(np.isnan(value) for value in ranges.values()):
                reason = "fewer than 2 objects in the window"
        if reason is not None:
            diagnostics["undefined_reason"] = reason
            quantities = _quantities("range", self.feature, self.descriptors, reason=reason)
        else:
            diagnostics["warning"] = assumption(_PLATEAU_CAUTION.format(window=self.window))
            quantities = _quantities("range", self.feature, self.descriptors, ranges)
        return Evidence(quantities, diagnostics)


@dataclass(frozen=True)
class DescriptorSpreadEstimator(Estimator):
    """Largest descriptor spread across resamples of the visible data.

    Draws ``n_resamples`` resamples, with replacement, of the currently visible
    data at its current size: objects for ``resampling_unit="object"``, whole
    FOVs (including zero-object FOVs) for ``"fov"``. For each descriptor, the
    spread is max - min across resamples, as in the paper's Fig. 2A, which used
    100 random samples. It reports, in the feature's units (``inf`` when a
    resample has fewer than 2 objects), the largest spread over the mean and
    median as ``max_location_spread`` (population), over the SD and IQR as
    ``max_scale_spread`` (distribution), and over all chosen descriptors as
    ``max_descriptor_spread`` (see the module docstring). Because a range grows
    with the number of resamples, ``n_resamples`` is part of the estimate and
    has no default.

    Fewer than 2 objects (object unit), fewer than ``MIN_SPREAD_FOVS`` (5)
    acquired FOVs with objects (FOV unit), or no spread among the visible
    objects so far, give ``insufficient_data``. Missing values in the
    feature, or rows that are not objects for the object unit, give
    ``not_applicable``. The object unit resamples objects as independent
    draws (``assumes_random_object_sampling`` in the diagnostics), as the
    paper assumes cells are sampled at random from the well. Resamples are
    drawn in a canonical order (the module docstring), so row order never
    changes them.
    """

    feature: str
    resampling_unit: str
    n_resamples: int
    random_state: int
    descriptors: tuple[str, ...] = PAPER_DESCRIPTORS

    def __post_init__(self) -> None:
        _check_common(self.feature, self.descriptors)
        _check_spread(self.resampling_unit, self.n_resamples, self.random_state)
        _as_ints(self, "n_resamples", "random_state")

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
        """Descriptors of each resample, or the reason resampling is not yet possible.

        Draws index the values in a canonical order (objects sorted by value;
        FOVs by ``base._canonical_fovs``, their objects by value), so the
        table's row order cannot change a resample: every descriptor of a
        resample is a function of the values it holds.
        """
        rng = np.random.default_rng(self.random_state)
        values = data.feature_values(self.feature)
        if self.resampling_unit == "object":
            if len(values) < 2:
                return "fewer than 2 objects"
            constant = _no_spread_reason(values, data.observation_unit)
            if constant is not None:
                return constant
            return _object_resamples(np.sort(values), self.n_resamples, rng, self.descriptors)

        k = len(data.fovs)
        rank = np.empty(k, dtype=int)
        rank[_canonical_fovs(data)] = np.arange(k)
        positions = rank[fov_positions(data, data.fovs)]  # each object's FOV, canonical
        filled = len(np.unique(positions))
        if filled < MIN_SPREAD_FOVS:
            return (
                f"fewer than {MIN_SPREAD_FOVS} acquired FOVs with objects (with 2 the resamples "
                "can only repeat them, so the spread is smallest there)"
            )
        constant = _no_spread_reason(values, data.observation_unit)
        if constant is not None:
            return constant
        return _fov_resamples(values, positions, k, self.n_resamples, rng, self.descriptors)

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The descriptor spreads across resamples of the visible data."""
        values = data.feature_values(self.feature)
        # Resampling objects assumes object rows; the FOV variant resamples whole
        # FOVs and applies to any observation unit.
        reason = None
        if self.resampling_unit == "object":
            reason = unsupported_observation_unit(
                data, OBJECT_LIKE_UNITS, method="the object-unit descriptor-spread rule"
            )
        if reason is None and np.isnan(values).any():
            reason = f"missing values in {self.feature}"
        if reason is not None:
            return _not_applicable("spread", self.feature, self.descriptors, reason)
        diagnostics: dict[str, Any] = {
            "descriptors": _reported(describe(values, self.descriptors)),
            "assumes_random_object_sampling": self.resampling_unit == "object",
        }
        resampled = self._resampled_descriptors(data)
        if isinstance(resampled, str):
            diagnostics["undefined_reason"] = resampled
            quantities = _quantities("spread", self.feature, self.descriptors, reason=resampled)
            return Evidence(quantities, diagnostics)
        # A resample with fewer than 2 objects has undefined descriptors: not stable.
        spreads = {
            name: float(np.ptp(series)) if np.isfinite(series).all() else float("inf")
            for name, series in resampled.items()
        }
        diagnostics["resample_spreads"] = spreads
        return Evidence(_quantities("spread", self.feature, self.descriptors, spreads), diagnostics)


@dataclass(frozen=True)
class DescriptorPlateauStrategy(SamplingStrategy):
    """Stop when cumulative descriptors stay within ``tolerance`` over ``window`` FOVs with objects.

    ``DescriptorPlateauEstimator`` plus ``ThresholdPolicy(tolerance)`` on
    ``max_location_range`` (population) and on ``max_scale_range``
    (distribution), each when the descriptors include its kind: its own
    verdict is stop once, over the last ``window + 1`` cumulative states,
    every descriptor's range is at most ``tolerance``, in the feature's units,
    and it continues until enough states exist. ``window`` counts FOVs that
    added objects; empty FOVs in between do not count. A stop says the
    descriptors changed by at most ``tolerance`` over the window, not that
    they are within ``tolerance`` of the well's: see the module docstring for
    how often they were not.
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
        _as_ints(self, "window")

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
    def policies(self) -> tuple[ThresholdPolicy, ...]:
        """``max_location_range <= tolerance`` and ``max_scale_range <= tolerance``."""
        return _policies("range", self.descriptors, self.tolerance)


@dataclass(frozen=True)
class DescriptorSpreadStrategy(SamplingStrategy):
    """Stop when descriptor spread across resamples of the visible data is within ``tolerance``.

    ``DescriptorSpreadEstimator`` plus ``ThresholdPolicy(tolerance)`` on
    ``max_location_spread`` (population) and on ``max_scale_spread``
    (distribution), each when the descriptors include its kind: its own
    verdict is stop when every descriptor's spread across resamples is at most
    ``tolerance``, in the feature's units.
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
        _as_ints(self, "n_resamples", "random_state")

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
    def policies(self) -> tuple[ThresholdPolicy, ...]:
        """``max_location_spread <= tolerance`` and ``max_scale_spread <= tolerance``."""
        return _policies("spread", self.descriptors, self.tolerance)


__all__ = [
    "LOCATION_DESCRIPTORS",
    "SPREAD_DESCRIPTORS",
    "DescriptorPlateauEstimator",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadEstimator",
    "DescriptorSpreadStrategy",
    "component_of",
]
