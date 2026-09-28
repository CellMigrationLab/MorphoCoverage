"""JCB/Zweifach-style scalar precision baseline.

Reproduces the averaging-noise criterion of Zweifach, "Determining how many
cells to average for statistical testing of microscopy experiments",
J Cell Biol 2024;223(8):e202401074, doi:10.1083/jcb.202401074.

Scope stated by the paper: one scalar property of whole cells at a single time
point, in spatially uniform samples, with each sample's object average treated
as n = 1. With the between-object SD expressed as a fraction of the mean
(the paper's % CV), the number of objects to average is

    n = ceil((SD / target) ** 2),

equivalently, the averaging noise of a mean of ``n`` objects is ``SD / sqrt(n)``.
The paper recommends a target of about 5% CV when samples are independent
(Student's t test) and 2.5% CV for matched/paired designs whose samples share
between-repeat variability. It reports both a point estimate and one based on
the upper bound of the 95% confidence interval for SD, recommended when only a
few objects are available; the upper bound is the chi-square bound with
``n - 1`` degrees of freedom, which reproduces the paper's worked example.

This is a **single-group JCB-style adaptation**: the requirement is estimated
from the objects of one analysis group and the group's own mean is used as the
normalizing mean. It exactly matches the paper's single-sample calculation
when that group is the normalization control, or when the supplied feature has
already been normalized to the control mean. For an arbitrary treated group,
the published procedure instead expresses its within-sample SD relative to the
control mean. The faithful multi-sample, control-normalized planning procedure
is therefore ``plan_objects_to_average`` plus ``PlannedObjectCountStrategy``
in ``jcb_planning``.

``resampling_unit="object"`` is the published criterion. ``resampling_unit="fov"``
is a MorphoCoverage variant, not part of the paper: the same averaging noise
estimated with FOVs as the sampling unit (ratio-estimator variance for cluster
sampling), so phenotypes clustered within FOVs inflate it. The gap between the
two is the comparison the regime map needs; it does not make either wrong.
The FOV unit assumes the acquired FOVs are representative, exchangeable draws
of the well's positions: it does not see correlation between neighbouring
FOVs, so a raster over a spatial gradient, or a readout that drifts during
the acquisition, can stop it early. Its upper bound uses Kish's effective
number of FOVs, ``(sum N_j) ** 2 / sum N_j ** 2`` over the FOVs' object
counts, minus one as its degrees of freedom (``fov_averaging_cv``): with one
crowded FOV among sparse ones the bound is wide, and below 2 effective FOVs
the quantity is ``insufficient_data`` in both SD estimates.

Assumptions the paper's arithmetic makes, stated here because the rule cannot
check them all:

- The chi-square upper bound of the SD holds for normal values. A readout
  with a heavy tail or a minority state (10% of cells at four times the
  level of the rest) makes it optimistic; the object unit's upper-95 variant
  then says so in ``warning`` (excess kurtosis above 2 or bimodality
  coefficient above 5/9, from 8 objects).
- A CV is not invariant to an additive offset: intensities must be
  background-subtracted (+500 of camera background moved the per-FOV stop from
  FOV 21 to FOV 5 in one well).
- The point SD estimate is optimistic with few FOVs: with fewer than 5 FOVs
  holding objects the FOV unit's point variant says so in ``warning``, since
  the paper advises the upper bound for small samples.

Not reproduced here, but in ``jcb_planning``, both parts of the multi-sample
procedure:

- normalizing values to the average of all control samples;
- when several samples are used, subtracting each sample's mean before
  estimating the pooled within-sample SD.

Within one group the sample's own mean is used by this adaptation. That is
not the general published normalization for treated samples; use the planning
procedure when a separate control normalization is required.

Estimation and decision are separate. ``ScalarPrecisionEstimator`` reports the
averaging CV (and, with a pilot, the averaging CV the pilot projects for the
visible effort); ``ScalarPrecisionStrategy`` applies the paper's target to it
with a ``ThresholdPolicy``. The projected CV reaches the target when the
visible effort reaches the pilot's required number of objects
(``required_objects``) or, up to floating-point rounding, of FOVs
(``required_fovs``: at a rounding boundary the projection can need one FOV
more than that count), so the stopping rule is the paper's.

A feature with no spread so far (a saturated intensity, the same integer
count in every object) gives an averaging CV of 0, which says nothing about
averaging noise: the quantity is ``insufficient_data`` until objects (object
unit) or FOV means (FOV unit) differ, in every variant, as the planning path
refuses such a pilot.

A missing value is never imputed or dropped: one object without a value makes
the rule undefined (``not_applicable``) for the group from that FOV on, and
the reason names the first FOV and object without one, so the value can be
filled in.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import _deprecated
from .._checks import is_integer
from .._wording import plural
from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput, fov_positions
from ..statistics import RELATIVE_TOLERANCE, sd_upper_bound_factor  # noqa: F401  (re-exported)
from ._averaging_cv import (  # the public names are re-exported (__all__)
    _central_sums,
    _field_size_check,
    _missing_reason,
    _no_spread,
    _shape_warning,
    fov_averaging_cv,
    object_averaging_cv,
    required_fovs,
    required_objects,
)
from ._scalar_precision_state import _ScalarPrecisionState
from .base import (
    Estimator,
    EstimatorState,
    SamplingStrategy,
    check_order_col,
    unsupported_observation_unit,
)

_UNITS = ("object", "fov")
_SD_ESTIMATES = ("point", "upper_95")
# The paper's advice: a pilot SD estimate above 50% CV calls for more data.
_MAX_TRUSTED_CV = 0.5
# Below this many FOVs with objects the FOV unit's point estimate carries a caution.
_FEW_FOVS_FOR_POINT = 5


_CV_UNITS = "fraction of the mean (CV)"


# Effort counts of the visible data belong to the framework (``StrategyResult``),
# not to an estimator's diagnostics; only the pilot's own counts are reported.
_EFFORT_KEYS = ("n_objects", "n_fovs")


def _cv_quantity(
    value: float | None, availability: str = "available", reason: str | None = None
) -> Quantity:
    """An averaging-CV quantity: population evidence, lower is better, fraction of the mean."""
    return Quantity(
        value, Direction.LOWER_IS_BETTER, _CV_UNITS, Component.POPULATION, availability, reason
    )


def _check_configuration(
    resampling_unit: str, sd_estimate: str, pilot_fovs: int | None, order_col: str | None
) -> None:
    """Validate the settings the estimator and the strategy share."""
    if resampling_unit not in _UNITS:
        raise ValueError(f"resampling_unit must be one of {_UNITS}")
    if sd_estimate not in _SD_ESTIMATES:
        raise ValueError(f"sd_estimate must be one of {_SD_ESTIMATES}")
    if (pilot_fovs is None) != (order_col is None):
        raise ValueError("pilot_fovs and order_col must be given together")
    if pilot_fovs is not None and not (is_integer(pilot_fovs) and pilot_fovs >= 2):
        raise ValueError("pilot_fovs must be an integer number of FOVs, at least 2")


@dataclass(frozen=True)
class ScalarPrecisionEstimator(Estimator):
    """Averaging CV of one scalar feature's mean, as evidence for population precision.

    Quantities (lower is better, as a fraction of the mean, component
    ``population``):

    - ``averaging_cv``: the averaging CV of the visible data (``SD / sqrt(n)``
      relative to the mean for the object unit; the cluster ratio-estimator CV
      for the FOV unit), from the point or upper-95 SD estimate.
    - ``pilot_projected_averaging_cv`` (only with ``pilot_fovs``): the averaging
      CV that the pilot's estimate projects for the visible effort,
      ``cv_objects_pilot / sqrt(n_objects)`` (object unit) or
      ``cv_mean_pilot * sqrt(k_pilot / k)`` (FOV unit), with ``k`` every
      acquired FOV, empty ones included. The pilot is the first ``pilot_fovs``
      visible FOVs by ``order_col``, which must hold no missing value. Only
      when ``order_col`` is the acquisition order is the pilot fixed once
      complete, as in the paper's pilot procedure; with another order, a later
      FOV that sorts before a pilot FOV takes its place, so the pilot and its
      requirement can change as acquisition goes on.

    Too few objects or FOVs, fewer than 2 effective FOVs (Kish, FOV unit), or
    a zero mean so far (only zeros), give ``insufficient_data``, as does no
    spread so far (every object alike for the object unit, every FOV mean
    alike for the FOV unit, up to rounding: a CV of 0 says nothing about
    averaging noise). Missing or negative values make the CV undefined
    (``not_applicable``; the reason names the first FOV and object without a
    value), as does a pilot that cannot define the requirement.

    With the FOV unit the diagnostics also give ``ratio_design_effect``, the
    FOV-unit variance of the mean over that of as many random objects (the
    squared ratio of the two averaging CVs); it is not
    ``sampling_unit.fov_variability``'s ``design_effect``, which comes from
    variance components (``statistics.design_effect``). When FOV size and FOV
    mean are related, ``field_size_warning`` says so; that check has little
    power, so no warning is not evidence that they are unrelated. ``warning``
    joins the other cautions: an SD above 50% CV, values far from normal for
    the upper-95 SD bound (object unit), a point estimate from fewer than 5
    FOVs with objects (FOV unit). See ``ScalarPrecisionStrategy`` for the
    parameters.
    """

    feature: str
    resampling_unit: str
    sd_estimate: str
    pilot_fovs: int | None = None
    order_col: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("feature must be one column name")
        _check_configuration(
            self.resampling_unit, self.sd_estimate, self.pilot_fovs, self.order_col
        )
        if self.pilot_fovs is not None:
            object.__setattr__(self, "pilot_fovs", int(self.pilot_fovs))

    @property
    def name(self) -> str:
        """Variant-specific name, e.g. ``averaging_cv_object_upper95_pilot``."""
        parts = ["averaging_cv", self.resampling_unit]
        if self.sd_estimate == "upper_95":
            parts.append("upper95")
        if self.pilot_fovs is not None:
            parts.append("pilot")
        return "_".join(parts)

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the estimate, for result provenance."""
        return {
            "feature": self.feature,
            "resampling_unit": self.resampling_unit,
            "sd_estimate": self.sd_estimate,
            "pilot_fovs": self.pilot_fovs,
            "order_col": self.order_col,
        }

    def _suffix(self) -> str:
        """Diagnostic-key suffix selecting the point or upper-95 SD estimate."""
        return "_upper_95" if self.sd_estimate == "upper_95" else ""

    def _precision(
        self, data: ValidatedInput, fovs: pd.DataFrame
    ) -> dict[str, Any] | tuple[str, str]:
        """Averaging-precision estimate on the given FOVs, or (availability, reason)."""
        # Per-FOV sums and counts in the order of ``fovs``; empty FOVs give zeros.
        positions = fov_positions(data, fovs)
        inside = positions >= 0
        values = data.feature_values(self.feature)[inside]
        sums = np.bincount(positions[inside], weights=values, minlength=len(fovs))
        counts = np.bincount(positions[inside], minlength=len(fovs)).astype(float)
        missing = None
        if np.isnan(values).any():
            where = positions[inside]
            gaps = where[np.isnan(values)]
            first = int(gaps.min())
            within = np.isnan(values[where == first])
            name = fovs[data.fov_key[-1]].tolist()[first]
            place = int(np.argmax(within)) + 1
            missing = _missing_reason(name, place, len(np.unique(gaps)), data.observation_unit)
        return self._precision_from(
            sums,
            counts,
            has_negative=bool((values < 0).any()),
            cells=lambda: object_averaging_cv(values),
            unit=data.observation_unit,
            moments=lambda: _central_sums(values),
            missing=missing,
        )

    def _precision_from(
        self,
        sums: np.ndarray,
        counts: np.ndarray,
        *,
        has_negative: bool,
        cells: Callable[[], dict[str, float]],
        unit: str,
        moments: Callable[[], tuple[int, float, float, float]] | None = None,
        missing: str | None = None,
    ) -> dict[str, Any] | tuple[str, str]:
        """The estimate from per-FOV sums and counts; ``cells()`` gives the object-level CVs.

        Conditions that no further FOV can change (missing or negative values)
        are ``not_applicable`` whatever the effort so far; those more FOVs can
        change are ``insufficient_data``. In the FOV unit, empty FOVs carry no
        between-FOV information, so at least two FOVs must hold objects, and
        at least 2 effective FOVs (Kish) must: one crowded FOV beside sparse
        ones is about one FOV of information. No spread so far is also not yet
        enough data: a CV of 0 would stop the rule after two objects of a
        saturated or integer feature, although a later FOV can show the
        spread. ``unit`` names the observation rows in the wording;
        ``moments()`` gives the objects' central sums for the shape check (none
        without it) and ``missing`` where the first missing value is
        (``_missing_reason``); both paths of the estimator pass them.
        """
        if np.isnan(sums).any():
            return "not_applicable", missing or "missing values in the selected scalar feature"
        if has_negative:
            return "not_applicable", (
                "negative values; the CV criterion applies to non-negative whole-cell measurements"
            )
        n_objects = int(counts.sum())
        if self.resampling_unit == "object":
            if n_objects < 2:
                return "insufficient_data", "fewer than 2 objects"
        elif int((counts > 0).sum()) < 2:
            return "insufficient_data", "fewer than 2 acquired FOVs with objects"
        if sums.sum() <= 0:
            # Only zeros so far (negatives are refused above): a later FOV can give a
            # positive mean, so this is not yet enough data rather than undefined.
            return "insufficient_data", "zero mean so far; the CV needs a positive mean"

        object_cvs = cells() if n_objects >= 2 else None
        if self.resampling_unit == "object":
            estimate: dict[str, Any] = dict(object_cvs)
            if _no_spread(estimate["cv_objects"] * estimate["mean"], estimate["mean"]):
                return "insufficient_data", f"no spread among the {plural(unit)} so far"
        else:
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
            # Values are non-negative here, so a between-object CV at rounding level is a
            # constant feature: its design effect would be rounding noise over noise.
            if object_cvs is not None and object_cvs["cv_objects"] > RELATIVE_TOLERANCE:
                # Variance inflation from treating FOVs, not objects, as the sampling unit.
                ratio = (estimate["cv_mean"] / object_cvs["cv_mean"]) ** 2
                estimate["ratio_design_effect"] = ratio
                estimate.update(_field_size_check(sums, counts))
        cautions = []
        if object_cvs is not None and object_cvs["cv_objects"] > _MAX_TRUSTED_CV:
            cautions.append(
                f"the SD between {plural(unit)} exceeds 50% CV; the paper advises acquiring "
                "more data and repeating the estimate"
            )
        if moments and self.resampling_unit == "object" and self.sd_estimate == "upper_95":
            cautions.append(_shape_warning(*moments(), plural(unit)))
        filled = int((counts > 0).sum())
        if (
            self.resampling_unit == "fov"
            and self.sd_estimate == "point"
            and filled < _FEW_FOVS_FOR_POINT
        ):
            cautions.append(
                f"the point estimate from {filled} FOVs with {plural(unit)} is optimistic; the "
                "paper advises the upper-95 bound for small samples (the point rule stopped at "
                "2-3 FOVs in 39% of simulated homogeneous wells, and 18% of its stops were "
                "more than 9.8% off the well mean)"
            )
        cautions = [text for text in cautions if text]
        if cautions:
            estimate["warning"] = "; ".join(cautions)
        # The data this estimate came from: kept for the pilot's provenance, and
        # dropped from the visible estimate's diagnostics (the framework owns effort).
        estimate["n_objects"] = n_objects
        estimate["n_fovs"] = len(sums)
        return estimate

    def _check_order_col(self, fovs: pd.DataFrame) -> None:
        """With a pilot, the order column must pass ``base.check_order_col``.

        It must be in the FOV table, complete, and numbers or times: not text
        (``"10"`` sorts before ``"2"``) and not categorical (a newly revealed
        FOV carries only its own category, so the incremental pilot would
        differ from the full one).
        """
        if self.pilot_fovs is None:
            return
        check_order_col(fovs, self.order_col, "the pilot")

    def _unit_reason(self, data: ValidatedInput) -> str | None:
        """The published criterion averages objects; the FOV-unit variant any observation."""
        if self.resampling_unit != "object":
            return None
        return unsupported_observation_unit(
            data, OBJECT_LIKE_UNITS, method="the object-unit JCB criterion"
        )

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """Averaging CV of the visible data, and the pilot's projection when configured."""
        data.feature_values(self.feature)  # fails early on a non-feature column
        self._check_order_col(data.fovs)
        reason = self._unit_reason(data)
        if reason is not None:
            return self._evidence(
                ("not_applicable", reason),
                None,
                len(data.objects),
                len(data.fovs),
                data.observation_unit,
            )
        current = self._precision(data, data.fovs)
        pilot = None
        if self.pilot_fovs is not None and len(data.fovs) >= self.pilot_fovs:
            ordered = data.fovs.sort_values(self.order_col, kind="stable")
            pilot = self._precision(data, ordered.iloc[: self.pilot_fovs])
        return self._evidence(
            current, pilot, len(data.objects), len(data.fovs), data.observation_unit
        )

    def _evidence(
        self,
        current: dict[str, Any] | tuple[str, str],
        pilot: dict[str, Any] | tuple[str, str] | None,
        n_objects: int,
        n_fovs: int,
        unit: str,
    ) -> Evidence:
        """Quantities and diagnostics from the current and (if complete) pilot estimates."""
        diagnostics: dict[str, Any] = {
            "scalar_parameter": f"mean of {self.feature!r} over {plural(unit)}",
            "assumes_spatial_uniformity": self.resampling_unit == "object",
        }
        quantities: dict[str, Quantity] = {}
        if isinstance(current, tuple):
            availability, reason = current
            diagnostics["undefined_reason"] = reason
            quantities["averaging_cv"] = _cv_quantity(None, availability, reason)
        else:
            diagnostics.update({k: v for k, v in current.items() if k not in _EFFORT_KEYS})
            quantities["averaging_cv"] = _cv_quantity(current["cv_mean" + self._suffix()])
        if self.pilot_fovs is not None:
            quantities["pilot_projected_averaging_cv"] = self._projection(
                current, pilot, n_objects, n_fovs, diagnostics
            )
        return Evidence(quantities, diagnostics)

    def _projection(
        self,
        current: dict[str, Any] | tuple[str, str],
        pilot: dict[str, Any] | tuple[str, str] | None,
        n_objects: int,
        n_fovs: int,
        diagnostics: dict[str, Any],
    ) -> Quantity:
        """Averaging CV the fixed pilot estimate projects for the visible effort."""
        if isinstance(current, tuple) and current[0] == "not_applicable":
            return _cv_quantity(None, *current)
        if n_fovs < self.pilot_fovs:
            diagnostics["pilot_status"] = "pilot not yet complete"
            return _cv_quantity(None, "insufficient_data", "pilot not yet complete")
        if isinstance(pilot, tuple):
            # With order_col the acquisition order the pilot is fixed, so more acquisition
            # cannot fix it; with another order a later FOV can replace a pilot FOV.
            reason = (
                f"pilot (the first {self.pilot_fovs} FOVs by {self.order_col!r}): {pilot[1]}; "
                "it changes only if a later FOV sorts before a pilot FOV"
            )
            diagnostics["pilot_status"] = reason
            return _cv_quantity(None, "not_applicable", reason)
        diagnostics["pilot"] = pilot
        if self.resampling_unit == "object":
            projected = pilot["cv_objects" + self._suffix()] / math.sqrt(n_objects)
        else:
            projected = pilot["cv_mean" + self._suffix()] * math.sqrt(self.pilot_fovs / n_fovs)
        return _cv_quantity(projected)

    def start(
        self, template: ValidatedInput, *, objective: ObjectiveConfig
    ) -> EstimatorState | None:
        """Incremental version: keeps sufficient statistics per acquired FOV.

        Returns ``None`` (full recomputation, which reports not applicable)
        when the observation unit does not fit the criterion.
        """
        template.feature_values(self.feature)
        self._check_order_col(template.fovs)
        if self._unit_reason(template) is not None:
            return None
        return _ScalarPrecisionState(self, template.observation_unit)


@dataclass(frozen=True)
class ScalarPrecisionStrategy(SamplingStrategy):
    """Stop when the averaging CV of one scalar feature's mean reaches ``target_cv``.

    ``ScalarPrecisionEstimator`` plus ``ThresholdPolicy(target_cv)`` on
    ``averaging_cv``, or on ``pilot_projected_averaging_cv`` with a pilot. It
    answers the population component only.

    Parameters
    ----------
    feature:
        The single scalar measurement whose mean is estimated. It must be one of
        the explicitly selected feature columns, with non-negative values and a
        positive mean, as for the whole-object properties (sizes, counts,
        intensities) the criterion was designed for. Negative values make the
        criterion undefined (``decision=None``).
    target_cv:
        Target averaging CV as a fraction: the paper suggests 0.05 for
        independent samples and 0.025 for matched/paired designs. No default.
    resampling_unit:
        ``"object"`` (the published criterion) or ``"fov"`` (MorphoCoverage variant).
    sd_estimate:
        ``"point"`` or ``"upper_95"`` (upper bound of the 95% confidence interval,
        which the paper recommends when few objects are available). No default.
    pilot_fovs, order_col:
        ``None`` (default) re-checks the criterion on all visible data after every
        FOV, an adaptation for sequential stopping. With ``pilot_fovs=k`` (an
        integer) the requirement is planned from the first ``k`` FOVs by
        ``order_col``: a number of objects (object unit) or of FOVs (fov unit).
        With ``order_col`` the acquisition order, the plan is then held fixed, as
        in the paper's pilot procedure; another order re-selects the pilot when a
        later FOV sorts before a pilot FOV.
    """

    feature: str
    target_cv: float
    resampling_unit: str
    sd_estimate: str
    pilot_fovs: int | None = None
    order_col: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("feature must be one column name")
        if not 0 < self.target_cv < 1:
            raise ValueError("target_cv must be between 0 and 1")
        _check_configuration(
            self.resampling_unit, self.sd_estimate, self.pilot_fovs, self.order_col
        )
        if self.pilot_fovs is not None:
            object.__setattr__(self, "pilot_fovs", int(self.pilot_fovs))

    @property
    def name(self) -> str:
        """Variant-specific name, e.g. ``scalar_precision_object_upper95_pilot``."""
        parts = ["scalar_precision", self.resampling_unit]
        if self.sd_estimate == "upper_95":
            parts.append("upper95")
        if self.pilot_fovs is not None:
            parts.append("pilot")
        return "_".join(parts)

    @property
    def parameters(self) -> dict[str, Any]:
        """All settings that define the criterion, for result provenance."""
        return {
            "feature": self.feature,
            "target_cv": self.target_cv,
            "resampling_unit": self.resampling_unit,
            "sd_estimate": self.sd_estimate,
            "pilot_fovs": self.pilot_fovs,
            "order_col": self.order_col,
        }

    @property
    def estimators(self) -> tuple[ScalarPrecisionEstimator]:
        """The averaging-CV estimator with this strategy's settings."""
        return (
            ScalarPrecisionEstimator(
                self.feature,
                self.resampling_unit,
                self.sd_estimate,
                self.pilot_fovs,
                self.order_col,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """The paper's target applied to the running or pilot-projected averaging CV."""
        quantity = "averaging_cv" if self.pilot_fovs is None else "pilot_projected_averaging_cv"
        return (ThresholdPolicy(quantity, self.target_cv),)


__all__ = [
    "ScalarPrecisionEstimator",
    "ScalarPrecisionStrategy",
    "fov_averaging_cv",
    "object_averaging_cv",
    "required_fovs",
    "required_objects",
]


# Former object names, resolved with a DeprecationWarning (see morphocoverage._deprecated).
__getattr__ = _deprecated.module_getattr(__name__, globals(), ("cell_averaging_cv",))
