"""JCB/Zweifach multi-sample planning: how many objects (the paper's cells) to average.

The full procedure of Zweifach, "Determining how many cells to average for
statistical testing of microscopy experiments", J Cell Biol 2024;223(8):
e202401074, doi:10.1083/jcb.202401074, as the paper states it:

1. Normalize: divide every value by the average of all the control samples,
   so errors are fractions of the control mean (the paper's % CV).
2. With several samples, subtract each sample's mean from that sample's
   values, "then calculate the SD for the entire data set", which removes
   differences between samples from the SD.
3. Cells to average: ``ceil((SD / target) ** 2)``, from the point SD or from
   the upper bound of its 95% confidence interval.

``ScalarPrecisionStrategy`` is the single-sample variant, re-estimated inside
each analysis group. This multi-sample procedure needs several samples, and
usually control samples from other conditions, so it cannot run inside one
group. It is therefore a separate planning step. ``plan_objects_to_average``
runs once on pilot data and returns a fixed ``ObjectCountPlan``;
``PlannedObjectCountStrategy`` then stops each evaluated group once it has
acquired the planned number of objects. An evaluated group that contains a
pilot sample would be tested on its own planning data, so its evidence is
reported as not applicable.

Supplementary Data S1 and S3 resolve the control normalization:

- "The average of all the control samples" is the mean of all control
  observations (objects), so control samples are weighted by the number of
  objects entered. With equal objects per control sample this is also the
  mean of the control-sample means.
- The SD of the centred data set uses ``ddof=1`` over all objects, and the
  upper bound uses a chi-square interval on ``n_objects - 1`` degrees of
  freedom: the SD "for the entire data set", as the paper's spreadsheet
  computes it. The paper reports a worked example of 17.8% point SD and 26.0%
  upper SD from 20 cells. Supplementary Data S1 carries cached result cells
  matching those reported values, but its visible 10 control and 10 treated
  inputs recalculate to about 20.38% and 29.77%; tests preserve both facts.

The spreadsheet arithmetic is kept, and it is biased for small multi-sample
pilots. Centring on ``n_samples`` sample means leaves ``n_objects -
n_samples`` degrees of freedom, so the point variance averages ``(n_objects
- n_samples) / (n_objects - 1)`` of the true within-sample variance, and the
point plan asks for that share of the objects needed; the upper bound is
computed on ``n_objects - 1`` degrees of freedom instead of ``n_objects -
n_samples``, so it covers the true SD less often than 95%. In a check with
normal values: 2 samples of 3 objects, 79% of the needed objects and 94%
coverage; 3 of 3, 76% and 90%; 4 of 5, 84% and 91%; 6 of 10, 92% and 93%
(a pooled-variance estimate on ``n_objects - n_samples`` covered 97-98% in
all four). With 10 or more objects per sample the difference is small; for
smaller pilots prefer ``sd_estimate="upper_95"`` and more objects per sample.

Sample columns may be categorical (``read_csv(dtype="category")``): samples
are the combinations present in the pilot, never unobserved combinations of
categories.

The paper's upper bound of the SD is the chi-square bound, which holds for
normal values. From normal pilots of 3 x 10, 3 x 30 and 5 x 40 objects the
upper-95 plan fell short of the objects actually needed in 5%, 2% and 3% of
pilots; with 10% of cells at four times the level of the rest (giant or
mitotic cells) in 19%, 12% and 11%, and for lognormal values (sigma 1) in
45-49% (audit 4, P4-7). The bound is kept as the paper computes it, and the
plan carries cautions instead (``ObjectCountPlan.warnings``, shown in the
planned rule's ``warning``): values far from normal (excess kurtosis above 2
or bimodality coefficient above 5/9 among the centred pilot values, from 8
objects, with ``sd_estimate="upper_95"``), and an SD above 50% CV of the
control mean, for which the paper advises acquiring more data and repeating
the estimate. A CV is not invariant to an additive offset, so intensities
must be background-subtracted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import _deprecated
from .._former_repeat import one_warning, read_former
from .._seeds import plain
from .._wording import listing, plural
from ..data_check import validate_input_tables
from ..evidence import Availability, Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import DecisionPolicy, Effort
from ..results import Decision
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput
from ..statistics import sd_upper_bound_factor, spread_is_zero
from ._averaging_cv import _central_sums, _shape_warning
from ._precision_steps import _MAX_TRUSTED_CV
from .base import Estimator, SamplingStrategy, unsupported_observation_unit
from .scalar_precision import _SD_ESTIMATES, required_objects

_REQUIRED = "planned_required_objects"
_MEASURED = "measured_objects"


@dataclass(frozen=True)
class ObjectCountPlan:
    """The number of objects to average, planned once from pilot samples.

    ``pilot_samples`` lists the ``sample_cols`` values of every pilot sample,
    so evaluation can refuse groups that were used for planning.
    ``warnings`` holds the plan's cautions (the module docstring); they are
    not part of ``parameters``, which define the plan.
    """

    feature: str
    target_cv: float
    sd_estimate: str
    sample_cols: tuple[str, ...]
    control: tuple[tuple[str, Any], ...]
    pilot_samples: tuple[tuple[Any, ...], ...]
    n_samples: int
    n_control_samples: int
    n_objects: int
    control_mean: float
    cv: float
    cv_upper_95: float
    required_objects: int
    warnings: tuple[str, ...] = ()

    @property
    def parameters(self) -> dict[str, Any]:
        """Everything that defines the plan, JSON-serializable, for provenance."""
        return {
            "feature": self.feature,
            "target_cv": self.target_cv,
            "sd_estimate": self.sd_estimate,
            "sample_cols": list(self.sample_cols),
            "control": {column: value for column, value in self.control},
            "pilot_samples": [list(sample) for sample in self.pilot_samples],
            "n_samples": self.n_samples,
            "n_control_samples": self.n_control_samples,
            "n_objects": self.n_objects,
            "control_mean": self.control_mean,
            "cv": self.cv,
            "cv_upper_95": self.cv_upper_95,
            "required_objects": self.required_objects,
        }


def _pilot_values(
    objects: pd.DataFrame,
    feature: str,
    sample_cols: tuple[str, ...],
    observation_unit: str | None,
    validation: Mapping[str, Any],
) -> tuple[ValidatedInput, np.ndarray]:
    """Validated pilot data and its feature values, or a clear error."""
    data = validate_input_tables(
        objects, feature_cols=[feature], observation_unit=observation_unit, **validation
    )
    reason = unsupported_observation_unit(data, OBJECT_LIKE_UNITS, method="mean-precision planning")
    if reason is not None:
        raise ValueError(reason)
    missing = [column for column in sample_cols if column not in data.objects.columns]
    if missing:
        raise ValueError(f"sample_cols {missing} are not columns of the pilot observations")
    blank = [column for column in sample_cols if data.objects[column].isna().any()]
    if blank:
        raise ValueError(
            f"sample_cols {blank} have missing values: every pilot object must belong to a "
            "sample; fill them in or leave those objects out of the pilot"
        )
    values = data.feature_values(feature)
    if np.isnan(values).any():
        raise ValueError(f"pilot values of {feature!r} contain missing values")
    if (values < 0).any():
        raise ValueError(
            f"pilot values of {feature!r} are negative; the CV criterion applies to "
            "non-negative whole-cell measurements"
        )
    return data, values


@one_warning
def plan_objects_to_average(
    objects: pd.DataFrame,
    *,
    feature: str,
    target_cv: float,
    sd_estimate: str,
    sample_cols: Sequence[str],
    control: Mapping[str, Any],
    observation_unit: str | None = None,
    **validation: Any,
) -> ObjectCountPlan:
    """Plan the number of objects to average from pilot samples (see the module docstring).

    ``sample_cols`` identify one sample (for example condition, repeat and
    well); ``control`` selects the control samples by column value, e.g.
    ``{"condition": "DMSO"}``. ``target_cv`` and ``sd_estimate`` have no
    defaults: the paper suggests 0.05 for independent and 0.025 for paired
    designs, and ``"upper_95"`` for small pilots. Other keyword arguments
    (``fovs``, ``fov_col``, ``group_cols``) go to input validation.
    """
    if isinstance(sample_cols, str) or not sample_cols:
        raise TypeError("sample_cols must be a non-empty sequence of column names")
    sample_cols, control = read_former(sample_cols=tuple(sample_cols), control=control)
    if not 0 < target_cv < 1:
        raise ValueError("target_cv must be between 0 and 1")
    if sd_estimate not in _SD_ESTIMATES:
        raise ValueError(f"sd_estimate must be one of {_SD_ESTIMATES}")
    if not isinstance(control, Mapping) or not control:
        raise ValueError("control must map at least one column to the control value")
    sample_cols = tuple(sample_cols)
    data, values = _pilot_values(objects, feature, sample_cols, observation_unit, validation)
    if len(values) < 2:
        raise ValueError("planning needs at least 2 pilot cells")
    by_sample, control_means, control_mean = _samples(data, values, sample_cols, control)
    sample_means = by_sample.mean()
    centred, sd, cv, cv_upper = _within_sample_cv(
        values, by_sample, len(sample_means), control_mean
    )
    required = required_objects(cv if sd_estimate == "point" else cv_upper, target_cv)
    if spread_is_zero(sd, values) or required < 1:
        # A plan of 0 objects would stop every evaluated group at its first FOV, even
        # an empty one: a constant feature within samples says nothing about averaging.
        raise ValueError(
            f"the within-sample SD of {feature!r} in the pilot is zero (every sample is "
            "constant), or so small that the plan would need no object, so no number of "
            "objects to average can be planned from it; use a feature that varies between "
            "the objects of a sample, or a larger pilot"
        )
    warnings = []
    if sd_estimate == "upper_95":
        warnings.append(_shape_warning(*_central_sums(centred), plural(data.observation_unit)))
    if cv > _MAX_TRUSTED_CV:
        warnings.append(
            f"the within-sample SD of {feature!r} exceeds 50% CV of the control mean; the paper "
            "advises acquiring more data and repeating the estimate"
        )
    return ObjectCountPlan(
        feature=feature,
        target_cv=float(target_cv),
        sd_estimate=sd_estimate,
        sample_cols=sample_cols,
        control=tuple((column, plain(value)) for column, value in control.items()),
        pilot_samples=tuple(
            tuple(plain(v) for v in (key if isinstance(key, tuple) else (key,)))
            for key in sample_means.index
        ),
        n_samples=len(sample_means),
        n_control_samples=len(control_means),
        n_objects=len(values),
        control_mean=control_mean,
        cv=cv,
        cv_upper_95=cv_upper,
        required_objects=required,
        warnings=tuple(text for text in warnings if text),
    )


def _samples(
    data: ValidatedInput,
    values: np.ndarray,
    sample_cols: tuple[str, ...],
    control: Mapping[str, Any],
) -> tuple[Any, pd.Series, float]:
    """The pilot values grouped by sample, the control samples' means and the control's
    mean; ``ValueError`` when the control is not in the pilot or its mean is not positive."""
    absent = [column for column in control if column not in data.objects.columns]
    if absent:
        raise ValueError(
            f"control columns {absent} are not columns of the pilot observations (they hold "
            f"{listing(list(data.objects.columns), limit=8)})"
        )
    table = data.objects.loc[:, list(sample_cols)].assign(_value=values)
    for column in control:
        if column not in table.columns:
            table[column] = data.objects[column]
    is_control = np.logical_and.reduce([table[c].eq(v).to_numpy() for c, v in control.items()])
    # observed=True: with categorical sample columns, only the samples present in the
    # pilot, never unobserved combinations of categories (pandas 2 defaults to all).
    by_sample = table.groupby(list(sample_cols), sort=True, observed=True)["_value"]
    control_rows = table.loc[is_control]
    control_means = control_rows.groupby(list(sample_cols), observed=True)["_value"].mean()
    if control_means.empty:
        present = {
            column: listing([repr(v) for v in table[column].drop_duplicates()], limit=5)
            for column in control
        }
        raise ValueError(
            f"no pilot sample matches the control {dict(control)}; the pilot holds "
            + "; ".join(f"{column}: {values}" for column, values in present.items())
        )
    control_mean = float(control_rows["_value"].mean())
    if not control_mean > 0:
        raise ValueError("the control mean must be positive to express SD as a CV")
    return by_sample, control_means, control_mean


def _within_sample_cv(
    values: np.ndarray, by_sample: Any, n_samples: int, control_mean: float
) -> tuple[np.ndarray, float, float, float]:
    """The values centred on their sample's mean, their SD, as a CV of the control mean, and
    that CV's upper 95% bound."""
    if len(values) <= n_samples:
        # Centring each sample on its own mean leaves n_objects - n_samples degrees of
        # freedom: with one object per sample every centred value is 0 and the SD is 0.
        raise ValueError(
            "planning needs at least one pilot sample with 2 or more cells: with one cell per "
            "sample the within-sample SD is undefined"
        )
    centred = values - by_sample.transform("mean").to_numpy()
    sd = float(np.std(centred, ddof=1))
    cv = sd / control_mean
    cv_upper = cv * sd_upper_bound_factor(len(values) - 1)
    return centred, sd, cv, cv_upper


def _samples_seen(data: ValidatedInput, sample_cols: tuple[str, ...]) -> set[tuple] | None:
    """The samples present in the evaluated data, or ``None`` if they cannot be identified."""
    for table in (data.fovs, data.objects):
        if all(column in table.columns for column in sample_cols):
            rows = table.loc[:, list(sample_cols)].itertuples(index=False, name=None)
            return {tuple(plain(v) for v in row) for row in rows}
    return None


@dataclass(frozen=True)
class PlannedObjectCountEstimator(Estimator):
    """Reports a plan's required object count as evidence, if the plan applies to the group.

    Quantities: ``planned_required_objects``, the plan's count, and
    ``measured_objects``, the visible objects with a measured (non-missing)
    value of the planned feature. The plan counts measured objects, so an
    object without a value does not bring the group closer to it. The planned
    feature must be one of the evaluated feature columns.

    The plan does not apply (``not_applicable``, with the reason) when the rows
    are not objects, when the group's samples cannot be identified, or when the
    group contains a pilot sample, since that group would be judged on its own
    planning data.
    """

    plan: ObjectCountPlan
    name: str = "planned_objects"

    @property
    def parameters(self) -> dict[str, Any]:
        """The plan, which fully defines the evidence."""
        return {"plan": self.plan.parameters}

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The required count, or why the plan does not apply to this group.

        No incremental state: whether the group holds a pilot sample is read
        from its visible rows, so each prefix is a cheap full recomputation.
        """
        values = data.feature_values(self.plan.feature)  # fails early on a non-feature column
        reason = unsupported_observation_unit(
            data, OBJECT_LIKE_UNITS, method="a mean-precision cell plan"
        )
        if reason is None:
            seen = _samples_seen(data, self.plan.sample_cols)
            if seen is None:
                reason = (
                    f"sample columns {list(self.plan.sample_cols)} are not in the evaluated "
                    "tables, so pilot samples cannot be excluded"
                )
            elif seen & set(self.plan.pilot_samples):
                reason = "the evaluated group contains pilot samples used to make the plan"
        meta = {"units": "objects", "component": Component.POPULATION}
        lower, higher = Direction.LOWER_IS_BETTER, Direction.HIGHER_IS_BETTER
        if reason is not None:
            return Evidence(
                {
                    _REQUIRED: Quantity.not_applicable(reason, direction=lower, **meta),
                    _MEASURED: Quantity.not_applicable(reason, direction=higher, **meta),
                },
                {"undefined_reason": reason},
            )
        measured = int(np.count_nonzero(~np.isnan(values)))
        diagnostics: dict[str, Any] = {
            "plan": self.plan.parameters,
            "objects_without_value": len(values) - measured,
        }
        if self.plan.warnings:
            diagnostics["warning"] = "plan: " + "; ".join(self.plan.warnings)
        return Evidence(
            {
                _REQUIRED: Quantity.available(self.plan.required_objects, direction=lower, **meta),
                _MEASURED: Quantity.available(measured, direction=higher, **meta),
            },
            diagnostics,
        )


@dataclass(frozen=True)
class _PlannedCountPolicy(DecisionPolicy):
    """Stop once the objects measured for the planned feature reach the planned requirement."""

    name: str = "planned_count"

    def decide(self, evidence: Evidence, effort: Effort) -> tuple[Component, Decision | None]:
        """Undefined if the plan does not apply; else stop at the planned count."""
        quantity, measured = evidence.quantities[_REQUIRED], evidence.quantities[_MEASURED]
        if quantity.availability is not Availability.AVAILABLE:
            return Component.POPULATION, None
        reached = measured.value >= quantity.value
        return Component.POPULATION, Decision.STOP if reached else Decision.CONTINUE


@dataclass(frozen=True)
class PlannedObjectCountStrategy(SamplingStrategy):
    """Stop each group once it has the number of objects a JCB pilot plan requires.

    Objects count once they have a measured value of the planned feature. The
    multi-sample counterpart of ``ScalarPrecisionStrategy``: the
    requirement is fixed before evaluation, from pilot samples, instead of
    being re-estimated from each group's visible objects.
    """

    plan: ObjectCountPlan
    name: str = "scalar_precision_planned"

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ObjectCountPlan):
            raise TypeError("plan must be an ObjectCountPlan from plan_objects_to_average")
        if not self.plan.required_objects >= 1:
            raise ValueError(
                "the plan requires no object; plan_objects_to_average refuses such a pilot"
            )

    @property
    def parameters(self) -> dict[str, Any]:
        """The plan."""
        return {"plan": self.plan.parameters}

    @property
    def estimators(self) -> tuple[PlannedObjectCountEstimator]:
        """The plan as evidence for each group."""
        return (PlannedObjectCountEstimator(self.plan),)

    @property
    def policies(self) -> tuple[_PlannedCountPolicy]:
        """Stop at the planned count."""
        return (_PlannedCountPolicy(),)


__all__ = [
    "ObjectCountPlan",
    "PlannedObjectCountEstimator",
    "PlannedObjectCountStrategy",
    "plan_objects_to_average",
]


# Former object names, resolved with a DeprecationWarning (see morphocoverage._deprecated).
__getattr__ = _deprecated.module_getattr(
    __name__,
    globals(),
    (
        "plan_cells_to_average",
        "CellCountPlan",
        "PlannedCellCountEstimator",
        "PlannedCellCountStrategy",
    ),
)
