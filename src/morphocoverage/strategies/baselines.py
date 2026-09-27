"""Simple baseline strategies.

These intentionally simple methods exist to exercise the strategy framework and
provide essential comparators. They are not endorsed as default MorphoCoverage
estimators. Each stops at a fixed amount of acquisition effort; they answer
the population component of an objective only.

``FixedAreaStrategy`` stops at a fixed physical imaged area, read from an
FOV-table area column. Area is acquisition metadata rather than framework
effort, so an estimator reports the visible total; it is more comparable than
an FOV count when field size differs between acquisitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import is_integer
from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import DecisionPolicy, Effort, EffortPolicy, ThresholdPolicy
from ..results import Decision
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput
from .base import (
    ConstantState,
    Estimator,
    EstimatorState,
    SamplingStrategy,
    unsupported_observation_unit,
)


def _require_positive_int(value: object, name: str) -> int:
    """``value`` as a Python int of at least 1 (numpy integers too; ``bool`` is rejected)."""
    if not is_integer(value):
        raise TypeError(f"{name} must be an integer count")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


@dataclass(frozen=True)
class FixedFOVStrategy(SamplingStrategy):
    """Stop after a user-defined number of acquired FOVs."""

    target_fovs: int
    name: str = "fixed_fov"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_fovs", _require_positive_int(self.target_fovs, "target_fovs")
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Configured target number of acquired FOVs."""
        return {"target_fovs": self.target_fovs}

    @property
    def estimators(self) -> tuple:
        """None: the rule uses framework-owned effort only."""
        return ()

    @property
    def policies(self) -> tuple[EffortPolicy]:
        """Stop once acquired FOVs, including zero-object FOVs, reach the target."""
        return (EffortPolicy("n_fovs", self.target_fovs),)


def _object_rows_evidence(data: ValidatedInput) -> Evidence:
    """No quantities; the reason the object count is undefined, if it is."""
    reason = unsupported_observation_unit(data, OBJECT_LIKE_UNITS, method="FixedObjectStrategy")
    return Evidence(diagnostics={} if reason is None else {"undefined_reason": reason})


class _ObjectRowsCheck(Estimator):
    """Explains, in the result's diagnostics, why an object count is undefined."""

    name = "object_rows_check"

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The reason, if rows are not objects."""
        return _object_rows_evidence(data)

    def start(self, template: ValidatedInput, *, objective: ObjectiveConfig) -> EstimatorState:
        """The unit is fixed per group, so the evidence never changes."""
        return ConstantState(_object_rows_evidence(template))


@dataclass(frozen=True)
class _ObjectCountPolicy(DecisionPolicy):
    """Legacy object-count policy with explicit row-semantics guard."""

    target: int
    name: str = "effort:n_objects"

    @property
    def parameters(self) -> dict[str, Any]:
        """The effort measure and its target."""
        return {"effort": "n_objects", "target": self.target}

    def decide(self, evidence: Evidence, effort: Effort) -> tuple[Component, Decision | None]:
        """Undefined (``None``) unless rows are objects; else stop at the target count.

        A count of slices or patches is not a count of objects, so the rule
        gives no decision rather than an error; use FixedObservationStrategy
        for generic rows.
        """
        if effort.observation_unit not in OBJECT_LIKE_UNITS:
            return Component.POPULATION, None
        reached = effort.n_observations >= self.target
        return Component.POPULATION, Decision.STOP if reached else Decision.CONTINUE


@dataclass(frozen=True)
class FixedObjectStrategy(SamplingStrategy):
    """Legacy row-count baseline for object-style feature tables.

    For generic feature rows prefer FixedObservationStrategy, which makes the
    row semantics explicit through the framework observation_unit.
    """

    target_objects: int
    name: str = "fixed_objects"
    effort_rule = True  # a fixed amount of effort, not a measure of the data (not a field)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_objects", _require_positive_int(self.target_objects, "target_objects")
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Configured target number of detected objects."""
        return {"target_objects": self.target_objects}

    @property
    def estimators(self) -> tuple[_ObjectRowsCheck]:
        """A check that records why the rule is undefined for non-object rows."""
        return (_ObjectRowsCheck(),)

    @property
    def policies(self) -> tuple[_ObjectCountPolicy]:
        """Stop once object-like observations reach the legacy object target."""
        return (_ObjectCountPolicy(self.target_objects),)


@dataclass(frozen=True)
class FixedObservationStrategy(SamplingStrategy):
    """Stop after a user-defined number of phenotype observations.

    Unlike FixedObjectStrategy, this baseline is agnostic to whether rows are
    cells, nuclei, patches, slices, embeddings, or another declared unit.
    """

    target_observations: int
    name: str = "fixed_observations"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_observations",
            _require_positive_int(self.target_observations, "target_observations"),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Configured target number of observations."""
        return {"target_observations": self.target_observations}

    @property
    def estimators(self) -> tuple:
        """None: the rule uses framework-owned effort only."""
        return ()

    @property
    def policies(self) -> tuple[EffortPolicy]:
        """Stop once observations, of any unit, reach the target."""
        return (EffortPolicy("n_observations", self.target_observations),)


_AREA = "imaged_area"


def _check_area_col(area_col: object) -> None:
    """The name of the FOV table's area column."""
    if not isinstance(area_col, str) or not area_col:
        raise TypeError("area_col must be the FOV table's area column")


@dataclass(frozen=True)
class ImagedAreaEstimator(Estimator):
    """Total physical area of the visible FOVs, from the FOV-table column ``area_col``.

    Acquired FOVs without observations count, as they were imaged. Missing,
    zero or negative areas make the quantity ``not_applicable``: an unknown or
    impossible field size cannot be summed. A missing column is a
    configuration error and raises.

    The sum is the nominal area acquired, not the unique specimen area
    covered: overlapping FOVs (tiles with overlap, repeated positions) count
    their shared area more than once.
    """

    area_col: str
    name: str = "imaged_area"

    def __post_init__(self) -> None:
        _check_area_col(self.area_col)

    @property
    def parameters(self) -> dict[str, Any]:
        """The area column."""
        return {"area_col": self.area_col}

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The summed area of every visible FOV."""
        if self.area_col not in data.fovs.columns:
            raise ValueError(f"area_col {self.area_col!r} is not a column of the FOV table")
        if self.area_col in data.feature_cols:
            raise ValueError(f"area_col {self.area_col!r} is metadata, not a phenotype feature")
        column = data.fovs[self.area_col]
        if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            raise ValueError(
                f"area_col {self.area_col!r} holds {column.dtype} values, not numbers; "
                "convert it to numeric areas first"
            )
        areas = column.to_numpy(dtype=float)
        meta = {
            "direction": Direction.HIGHER_IS_BETTER,
            "units": f"units of {self.area_col!r}",
            "component": Component.POPULATION,
        }
        if not np.isfinite(areas).all() or (areas <= 0).any():
            reason = f"missing, infinite, zero or negative areas in {self.area_col!r}"
            quantity = Quantity.not_applicable(reason, **meta)
            return Evidence({_AREA: quantity}, {"undefined_reason": reason})
        return Evidence({_AREA: Quantity.available(float(areas.sum()), **meta)})


@dataclass(frozen=True)
class FixedAreaStrategy(SamplingStrategy):
    """Stop once the visible FOVs cover at least ``target_area`` (in ``area_col`` units)."""

    target_area: float
    area_col: str
    name: str = "fixed_area"
    effort_rule = True  # a fixed amount of effort, not a measure of the data (not a field)

    def __post_init__(self) -> None:
        _check_area_col(self.area_col)
        if isinstance(self.target_area, bool) or not (
            self.target_area > 0 and math.isfinite(self.target_area)
        ):
            raise ValueError("target_area must be a positive, finite area")

    @property
    def parameters(self) -> dict[str, Any]:
        """The target area and the area column."""
        return {"target_area": float(self.target_area), "area_col": self.area_col}

    @property
    def estimators(self) -> tuple[ImagedAreaEstimator]:
        """The imaged-area sum."""
        return (ImagedAreaEstimator(self.area_col),)

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``imaged_area >= target_area``."""
        return (ThresholdPolicy(_AREA, float(self.target_area)),)
