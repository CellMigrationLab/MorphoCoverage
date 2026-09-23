"""Simple baseline strategies.

These intentionally simple methods exist to exercise the strategy framework and
provide essential comparators. They are not endorsed as default MorphoCoverage
estimators. Each is an effort policy; they answer the population component
of an objective only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..evidence import Component, Evidence
from ..objectives import ObjectiveConfig
from ..policies import DecisionPolicy, Effort, EffortPolicy
from ..results import Decision
from ..schema import OBJECT_LIKE_UNITS, ValidatedInput
from .base import Estimator, EstimatorState, SamplingStrategy, unsupported_observation_unit


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer count")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True)
class FixedFOVStrategy(SamplingStrategy):
    """Stop after a user-defined number of acquired FOVs."""

    target_fovs: int
    name: str = "fixed_fov"

    def __post_init__(self) -> None:
        _require_positive_int(self.target_fovs, "target_fovs")

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
        return _ConstantState(_object_rows_evidence(template))


class _ConstantState(EstimatorState):
    """Incremental state whose evidence does not depend on the revealed FOVs."""

    def __init__(self, evidence: Evidence) -> None:
        self.evidence = evidence

    def update(self, new: ValidatedInput) -> Evidence:
        """The same evidence as at the start."""
        return self.evidence


@dataclass(frozen=True)
class _ObjectCountPolicy(DecisionPolicy):
    """Legacy object-count policy with explicit row-semantics guard."""

    target: int
    name: str = "effort:n_objects"

    @property
    def parameters(self) -> dict[str, Any]:
        """The effort measure and its target."""
        return {"effort": "n_objects", "target": self.target}

    def decide(self, evidence, effort: Effort):
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

    def __post_init__(self) -> None:
        _require_positive_int(self.target_objects, "target_objects")

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
        _require_positive_int(self.target_observations, "target_observations")

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
