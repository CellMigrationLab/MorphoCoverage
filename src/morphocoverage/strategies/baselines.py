"""Simple baseline strategies.

These intentionally simple methods exist to exercise the strategy framework and
provide essential comparators. They are not endorsed as default MorphoCoverage
estimators. Both are an effort policy with no estimator; they answer the
population component of an objective only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..policies import EffortPolicy
from .base import SamplingStrategy


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
    def estimators(self) -> tuple:
        """None: the rule uses framework-owned effort only."""
        return ()

    @property
    def policies(self) -> tuple[EffortPolicy]:
        """Stop once feature-table rows reach the legacy object target."""
        return (EffortPolicy("n_objects", self.target_objects),)


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
        return {"target_observations": self.target_observations}

    @property
    def estimators(self) -> tuple:
        return ()

    @property
    def policies(self) -> tuple[EffortPolicy]:
        return (EffortPolicy("n_observations", self.target_observations),)
