"""Simple baseline strategies.

These intentionally simple methods exist to exercise the strategy framework and
provide essential comparators. They are not endorsed as default MorphoCoverage
estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..objectives import ObjectiveConfig
from ..results import Decision, StrategyResult
from ..schema import ValidatedInput
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

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        """Stop once the acquired FOVs, including zero-object FOVs, reach the target."""
        n_fovs = len(data.fovs)
        return StrategyResult(
            strategy=self.name,
            objective=objective,
            score=float(n_fovs),
            decision=Decision.STOP if n_fovs >= self.target_fovs else Decision.CONTINUE,
            diagnostics={"target_fovs": self.target_fovs},
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class FixedObjectStrategy(SamplingStrategy):
    """Stop after a user-defined number of detected objects."""

    target_objects: int
    name: str = "fixed_objects"

    def __post_init__(self) -> None:
        _require_positive_int(self.target_objects, "target_objects")

    @property
    def parameters(self) -> dict[str, Any]:
        """Configured target number of detected objects."""
        return {"target_objects": self.target_objects}

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        """Stop once the detected objects reach the target, regardless of FOV count."""
        n_objects = len(data.objects)
        return StrategyResult(
            strategy=self.name,
            objective=objective,
            score=float(n_objects),
            decision=(
                Decision.STOP if n_objects >= self.target_objects else Decision.CONTINUE
            ),
            diagnostics={"target_objects": self.target_objects},
            parameters=self.parameters,
        )
