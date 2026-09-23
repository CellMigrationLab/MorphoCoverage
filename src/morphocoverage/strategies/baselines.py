"""Simple baseline strategies.

These intentionally simple methods exist to exercise the strategy framework and
provide essential comparators. They are not endorsed as default MorphoCoverage
estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..objectives import ObjectiveConfig
from ..results import StrategyResult
from ..schema import ValidatedInput
from .base import SamplingStrategy


@dataclass(frozen=True)
class FixedFOVStrategy(SamplingStrategy):
    """Stop after a user-defined number of acquired FOVs."""

    target_fovs: int
    name: str = "fixed_fov"

    def __post_init__(self) -> None:
        if self.target_fovs < 1:
            raise ValueError("target_fovs must be at least 1")

    @property
    def parameters(self):
        return {"target_fovs": self.target_fovs}

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        n_fovs = len(data.fovs)
        return StrategyResult(
            strategy=self.name,
            objective=objective,
            score=float(n_fovs),
            decision="stop" if n_fovs >= self.target_fovs else "continue",
            n_fovs=n_fovs,
            n_objects=len(data.objects),
            diagnostics={"target_fovs": self.target_fovs},
            parameters=self.parameters,
        )


@dataclass(frozen=True)
class FixedObjectStrategy(SamplingStrategy):
    """Stop after a user-defined number of detected objects."""

    target_objects: int
    name: str = "fixed_objects"

    def __post_init__(self) -> None:
        if self.target_objects < 1:
            raise ValueError("target_objects must be at least 1")

    @property
    def parameters(self):
        return {"target_objects": self.target_objects}

    def evaluate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> StrategyResult:
        n_objects = len(data.objects)
        return StrategyResult(
            strategy=self.name,
            objective=objective,
            score=float(n_objects),
            decision="stop" if n_objects >= self.target_objects else "continue",
            n_fovs=len(data.fovs),
            n_objects=n_objects,
            diagnostics={"target_objects": self.target_objects},
            parameters=self.parameters,
        )
