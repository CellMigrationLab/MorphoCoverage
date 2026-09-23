"""Sampling objectives for MorphoCoverage analyses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SamplingObjective(str, Enum):
    """High-level purpose of an acquisition decision."""

    POPULATION = "population"
    RARE_DISCOVERY = "rare_discovery"
    BALANCED = "balanced"


@dataclass(frozen=True)
class ObjectiveConfig:
    """User-defined acquisition objective.

    Rare-state settings are only required when rare-state discovery contributes
    to the objective. They describe the scientific target, not a particular
    estimator.
    """

    objective: SamplingObjective = SamplingObjective.POPULATION
    rare_prevalence: float | None = None
    target_detection_probability: float | None = None
    min_fov_occurrences: int | None = None

    def __post_init__(self) -> None:
        needs_rare_target = self.objective in {
            SamplingObjective.RARE_DISCOVERY,
            SamplingObjective.BALANCED,
        }
        rare_fields = (
            self.rare_prevalence,
            self.target_detection_probability,
            self.min_fov_occurrences,
        )
        if not needs_rare_target and any(value is not None for value in rare_fields):
            raise ValueError(
                "rare-state targets were given for a population objective; use "
                "rare_discovery or balanced so the rare target is not silently ignored"
            )
        if needs_rare_target and self.rare_prevalence is None:
            raise ValueError(
                "rare_prevalence is required for rare_discovery and balanced objectives"
            )
        if self.rare_prevalence is not None and not 0 < self.rare_prevalence < 1:
            raise ValueError("rare_prevalence must be between 0 and 1")
        if (
            self.target_detection_probability is not None
            and not 0 < self.target_detection_probability < 1
        ):
            raise ValueError("target_detection_probability must be between 0 and 1")
        if self.min_fov_occurrences is not None and (
            isinstance(self.min_fov_occurrences, bool)
            or not isinstance(self.min_fov_occurrences, int)
        ):
            raise TypeError("min_fov_occurrences must be an integer count of FOVs")
        if self.min_fov_occurrences is not None and self.min_fov_occurrences < 1:
            raise ValueError("min_fov_occurrences must be at least 1")
