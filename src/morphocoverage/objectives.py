"""Sampling objectives for MorphoCoverage analyses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .evidence import Component


class SamplingObjective(str, Enum):
    """High-level purpose of an acquisition decision."""

    POPULATION = "population"
    RARE_DISCOVERY = "rare_discovery"
    BALANCED = "balanced"


# Objective components a decision must cover for each objective.
REQUIRED_COMPONENTS: dict[SamplingObjective, frozenset[Component]] = {
    SamplingObjective.POPULATION: frozenset({Component.POPULATION}),
    SamplingObjective.RARE_DISCOVERY: frozenset({Component.RARE}),
    SamplingObjective.BALANCED: frozenset({Component.POPULATION, Component.RARE}),
}


def _probability(value: float | None, name: str) -> None:
    """Allow ``None``; otherwise require a number strictly between 0 and 1."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, kw_only=True)
class RareTarget:
    """The rare state an acquisition should detect, stated so that it is testable.

    Parameters
    ----------
    object_prevalence:
        Fraction of objects in the rare state.
    fov_incidence:
        Fraction of FOVs containing at least one object in the rare state. The
        same object prevalence spread over every FOV or confined to a few is a
        different acquisition problem, so at least one of the two is required.
    detection_probability:
        Required probability of detecting the state, if present at the target.
    min_fov_occurrences:
        Number of distinct FOVs the state must be seen in to count as detected,
        so that one FOV (possibly one artefact) is not enough.
    phenotype:
        How the rare state is defined: the name of an externally labelled
        column, or the estimator's resolution in phenotype space (for example
        "radius 0.5 in standardized feature space"). Without a definition a
        prevalence is not a testable quantity. It is provenance; estimators
        state how they use it.
    """

    detection_probability: float
    min_fov_occurrences: int
    phenotype: str
    object_prevalence: float | None = None
    fov_incidence: float | None = None

    def __post_init__(self) -> None:
        if self.object_prevalence is None and self.fov_incidence is None:
            raise ValueError("a rare target needs object_prevalence, fov_incidence or both")
        _probability(self.object_prevalence, "object_prevalence")
        _probability(self.fov_incidence, "fov_incidence")
        _probability(self.detection_probability, "detection_probability")
        if isinstance(self.min_fov_occurrences, bool) or not isinstance(
            self.min_fov_occurrences, int
        ):
            raise TypeError("min_fov_occurrences must be an integer count of FOVs")
        if self.min_fov_occurrences < 1:
            raise ValueError("min_fov_occurrences must be at least 1")
        if not isinstance(self.phenotype, str) or not self.phenotype.strip():
            raise ValueError(
                "phenotype must declare how the rare state is defined: a labelled "
                "column or the estimator's resolution"
            )


@dataclass(frozen=True)
class ObjectiveConfig:
    """User-defined acquisition objective.

    ``rare`` is required when rare-state discovery contributes to the objective
    (``rare_discovery`` and ``balanced``), and rejected otherwise so that it is
    never silently ignored. It describes the scientific target, not an
    estimator.
    """

    objective: SamplingObjective = SamplingObjective.POPULATION
    rare: RareTarget | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "objective", SamplingObjective(self.objective))
        except ValueError:
            allowed = [o.value for o in SamplingObjective]
            raise ValueError(
                f"unknown sampling objective {self.objective!r}; expected one of {allowed}"
            ) from None
        if self.rare is not None and not isinstance(self.rare, RareTarget):
            raise TypeError("rare must be a RareTarget")
        needs_rare = Component.RARE in self.required_components
        if not needs_rare and self.rare is not None:
            raise ValueError(
                "a rare target was given for a population objective; use "
                "rare_discovery or balanced so the rare target is not silently ignored"
            )
        if needs_rare and self.rare is None:
            raise ValueError("a RareTarget is required for rare_discovery and balanced objectives")

    @property
    def required_components(self) -> frozenset[Component]:
        """The objective components a decision must cover."""
        return REQUIRED_COMPONENTS[self.objective]
