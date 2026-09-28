"""Sampling objectives for MorphoCoverage analyses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._checks import is_integer, is_number
from .evidence import Component


class SamplingObjective(str, Enum):
    """High-level purpose of an acquisition decision.

    Each objective names the components a decision must cover: the population
    summary, the phenotypic fingerprint (distribution), the rare phenotype.
    ``POPULATION``, ``DISTRIBUTION`` and ``RARE_DISCOVERY`` need one each;
    ``PHENOTYPE`` the summary and the fingerprint; ``BALANCED`` the summary
    and the rare state; ``ALL`` every one.
    """

    POPULATION = "population"
    DISTRIBUTION = "distribution"
    RARE_DISCOVERY = "rare_discovery"
    PHENOTYPE = "phenotype"
    BALANCED = "balanced"
    ALL = "all"


# Objective components a decision must cover for each objective.
REQUIRED_COMPONENTS: dict[SamplingObjective, frozenset[Component]] = {
    SamplingObjective.POPULATION: frozenset({Component.POPULATION}),
    SamplingObjective.DISTRIBUTION: frozenset({Component.DISTRIBUTION}),
    SamplingObjective.RARE_DISCOVERY: frozenset({Component.RARE}),
    SamplingObjective.PHENOTYPE: frozenset({Component.POPULATION, Component.DISTRIBUTION}),
    SamplingObjective.BALANCED: frozenset({Component.POPULATION, Component.RARE}),
    SamplingObjective.ALL: frozenset(
        {Component.POPULATION, Component.DISTRIBUTION, Component.RARE}
    ),
}


def _probability(value: float | None, name: str) -> float | None:
    """``None``, or ``value`` as a Python float strictly between 0 and 1.

    numpy numbers are accepted like Python ones (``np.float32(0.01)``) and
    returned as a Python float, so the target stays JSON-serializable.
    """
    if value is None:
        return None
    if not is_number(value):
        raise TypeError(f"{name} must be a number")
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


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
        for name in ("object_prevalence", "fov_incidence"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.detection_probability is None:
            raise TypeError("detection_probability is required: the chance of seeing the state")
        object.__setattr__(
            self,
            "detection_probability",
            _probability(self.detection_probability, "detection_probability"),
        )
        if not is_integer(self.min_fov_occurrences):
            raise TypeError("min_fov_occurrences must be an integer count of FOVs")
        if self.min_fov_occurrences < 1:
            raise ValueError("min_fov_occurrences must be at least 1")
        object.__setattr__(self, "min_fov_occurrences", int(self.min_fov_occurrences))
        if not isinstance(self.phenotype, str) or not self.phenotype.strip():
            raise ValueError(
                "phenotype must declare how the rare state is defined: a labelled "
                "column or the estimator's resolution"
            )


@dataclass(frozen=True)
class ObjectiveConfig:
    """User-defined acquisition objective.

    ``rare`` is required when rare-state discovery contributes to the objective
    (``rare_discovery``, ``balanced`` and ``all``), and rejected otherwise so
    that it is never silently ignored. It describes the scientific target, not
    an estimator.
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
                f"a rare target was given for a {self.objective.value} objective, which has no "
                "rare component; use rare_discovery, balanced or all so the rare target is not "
                "silently ignored"
            )
        if needs_rare and self.rare is None:
            raise ValueError(
                "a RareTarget is required for rare_discovery, balanced and all objectives"
            )

    @property
    def required_components(self) -> frozenset[Component]:
        """The objective components a decision must cover."""
        return REQUIRED_COMPONENTS[self.objective]


__all__ = [
    "REQUIRED_COMPONENTS",
    "ObjectiveConfig",
    "RareTarget",
    "SamplingObjective",
]
