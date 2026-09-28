"""Outcome criteria: did a set of FOVs preserve what the experiment needs?

An ``OutcomeMetric`` scores a set of FOVs against a reference of FOVs the
estimators never saw (a held-out reservoir, or the rest of a completed
acquisition) and returns a number, for example a relative mean error of 0.043.
An ``OutcomeCriterion`` adds the declared tolerance that turns the number into
an answer: *preserved*, *not preserved*, or *undefined* when the metric cannot
be computed. Every retrospective analysis uses this one contract, so threshold
logic exists in one place.

A criterion is deliberately not a ``DecisionPolicy``. A policy acts
prospectively on an estimator's evidence and decides whether to acquire more;
a criterion acts retrospectively on an outcome measured against hidden FOVs
and says whether a set of FOVs was in fact enough. Keeping them apart is what
lets a report say "the rule said enough" and "it was enough" separately.

Thresholds without a published meaning are provisional user choices: the
tolerance is part of the question, and reports state it with every answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._checks import is_integer, is_number
from ..outcomes import OutcomeMetric
from ..schema import ValidatedInput


class Preservation(str, Enum):
    """Result of assessing one set of FOVs against a reference."""

    PRESERVED = "preserved"
    NOT_PRESERVED = "not_preserved"
    UNDEFINED = "undefined"


_DIRECTIONS = ("lower_is_better", "higher_is_better")


@dataclass(frozen=True)
class OutcomeCriterion:
    """``metric`` within ``threshold`` in the given ``direction`` counts as preserved.

    ``direction`` is ``"lower_is_better"`` (errors: preserved when
    ``value <= threshold``) or ``"higher_is_better"`` (for example
    ``RareStateDetected``: preserved when ``value >= threshold``). An undefined
    metric value (``None``) is never preserved; ``assess`` counts a subset with
    too few observations for the metric (none, or fewer than the metric's
    ``min_observations``: one object has no covariance) as not preserved when
    the reference itself can be judged. ``description`` is the plain
    wording reports use, for example "the well mean of YAP/TAZ ratio within
    5%"; without it they use ``name``.
    """

    metric: OutcomeMetric
    threshold: float
    direction: str
    description: str | None = None

    def __post_init__(self) -> None:
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be text or None")
        if not isinstance(self.metric, OutcomeMetric):
            raise TypeError("metric must be an OutcomeMetric")
        if not is_number(self.threshold):
            raise TypeError("threshold must be a number")
        minimum = self.metric.min_observations
        if not is_integer(minimum) or minimum < 1:
            raise ValueError(
                f"the metric's min_observations must be an integer of at least 1, not {minimum!r}"
            )
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")
        object.__setattr__(self, "threshold", float(self.threshold))

    @property
    def name(self) -> str:
        """``<metric> <= threshold`` or ``<metric> >= threshold``."""
        comparison = "<=" if self.direction == "lower_is_better" else ">="
        return f"{self.metric.name} {comparison} {self.threshold:g}"

    @property
    def label(self) -> str:
        """The plain description if given, else ``name``."""
        return self.description or self.name

    @property
    def parameters(self) -> dict[str, Any]:
        """Provenance: the metric's identity and settings, the threshold and direction."""
        return {
            "metric": self.metric.name,
            "metric_config_id": self.metric.config_id,
            "metric_parameters": dict(self.metric.parameters),
            "threshold": self.threshold,
            "direction": self.direction,
            "description": self.description,
        }

    def judge(self, value: float | None) -> Preservation:
        """Preservation for an already computed metric value."""
        if value is None or not math.isfinite(value):
            return Preservation.UNDEFINED
        met = (
            value <= self.threshold
            if self.direction == "lower_is_better"
            else (value >= self.threshold)
        )
        return Preservation.PRESERVED if met else Preservation.NOT_PRESERVED

    def assess(
        self, reference: ValidatedInput, subset: ValidatedInput
    ) -> tuple[Preservation, float | None]:
        """Whether ``subset`` preserves ``reference`` under this criterion, and the value.

        FOVs without observations are acquired effort, not missing data: a
        subset with no observations at all, or fewer than the metric needs
        (``OutcomeMetric.min_observations``), cannot preserve a reference that
        has some, so when the metric is undefined for that reason the answer
        is not preserved rather than undefined. Otherwise a few fields holding
        a single object would leave the denominator of a reliability as
        undefined draws instead of counting as the failures they are. When the
        reference itself makes the metric undefined (a missing value in it, a
        rare state it does not show), no subset could have been judged, so the
        answer stays undefined: the metric of the reference against itself
        decides which.
        """
        value = self.metric.evaluate(reference, subset)
        too_few = len(subset.objects) < self.metric.min_observations
        if value is None and too_few and len(reference.objects) > 0:
            if self.metric.evaluate(reference, reference) is None:
                return Preservation.UNDEFINED, None
            return Preservation.NOT_PRESERVED, None
        return self.judge(value), value


__all__ = ["OutcomeCriterion", "Preservation"]
