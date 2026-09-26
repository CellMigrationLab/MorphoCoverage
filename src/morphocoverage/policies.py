"""Decision policies: turn evidence and effort into continue/stop.

Estimation and decision are separate so that thresholds are explicit and can
be swept after one run (``redecide``), and so that an objective is only
answered by evidence about it. A policy decides one objective component; the
framework combines component decisions per objective:

- a required component without any policy is ``UNSUPPORTED``, never ``STOP``;
- within a component: ``None`` (undefined) if any policy returns ``None``,
  ``STOP`` if all stop, otherwise ``CONTINUE``;
- overall: ``UNSUPPORTED`` if any required component is, else ``None`` if any
  is undefined, ``STOP`` if all stop, otherwise ``CONTINUE``. ``balanced``
  therefore stops only when both population and rare components stop.
"""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .evidence import Availability, Component, Direction, Evidence
from .objectives import ObjectiveConfig
from .provenance import class_identity, config_hash
from .results import Decision, StrategyResult, combine_decisions

_EFFORTS = ("n_fovs", "n_observations", "n_objects")


@dataclass(frozen=True, init=False)
class Effort:
    """Framework-owned acquisition effort.

    n_observations is canonical. Legacy construction with n_objects= is
    accepted so existing policy code remains source-compatible.
    """

    n_fovs: int
    n_observations: int
    observation_unit: str = "object"

    def __init__(
        self,
        n_fovs: int,
        n_observations: int | None = None,
        observation_unit: str = "object",
        *,
        n_objects: int | None = None,
    ) -> None:
        if n_observations is None and n_objects is None:
            raise TypeError("provide n_observations (or legacy n_objects)")
        if n_observations is not None and n_objects is not None and n_observations != n_objects:
            raise ValueError("n_observations and n_objects disagree")
        count = n_objects if n_observations is None else n_observations
        object.__setattr__(self, "n_fovs", n_fovs)
        object.__setattr__(self, "n_observations", count)
        object.__setattr__(self, "observation_unit", observation_unit)

    @property
    def n_objects(self) -> int:
        """Backward-compatible alias for n_observations."""
        return self.n_observations


class DecisionPolicy(ABC):
    """Maps evidence and effort to a decision for one objective component.

    A policy is configuration only and must be a pure function of its inputs.
    """

    name: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Serializable settings that define the policy, for provenance."""
        return {}

    @property
    def config_id(self) -> str:
        """Stable hash of the policy class and its parameters."""
        return config_hash(class_identity(type(self)), self.parameters)

    @abstractmethod
    def decide(self, evidence: Evidence, effort: Effort) -> tuple[Component, Decision | None]:
        """Return the component decided and ``CONTINUE``, ``STOP`` or ``None`` (undefined)."""
        raise NotImplementedError


@dataclass(frozen=True)
class ThresholdPolicy(DecisionPolicy):
    """Stop when a named quantity reaches ``threshold`` in its better direction.

    The component is the quantity's own, so population evidence can never
    answer a rare-discovery objective. An insufficient-data quantity gives
    ``CONTINUE``; a not-applicable one gives ``None``.
    """

    quantity: str
    threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, str) or not self.quantity:
            raise TypeError("quantity must be the name of an evidence quantity")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, numbers.Real):
            raise TypeError("threshold must be a number")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        object.__setattr__(self, "threshold", float(self.threshold))

    @property
    def name(self) -> str:
        """``threshold:<quantity>``."""
        return f"threshold:{self.quantity}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The quantity and its threshold."""
        return {"quantity": self.quantity, "threshold": self.threshold}

    def decide(self, evidence: Evidence, effort: Effort) -> tuple[Component, Decision | None]:
        """Compare the quantity with the threshold according to its direction."""
        if self.quantity not in evidence.quantities:
            raise ValueError(
                f"policy {self.name!r} needs quantity {self.quantity!r}; the estimators "
                f"reported {sorted(evidence.quantities)}"
            )
        quantity = evidence.quantities[self.quantity]
        if quantity.availability is Availability.INSUFFICIENT_DATA:
            return quantity.component, Decision.CONTINUE
        if quantity.availability is Availability.NOT_APPLICABLE:
            return quantity.component, None
        if quantity.direction is Direction.LOWER_IS_BETTER:
            reached = quantity.value <= self.threshold
        else:
            reached = quantity.value >= self.threshold
        return quantity.component, Decision.STOP if reached else Decision.CONTINUE


@dataclass(frozen=True)
class EffortPolicy(DecisionPolicy):
    """Stop once acquired effort (``"n_fovs"`` or ``"n_objects"``) reaches ``target``.

    Needs no estimator. ``component`` states which objective component the
    rule is claimed to answer; fixed counts are conventionally used for
    population summaries.
    """

    effort: str
    target: int
    component: Component = Component.POPULATION

    def __post_init__(self) -> None:
        if self.effort not in _EFFORTS:
            raise ValueError(f"effort must be one of {_EFFORTS}")
        if self.effort == "n_objects":
            # Legacy spelling of the same count: one rule, one config_id.
            object.__setattr__(self, "effort", "n_observations")
        if isinstance(self.target, bool) or not isinstance(self.target, int):
            raise TypeError("target must be an integer count")
        if self.target < 1:
            raise ValueError("target must be at least 1")
        object.__setattr__(self, "component", Component(self.component))

    @property
    def name(self) -> str:
        """``effort:<effort>``."""
        return f"effort:{self.effort}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The effort measure, its target and the component answered."""
        return {"effort": self.effort, "target": self.target, "component": self.component.value}

    def decide(self, evidence: Evidence, effort: Effort) -> tuple[Component, Decision | None]:
        """Stop once the effort measure reaches the target."""
        reached = getattr(effort, self.effort) >= self.target
        return self.component, Decision.STOP if reached else Decision.CONTINUE


def _combine(decisions: Sequence[Decision | None]) -> Decision | None:
    """Undefined if any decision is, stop if all stop, otherwise continue."""
    return combine_decisions(decisions)


def decide(
    evidence: Evidence,
    effort: Effort,
    policies: Iterable[DecisionPolicy],
    objective: ObjectiveConfig,
) -> tuple[Decision | None, dict[str, Decision | None]]:
    """Apply policies and combine them per objective (see the module docstring).

    Returns the overall decision and the decision per component. Components
    that have a policy but are not required are reported but do not affect the
    overall decision.
    """
    by_component: dict[Component, list[Decision | None]] = {}
    for policy in policies:
        outcome = policy.decide(evidence, effort)
        if not (isinstance(outcome, tuple) and len(outcome) == 2):
            raise TypeError(f"policy {policy.name!r} must return (component, decision)")
        component, decision = outcome
        if not isinstance(component, Component):
            raise TypeError(f"policy {policy.name!r} returned component {component!r}")
        if decision is not None:
            try:
                decision = Decision(decision)
            except ValueError:
                decision = Decision.UNSUPPORTED  # rejected below with the same message
            if decision is Decision.UNSUPPORTED:
                raise ValueError(
                    f"policy {policy.name!r} returned decision {outcome[1]!r}; expected "
                    "Decision.CONTINUE, Decision.STOP or None"
                )
        by_component.setdefault(component, []).append(decision)

    per_component: dict[Component, Decision | None] = {
        component: _combine(decisions) for component, decisions in by_component.items()
    }
    required = objective.required_components
    for component in required:
        per_component.setdefault(component, Decision.UNSUPPORTED)

    required_decisions = [per_component[c] for c in Component if c in required]
    if Decision.UNSUPPORTED in required_decisions:
        overall: Decision | None = Decision.UNSUPPORTED
    else:
        overall = _combine(required_decisions)
    ordered = {c.value: per_component[c] for c in Component if c in per_component}
    return overall, ordered


def describe(component: Any) -> dict[str, Any]:
    """Provenance entry for an estimator or policy."""
    return {
        "name": component.name,
        "type": class_identity(type(component)),
        "config_id": component.config_id,
        "parameters": dict(component.parameters),
    }


def strategy_config_id(estimator_ids: Iterable[str], policy_ids: Iterable[str]) -> str:
    """Configuration id of an estimator-plus-policy combination; names do not enter it."""
    return config_hash(
        "morphocoverage.strategy",
        {"estimators": list(estimator_ids), "policies": list(policy_ids)},
    )


def redecide(
    result: StrategyResult, policies: Sequence[DecisionPolicy], *, name: str
) -> StrategyResult:
    """Apply other policies to a stored result without re-running its estimators.

    The returned result equals evaluating ``SamplingStrategy(estimators,
    policies, name=name)`` on the same data, apart from ``compute_seconds``
    (kept from the original estimate). Use it for threshold sweeps.
    """
    policies = tuple(policies)
    effort = Effort(
        n_fovs=result.n_fovs,
        n_observations=result.n_observations,
        observation_unit=result.observation_unit,
    )
    decision, component_decisions = decide(result.evidence, effort, policies, result.objective)
    if "estimators" not in result.configuration:
        raise ValueError("result has no recorded estimator configuration to re-decide from")
    estimators = result.configuration["estimators"]
    configuration = {
        "estimators": estimators,
        "policies": [describe(policy) for policy in policies],
    }
    config_id = strategy_config_id(
        [e["config_id"] for e in estimators], [p.config_id for p in policies]
    )
    return replace(
        result,
        strategy=name,
        config_id=config_id,
        decision=decision,
        component_decisions=component_decisions,
        parameters=configuration,
        configuration=configuration,
    )
