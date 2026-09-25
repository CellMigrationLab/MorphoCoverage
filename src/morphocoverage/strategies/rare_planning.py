"""Binomial FOV-incidence planning for a rare target: a fixed-FOV rare-state comparator.

Roadmap 7.1 (docs/proposals/candidate_estimators_roadmap.md). If a rare
state is present in a fraction ``p`` of representative FOVs (the target's
``fov_incidence``) and FOVs are independent, the number of FOVs showing it
after ``n`` FOVs is Binomial(n, p). The required number of FOVs is the
smallest ``n`` with

    P(X >= r) >= beta,

for ``r = min_fov_occurrences`` and ``beta = detection_probability``. For
``r = 1`` this is ``ceil(log(1 - beta) / log(1 - p))``.

The plan is fixed before acquisition: the rule uses no phenotype data, so it
is a planning calculation and a comparator for adaptive rare-state methods,
not an adaptive coverage estimator. Its assumptions:

- ``fov_incidence`` is the design value supplied by the user. An
  ``object_prevalence`` is never converted into FOV incidence, since that
  needs observations per FOV and within-FOV clustering; such a target gives
  ``not_applicable`` evidence (roadmap 7.2).
- FOVs are independent. If the rare state clusters across neighbouring FOVs,
  the binomial is anti-conservative and overstates detection probability.
- Detection within a positive FOV is assumed certain.

The rule answers the rare component only, so on its own it is ``unsupported``
for population objectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from ..evidence import Availability, Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import DecisionPolicy, Effort
from ..results import Decision
from ..schema import ValidatedInput
from .base import ConstantState, Estimator, EstimatorState, SamplingStrategy

_QUANTITY = "planned_fovs_for_rare_target"


def fovs_to_detect(fov_incidence: float, detection_probability: float, occurrences: int) -> int:
    """Smallest n with ``P(Binomial(n, fov_incidence) >= occurrences) >= detection_probability``.

    Uses ``scipy.stats.binom.sf``, which is monotone increasing in ``n``;
    the answer is found by doubling then bisection.
    """
    for value, label in ((fov_incidence, "fov_incidence"), (detection_probability, "detection")):
        if not 0 < value < 1:
            raise ValueError(f"{label} must be between 0 and 1")
    if isinstance(occurrences, bool) or not isinstance(occurrences, int) or occurrences < 1:
        raise ValueError("occurrences must be a positive integer")

    def detected(n: int) -> bool:
        return stats.binom.sf(occurrences - 1, n, fov_incidence) >= detection_probability

    low, high = occurrences - 1, occurrences
    while not detected(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if detected(middle) else (middle, high)
    return high


def first_detection_fovs(fov_incidence: float, detection_probability: float) -> int:
    """Closed form for one occurrence: ``ceil(log(1 - beta) / log(1 - p))``."""
    return math.ceil(math.log(1 - detection_probability) / math.log(1 - fov_incidence))


def _plan_evidence(objective: ObjectiveConfig) -> Evidence:
    """The planned FOV count for the objective's rare target, or why there is none."""
    meta = {"direction": Direction.LOWER_IS_BETTER, "units": "FOVs", "component": Component.RARE}
    target = objective.rare
    if target is None:
        reason = "the objective has no rare target"
    elif target.fov_incidence is None:
        reason = (
            "the rare target states object prevalence only; converting it to FOV incidence "
            "needs observations per FOV and within-FOV clustering, so it is not assumed"
        )
    else:
        planned = fovs_to_detect(
            target.fov_incidence, target.detection_probability, target.min_fov_occurrences
        )
        diagnostics = {
            "fov_incidence": target.fov_incidence,
            "detection_probability": target.detection_probability,
            "min_fov_occurrences": target.min_fov_occurrences,
            "assumes_independent_fovs": True,
        }
        return Evidence({_QUANTITY: Quantity.available(planned, **meta)}, diagnostics)
    return Evidence(
        {_QUANTITY: Quantity.not_applicable(reason, **meta)}, {"undefined_reason": reason}
    )


class FOVIncidencePlanEstimator(Estimator):
    """Reports the binomial FOV count planned for the objective's rare target.

    The evidence depends only on the objective, so it is the same at every
    step and for every observation unit.
    """

    name = "fov_incidence_plan"

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The planned FOV count, or ``not_applicable`` with the reason."""
        return _plan_evidence(objective)

    def start(self, template: ValidatedInput, *, objective: ObjectiveConfig) -> EstimatorState:
        """Constant evidence: nothing to update as FOVs arrive."""
        return ConstantState(_plan_evidence(objective))


@dataclass(frozen=True)
class _PlannedFOVPolicy(DecisionPolicy):
    """Stop the rare component once the acquired FOVs reach the planned count."""

    name: str = "planned_fovs"

    def decide(self, evidence: Evidence, effort: Effort) -> tuple[Component, Decision | None]:
        """Undefined without a plan; otherwise stop at the planned number of FOVs."""
        quantity = evidence.quantities[_QUANTITY]
        if quantity.availability is not Availability.AVAILABLE:
            return Component.RARE, None
        reached = effort.n_fovs >= quantity.value
        return Component.RARE, Decision.STOP if reached else Decision.CONTINUE


class FOVIncidencePlanStrategy(SamplingStrategy):
    """Acquire the binomial FOV count planned from the objective's ``RareTarget``.

    Configuration comes from the objective, so the strategy has no settings
    of its own. Combine it with a population strategy for a ``balanced``
    objective.
    """

    def __init__(self) -> None:
        super().__init__(
            [FOVIncidencePlanEstimator()], [_PlannedFOVPolicy()], name="fov_incidence_plan"
        )
