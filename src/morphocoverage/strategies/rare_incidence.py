"""Beta-binomial evidence on a labelled rare state's FOV incidence, with recurrence.

Described in ``docs/architecture.md``. The
objective's ``RareTarget`` states the incidence ``p*`` (``fov_incidence``) at
which a state must be detected in ``r`` (``min_fov_occurrences``) FOVs. With
``x`` of the ``k`` visible FOVs showing the labelled state (acquired FOVs
without observations count as negative) and a ``Beta(a, b)`` prior on the
per-FOV incidence ``p``, the posterior is ``Beta(a + x, b + k - x)``. The
estimator reports

    rare_incidence_risk = P(p >= p* | x, k)     while x < r,
    rare_incidence_risk = 0                     once x >= r,

the posterior probability that the state is at least as common as the target
yet has not been seen in ``r`` FOVs. A strategy stops the rare component when
the risk is at most ``max_risk``.

This is a Bayesian statement, not a frequentist guarantee. "Posterior risk
at most 5%" means: given the prior and these FOVs, the probability that the
incidence is at least ``p*`` is at most 5%. It does not mean that an
experiment run this way detects a state at incidence ``p*`` with probability
at least 95%; that is the binomial plan of ``rare_planning``. Choosing
``max_risk = 1 - detection_probability`` makes the two rules similar, not
equal, and how far apart they are depends on the prior.

With a uniform prior (``a = b = 1``) and no detections the risk is
``(1 - p*) ** (k + 1)``, so for ``min_fov_occurrences = 1`` the rule stops one
FOV before the binomial plan of ``rare_planning``; unlike that plan it stops
as soon as the state recurs. For ``min_fov_occurrences = r > 1`` the risk does
not depend on ``r`` while fewer than ``r`` positives are seen, so with no
detections the rule stops at the ``r = 1`` point, well before the plan for
``r`` occurrences (with ``p* = 0.05`` and 5% risk: 58 FOVs, against 93 for
``r = 2`` and 124 for ``r = 3``).
Recurrence counts positions, not objects: many labelled objects in one FOV are one
positive FOV.

How often it stops too early (audit 4, P4-8; 20,000 simulated acquisitions
with the state at exactly the target, ``p* = 0.05``, and ``max_risk =
0.05``): it stopped without the state seen in ``r`` FOVs in 5.2% of
acquisitions for ``r = 1``, 7.7% for ``r = 2`` and 9.6% for ``r = 3`` with a
uniform prior, in 13.9%, 18.3% and 20.9% with Jeffreys' prior ``(0.5, 0.5)``,
and in 55-69% when the state came in runs of 5 neighbouring FOVs (the
binomial plan: 29-55%). So ``max_risk`` is the false-stop rate only for
``r = 1`` under a uniform prior and independent FOVs; the rule is kept as
designed and not recalibrated, and its evidence carries this caution in
``warning``.

Assumptions and limits:

- ``label_col`` marks the rare state independently of the acquisition (an
  observation column of booleans or 0/1), and must be available per FOV when
  it is acquired for live use;
- FOVs are exchangeable and independent; clustering across neighbouring FOVs
  makes the posterior overconfident;
- detection within a positive FOV is certain;
- the prior is explicit and part of the result's provenance.

A target without ``fov_incidence`` gives ``not_applicable``: object prevalence
is never converted to FOV incidence.

Shared-private: ``_positive_fovs`` (the one count of FOVs holding a labelled
state, also used by ``outcomes.RareStateDetected``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from .base import Estimator, SamplingStrategy

QUANTITY = "rare_incidence_risk"
# The false-stop rates beside every available risk (audit 4, P4-8; the module docstring).
_CAUTION = (
    "a posterior risk, not a detection guarantee: with the state at exactly the target "
    "incidence (5%, max_risk 5%) the rule stopped short of r positive FOVs in 5.2%, 7.7% and "
    "9.6% of simulated acquisitions for r = 1, 2 and 3 with a uniform prior, 14-21% with "
    "Jeffreys' prior, and 55-69% when the state came in runs of 5 neighbouring FOVs"
)
_META = {
    "direction": Direction.LOWER_IS_BETTER,
    "units": "probability",
    "component": Component.RARE,
}


def incidence_risk(
    positive: int, fovs: int, target: float, occurrences: int, a: float, b: float
) -> float:
    """``P(p >= target)`` under ``Beta(a + positive, b + fovs - positive)``, or 0 once recurrent."""
    if positive >= occurrences:
        return 0.0
    return float(stats.beta.sf(target, a + positive, b + fovs - positive))


def _positive_fovs(data: ValidatedInput, label_col: str) -> int | None:
    """Number of FOVs of ``data`` with at least one observation labelled with the state.

    ``None`` when a label is missing (the count is then unknown). Raises when
    ``label_col`` is not an observation column or holds values other than
    booleans or 0/1.
    """
    if label_col not in data.objects.columns:
        raise ValueError(f"label_col {label_col!r} is not a column of the observations")
    labels = data.objects[label_col]
    if labels.isna().any():
        return None
    if not (pd.api.types.is_bool_dtype(labels) or set(labels.unique()) <= {0, 1}):
        raise ValueError(
            f"label_col {label_col!r} must hold booleans or 0/1, got "
            f"{sorted(map(str, labels.unique()))[:5]}"
        )
    positions = fov_positions(data, data.fovs)[labels.to_numpy(dtype=bool)]
    return int(np.unique(positions[positions >= 0]).size)


def _check(label_col: str, prior_a: float, prior_b: float) -> None:
    """A label column and a proper Beta prior."""
    if not isinstance(label_col, str) or not label_col:
        raise TypeError("label_col must be the name of an observation label column")
    for value, name in ((prior_a, "prior_a"), (prior_b, "prior_b")):
        if isinstance(value, bool) or not (float(value) > 0 and np.isfinite(value)):
            raise ValueError(f"{name} must be a positive, finite Beta parameter")


@dataclass(frozen=True)
class RareIncidenceEstimator(Estimator):
    """Posterior risk that the target rare state is common enough yet not yet recurrent.

    Reports ``rare_incidence_risk`` (rare component, lower is better). See the
    module docstring.
    """

    label_col: str
    prior_a: float
    prior_b: float
    name: str = "rare_incidence"

    def __post_init__(self) -> None:
        _check(self.label_col, self.prior_a, self.prior_b)
        object.__setattr__(self, "prior_a", float(self.prior_a))
        object.__setattr__(self, "prior_b", float(self.prior_b))

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column and the Beta prior."""
        return {"label_col": self.label_col, "prior_a": self.prior_a, "prior_b": self.prior_b}

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The posterior risk for the objective's rare target."""
        target = objective.rare
        if target is None or target.fov_incidence is None:
            reason = (
                "the objective has no rare target"
                if target is None
                else "the rare target states object prevalence only, not FOV incidence"
            )
            return Evidence({QUANTITY: Quantity.not_applicable(reason, **_META)}, {})
        if self.label_col in data.feature_cols:
            raise ValueError(f"label_col {self.label_col!r} is a phenotype feature, not a label")
        positive, k = _positive_fovs(data, self.label_col), len(data.fovs)
        if positive is None:
            reason = f"missing labels in {self.label_col!r}"
            quantity = Quantity.not_applicable(reason, **_META)
            return Evidence({QUANTITY: quantity}, {"undefined_reason": reason})
        risk = incidence_risk(
            positive,
            k,
            target.fov_incidence,
            target.min_fov_occurrences,
            self.prior_a,
            self.prior_b,
        )
        diagnostics = {
            "positive_fovs": positive,
            "visible_fovs": k,
            "posterior": [self.prior_a + positive, self.prior_b + k - positive],
            "recurrent": positive >= target.min_fov_occurrences,
            "warning": _CAUTION,
        }
        return Evidence({QUANTITY: Quantity.available(risk, **_META)}, diagnostics)


@dataclass(frozen=True)
class RareIncidenceStrategy(SamplingStrategy):
    """Stop the rare component once ``rare_incidence_risk`` is at most ``max_risk``.

    ``RareIncidenceEstimator`` plus ``ThresholdPolicy(max_risk)``.
    ``max_risk`` bounds a posterior probability; ``1 - detection_probability``
    gives a rule close to, but not the same as, the binomial detection plan.
    It is the rate of stopping too early only for ``r = 1``, a uniform prior
    and independent FOVs: more occurrences, Jeffreys' prior or states in runs
    of neighbouring FOVs raise it (5-21%, and 55-69% in runs, at a nominal 5%;
    see the module docstring).
    """

    label_col: str
    prior_a: float
    prior_b: float
    max_risk: float
    name: str = "rare_incidence"

    def __post_init__(self) -> None:
        _check(self.label_col, self.prior_a, self.prior_b)
        if not 0 < self.max_risk < 1:
            raise ValueError("max_risk must be a probability between 0 and 1")

    @property
    def parameters(self) -> dict[str, Any]:
        """The estimator's settings and the risk threshold."""
        return {**self.estimators[0].parameters, "max_risk": self.max_risk}

    @property
    def estimators(self) -> tuple[RareIncidenceEstimator]:
        """The beta-binomial estimator."""
        return (RareIncidenceEstimator(self.label_col, self.prior_a, self.prior_b),)

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``rare_incidence_risk <= max_risk``."""
        return (ThresholdPolicy(QUANTITY, self.max_risk),)
