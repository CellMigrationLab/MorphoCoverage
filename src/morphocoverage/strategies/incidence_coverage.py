"""Incidence-based sample coverage and Chao2 richness for externally labelled states.

Described in ``docs/architecture.md``. Restricted to the category source that
the critical category rule (``docs/proposals/candidate_estimators_roadmap.md``)
accepts first: labels fixed independently of the evaluated acquisition
(``label_col``, for example an annotated object type). The FOV is the sampling
unit: with ``T`` visible FOVs (acquired FOVs without observations included),
``Q_k`` states seen in exactly ``k`` FOVs and ``U = sum_k k Q_k`` incidences,

- Chao2 richness (Chao 1987), with the bias-corrected form when ``Q_2 = 0``:

      S = S_obs + Q_0,  Q_0 = ((T - 1) / T) Q_1^2 / (2 Q_2)      if Q_2 > 0
                        Q_0 = ((T - 1) / T) Q_1 (Q_1 - 1) / 2    otherwise,

  reported as completeness ``S_obs / S``;

- sample coverage (Chao and Jost 2012, incidence data), the estimated share
  of incidence belonging to states already seen, as iNEXT computes it
  (``Chat.Sam``, Chao et al. 2014), with the same ``Q_0``:

      C = 1 - (Q_1 / U) * A,  A = T Q_0 / (T Q_0 + Q_1)  (A = 1 when Q_1 = 0).

  With ``Q_2 > 0``, ``A = (T - 1) Q_1 / ((T - 1) Q_1 + 2 Q_2)``, Chao and
  Jost's own form. With ``Q_2 = 0`` their paper writes ``A = 1``; the
  bias-corrected ``Q_0`` keeps coverage consistent with Chao2 and with the
  expected new states below, ``E[new] = (U / T) (1 - C)`` (for 1, 1, 1 and 4
  incidences in 5 FOVs: 0.657 rather than 0.571);

- expected new states in the next FOV (Chao et al. 2014, incidence
  extrapolation, Good-Toulmin type), with ``Q_0 = S - S_obs``:

      Q_0 * (1 - (1 - Q_1 / (T Q_0 + Q_1)) ** 1),

  zero when there are no singletons. This asks directly whether one more
  representative FOV is likely to reveal a new labelled state. Averaged over
  simulations with independent FOVs it tracks, and slightly undershoots, the
  number of new states the next FOV actually reveals, as Chao2 is a lower
  bound;

- the probability that the next FOV shows at least one new state. Under the
  same model each of the ``Q_0`` unseen states appears in a new unit with
  probability ``q = Q_1 / (T Q_0 + Q_1)``, independently, so

      P(new) = 1 - (1 - q) ** Q_0;

  in simulations with independent FOVs it matches the realized rate from
  about 20 FOVs and runs low with fewer, which would make stopping look safe
  too early;

- the Good-Turing plug-in estimate ``min(1, Q_1 / T)``: Q_1 / T estimates the
  expected number of new states in one more unit. That expectation would
  bound the probability of at least one (Markov's inequality), but an
  estimate of it is not a bound: two identical FOVs give Q_1 = 0 and so 0,
  although a third FOV can clearly show something new. In the independent-FOV
  simulations it stayed at or above the realized rate on average, which is an
  empirical property, not a guarantee;

- the leave-one-FOV-out rate ``x / T``, where ``x`` is the number of FOVs
  holding a state seen in no other FOV, and its one-sided Clopper-Pearson
  upper bound at nominal level ``confidence`` (``morphocoverage.bounds``;
  approximate, since the leave-one-FOV-out outcomes are not independent). The model-based
  probabilities above are 0 whenever no state is a singleton, even after two
  FOVs; the bound is not (0.78 for 0 of 2 at 95%) and is the quantity to stop
  on when few FOVs have been seen.

With one singleton and no doubletons, Chao2 adds ``(T - 1) / T * 1 * 0 / 2
= 0`` unseen states, so completeness and coverage are exactly 1 and the
expected new states 0, although a state seen once is evidence that more may
exist; a stop on these alone would ignore the singleton, and the
leave-one-FOV-out bound does not.

Many objects of one state in one FOV are one incidence, so these count
recurrence across positions, not objects. Assumptions and limits:

- the labels must be fixed without looking at the evaluated acquisition; a
  partition fitted to it breaks the meaning of singletons and doubletons;
- FOVs are treated as independent sampling units, which spatially correlated
  neighbours are not;
- coverage refers to the declared label system only, never to phenotypes the
  labels cannot express;
- detection within an FOV is taken as certain.

Missing labels make all quantities ``not_applicable``. Fewer than
``min_fovs`` visible FOVs, or no labelled observations, give
``insufficient_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import is_integer
from ..bounds import check_confidence, upper_bound
from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from .base import Estimator, SamplingStrategy

QUANTITIES = (
    "incidence_coverage",
    "chao2_completeness",
    "expected_new_states_next_fov",
    "prob_new_state_next_fov",
    "good_turing_new_state_estimate",
    "lofo_new_state_fov_fraction",
    "new_state_fov_upper",
)
_UNITS = {
    "incidence_coverage": "fraction of incidence in observed states",
    "chao2_completeness": "observed states / Chao2 estimated states",
    "expected_new_states_next_fov": "states",
    "prob_new_state_next_fov": "probability",
    "good_turing_new_state_estimate": "probability (plug-in estimate)",
    "lofo_new_state_fov_fraction": "fraction of FOVs (probability)",
    "new_state_fov_upper": "probability (nominal upper bound)",
}
_LOWER_IS_BETTER = {
    "expected_new_states_next_fov",
    "prob_new_state_next_fov",
    "good_turing_new_state_estimate",
    "lofo_new_state_fov_fraction",
    "new_state_fov_upper",
}
_PROBABILITIES = _LOWER_IS_BETTER - {"expected_new_states_next_fov"}


def fovs_with_unique_states(labels: np.ndarray, positions: np.ndarray) -> int:
    """Number of FOVs holding at least one label that no other FOV shows."""
    pairs = pd.DataFrame({"label": labels, "fov": positions}).drop_duplicates()
    singletons = pairs.groupby("label")["fov"].transform("size") == 1
    return int(pairs.loc[singletons, "fov"].nunique())


def incidence_frequencies(labels: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Number of distinct FOVs in which each observed label occurs."""
    pairs = pd.DataFrame({"label": labels, "fov": positions}).drop_duplicates()
    return pairs.groupby("label", sort=True).size().to_numpy()


def incidence_coverage(incidences: np.ndarray, n_units: int) -> float:
    """Chao and Jost (2012) sample coverage of incidence data from ``n_units`` sampling units.

    ``1 - (Q_1 / U) A`` with ``A = T Q_0 / (T Q_0 + Q_1)`` and the Chao2 ``Q_0``
    (bias-corrected when ``Q_2 = 0``), as iNEXT's ``Chat.Sam``.
    """
    q1 = int(np.sum(incidences == 1))
    if q1 == 0:
        return 1.0
    unseen = chao2(incidences, n_units) - len(incidences)
    shrink = n_units * unseen / (n_units * unseen + q1)
    return 1.0 - q1 / int(np.sum(incidences)) * shrink


def chao2(incidences: np.ndarray, n_units: int) -> float:
    """Chao2 estimate of the number of states, bias-corrected when no state occurs twice."""
    observed = len(incidences)
    q1, q2 = int(np.sum(incidences == 1)), int(np.sum(incidences == 2))
    factor = (n_units - 1) / n_units
    if q2 > 0:
        return observed + factor * q1**2 / (2 * q2)
    return observed + factor * q1 * (q1 - 1) / 2


def expected_new_states(incidences: np.ndarray, n_units: int, additional: int = 1) -> float:
    """Chao et al. (2014) expected number of new states in ``additional`` more sampling units."""
    q1 = int(np.sum(incidences == 1))
    unseen = chao2(incidences, n_units) - len(incidences)
    if q1 == 0 or unseen <= 0:
        return 0.0
    return unseen * (1 - (1 - q1 / (n_units * unseen + q1)) ** additional)


def prob_new_state(incidences: np.ndarray, n_units: int) -> float:
    """Chao et al. (2014) model probability that one more unit shows at least one new state."""
    q1 = int(np.sum(incidences == 1))
    unseen = chao2(incidences, n_units) - len(incidences)
    if q1 == 0 or unseen <= 0:
        return 0.0
    return 1 - (1 - q1 / (n_units * unseen + q1)) ** unseen


def good_turing_estimate(incidences: np.ndarray, n_units: int) -> float:
    """``min(1, Q_1 / T)``: Good-Turing plug-in estimate of a new state in the next unit.

    An estimate, not a bound: it is 0 whenever no state is a singleton.
    """
    return min(1.0, int(np.sum(incidences == 1)) / n_units)


def _quantities(**values: Any) -> dict[str, Quantity]:
    """All quantities: available values, or the same unavailable state with its reason."""
    out = {}
    for name in QUANTITIES:
        meta = {
            "direction": (
                Direction.LOWER_IS_BETTER
                if name in _LOWER_IS_BETTER
                else Direction.HIGHER_IS_BETTER
            ),
            "units": _UNITS[name],
            # How complete the set of labelled states is describes the distribution
            # of phenotypes, not its summary and not a declared RareTarget.
            "component": Component.DISTRIBUTION,
        }
        if "not_applicable" in values:
            out[name] = Quantity.not_applicable(values["not_applicable"], **meta)
        elif "insufficient" in values:
            out[name] = Quantity.insufficient(values["insufficient"], **meta)
        else:
            out[name] = Quantity.available(values[name], **meta)
    return out


def _check(label_col: str, min_fovs: int) -> None:
    """A label column name and a minimum of at least 2 FOVs."""
    if not isinstance(label_col, str) or not label_col:
        raise TypeError("label_col must be the name of an observation label column")
    if not is_integer(min_fovs) or min_fovs < 2:
        raise ValueError("min_fovs must be an integer of at least 2")


@dataclass(frozen=True)
class IncidenceCoverageEstimator(Estimator):
    """Incidence coverage, Chao2 completeness and next-FOV discovery for fixed labels.

    Reports ``incidence_coverage`` and ``chao2_completeness`` (higher is
    better) and ``expected_new_states_next_fov``, ``prob_new_state_next_fov``,
    ``good_turing_new_state_estimate``, ``lofo_new_state_fov_fraction`` and
    ``new_state_fov_upper`` (lower is better), all distribution evidence.
    ``label_col`` is an observation column of labels fixed independently of
    the acquisition; it is metadata and must not be a
    phenotype feature.
    """

    label_col: str
    min_fovs: int
    confidence: float = 0.95
    name: str = "incidence_coverage"

    def __post_init__(self) -> None:
        _check(self.label_col, self.min_fovs)
        object.__setattr__(self, "min_fovs", int(self.min_fovs))  # numpy ints too
        object.__setattr__(self, "confidence", check_confidence(self.confidence))

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column, the minimum number of FOVs and the bound's confidence."""
        return {
            "label_col": self.label_col,
            "min_fovs": self.min_fovs,
            "confidence": self.confidence,
        }

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """Coverage and Chao2 completeness from the visible FOVs' label incidences."""
        if self.label_col not in data.objects.columns:
            raise ValueError(f"label_col {self.label_col!r} is not a column of the observations")
        if self.label_col in data.feature_cols:
            raise ValueError(f"label_col {self.label_col!r} is a phenotype feature, not a label")
        labels = data.objects[self.label_col]
        if labels.isna().any():
            reason = f"missing labels in {self.label_col!r}"
            return Evidence(_quantities(not_applicable=reason), {"undefined_reason": reason})
        n_units = len(data.fovs)
        if n_units < self.min_fovs or len(labels) == 0:
            reason = f"fewer than {self.min_fovs} visible FOVs, or no labelled observations"
            return Evidence(_quantities(insufficient=reason), {"undefined_reason": reason})
        positions = fov_positions(data, data.fovs)
        incidences = incidence_frequencies(labels.to_numpy(), positions)
        unique_fovs = fovs_with_unique_states(labels.to_numpy(), positions)
        estimated = chao2(incidences, n_units)
        diagnostics = {
            "sampling_units": n_units,
            "observed_states": len(incidences),
            "singletons": int(np.sum(incidences == 1)),
            "doubletons": int(np.sum(incidences == 2)),
            "incidences": int(np.sum(incidences)),
            "chao2_states": estimated,
            "fovs_with_unique_states": unique_fovs,
        }
        values = {
            "incidence_coverage": incidence_coverage(incidences, n_units),
            "chao2_completeness": len(incidences) / estimated,
            "expected_new_states_next_fov": expected_new_states(incidences, n_units),
            "prob_new_state_next_fov": prob_new_state(incidences, n_units),
            "good_turing_new_state_estimate": good_turing_estimate(incidences, n_units),
            "lofo_new_state_fov_fraction": unique_fovs / n_units,
            "new_state_fov_upper": upper_bound(unique_fovs, n_units, self.confidence),
        }
        return Evidence(_quantities(**values), diagnostics)


@dataclass(frozen=True)
class IncidenceCoverageStrategy(SamplingStrategy):
    """Stop when ``quantity`` reaches ``target``.

    ``IncidenceCoverageEstimator`` plus ``ThresholdPolicy(target)`` on the
    chosen quantity: coverage or Chao2 completeness at least ``target``, or
    the expected new states, or either probability of a new state in the next
    FOV, at most ``target``.
    """

    label_col: str
    quantity: str
    target: float
    min_fovs: int
    confidence: float = 0.95

    def __post_init__(self) -> None:
        _check(self.label_col, self.min_fovs)
        object.__setattr__(self, "min_fovs", int(self.min_fovs))  # numpy ints too
        check_confidence(self.confidence)
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        if self.quantity in _PROBABILITIES:
            if not 0 < self.target < 1:
                raise ValueError("target must be a probability between 0 and 1")
        elif self.quantity in _LOWER_IS_BETTER:
            if not (self.target > 0 and np.isfinite(self.target)):
                raise ValueError("target must be a positive, finite number of states")
        elif not 0 < self.target <= 1:
            raise ValueError("target must be a fraction in (0, 1]")

    @property
    def name(self) -> str:
        """The chosen quantity's name."""
        return self.quantity

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column, quantity, target and minimum FOVs."""
        return {
            "label_col": self.label_col,
            "quantity": self.quantity,
            "target": self.target,
            "min_fovs": self.min_fovs,
            "confidence": self.confidence,
        }

    @property
    def estimators(self) -> tuple[IncidenceCoverageEstimator]:
        """The incidence estimator."""
        return (IncidenceCoverageEstimator(self.label_col, self.min_fovs, self.confidence),)

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``quantity`` reaches ``target`` in its better direction: ``>=`` for coverage and
        completeness, ``<=`` for the others."""
        return (ThresholdPolicy(self.quantity, self.target),)
