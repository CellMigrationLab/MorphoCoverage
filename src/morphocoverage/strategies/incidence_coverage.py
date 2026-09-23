"""Incidence-based sample coverage and Chao2 richness for externally labelled states.

Roadmap Phase 5 (docs/proposals/candidate_estimators_roadmap.md), restricted
to the category source the roadmap accepts first: labels fixed independently
of the evaluated acquisition (``label_col``, for example an annotated cell
type). The FOV is the sampling unit: with ``T`` visible FOVs (acquired FOVs
without observations included), ``Q_k`` states seen in exactly ``k`` FOVs and
``U = sum_k k Q_k`` incidences,

- sample coverage (Chao and Jost 2012, incidence data), the estimated share
  of incidence belonging to states already seen:

      C = 1 - (Q_1 / U) * ((T - 1) Q_1 / ((T - 1) Q_1 + 2 Q_2));

- Chao2 richness (Chao 1987), with the bias-corrected form when ``Q_2 = 0``:

      S = S_obs + ((T - 1) / T) Q_1^2 / (2 Q_2)          if Q_2 > 0
      S = S_obs + ((T - 1) / T) Q_1 (Q_1 - 1) / 2        otherwise,

  reported as completeness ``S_obs / S``.

Many cells of one state in one FOV are one incidence, so these count
recurrence across positions, not cells. Assumptions and limits:

- the labels must be fixed without looking at the evaluated acquisition; a
  partition fitted to it breaks the meaning of singletons and doubletons;
- FOVs are treated as independent sampling units, which spatially correlated
  neighbours are not;
- coverage refers to the declared label system only, never to phenotypes the
  labels cannot express;
- detection within an FOV is taken as certain.

Missing labels make both quantities ``not_applicable``. Fewer than
``min_fovs`` visible FOVs, or no labelled observations, give
``insufficient_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from .base import Estimator, SamplingStrategy

QUANTITIES = ("incidence_coverage", "chao2_completeness")
_UNITS = {
    "incidence_coverage": "fraction of incidence in observed states",
    "chao2_completeness": "observed states / Chao2 estimated states",
}


def incidence_frequencies(labels: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Number of distinct FOVs in which each observed label occurs."""
    pairs = pd.DataFrame({"label": labels, "fov": positions}).drop_duplicates()
    return pairs.groupby("label", sort=True).size().to_numpy()


def incidence_coverage(incidences: np.ndarray, n_units: int) -> float:
    """Chao and Jost (2012) sample coverage of incidence data from ``n_units`` sampling units."""
    q1, q2 = int(np.sum(incidences == 1)), int(np.sum(incidences == 2))
    total = int(np.sum(incidences))
    if q1 == 0:
        return 1.0
    shrink = (n_units - 1) * q1 / ((n_units - 1) * q1 + 2 * q2)
    return 1.0 - q1 / total * shrink


def chao2(incidences: np.ndarray, n_units: int) -> float:
    """Chao2 estimate of the number of states, bias-corrected when no state occurs twice."""
    observed = len(incidences)
    q1, q2 = int(np.sum(incidences == 1)), int(np.sum(incidences == 2))
    factor = (n_units - 1) / n_units
    if q2 > 0:
        return observed + factor * q1**2 / (2 * q2)
    return observed + factor * q1 * (q1 - 1) / 2


def _quantities(**values: Any) -> dict[str, Quantity]:
    """Both quantities: available values, or the same unavailable state with its reason."""
    out = {}
    for name in QUANTITIES:
        meta = {
            "direction": Direction.HIGHER_IS_BETTER,
            "units": _UNITS[name],
            "component": Component.POPULATION,
        }
        if "not_applicable" in values:
            out[name] = Quantity.not_applicable(values["not_applicable"], **meta)
        elif "insufficient" in values:
            out[name] = Quantity.insufficient(values["insufficient"], **meta)
        else:
            out[name] = Quantity.available(values[name], **meta)
    return out


def _check(label_col, min_fovs) -> None:
    """A label column name and a minimum of at least 2 FOVs."""
    if not isinstance(label_col, str) or not label_col:
        raise TypeError("label_col must be the name of an observation label column")
    if isinstance(min_fovs, bool) or not isinstance(min_fovs, int) or min_fovs < 2:
        raise ValueError("min_fovs must be an integer of at least 2")


@dataclass(frozen=True)
class IncidenceCoverageEstimator(Estimator):
    """Incidence sample coverage and Chao2 completeness of an external label system.

    Reports ``incidence_coverage`` and ``chao2_completeness`` (population,
    higher is better). ``label_col`` is an observation column of labels fixed
    independently of the acquisition; it is metadata and must not be a
    phenotype feature.
    """

    label_col: str
    min_fovs: int
    name: str = "incidence_coverage"

    def __post_init__(self) -> None:
        _check(self.label_col, self.min_fovs)

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column and the minimum number of FOVs."""
        return {"label_col": self.label_col, "min_fovs": self.min_fovs}

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
        incidences = incidence_frequencies(labels.to_numpy(), fov_positions(data, data.fovs))
        estimated = chao2(incidences, n_units)
        diagnostics = {
            "sampling_units": n_units,
            "observed_states": len(incidences),
            "singletons": int(np.sum(incidences == 1)),
            "doubletons": int(np.sum(incidences == 2)),
            "incidences": int(np.sum(incidences)),
            "chao2_states": estimated,
        }
        values = {
            "incidence_coverage": incidence_coverage(incidences, n_units),
            "chao2_completeness": len(incidences) / estimated,
        }
        return Evidence(_quantities(**values), diagnostics)


@dataclass(frozen=True)
class IncidenceCoverageStrategy(SamplingStrategy):
    """Stop when ``quantity`` (coverage or Chao2 completeness) reaches ``target``.

    ``IncidenceCoverageEstimator`` plus ``ThresholdPolicy(target)`` on the
    chosen quantity, which is higher-is-better.
    """

    label_col: str
    quantity: str
    target: float
    min_fovs: int

    def __post_init__(self) -> None:
        _check(self.label_col, self.min_fovs)
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        if not 0 < self.target <= 1:
            raise ValueError("target must be a fraction in (0, 1]")

    @property
    def name(self) -> str:
        """``incidence_coverage`` or ``chao2_completeness``."""
        return self.quantity

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column, quantity, target and minimum FOVs."""
        return {
            "label_col": self.label_col,
            "quantity": self.quantity,
            "target": self.target,
            "min_fovs": self.min_fovs,
        }

    @property
    def estimators(self) -> tuple[IncidenceCoverageEstimator]:
        """The incidence estimator."""
        return (IncidenceCoverageEstimator(self.label_col, self.min_fovs),)

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``quantity >= target``."""
        return (ThresholdPolicy(self.quantity, self.target),)
