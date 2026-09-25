"""Canonical strategy result types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .evidence import Evidence
from .objectives import ObjectiveConfig
from .schema import unit_plural


class Decision(str, Enum):
    """Continue/stop recommendations; any other value is rejected.

    ``UNSUPPORTED`` means the strategy has no evidence about a component the
    objective requires, for example a population-precision rule asked about
    rare-state discovery. It is never a disguised ``STOP``.
    """

    CONTINUE = "continue"
    STOP = "stop"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Timing:
    """Where the time for one decision went, in wall-clock seconds.

    - ``recompute_seconds``: full ``estimate`` calls on the visible prefix.
    - ``update_seconds``: incremental ``update`` calls on the newly revealed FOV.
    - ``initial_fit_seconds``: ``start`` calls, and warm-up updates for steps
      before ``min_fovs``; non-zero only on the first result of a replay.
    - ``framework_overhead_seconds``: everything else for this decision:
      building the visible prefix, isolating inputs, copying estimators,
      applying policies and building the result.
    - ``total_decision_seconds``: the sum, from the reveal of the FOV to the
      result. It is MorphoCoverage's own contribution to smart-microscopy
      latency; feature extraction and microscope control are not included.

    ``StrategyResult.compute_seconds`` equals ``recompute_seconds +
    update_seconds``.
    """

    recompute_seconds: float
    update_seconds: float
    initial_fit_seconds: float
    framework_overhead_seconds: float

    @property
    def compute_seconds(self) -> float:
        """Estimator time for this decision: recomputation plus incremental updates."""
        return self.recompute_seconds + self.update_seconds

    @property
    def total_decision_seconds(self) -> float:
        """Everything MorphoCoverage spent on this decision."""
        return self.compute_seconds + self.initial_fit_seconds + self.framework_overhead_seconds


@dataclass(frozen=True)
class StrategyResult:
    """One strategy evaluation for one analysis group and acquisition state.

    Built by the framework, never by a strategy: estimators supply
    ``evidence``, decision policies supply ``decision`` and
    ``component_decisions``, and the framework fills everything else from the
    acquisition record.

    - ``config_id`` identifies the estimators and policies exactly; ``strategy``
      is a human-readable name that several configurations may share.
    - ``component_decisions`` maps each objective component
      (``"population"``, ``"rare"``) that has a policy, or that the objective
      requires, to its decision. ``decision`` combines the required ones.
    - ``compute_seconds`` is the wall-clock time of the estimators' own calls
      for this decision (``estimate``, or ``update`` when replaying
      incrementally). ``runtime_seconds`` is an alias. ``timing`` breaks the
      whole decision down, including framework overhead; see ``Timing``.
    - ``feature_cols``, ``fov_key`` and ``observation_unit`` record the phenotype
      representation, spatial identity and row semantics;
      ``observation_unit_declared`` is ``False`` when the unit was assumed
      because the caller did not state it. Method settings
      (preprocessing, metric, thresholds) are in ``parameters`` and
      ``configuration``.
    """

    strategy: str
    config_id: str
    objective: ObjectiveConfig
    evidence: Evidence
    decision: Decision | None
    component_decisions: Mapping[str, Decision | None] = field(default_factory=dict)
    compute_seconds: float | None = None
    timing: Timing | None = None
    n_fovs: int | None = None
    n_observations: int | None = None
    observation_unit: str = "object"
    observation_unit_declared: bool = False
    group: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    configuration: Mapping[str, Any] = field(default_factory=dict)
    feature_cols: tuple[str, ...] = ()
    fov_key: tuple[str, ...] = ()

    def explain(self) -> str:
        """One line: the decision, each policy's rule with its evidence, and the effort.

        For example ``stop: lofo_novel_fov_upper = 0.24 <= 0.25 (population); after
        11 FOVs, 1100 objects``. A rule's own threshold is not a held-out check:
        the line names the quantity so the reader can see when a stop rests
        on the number of FOVs alone.
        """
        decision = "undefined" if self.decision is None else self.decision.value
        parts = []
        for policy in self.configuration.get("policies", []):
            parameters = policy.get("parameters", {})
            name = parameters.get("quantity")
            quantity = self.evidence.quantities.get(name) if name else None
            if quantity is not None and "threshold" in parameters:
                better = "<=" if quantity.direction.value == "lower_is_better" else ">="
                if quantity.value is None:
                    parts.append(f"{name} unavailable ({quantity.reason})")
                else:
                    parts.append(
                        f"{name} = {quantity.value:.3g} {better} {parameters['threshold']:g} "
                        f"({quantity.component.value})"
                    )
            elif "effort" in parameters and "target" in parameters:
                current = getattr(self, parameters["effort"], None)
                parts.append(
                    f"{parameters['effort']} = {current} >= {parameters['target']} "
                    f"({parameters.get('component', 'population')})"
                )
            else:
                parts.append(policy.get("name", "policy"))
        unit = self.observation_unit if self.observation_unit_declared else "observation"
        effort = f"after {self.n_fovs} FOVs, {self.n_observations} {unit_plural(unit)}"
        rules = "; ".join(parts) if parts else "no decision policy"
        return f"{decision}: {rules}; {effort}"

    def without_timing(self) -> StrategyResult:
        """This result with the time measurements removed, for comparing results."""
        return replace(self, compute_seconds=None, timing=None)

    @property
    def n_objects(self) -> int | None:
        """Backward-compatible alias for n_observations."""
        return self.n_observations

    @property
    def runtime_seconds(self) -> float | None:
        """Alias of ``compute_seconds``."""
        return self.compute_seconds

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """The estimators' diagnostics."""
        return self.evidence.diagnostics
