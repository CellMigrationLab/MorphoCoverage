"""Canonical strategy result types, and the one view of a set of results per target.

Two decisions live in every result. ``decision`` is the objective's: it
combines the components the objective requires, and a rule with no evidence
about one of them is ``unsupported``, never ``stop``. ``rule_decision`` is
the rule's own verdict: what its policies say about the components they do
decide, whatever the objective. The first governs the live decision; the
second is what a self-assessment ("did each criterion say enough, by its own
threshold?"), a replay or a calibration reads, since those ask what the rule
said, not whether the rule alone could settle the objective.
``decision_view`` renders a set of results (one acquisition state, several
rules) as one line per target and the overall decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from ._wording import count
from .evidence import Component, Evidence
from .objectives import ObjectiveConfig
from .schema import unit_plural

# Plain words for each objective component, as reports and the live view print them.
COMPONENT_WORDS: dict[Component, str] = {
    Component.POPULATION: "Population summary",
    Component.DISTRIBUTION: "Phenotypic fingerprint",
    Component.RARE: "Rare phenotype",
}


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
      (``"population"``, ``"distribution"``, ``"rare"``) that has a policy, or
      that the objective requires, to its decision. ``decision`` combines the
      required ones; ``rule_decision`` combines the ones the rule decides.
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

        For example ``stop: lofo_novel_fov_upper = 0.24 (stop at <= 0.25, distribution);
        after 11 FOVs, 1100 objects``: the value, then the rule it is compared with,
        so a ``continue`` never reads as a false inequality. When the objective's
        decision differs from the rule's own (a rule with no evidence about a
        required target is ``unsupported``), both are printed: ``unsupported (the
        rule itself: continue): ...``. A rule's own threshold is not a held-out
        check: the line names the quantity so the reader can see when a stop rests
        on the number of FOVs alone.
        """
        decision = _word(self.decision)
        if self.rule_decision is not self.decision:
            # A rule that cannot settle the objective still said something itself.
            decision += f" (the rule itself: {_word(self.rule_decision)})"
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
                    value, threshold = _distinct(quantity.value, parameters["threshold"])
                    parts.append(
                        f"{name} = {value} (stop at {better} {threshold}, "
                        f"{quantity.component.value})"
                    )
            elif "effort" in parameters and "target" in parameters:
                current = getattr(self, parameters["effort"], None)
                parts.append(
                    f"{parameters['effort']} = {current} (stop at >= {parameters['target']}, "
                    f"{parameters.get('component', 'population')})"
                )
            else:
                parts.append(policy.get("name", "policy"))
        unit = self.observation_unit if self.observation_unit_declared else "observation"
        fovs = "an unknown number of FOVs" if self.n_fovs is None else count(self.n_fovs, "FOV")
        observations = (
            f"an unknown number of {unit_plural(unit)}"
            if self.n_observations is None
            else count(self.n_observations, _SINGULAR_WORDS.get(unit, unit), unit_plural(unit))
        )
        effort = f"after {fovs}, {observations}"
        rules = "; ".join(parts) if parts else "no decision policy"
        return f"{decision}: {rules}; {effort}"

    @property
    def rule_decision(self) -> Decision | None:
        """The rule's own verdict, whatever the objective (see the module).

        Combined over the components its policies decide: ``None`` if any is
        undefined, ``stop`` if all stop, otherwise ``continue``; ``None`` for a
        rule without a policy. Never ``unsupported``.
        """
        decided = [d for d in self.component_decisions.values() if d is not Decision.UNSUPPORTED]
        return combine_decisions(decided) if decided else None

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


def combine_decisions(decisions: Sequence[Decision | None]) -> Decision | None:
    """Undefined if any decision is, stop if all stop, otherwise continue.

    At least one decision is needed: with none, "all stop" would hold vacuously
    and nothing would read as ``STOP``.
    """
    decisions = list(decisions)
    if not decisions:
        raise ValueError("combine_decisions needs at least one decision; no decision is not a stop")
    if any(decision is None for decision in decisions):
        return None
    if all(decision is Decision.STOP for decision in decisions):
        return Decision.STOP
    return Decision.CONTINUE


def _word(decision: Decision | None) -> str:
    return "undefined" if decision is None else decision.value


# Units whose label is not the word a sentence uses for one of them.
_SINGULAR_WORDS = {"fov_summary": "field summary"}


def _distinct(value: float, threshold: float) -> tuple[str, str]:
    """``value`` and ``threshold`` as text, with the digits that tell them apart.

    Three significant digits, more when those would print a value that differs
    from the threshold as the threshold itself (0.2504 against 0.25 is not
    ``0.25 (stop at <= 0.25)``, which reads as a stop).
    """
    digits = 3
    while value != threshold and digits < 17 and f"{value:.{digits}g}" == f"{threshold:.{digits}g}":
        digits += 1
    return f"{value:.{digits}g}", f"{threshold:.{max(digits, 6)}g}"


def decision_view(results: Sequence[StrategyResult]) -> str:
    """One view of several rules on the same acquisition state: each target, then the overall.

    For each objective component, the decision combined over the rules that
    decide it (``combine_decisions``), with the number of rules; a required
    component that no rule decides is said so. The overall decision is the
    objective's: ``unsupported`` while a required component has no rule,
    otherwise the combination of the required components. The results must
    share one objective (one session, one acquisition state); the view
    answers "if the targets disagree, is that obvious?" (``docs/questions.md``
    O4).
    """
    results = list(results)
    if not results:
        raise ValueError("decision_view needs at least one result")
    objectives = {r.objective for r in results}
    if len(objectives) != 1:
        raise ValueError("decision_view needs results of one objective")
    objective = results[0].objective
    required = objective.required_components
    parts, overall = [], []
    for component in Component:
        decided = [
            r.component_decisions[component.value]
            for r in results
            if r.component_decisions.get(component.value, Decision.UNSUPPORTED)
            is not Decision.UNSUPPORTED
        ]
        word = COMPONENT_WORDS[component]
        if decided:
            combined = combine_decisions(decided)
            rules = count(len(decided), "rule")
            parts.append(f"{word}: {_word(combined)} ({rules})")
            if component in required:
                overall.append(combined)
        elif component in required:
            parts.append(f"{word}: no rule decides it")
            overall.append(Decision.UNSUPPORTED)
    if Decision.UNSUPPORTED in overall:
        missing = [
            COMPONENT_WORDS[c].lower()
            for c in Component
            if c in required
            and not any(
                r.component_decisions.get(c.value, Decision.UNSUPPORTED) is not Decision.UNSUPPORTED
                for r in results
            )
        ]
        final = f"unsupported (no rule decides the {', '.join(missing)})"
    else:
        final = _word(combine_decisions(overall))
    return "; ".join([*parts, f"Overall ({objective.objective.value} objective): {final}"])


__all__ = [
    "COMPONENT_WORDS",
    "Decision",
    "StrategyResult",
    "Timing",
    "combine_decisions",
    "decision_view",
]
