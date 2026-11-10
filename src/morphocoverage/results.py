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

Every stop is printed with what it rests on: the fields seen so far read as
a random sample of the well's positions (``order_check``). ``explain`` and
``decision_view`` take the ``OrderCheck`` of the fields seen (live, the
``LiveUpdate`` passes it) and print its caution beside a stop, or the
assumption itself when no check was run; a flagged trend never changes a
decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from ._frozen import own
from ._wording import count, listing, rate
from .evidence import Component, Evidence
from .objectives import ObjectiveConfig
from .order_check import RANDOM_SAMPLE_ASSUMPTION, RARE_ASSUMPTION, OrderCheck
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

    def __post_init__(self) -> None:
        # Owned, read-only snapshots (issue 108): ``parameters`` and ``configuration`` are
        # separate copies even when built from one dictionary (``redecide``, issue 112).
        for name in ("component_decisions", "group", "parameters", "configuration"):
            object.__setattr__(self, name, own(getattr(self, name)))

    def explain(self, order_check: OrderCheck | None = None) -> str:
        """One line: the decision, each policy's rule with its evidence, and the effort.

        For example ``stop: lofo_novel_fov_upper = 0.24 (stop at <= 0.25, distribution);
        after 11 FOVs, 1100 objects; treats the fields as a random sample ...``: the
        value, then the rule it is compared with, so a ``continue`` never reads as a
        false inequality. When the objective's decision differs from the rule's own,
        both are printed: a rule that decides some of the objective's targets but not
        all reads ``stop (decides the population summary only)``, and a rule that
        decides none of them ``unsupported (the rule itself: continue)``. A rule's
        own threshold is not a held-out check: the line names the quantity so the
        reader can see when a stop rests on the number of FOVs alone.

        A rule that treats objects as independent draws says so
        (``assumes_spatial_uniformity``). Beside a stop, the line says what the
        stop rests on: the caution of ``order_check`` (the acquisition-order check
        of the fields seen so far), or without one the assumption that the fields
        are a random sample of the well's positions; a rule deciding the rare
        phenotype adds where the fields have looked.
        """
        effort = f"after {self._fovs()}, {self._observations()}"
        line = f"{self._decision_words()}: {'; '.join(self._rules())}; {effort}"
        diagnostics = self.evidence.diagnostics
        if diagnostics.get("assumes_spatial_uniformity") or diagnostics.get(
            "assumes_random_object_sampling"
        ):
            line += "; treats observations as independent draws: assumes a spatially uniform sample"
        if Decision.STOP in (self.decision, self.rule_decision):
            line += f"; {stop_caution(order_check, rare=self.decides_rare)}"
        return line

    def _decision_words(self) -> str:
        """The decision, with the rule's own verdict when the objective's differs."""
        if self.rule_decision is self.decision:
            return _word(self.decision)
        required = self.objective.required_components
        decided = [
            c
            for c in Component
            if c in required
            and self.component_decisions.get(c.value, Decision.UNSUPPORTED)
            is not Decision.UNSUPPORTED
        ]
        if self.decision is Decision.UNSUPPORTED and decided:
            # It settles some required targets, not all: say which, not "unsupported".
            own = combine_decisions([self.component_decisions[c.value] for c in decided])
            words = listing([COMPONENT_WORDS[c].lower() for c in decided])
            return f"{_word(own)} (decides the {words} only)"
        return f"{_word(self.decision)} (the rule itself: {_word(self.rule_decision)})"

    def _rules(self) -> list[str]:
        """Each policy's rule beside its evidence, or ``no decision policy``."""
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
        return parts or ["no decision policy"]

    def _fovs(self) -> str:
        return "an unknown number of FOVs" if self.n_fovs is None else count(self.n_fovs, "FOV")

    def _observations(self) -> str:
        unit = self.observation_unit if self.observation_unit_declared else "observation"
        if self.n_observations is None:
            return f"an unknown number of {unit_plural(unit)}"
        return count(self.n_observations, _SINGULAR_WORDS.get(unit, unit), unit_plural(unit))

    @property
    def decides_rare(self) -> bool:
        """Whether a policy of this rule decides the rare phenotype."""
        decision = self.component_decisions.get(Component.RARE.value, Decision.UNSUPPORTED)
        return Component.RARE.value in self.component_decisions and (
            decision is not Decision.UNSUPPORTED
        )

    def criteria(self, component: Component) -> list[str]:
        """This rule's thresholds on ``component``'s quantities, as ``averaging_cv <= 0.05``."""
        found = []
        for policy in self.configuration.get("policies", []):
            parameters = policy.get("parameters", {})
            quantity = self.evidence.quantities.get(parameters.get("quantity"))
            if quantity is None or "threshold" not in parameters:
                continue
            if quantity.component is component:
                better = "<=" if quantity.direction.value == "lower_is_better" else ">="
                found.append(f"{parameters['quantity']} {better} {parameters['threshold']:g}")
        return found

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
    def runtime_seconds(self) -> float | None:
        """Alias of ``compute_seconds``."""
        return self.compute_seconds

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """The estimators' diagnostics."""
        return self.evidence.diagnostics


# An objective with no target (``SamplingObjective.NO_TARGET``): what the live decision says.
NO_TARGET_WORDS = "no target: nothing decides a stop"


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


def stop_caution(order_check: OrderCheck | None, *, rare: bool = False) -> str:
    """What a stop rests on: the caution of ``order_check``, or the assumption without one."""
    return "; ".join(stop_caution_parts(order_check, rare=rare))


def stop_caution_parts(order_check: OrderCheck | None, *, rare: bool = False) -> list[str]:
    """``stop_caution`` in its parts: what every stop rests on, then (``rare``) what a
    rare-phenotype stop adds, so that a view printing both kinds of stop states each once."""
    plain = RANDOM_SAMPLE_ASSUMPTION if order_check is None else order_check.caution()
    if not rare:
        return [plain]
    if order_check is None:
        return [plain, f"it {RARE_ASSUMPTION}"]
    return [plain, order_check.caution(rare=True).removeprefix(f"{plain}; ")]


def _averaging_cv_note(thresholds: Sequence[float]) -> str:
    """An averaging CV is a standard error as a fraction of the mean, not a tolerance."""
    notes = [
        f"an averaging CV of {rate(t)}: the standard error is {rate(t)} of the mean, "
        f"not a ±{rate(t)} bound"
        for t in sorted(set(thresholds))
    ]
    return "; ".join(notes)


def _component_entry(
    component: Component, results: Sequence[StrategyResult], decided: list[Decision | None]
) -> str:
    """One target of the view: its combined decision, and which criteria decided it."""
    combined = combine_decisions(decided)
    rules = count(len(decided), "rule")
    deciding = [
        r
        for r in results
        if r.component_decisions.get(component.value, Decision.UNSUPPORTED)
        is not Decision.UNSUPPORTED
    ]
    if combined is Decision.STOP:
        chosen = deciding
    else:
        chosen = [
            r for r in deciding if r.component_decisions[component.value] is not Decision.STOP
        ]
    named = [
        f"{r.strategy} {' and '.join(r.criteria(component))}"
        for r in chosen
        if r.criteria(component)
    ]
    word = COMPONENT_WORDS[component]
    if not named:
        return f"{word}: {_word(combined)} ({rules})"
    if combined is Decision.STOP:
        thresholds = [
            float(policy["parameters"]["threshold"])
            for r in chosen
            for policy in r.configuration.get("policies", [])
            if str(policy.get("parameters", {}).get("quantity", "")).endswith("averaging_cv")
            and "threshold" in policy.get("parameters", {})
        ]
        note = f"; {_averaging_cv_note(thresholds)}" if thresholds else ""
        return f"{word}: stop ({rules}, met: {', '.join(named)}{note})"
    return f"{word}: {_word(combined)} ({rules}; not yet met: {', '.join(named)})"


def decision_view(results: Sequence[StrategyResult], order_check: OrderCheck | None = None) -> str:
    """One view of several rules on the same acquisition state: each target, then the overall.

    For each objective component, the decision combined over the rules that
    decide it (``combine_decisions``), with the number of rules and the
    criteria that decided it: at a stop every rule's threshold (an averaging
    CV is named as the standard error it is, not a tolerance on the mean),
    otherwise the thresholds not yet met. A required component that no rule
    decides is said so. The overall decision is the objective's:
    ``unsupported`` while a required component has no rule, otherwise the
    combination of the required components; an overall stop is followed by
    what it rests on (``stop_caution``: the caution of ``order_check``, or the
    assumption that the fields are a random sample of the well's positions).
    The results must share one objective (one session, one acquisition
    state); the view answers "if the targets disagree, is that obvious?"
    (``docs/live/live_session.md`` O4). The live view prints its entries one per line
    (``_target_decisions``).
    """
    return "; ".join(entry.text for entry in _target_decisions(results, order_check))


@dataclass(frozen=True)
class _TargetDecision:
    """One entry of ``decision_view``: a target (``None`` for the overall), its decision
    (``UNSUPPORTED`` when no rule decides a required target) and its text."""

    component: Component | None
    decision: Decision | None
    text: str


def _target_decisions(
    results: Sequence[StrategyResult], order_check: OrderCheck | None = None
) -> list[_TargetDecision]:
    """The entries of ``decision_view``: each target the rules decide or the objective
    requires, then the overall (shared-private: the live view reads them)."""
    results = list(results)
    if not results:
        raise ValueError("decision_view needs at least one result")
    objectives = {r.objective for r in results}
    if len(objectives) != 1:
        raise ValueError("decision_view needs results of one objective")
    objective = results[0].objective
    required = objective.required_components
    entries, overall = [], []
    for component in Component:
        decided = [
            r.component_decisions[component.value]
            for r in results
            if r.component_decisions.get(component.value, Decision.UNSUPPORTED)
            is not Decision.UNSUPPORTED
        ]
        if decided:
            text = _component_entry(component, results, decided)
            entries.append(_TargetDecision(component, combine_decisions(decided), text))
            if component in required:
                overall.append(combine_decisions(decided))
        elif component in required:
            text = f"{COMPONENT_WORDS[component]}: no rule decides it"
            entries.append(_TargetDecision(component, Decision.UNSUPPORTED, text))
            overall.append(Decision.UNSUPPORTED)
    if not required:
        combined: Decision | None = Decision.UNSUPPORTED
        final = NO_TARGET_WORDS
    elif Decision.UNSUPPORTED in overall:
        missing = [
            COMPONENT_WORDS[e.component].lower()
            for e in entries
            if e.decision is Decision.UNSUPPORTED and e.component is not None
        ]
        combined = Decision.UNSUPPORTED
        final = f"unsupported (no rule decides the {', '.join(missing)})"
    else:
        combined = combine_decisions(overall)
        final = _word(combined)
        if combined is Decision.STOP:
            # The overall stop rests on the rare phenotype only when the objective requires
            # it; a rare rule shown beside another objective keeps its caveat out of it.
            rare = Component.RARE in required
            final += f"; {stop_caution(order_check, rare=rare)}"
    text = f"Overall ({objective.objective.value} objective): {final}"
    return [*entries, _TargetDecision(None, combined, text)]


__all__ = [
    "COMPONENT_WORDS",
    "NO_TARGET_WORDS",
    "Decision",
    "StrategyResult",
    "Timing",
    "combine_decisions",
    "decision_view",
    "stop_caution",
    "stop_caution_parts",
]
