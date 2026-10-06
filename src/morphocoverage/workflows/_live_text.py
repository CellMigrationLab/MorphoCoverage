"""The text of one live update (private; ``LiveUpdate.summary`` uses it).

It is read during acquisition, one block per field, so the short form (the
default) is compact:

- a headline: the field's number in its group, the group
  (``MOCK · 1A · B6``) and the decision. ``✅ stop`` when the objective's
  targets are met; ``⚠️ stop`` when they are but the acquisition-order check
  flags the fields; ``⏳ continue`` while a rule a target needs says not yet
  (the ⏳ marker is the live view's own); ``➖`` while a declared target has no
  rule, or a decision is undefined;
- the field's counts: its observations, the group's so far, the read time;
- one table of the rules: each one's own verdict, its evidence now and its
  threshold; a quantity not yet computable says why below it;
- when more than one target is shown, one line with each target's decision;
- unless the objective says continue (no stop is acted on yet), when a rule
  says stop: what the stop rests on, each caveat once (``_caveats``). A flag
  of the order check, or no check run, is a ⚠️ caution about these fields;
  the standing assumptions (the fields as a random sample of the well's
  positions when the check found nothing, where a rare-phenotype stop has
  looked, the rules that treat observations as independent draws, what an
  averaging CV is at an overall stop of the population summary) are listed
  under "Assumptions:". The full form states them beside every stop, as
  ``explain`` does.

The full form keeps every line the view printed before: the field's
metadata, every rule's ``StrategyResult.explain`` line and every entry of
``decision_view``, one per line with its marker.

The text is a pure function of the update (its results, order check and
counts): a replay of the same fields gives the same results
(``tests/test_guards.py``), hence the same text but for the read time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._caveats import assumption, caveat_lines
from .._style import (
    CAUTION,
    DASH,
    NOT_ASSESSED,
    OK,
    SEPARATOR,
    check_detail,
    group_label,
    rule_label,
    table,
    wrap,
)
from .._wording import count, listing, plural
from ..evidence import Component
from ..order_check import order_caveat
from ..results import (
    COMPONENT_WORDS,
    NO_TARGET_WORDS,
    Decision,
    StrategyResult,
    _averaging_cv_note,
    _distinct,
    _target_decisions,
    stop_caution_parts,
)
from ._manifest import OBJECTS_FILE

if TYPE_CHECKING:
    from .live import LiveUpdate

WAITING = "⏳"
_QUANTITY_WORDS = {
    "averaging_cv": "averaging CV",
    "pilot_projected_averaging_cv": "projected averaging CV",
}
INDEPENDENT_DRAWS = "observations as independent draws: assumes a spatially uniform sample"


def mark(decision: Decision | None) -> str:
    """The live marker of a decision: ✅ stop, ⏳ continue, ➖ undefined or unsupported."""
    if decision is Decision.STOP:
        return OK
    return WAITING if decision is Decision.CONTINUE else NOT_ASSESSED


def _says(decision: Decision | None) -> str:
    """A rule's own verdict in a table cell: ``✅ enough``, ``⏳ not yet``, ``➖ undefined``."""
    words = {Decision.STOP: "enough", Decision.CONTINUE: "not yet"}
    return f"{mark(decision)} {words.get(decision, 'undefined')}"


def _label(result: StrategyResult, unit: str, detail: str = "short") -> str:
    """The rule in words (``_style.rule_label``), from what its result records."""
    row = {
        "strategy": result.strategy,
        "policies": list(result.configuration.get("policies", [])),
        "parameters": dict(result.parameters),
        "config_id": result.config_id,
    }
    return rule_label(row, unit, detail=detail)


def _effort_words(effort: str, unit: str) -> tuple[str, str]:
    """The singular and plural of an effort such as ``n_fovs`` (fields) or observations."""
    if effort == "n_fovs":
        return "field", "fields"
    if effort in ("n_observations", "n_objects"):
        return unit, plural(unit)
    words = effort.replace("_", " ")
    return words, words


def _evidence(result: StrategyResult, unit: str) -> tuple[str, str, list[str], list[str]]:
    """The rule's evidence now, its thresholds, what a fixed amount still needs, and the
    reasons a quantity is not yet computable: each policy of the rule once."""
    now, stop_at, needs, unavailable = [], [], [], []
    for policy in result.configuration.get("policies", []):
        parameters = policy.get("parameters", {})
        name = parameters.get("quantity")
        quantity = result.evidence.quantities.get(name) if name else None
        if quantity is not None and "threshold" in parameters:
            words = _QUANTITY_WORDS.get(name, name.replace("_", " "))
            better = "≤" if quantity.direction.value == "lower_is_better" else "≥"
            if quantity.value is None:
                now.append("not computable yet")
                stop_at.append(f"{better}{parameters['threshold']:g}")
                unavailable.append(f"{words} not computable yet ({quantity.reason})")
            else:
                value, threshold = _distinct(quantity.value, parameters["threshold"])
                now.append(f"{words} {value}")
                stop_at.append(f"{better}{threshold}")
        elif "effort" in parameters and "target" in parameters:
            one, many = _effort_words(parameters["effort"], unit)
            current = getattr(result, parameters["effort"], None)
            target = parameters["target"]
            now.append("unknown" if current is None else count(current, one, many))
            stop_at.append(f"≥{target:g}")
            if current is not None and current < target:
                needs.append(count(target - current, f"more {one}", f"more {many}"))
        else:
            now.append(policy.get("name", "policy"))
    return ", ".join(now) or DASH, ", ".join(stop_at) or DASH, needs, unavailable


def _needs(result: StrategyResult, target: Component) -> bool:
    """Whether the rule decides ``target`` and has not met it."""
    decision = result.component_decisions.get(target.value, Decision.UNSUPPORTED)
    return decision is not Decision.UNSUPPORTED and decision is not Decision.STOP


def _saying(k: int, n: int) -> str:
    """``all 3 rules say``, ``2 of 3 rules say``, ``the rule says``, ``no rule says``."""
    if n == 1:
        return "the rule says" if k else "the rule does not yet say"
    if k == n:
        return "both rules say" if n == 2 else f"all {n} rules say"
    return f"none of {n} rules say" if k == 0 else f"{k} of {n} rules say"


def headline(update: LiveUpdate) -> str:
    """``✅ field 8 · MOCK · 1A · B6 — stop: all 3 rules say enough`` (see the module)."""
    results = update.results
    if not results:
        return f"{NOT_ASSESSED} {_field(update)} {DASH} no rule was run"
    unit = update.observation_unit
    entries = _target_decisions(results, update.order_check)
    overall = entries[-1].decision
    required = results[0].objective.required_components
    stops = sum(r.rule_decision is Decision.STOP for r in results)
    saying = f"{_saying(stops, len(results))} enough"
    waiting = []
    for result in results:
        if any(_needs(result, c) for c in required):
            _, _, needs, _ = _evidence(result, unit)
            more = f" ({listing(needs)})" if needs else ""
            waiting.append(f"{_label(result, unit)}{more}")
    not_yet = f"; not yet: {'; '.join(waiting)}" if waiting else ""
    if overall is Decision.UNSUPPORTED and not required:
        verdict = f"not decided: {NO_TARGET_WORDS}; {saying}"
    elif overall is Decision.UNSUPPORTED:
        missing = [
            COMPONENT_WORDS[e.component].lower()
            for e in entries
            if e.decision is Decision.UNSUPPORTED and e.component is not None
        ]
        verdict = f"not decided: no rule decides the {listing(missing)}; {saying}{not_yet}"
    elif overall is Decision.STOP:
        if update.order_check is not None and update.order_check.flagged:
            verdict = f"stop: {saying}; but the order check flags these fields (below)"
            return wrap(f"{CAUTION} {_field(update)} {DASH} {verdict}", hang=3)
        verdict = f"stop: {saying}"
    elif overall is Decision.CONTINUE:
        verdict = f"continue: {saying}{not_yet}"
    else:
        undefined = [_label(r, unit) for r in results if r.rule_decision is None]
        verdict = f"undefined: {listing(undefined) or 'a rule'} cannot decide yet; {saying}"
    return wrap(f"{mark(overall)} {_field(update)} {DASH} {verdict}", hang=3)


def _field(update: LiveUpdate) -> str:
    """``field 8 · MOCK · 1A · B6``: the field's number in its group, and the group."""
    results = update.results
    if not results:
        return "field"
    number = "field" if results[0].n_fovs is None else f"field {results[0].n_fovs}"
    group = dict(results[0].group)
    return f"{number}{SEPARATOR}{group_label(group)}" if group else number


def _identity(update: LiveUpdate) -> str:
    """The field's identifier (its FOV column's value), or its metadata when unknown."""
    key = update.results[0].fov_key if update.results else ()
    if key and key[-1] in update.fov_metadata:
        return str(update.fov_metadata[key[-1]])
    return _where(update)


def _where(update: LiveUpdate) -> str:
    """Every metadata value of the field, as ``condition=MOCK, repeat=1A, ...``."""
    return ", ".join(f"{k}={v}" for k, v in update.fov_metadata.items() if k != OBJECTS_FILE)


def _read(update: LiveUpdate) -> str:
    objects = count(update.n_objects, update.observation_unit)
    return f"{objects} (read in {1000 * update.read_seconds:.0f} ms)"


def _pending(update: LiveUpdate) -> list[str]:
    """The observation rows still waiting for their FOV row, when there are any."""
    if not update.pending_objects:
        return []
    unit = update.observation_unit
    waiting = (
        f"1 {unit} row waits for its FOV row"
        if update.pending_objects == 1
        else f"{update.pending_objects} {unit} rows wait for their FOV rows"
    )
    return [f"  {waiting} (not yet counted in any field)"]


def _cautions(update: LiveUpdate) -> list[str]:
    """What the stops of this update rest on, each once (see the module)."""
    results = update.results
    unit = update.observation_unit
    stopped = [r for r in results if Decision.STOP in (r.decision, r.rule_decision)]
    entries = _target_decisions(results, update.order_check) if results else []
    overall = entries[-1].decision if entries else None
    # While the objective says continue, no stop is acted on: its caveats wait for it.
    if not stopped or overall is Decision.CONTINUE:
        return []
    overall_stop = overall is Decision.STOP
    rare = any(r.decides_rare for r in stopped) or (
        overall_stop
        and any(r.component_decisions.get(Component.RARE.value) is Decision.STOP for r in results)
    )
    # A flag, or no check run, is a caution about these fields; a check that ran and found
    # nothing leaves the standing random-sample assumption (order_check.order_caveat).
    cautions = stop_caution_parts(update.order_check, rare=rare)
    cautions[0] = order_caveat(cautions[0])
    if rare:
        cautions[-1] = assumption(f"the rare-phenotype stop: {cautions[-1]}")
    independent = [
        _label(r, unit)
        for r in stopped
        if r.diagnostics.get("assumes_spatial_uniformity")
        or r.diagnostics.get("assumes_random_object_sampling")
    ]
    if independent:
        verb = "treats" if len(independent) == 1 else "treat"
        cautions.append(assumption(f"{listing(independent)} {verb} {INDEPENDENT_DRAWS}"))
    population = next((e for e in entries if e.component is Component.POPULATION), None)
    if population is not None and population.decision is Decision.STOP:
        thresholds = [
            float(policy["parameters"]["threshold"])
            for r in results
            if r.component_decisions.get(Component.POPULATION.value) is Decision.STOP
            for policy in r.configuration.get("policies", [])
            if str(policy.get("parameters", {}).get("quantity", "")).endswith("averaging_cv")
            and "threshold" in policy.get("parameters", {})
        ]
        if thresholds:
            cautions.append(assumption(_averaging_cv_note(thresholds)))
    return caveat_lines(cautions, indent=2)


def _targets(update: LiveUpdate) -> list[str]:
    """One line with each target's decision, when the view shows more than one target."""
    entries = [e for e in _target_decisions(update.results) if e.component is not None]
    if len(entries) < 2:
        return []
    required = update.results[0].objective.required_components
    parts = []
    for entry in entries:
        words = {Decision.STOP: "stop", Decision.CONTINUE: "continue"}
        said = words.get(entry.decision, "undefined")
        if entry.decision is Decision.UNSUPPORTED:
            said = "no rule decides it"
        extra = "" if entry.component in required else " (not required)"
        target = COMPONENT_WORDS[entry.component].lower()
        parts.append(f"{target} {mark(entry.decision)} {said}{extra}")
    return [wrap(f"Targets: {SEPARATOR.join(parts)}", 2)]


def _short(update: LiveUpdate) -> str:
    lines = [headline(update)]
    results = update.results
    unit = update.observation_unit
    counts = f"{_identity(update)}: {_read(update)}"
    if results and results[0].n_observations is not None and results[0].n_fovs is not None:
        so_far = count(results[0].n_observations, unit, plural(unit))
        counts += f"; {so_far} in {count(results[0].n_fovs, 'field')} so far"
    lines.append(wrap(counts, 2))
    rows, notes = [], []
    for result in results:
        now, stop_at, _, unavailable = _evidence(result, unit)
        label = _label(result, unit)
        rows.append([label, _says(result.rule_decision), now, stop_at])
        notes += [f"{label}: {reason}" for reason in unavailable]
    if rows:
        lines.append(table(rows, ["rule", "says", "now", "stop at"], indent=4))
    lines += [wrap(f"{note}.", 4) for note in notes]
    lines += _targets(update)
    lines += _cautions(update)
    lines += _pending(update)
    return "\n".join(lines)


def _full(update: LiveUpdate) -> str:
    lines = [headline(update), wrap(f"{_where(update)}: {_read(update)}", 2, hang=4)]
    unit = update.observation_unit
    for result in update.results:
        own = f"{mark(result.rule_decision)} {_label(result, unit, 'full')}:"
        lines.append(wrap(own, 2, hang=5))
        lines.append(wrap(result.explain(update.order_check), 5))
    if update.results:
        lines.append("  Targets and the overall decision:")
        for entry in _target_decisions(update.results, update.order_check):
            lines.append(wrap(f"{mark(entry.decision)} {entry.text}", 4, hang=7))
    lines += _pending(update)
    return "\n".join(lines)


def live_summary(update: LiveUpdate, detail: str = "short") -> str:
    """The update's text: ``"short"`` (the default) or ``"full"`` (see the module)."""
    check_detail(detail)
    return _short(update) if detail == "short" else _full(update)


__all__: list[str] = []
