"""The guided session's control questions (private; ``guided.guide``; owner, 2026-10-05).

A gate first: "Are all your conditions compared with the same control?". Yes asks for the one
control as before (``_guided_ask.ask_control``). No asks for several controls, each with a
proposal to confirm: which conditions are controls (named like one: ``siCTRL``,
``VEC_SiNC``); the conditions of each (those sharing its name's stem), with treatment keys
proposed by stripping the shared stem (``THBS5_SiCol6a1`` under ``THBS5_SiNC`` is
``SiCol6a1``; no shared stem keeps the full name); a parent when a control's name extends
another condition's (``VEC_SiNC`` under ``VEC``, the key ``transfection`` for an siRNA or
shRNA suffix, else the suffix); and an optional master control, the control imaged in the
most biological repeats proposed (ties to the first declared). After the conditions of each
control, a condition still compared with nothing is asked for its control, none by default,
so a condition not named like a control can become one (``VEC`` for ``THBS5``; owner,
2026-10-05). The gate proposes No when
the likeliest control is missing from biological repeats where other conditions are
imaged, and then shows the condition by repeat table of images first. Then the contrast
table the answers give is shown, one test per line. Then, when sets of conditions could be
separate experiments, whether they are (``ask_experiment``; audit 10, owner, 2026-10-06):
with several controls, whether each control's conditions are an experiment of their own;
with one control whose conditions fall into sets imaged in separate biological repeats,
whether they are one screen, corrected together. No, the default, keeps the families the
design gives.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .._control import NO_CONTROL
from .._style import table
from .._wording import listing
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit._experiments import experiments
from ._guided_ask import Ask, Say, ask_control, choose, choose_many, text, yes_no

# The last part of a name that reads like a negative control (VEC_SiNC, x_siCTRL, y_scr).
_CONTROL_PART = re.compile(
    r"^(si|sh)?(nc|ctrl|control|scr|scramble|scrambled|nt|neg|negative|mock)$", re.IGNORECASE
)
_KIND_WORDS = {
    "own_control": "against its own control, the main verdict",
    "across_normalised": "across experiments, each normalised to its own control",
    "vs_master_raw": "against the master control, on raw values",
}
NONE = "none"


def _last(name: Any) -> str:
    return str(name).rsplit("_", 1)[-1]


def proposed_controls(scan: Any) -> list[Any]:
    """The conditions named like a control: the scan's guesses, or a last name part such as
    ``SiNC``."""
    guesses = set(scan.design.get("control_guesses") or [])
    return [c for c in scan.design["conditions"] if c in guesses or _CONTROL_PART.match(_last(c))]


def _stem(name: Any) -> str:
    """``VEC_`` for ``VEC_SiNC``; the name and ``_`` when it has no part."""
    text_ = str(name)
    return text_.rsplit("_", 1)[0] + "_" if "_" in text_ else text_ + "_"


def treatment_key(control: Any, member: Any) -> str:
    """The member's name without the stem it shares with its control (``SiCol6a1``), the
    full name when they share none."""
    stem, name = _stem(control), str(member)
    return name[len(stem) :] if name.startswith(stem) and len(name) > len(stem) else name


def image_table(scan: Any) -> str:
    """The fields imaged per condition and biological repeat, as a table."""
    fields = scan.fovs
    if CONDITION_COLUMN not in fields or REPEAT_COLUMN not in fields:
        return ""
    counts = pd.crosstab(fields[CONDITION_COLUMN], fields[REPEAT_COLUMN])
    counts = counts.reindex(index=scan.design["conditions"])
    rows = [[c, *(int(n) if n else "–" for n in counts.loc[c])] for c in counts.index]
    return table(rows, ["images", *map(str, counts.columns)], indent=2)


def _missing_repeats(scan: Any, condition: Any) -> list[Any]:
    """The repeats where other conditions are imaged and ``condition`` is not."""
    fields = scan.fovs
    if REPEAT_COLUMN not in fields or CONDITION_COLUMN not in fields:
        return []
    held = set(fields.loc[fields[CONDITION_COLUMN] == condition, REPEAT_COLUMN])
    return [r for r in pd.unique(fields[REPEAT_COLUMN]) if r not in held]


def ask_gate(ask: Ask, say: Say, scan: Any) -> bool:
    """Whether every condition is compared with the same control; No is proposed (after the
    table of images) when the likeliest control is missing from some repeats."""
    guess = scan.design.get("control_guess")
    likely = guess if guess is not None else next(iter(proposed_controls(scan)), None)
    missing = _missing_repeats(scan, likely) if likely is not None else []
    if missing:
        say(f"{likely} is not imaged in every biological repeat. The images per condition:")
        say(image_table(scan))
    return yes_no(ask, say, "Are all your conditions compared with the same control?", not missing)


def _keys(ask: Ask, say: Say, control: Any, members: list[Any]) -> dict[Any, Any]:
    keys = [treatment_key(control, m) for m in members]
    if len(set(keys)) < len(keys):
        keys = [str(m) for m in members]
    say("  Treatment keys: " + "; ".join(f"{k}: {m}" for k, m in zip(keys, members, strict=True)))
    if not yes_no(ask, say, f"Use these keys for {control}?", True):
        keys = [text(ask, f"Treatment key of {m}:", k) for k, m in zip(keys, members, strict=True)]
    return dict(zip(keys, members, strict=True))


def _parent(ask: Ask, say: Say, control: Any, conditions: list[Any]) -> Any:
    """The parent proposed when the control's name extends another condition's, confirmed;
    ``None`` without."""
    stems = [c for c in conditions if c != control and str(control).startswith(f"{c}_")]
    if not stems:
        return None
    parent = max(stems, key=lambda c: len(str(c)))
    suffix = str(control)[len(str(parent)) + 1 :]
    key = "transfection" if re.match(r"^(si|sh)", suffix, re.IGNORECASE) else suffix
    if not yes_no(ask, say, f"Is {control} compared with {parent}, as its parent?", True):
        return None
    return (parent, text(ask, f"Treatment key of {control} under {parent}:", key))


def _unlinked(
    ask: Ask,
    say: Say,
    conditions: list[Any],
    controls: list[Any],
    found: dict[Any, dict],
    assigned: set[Any],
) -> list[Any]:
    """Each condition not yet compared with anything asked for its control (default none):
    the one named becomes a control if it was not (``VEC`` for ``THBS5``). The controls, in
    the declared order, the new ones last."""
    for condition in conditions:
        if condition in assigned:
            continue
        options = [c for c in conditions if c != condition]
        prompt = f"Which control should {condition} be compared with?"
        chosen = choose(ask, say, prompt, [*options, NONE], NONE)
        if chosen == NONE:
            continue
        members = found.setdefault(chosen, {})
        key = treatment_key(chosen, condition)
        members[str(condition) if key in members else key] = condition
        assigned.update((condition, chosen))
        controls += [] if chosen in controls else [chosen]
    return controls


def most_imaged(scan: Any, controls: list[Any]) -> Any:
    """The control imaged in the most biological repeats, ties to the first declared: the
    master proposed (owner, 2026-10-05)."""
    fields = scan.fovs
    if REPEAT_COLUMN not in fields or CONDITION_COLUMN not in fields:
        return controls[0]
    held = fields.groupby(CONDITION_COLUMN)[REPEAT_COLUMN].nunique()
    return max(controls, key=lambda c: int(held.get(c, 0)))


def ask_several(ask: Ask, say: Say, scan: Any) -> tuple[dict, dict | None, Any]:
    """Several controls: ``(control, control_parents, master_control)`` (see the module)."""
    from .setup import Setup  # setup imports the workflows that import this module

    conditions = list(scan.design["conditions"])
    while True:
        proposal = proposed_controls(scan) or None
        # No default of every condition: with nothing under any control, nothing is compared.
        prompt = "Which conditions are controls?"
        controls = choose_many(ask, say, prompt, conditions, proposal, enter_all=False)
        found: dict[Any, dict] = {}
        assigned = set(controls)
        named = set(proposal or [])
        # The controls named like one first, the longest stem first: VEC_SiNC takes the
        # VEC_ conditions before VEC is asked for its own.
        order = sorted(controls, key=lambda c: (c not in named, -len(_stem(c))))
        for control in order:
            left = [c for c in conditions if c not in assigned]
            stem = _stem(control)
            near = [c for c in left if str(c).startswith(stem)] or [NONE]
            prompt = f"Which conditions belong to {control}?"
            members = [m for m in choose_many(ask, say, prompt, [*left, NONE], near) if m != NONE]
            assigned.update(members)
            found[control] = _keys(ask, say, control, members) if members else {}
        controls = _unlinked(ask, say, conditions, list(controls), found, assigned)
        spec = {control: found[control] for control in controls}
        parents = {}
        for control in controls:
            found = _parent(ask, say, control, conditions)
            if found is not None:
                parents[control] = found
        prompt = "Which control are the others compared with on raw values (the master)?"
        master = choose(ask, say, prompt, [*controls, NONE], most_imaged(scan, controls))
        declared = {
            "control": {CONDITION_COLUMN: spec},
            "control_parents": parents or None,
            "master_control": None if master == NONE else master,
        }
        try:
            Setup(readout="_", **declared)
        except (TypeError, ValueError) as error:  # fallback: asked again, with what was wrong
            say(f"  These controls cannot be used ({error}). Please answer again.")
            continue
        return declared["control"], declared["control_parents"], declared["master_control"]


def ask_controls(ask: Ask, say: Say, scan: Any) -> tuple[dict | None, dict | None, Any]:
    """The gate, then one control or several: ``(control, control_parents, master)``."""
    if not scan.design["conditions"] or ask_gate(ask, say, scan):
        return ask_control(ask, say, scan), None, None
    return ask_several(ask, say, scan)


def _sets(scan: Any, control: Any) -> list[list[Any]]:
    """The conditions compared with ``control``, in sets linked by the biological repeats
    they share with it (``_experiments.experiments``)."""
    fields = scan.fovs
    if REPEAT_COLUMN not in fields or CONDITION_COLUMN not in fields:
        return []
    repeats = fields.groupby(CONDITION_COLUMN)[REPEAT_COLUMN].agg(set).to_dict()
    own = repeats.get(control, set())
    held = {c: r & own for c, r in repeats.items() if c != control and r & own}
    numbers = experiments(held)
    return [[c for c in held if numbers[c] == n] for n in sorted(set(numbers.values()))]


def ask_experiment(ask: Ask, say: Say, scan: Any, control: Any) -> dict[Any, Any] | None:
    """Whether sets of conditions are separate experiments (see the module): the mapping
    condition -> experiment ``Setup(experiment=...)`` takes, or ``None`` (the default)."""
    spec = (control or {}).get(CONDITION_COLUMN)
    if isinstance(spec, dict):  # several controls
        members = {c: list(m.values() if isinstance(m, dict) else m) for c, m in spec.items()}
        if sum(bool(m) for m in members.values()) < 2:
            return None
        prompt = (
            "Are the conditions of each control a separate experiment, corrected on its own? "
            "No: corrected together wherever they share biological repeats."
        )
        if not yes_no(ask, say, prompt, default=False):
            return None
        return {m: str(c) for c, ms in members.items() for m in ms}
    sets = _sets(scan, spec) if spec is not None else []
    if len(sets) < 2:
        return None
    named = listing([listing([str(c) for c in s], 3) for s in sets], 4)
    say(f"The conditions fall into {len(sets)} sets imaged in separate repeats: {named}.")
    prompt = (
        f"Are they one experiment against {spec}, a screen corrected together? No: each set "
        "corrected on its own, the screen-wide q beside each verdict."
    )
    if not yes_no(ask, say, prompt, default=False):
        return None
    return {c: "screen" for s in sets for c in s}


def contrast_lines(control: Any, parents: Any, master: Any) -> list[str]:
    """The tests these answers make, one per line, with their family."""
    from .setup import Setup

    if control is None:
        return [f"{NO_CONTROL[:1].upper()}{NO_CONTROL[1:]}."]
    setup = Setup(readout="_", control=control, control_parents=parents, master_control=master)
    found = setup.contrast_table
    if found is None:
        name = setup.control[CONDITION_COLUMN]
        return [f"Each condition against {name}, within each biological repeat."]
    return [
        f"  {row['label']}: {_KIND_WORDS[row['kind']]}"
        + (", corrected within its sub-experiment" if row["kind"] == "own_control" else "")
        + ("" if row["kind"] == "own_control" else f", corrected in the family {row['family']}")
        for row in found.to_dict("records")
    ]


__all__ = [
    "ask_controls",
    "ask_experiment",
    "ask_gate",
    "ask_several",
    "contrast_lines",
    "image_table",
    "most_imaged",
    "proposed_controls",
    "treatment_key",
]
