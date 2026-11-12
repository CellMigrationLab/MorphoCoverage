"""The guided session reads a MorphoFeature export's experiment description (private; #631).

``guide(description=...)`` takes the description of the export
(``morphocoverage.described.read_experiment``; the scan reads it, ``workflows._described``):

- when it gives a default control (its references compare at least one condition that has a
  field with its control), the control questions are skipped (``ask_any_controls``): the
  ``Setup`` takes the description's references (``described_setup``; a chain is compared as
  a chain), and the session says the tests they make (``described_lines``). The
  positive-control and experiment questions read those controls (``question_control``).
  Without a default (no reference, or an export without the block), the control questions
  are asked as before;
- its target suggestions are shown, as written, and the user is asked whether to use them as
  suggested, edit them or leave them (``ask_targets``). Enter leaves them: a suggestion is
  never used without the user's answer (design note, section 4, item 6), and only the
  grouping the user confirms is saved and read (``Setup(targets=...)``).

Shared-private: used by ``guided``.
"""

from __future__ import annotations

from typing import Any

from .._wording import listing
from ..described import DescribedExperiment
from ..metadata import CONDITION_COLUMN
from ._guided_ask import Ask, Say, choose, text
from ._guided_controls import NONE, ask_controls, contrast_lines

CONFIRM = "use them as suggested"
EDIT = "edit them"
LEAVE = "leave them unconfirmed (not used)"


def described_default(scan: Any) -> DescribedExperiment | None:
    """The scan's description when it gives a default control, else ``None``."""
    described = scan.design.get("described")
    if described is None or described.block is None or not described.pairs:
        return None
    return described


def described_lines(described: DescribedExperiment) -> list[str]:
    """The controls the description gives and the tests they make, one per line."""
    sha = described.block.description_sha256[:12]
    tests = [
        f"  {c} against {r}, its control in the description, within each biological repeat."
        for c, r in described.pairs
    ]
    head = (
        f"The controls come from the experiment description (description {sha}), so they are "
        "not asked. The tests that will run:"
    )
    return [head, *tests, *(f"  {note}" for note in described.notes)]


def question_control(described: DescribedExperiment) -> dict[str, Any]:
    """The description's controls as the positive-control and experiment questions read them:
    ``{"condition": X}`` for one control, else each control with no conditions listed."""
    default = described.default_control()
    if "control" in default:
        return default["control"]
    controls = dict.fromkeys(r for _, r in described.pairs)
    return {CONDITION_COLUMN: {control: {} for control in controls}}


def ask_any_controls(ask: Ask, say: Say, scan: Any) -> tuple[Any, Any, Any, Any]:
    """``(control, control_parents, master_control, controls)``: the description's (``None``
    three times, its references being the default) when it gives them, else asked
    (``ask_controls``); ``controls`` is what the positive-control and experiment questions
    read. An export without the block defaults nothing: its controls are asked until one is
    declared (``MISSING``: "no control" would be refused when the ``Setup`` is made)."""
    described = described_default(scan)
    if described is None:
        missing = getattr(scan.design.get("described"), "missing", None)
        if missing is not None:
            say(f"No control is defaulted: {missing}.")
        control, parents, master = ask_controls(ask, say, scan)
        while missing is not None and control is None:
            say("This export needs its controls declared by their IDs; please choose one.")
            control, parents, master = ask_controls(ask, say, scan)
        say("The tests that will run:\n" + "\n".join(contrast_lines(control, parents, master)))
        return control, parents, master, control
    say("\n".join(described_lines(described)))
    return None, None, None, question_control(described)


def ask_targets(ask: Ask, say: Say, scan: Any) -> dict[str, str] | None:
    """The reagent grouping the user confirms from the description's target suggestions
    (see the module), ``None`` when there are none or the user leaves them."""
    described = scan.design.get("described")
    suggestions = None if described is None else described.target_suggestions
    if not suggestions:
        return None
    say(
        "The experiment description suggests targets, as written (descriptive only, compared "
        "exactly, case included):\n" + "\n".join(f"  {line}" for line in suggestions.lines())
    )
    prompt = "Group the reagents of one target with them?"
    choice = choose(ask, say, prompt, [CONFIRM, EDIT, LEAVE], LEAVE)
    if choice == LEAVE:
        return None
    edits: dict[str, str | None] = {}
    if choice == EDIT:
        for condition, target in suggestions.suggested.items():
            answer = text(ask, f"Target of {condition} ({NONE!r} leaves it out):", target)
            edits[condition] = None if answer == NONE else answer
    confirmed = suggestions.confirm(edits)
    if confirmed:
        named = listing([f"{c}: {t}" for c, t in confirmed.items()], 8)
        say(f"Targets confirmed: {named}.")
    return confirmed or None


__all__ = [
    "ask_any_controls",
    "ask_targets",
    "described_default",
    "described_lines",
    "question_control",
]
