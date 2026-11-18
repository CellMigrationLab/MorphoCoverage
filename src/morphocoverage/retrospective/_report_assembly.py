"""The steps of ``microscopist_report`` (private): each answer's text, the notes, the gate of
the representation control, the sections and every answer's status.

Split from ``report`` to keep that module under 600 lines; the report is described there.
Shared-private: used by ``retrospective.report``. ``given`` is every result passed to
``microscopist_report``, by its argument name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._caveats import assumption, caveat_lines, pool_assumptions
from .._style import PROBLEM, Status, indented, wrap
from .._wording import listing
from .controls import ControlResult
from .controls_text import representation_gate
from .inspection_text import inspection_parts
from .order_text import order_parts
from .report_answers import conditions_sections, short_section
from .report_enough import enough_full, enough_short
from .report_needs import _NEEDS, _SHORT_NEEDS
from .report_plan import bought_short, effort_short, next_time_short
from .report_profile import sampling_profile
from .report_sampling import (
    controls_full,
    controls_short,
    profile_full,
    profile_short,
    unit_full,
    unit_short,
)
from .report_status import caveat_status, question_statuses, weak_status

# Said only where a definition of new ran (audit 10, C24).
_NO_CONTROL_NOTE = (
    "No representation control: a 'nothing new' answer says nothing about phenotypes these "
    "features and this definition of new cannot see."
)


@dataclass(frozen=True)
class _Gate:
    """The answers the representation control's sentence was put before (``weakened``), and
    whether the representation is weak (else the small-state check was not assessed)."""

    weakened: list[str]
    weak: bool


def _foundation_parts(
    given: Mapping[str, Any], parts: dict[str, list[str]], short: dict[str, str]
) -> None:
    """Add the Foundation's, question 1's and question 2's texts to ``parts`` and ``short``."""
    variability, object_rules = given["variability"], given["object_rules"]
    descriptor_spread, controls = given["descriptor_spread"], given["controls"]
    sufficiency, calibration, contrast = (
        given["sufficiency"],
        given["calibration"],
        given["contrast"],
    )
    for check in (variability, object_rules, descriptor_spread):
        if check is not None:
            parts["unit"].append(check.summary(detail="full"))
    if parts["unit"]:
        short["unit"] = unit_short(variability, object_rules, descriptor_spread)
    if controls is not None:
        parts["controls"].append(controls.summary(detail="full"))
        short["controls"] = controls_short(controls)
    parts["enough"] += enough_full(sufficiency, calibration, controls, contrast)
    if parts["enough"]:
        short["enough"] = enough_short(sufficiency, calibration, controls, contrast=contrast)


def _planning_parts(
    given: Mapping[str, Any], parts: dict[str, list[str]], short: dict[str, str]
) -> None:
    """Add the texts of questions 4 to 8 to ``parts`` and ``short``."""
    adequacy, reliability, protocol = given["adequacy"], given["reliability"], given["protocol"]
    inspection, effort, efficiency = given["inspection"], given["effort"], given["efficiency"]
    if adequacy is not None:
        parts["bought"].append(adequacy.summary(detail="full"))
    if reliability is not None:
        parts["next_time"].append(reliability.summary(detail="full"))
    if protocol is not None:
        parts["next_time"].append(protocol.summary(detail="full"))
    if inspection is not None:
        parts["inspect"].append(inspection.summary(detail="full"))
        short["inspect"] = short_section([inspection_parts(inspection, indent=2)])
    if effort is not None:
        parts["effort"].append(effort.summary(detail="full"))
    for key, text in (
        ("next_time", next_time_short(reliability, protocol)),
        ("bought", bought_short(adequacy)),
        ("effort", effort_short(effort)),
    ):
        if text:
            short[key] = text
    if efficiency is not None:
        parts["order"].append(efficiency.summary(detail="full"))


def _answer_parts(
    given: Mapping[str, Any], keys: list[str]
) -> tuple[dict[str, list[str]], dict[str, str], pd.DataFrame]:
    """The full texts of every question (``keys``, in order) and the short forms there are,
    and the sampling profile."""
    parts: dict[str, list[str]] = {key: [] for key in keys}
    short: dict[str, str] = {}
    _foundation_parts(given, parts, short)
    _planning_parts(given, parts, short)
    unusual, clustering = given["unusual"], given["clustering"]
    profile = sampling_profile(
        given["controls"],
        given["effort"],
        None if given["redundancy"] is None else given["redundancy"].table,
        given["inspection"],
        given["variability"],
        clustering,
    )
    if len(profile) or unusual is not None or clustering is not None:
        parts["profile"].append("profile")  # its text, wrapped at its indent, is set below
    parts["conditions"], conditions = conditions_sections(
        given["contrast"], given["ranking"], given["discovery"]
    )
    if conditions:
        short["conditions"] = conditions
    return parts, short, profile


def _first_notes(given: Mapping[str, Any]) -> tuple[list[str], bool]:
    """The notes every answer is read with, and whether a definition of new ran."""
    sufficiency = given["sufficiency"]
    # Only what was stated: no tolerance or definition of new where none ran (C24, B21).
    novel = sufficiency is not None and len(sufficiency.next_fov) > 0
    stated = ["features"]
    stated += ["tolerances"] if sufficiency is not None and len(sufficiency.enough) else []
    stated += ["definition of new"] if novel else []
    notes = [
        assumption(f"Every answer holds for the stated {listing(stated)}."),
        assumption(
            "Replays, draws and hold-outs describe these samples; they are not independent "
            "experiments. Statements across samples count samples or repeats."
        ),
        assumption(
            "A field count that preserved this acquisition is not a protocol until it holds on "
            "samples it was not chosen on."
        ),
    ]
    return notes, novel


def _gate(
    controls: ControlResult | None,
    minimum_detection: float,
    novel: bool,
    parts: dict[str, list[str]],
    short: dict[str, str],
    notes: list[str],
) -> _Gate:
    """Put the representation control's sentence before the adequacy answers (in ``parts``
    and ``short``) and in ``notes`` when it gates them; without a control, the note that
    there is none where a definition of new ran."""
    weakened: list[str] = []
    if controls is None or controls.status == "not assessed":
        notes += [assumption(_NO_CONTROL_NOTE)] if novel else []
        return _Gate(weakened, False)
    # With planted objects declared, the rare-state rate gates the warning (P6-4); none
    # planted, the gate is not assessed, never read from the whole fields (#270).
    gate, weak = representation_gate(controls, float(minimum_detection))
    after = ", so the answers below cannot exclude a phenotype like it." if weak else "."
    marker = PROBLEM if weak else Status.NOT_ASSESSED.marker
    for key in ("enough", "next_time", "bought") if gate else ():
        if parts[key]:
            weakened.append(key)
            parts[key].insert(0, f"{gate}{after}")
            if key in short:
                short[key] = f"{wrap(f'{marker} {gate}{after}', 2, hang=5)}\n{short[key]}"
    if gate:
        # The notes close the report: the adequacy answers are above them.
        notes.append(
            f"{gate}, so the adequacy answers (questions 2, 4 and 5) "
            + ("cannot exclude a phenotype like it." if weak else "are not checked for it.")
        )
    return _Gate(weakened, weak)


def _closing_notes(
    given: Mapping[str, Any],
    parts: dict[str, list[str]],
    short: dict[str, str],
    notes: list[str],
) -> str | None:
    """Add the drift note to ``notes``, and the sampling-frame caveat where it belongs: with
    question 8, in ``notes``, or returned to close question 3 (else None)."""
    # Redundancy alone feeds the profile; question 8 is answered by an order replay.
    frame = assumption(
        "Whether these were the right positions needs every candidate position from an "
        "overview (not available: sampling-frame bias is not assessed)."
    )
    notes.append(
        assumption(
            "Technical drift between repeats, batches or plates is not assessed: groups are "
            "analysed apart, and a shift shared by all fields of a group is not new within it."
        )
    )
    efficiency, redundancy = given["efficiency"], given["redundancy"]
    if efficiency is not None:
        parts["order"].append("\n".join(caveat_lines([frame], 0, detail="full")))
        short["order"] = short_section([order_parts(efficiency, indent=2)], frame)
    elif redundancy is not None and parts["profile"]:
        return frame  # it closes question 3 (profile_full, profile_short)
    elif redundancy is not None:
        # A caveat alone does not assess question 3: it stays "not assessed", and
        # the caveat is kept with the notes.
        notes.append(frame)
    return None


def _sections(
    given: Mapping[str, Any],
    parts: dict[str, list[str]],
    short: dict[str, str],
    profile: pd.DataFrame,
    framed: str | None,
) -> dict[str, str]:
    """Every question's full text (``Not assessed`` with what it needs when it has none),
    after adding question 3's short form to ``short``."""
    variability, unusual, clustering = given["variability"], given["unusual"], given["clustering"]
    if parts["profile"]:
        short["profile"] = profile_short(
            profile,
            variability=variability,
            effort=given["effort"],
            unusual=unusual,
            clustering=clustering,
            frame=framed,
        )
    sections = {
        key: indented("\n\n".join(texts)) if texts else f"  Not assessed: {_NEEDS[key]}."
        for key, texts in parts.items()
    }
    # The Foundation, 1 and 3 are wrapped at the indent they are printed at.
    if parts["unit"]:
        sections["unit"] = unit_full(variability, given["object_rules"], given["descriptor_spread"])
    if given["controls"] is not None:
        sections["controls"] = controls_full(given["controls"])
    if parts["profile"]:
        sections["profile"] = profile_full(profile, unusual, clustering, framed)
    return sections


def _statuses(
    results: Mapping[str, Any],
    profile: pd.DataFrame,
    minimum_detection: float,
    parts: dict[str, list[str]],
    sections: dict[str, str],
    short: dict[str, str],
    gate: _Gate,
) -> tuple[dict[str, Status], dict[str, str], dict[str, str], dict[str, str]]:
    """Every question's status and headline, with the sections and short forms each with
    one "Assumptions:" block."""
    # One "Assumptions:" block per section, after its answers; a caution raises its status.
    sections = {key: pool_assumptions(text) for key, text in sections.items()}
    short = {key: pool_assumptions(text) for key, text in short.items()}
    answers = question_statuses(results, profile, minimum_detection)
    for key, (status, glance) in answers.items():
        answers[key] = caveat_status(status, short.get(key, sections[key]), key), glance
    statuses = {key: answers[key][0] for key in parts}
    headlines = {key: answers[key][1] for key in parts}
    for key, texts in parts.items():
        if not texts:
            statuses[key] = Status.NOT_ASSESSED
            headlines[key] = f"not assessed; {_SHORT_NEEDS[key]}"
    for key in gate.weakened:
        gated = (statuses[key], f"small-state representation not assessed; {headlines[key]}")
        statuses[key], headlines[key] = weak_status(answers[key]) if gate.weak else gated
    return statuses, headlines, sections, short


__all__ = [
    "_answer_parts",
    "_closing_notes",
    "_first_notes",
    "_gate",
    "_sections",
    "_statuses",
]
