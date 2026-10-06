"""The microscopist's report: every answer in one document, in the order of the questions.

``microscopist_report`` takes the results a user has computed
(each analysis stays a separate, inspectable call) and orders their plain
sentences by the questions of ``docs/report/questions.md``. It opens with the
foundation every other answer rests on, then answers the questions in order:

- Foundation: why count fields, and not only objects? (``sampling_unit``:
  ``fov_variability``, ``object_rule_check``, ``descriptor_spread_check``)
  How much do the acquired fields differ, and would an object-count rule have
  held when its objects came as whole fields?

1. Can these features detect what I care about? (``representation_control``)
2. Did I image enough? (``sufficiency_report``, ``stopping_calibration``)
3. What is limiting my sampling? (the sampling profile, below, and
   ``sampling_unit.fov_clustering`` of rare or unusual objects)
4. How many fields next time? (``sampling_reliability``, ``protocol_reliability``)
5. What did the extra fields buy? (``adequacy_points``)
6. Which fields should I inspect? (``inspect_fovs``)
7. More fields or more samples? (``effort_projection``)
8. Could the same positions have been visited in a better order?
   (``order_efficiency``, ``redundancy_distance``)
9. Do the conditions differ, and how confidently? (``sampling_unit.condition_contrast``
   and, with several features, ``sampling_unit.feature_ranking``:
   each condition against the control, with objects and with fields as the unit)

The report opens with what the tables hold (``dataset_overview``: objects,
fields, features, conditions and repeats, normalisation) and with the
declaration of what the acquisition was asked to preserve
(``workflows.Setup.declaration()``), so every answer is read against a named
target; it closes with "What this dataset cannot tell me" (R6): every question
not assessed, with what it needs, and the notes to read the answers with. A
question without its analysis says "not assessed" and what it needs; nothing
is filled in. The **sampling
profile** lists only measured quantities, each with its numbers and its
source: how well a declared control is seen, the design effect of whole
fields, where the variance of the mean sits, the distance over which
neighbouring fields are alike, how many fields hold the new states, and
whether labelled rare or unusual objects cluster in a few fields. It
describes this acquisition under these features; it is not a biological
conclusion. Where the sampling frame (candidate positions) would
be needed, the report says so (P9, not implemented).

``MicroscopistReport.summary()`` is short by default: a title line, an "At a
glance" block with each answer's status marker and one-line headline
(``report_status``, the rules question by question), then each answer in its
short form where it has one (the opening, ``opening``; the Foundation and
questions 1 and 3, ``report_sampling``; question 2, ``report_enough``;
questions 4, 5 and 7, ``report_plan``; questions 6, 8 and 9 and the closing,
``report_answers``; the declaration, ``workflows``), each caveat once;
``summary(detail="full")`` gives every sentence, each section's header with
its marker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from .._caveats import assumption, caveat_lines, pool_assumptions
from .._style import DASH, PROBLEM, Status, check_detail, fitted, headline, wrap
from .._wording import count, listing
from ..sampling_unit import (
    ConditionContrast,
    DescriptorSpreadCheck,
    FeatureDiscovery,
    FeatureRanking,
    FOVClustering,
    FOVVariability,
    ObjectRuleCheck,
    UnusualObjects,
)
from ..schema import observation_words, unit_words
from ..sufficiency import SufficiencyReport
from .adequacy import AdequacyResult
from .calibration import StoppingCalibration
from .controls import ControlResult
from .controls_text import representation_gate
from .effort import EffortResult
from .inspection import InspectionResult
from .inspection_text import inspection_parts
from .order_text import order_parts
from .overview import DatasetOverview
from .protocol import ProtocolResult
from .reliability import ReliabilityResult
from .report_answers import conditions_sections, limits_text, short_section
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
from .report_status import (
    caveat_status,
    integrity_status,
    overview_status,
    question_statuses,
    weak_status,
)
from .spatial import EfficiencyResult

OVERVIEW_TITLE = "What is in the dataset"
DECLARATION_TITLE = "What this acquisition was asked to preserve"
LIMITS_TITLE = "What this dataset cannot tell me"
REPORT_TITLE = "Microscopist's report"
GLANCE_TITLE = "At a glance"
# The planning questions, answered by workflows.optimise rather than by report.
_PLANNING = ("next_time", "bought", "effort", "order")
# What the markers mean: what each answer found (owner, 2026-10-01).
GLANCE_KEY = (
    "Key: ✅ yes, or a clear answer · ⚠️ partly, a trend only, or something to look at · ❌ "
    "check the data first · ➖ no conclusion possible here. The word beside each marker says "
    "which."
)


def questions_for(units: str = "cells") -> tuple[tuple[str, str], ...]:
    """The report's questions, keyed, in the words of its observations (``tracks``)."""
    return (
        ("unit", f"Foundation: why count fields, and not only {units}?"),
        ("controls", "1. Can these features detect what I care about?"),
        ("enough", "2. Did I image enough?"),
        ("profile", "3. What is limiting my sampling?"),
        ("next_time", "4. How many fields should I image next time?"),
        ("bought", "5. What did the extra fields buy?"),
        ("inspect", "6. Which fields should I look at?"),
        ("effort", "7. More fields or more samples?"),
        ("order", "8. Could I have sampled more efficiently?"),
        ("conditions", "9. Do the conditions differ, and how confidently?"),
        ("populations", f"10. Are there groups of {units}, and is a condition enriched in one?"),
    )


def short_questions_for(units: str = "cells") -> dict[str, str]:
    """The questions as the "At a glance" block names them, in the observations' words."""
    return {
        "overview": "The data",
        "unit": "Foundation. Why count fields?",
        "controls": "1. Can the features see the phenotype?",
        "enough": "2. Did I image enough?",
        "profile": "3. What limits my sampling?",
        "next_time": "4. How many fields next time?",
        "bought": "5. What did the extra fields buy?",
        "inspect": "6. Which fields should I look at?",
        "effort": "7. More fields or more samples?",
        "order": "8. A better order?",
        "conditions": "9. Do the conditions differ?",
        "populations": f"10. Groups of {units}?",
    }


QUESTIONS = questions_for()
SHORT_QUESTIONS = short_questions_for()


@dataclass(frozen=True)
class MicroscopistReport:
    """The answers by question (``sections``) and the sampling profile (``profile``).

    ``sections`` hold the full text of each question, ``short`` the short form
    of the questions that have one (and of the opening, ``"overview"``; the
    others print their full text in both), ``statuses`` and ``headlines`` each
    answer's marker and one-line answer (``report_status``), keyed like
    ``sections`` and ``"overview"``; ``units`` the observations, for the titles.
    """

    sections: dict[str, str]
    profile: pd.DataFrame
    notes: tuple[str, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    declaration: tuple[str, ...] = ()
    overview: str | None = None
    short: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, Status] = field(default_factory=dict)
    headlines: dict[str, str] = field(default_factory=dict)
    title: str = REPORT_TITLE
    units: str = "cells"

    def summary(self, questions: Sequence[str] | None = None, detail: str = "short") -> str:
        """The whole report, question by question, or only the ``questions`` keys given.

        ``detail="short"`` (the default) opens with a title line and an "At a
        glance" block, one line per answer with its marker, then gives each
        answer in its short form; ``"full"`` gives every sentence, each
        section's header with its marker. The dataset overview and the
        declaration of what the acquisition was asked to preserve, when given,
        come first, and the closing section, "What this dataset cannot tell
        me" (R6 of ``docs/report/questions.md``), lists every question not assessed
        with what it needs and the notes to read the answers with, whatever
        ``questions`` selects.
        """
        short = check_detail(detail) == "short"
        titles = questions_for(self.units)
        keys = [key for key, _ in titles]
        if questions is not None:
            unknown = sorted(set(questions) - set(keys))
            if unknown:
                raise ValueError(f"unknown report questions {unknown}; choose from {keys}")
        chosen = [(k, t) for k, t in titles if questions is None or k in questions]
        blocks = [self.title, self.at_a_glance([k for k, _ in chosen])] if short else []
        if self.overview:
            text = self.short.get("overview") if short else None
            text = text or _indent(self.overview)
            blocks.append(f"{self._header('overview', OVERVIEW_TITLE)}\n{text}")
        if self.declaration:
            text = self.short.get("declaration") if short else None
            text = text or "\n".join(wrap(f"- {line}", 2, hang=4) for line in self.declaration)
            blocks.append(f"{DECLARATION_TITLE}\n{text}")
        for key, title in chosen:
            text = self.short.get(key, self.sections[key]) if short else self.sections[key]
            blocks.append(f"{self._header(key, title)}\n{text}")
        blocks.append(f"{LIMITS_TITLE}\n{self.limits(questions, detail)}")
        return fitted("\n\n".join(blocks))

    def _header(self, key: str, title: str) -> str:
        """A section's title with its marker, when the answer has a status."""
        status = self.statuses.get(key)
        return title if status is None else headline(status, title)

    def at_a_glance(self, keys: Sequence[str] | None = None) -> str:
        """One line per answer: ``<marker> <n>. <short question> — <headline>``."""
        lines = [GLANCE_TITLE, wrap(GLANCE_KEY, "  ", hang=4)]
        short_titles = short_questions_for(self.units)
        chosen = [k for k, _ in QUESTIONS if keys is None or k in keys]
        for key in ["overview", *chosen]:
            status = self.statuses.get(key)
            if status is None:
                continue
            line = headline(status, short_titles[key], self.headlines.get(key))
            lines.append(wrap(line, "  ", hang=5))
        # The planning questions come from optimise; say so rather than skip their numbers.
        missing = [
            title.split(".")[0]
            for key, title in QUESTIONS
            if key in _PLANNING and key not in chosen
        ]
        if missing and any(key in chosen for key in ("inspect", "conditions")):
            subject = (
                f"Questions {listing(missing, 8)}"
                if len(missing) > 1
                else (f"Question {missing[0]}")
            )
            note = f"{subject}, planning the next experiment, come from workflows.optimise."
            lines.append(wrap(note, "  ", hang=4))
        return "\n".join(lines)

    def limits(self, questions: Sequence[str] | None = None, detail: str = "full") -> str:
        """The questions not assessed (among ``questions``, or all), what each needs, the
        notes (``report_answers.limits_text``; ``detail="short"`` names each need in a few
        words and leaves out a note a section above already states)."""
        return limits_text(
            self,
            questions,
            detail,
            titles=questions_for(self.units),
            needs=_NEEDS,
            short_titles=short_questions_for(self.units),
        )


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else line for line in text.splitlines())


def _check(value: object, kind: type, name: str) -> None:
    if value is not None and not isinstance(value, kind):
        raise TypeError(f"{name} must be a {kind.__name__}")


def microscopist_report(
    *,
    controls: ControlResult | None = None,
    sufficiency: SufficiencyReport | None = None,
    calibration: StoppingCalibration | None = None,
    adequacy: AdequacyResult | None = None,
    reliability: ReliabilityResult | None = None,
    protocol: ProtocolResult | None = None,
    inspection: InspectionResult | None = None,
    effort: EffortResult | None = None,
    efficiency: EfficiencyResult | None = None,
    redundancy: pd.DataFrame | None = None,
    variability: FOVVariability | None = None,
    object_rules: ObjectRuleCheck | None = None,
    descriptor_spread: DescriptorSpreadCheck | None = None,
    unusual: UnusualObjects | None = None,
    clustering: FOVClustering | None = None,
    contrast: ConditionContrast | None = None,
    ranking: FeatureRanking | None = None,
    discovery: FeatureDiscovery | None = None,
    overview: DatasetOverview | None = None,
    minimum_detection: float = 0.9,
    declaration: Sequence[str] = (),
) -> MicroscopistReport:
    """Assemble computed results into one report, question by question (see the module).

    ``minimum_detection`` is the share of declared positive-control fields the
    representation must recognize before the adequacy answers are read
    without a warning (provisional preset 0.9, as for the protocol threshold).
    Below it, every adequacy section is prefixed with the detection rate, since
    a "nothing new" answer cannot exclude a phenotype the features miss. With
    planted objects declared and none planted, it says the planted check could
    not run instead, never reading the whole fields (owner, 2026-10-05).
    ``declaration`` is what the acquisition was asked to preserve, one sentence
    per target (``workflows.Setup.declaration()``), and ``overview`` what the
    tables hold (``dataset_overview``); the report prints them first.
    """
    if isinstance(minimum_detection, bool) or not 0 < float(minimum_detection) <= 1:
        raise ValueError("minimum_detection must be a share in (0, 1]")
    if isinstance(declaration, str) or any(not isinstance(d, str) for d in declaration):
        raise TypeError("declaration must be a sequence of sentences")
    for value, kind, name in (
        (controls, ControlResult, "controls"),
        (sufficiency, SufficiencyReport, "sufficiency"),
        (calibration, StoppingCalibration, "calibration"),
        (adequacy, AdequacyResult, "adequacy"),
        (reliability, ReliabilityResult, "reliability"),
        (protocol, ProtocolResult, "protocol"),
        (inspection, InspectionResult, "inspection"),
        (effort, EffortResult, "effort"),
        (efficiency, EfficiencyResult, "efficiency"),
        (redundancy, pd.DataFrame, "redundancy"),
        (variability, FOVVariability, "variability"),
        (object_rules, ObjectRuleCheck, "object_rules"),
        (descriptor_spread, DescriptorSpreadCheck, "descriptor_spread"),
        (unusual, UnusualObjects, "unusual"),
        (clustering, FOVClustering, "clustering"),
        (contrast, ConditionContrast, "contrast"),
        (ranking, FeatureRanking, "ranking"),
        (discovery, FeatureDiscovery, "discovery"),
        (overview, DatasetOverview, "overview"),
    ):
        _check(value, kind, name)
    parts: dict[str, list[str]] = {key: [] for key, _ in QUESTIONS}
    short = {}
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
    profile = sampling_profile(controls, effort, redundancy, inspection, variability, clustering)
    if len(profile) or unusual is not None or clustering is not None:
        parts["profile"].append("profile")  # its text, wrapped at its indent, is set below
    parts["conditions"], conditions = conditions_sections(contrast, ranking, discovery)
    if conditions:
        short["conditions"] = conditions
    # Redundancy alone feeds the profile; question 8 is answered by an order replay.
    frame = assumption(
        "Whether these were the right positions needs every candidate position from an "
        "overview (not available: sampling-frame bias is not assessed)."
    )
    notes = [
        assumption("Every answer holds for the stated features, tolerances and definition of new."),
        assumption(
            "Replays, draws and hold-outs describe these samples; they are not independent "
            "experiments. Statements across samples count samples or repeats."
        ),
        assumption(
            "A field count that preserved this acquisition is not a protocol until it holds on "
            "samples it was not chosen on."
        ),
    ]
    weakened: list[str] = []
    weak = False
    if controls is None or controls.status == "not assessed":
        notes.append(
            assumption(
                "No representation control: a 'nothing new' answer says nothing about "
                "phenotypes these features and this definition of new cannot see."
            )
        )
    else:
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
    notes.append(
        assumption(
            "Technical drift between repeats, batches or plates is not assessed: groups are "
            "analysed apart, and a shift shared by all fields of a group is not new within it."
        )
    )
    framed = None
    if efficiency is not None:
        parts["order"].append("\n".join(caveat_lines([frame], 0, detail="full")))
        short["order"] = short_section([order_parts(efficiency, indent=2)], frame)
    elif redundancy is not None and parts["profile"]:
        framed = frame  # it closes question 3 (profile_full, profile_short)
    elif redundancy is not None:
        # A caveat alone does not assess question 3: it stays "not assessed", and
        # the caveat is kept with the notes.
        notes.append(frame)
    if parts["profile"]:
        short["profile"] = profile_short(
            profile,
            variability=variability,
            effort=effort,
            unusual=unusual,
            clustering=clustering,
            frame=framed,
        )
    sections = {
        key: _indent("\n\n".join(texts)) if texts else f"  Not assessed: {_NEEDS[key]}."
        for key, texts in parts.items()
    }
    # The Foundation, 1 and 3 are wrapped at the indent they are printed at.
    if parts["unit"]:
        sections["unit"] = unit_full(variability, object_rules, descriptor_spread)
    if controls is not None:
        sections["controls"] = controls_full(controls)
    if parts["profile"]:
        sections["profile"] = profile_full(profile, unusual, clustering, framed)
    inputs = {key: ("given" if parts[key] else "not assessed") for key, _ in QUESTIONS}
    results = {
        "variability": variability,
        "object_rules": object_rules,
        "descriptor_spread": descriptor_spread,
        "controls": controls,
        "sufficiency": sufficiency,
        "calibration": calibration,
        "clustering": clustering,
        "unusual": unusual,
        "reliability": reliability,
        "protocol": protocol,
        "adequacy": adequacy,
        "inspection": inspection,
        "effort": effort,
        "efficiency": efficiency,
        "contrast": contrast,
        "ranking": ranking,
        "discovery": discovery,
    }
    # One "Assumptions:" block per section, after its answers; a caution raises its status.
    sections = {key: pool_assumptions(text) for key, text in sections.items()}
    short = {key: pool_assumptions(text) for key, text in short.items()}
    answers = question_statuses(results, profile, minimum_detection)
    for key, (status, glance) in answers.items():
        answers[key] = caveat_status(status, short.get(key, sections[key]), key), glance
    statuses = {key: answers[key][0] for key, _ in QUESTIONS}
    headlines = {key: answers[key][1] for key, _ in QUESTIONS}
    for key, _ in QUESTIONS:
        if not parts[key]:
            statuses[key] = Status.NOT_ASSESSED
            headlines[key] = f"not assessed; {_SHORT_NEEDS[key]}"
    for key in weakened:
        gated = (statuses[key], f"small-state representation not assessed; {headlines[key]}")
        statuses[key], headlines[key] = weak_status(answers[key]) if weak else gated
    document = MicroscopistReport(
        sections,
        profile,
        tuple(notes),
        inputs,
        tuple(declaration),
        None,
        short,
        statuses,
        headlines,
        units=_units(results),
    )
    return with_opening(document, overview) if overview is not None else document


def _units(results: Mapping[str, Any]) -> str:
    """The observations the results were computed on, plural (``cells`` when undeclared)."""
    found = [getattr(r, "settings", None) for r in results.values()]
    found = [f for f in found if isinstance(f, dict) and f.get("observation_unit")]
    return observation_words(found[0] if found else {})[1]


def opening(overview: DatasetOverview, integrity: Any = None) -> dict[str, Any]:
    """The report's opening, "What is in the dataset", short and full, with its status.

    The overview (``DatasetOverview.summary``) with its data-check marker, and
    the data-integrity checks as one marked line (short) or their summary
    (full); ``integrity`` is an ``IntegrityResult``, or ``None`` when the
    checks were not run (the short form says so).
    """
    status, text = integrity_status(integrity)
    line = f"{status.marker} {text[:1].upper()}{text[1:]}."
    short = "\n".join(wrap(text, 2, hang=4) for text in overview.short_lines())
    full = overview.summary(detail="full")
    if integrity is not None:
        involved = integrity.field_labels()
        if involved:
            line = f"{line[:-1]}: {listing(involved, limit=3)}."
        full = f"{full}\n\n{integrity.summary(detail='full')}"
    overall, glance = overview_status(overview, integrity)
    counts = overview.counts
    settings = overview.settings
    unit, units = unit_words(settings["observation_unit"], settings["observation_unit_declared"])
    title = (
        f"{REPORT_TITLE} {DASH} {count(counts['objects'], unit, units)} in "
        f"{count(counts['fields'], 'field')}"
    )
    return {
        "short": f"{short}\n{wrap(line, 2, hang=5)}",
        "full": full,
        "status": overall,
        "headline": glance,
        "title": title,
        "units": units,
    }


def with_opening(
    document: MicroscopistReport, overview: DatasetOverview, integrity: Any = None
) -> MicroscopistReport:
    """The report with ``opening(overview, integrity)`` as its first section and title."""
    parts = opening(overview, integrity)
    return replace(
        document,
        overview=parts["full"],
        short={**document.short, "overview": parts["short"]},
        statuses={**document.statuses, "overview": parts["status"]},
        headlines={**document.headlines, "overview": parts["headline"]},
        title=parts["title"],
        units=parts["units"],
    )


__all__ = [
    "DECLARATION_TITLE",
    "GLANCE_TITLE",
    "LIMITS_TITLE",
    "OVERVIEW_TITLE",
    "QUESTIONS",
    "REPORT_TITLE",
    "SHORT_QUESTIONS",
    "MicroscopistReport",
    "microscopist_report",
    "opening",
    "questions_for",
    "short_questions_for",
    "with_opening",
]
