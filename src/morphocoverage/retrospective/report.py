"""The microscopist's report: every answer in one document, in the order of the questions.

``microscopist_report`` takes the results a user has computed
(each analysis stays a separate, inspectable call) and orders their plain
sentences by the questions of ``docs/questions.md``. It opens with the
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
short form where it has one (the opening, ``opening``; question 2,
``report_enough``; questions 6, 8 and 9 and the closing, ``report_answers``;
the declaration, ``workflows``), each caveat once; ``summary(detail="full")``
gives every sentence, each section's header with its marker.

Shared-private: re-exports ``_detection_adjusted_bounds`` and
``_false_alarm_rate`` of ``report_profile`` for its callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from .._style import DASH, PROBLEM, Status, caution_lines, check_detail, headline, wrap
from .._wording import count, listing, percent
from ..sampling_unit import (
    ConditionContrast,
    DescriptorSpreadCheck,
    FeatureRanking,
    FOVClustering,
    FOVVariability,
    ObjectRuleCheck,
    UnusualObjects,
)
from ..sampling_unit.contrast_text import contrast_parts
from ..schema import unit_plural
from ..sufficiency import SufficiencyReport
from .adequacy import AdequacyResult
from .calibration import StoppingCalibration
from .controls import ControlResult
from .effort import EffortResult
from .inspection import InspectionResult
from .inspection_text import inspection_parts
from .order_text import order_parts
from .overview import DatasetOverview
from .protocol import ProtocolResult
from .reliability import ReliabilityResult
from .report_answers import limits_text, short_section
from .report_enough import enough_full, enough_short
from .report_profile import (  # noqa: F401  (private helpers re-exported for callers)
    _control_rate,
    _detection_adjusted_bounds,
    _false_alarm_rate,
    sampling_profile,
)
from .report_status import integrity_status, overview_status, question_statuses, weak_status
from .spatial import EfficiencyResult

OVERVIEW_TITLE = "What is in the dataset"
DECLARATION_TITLE = "What this acquisition was asked to preserve"
LIMITS_TITLE = "What this dataset cannot tell me"
REPORT_TITLE = "Microscopist's report"
GLANCE_TITLE = "At a glance"

QUESTIONS = (
    ("unit", "Foundation: why count fields, and not only objects?"),
    ("controls", "1. Can these features detect what I care about?"),
    ("enough", "2. Did I image enough?"),
    ("profile", "3. What is limiting my sampling?"),
    ("next_time", "4. How many fields should I image next time?"),
    ("bought", "5. What did the extra fields buy?"),
    ("inspect", "6. Which fields should I look at?"),
    ("effort", "7. More fields or more samples?"),
    ("order", "8. Could I have sampled more efficiently?"),
    ("conditions", "9. Do the conditions differ, and how confidently?"),
)

# The questions as the "At a glance" block names them.
SHORT_QUESTIONS = {
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
}

# What a question not assessed needs, in a few words for its "At a glance" line.
_SHORT_NEEDS = {
    "unit": "needs fov_variability",
    "controls": "needs a declared positive control",
    "enough": "needs sufficiency_report or a stopping calibration",
    "profile": "needs the controls, variability, clustering or effort analyses",
    "next_time": "needs sampling_reliability and protocol_reliability",
    "bought": "needs adequacy_points on a replay",
    "inspect": "needs inspect_fovs",
    "effort": "needs technical repeats or a sampling hierarchy",
    "order": "needs stage coordinates and an order replay",
    "conditions": "needs a control condition and condition_contrast",
}

_NEEDS = {
    "unit": "run fov_variability (and object_rule_check or descriptor_spread_check) "
    "from morphocoverage.sampling_unit",
    "controls": "declare a positive control (and a reference) with representation_control",
    "enough": "run sufficiency_report, or stopping_calibration on a replay",
    "profile": "run representation_control, fov_variability, fov_clustering, "
    "effort_projection, redundancy_distance or inspect_fovs; the profile lists what they measured",
    "next_time": "run sampling_reliability, then protocol_reliability across samples",
    "bought": "run adequacy_points on a replay",
    "inspect": "run inspect_fovs",
    "effort": "declare a sampling hierarchy, or give the FOV table a technical_repeat column "
    "(a well, a dish) kept among the group columns, and run optimise (effort_projection)",
    "order": "give stage coordinates and run order_efficiency on a replay with dispersed orders",
    "conditions": "declare a control condition and run condition_contrast on data with more "
    "than one condition, with analysis groups that keep conditions and repeats apart",
}


@dataclass(frozen=True)
class MicroscopistReport:
    """The answers by question (``sections``) and the sampling profile (``profile``).

    ``sections`` hold the full text of each question, ``short`` the short form
    of the questions that have one (and of the opening, ``"overview"``; the
    others print their full text in both), ``statuses`` and ``headlines`` each
    answer's marker and one-line answer (``report_status``), keyed like
    ``sections`` and ``"overview"``.
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

    def summary(self, questions: Sequence[str] | None = None, detail: str = "short") -> str:
        """The whole report, question by question, or only the ``questions`` keys given.

        ``detail="short"`` (the default) opens with a title line and an "At a
        glance" block, one line per answer with its marker, then gives each
        answer in its short form; ``"full"`` gives every sentence, each
        section's header with its marker. The dataset overview and the
        declaration of what the acquisition was asked to preserve, when given,
        come first, and the closing section, "What this dataset cannot tell
        me" (R6 of ``docs/questions.md``), lists every question not assessed
        with what it needs and the notes to read the answers with, whatever
        ``questions`` selects.
        """
        short = check_detail(detail) == "short"
        keys = [key for key, _ in QUESTIONS]
        if questions is not None:
            unknown = sorted(set(questions) - set(keys))
            if unknown:
                raise ValueError(f"unknown report questions {unknown}; choose from {keys}")
        chosen = [(k, t) for k, t in QUESTIONS if questions is None or k in questions]
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
        return "\n\n".join(blocks)

    def _header(self, key: str, title: str) -> str:
        """A section's title with its marker, when the answer has a status."""
        status = self.statuses.get(key)
        return title if status is None else headline(status, title)

    def at_a_glance(self, keys: Sequence[str] | None = None) -> str:
        """One line per answer: ``<marker> <n>. <short question> — <headline>``."""
        lines = [GLANCE_TITLE]
        chosen = [k for k, _ in QUESTIONS if keys is None or k in keys]
        for key in ["overview", *chosen]:
            status = self.statuses.get(key)
            if status is None:
                continue
            line = headline(status, SHORT_QUESTIONS[key], self.headlines.get(key))
            lines.append(wrap(line, "  ", hang=5))
        return "\n".join(lines)

    def limits(self, questions: Sequence[str] | None = None, detail: str = "full") -> str:
        """The questions not assessed (among ``questions``, or all), what each needs, the
        notes (``report_answers.limits_text``; ``detail="short"`` names each need in a few
        words and leaves out a note a section above already states)."""
        return limits_text(
            self, questions, detail, titles=QUESTIONS, needs=_NEEDS, short_titles=SHORT_QUESTIONS
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
    overview: DatasetOverview | None = None,
    minimum_detection: float = 0.9,
    declaration: Sequence[str] = (),
) -> MicroscopistReport:
    """Assemble computed results into one report, question by question (see the module).

    ``minimum_detection`` is the share of declared positive-control fields the
    representation must recognize before the adequacy answers are read
    without a warning (provisional preset 0.9, as for the protocol threshold).
    Below it, every adequacy section is prefixed with the detection rate, since
    a "nothing new" answer cannot exclude a phenotype the features miss.
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
        (overview, DatasetOverview, "overview"),
    ):
        _check(value, kind, name)
    parts: dict[str, list[str]] = {key: [] for key, _ in QUESTIONS}
    for check in (variability, object_rules, descriptor_spread):
        if check is not None:
            parts["unit"].append(check.summary())
    if controls is not None:
        parts["controls"].append(controls.summary())
    parts["enough"] += enough_full(sufficiency, calibration, controls)
    short = {}
    if parts["enough"]:
        short["enough"] = enough_short(sufficiency, calibration, controls)
    if adequacy is not None:
        parts["bought"].append(adequacy.summary())
    if reliability is not None:
        parts["next_time"].append(reliability.summary())
    if protocol is not None:
        parts["next_time"].append(protocol.summary())
    if inspection is not None:
        parts["inspect"].append(inspection.summary(detail="full"))
        short["inspect"] = short_section([inspection_parts(inspection, indent=2)])
    if effort is not None:
        parts["effort"].append(effort.summary())
    if efficiency is not None:
        parts["order"].append(efficiency.summary(detail="full"))
    profile = sampling_profile(controls, effort, redundancy, inspection, variability, clustering)
    if len(profile):
        parts["profile"].append(
            "\n".join(
                f"{row.statement} [{row.where}; {row.source}]" for row in profile.itertuples()
            )
        )
    for check in (unusual, clustering):
        if check is not None:
            parts["profile"].append(check.summary())
    generalizing = (
        "This describes what these fields support about a difference under this feature; "
        "generalizing it needs independent biological repeats, which is the user's inference."
    )
    if contrast is not None:
        parts["conditions"].append(contrast.summary(detail="full"))
        parts["conditions"].append(caution_lines([generalizing], indent=0)[0])
    if ranking is not None:
        parts["conditions"].append(ranking.summary(detail="full"))
    if contrast is not None or ranking is not None:
        answers = [contrast_parts(contrast, indent=2)] if contrast is not None else []
        answers += [ranking.short_parts(indent=2)] if ranking is not None else []
        closing = [generalizing] if contrast is not None else []
        short["conditions"] = short_section(answers, *closing)
    # Redundancy alone feeds the profile; question 8 is answered by an order replay.
    frame = (
        "Whether these were the right positions needs every candidate position from an "
        "overview (not available: sampling-frame bias is not assessed)."
    )
    notes = [
        "Every answer holds for the stated features, tolerances and definition of new.",
        (
            "Replays, draws and hold-outs describe these samples; they are not independent "
            "experiments. Statements across samples count samples or repeats."
        ),
        (
            "A field count that preserved this acquisition is not a protocol until it holds on "
            "samples it was not chosen on."
        ),
    ]
    weakened: list[str] = []
    if controls is None or controls.status == "not assessed":
        notes.append(
            "No representation control: a 'nothing new' answer says nothing about phenotypes "
            "these features and this definition of new cannot see."
        )
    else:
        # With planted objects declared, the rare-state rate gates the warning (P6-4).
        planted = (controls.settings.get("control") or {}).get("planted_objects")
        planted_rate = _control_rate(controls, "planted") if planted else None
        detection = (
            planted_rate if planted_rate is not None else _control_rate(controls, "positive")
        )
        seen = (
            f"ordinary fields with {planted} planted positive-control objects"
            if planted_rate is not None
            else "fields of the declared positive control"
        )
        if detection is not None and detection < float(minimum_detection):
            weak = (
                f"Weak representation: {seen} were recognized "
                f"in only {percent(detection)} of checks (below {percent(minimum_detection)})"
            )
            for key in ("enough", "next_time", "bought"):
                if parts[key]:
                    weakened.append(key)
                    parts[key].insert(
                        0, f"{weak}, so the answers below cannot exclude a phenotype like it."
                    )
            if "enough" in short:
                line = f"{PROBLEM} {weak}, so the answers below cannot exclude a phenotype like it."
                short["enough"] = f"{wrap(line, 2, hang=5)}\n{short['enough']}"
            # The notes close the report: the adequacy answers are above them.
            notes.append(
                f"{weak}, so the adequacy answers (questions 2, 4 and 5) cannot exclude a "
                "phenotype like it."
            )
    notes.append(
        "Technical drift between repeats, batches or plates is not assessed: groups are "
        "analysed apart, and a shift shared by all fields of a group is not new within it."
    )
    if efficiency is not None:
        parts["order"].append(caution_lines([frame], indent=0)[0])
        short["order"] = short_section([order_parts(efficiency, indent=2)], frame)
    elif redundancy is not None and parts["profile"]:
        parts["profile"].append(frame)
    elif redundancy is not None:
        # A caveat alone does not assess question 3: it stays "not assessed", and
        # the caveat is kept with the notes.
        notes.append(frame)
    sections = {
        key: _indent("\n\n".join(texts)) if texts else f"  Not assessed: {_NEEDS[key]}."
        for key, texts in parts.items()
    }
    inputs = {key: ("given" if parts[key] else "not assessed") for key, _ in QUESTIONS}
    results = {
        "variability": variability,
        "object_rules": object_rules,
        "descriptor_spread": descriptor_spread,
        "controls": controls,
        "sufficiency": sufficiency,
        "calibration": calibration,
        "clustering": clustering,
        "reliability": reliability,
        "protocol": protocol,
        "adequacy": adequacy,
        "inspection": inspection,
        "effort": effort,
        "efficiency": efficiency,
        "contrast": contrast,
        "ranking": ranking,
    }
    answers = question_statuses(results, profile, minimum_detection)
    statuses = {key: answers[key][0] for key, _ in QUESTIONS}
    headlines = {key: answers[key][1] for key, _ in QUESTIONS}
    for key, _ in QUESTIONS:
        if not parts[key]:
            statuses[key] = Status.NOT_ASSESSED
            headlines[key] = f"not assessed; {_SHORT_NEEDS[key]}"
    for key in weakened:
        statuses[key], headlines[key] = weak_status(answers[key])
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
    )
    return with_opening(document, overview) if overview is not None else document


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
        involved = integrity.fields_involved()
        if involved:
            line = f"{line[:-1]}: {listing(involved, limit=3)}."
        full = f"{full}\n\n{integrity.summary(detail='full')}"
    overall, glance = overview_status(overview, integrity)
    counts = overview.counts
    unit = overview.settings["observation_unit"]
    title = (
        f"{REPORT_TITLE} {DASH} {count(counts['objects'], unit, unit_plural(unit))} in "
        f"{count(counts['fields'], 'field')}"
    )
    return {
        "short": f"{short}\n{wrap(line, 2, hang=5)}",
        "full": full,
        "status": overall,
        "headline": glance,
        "title": title,
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
    "with_opening",
]
