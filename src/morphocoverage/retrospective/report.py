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

Shared-private: re-exports ``_detection_adjusted_bounds`` and
``_false_alarm_rate`` of ``report_profile`` for its callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from .._wording import percent
from ..sampling_unit import (
    ConditionContrast,
    DescriptorSpreadCheck,
    FeatureRanking,
    FOVClustering,
    FOVVariability,
    ObjectRuleCheck,
    UnusualObjects,
)
from ..sufficiency import SufficiencyReport
from .adequacy import AdequacyResult
from .calibration import StoppingCalibration
from .controls import ControlResult
from .effort import EffortResult
from .inspection import InspectionResult
from .overview import DatasetOverview
from .protocol import ProtocolResult
from .reliability import ReliabilityResult
from .report_profile import (  # noqa: F401  (private helpers re-exported for callers)
    _control_rate,
    _detection_adjusted_bounds,
    _false_alarm_rate,
    sampling_profile,
)
from .spatial import EfficiencyResult

OVERVIEW_TITLE = "What is in the dataset"
DECLARATION_TITLE = "What this acquisition was asked to preserve"
LIMITS_TITLE = "What this dataset cannot tell me"

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
    """The answers by question (``sections``) and the sampling profile (``profile``)."""

    sections: dict[str, str]
    profile: pd.DataFrame
    notes: tuple[str, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    declaration: tuple[str, ...] = ()
    overview: str | None = None

    def summary(self, questions: Sequence[str] | None = None) -> str:
        """The whole report, question by question, or only the ``questions`` keys given.

        The dataset overview and the declaration of what the acquisition was
        asked to preserve, when given, come first, and the closing section,
        "What this dataset cannot tell me" (R6 of ``docs/questions.md``), lists
        every question not assessed with what it needs and the notes to read
        the answers with, whatever ``questions`` selects.
        """
        keys = [key for key, _ in QUESTIONS]
        if questions is not None:
            unknown = sorted(set(questions) - set(keys))
            if unknown:
                raise ValueError(f"unknown report questions {unknown}; choose from {keys}")
        blocks = []
        if self.overview:
            blocks.append(f"{OVERVIEW_TITLE}\n{_indent(self.overview)}")
        if self.declaration:
            blocks.append(
                f"{DECLARATION_TITLE}\n" + "\n".join(f"  - {line}" for line in self.declaration)
            )
        for key, title in QUESTIONS:
            if questions is not None and key not in questions:
                continue
            blocks.append(f"{title}\n{self.sections[key]}")
        blocks.append(f"{LIMITS_TITLE}\n{self.limits(questions)}")
        return "\n\n".join(blocks)

    def limits(self, questions: Sequence[str] | None = None) -> str:
        """The questions not assessed (among ``questions``, or all), what each needs, the notes."""
        missing = [
            f"  - {title}: not assessed; needs: {_NEEDS[key]}."
            for key, title in QUESTIONS
            if (questions is None or key in questions) and self.inputs.get(key) == "not assessed"
        ]
        # Assessed is not "everything was checked": each answer holds for what it read.
        assessed = (
            "  - Every question was assessed, each on the columns and under the assumptions "
            "stated with it and in the notes below."
        )
        lines = missing or [assessed]
        lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)


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
    if sufficiency is not None:
        parts["enough"].append(sufficiency.summary())
        adjusted = _detection_adjusted_bounds(sufficiency, controls)
        if adjusted:
            parts["enough"].append(adjusted)
    if calibration is not None:
        parts["enough"].append(calibration.summary())
    if adequacy is not None:
        parts["bought"].append(adequacy.summary())
    if reliability is not None:
        parts["next_time"].append(reliability.summary())
    if protocol is not None:
        parts["next_time"].append(protocol.summary())
    if inspection is not None:
        parts["inspect"].append(inspection.summary())
    if effort is not None:
        parts["effort"].append(effort.summary())
    if efficiency is not None:
        parts["order"].append(efficiency.summary())
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
    if contrast is not None:
        parts["conditions"].append(contrast.summary())
        parts["conditions"].append(
            "This describes what these fields support about a difference under this feature; "
            "generalizing it needs independent biological repeats, which is the user's inference."
        )
    if ranking is not None:
        parts["conditions"].append(ranking.summary())
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
    if controls is None or controls.status == "not assessed":
        notes.append(
            "No representation control: a 'nothing new' answer says nothing about phenotypes "
            "these features cannot see."
        )
    else:
        detection = _control_rate(controls, "positive")
        if detection is not None and detection < float(minimum_detection):
            weak = (
                f"Weak representation: fields of the declared positive control were recognized "
                f"in only {percent(detection)} of checks (below {percent(minimum_detection)})"
            )
            for key in ("enough", "next_time", "bought"):
                if parts[key]:
                    parts[key].insert(
                        0, f"{weak}, so the answers below cannot exclude a phenotype like it."
                    )
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
        parts["order"].append(frame)
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
    return MicroscopistReport(
        sections,
        profile,
        tuple(notes),
        inputs,
        tuple(declaration),
        None if overview is None else overview.summary(),
    )


__all__ = [
    "DECLARATION_TITLE",
    "LIMITS_TITLE",
    "OVERVIEW_TITLE",
    "QUESTIONS",
    "MicroscopistReport",
    "microscopist_report",
]
