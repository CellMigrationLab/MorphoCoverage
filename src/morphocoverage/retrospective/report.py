"""The microscopist's report: every answer in one document, in the order of the questions.

Roadmap Step 10. ``microscopist_report`` takes the results a user has computed
(each analysis stays a separate, inspectable call) and orders their plain
sentences by the questions of ``docs/proposals/microscopist_report.md``:

1. Can these features detect what I care about? (``representation_control``)
2. Did I image enough? (``sufficiency_report``, ``adequacy_points``,
   ``stopping_calibration``)
3. What is limiting my sampling? (the sampling profile, below)
4. How many fields next time? (``sampling_reliability``, ``protocol_reliability``)
5. What did the extra fields buy? (``adequacy_points``)
6. Which fields should I inspect? (``inspect_fovs``)
7. More fields or more samples? (``effort_projection``)
8. Could the same positions have been visited in a better order?
   (``order_efficiency``, ``redundancy_distance``)

A question without its analysis says "not assessed" and what it needs; nothing
is filled in. The **sampling profile** lists only measured quantities, each
with its numbers and its source: how well a declared control is seen, the
design effect of whole fields, where the variance of the mean sits, the
distance over which neighbouring fields are alike, and how many fields hold
the new states. It describes this acquisition under these features; it is not
a biological conclusion. Where the sampling frame (candidate positions) would
be needed, the report says so (roadmap Step 9, not implemented).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..sufficiency import SufficiencyReport
from ._text import unit_plural
from .adequacy import AdequacyResult
from .calibration import StoppingCalibration
from .controls import ControlResult
from .effort import EffortResult, design_effect, share_caveat
from .inspection import InspectionResult
from .protocol import ProtocolResult
from .reliability import ReliabilityResult
from .spatial import EfficiencyResult

QUESTIONS = (
    ("controls", "1. Can these features detect what I care about?"),
    ("enough", "2. Did I image enough?"),
    ("profile", "3. What is limiting my sampling?"),
    ("next_time", "4. How many fields should I image next time?"),
    ("bought", "5. What did the extra fields buy?"),
    ("inspect", "6. Which fields should I look at?"),
    ("effort", "7. More fields or more samples?"),
    ("order", "8. Could I have sampled more efficiently?"),
)

_NEEDS = {
    "controls": "declare a positive control (and a reference) with representation_control",
    "enough": "run sufficiency_report, or adequacy_points and stopping_calibration on a replay",
    "profile": "run any of the analyses below",
    "next_time": "run sampling_reliability, then protocol_reliability across samples",
    "bought": "run adequacy_points on a replay",
    "inspect": "run inspect_fovs",
    "effort": "declare a sampling hierarchy and run effort_projection",
    "order": "give stage coordinates and run order_efficiency on a replay with dispersed orders",
}


@dataclass(frozen=True)
class MicroscopistReport:
    """The answers by question (``sections``) and the sampling profile (``profile``)."""

    sections: dict[str, str]
    profile: pd.DataFrame
    notes: tuple[str, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """The whole report, question by question."""
        blocks = []
        for key, title in QUESTIONS:
            blocks.append(f"{title}\n{self.sections[key]}")
        if self.notes:
            blocks.append("Read with care\n" + "\n".join(f"  - {note}" for note in self.notes))
        return "\n\n".join(blocks)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else line for line in text.splitlines())


def _profile(controls, effort, redundancy, inspection) -> pd.DataFrame:
    units = [
        r.settings.get("observation_unit")
        for r in (controls, effort)
        if r is not None and r.settings.get("observation_unit")
    ]
    observations = unit_plural(units[0] if units else "object")
    rows = []
    if controls is not None and controls.status != "not assessed":
        rates = controls.pooled.set_index("candidate")["rate"]
        for kind, words in (
            ("positive", "declared positive-control fields recognized as different"),
            ("planted", f"fields with planted positive-control {observations} recognized"),
            ("reference", "reference fields called different (false alarms)"),
        ):
            if kind in rates.index and pd.notna(rates[kind]):
                rows.append(
                    {
                        "item": "representation",
                        "where": "pooled",
                        "value": float(rates[kind]),
                        "statement": f"{100 * rates[kind]:.0f}% of {words}",
                        "source": "representation_control",
                    }
                )
    if effort is not None:
        hierarchy = effort.settings.get("hierarchy", {})
        level_words = {
            hierarchy.get("fov"): "between fields within a sample",
            hierarchy.get("sample"): "between samples",
            hierarchy.get("biological_repeat"): "between repeats",
            "observation": f"between {observations} within a field",
        }
        for stratum, block in effort.components.groupby("stratum", sort=False):
            effect = design_effect(effort, stratum)
            if effect is not None:
                rows.append(
                    {
                        "item": "field-to-field heterogeneity",
                        "where": stratum,
                        "value": effect,
                        "statement": (
                            f"a mean from whole fields varies {effect:.1f} times as much as one "
                            f"from the same number of random {observations} (design effect)"
                        ),
                        "source": "effort_projection",
                    }
                )
            top = block.dropna(subset=["share"]).sort_values("share", ascending=False)
            if len(top):
                first = top.iloc[0]
                unit = effort.settings.get("independent_unit", "unit")
                unit = "sample" if unit == hierarchy.get("sample") else unit
                caveat = share_caveat(first, int(first["units"]), unit)
                rows.append(
                    {
                        "item": "largest source of uncertainty of the mean",
                        "where": stratum,
                        "value": float(first["share"]),
                        "statement": (
                            f"{100 * first['share']:.0f}% of the variance of the mean is "
                            f"{level_words.get(first['level'], first['level'])}{caveat}"
                        ),
                        "source": "effort_projection",
                    }
                )
    if redundancy is not None and len(redundancy):
        measured = redundancy.dropna(subset=["smallest_lag"])
        always = measured[measured["redundancy_distance"].isna()]
        found = measured.dropna(subset=["redundancy_distance"])
        alike = found[found["redundancy_distance"] > found["smallest_lag"]]
        smallest = float(measured["smallest_lag"].min()) if len(measured) else None
        unit = str(redundancy["unit"].iloc[0]) if "unit" in redundancy.columns else "stage units"
        parts = []
        if len(alike):
            parts.append(
                f"in {len(alike)} of {len(redundancy)} groups, fields closer than about "
                f"{np.median(alike['redundancy_distance']):.0f} {unit} (median) are more "
                "alike than random pairs"
            )
        if len(always):
            largest = float(always["largest_lag"].max()) if "largest_lag" in always else None
            reach = f" (up to {largest:.0f} {unit})" if largest is not None else ""
            parts.append(
                f"in {len(always)} of {len(redundancy)} groups, fields are more alike than "
                f"random pairs at every distance measured{reach}"
            )
        if parts:
            statement = "; ".join(parts)
        elif smallest is not None:
            statement = (
                f"no group shows neighbouring fields more alike than random pairs, down to the "
                f"closest spacing measured ({smallest:.0f} {unit})"
            )
        else:
            statement = None
        if statement is not None:
            rows.append(
                {
                    "item": "spatial correlation",
                    "where": "all groups",
                    "value": float(len(alike)),
                    "statement": statement,
                    "source": "redundancy_distance",
                }
            )
    if inspection is not None and len(inspection.fovs):
        new = inspection.fovs[inspection.fovs["reason"] == "new state"]
        keys = [c for c in inspection.settings.get("fov_key", []) if c in new.columns]
        if len(new):
            rows.append(
                {
                    "item": "new states",
                    "where": "all groups",
                    "value": float(new[keys].drop_duplicates().shape[0]),
                    "statement": (
                        f"new states come from {new[keys].drop_duplicates().shape[0]} fields"
                    ),
                    "source": "inspect_fovs",
                }
            )
    return pd.DataFrame(rows, columns=["item", "where", "value", "statement", "source"])


def _check(value, kind, name):
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
) -> MicroscopistReport:
    """Assemble computed results into one report, question by question (see the module)."""
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
    ):
        _check(value, kind, name)
    parts: dict[str, list[str]] = {key: [] for key, _ in QUESTIONS}
    if controls is not None:
        parts["controls"].append(controls.summary())
    if sufficiency is not None:
        parts["enough"].append(sufficiency.summary())
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
    profile = _profile(controls, effort, redundancy, inspection)
    if len(profile):
        parts["profile"].append(
            "\n".join(
                f"{row.statement} [{row.where}; {row.source}]" for row in profile.itertuples()
            )
        )
    if efficiency is not None or redundancy is not None:
        parts["order"].append(
            "Whether these were the right positions needs every candidate position from an "
            "overview (not available: sampling-frame bias is not assessed)."
        )
    sections = {
        key: _indent("\n\n".join(texts)) if texts else f"  Not assessed: {_NEEDS[key]}."
        for key, texts in parts.items()
    }
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
    inputs = {key: ("given" if parts[key] else "not assessed") for key, _ in QUESTIONS}
    return MicroscopistReport(sections, profile, tuple(notes), inputs)


__all__ = ["QUESTIONS", "MicroscopistReport", "microscopist_report"]
