"""The microscopist's report: every answer in one document, in the order of the questions.

Roadmap Step 10. ``microscopist_report`` takes the results a user has computed
(each analysis stays a separate, inspectable call) and orders their plain
sentences by the questions of ``docs/questions.md``. It opens with the
foundation every other answer rests on, then answers the questions in order:

- Foundation: why count fields, and not only objects? (``sampling_unit``:
  ``fov_variability``, ``object_rule_check``, ``descriptor_spread_check``)
  How much do the acquired fields differ, and would an object-count rule have
  held when its objects came as whole fields?

1. Can these features detect what I care about? (``representation_control``)
2. Did I image enough? (``sufficiency_report``, ``adequacy_points``,
   ``stopping_calibration``)
3. What is limiting my sampling? (the sampling profile, below, and
   ``sampling_unit.fov_clustering`` of rare or unusual objects)
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
distance over which neighbouring fields are alike, how many fields hold the
new states, and whether labelled rare or unusual objects cluster in a few
fields. It describes this acquisition under these features; it is not
a biological conclusion. Where the sampling frame (candidate positions) would
be needed, the report says so (roadmap Step 9, not implemented).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:  # sampling_unit imports retrospective helpers, so it is imported late
    from ..sampling_unit import (
        DescriptorSpreadCheck,
        FOVClustering,
        FOVVariability,
        ObjectRuleCheck,
        UnusualObjects,
    )

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
)

_NEEDS = {
    "unit": "run fov_variability (and object_rule_check or descriptor_spread_check) "
    "from morphocoverage.sampling_unit",
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

    def summary(self, questions: Sequence[str] | None = None) -> str:
        """The whole report, question by question, or only the ``questions`` keys given."""
        keys = [key for key, _ in QUESTIONS]
        if questions is not None:
            unknown = sorted(set(questions) - set(keys))
            if unknown:
                raise ValueError(f"unknown report questions {unknown}; choose from {keys}")
        blocks = []
        for key, title in QUESTIONS:
            if questions is not None and key not in questions:
                continue
            blocks.append(f"{title}\n{self.sections[key]}")
        if self.notes:
            blocks.append("Read with care\n" + "\n".join(f"  - {note}" for note in self.notes))
        return "\n\n".join(blocks)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else line for line in text.splitlines())


def _false_alarm_rate(controls) -> float | None:
    """The declared reference's false-alarm rate, when it was measured."""
    if controls is None or controls.status != "assessed":
        return None
    rates = controls.pooled.set_index("candidate")["rate"]
    if "reference" not in rates.index or pd.isna(rates["reference"]):
        return None
    return float(rates["reference"])


def _detection_adjusted_bounds(sufficiency, controls) -> str | None:
    """The unseen-state bound divided by the declared control's detection rate.

    The rule-of-three bound holds for states this definition would detect in
    every FOV that has them. A phenotype recognized like the declared
    positive control, in a share ``d`` of the fields that show it, is seen
    with probability ``d * f`` per FOV when present in a share ``f``, so the
    same run of FOVs with nothing new bounds ``f`` at ``bound / d`` (at most 1),
    assuming detection is independent between fields.
    """
    if controls is None or controls.status != "assessed" or sufficiency.next_fov.empty:
        return None
    rates = controls.pooled.set_index("candidate")["rate"]
    if "positive" not in rates.index or pd.isna(rates["positive"]) or rates["positive"] <= 0:
        return None
    detection = float(rates["positive"])
    keys = list(sufficiency.group_cols)
    heading = (
        f"Adjusted for the declared control's detection rate ({100 * detection:.0f}% of its "
        "fields recognized): a phenotype seen as well as the control, and shown by no FOV yet,"
        " could still be in"
    )
    lines = [heading]
    for row in sufficiency.next_fov.itertuples(index=False):
        bound = row.unseen_state_max_frequency
        if bound is None or pd.isna(bound):
            continue
        adjusted = min(1.0, float(bound) / detection)
        label = ", ".join(f"{k}={getattr(row, k)}" for k in keys) or "all FOVs (pooled)"
        lines.append(
            f"  {label}: up to {100 * adjusted:.0f}% of FOVs (the {100 * float(bound):.0f}% "
            f"bound above divided by {100 * detection:.0f}%), if detection is independent "
            "between fields."
        )
    return "\n".join(lines) if len(lines) > 1 else None


def _unit_rows(variability, clustering, observations: str) -> list[dict]:
    """Profile rows from the sampling-unit checks: FOVs against random objects, clustering."""
    rows = []
    if variability is not None:
        ok = variability.groups[variability.groups["status"] == "ok"]
        if len(ok):
            effect = ok["design_effect"]
            where = f"median of {len(ok)} groups" if len(ok) > 1 else "1 group"
            rows.append(
                {
                    "item": "fields against random objects",
                    "where": where,
                    "value": float(effect.median()),
                    "statement": (
                        f"a mean from whole fields varies {effect.median():.1f} times as much as "
                        f"one from as many random {observations} (design effect, range "
                        f"{effect.min():.1f}-{effect.max():.1f}); "
                        f"{100 * ok['unusual_fov_share'].median():.0f}% of fields differ more "
                        f"than random {observations} would (5% expected)"
                    ),
                    "source": "fov_variability",
                }
            )
    if clustering is not None:
        table = clustering.table
        for state, block in table.groupby("state", sort=False):
            seen = block[block["state_objects"] > 0]
            if not len(seen):
                continue
            clustered = seen[seen["p_clustered"] <= 0.05]
            statement = (
                f"{clustering.state_name(state)}: clustered in a few fields (p <= 0.05) in "
                f"{len(clustered)} of the {len(seen)} groups where it was seen"
            )
            if len(clustered):
                factor = clustered["fovs_needed"] / clustered["fovs_needed_if_random"]
                statement += (
                    f"; there, seeing it takes about {factor.median():.1f} times the fields "
                    f"that a count of {observations} predicts"
                )
            rows.append(
                {
                    "item": "rare-state clustering",
                    "where": "all groups",
                    "value": float(len(clustered)),
                    "statement": statement,
                    "source": "fov_clustering",
                }
            )
    return rows


def _profile(
    controls, effort, redundancy, inspection, variability=None, clustering=None
) -> pd.DataFrame:
    units = [
        r.settings.get("observation_unit")
        for r in (variability, clustering, controls, effort)
        if r is not None and r.settings.get("observation_unit")
    ]
    observations = unit_plural(units[0] if units else "object")
    rows = _unit_rows(variability, clustering, observations)
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
            # fov_variability, when given, states the design effect per group already
            if effect is not None and variability is None:
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
            count = new[keys].drop_duplicates().shape[0]
            statement = f"new states come from {count} fields"
            false_alarms = _false_alarm_rate(controls)
            n_fovs = inspection.settings.get("n_fovs")
            if false_alarms is not None and n_fovs:
                # The same definition called that share of reference fields new.
                expected = false_alarms * n_fovs
                statement += (
                    f" (about {expected:.0f} of {n_fovs} expected by chance at the "
                    f"{100 * false_alarms:.0f}% false-alarm rate of the declared reference)"
                )
            rows.append(
                {
                    "item": "new states",
                    "where": "all groups",
                    "value": float(count),
                    "statement": statement,
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
    variability: FOVVariability | None = None,
    object_rules: ObjectRuleCheck | None = None,
    descriptor_spread: DescriptorSpreadCheck | None = None,
    unusual: UnusualObjects | None = None,
    clustering: FOVClustering | None = None,
    minimum_detection: float = 0.9,
) -> MicroscopistReport:
    """Assemble computed results into one report, question by question (see the module).

    ``minimum_detection`` is the share of declared positive-control fields the
    representation must recognize before the adequacy answers are read
    without a warning (provisional preset 0.9, as for the protocol threshold).
    Below it, every adequacy section is prefixed with the detection rate, since
    a "nothing new" answer cannot exclude a phenotype the features miss.
    """
    from ..sampling_unit import (
        DescriptorSpreadCheck,
        FOVClustering,
        FOVVariability,
        ObjectRuleCheck,
        UnusualObjects,
    )

    if isinstance(minimum_detection, bool) or not 0 < float(minimum_detection) <= 1:
        raise ValueError("minimum_detection must be a share in (0, 1]")
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
    profile = _profile(controls, effort, redundancy, inspection, variability, clustering)
    if len(profile):
        parts["profile"].append(
            "\n".join(
                f"{row.statement} [{row.where}; {row.source}]" for row in profile.itertuples()
            )
        )
    for check in (unusual, clustering):
        if check is not None:
            parts["profile"].append(check.summary())
    if efficiency is not None or redundancy is not None:
        parts["order"].append(
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
        rates = controls.pooled.set_index("candidate")["rate"]
        detection = float(rates["positive"]) if "positive" in rates.index else None
        if detection is not None and detection < float(minimum_detection):
            warning = (
                f"Weak representation: fields of the declared positive control were recognized "
                f"in only {100 * detection:.0f}% of checks (below {100 * minimum_detection:.0f}%)"
                ", so the answers below cannot exclude a phenotype like it."
            )
            for key in ("enough", "next_time", "bought"):
                if parts[key]:
                    parts[key].insert(0, warning)
            notes.append(warning)
    notes.append(
        "Technical drift between repeats, batches or plates is not assessed: groups are "
        "analysed apart, and a shift shared by all fields of a group is not new within it."
    )
    sections = {
        key: _indent("\n\n".join(texts)) if texts else f"  Not assessed: {_NEEDS[key]}."
        for key, texts in parts.items()
    }
    inputs = {key: ("given" if parts[key] else "not assessed") for key, _ in QUESTIONS}
    return MicroscopistReport(sections, profile, tuple(notes), inputs)


__all__ = ["QUESTIONS", "MicroscopistReport", "microscopist_report"]
