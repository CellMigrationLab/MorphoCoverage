"""The sampling profile of the microscopist's report, and the detection-adjusted bound.

Kept apart from the assembly of the report (``report``). The profile lists
only measured quantities, each with its numbers and its source; one helper
per source keeps each rule readable.

Rates that compare with ``inspect_fovs`` or with the unseen-state bound of the
sufficiency report are the control's leave-one-field-out rates
(``reference_lofo``, ``positive_lofo``): those analyses judge each field
against every other field, and a rate measured after only ``fovs_seen``
reference fields can be many times larger (41% against 0% on one plate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..sufficiency import SufficiencyReport
from ._text import unit_plural
from .controls import ControlResult
from .effort import EffortResult, design_effect
from .effort_text import level_words, share_caveat, unit_word
from .inspection import InspectionResult

if TYPE_CHECKING:  # sampling_unit is imported late by the report
    from ..sampling_unit import FOVClustering, FOVVariability

PROFILE_COLUMNS = ["item", "where", "value", "statement", "source"]


def _lofo_rate(controls: ControlResult | None, kind: str) -> float | None:
    """A leave-one-field-out rate of the declared control, when it was measured."""
    if controls is None or controls.status == "not assessed":
        return None
    rates = controls.pooled.set_index("candidate")["rate"]
    if kind not in rates.index or pd.isna(rates[kind]):
        return None
    return float(rates[kind])


def _false_alarm_rate(controls: ControlResult | None) -> float | None:
    """The declared reference's false-alarm rate, judged as ``inspect_fovs`` judges a field."""
    return _lofo_rate(controls, "reference_lofo")


def _detection_adjusted_bounds(
    sufficiency: SufficiencyReport, controls: ControlResult | None
) -> str | None:
    """The unseen-state bound divided by the declared control's detection rate.

    The rule-of-three bound holds for states this definition would detect in
    every FOV that has them. A phenotype recognized like the declared
    positive control, in a share ``d`` of the fields that show it, is seen
    with probability ``d * f`` per FOV when present in a share ``f``, so the
    same run of FOVs with nothing new bounds ``f`` at ``bound / d`` (at most 1),
    assuming detection is independent between fields. ``d`` is the rate
    against every reference field (``positive_lofo``), as the bound is about
    fields judged against a whole acquisition; it needs a positive control
    only, so it is given under "positive only" too.
    """
    if controls is None or controls.status == "not assessed" or sufficiency.next_fov.empty:
        return None
    detection = _lofo_rate(controls, "positive_lofo")
    if detection is None:
        return (
            "The unseen-state bound is not adjusted for detection: the declared positive "
            "control could not be judged against every reference field."
        )
    if detection <= 0:
        return (
            "The declared positive control was never recognized against every reference "
            "field, so the unseen-state bound says nothing about a phenotype seen like it."
        )
    keys = list(sufficiency.group_cols)
    heading = (
        f"Adjusted for the declared control's detection rate ({100 * detection:.0f}% of its "
        "fields recognized against every reference field): a phenotype seen as well as the "
        "control, and shown by no FOV yet, could still be in"
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


def _variability_rows(variability: FOVVariability | None, observations: str) -> list[dict]:
    """Fields against random objects, from ``fov_variability``."""
    if variability is None:
        return []
    ok = variability.groups[variability.groups["status"] == "ok"]
    if not len(ok):
        return []
    effect = ok["design_effect"]
    shares = pd.to_numeric(ok["unusual_fov_share"], errors="coerce").dropna()
    unusual = (
        f"; {100 * shares.median():.0f}% of fields differ more than random "
        f"{observations} would (5% expected)"
        if len(shares)
        else ""
    )
    where = f"median of {len(ok)} groups" if len(ok) > 1 else "1 group"
    return [
        {
            "item": "fields against random objects",
            "where": where,
            "value": float(effect.median()),
            "statement": (
                f"a mean from whole fields varies {effect.median():.1f} times as much as "
                f"one from as many random {observations} (design effect, range "
                f"{effect.min():.1f}-{effect.max():.1f}){unusual}"
            ),
            "source": "fov_variability",
        }
    ]


def _clustering_rows(clustering: FOVClustering | None, observations: str) -> list[dict]:
    """Whether labelled rare or unusual objects cluster in a few fields."""
    if clustering is None:
        return []
    rows = []
    for state, block in clustering.table.groupby("state", sort=False):
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


def _representation_rows(controls: ControlResult | None, observations: str) -> list[dict]:
    """How well the declared control is seen, and the reference's false alarms."""
    if controls is None or controls.status == "not assessed":
        return []
    rates = controls.pooled.set_index("candidate")["rate"]
    rows = []
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
    return rows


def _effort_rows(
    effort: EffortResult | None, variability: FOVVariability | None, observations: str
) -> list[dict]:
    """The design effect (unless ``fov_variability`` states it) and the largest variance share."""
    if effort is None:
        return []
    hierarchy = effort.settings.get("hierarchy", {})
    words = level_words(hierarchy, observations)
    rows = []
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
            caveat = share_caveat(
                first, int(first["units"]), unit_word(hierarchy, str(first["unit"]))
            )
            rows.append(
                {
                    "item": "largest source of uncertainty of the mean",
                    "where": stratum,
                    "value": float(first["share"]),
                    "statement": (
                        f"{100 * first['share']:.0f}% of the variance of the mean is "
                        f"{words.get(first['level'], first['level'])}{caveat}"
                    ),
                    "source": "effort_projection",
                }
            )
    return rows


def _redundancy_rows(redundancy: pd.DataFrame | None) -> list[dict]:
    """The distance over which neighbouring fields are alike, when it was measured."""
    if redundancy is None or not len(redundancy):
        return []
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
        return []
    return [
        {
            "item": "spatial correlation",
            "where": "all groups",
            "value": float(len(alike)),
            "statement": statement,
            "source": "redundancy_distance",
        }
    ]


def _new_state_rows(
    inspection: InspectionResult | None, controls: ControlResult | None
) -> list[dict]:
    """How many fields hold the new states, beside what the reference's false alarms expect."""
    if inspection is None or not len(inspection.fovs):
        return []
    new = inspection.fovs[inspection.fovs["reason"] == "new state"]
    if not len(new):
        return []
    keys = [c for c in inspection.settings.get("fov_key", []) if c in new.columns]
    count = new[keys].drop_duplicates().shape[0]
    statement = f"new states come from {count} fields"
    false_alarms = _false_alarm_rate(controls)
    n_fovs = inspection.settings.get("n_fovs")
    if false_alarms is not None and n_fovs:
        # The same definition, judged the same way (each field against every
        # other), called that share of reference fields new.
        expected = false_alarms * n_fovs
        statement += (
            f" (about {expected:.0f} of {n_fovs} expected by chance at the "
            f"{100 * false_alarms:.0f}% false-alarm rate of the declared reference, each "
            "field judged against every other)"
        )
    return [
        {
            "item": "new states",
            "where": "all groups",
            "value": float(count),
            "statement": statement,
            "source": "inspect_fovs",
        }
    ]


def sampling_profile(
    controls: ControlResult | None,
    effort: EffortResult | None,
    redundancy: pd.DataFrame | None,
    inspection: InspectionResult | None,
    variability: FOVVariability | None = None,
    clustering: FOVClustering | None = None,
) -> pd.DataFrame:
    """The measured quantities of question 3, one row each with its source."""
    units = [
        r.settings.get("observation_unit")
        for r in (variability, clustering, controls, effort)
        if r is not None and r.settings.get("observation_unit")
    ]
    observations = unit_plural(units[0] if units else "object")
    rows = [
        *_variability_rows(variability, observations),
        *_clustering_rows(clustering, observations),
        *_representation_rows(controls, observations),
        *_effort_rows(effort, variability, observations),
        *_redundancy_rows(redundancy),
        *_new_state_rows(inspection, controls),
    ]
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


__all__ = ["PROFILE_COLUMNS", "sampling_profile"]
