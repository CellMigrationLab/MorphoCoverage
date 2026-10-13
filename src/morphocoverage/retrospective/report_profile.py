"""The sampling profile of the microscopist's report, and the detection-adjusted bound.

Kept apart from the assembly of the report (``report``). The profile lists only measured
quantities, each with its numbers and its source; one helper per source keeps each rule
readable.

Rates that compare with ``inspect_fovs`` or with the unseen-state bound of the sufficiency
report are the control's leave-one-field-out rates (``reference_lofo``, ``positive_lofo``),
measured within the same analysis groups: those analyses judge each field against the other
fields of its group, and a rate measured after only ``fovs_seen`` reference fields can be many
times larger (41% against 0% on one plate). A rate is combined with another analysis only when
both used the same definition of new, on the same features, within the same analysis groups;
otherwise the report says why it was not.

Shared-private: ``_detection_adjusted_bounds`` (used by ``report_enough``); the control's
rates are in ``_control_rates`` and the new states' row in ``_new_states``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._style import low_to_high
from .._wording import count, percent
from ..bounds import lower_bound
from ..sampling_unit._state_text import CLUSTER_ALPHA
from ..sampling_unit._state_text import clustered as clustered_rows
from ..sampling_unit._unit_text import one_per_field_rows
from ..schema import observation_words
from ..sufficiency import SufficiencyReport
from ._control_rates import _CONFIDENCE, _control_rate, _mismatch
from ._new_states import new_state_rows as _new_state_rows
from .controls import ControlResult
from .controls_text import pooled_over
from .effort import EffortResult, design_effect
from .effort_text import level_words, share_caveat, unit_word
from .inspection import InspectionResult

if TYPE_CHECKING:  # the report takes these as results; only their types are needed here
    from ..sampling_unit import FOVClustering, FOVVariability

PROFILE_COLUMNS = ["item", "where", "value", "statement", "source"]


def _detection(controls: ControlResult) -> tuple[str, int, int] | None:
    """The detection rate the unseen-state bound is adjusted by: kind, recognized, checks.

    With ``planted_objects`` declared, the rate of planted objects in ordinary
    reference fields (``planted_lofo``): the bound is about a state not yet
    seen, which a whole-field phenotype does not stand for (P6-4). Otherwise,
    or when no planting could be judged, the whole positive fields
    (``positive_lofo``).
    """
    rates = controls.pooled.set_index("candidate")
    planted = (getattr(controls, "settings", {}) or {}).get("control", {}) or {}
    kinds = ["planted_lofo", "positive_lofo"] if planted.get("planted_objects") else []
    for kind in kinds or ["positive_lofo"]:
        if kind in rates.index and int(rates.loc[kind, "candidates"]) > 0:
            return kind, int(rates.loc[kind, "judged_new"]), int(rates.loc[kind, "candidates"])
    return None


def _words(controls: ControlResult) -> str:
    """The control's observation unit, plural: ``nuclei`` (``objects`` when none declared)."""
    return observation_words(controls.settings)[1]


def _detection_words(controls: ControlResult, kind: str, detection: float) -> tuple[str, str]:
    """The detection rate in words (planted objects, or whole positive fields), and what it
    holds for ('' for planted objects)."""
    if kind == "planted_lofo":
        planted = controls.settings["control"]["planted_objects"]
        what = (
            f"for {planted} planted positive-control {_words(controls)}: {percent(detection)} of "
            "ordinary "
            "reference fields with them recognized against the other reference fields of "
            "their analysis group"
        )
        return what, ""
    what = (
        f"{percent(detection)} of its fields recognized against the reference fields of an "
        "analysis group"
    )
    scope = (
        "  This holds for whole-field phenotypes like the control only: a small state in "
        "an ordinary field is recognized less often (declare planted_objects to measure it)."
    )
    return what, scope


def _adjusted_lines(frame: pd.DataFrame, keys: list[str], low: float) -> list[str]:
    """Each group's unseen-state bound divided by the low end of the detection rate."""
    lines = []
    for row in frame.itertuples(index=False):
        bound = row.unseen_state_max_frequency
        if bound is None or pd.isna(bound):
            continue
        adjusted = min(1.0, float(bound) / low)
        label = ", ".join(f"{k}={getattr(row, k)}" for k in keys) or "all FOVs (pooled)"
        lines.append(
            f"  {label}: up to {percent(adjusted)} of FOVs (the {percent(float(bound))} "
            f"bound above divided by {percent(low)}), if detection is independent "
            "between fields."
        )
    return lines


def _detection_adjusted_bounds(
    sufficiency: SufficiencyReport, controls: ControlResult | None
) -> str | None:
    """The unseen-state bound divided by the low end of the control's detection rate.

    The rule-of-three bound holds for states this definition would detect in
    every FOV that has them. A phenotype recognized like the declared control,
    in a share ``d`` of the fields that show it, is seen with probability
    ``d * f`` per FOV when present in a share ``f``, so the same run of FOVs
    with nothing new bounds ``f`` at ``bound / d`` (at most 1), assuming
    detection is independent between fields. ``d`` is judged as the bound
    judges fields, against the other fields of an analysis group: the rate
    of planted positive objects in ordinary reference fields
    (``planted_lofo``) when ``planted_objects`` is declared, else of whole
    positive fields (``positive_lofo``), which holds for whole-field
    phenotypes only, and the text says so. The bound is divided by the
    one-sided Clopper-Pearson lower bound of ``d`` at the bound's confidence,
    not by its point estimate, which with few control fields let the
    adjusted bound exceed its level (P3-9). It needs a positive control only,
    so it is given under "positive only" too, and it is not given when the
    sufficiency report records another definition of new or other groups.
    """
    if controls is None or controls.status == "not assessed" or sufficiency.next_fov.empty:
        return None
    different = _mismatch(
        controls,
        getattr(sufficiency, "novelty_definition", None),
        getattr(sufficiency, "group_cols", None),
    )
    if different is not None:
        return f"The unseen-state bound is not adjusted for detection: {different}."
    measured = _detection(controls)
    if measured is None:
        return (
            "The unseen-state bound is not adjusted for detection: the declared positive "
            "control could not be judged against the reference fields of an analysis group."
        )
    kind, recognized, checks = measured
    if recognized == 0:
        return (
            "The declared positive control was never recognized against the reference fields "
            "of an analysis group, so the unseen-state bound says nothing about a phenotype "
            "seen like it."
        )
    frame = sufficiency.next_fov
    confidence = float(frame["confidence"].iloc[0]) if "confidence" in frame else _CONFIDENCE
    detection, low = recognized / checks, lower_bound(recognized, checks, confidence)
    what, scope = _detection_words(controls, kind, detection)
    # One rate for every group: pooled over the control's strata, and said so (#253).
    pooled = pooled_over(controls)
    heading = (
        f"Adjusted for the declared control's detection rate ({what}; at least {percent(low)} "
        f"at {percent(confidence)} confidence, of {checks} checks{', ' if pooled else ''}"
        f"{pooled}): a phenotype seen as well as the control, and shown by no FOV yet, could "
        "still be in"
    )
    lines = [heading, *_adjusted_lines(frame, list(sufficiency.group_cols), low)]
    if len(lines) == 1:
        return None
    return "\n".join([*lines, scope] if scope else lines)


def _variability_rows(variability: FOVVariability | None, observations: str) -> list[dict]:
    """Fields against random objects, from ``fov_variability``."""
    if variability is None:
        return []
    ok = variability.groups[variability.groups["status"] == "ok"]
    if not len(ok):  # one observation per field: the fields are the observations
        return one_per_field_rows(variability, observations)
    effect = ok["design_effect"]
    shares = pd.to_numeric(ok["unusual_fov_share"], errors="coerce").dropna()
    unusual = (
        f"; {percent(shares.median())} of fields differ more than random "
        f"{observations} would (5% expected)"
        if len(shares)
        else ""
    )
    where = f"median of {len(ok)} groups" if len(ok) > 1 else "1 group"
    # The level named, since the effort projection's design effect is another estimand.
    level = (
        f"between the fields of each group ({', '.join(variability.group_cols)})"
        if variability.group_cols
        else "between all fields, pooled"
    )
    return [
        {
            "item": "fields against random objects",
            "where": where,
            "value": float(effect.median()),
            "statement": (
                f"a mean from whole fields varies {effect.median():.1f} times as much as "
                f"one from as many random {observations} (design effect of the variance "
                f"{level}, range {low_to_high(f'{effect.min():.1f}', f'{effect.max():.1f}')})"
                f"{unusual}"
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
        clustered = seen[clustered_rows(seen)]
        run = int(pd.to_numeric(seen["p_clustered"], errors="coerce").notna().sum())
        statement = (
            f"{clustering.state_name(state)}: clustered in a few fields "
            f"(q <= {CLUSTER_ALPHA:g}) in {len(clustered)} of the {run} groups where it could be "
            f"tested, of {len(seen)} where it was seen"
        )
        # Fields holding the state (at least m of its objects, for unusual labels);
        # a group where no field holds it has no field count to compare.
        factor = (
            pd.to_numeric(clustered["fovs_needed"], errors="coerce")
            / pd.to_numeric(clustered["fovs_needed_if_random"], errors="coerce")
        ).dropna()
        if len(factor):
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
    """How well the declared control is seen, and the reference's false alarms, by method."""
    if controls is None or controls.status == "not assessed":
        return []
    settings = getattr(controls, "settings", {}) or {}
    seen = settings.get("fovs_seen")
    after = f" after {seen} reference fields" if seen is not None else ""
    groups = settings.get("lofo_group_cols")
    within = f" ({', '.join(groups)})" if groups else ""
    rows = []
    for kind, item, words in (
        (
            "positive",
            "representation",
            f"declared positive-control fields recognized as different{after}",
        ),
        (
            "planted",
            "representation",
            f"fields with planted positive-control {observations} recognized{after}",
        ),
        ("reference", "representation", f"reference fields called different{after} (false alarms)"),
        (
            "reference_lofo",
            "representation, as inspect_fovs judges",
            (
                "reference fields called different against the other reference fields of "
                f"their analysis group{within} (the false-alarm rate of inspect_fovs)"
            ),
        ),
        (
            "positive_lofo",
            "representation, as inspect_fovs judges",
            (
                "checks of a positive-control field against the reference fields of an "
                f"analysis group{within} recognized as different"
            ),
        ),
        (
            "planted_lofo",
            "representation, as inspect_fovs judges",
            (
                f"reference fields with planted positive-control {observations} recognized "
                f"against the other reference fields of their analysis group{within}"
            ),
        ),
    ):
        rate = _control_rate(controls, kind)
        if rate is not None:
            rows.append(
                {
                    "item": item,
                    "where": "pooled",
                    "value": rate,
                    "statement": f"{percent(rate)} of {words}",
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
            level = words.get(hierarchy.get("fov"), "between fields")
            rows.append(
                {
                    "item": "field-to-field heterogeneity",
                    "where": stratum,
                    "value": effect,
                    "statement": (
                        f"a mean from whole fields varies {effect:.1f} times as much as one "
                        f"from the same number of random {observations} (design effect of the "
                        f"variance {level})"
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
                        f"{percent(first['share'])} of the variance of the mean is "
                        f"{words.get(first['level'], first['level'])}{caveat}"
                    ),
                    "source": "effort_projection",
                }
            )
    return rows


def _pairs(found: pd.DataFrame) -> str:
    """``, 12 to 40 pairs of fields closer than that per group``, when the pairs were counted."""
    if "pairs_closer" not in found.columns:
        return ""
    pairs = pd.to_numeric(found["pairs_closer"], errors="coerce").dropna()
    if not len(pairs):
        return ""
    low, high = int(pairs.min()), int(pairs.max())
    span = f"{low}" if low == high else f"{low} to {high}"
    return f", from {span} pairs of fields closer than that per group"


# Why a group has no spatial estimate, in a few words (``spatial.redundancy_distance``).
_NOT_MEASURED = (
    ("missing values", "missing values"),
    ("only ", "too few fields"),
    ("no pair", "no pair of fields in the distance bins"),
    ("the FOV means", "the field means do not vary"),
)


def _not_measured(rows: pd.DataFrame) -> str:
    """``; 1 more not assessed (too few fields)``, or '' when every group was assessed."""
    reasons: list[str] = []
    for text in rows.get("reason", pd.Series(dtype=object)).fillna("not measured").astype(str):
        short = next((words for start, words in _NOT_MEASURED if text.startswith(start)), text)
        reasons += [] if short in reasons else [short]
    return f"; {len(rows)} more not assessed ({'; '.join(reasons)})" if len(rows) else ""


def _alike_part(alike: pd.DataFrame, of: str, unit: str, tested: bool) -> list[str]:
    """The groups whose neighbouring fields are alike up to a distance, in words."""
    if not len(alike):
        return []
    tested_text = (
        "beyond what shuffling the positions within each group gives"
        if tested
        else "not tested against chance"
    )
    text = (
        f"in {len(alike)} {of}, fields closer than about "
        f"{np.median(alike['redundancy_distance']):.0f} {unit} (the median of point "
        f"estimates, {tested_text}{_pairs(alike)}) are more alike than "
        "random pairs"
    )
    return [text]


def _always_part(always: pd.DataFrame, of: str, unit: str) -> list[str]:
    """The groups whose fields are alike at every distance measured, in words."""
    if not len(always):
        return []
    largest = float(always["largest_lag"].max()) if "largest_lag" in always else None
    reach = f" (up to {largest:.0f} {unit})" if largest is not None else ""
    text = (
        f"in {len(always)} {of}, fields are more alike than "
        f"random pairs at every distance measured{reach}"
    )
    return [text]


def _chance_part(chance: pd.DataFrame | None, of: str) -> list[str]:
    """The groups whose point estimate suggested redundancy that the shuffles do not
    support, in words."""
    suggested = (
        chance[
            chance["null_share"].notna()
            & (
                chance["redundancy_distance"].isna()
                | (chance["redundancy_distance"] > chance["smallest_lag"])
            )
        ]
        if chance is not None
        else pd.DataFrame()
    )
    if not len(suggested):
        return []
    typical = float(suggested["null_share"].median())
    text = (
        f"in {len(suggested)} {of} the point estimate suggested "
        "that closer fields are more alike, but shuffling their positions within the "
        f"group gave as large a distance in a median of {percent(typical)} of shuffles, "
        "so no spatial redundancy is shown there"
    )
    return [text]


def _redundancy_rows(redundancy: pd.DataFrame | None) -> list[dict]:
    """The distance over which neighbouring fields are alike, when it was measured.

    Each group's distance is a point estimate from the semivariogram, with no
    test against chance (independent fields can dip below the variance at a
    short lag), so the statement says so and gives the pairs behind it.
    """
    if redundancy is None or not len(redundancy):
        return []
    measured = redundancy.dropna(subset=["smallest_lag"])
    # Groups with no estimate are counted apart, neither evidence for nor against (#272).
    unassessed, assessed = redundancy[redundancy["smallest_lag"].isna()], len(measured)
    of = (
        f"of the {count(assessed, 'group')} assessed"
        if len(unassessed)
        else f"of {len(redundancy)} groups"
    )
    apart = _not_measured(unassessed)
    # Only distances beyond the position-shuffle null count (audit 4, P8-2): without
    # spatial structure, a distance beyond the first bin arises in about 40% of groups.
    tested = "beyond_chance" in measured.columns
    chance = measured[~measured["beyond_chance"].fillna(False).astype(bool)] if tested else None
    if tested:
        measured = measured[measured["beyond_chance"].fillna(False).astype(bool)]
    always = measured[measured["redundancy_distance"].isna()]
    found = measured.dropna(subset=["redundancy_distance"])
    alike = found[found["redundancy_distance"] > found["smallest_lag"]]
    smallest = float(measured["smallest_lag"].min()) if len(measured) else None
    unit = str(redundancy["unit"].iloc[0]) if "unit" in redundancy.columns else "stage units"
    parts = [
        *_alike_part(alike, of, unit, tested),
        *_always_part(always, of, unit),
        *_chance_part(chance, of),
    ]
    if parts:
        statement = "; ".join(parts) + apart
    elif smallest is not None:
        among = f"in the {count(assessed, 'group')} assessed, " if len(unassessed) else ""
        statement = (
            f"{among}no group shows neighbouring fields more alike than random pairs, down to "
            f"the closest spacing measured ({smallest:.0f} {unit}){apart}"
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
        observation_words(r.settings)[1]
        for r in (variability, clustering, controls, effort)
        if r is not None and r.settings.get("observation_unit")
    ]
    observations = units[0] if units else observation_words({})[1]
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
