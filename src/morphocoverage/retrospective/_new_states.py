"""How many fields hold the new states of question 6, beside what chance alone would give
(private; the sampling profile's row, ``report_profile``).

The count chance gives comes from the declared reference's false-alarm rate when a positive
control is declared and judged as ``inspect_fovs`` judges a field (``reference_lofo``), from
its point estimate to its one-sided upper bound; without one, from the definition of new's
own null rate at each judged group's size (``strategies.novelty_null``), its expected count
and the 95th percentile of a binomial count at that rate. The sentence says whether the
fields found are compatible with chance (owner, 2026-10-06, audit 10: "New states come from
4 fields; false alarms alone would give about 8, up to 25, so these are compatible with
chance"; before, "8 to 24.5 of the 336 fields judged expected by chance", and nothing
without a positive control, where 18 null fields of 150 read as findings).

Shared-private: used by ``report_profile``.
"""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._wording import number, percent
from ..bounds import upper_bound
from ._control_rates import _CONFIDENCE, _false_alarm_rate, _mismatch
from .controls import ControlResult
from .controls_text import pooled_over
from .inspection import InspectionResult


def _reference(
    controls: ControlResult, inspection: InspectionResult, judged: int
) -> tuple[float, float, str] | str:
    """``(expected, at most, where the rate comes from)`` from the declared reference's
    false alarms, or why they are not compared."""
    rate = _false_alarm_rate(controls)
    assert rate is not None
    settings: dict[str, Any] = inspection.settings
    different = _mismatch(
        controls, settings.get("novelty_definition"), getattr(inspection, "group_cols", None)
    )
    if different is not None:
        return f" (not compared with the declared reference's false alarms: {different})"
    row = controls.pooled.set_index("candidate").loc["reference_lofo"]
    # One rate for the fields of every group: pooled over the control's strata, said so (#253).
    over = pooled_over(controls)
    pooled = f", {over}," if over else ","
    source = (
        f"the declared reference's false-alarm rate{pooled} each field judged against the other "
        "fields of its group"
    )
    if "judged_new" not in row.index or "candidates" not in row.index:
        return rate * judged, rate * judged, f"{source}, {percent(rate)}"
    new, total = int(row["judged_new"]), int(row["candidates"])
    high = upper_bound(new, total, _CONFIDENCE)
    bound = f"at most {percent(high)} at {percent(_CONFIDENCE)} confidence"
    return (
        rate * judged,
        high * judged,
        f"{source}, was {new} of {total} fields, {percent(rate)}, {bound}",
    )


def _null(inspection: InspectionResult) -> tuple[float, float, str, int] | None:
    """``(expected, at most, where it comes from, fields)`` from the definition's own null
    rate at each judged group's size, or ``None`` when no group has one."""
    rated = [g for g in inspection.settings.get("novelty_null") or [] if g["rate"] is not None]
    fields = sum(int(g["fields"]) for g in rated)
    if not fields:
        return None
    expected = sum(float(g["rate"]) * int(g["fields"]) for g in rated)
    most = float(stats.binom.ppf(_CONFIDENCE, fields, expected / fields))
    source = (
        "this definition of new on a homogeneous population of the same sizes, the features "
        "independent and Gaussian"
    )
    return expected, most, source, fields


def chance_words(found: int, controls: ControlResult | None, inspection: InspectionResult) -> str:
    """``; false alarms alone would give about 8, up to 25, of the 336 fields judged, so these
    are compatible with chance: <where the rate comes from>``, or empty (see the module)."""
    settings: dict[str, Any] = inspection.settings
    judged = settings.get("n_fovs_judged", settings.get("n_fovs"))
    if not judged:
        return ""
    if _false_alarm_rate(controls) is not None:
        assert controls is not None
        reference = _reference(controls, inspection, int(judged))
        if isinstance(reference, str):
            return reference
        expected, most, source = reference
    else:
        null = _null(inspection)
        if null is None:
            return ""
        expected, most, source, judged = null
    verdict = "compatible with chance" if found <= most else "more than chance alone gives"
    return (
        f"; false alarms alone would give about {number(expected)}, up to {number(most)}, of "
        f"the {judged} fields judged, so these are {verdict}: {source}"
    )


def new_state_rows(
    inspection: InspectionResult | None, controls: ControlResult | None
) -> list[dict]:
    """How many fields hold the new states, beside what chance alone would give."""
    if inspection is None or not len(inspection.fovs):
        return []
    new = inspection.fovs[inspection.fovs["reason"] == "new state"]
    if not len(new):
        return []
    keys = [c for c in inspection.settings.get("fov_key", []) if c in new.columns]
    found = new[keys].drop_duplicates().shape[0]
    statement = f"new states come from {found} fields" + chance_words(found, controls, inspection)
    return [
        {
            "item": "new states",
            "where": "all groups",
            "value": float(found),
            "statement": statement,
            "source": "inspect_fovs",
        }
    ]


__all__ = ["chance_words", "new_state_rows"]
