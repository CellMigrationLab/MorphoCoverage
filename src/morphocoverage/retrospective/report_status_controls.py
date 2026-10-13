"""The status of question 1, "Can these features detect what I care about?" (private,
re-exported by ``report_status``, whose module text gives the rule).

❌ weak representation when the declared positive control (the planted objects when
declared) was recognized in fewer than ``minimum_detection`` of checks; ➖ "cannot tell the
positive control from the reference with this definition of new", not judged, when it was
recognized no more often than the reference's false alarms (owner, 2026-10-06, audit 10:
a definition of new that never calls a field of 4 cells new read as a weak representation
of features that show the phenotype); ⚠️ when a stratum could not be judged, the false-alarm
rate was not assessed or is above ``FALSE_ALARM_CAUTION``; ➖ without a control; ✅
otherwise. Split from ``report_status`` to keep it under 600 lines.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._style import Status
from .._wording import percent
from ..sampling_unit._group_text import selector_values
from ..schema import observation_words
from .controls_text import pooled_over
from .report_sampling import skipped_strata

# Above this false-alarm rate of the declared reference, the control answer asks the reader
# to look (a provisional preset, as minimum_detection).
FALSE_ALARM_CAUTION = 0.1


def _rate(controls: Any, kind: str) -> float | None:
    rates = controls.pooled.set_index("candidate")["rate"]
    return None if kind not in rates.index or pd.isna(rates[kind]) else float(rates[kind])


def _not_judged(controls: Any) -> tuple[Status, str] | None:
    """The answer without a control, or with one that could not be judged; else None."""
    if controls is None:
        return Status.NOT_ASSESSED, "no positive control declared"
    if controls.status != "not assessed":
        return None
    declared = controls.settings.get("reason") != "none"
    if declared:
        return Status.CAUTION, "a control was declared, but could not be judged"
    return Status.NOT_ASSESSED, "no positive control declared"


def _recognized(
    controls: Any, what: str, detection: float, false_alarm: float | None, pooled: str
) -> tuple[Status, str]:
    """The control recognized well enough: the headline leads with the positive control
    itself, then the false alarms and the strata that could not be judged."""
    over = f", {pooled}" if pooled else ""
    control = controls.settings.get("control") or {}
    # The headline leads with the positive control itself; the planted rate gates the status.
    positive = _rate(controls, "positive")
    name = selector_values(control.get("positive")) or "positive-control"
    headline = (
        f"{name} fields recognized in {percent(positive)} of checks"
        if positive is not None
        else f"{what} recognized in {percent(detection)} of checks"
    )
    status = Status.OK
    if false_alarm is None:
        status, headline = Status.CAUTION, f"{headline}{over}; false alarms not assessed"
    else:
        headline += f"; false alarms {percent(false_alarm)}{over and f', both {pooled}'}"
        if false_alarm > FALSE_ALARM_CAUTION:
            status = Status.CAUTION
    if controls.settings.get("skipped_strata"):
        status = Status.CAUTION
        headline += f"; {skipped_strata(controls)}"
    return status, headline


def controls_status(controls: Any, minimum_detection: float = 0.9) -> tuple[Status, str]:
    """1. Can these features detect what I care about? (see the module)."""
    unjudged = _not_judged(controls)
    if unjudged is not None:
        return unjudged
    words = observation_words(controls.settings)[1]
    control = controls.settings.get("control") or {}
    planted = control.get("planted_objects")
    kind = "planted" if planted and _rate(controls, "planted") is not None else "positive"
    detection = _rate(controls, kind)
    what = f"{planted} planted {words}" if kind == "planted" else "positive-control fields"
    if detection is None:
        return Status.CAUTION, f"{what}: not judged"
    # The rates are pooled over the control's strata, and the headline says so (#253).
    pooled = pooled_over(controls)
    over = f", {pooled}" if pooled else ""
    false_alarm = _rate(controls, "reference")
    if false_alarm is not None and detection <= false_alarm:
        # Not a weak representation: this definition of new tells nothing apart here (C1).
        rates = f"{percent(detection)} of checks, false alarms {percent(false_alarm)}{over}"
        cannot = "cannot tell the positive control from the reference with this definition of new"
        return Status.NOT_ASSESSED, f"{cannot}: {what} recognized in {rates}; not judged"
    if detection < float(minimum_detection):
        return (
            Status.PROBLEM,
            f"weak representation: {what} recognized in {percent(detection)} of checks{over}",
        )
    return _recognized(controls, what, detection, false_alarm, pooled)


__all__ = ["FALSE_ALARM_CAUTION", "controls_status"]
