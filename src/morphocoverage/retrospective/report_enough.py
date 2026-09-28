"""Question 2 of the microscopist's report, "Did I image enough?", short and full.

The section puts three results together: what each rule says by its own
threshold and whether one more field would show something new
(``sufficiency_report``), the unseen-state bound adjusted for the declared
control's detection rate (``report_profile``), and how often each rule's stop
preserved the criterion in held-back fields (``stopping_calibration``).

- ``enough_short``: the rules in one table across the analysis groups, the
  novelty answers in a few lines (``sufficiency_short``), the adjusted bound
  in one line, the held-back check in one table per criterion
  (``calibration.calibration_short_parts``), then every caveat of the three,
  once each, as ⚠️ lines at the end.
- ``enough_full``: every sentence of the three summaries, restyled (groups by
  their values, rules by their readable names with their config ids).
"""

from __future__ import annotations

from typing import Any

from .._style import caution_lines, span, wrap
from .._wording import percent
from ..bounds import lower_bound
from ..sufficiency_short import short_parts
from .calibration import calibration_short_parts
from .report_profile import _CONFIDENCE, _detection, _detection_adjusted_bounds


def _span(values: list[float]) -> str:
    return span(values, "percent")


def detection_line(sufficiency: Any, controls: Any) -> tuple[str | None, list[str]]:
    """The unseen-state bound adjusted for the control's detection, in one line, and its
    caveats; or the sentence saying why it is not adjusted (``_detection_adjusted_bounds``)."""
    text = _detection_adjusted_bounds(sufficiency, controls)
    if text is None:
        return None, []
    if not text.startswith("Adjusted"):
        return text, []
    kind, recognized, checks = _detection(controls)
    frame = sufficiency.next_fov
    confidence = float(frame["confidence"].iloc[0]) if "confidence" in frame else _CONFIDENCE
    low = lower_bound(recognized, checks, confidence)
    bounds = [float(b) for b in frame["unseen_state_max_frequency"].dropna()]
    adjusted = [min(1.0, bound / low) for bound in bounds]
    if kind == "planted_lofo":
        planted = controls.settings["control"]["planted_objects"]
        what = f"{planted} planted positive-control objects"
        cautions = []
    else:
        what = "positive-control fields"
        cautions = [
            (
                "the detection-adjusted bound holds for whole-field phenotypes like the control "
                "only: a small state in an ordinary field is recognized less often (declare "
                "planted_objects to measure it)"
            )
        ]
    cautions.append("the detection-adjusted bound assumes detection is independent between fields")
    line = (
        f"Adjusted for detection ({what} recognized in {percent(recognized / checks)} of "
        f"{checks} checks, at least {percent(low)} at {percent(confidence)} confidence): a "
        f"phenotype seen as well as the control, and shown by no field yet, could still be in "
        f"up to {_span(adjusted)} of fields (the {_span(bounds)} bound divided by "
        f"{percent(low)})."
    )
    return line, cautions


def enough_short(sufficiency: Any, calibration: Any, controls: Any, indent: int = 2) -> str:
    """The short form of question 2 (see the module), under ``indent``."""
    lines: list[str] = []
    cautions: list[str] = []
    if sufficiency is not None:
        own, own_cautions = short_parts(sufficiency, indent=indent)
        lines += own
        cautions += own_cautions
        adjusted, adjusted_cautions = detection_line(sufficiency, controls)
        if adjusted:
            lines.append(wrap(adjusted, indent, hang=indent + 2))
            cautions += adjusted_cautions
    if calibration is not None:
        held, held_cautions = calibration_short_parts(calibration, indent=indent)
        lines += held
        cautions += held_cautions
    return "\n".join([*lines, *caution_lines(cautions, indent=indent)])


def enough_full(sufficiency: Any, calibration: Any, controls: Any) -> list[str]:
    """The full form of question 2: each summary's every sentence, one block each."""
    blocks = []
    if sufficiency is not None:
        blocks.append(sufficiency.summary(detail="full"))
        adjusted = _detection_adjusted_bounds(sufficiency, controls)
        if adjusted:
            blocks.append(adjusted)
    if calibration is not None:
        blocks.append(calibration.summary(detail="full"))
    return blocks


__all__ = ["detection_line", "enough_full", "enough_short"]
