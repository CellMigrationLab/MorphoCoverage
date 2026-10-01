"""Question 2 of the microscopist's report, "Did I image enough?", short and full.

The section puts three results together: what each rule says by its own
threshold and whether one more field would show something new
(``sufficiency_report``), the unseen-state bound adjusted for the declared
control's detection rate (``report_profile``), and how often each rule's stop
preserved the criterion in held-back fields (``stopping_calibration``).

With a control and other conditions, the section opens with whether the
fields acquired resolve the difference from the control (``report_resolved``);
the precision of each group's own mean follows.

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

from .._caveats import assumption, caveat_lines, is_assumption
from .._style import span, wrap
from .._wording import percent
from ..bounds import lower_bound
from ..sufficiency_short import short_parts
from .calibration import calibration_short_parts
from .report_profile import _CONFIDENCE, _detection, _detection_adjusted_bounds, _words
from .report_resolved import has_resolution, resolved_full, resolved_parts


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
        what = f"{planted} planted positive-control {_words(controls)}"
        cautions = []
    else:
        what = "positive-control fields"
        # No planted objects declared: a measurement not made on these data, so a caution.
        cautions = [
            (
                "the detection-adjusted bound holds for whole-field phenotypes like the control "
                "only: a small state in an ordinary field is recognized less often (declare "
                "planted_objects to measure it)"
            )
        ]
    cautions.append(
        assumption("the detection-adjusted bound assumes detection is independent between fields")
    )
    line = (
        f"Adjusted for detection ({what} recognized in {percent(recognized / checks)} of "
        f"{checks} checks, at least {percent(low)} at {percent(confidence)} confidence): a "
        f"phenotype seen as well as the control, and shown by no field yet, could still be in "
        f"up to {_span(adjusted)} of fields (the {_span(bounds)} bound divided by "
        f"{percent(low)})."
    )
    return line, cautions


def enough_short(
    sufficiency: Any, calibration: Any, controls: Any, indent: int = 2, contrast: Any = None
) -> str:
    """The short form of question 2 (see the module), under ``indent``; with a contrast that
    carries a resolution answer, that answer first (``report_resolved``)."""
    lines: list[str] = []
    cautions: list[str] = []
    if has_resolution(contrast):
        lines, cautions = resolved_parts(contrast, indent=indent)
        if sufficiency is not None:
            lines.append(wrap("The precision of each group's own mean:", indent, hang=indent + 2))
    if sufficiency is not None:
        own, own_cautions = short_parts(sufficiency, indent=indent)
        lines += own
        if has_resolution(contrast):
            # The precision of each mean is the secondary answer here: its own cautions are
            # in the full report, so they do not stand for the answer to the question.
            if any(not is_assumption(c) for c in own_cautions):
                lines.append(
                    wrap(
                        "The cautions on these rules are in the full report.",
                        indent,
                        hang=indent + 2,
                    )
                )
            own_cautions = [c for c in own_cautions if is_assumption(c)]
        cautions += own_cautions
        adjusted, adjusted_cautions = detection_line(sufficiency, controls)
        if adjusted:
            lines.append(wrap(adjusted, indent, hang=indent + 2))
            cautions += adjusted_cautions
    if calibration is not None:
        held, held_cautions = calibration_short_parts(calibration, indent=indent)
        lines += held
        cautions += held_cautions
    return "\n".join([*lines, *caveat_lines(cautions, indent=indent)])


def enough_full(
    sufficiency: Any, calibration: Any, controls: Any, contrast: Any = None
) -> list[str]:
    """The full form of question 2: each summary's every sentence, one block each (the
    resolution of the difference from the control first, when there is one)."""
    blocks = [resolved_full(contrast)] if has_resolution(contrast) else []
    if sufficiency is not None:
        blocks.append(sufficiency.summary(detail="full"))
        adjusted = _detection_adjusted_bounds(sufficiency, controls)
        if adjusted:
            blocks.append(adjusted)
    if calibration is not None:
        blocks.append(calibration.summary(detail="full"))
    return blocks


__all__ = ["detection_line", "enough_full", "enough_short"]
