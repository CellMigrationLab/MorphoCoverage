"""Question 9's status in "At a glance" (``conditions_status``; the rule is in
``report_status``'s module docstring, which lists every question's rule)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._style import Status, span
from .._wording import count
from ..sampling_unit._units import opposite_signs
from ..sampling_unit.contrast_text import (
    P_ALPHA,
    across_headline,
    across_tested,
    design_limited,
)

Answer = tuple[Status, str]


_UNTESTED = {
    "only one technical repeat": "one technical repeat on a side",
    "too few technical repeats": "too few technical repeats on a side",
    "fewer than 2 fields": "too few fields on a side",
    "fields of one dish or well": "the fields' wells unknown",
    "no control group": "no control fields",
}


def _untested_words(statuses: Sequence[str]) -> str:
    """Why contrasts have no test, in a few words."""
    words = {
        next((short for start, short in _UNTESTED.items() if s.startswith(start)), s)
        for s in statuses
    }
    return words.pop() if len(words) == 1 else "too few units on a side"


def conditions_status(contrast: Any, ranking: Any) -> Answer:
    """9. Do the conditions differ, and how confidently? (see the module)."""
    if contrast is None and ranking is None:
        return Status.NOT_ASSESSED, "not assessed"
    status, parts, limited = Status.OK, [], False
    if contrast is not None and len(contrast.table):
        table = contrast.table
        across = getattr(contrast, "repeats", None)
        answered = across_tested(contrast)
        if across is not None and len(across):
            parts.append(across_headline(across, contrast.settings["repeat_col"]))
        else:
            sizes = table["d_fields"].abs().dropna()
            size = f", |d| on field means {span(sizes)}" if len(sizes) else ""
            parts.append(f"{count(len(table), 'contrast')}{size}")
        tested = table[table["status"] == "ok"]
        untested = table[table["status"] != "ok"]
        within = "within repeats: " if across is not None and len(across) else ""
        if len(tested):
            differ = int((tested["p_bonferroni"] <= 0.05).sum())
            parts.append(f"{within}{differ} of {len(tested)} tested differ (Bonferroni p ≤ 0.05)")
        limited = bool(len(untested)) and bool(untested["status"].map(design_limited).all())
        if len(untested) and not answered:
            # A test the design cannot run is left out, not flagged (see the module).
            if not limited:
                status = Status.CAUTION
            elif not len(tested):
                status = Status.NOT_ASSESSED
            which = "not tested" if not len(tested) else f"{len(untested)} not tested"
            parts.append(f"{within}{which} ({_untested_words(list(untested['status']))})")
        passing = tested[tested["p_value"] <= P_ALPHA]
        if opposite_signs(passing, ["condition"], contrast.settings["repeat_col"], str):
            status = Status.CAUTION
            parts.append("opposite signs between repeats")
    elif contrast is not None:
        parts.append("no condition to compare with the control")
    if ranking is not None and len(ranking.ranking):
        passing = int(ranking.ranking["score"].notna().sum())
        parts.append(f"{passing} of {count(len(ranking.ranking), 'feature')} pass the gate")
        if not ranking.ranking["tests"].any() and not (
            limited or (contrast is not None and across_tested(contrast))
        ):
            status = Status.CAUTION
    return status, "; ".join(parts)


__all__ = ["conditions_status"]
