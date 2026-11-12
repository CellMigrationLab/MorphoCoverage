"""Conditions read by their names, for the design scan of undescribed tables (private; split
out of ``workflows.scan``): a condition named like a negative control (siCTRL, MOCK, DMSO) is
suggested as the control, never used, and conditions named like reagents of one target
(siCCT8#5, siCCT8#8) are suggested as a reagent group. A described export reads neither from
the names: its description gives the controls and suggests the targets (MC-2,
``workflows._described``).

Shared-private: used by ``workflows.scan``, which re-exports ``reagent_groups``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

# A control named as labs name it: siCTRL, Control, MOCK, DMSO, NT, scramble, vehicle.
_CONTROL_WORDS = re.compile(
    r"^(ctrl|control|mock|dmso|untreated|scr|scramble|scrambled|nt|nontargeting|non_targeting|"
    r"vehicle|neg|negative|empty|luc|gfp)$"
)
# Two reagents of one target: siCCT8#5 and siCCT8#8, shRNA_1 and shRNA_2.
_REAGENT = re.compile(r"^(?P<stem>.+?)[#_\-. ](?P<number>\w*\d\w*)$")
_PREFIX = re.compile(r"^(si|sh|sg)(?=[A-Z0-9])")


def control_guess(conditions: Sequence[Any]) -> list[Any]:
    """The conditions whose name reads like a negative control (siCTRL, MOCK, DMSO)."""
    guesses = []
    for condition in conditions:
        name = _PREFIX.sub("", str(condition).strip())
        words = [w for w in re.split(r"[^A-Za-z0-9]+", name.lower()) if w]
        if any(_CONTROL_WORDS.match(w) for w in words) or _CONTROL_WORDS.match(
            name.lower().replace(" ", "").replace("-", "")
        ):
            guesses.append(condition)
    return guesses


def reagent_groups(conditions: Sequence[Any], control: Any = None) -> dict[str, list[Any]]:
    """Conditions that look like several reagents of one target, by name: ``siCCT8#5`` and
    ``siCCT8#8`` give ``{"CCT8": ["siCCT8#5", "siCCT8#8"]}``. Only groups of two or more,
    never the control."""
    groups: dict[str, list[Any]] = {}
    for condition in conditions:
        if condition == control:
            continue
        match = _REAGENT.match(str(condition).strip())
        if match is None:
            continue
        target = _PREFIX.sub("", match["stem"])
        groups.setdefault(target, []).append(condition)
    return {t: members for t, members in groups.items() if len(members) > 1}


__all__ = ["control_guess", "reagent_groups"]
