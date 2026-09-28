"""Questions 4, 5 and 7 of the microscopist's report in their short form.

- **4. How many fields should I image next time?** The protocol across
  samples first (``protocol_text.protocol_short_parts``: one table, a row per
  criterion), then the per-group curves (``reliability_text``: one table
  across groups), then the caveats of both, once each.
- **5. What did the extra fields buy?** When held-back fields confirmed
  everything wanted, and when each rule would have said stop, across groups
  (``adequacy_text.adequacy_short_parts``).
- **7. More fields or more samples?** One table over the strata
  (``effort_text.effort_short_parts``).

Each function returns the section's text under ``indent`` (the report
indents its sections by 2), or ``None`` without its results; the full forms
are each result's ``summary(detail="full")``.
"""

from __future__ import annotations

from typing import Any

from .._style import caution_lines
from .adequacy_text import adequacy_short_parts
from .effort_text import effort_short_parts
from .protocol_text import protocol_short_parts
from .reliability_text import reliability_short_parts


def _join(parts: list[tuple[list[str], list[str]]], indent: int) -> str | None:
    """The parts' lines, a blank line between parts, then every caveat of all of them once."""
    blocks = [lines for lines, _ in parts if lines]
    cautions = [text for _, own in parts for text in own]
    if not blocks and not cautions:
        return None
    lines = []
    for block in blocks:
        lines += ([""] if lines else []) + block
    return "\n".join([*lines, *caution_lines(cautions, indent=indent)])


def next_time_short(reliability: Any, protocol: Any, indent: int = 2) -> str | None:
    """Question 4, short: the protocol table, the per-group curves, the caveats once."""
    parts = []
    if protocol is not None:
        parts.append(protocol_short_parts(protocol, indent=indent))
    if reliability is not None:
        parts.append(reliability_short_parts(reliability, indent=indent))
    return _join(parts, indent) if parts else None


def bought_short(adequacy: Any, indent: int = 2) -> str | None:
    """Question 5, short: held-back confirmation and the rules' stops across groups."""
    return None if adequacy is None else _join([adequacy_short_parts(adequacy, indent)], indent)


def effort_short(effort: Any, indent: int = 2) -> str | None:
    """Question 7, short: one table over the strata, the caveats once."""
    return None if effort is None else _join([effort_short_parts(effort, indent)], indent)


__all__ = ["bought_short", "effort_short", "next_time_short"]
