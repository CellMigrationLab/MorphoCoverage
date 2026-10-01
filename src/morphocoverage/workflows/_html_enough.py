""" "Did I image enough?" on the HTML page over every feature (private; ``html_report``).

A summary first, in sentences and in the table across the features (``_enough_features``):
for which features the fields imaged are enough to compare with the control, partly or not
yet, and for which each well's own mean meets the precision asked. Then one feature at a
time, as tabs in the order of "Do the conditions differ?" (significant across repeats
first): its comparisons within single repeats, what more would show the others, what to do
across repeats, and its precision. The stopping rules on the readout follow, as optimise and
live run them. Each tab says when it is enough, not only when it is not.
"""

from __future__ import annotations

import html
import re
from typing import Any

from .._wording import count, listing
from ..retrospective.report_resolved import resolved_parts
from ._enough_features import ENOUGH, NOT_YET, PARTLY, FeatureEnough
from ._html_render import text_html
from ._html_sections import _resolution
from ._html_tabs import tabs_html

_MARKS = {ENOUGH: "✅", PARTLY: "⚠️", NOT_YET: "⚠️"}
# The summary table's rows shown before "Show all" (the style hides the others).
_FIRST_ROWS = 20
_STOPPING = "  The stopping rules, as optimise and live run them (on the readout):"


def _names(rows: Any, verdict: str) -> list[str]:
    return list(rows.loc[rows["verdict"] == verdict, "feature"].astype(str))


def summary_lines(features: FeatureEnough, word: str, fields: str) -> list[str]:
    """The answer over the features in sentences: enough, partly, not yet; each well's mean."""
    rows = features.table
    parts = []
    for verdict, said in (
        (ENOUGH, "enough for {} (every comparison shown)"),
        (PARTLY, "partly for {}"),
        (NOT_YET, "not yet for {}"),
    ):
        names = _names(rows, verdict)
        if names:
            parts.append(said.format(listing(names, 6)))
    lines = [
        f"Enough {fields} to compare with the control, within single repeats? "
        + "; ".join(parts)[:1].upper()
        + "; ".join(parts)[1:]
        + "."
    ]
    precise = rows[rows["precision_met"].notna()]
    if len(precise):
        every = list(
            precise.loc[precise["precision_met"] == precise["precision_groups"], "feature"]
        )
        rule = precise["precision_rule"].iloc[0]
        met = f"met in every {word} for {listing(every, 6)}" if every else f"met in no {word}"
        lines.append(
            f"Each {word}'s own mean to the precision asked ({rule}): {met}, of "
            f"{count(len(precise), 'feature')}. A stricter goal than the comparison: it "
            f"matters when the value of a mean is reported, not for comparing conditions."
        )
    lines.append(
        "One feature at a time below, in the order of “Do the conditions differ?” "
        "(significant across repeats first)."
    )
    return lines


def _precision_line(features: FeatureEnough, feature: str, word: str, fields: str) -> str:
    tally = features.precision.get(feature)
    if not tally:
        return ""
    met, groups = tally["met"], tally["groups"]
    line = (
        f"Enough to pin down each {word}'s own mean of {feature}? The {tally['label']} was met "
        f"in {met} of {count(groups, word)}"
    )
    if met == groups:
        return line + ": enough."
    return line + (f". A stricter goal than the comparison; more {fields} per {word} is the lever.")


def _panel(features: FeatureEnough, feature: str, word: str, fields: str) -> str:
    contrast = features.contrasts[feature]
    lines = [*_resolution(contrast), _precision_line(features, feature, word, fields)]
    reading = "".join(f"<p>{html.escape(line)}</p>" for line in lines if line)
    table, _ = resolved_parts(contrast, indent=0)
    return f'<div class="reading">{reading}</div>' + text_html("\n".join(table))


def enough_body(
    features: FeatureEnough, short: str, word: str, fields: str, annotate: Any, unnumbered: Any
) -> str:
    """The card's body (see the module); ``short`` is the section's short text, its table
    across the features first and the stopping rules after; ``annotate`` links the
    glossary's words, ``unnumbered`` gives question numbers this page's titles."""
    head, _, rest = short.partition(f"\n\n{_STOPPING}")
    # The sentences above say what the table's intro line says.
    head = re.sub(r"^\s*Each feature alike.*?first\):\n", "", head, flags=re.DOTALL)
    summary = "".join(
        f"<p>{html.escape(line)}</p>" for line in summary_lines(features, word, fields)
    )
    rows = features.table.to_dict("records")
    items = [
        (r["feature"], _MARKS[r["verdict"]], annotate(_panel(features, r["feature"], word, fields)))
        for r in rows
    ]
    shown = text_html(head)
    if len(rows) > _FIRST_ROWS:
        # Hundreds of features: the first rows, sortable over them all, and a button.
        shown = (
            f'<div class="long-table">{shown}<button type="button" class="show-all" '
            f'data-rows="{_FIRST_ROWS}" aria-expanded="false">Show all {len(rows)} features'
            "</button></div>"
        )
    body = (
        f'<div class="reading">{annotate(summary)}</div>{shown}'
        "<h3>One feature at a time</h3>" + tabs_html("enough", items, "Features")
    )
    if rest.strip():
        body += "<h3>The stopping rules, as optimise and live run them (on the readout)</h3>"
        body += annotate(unnumbered(text_html(rest)))
    return body


__all__ = ["enough_body", "summary_lines"]
