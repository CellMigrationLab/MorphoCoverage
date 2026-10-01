"""Tabs on the HTML page (private; ``_html_enough``, ``html_report``): one panel at a time,
the arrow keys moving between them (``_html_script``).

- ``tabs_html``: tabs from ``(label, mark, markup)`` items, the first one shown.
- ``tests_tabs``: "The tests behind the reading" one kind of test at a time: across the
  repeats, the reagents of a target, the cell-level screen, within single repeats, planning
  the next experiment, every feature. Text before the first kind stays above the tabs, and
  the section's cautions and assumptions below them, for every tab.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Sequence

# Each kind of test by the words its paragraph starts with in the text report.
TEST_KINDS = (
    ("Across repeats", ("Across the ",)),
    ("Reagents", ("Independent reagents",)),
    ("Cell-level screen", ("Cell-level screen",)),
    ("Within repeats", ("No test within a single repeat", "Within each repeat")),
    ("Planning", ("Planning the next experiment",)),
    ("Every feature", ("Which features separate the conditions?",)),
)
_CLOSING = ("⚠️", "Assumptions")


def tabs_html(name: str, items: Sequence[tuple[str, str, str]], label: str) -> str:
    """Tabs called ``name`` (an id prefix) from ``(label, mark, markup)`` items; ``label``
    names the tab list for a screen reader."""
    buttons = "".join(
        f'<button type="button" role="tab" id="{name}-tab-{i}" aria-controls="{name}-panel-{i}" '
        f'aria-selected="{str(i == 0).lower()}" tabindex="{0 if i == 0 else -1}">'
        f"{html.escape(text)}" + (f'<span class="mark">{mark}</span>' if mark else "") + "</button>"
        for i, (text, mark, _) in enumerate(items)
    )
    panels = "".join(
        f'<div class="tab-panel" role="tabpanel" id="{name}-panel-{i}" '
        f'aria-labelledby="{name}-tab-{i}"{"" if i == 0 else " hidden"}>{markup}</div>'
        for i, (_, _, markup) in enumerate(items)
    )
    return (
        f'<div class="feature-tabs"><div class="tabs" role="tablist" '
        f'aria-label="{html.escape(label)}">{buttons}</div>{panels}</div>'
    )


def _kind(line: str) -> str | None:
    text = line.strip()
    return next((name for name, starts in TEST_KINDS if text.startswith(starts)), None)


def tests_tabs(text: str, render: Callable[[str], str]) -> str:
    """The tests behind the reading as tabs (see the module); ``render`` turns text into
    markup. One kind or none found: the text as it is."""
    paragraphs = re.split(r"\n(?=  \S)", text)
    head, blocks, tail = [], {}, []
    current = None
    for paragraph in paragraphs:
        kind = _kind(paragraph)
        if paragraph.strip().startswith(_CLOSING):
            current = "tail"
        elif kind is not None:
            current = kind
        target = (
            tail
            if current == "tail"
            else head
            if current is None
            else blocks.setdefault(current, [])
        )
        target.append(paragraph)
    if len(blocks) < 2:
        return render(text)
    items = [
        (name, "", render("\n".join(blocks[name]))) for name, _ in TEST_KINDS if name in blocks
    ]
    markup = render("\n".join(head)) if head else ""
    markup += tabs_html("tests", items, "Tests")
    return markup + (render("\n".join(tail)) if tail else "")


__all__ = ["TEST_KINDS", "tabs_html", "tests_tabs"]
