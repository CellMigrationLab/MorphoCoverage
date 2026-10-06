"""Tabs on the HTML page (private; ``_html_enough``, ``html_report``): one panel at a time,
the arrow keys moving between them (``_html_script``).

- ``tabs_html``: tabs from ``(label, mark, markup)`` items, the first one shown; above
  ``TABS_AT_MOST`` items (hundreds of features), a list to search and pick from instead,
  one panel shown at a time all the same.
- ``test_blocks``: the text of "The tests behind the reading" cut by kind of test (across
  the repeats, the reagents of a target, the cell-level screen, within single repeats,
  planning, every feature, beyond the mean), for ``_html_tests``: the text before the first
  kind, each kind's, and the closing cautions and assumptions. The page builds its tabs
  from the results and takes only the reagents' paragraph from the text; the kinds in
  ``OFF_PAGE`` are cut out so they never reach it (owner, 2026-10-02: the plan for the
  next experiment, the ranking and the within-repeat tables stay in the text report).
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

# Each kind of test by the words its paragraph starts with in the text report.
TEST_KINDS = (
    ("Across repeats", ("Across the ",)),
    ("Reagents", ("Independent reagents",)),
    ("Cell-level screen", ("Cell-level screen",)),
    ("Within repeats", ("No test within a single repeat", "Within each repeat")),
    ("Planning", ("Planning the next experiment",)),
    ("Every feature", ("Which features separate the conditions?",)),
    ("Beyond the mean", ("Beyond the mean",)),
)
# Kinds the page never shows from the text: the plan (for its own section), the ranking (the
# heatmap and the Across repeats tab give it) and the tests within single repeats.
OFF_PAGE = ("Planning", "Every feature", "Within repeats")
_CLOSING = ("⚠️", "Assumptions")
# The screen's paragraph starts with the observations' own word ("Nucleus-level screen").
_SCREEN_START = re.compile(r"\w+-level screen\b")
# More tabs than this do not fit a row or two: a list to search and pick from instead.
TABS_AT_MOST = 12


def _picker(name: str, items: Sequence[tuple[str, str, str]], label: str) -> str:
    """Many items: a search box filtering a list, the picked item's panel shown."""
    options = "".join(
        f'<option value="{i}"{" selected" if i == 0 else ""}>'
        f"{html.escape(f'{mark} {text}' if mark else text)}</option>"
        for i, (text, mark, _) in enumerate(items)
    )
    panels = "".join(
        f'<div class="tab-panel" role="region" id="{name}-panel-{i}" '
        f'aria-label="{html.escape(text)}"{"" if i == 0 else " hidden"}>'
        # The list may be scrolled away: the panel names what it shows.
        f'<h4 class="panel-name">{html.escape(f"{text} {mark}" if mark else text)}</h4>'
        f"{markup}</div>"
        for i, (text, mark, markup) in enumerate(items)
    )
    find = f"Find one of {len(items)} {label.lower()}"
    return (
        f'<div class="feature-tabs picker"><div class="pick">'
        f'<input type="search" id="{name}-find" placeholder="{html.escape(find)}" '
        f'aria-label="{html.escape(find)}" autocomplete="off">'
        f'<select id="{name}-pick" size="8" aria-label="{html.escape(label)}">{options}</select>'
        f"</div>{panels}</div>"
    )


def tabs_html(name: str, items: Sequence[tuple[str, str, str]], label: str) -> str:
    """Tabs called ``name`` (an id prefix) from ``(label, mark, markup)`` items; ``label``
    names the tab list for a screen reader. Above ``TABS_AT_MOST`` items, a list to search
    and pick from."""
    if len(items) > TABS_AT_MOST:
        return _picker(name, items, label)
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
    if _SCREEN_START.match(text):
        return "Cell-level screen"
    return next((name for name, starts in TEST_KINDS if text.startswith(starts)), None)


def test_blocks(text: str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """The text of the tests behind the reading cut by kind of test (``TEST_KINDS``): the
    paragraphs before the first kind, each kind's, and the closing cautions and assumptions."""
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
    return head, blocks, tail


# An embedding-only report has no mean to plan for (owner, 2026-10-06).
NOTHING = (
    "Nothing to plan here: without explainable features the targets are the fingerprint of "
    "each table and a labelled phenotype, and none can be judged here."
)
NOTHING_LIVE = (
    "Without a readout there is no default stopping rule: pass live_session(setup, "
    "rules=(...)), a FOVBootstrapStrategy on a table's columns or a RareIncidenceStrategy on "
    "the labelled phenotype."
)


def choose(name: str, label: str, items: Sequence[tuple[str, str]], empty: str = NOTHING) -> str:
    """A ``<select>`` called ``name`` (an id prefix) and one panel per ``(option, markup)``,
    the first shown; without any, ``empty`` says why (an embedding-only report)."""
    if not items:
        return f'<p class="note">{html.escape(empty)}</p>'
    options = "".join(
        f'<option value="{i}"{" selected" if i == 0 else ""}>{html.escape(text)}</option>'
        for i, (text, _) in enumerate(items)
    )
    panels = "".join(
        f'<div class="choice-panel" id="{name}-panel-{i}"{"" if i == 0 else " hidden"}>'
        f"{markup}</div>"
        for i, (_, markup) in enumerate(items)
    )
    return (
        f'<div class="choose"><label for="{name}-pick">{html.escape(label)}</label>'
        f'<select id="{name}-pick">{options}</select>{panels}</div>'
    )


__all__ = [
    "NOTHING",
    "NOTHING_LIVE",
    "OFF_PAGE",
    "TABS_AT_MOST",
    "TEST_KINDS",
    "choose",
    "tabs_html",
    "test_blocks",
]
