"""The page's fields to open (private; ``html_report``; owner, 2026-10-06, audit 10, A14
and B3): two cards.

- "Fields to check (quality control)": the fields the copy checks name, those far from the
  other fields of their condition and repeat (the acquisition anomalies), and those that
  sway the result, hold an unusual count or differ from their stage neighbours, with the
  code to run the report again without the ones the images show to be artefacts, every one
  commented out (removing a field is opt-in).
- "Fields to look at (new states, exploratory)": the fields holding a state no other field
  shows, exploratory, to look at and never among the fields to remove; or why a group lists
  none (the definition of new not informative at its size).
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ._exploratory import exploratory_line
from ._html_cards import fields_table, flagged_fields, new_state_fields
from ._html_next import fields_next

NEW_STATES = "new-states"
NEW_STATES_TITLE = "Fields to look at (new states, exploratory)"
NEW_STATES_HELP = (
    "Fields holding a state no other field shows, by the declared definition of new: a rare "
    "phenotype, or an artefact. Exploratory: open their images; they are never removed."
)
_NOT_AN_ERROR = (
    '<p class="note">None of these is called an error: look at the images to see whether it '
    "is biology, technique or simply a large field.</p>"
)


def has_new_states(result: Any) -> bool:
    """Whether the page has the new states' card."""
    rows, reasons = new_state_fields(result)
    return bool(rows or reasons)


def given_identities(
    result: Any, identities: list[tuple[Any, ...]], key: list[str]
) -> list[tuple[Any, ...]]:
    """The flagged fields' identities in the tables as given: the report reads conditions and
    repeats by their IDs (``_pairing``), and the code under "What to do next" runs on the
    user's tables, which hold the folder labels (``settings["folder_labels"]``)."""
    labels = (getattr(result, "settings", None) or {}).get("folder_labels")
    if not labels or CONDITION_COLUMN not in key:
        return identities
    folders = {(c, r): (cf, rf) for c, r, cf, rf in labels}
    at = key.index(CONDITION_COLUMN)
    repeat_at = key.index(REPEAT_COLUMN) if REPEAT_COLUMN in key else None
    given = []
    for identity in identities:
        repeat = None if repeat_at is None else identity[repeat_at]
        found = folders.get((identity[at], repeat))
        if found is None:
            given.append(identity)
            continue
        values = list(identity)
        values[at] = found[0]
        if repeat_at is not None:
            values[repeat_at] = found[1]
        given.append(tuple(values))
    return given


def fields_cards(
    result: Any,
    chip_html: str,
    short_html: str,
    full_html: str,
    card: Callable[[str, str, str, str], str],
    annotate: Callable[[str], str],
    title: str,
) -> str:
    """The two cards (see the module): ``card(key, title, chip, body)`` draws one,
    ``short_html`` and ``full_html`` are question 6's text, ``annotate`` links the glossary."""
    rows, identities, key = flagged_fields(result)
    identities = given_identities(result, identities, key)
    if rows:
        body = (
            "<p>Please check these fields of view, and why:</p>"
            + fields_table(rows, annotate)
            + _NOT_AN_ERROR
        )
    else:
        body = short_html
    body += f"<details><summary>Show more</summary>{full_html}</details>"
    body += fields_next(identities, key)
    cards = card("inspect", title, chip_html, body)
    new, reasons = new_state_fields(result)
    if not (new or reasons):
        return cards
    parts = []
    if new:
        parts.append("<p>These fields hold a state no other field shows:</p>")
        parts.append(fields_table(new, annotate))
    for reason in reasons:
        parts.append(f'<p class="note">{html.escape(reason)}.</p>')
    parts.append(f'<p class="note">{html.escape(exploratory_line(["novelty"]))}</p>')
    return cards + card(NEW_STATES, NEW_STATES_TITLE, "", "".join(parts))


__all__ = [
    "NEW_STATES",
    "NEW_STATES_HELP",
    "NEW_STATES_TITLE",
    "fields_cards",
    "given_identities",
    "has_new_states",
]
