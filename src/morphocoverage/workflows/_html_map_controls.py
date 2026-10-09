"""The map controls of the report page (private; _html_maps): the "Colour by" menu, the search
box, and the values buttons: Normalised and Raw, with a menu of the methods behind
"Normalised" when there are several (owner, 2026-10-09)."""

from __future__ import annotations

import html
from typing import Any

from .._palette import GREY
from ._html_map_text import METHOD_NAMES, METHOD_WORDS, VALUE_HELP


def colour_menu(data: dict[str, Any], n_features: int) -> str:
    """The "Colour by" menu: condition, population (when found) or a feature, and how many of
    the ``n_features`` it offers when it cannot offer them all (``MAP_FEATURES``)."""
    options = ['<option value="condition">condition</option>']
    if data["populations"]:
        options.append('<option value="population">population</option>')
    options += [
        f'<option value="{html.escape(f)}">{html.escape(f)}</option>' for f in data["features"]
    ]
    shown = len(data["features"])
    limit = (
        f'<span class="note">Colour by and the selection use the {shown} best-ranked of '
        f"{n_features} features.</span>"
        if shown < n_features
        else ""
    )
    return (
        '<label class="colour-by-label">Colour by <select class="colour-by">'
        + "".join(options)
        + "</select></label>"
        + limit
    )


def find_box(names: list[str], colours: list[str]) -> str:
    """Past 12 conditions, a search box to pick one and see it highlighted on the map."""
    if GREY not in colours:
        return ""
    options = "".join(f'<option value="{html.escape(n)}"></option>' for n in names)
    return (
        '<label class="find-label">Find a condition <input type="search" class="find" '
        'list="embedding-conditions" placeholder="name" aria-label="Find a condition">'
        f'</label><datalist id="embedding-conditions">{options}</datalist>'
    )


def _corrected_label(methods: list[str]) -> str:
    """The first button's word: "Normalised" when that method is there, else the lone
    method's own name (a ``KeyError`` for a view the page does not know)."""
    return "Normalised" if "normalised" in methods else METHOD_NAMES[methods[0]]


def _values_buttons(values: list[str], methods: list[str]) -> str:
    """Normalised (the methods' button, when there are methods) and Raw."""
    pressed = values[0] != "raw"
    corrected = (
        '<button type="button" data-key="values" data-value="@corrected" '
        f'aria-pressed="{str(pressed).lower()}">{html.escape(_corrected_label(methods))}</button>'
        if methods
        else ""
    )
    return (
        corrected
        + '<button type="button" data-key="values" data-value="raw" '
        + f'aria-pressed="{str(not pressed).lower()}">Raw</button>'
    )


def _method_menu(methods: list[str]) -> str:
    """The menu of the methods behind Normalised; empty with a single method."""
    if len(methods) < 2:
        return ""
    options = "".join(
        f'<option value="{v}" title="{html.escape(VALUE_HELP[v])}">'
        f"{html.escape(METHOD_WORDS[v])}</option>"
        for v in methods
    )
    return (
        f'<label class="values-label">Method <select class="values-menu">{options}</select></label>'
    )


def values_controls(values: list[str]) -> tuple[str, str]:
    """The Normalised / Raw buttons and the method menu (empty with one method) for the views
    ``values`` (``normalised``, ``raw``, ``centred``)."""
    methods = [v for v in values if v != "raw"]
    return _values_buttons(values, methods), _method_menu(methods)
