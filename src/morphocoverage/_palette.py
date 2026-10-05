"""The figures' shared look (private): one colour per condition, the same in the static
figures (matplotlib) and the HTML report's interactive ones (Plotly), the marker shapes of
the technical repeats, and the axis style of every static figure.

``COLOURS`` holds 12 colours, the control included (no grey: it is hard to see). Up to 12
conditions each has its own (``condition_palette``); past 12, the control and the conditions
the caller highlights (the hits, at most 11) keep a colour and every other is ``GREY``
(owner, 2026-10-05), so no other condition is ever drawn in the control's blue.
``MARKERS`` tells up to 8 technical repeats of a biological repeat apart by shape, so the
colour stays the condition's; more share ``OTHER_MARKER``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

COLOURS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#00a3b4",
    "#8c564b",
    "#b0469c",
    "#6b7f00",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
OTHER_MARKER = "."
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# The conditions left uncoloured past 12 (``condition_palette``).
GREY = "#bdbcb6"
FONT = 11


def condition_palette(
    conditions: Sequence[Any], control: Any = None, highlight: Sequence[Any] = ()
) -> dict[Any, str]:
    """A colour per condition, in their order: up to ``len(COLOURS)`` each its own, by its
    place; past that the ``control`` in the first colour, the ``highlight`` conditions after
    it in the next (in their order, as many as the colours hold) and every other ``GREY``."""
    conditions = list(conditions)
    if len(conditions) <= len(COLOURS):
        return {c: COLOURS[i] for i, c in enumerate(conditions)}
    present = set(conditions)
    lead = [control] if control in present else []
    picked = [h for h in dict.fromkeys(highlight) if h in present and h != control]
    # The first colour stays the control's, even in a figure that does not draw it.
    first = 0 if lead or control is None else 1
    coloured = dict(zip([*lead, *picked], COLOURS[first:], strict=False))
    return {c: coloured.get(c, GREY) for c in conditions}


def style_axes(ax: Any, *, grid: str = "y") -> None:
    """The look of every static plot: light grid lines behind the data, no box, ticks and
    labels in the text colour."""
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(colors=INK, labelsize=FONT, length=0, pad=6)
    ax.yaxis.label.set_size(FONT + 1)
    ax.xaxis.label.set_size(FONT + 1)


__all__ = [
    "COLOURS",
    "FONT",
    "GREY",
    "GRID",
    "INK",
    "MARKERS",
    "MUTED",
    "OTHER_MARKER",
    "condition_palette",
    "style_axes",
]
