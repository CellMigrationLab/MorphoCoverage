"""The figures' shared look (private): one colour per condition, the same in the static
figures (matplotlib) and the HTML report's interactive ones (Plotly), the marker shapes of
the technical repeats, and the axis style of every static figure.

``COLOURS`` holds 12 colours, the control included (no grey: it is hard to see); beyond
12 conditions they repeat, and the labels say which is which. ``MARKERS`` tells up to 8
technical repeats of a biological repeat apart by shape, so the colour stays the
condition's; more share ``OTHER_MARKER``.
"""

from __future__ import annotations

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
FONT = 11


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


__all__ = ["COLOURS", "FONT", "GRID", "INK", "MARKERS", "MUTED", "OTHER_MARKER", "style_axes"]
