"""The layout of a static figure's panels and legend (private; ``plots``, ``plots_pairwise``).

A figure's words are a legend at its bottom, as in a paper: "Legend:", then one paragraph
per panel, "(A) ...", "(B) ...", lettered when the figure has several panels, each panel
marked with its letter at its top left. ``set_legend`` replaces the legend a figure already
has (the figure is made taller or shorter for it), so a panel added under the plot goes
between the plot and its legend.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Any

from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy

from ._palette import INK, MUTED

LEGEND_FONT = 9.5
# Characters per inch of a line of the legend, and the height of a line in inches.
CHARS_PER_INCH = 14.5
LINE_INCHES = LEGEND_FONT * 1.38 / 72
MARGIN = 0.18


def sentences(caption: str) -> str:
    """A figure's caption (one clause per line) as sentences; the first line starts with the
    feature's own name, kept as written."""
    lines = [line.strip() for line in caption.split("\n") if line.strip()]
    lines = lines[:1] + [line[:1].upper() + line[1:] for line in lines[1:]]
    return ". ".join(lines) + ("." if lines else "")


def grow_bottom(figure: Figure, inches: float) -> None:
    """``inches`` more at the bottom of ``figure`` (``inches`` < 0 takes them away), every
    axes keeping its size and its place from the top."""
    width, height = figure.get_size_inches()
    total = height + inches
    figure.set_size_inches(width, total)
    for ax in figure.axes:
        box = ax.get_position()
        ax.set_position(
            [box.x0, (box.y0 * height + inches) / total, box.width, box.height * height / total]
        )
    for text in figure.texts:
        x, y = text.get_position()
        text.set_position((x, (y * height + inches) / total))


def panel_letter(ax: Any, letter: str, *, pad: float = 6.0) -> None:
    """The panel's letter, bold, left of the axes' title, on its baseline (``pad``, in
    points, is the title's)."""
    ax.text(
        0,
        1,
        letter,
        transform=offset_copy(ax.transAxes, ax.figure, x=-10, y=pad, units="points"),
        ha="right",
        va="baseline",
        fontsize=15,
        weight="bold",
        color=INK,
    )


def remove_legend(figure: Figure) -> None:
    """Take the figure's legend away, and the space it took."""
    previous = getattr(figure, "legend_artist", None)
    if previous is not None:
        previous.remove()
        grow_bottom(figure, -figure.legend_inches)
        figure.legend_artist, figure.legend_inches = None, 0.0


def set_legend(figure: Figure, entries: Sequence[str]) -> None:
    """The legend at the bottom of ``figure``: one paragraph per entry, lettered (A), (B) ...
    when there are several; replaces the legend the figure had."""
    remove_legend(figure)
    width = figure.get_size_inches()[0]
    chars = max(40, int((width - 2 * MARGIN) * CHARS_PER_INCH))
    lettered = len(entries) > 1
    paragraphs = []
    for index, entry in enumerate(entries):
        head = f"({chr(ord('A') + index)}) " if lettered else ""
        paragraphs.append("\n".join(textwrap.wrap(head + entry, chars)))
    text = "Legend:\n" + "\n\n".join(paragraphs)
    lines = text.count("\n") + 1
    inches = lines * LINE_INCHES + 2 * MARGIN
    grow_bottom(figure, inches)
    total = figure.get_size_inches()[1]
    figure.legend_artist = figure.text(
        MARGIN / width,
        (inches - MARGIN) / total,
        text,
        ha="left",
        va="top",
        fontsize=LEGEND_FONT,
        color=MUTED,
        linespacing=1.3,
    )
    figure.legend_inches = inches


__all__ = ["grow_bottom", "panel_letter", "remove_legend", "sentences", "set_legend"]
