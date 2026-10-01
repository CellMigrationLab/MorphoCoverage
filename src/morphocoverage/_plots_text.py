"""The words of a ``plots.plots_of_data`` figure (private): its title, which says what is
plotted, what a point is and what the interval resamples, and its legend of technical
repeats."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes

from ._palette import MUTED
from ._wording import ordinal, rate
from .metadata import REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN


def legend(ax: Axes, layout: Any, shapes: Sequence[str], n_technical: int) -> None:
    """The technical repeats' shapes, by rank, beside the plot; nothing when every biological
    repeat has one, and a note when there are more than shapes."""
    if layout.technical_col is None or n_technical < 2:
        return
    if n_technical <= len(shapes):
        handles = [
            ax.scatter(
                [],
                [],
                color=MUTED,
                marker=shapes[rank],
                s=30,
                label=f"{ordinal(rank + 1)}",
            )
            for rank in range(n_technical)
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=9.5,
            title=(
                f"{_named('well', layout.technical_col, TECHNICAL_REPEAT_COLUMN)}"
                ",\nin order within each\nbiological repeat"
            ),
            title_fontsize=9.5,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            alignment="left",
        )
    else:
        ax.text(
            0.01,
            0.99,
            f"up to {n_technical} "
            f"{_named('wells', layout.technical_col, TECHNICAL_REPEAT_COLUMN)} per "
            f"biological repeat: the first {len(shapes)} shaped in order, the others small dots",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color=MUTED,
        )


def _span(counts: Sequence[int]) -> str:
    """``3`` or ``2 to 5``."""
    return f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)} to {max(counts)}"


def _named(word: str, column: str | None, standard: str) -> str:
    """``word``, with its column in brackets when the column has another name than the
    MorphoBricks one (``technical repeat (well)``)."""
    if column in (None, standard) or str(column).lower() in (word, word.rstrip("s")):
        return word
    return f"{word} ({column})"


def _pooled_name(layout: Any) -> str:
    """The pooled top-level units in words: ``biological repeats`` or their columns."""
    if len(layout.pooled_cols) == 1 and layout.pooled_cols[0] in (
        layout.repeat_col,
        REPEAT_COLUMN,
    ):
        return _named("biological repeats", layout.pooled_cols[0], REPEAT_COLUMN)
    return f"units of {'/'.join(layout.pooled_cols)}"


def title(
    feature: str,
    part: str,
    layout: Any,
    unit: str,
    summary: str,
    level: float,
    clusters: Sequence[tuple[int, list[int]]] = (),
) -> str:
    """Three lines: what is plotted and how it is split, what a point is, what the interval is.

    ``clusters`` holds, per cluster, its top-level units and each one's
    technical repeats (``_cluster_units``): the interval line says how many
    units are resampled, and a fourth line warns when a cluster has only 2 or
    3, which no resampling turns into a reliable interval, or a single
    technical repeat, whose interval resamples its fields and so shows
    field-to-field variation only (it covered the repeat's median about half
    the time with a well SD above the field SD).
    """
    repeats_word = _named("biological repeat", layout.repeat_col, REPEAT_COLUMN)
    # A technical repeat is a well, dish or coverslip: "well" in lab words (owner, 2026-10-01).
    technical_word = _named("well", layout.technical_col, TECHNICAL_REPEAT_COLUMN)
    technicals = _named("wells", layout.technical_col, TECHNICAL_REPEAT_COLUMN)
    # A cluster is the points of one condition, or of one condition in one repeat.
    cluster = "condition" if layout.pooled_repeats or not layout.repeat_col else "repeat"
    repeats = ""
    if layout.pooled_repeats:
        repeats = (
            f"; each condition's {_pooled_name(layout)} pooled (more than the {layout.max_repeats} "
            "that can be shown side by side)"
        )
    elif layout.repeat_col:
        repeats = f"; each {repeats_word} side by side"
    unshown = [
        c for c in layout.pooled_cols if not (layout.pooled_repeats and c == layout.repeat_col)
    ]
    if unshown:
        repeats += f"; {'/'.join(unshown)} not shown apart, pooled within each cluster"
    by_top = [n for n, _ in clusters if n > 1]
    single = [technical[0] for n, technical in clusters if n == 1]
    coloured, phrases, caveat = "", [], ""
    if layout.technical_col:
        coloured = f", its shape the {technical_word}"
    if by_top:
        phrases.append(
            f"whole {_pooled_name(layout)} with all their fields ({_span(by_top)} per {cluster})"
        )
        if any(n in (2, 3) for n in by_top):
            caveat = f"\nwith 2-3 {_pooled_name(layout)} per {cluster} the interval is only a guide"
    if single or not by_top:
        if layout.technical_col:
            several = [n for n in single if n > 1]
            if not single:
                phrases.append(
                    f"whole {technicals} with their fields, or the fields where a "
                    f"{cluster} has one {technical_word}"
                )
            if several:
                phrases.append(
                    f"whole {technicals} with their fields ({_span(several)} per {cluster})"
                )
            if single and all(n == 1 for n in single) and not by_top:
                phrases.append(f"its fields (one {technical_word} per {cluster})")
            elif any(n == 1 for n in single):
                phrases.append(f"the fields, where a {cluster} has one {technical_word}")
            if any(n in (2, 3) for n in several):
                caveat = f"\nwith 2-3 {technicals} the interval is only a guide"
            if any(n == 1 for n in single):
                caveat += (
                    f"\nwith one {technical_word} the interval shows field-to-field "
                    "variation only, not well-to-well"
                )
        else:
            phrases.append("whole fields")
    resampled = "; or ".join(phrases)
    if summary == "median":
        caveat += "\nthe median is drawn; the tests across repeats use the repeat means"
    return (
        f"{feature}{part}{repeats}\nevery {unit} a point in its condition's colour{coloured}\n"
        f"{summary} with its {rate(level)} interval from resampling {resampled}{caveat}"
    )


__all__ = ["legend", "title"]
