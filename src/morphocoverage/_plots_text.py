"""The words of a ``plots.plots_of_data`` figure (private): its title, which says what is
plotted, what a point is and what the interval resamples, and its legend of technical
repeats."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes

from ._palette import MUTED
from ._wording import ordinal, rate
from .metadata import REPEAT_COLUMN


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
                label=f"{ordinal(rank + 1)} {layout.technical_col}",
            )
            for rank in range(n_technical)
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=9.5,
            title=(
                f"technical repeat\n({layout.technical_col}),\nin order within each\n"
                "biological repeat"
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
            f"up to {n_technical} technical repeats ({layout.technical_col}) per biological "
            f"repeat: the first {len(shapes)} shaped in order, the others small dots",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color=MUTED,
        )


def _span(counts: Sequence[int]) -> str:
    """``3`` or ``2 to 5``."""
    return f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)} to {max(counts)}"


def _pooled_name(layout: Any) -> str:
    """The pooled top-level units in words: ``biological repeats (repeat)`` or their columns."""
    if len(layout.pooled_cols) == 1 and layout.pooled_cols[0] in (
        layout.repeat_col,
        REPEAT_COLUMN,
    ):
        return f"biological repeats ({layout.pooled_cols[0]})"
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
    repeats = ""
    if layout.pooled_repeats:
        repeats = (
            f"; biological repeats ({layout.repeat_col}) pooled, more than the "
            f"{layout.max_repeats} shown side by side"
        )
    elif layout.repeat_col:
        repeats = f"; biological repeats ({layout.repeat_col}) side by side"
    unshown = [
        c for c in layout.pooled_cols if not (layout.pooled_repeats and c == layout.repeat_col)
    ]
    if unshown:
        repeats += f"; {'/'.join(unshown)} not shown apart, pooled within each cluster"
    by_top = [n for n, _ in clusters if n > 1]
    single = [technical[0] for n, technical in clusters if n == 1]
    coloured, phrases, caveat = "", [], ""
    if layout.technical_col:
        coloured = f", shaped by technical repeat ({layout.technical_col})"
    if by_top:
        phrases.append(
            f"whole {_pooled_name(layout)}, each with all its fields as acquired "
            f"({_span(by_top)} per cluster)"
        )
        if any(n in (2, 3) for n in by_top):
            caveat = f"\nwith only 2-3 {_pooled_name(layout)} a cluster's interval is not reliable"
    if single or not by_top:
        if layout.technical_col:
            span = f"{_span(single)} per cluster; " if single else ""
            phrases.append(
                f"whole technical repeats with their fields as acquired ({span}whole fields "
                "where there is one)"
            )
            if any(n in (2, 3) for n in single):
                caveat = "\nwith only 2-3 technical repeats a cluster's interval is not reliable"
            if any(n == 1 for n in single):
                caveat += (
                    "\none technical repeat: the interval shows field-to-field variation "
                    "only, not well-to-well"
                )
        else:
            phrases.append("whole fields")
    resampled = " or, in a cluster of one, ".join(phrases)
    return (
        f"{feature}{part}{repeats}\nevery {unit} a point in its condition's colour{coloured}\n"
        f"{summary} with its {rate(level)} interval from resampling {resampled}{caveat}"
    )


__all__ = ["legend", "title"]
