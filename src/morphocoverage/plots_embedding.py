"""Figures of an ``embedding.Embedding``: the PCA biplot and the UMAP, coloured by condition.

- **PCA**: every observation a point on the first two axes (each labelled with its share of
  the variance), coloured by condition (the control in the first colour, not grey, and drawn
  first so the conditions sit on top; past 12 conditions only the control and the
  ``highlight`` conditions keep a colour, ``_palette.condition_palette``, the others grey
  under them, one entry in the legend), and an arrow per highlighted feature: its loadings on
  the two axes, scaled to the cloud, so the direction a feature pulls the observations is
  read off the figure. The highlighted features are the discovery's best-ranked
  (``Embedding.loadings["top"]``), or the ``ARROWS`` features with the largest loadings when
  none is ranked.
- **UMAP**: the observations UMAP placed, coloured the same way; its axes have no units.

Figures are ``matplotlib.figure.Figure`` objects built without pyplot, as in ``plots``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke

from ._palette import COLOURS, FONT, GREY, INK, condition_palette, style_axes
from ._wording import count, percent
from .embedding import Embedding
from .metadata import CONDITION_COLUMN, sorted_labels
from .schema import observation_words

if TYPE_CHECKING:
    from matplotlib.axes import Axes

ARROWS = 5


def _colours(points: pd.DataFrame, control: Any, highlight: Sequence[Any]) -> dict[Any, str]:
    """A colour per condition in sorted order, the control first in the palette's first
    colour (a grey control is hard to see); past 12, grey but for the control and the
    ``highlight`` conditions (``_palette.condition_palette``)."""
    if CONDITION_COLUMN not in points:
        return {None: COLOURS[0]}
    conditions = sorted_labels(points[CONDITION_COLUMN], control)
    return condition_palette(conditions, control, highlight)


def _scatter(ax: Axes, points: pd.DataFrame, x: str, y: str, colours: dict[Any, str]) -> None:
    """The points by condition: the grey ones first, as one entry of the legend."""
    greyed = [c for c, colour in colours.items() if colour == GREY]
    if greyed:
        mine = points[points[CONDITION_COLUMN].isin(greyed)].dropna(subset=[x, y])
        ax.scatter(
            mine[x],
            mine[y],
            s=14,
            alpha=0.45,
            color=GREY,
            linewidths=0,
            label=f"the other {len(greyed)} conditions ({len(mine)})",
            rasterized=True,
        )
    coloured = {c: colour for c, colour in colours.items() if colour != GREY}
    for condition, colour in coloured.items():
        mine = points if condition is None else points[points[CONDITION_COLUMN] == condition]
        mine = mine.dropna(subset=[x, y])
        ax.scatter(
            mine[x],
            mine[y],
            s=18,
            alpha=0.6,
            color=colour,
            linewidths=0,
            label=f"{condition} ({len(mine)})" if condition is not None else None,
            rasterized=True,
        )
    if any(c is not None for c in colours):
        ax.legend(
            frameon=False,
            fontsize=FONT,
            markerscale=1.6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(coloured) + bool(greyed), 4),
        )


def _spread_labels(
    positions: list[tuple[float, float]], x_span: float, y_span: float
) -> list[tuple[float, float]]:
    """Label positions with labels that fall on one another moved apart vertically: arrows of
    duplicated or collinear features end at nearly one point (audit A7)."""
    gap = 0.05 * y_span
    placed: list[tuple[float, float]] = []
    for x, y in sorted(positions, key=lambda p: p[1]):
        while any(abs(y - py) < gap and abs(x - px) < 0.3 * x_span for px, py in placed):
            y += gap
        placed.append((x, y))
    return placed


def _arrows(ax: Axes, embedding: Embedding) -> None:
    loadings = embedding.loadings
    shown = loadings[loadings["top"]]
    if shown.empty:
        size = np.hypot(loadings["pc1"], loadings["pc2"])
        shown = loadings.loc[size.sort_values(ascending=False).index[:ARROWS]]
    points = embedding.points
    reach = 0.8 * float(np.nanmax(np.hypot(points["pc1"], points["pc2"])) or 1.0)
    longest = float(np.hypot(shown["pc1"], shown["pc2"]).max()) or 1.0
    tips = [
        (float(r.pc1) / longest * reach, float(r.pc2) / longest * reach) for r in shown.itertuples()
    ]
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    spots = _spread_labels(
        [(dx * 1.06, dy * 1.06) for dx, dy in tips], x_high - x_low, y_high - y_low
    )
    for row, (dx, dy), (lx, ly) in zip(shown.itertuples(), tips, spots, strict=True):
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6},
        )
        label = ax.text(
            lx,
            ly,
            row.feature,
            fontsize=FONT - 1,
            weight="bold",
            color=INK,
            ha="left" if dx >= 0 else "right",
            va="center",
        )
        label.set_path_effects([withStroke(linewidth=3, foreground="white")])
    # Keep every arrow and label inside the axes: they must not run into the title.
    x_pad, y_pad = 0.04 * (x_high - x_low), 0.06 * (y_high - y_low)
    words = max((len(str(f)) for f in shown["feature"]), default=0)
    x_room = 0.012 * words * (x_high - x_low)
    lows = [
        min(
            x_low,
            *(
                x - (x_room if dx < 0 else 0) - x_pad
                for (x, _), (dx, _) in zip(spots, tips, strict=True)
            ),
        )
    ]
    highs = [
        max(
            x_high,
            *(
                x + (x_room if dx >= 0 else 0) + x_pad
                for (x, _), (dx, _) in zip(spots, tips, strict=True)
            ),
        )
    ]
    ax.set_xlim(lows[0], highs[0])
    ax.set_ylim(
        min(y_low, *(y - y_pad for _, y in spots)), max(y_high, *(y + y_pad for _, y in spots))
    )


def embedding_figures(
    embedding: Embedding,
    *,
    control: Mapping[str, Any] | None = None,
    highlight: Sequence[Any] = (),
) -> dict[str, Figure]:
    """``{"pca": Figure, "umap": Figure}`` (UMAP only when it was computed; see the module).
    ``control`` (``{"condition": ...}``) is drawn first, in the first colour; past 12
    conditions, the ``highlight`` conditions are the others in colour."""
    if not isinstance(embedding, Embedding):
        raise TypeError("embedding must be an embedding.Embedding")
    points = embedding.points
    control_value = dict(control or {}).get(CONDITION_COLUMN)
    colours = _colours(points, control_value, highlight)
    unit = observation_words(embedding.settings)[0]
    n_features = len(embedding.settings["features"])
    figures: dict[str, Figure] = {}
    figure = Figure(figsize=(7.5, 6.8), layout="constrained")
    ax = figure.add_subplot()
    _scatter(ax, points, "pc1", "pc2", colours)
    style_axes(ax, grid="both")
    _arrows(ax, embedding)
    first, second = embedding.explained
    ax.set_xlabel(f"PC1 ({percent(first)} of the variance)")
    ax.set_ylabel(f"PC2 ({percent(second)} of the variance)")
    ax.set_title(
        f"PCA of {count(n_features, 'feature')}, standardised; "
        f"{count(len(points), unit)}; arrows: the features' loadings",
        fontsize=10,
        loc="left",
    )
    figures["pca"] = figure
    if points["umap1"].notna().any():
        figure = Figure(figsize=(7.5, 6.8), layout="constrained")
        ax = figure.add_subplot()
        _scatter(ax, points, "umap1", "umap2", colours)
        style_axes(ax, grid="")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        settings = embedding.settings
        ax.set_title(
            f"UMAP of {count(n_features, 'feature')}, standardised; "
            f"{count(settings['umap_points'], unit)} (n_neighbors {settings['n_neighbors']}, "
            f"min_dist {settings['min_dist']:g})",
            fontsize=10,
            loc="left",
        )
        figures["umap"] = figure
    return figures


__all__ = ["embedding_figures"]
