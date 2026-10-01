"""Figures of an ``embedding.Embedding``: the PCA biplot and the UMAP, coloured by condition.

- **PCA**: every observation a point on the first two axes (each labelled with its share of
  the variance), coloured by condition (the control in the first colour, not grey, and drawn
  first so the conditions sit on top), and an arrow per highlighted feature: its loadings on
  the two axes, scaled to the cloud, so the direction a feature pulls the observations is
  read off the figure. The highlighted features are the discovery's best-ranked
  (``Embedding.loadings["top"]``), or the ``ARROWS`` features with the largest loadings when
  none is ranked.
- **UMAP**: the observations UMAP placed, coloured the same way; its axes have no units.

Figures are ``matplotlib.figure.Figure`` objects built without pyplot, as in ``plots``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke

from ._palette import COLOURS, FONT, INK, style_axes
from ._wording import count, percent
from .embedding import Embedding
from .metadata import CONDITION_COLUMN, sorted_labels

ARROWS = 5
MAX_CONDITIONS = len(COLOURS)


def _colours(points: pd.DataFrame, control: Any) -> dict[Any, str]:
    """A colour per condition, the palette in sorted order (the control first, in the palette's
    first colour: a grey control is hard to see)."""
    if CONDITION_COLUMN not in points:
        return {None: COLOURS[0]}
    conditions = sorted_labels(points[CONDITION_COLUMN], control)
    return {c: COLOURS[i % len(COLOURS)] for i, c in enumerate(conditions)}


def _scatter(ax, points: pd.DataFrame, x: str, y: str, colours: dict[Any, str]) -> None:
    for condition, colour in colours.items():
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
            ncol=min(len(colours), 4),
        )


def _arrows(ax, embedding: Embedding) -> None:
    loadings = embedding.loadings
    shown = loadings[loadings["top"]]
    if shown.empty:
        size = np.hypot(loadings["pc1"], loadings["pc2"])
        shown = loadings.loc[size.sort_values(ascending=False).index[:ARROWS]]
    points = embedding.points
    reach = 0.8 * float(np.nanmax(np.hypot(points["pc1"], points["pc2"])) or 1.0)
    longest = float(np.hypot(shown["pc1"], shown["pc2"]).max()) or 1.0
    for row in shown.itertuples():
        dx, dy = row.pc1 / longest * reach, row.pc2 / longest * reach
        ax.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6},
        )
        label = ax.text(
            dx * 1.06,
            dy * 1.06,
            row.feature,
            fontsize=FONT - 1,
            weight="bold",
            color=INK,
            ha="left" if dx >= 0 else "right",
            va="center",
        )
        label.set_path_effects([withStroke(linewidth=3, foreground="white")])


def embedding_figures(
    embedding: Embedding, *, control: Mapping[str, Any] | None = None
) -> dict[str, Figure]:
    """``{"pca": Figure, "umap": Figure}`` (UMAP only when it was computed; see the module).
    ``control`` (``{"condition": ...}``) is drawn first, in the first colour."""
    if not isinstance(embedding, Embedding):
        raise TypeError("embedding must be an embedding.Embedding")
    points = embedding.points
    control_value = dict(control or {}).get(CONDITION_COLUMN)
    colours = _colours(points, control_value)
    if len(colours) > MAX_CONDITIONS + 1:
        raise ValueError(
            f"{len(colours)} conditions are too many to tell apart by colour (at most "
            f"{MAX_CONDITIONS} besides the control); embed a subset of the conditions"
        )
    unit = embedding.settings["observation_unit"]
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
