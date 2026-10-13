"""One map's data for the report page (private; ``_html_maps``): its points in the order they
are drawn, its caption, axis titles and arrows, and what rides on each point (its population,
its label, its standardised values, its identifiers).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._wording import percent
from ..embedding import Embedding
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit.populations import MAX_COMPONENTS, VARIANCE
from ._html_map_text import combined_words, share_lines
from .retrospective import Report


def _extras(
    points: pd.DataFrame, features: list[str], populations: list[str], label: str | None
) -> dict[str, Any]:
    """Per point: its population (an index into ``populations``, -1 none), whether it carries
    the label, its standardised value of each feature (2 decimals), and its ``observation``
    and ``label_id`` (``obs``, ``lid``), which its hover names (#74)."""
    extra: dict[str, Any] = {"obs": _ids(points["observation"])} if "observation" in points else {}
    if "label_id" in points:
        extra["lid"] = _ids(points["label_id"])
    if populations and "population" in points:
        index = {name: i for i, name in enumerate(populations)}
        extra["pop"] = [index.get(v, -1) for v in points["population"]]
    if label and label in points:
        extra["lab"] = [int(bool(v)) if pd.notna(v) else 0 for v in points[label]]
    # On another table's map, an observation the explainable map left out (a missing value)
    # has none (null): drawn grey when the map is coloured by a feature.
    extra["z"] = {
        f: [round(float(v), 2) if pd.notna(v) else None for v in points[f"z:{f}"]]
        for f in features
        if f"z:{f}" in points
    }
    return extra


def _ids(values: pd.Series) -> list[Any]:
    """Identifiers as the page's JSON holds them: a whole number as an int, anything else as
    its text, a missing one as ``None``."""
    ids: list[Any] = []
    for value in values.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            ids.append(None)
        elif isinstance(value, int | float) and float(value).is_integer():
            ids.append(int(value))
        else:
            ids.append(str(value))
    return ids


# The strongest loadings named under the PCA for each axis (#75).
TOP_LOADINGS = 3


def _held(result: Report) -> str:
    """How many PCs hold ``VARIANCE`` of the variance of the declared values, as the
    populations counted them (``populations._components``), as a clause; empty without."""
    found = result.results.get("populations")
    if found is None:
        return ""
    k = found.populations.settings.get("components")
    if not k:
        return ""
    many = f"{k} or more" if k >= MAX_COMPONENTS else str(k)
    return f", {many} PCs hold {percent(VARIANCE)} of it"


def _loading_lines(loadings: pd.DataFrame, explained: tuple[float, float]) -> list[str]:
    """One line per shown PC: its ``TOP_LOADINGS`` largest loadings by size, with signs on
    the axis as drawn (a PC that holds no variance is left out)."""
    lines = []
    sides = ((1, explained[0], "to the right"), (2, explained[1], "towards the top"))
    for axis, share, side in sides:
        if share <= 0:
            continue
        column = loadings[f"pc{axis}"]
        strongest = column.abs().sort_values(ascending=False, kind="stable").index
        named = ", ".join(
            f"{loadings.at[i, 'feature']} {column[i]:+.2f}" for i in strongest[:TOP_LOADINGS]
        )
        lines.append(f"PC{axis}, strongest loadings: {named}; a feature with + is higher {side}.")
    return lines


def _set(
    embedding: Embedding,
    method: str,
    order: dict,
    values: str,
    extras: tuple[list[str], list[str], str | None] = ([], [], None),
    held: str = "",
    line: str = "",
) -> dict[str, Any] | None:
    """One map's data: its points, caption (with ``line``, how far the conditions separate in
    it, last), axis titles, arrows and what rides on each point (``_extras``)."""
    x, y = (f"{method}1", f"{method}2")
    points = embedding.points.dropna(subset=[x, y])
    if points.empty:
        return None
    points, cond = _drawing_order(points, order)
    named = [c for c in (REPEAT_COLUMN, "field") if c in points]
    labels = [" · ".join(str(v) for v in row if pd.notna(v)) for row in points[named].values]
    n = _space_words(embedding.settings)
    if method == "pc":
        arrows, caption, labels_xy = _pca(embedding, points, n, values, held)
    else:
        arrows, caption, labels_xy = _umap(embedding.settings, n, values)
    caption = f"{caption}\n{line}" if line else caption
    return {
        "x": [round(float(v), 4) for v in points[x]],
        "y": [round(float(v), 4) for v in points[y]],
        "cond": cond,
        "labels": labels,
        "caption": caption,
        "xlabel": labels_xy[0],
        "ylabel": labels_xy[1],
        "arrows": arrows,
        **_extras(points, *extras),
    }


def _drawing_order(points: pd.DataFrame, order: dict) -> tuple[pd.DataFrame, list[int]]:
    """The points in the order they are drawn and each one's condition (its place in
    ``order``; 0 for all without conditions)."""
    if CONDITION_COLUMN not in points:
        return points, [0] * len(points)
    # The control first, so the conditions are drawn over it.
    points = points.assign(_rank=points[CONDITION_COLUMN].map(order)).sort_values(
        "_rank", kind="stable"
    )
    return points, points["_rank"].astype(int).tolist()


def _space_words(settings: dict[str, Any]) -> str:
    """What a map is of: its standardised features, or the feature tables combined."""
    n = f"{len(settings['features'])} standardised features"
    shares = settings.get("table_shares")
    if shares:  # the combined space of several feature tables (owner, 2026-10-06)
        n = combined_words(settings["table_columns"])
    return n


def _pca(
    embedding: Embedding, points: pd.DataFrame, n: str, values: str, held: str
) -> tuple[list[list[Any]], str, tuple[str, str]]:
    """A PCA map's arrows, caption and axis titles."""
    first, second = embedding.explained
    loadings = embedding.loadings
    shares = embedding.settings.get("table_shares")
    arrows, drawn = _arrows(loadings, points)
    head = (
        f"PCA of {n} ({values} values): PC1 {percent(first)} and "
        f"PC2 {percent(second)} of the variance{held}; arrows: {drawn}."
    )
    # The combined map: each table's share of each axis, not loadings of a table's
    # columns under arrows of explainable ones (owner, 2026-10-06).
    shown_lines = (
        share_lines(shares, embedding.explained)
        if shares
        else _loading_lines(loadings, embedding.explained)
    )
    caption = "\n".join([head, *shown_lines])
    return arrows, caption, (f"PC1 ({percent(first)})", f"PC2 ({percent(second)})")


def _arrows(loadings: pd.DataFrame, points: pd.DataFrame) -> tuple[list[list[Any]], str]:
    """The PCA's arrows, scaled to the points, and what they draw: the loadings of the
    best-ranked features, or the 5 largest when no feature ranks."""
    shown = loadings[loadings["top"]]
    drawn = "the loadings of the best-ranked features"
    if shown.empty:
        size = np.hypot(loadings["pc1"], loadings["pc2"])
        shown = loadings.loc[size.sort_values(ascending=False).index[:5]]
        drawn = f"the {len(shown)} largest loadings, as no feature ranks"
    reach = 0.8 * float(np.nanmax(np.hypot(points["pc1"], points["pc2"])) or 1.0)
    longest = float(np.hypot(shown["pc1"], shown["pc2"]).max()) or 1.0
    arrows = [
        [r.feature, round(r.pc1 / longest * reach, 3), round(r.pc2 / longest * reach, 3)]
        for r in shown.itertuples()
    ]
    return arrows, drawn


def _umap(
    settings: dict[str, Any], n: str, values: str
) -> tuple[list[list[Any]], str, tuple[str, str]]:
    """A UMAP map's arrows (none), caption and axis titles."""
    caption = (
        f"UMAP of {n} ({values} values), "
        f"{settings['umap_points']} observations (n_neighbors {settings['n_neighbors']}, "
        f"min_dist {settings['min_dist']:g}); its axes have no units."
    )
    return [], caption, ("UMAP 1", "UMAP 2")
