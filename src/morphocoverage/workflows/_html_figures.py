"""The HTML report's figures (private; html_report): each saved as PNG and PDF and
embedded, and the data and markup of the interactive PCA and UMAP.

- feature_figures: the readout and the discovery's best-ranked features (plots).
- embeddings: the PCA and UMAP of every feature, raw and normalised to the control, as
  data for the page's script (with the static figures beside): every condition, the control
  included, in a colour of COLOURS.
- figure_block and download: a figure with its download buttons; a button that saves
  a file the page carries.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._wording import percent
from ..embedding import Embedding, embed
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from ..normalize import normalize_to_control
from ..plots_embedding import embedding_figures
from .figures import plots
from .retrospective import Report

# The categorical palette of the interactive figures, the control included (no grey: it is
# hard to see); beyond its 12 colours they repeat, and the legend and hover name each point.
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


def _save(figure: Any, folder: Path, name: str) -> str:
    """``name``.png and .pdf in ``folder``; the PNG as a data URI for the page."""
    figure.savefig(folder / f"{name}.png", dpi=150)
    figure.savefig(folder / f"{name}.pdf")
    data = base64.b64encode((folder / f"{name}.png").read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def download(path: str, label: str) -> str:
    """A button that saves the file the page carries at ``path`` (``_files``)."""
    return (
        f'<button type="button" class="dl" data-file="{html.escape(path)}">'
        f"{html.escape(label)}</button>"
    )


def figure_block(uri: str, name: str, caption: str) -> str:
    return (
        f'<figure><img src="{uri}" alt="{html.escape(caption)}" loading="lazy">'
        f"<figcaption>{html.escape(caption)} · "
        f"{download(f'figures/{name}.pdf', 'PDF')} {download(f'figures/{name}.png', 'PNG')}"
        "</figcaption></figure>"
    )


def feature_figures(result: Report, objects, fovs, folder: Path) -> list[str]:
    drawn = plots(objects, fovs, result.setup, ranking=result.discovery)
    top = result.discovery.top() if result.discovery is not None else []
    blocks = []
    for feature, parts in drawn.items():
        for index, figure in enumerate(parts, start=1):
            name = f"{feature}_{index}" if len(parts) > 1 else str(feature)
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            role = "the readout" if feature == result.setup.readout else "best-ranked feature"
            if feature in top:
                role = f"best-ranked feature {top.index(feature) + 1}"
            if len(parts) > 1:
                role += f", part {index} of {len(parts)}"
            blocks.append(figure_block(_save(figure, folder, name), name, f"{feature}: {role}"))
    return blocks


def _set(embedding: Embedding, method: str, order: dict, values: str) -> dict[str, Any] | None:
    x, y = (f"{method}1", f"{method}2")
    points = embedding.points.dropna(subset=[x, y])
    if points.empty:
        return None
    if CONDITION_COLUMN in points:
        # The control first, so the conditions are drawn over it.
        points = points.assign(_rank=points[CONDITION_COLUMN].map(order)).sort_values(
            "_rank", kind="stable"
        )
        cond = points["_rank"].astype(int).tolist()
    else:
        cond = [0] * len(points)
    named = [c for c in (REPEAT_COLUMN, "field") if c in points]
    labels = [" · ".join(str(v) for v in row if pd.notna(v)) for row in points[named].values]
    settings = embedding.settings
    n = len(settings["features"])
    if method == "pc":
        first, second = embedding.explained
        loadings = embedding.loadings
        shown = loadings[loadings["top"]]
        if shown.empty:
            size = np.hypot(loadings["pc1"], loadings["pc2"])
            shown = loadings.loc[size.sort_values(ascending=False).index[:5]]
        reach = 0.8 * float(np.nanmax(np.hypot(points[x], points[y])) or 1.0)
        longest = float(np.hypot(shown["pc1"], shown["pc2"]).max()) or 1.0
        arrows = [
            [r.feature, round(r.pc1 / longest * reach, 3), round(r.pc2 / longest * reach, 3)]
            for r in shown.itertuples()
        ]
        caption = (
            f"PCA of {n} standardised features ({values} values): PC1 {percent(first)} and "
            f"PC2 {percent(second)} of the variance; arrows: the loadings of the best-ranked "
            "features."
        )
        labels_xy = (f"PC1 ({percent(first)})", f"PC2 ({percent(second)})")
    else:
        arrows = []
        caption = (
            f"UMAP of {n} standardised features ({values} values), "
            f"{settings['umap_points']} observations (n_neighbors {settings['n_neighbors']}, "
            f"min_dist {settings['min_dist']:g}); its axes have no units."
        )
        labels_xy = ("UMAP 1", "UMAP 2")
    return {
        "x": [round(float(v), 4) for v in points[x]],
        "y": [round(float(v), 4) for v in points[y]],
        "cond": cond,
        "labels": labels,
        "caption": caption,
        "xlabel": labels_xy[0],
        "ylabel": labels_xy[1],
        "arrows": arrows,
    }


def embeddings(result: Report, objects, fovs, folder: Path) -> tuple[str, list[str]]:
    """The interactive PCA and UMAP (its data and its markup), and the static figures."""
    setup = result.setup
    top = result.discovery.top() if result.discovery is not None else ()
    tables = {"raw": objects}
    control = setup.control
    if control is not None and CONDITION_COLUMN in control:
        try:
            tables["normalised"] = normalize_to_control(
                objects, fovs, feature_cols=list(setup.columns), control=control, **setup.common
            )
        except ValueError:
            pass  # a control mean at or below 0: the raw values only
    embeddings = {
        values: embed(
            table,
            fovs,
            features=list(setup.columns),
            top=top,
            random_state=setup.random_state,
            **setup.common,
        )
        for values, table in tables.items()
    }
    first = next(iter(embeddings.values())).points
    control_value = (control or {}).get(CONDITION_COLUMN)
    names = (
        sorted_labels(first[CONDITION_COLUMN], control_value)
        if CONDITION_COLUMN in first
        else ["all"]
    )
    order = {name: i for i, name in enumerate(names)}
    colours = [COLOURS[i % len(COLOURS)] for i in range(len(names))]
    counts = (
        [int((first[CONDITION_COLUMN] == name).sum()) for name in names]
        if CONDITION_COLUMN in first
        else [len(first)]
    )
    sets = {
        values: {
            method: _set(embedding, short, order, values)
            for method, short in (("PCA", "pc"), ("UMAP", "umap"))
        }
        for values, embedding in embeddings.items()
    }
    data = {
        "conditions": [str(n) for n in names],
        "colours": colours,
        "counts": counts,
        "methods": ["PCA", "UMAP"],
        "values": list(sets),
        "sets": sets,
    }
    static = []
    for values, embedding in embeddings.items():
        for method, figure in embedding_figures(embedding, control=control).items():
            name = f"{method}_{values}"
            static.append(
                figure_block(_save(figure, folder, name), name, f"{method.upper()}, {values}")
            )
    buttons = "".join(
        f'<button type="button" data-key="method" data-value="{m}" '
        f'aria-pressed="{str(i == 0).lower()}">{m}</button>'
        for i, m in enumerate(data["methods"])
    )
    values_buttons = "".join(
        f'<button type="button" data-key="values" data-value="{v}" '
        f'aria-pressed="{str(i == 0).lower()}">{v.capitalize()}</button>'
        for i, v in enumerate(data["values"])
    )
    markup = (
        '<div class="viz" data-viz="embedding"><div class="controls">'
        f'<span class="seg" role="group" aria-label="Method">{buttons}</span>'
        + (
            f'<span class="seg" role="group" aria-label="Values">{values_buttons}</span>'
            if len(data["values"]) > 1
            else ""
        )
        + '<button type="button" class="dl full">Enlarge</button>'
        '</div><div class="legend" aria-label="Conditions (click to hide)"></div>'
        '<div class="viz-body"><div class="plot-box"><div class="svg-slot"></div>'
        '<div class="tooltip"></div></div><div class="selection" aria-live="polite"></div></div>'
        '<p class="caption note"></p>'
        '<p class="note">Both describe, they do not test: a separation is a pattern to read '
        "with the tests between conditions. On raw values a separation can be a batch between "
        "repeats, which the normalised values (each divided by its repeat's control mean) take "
        "out.</p></div>"
        '<script type="application/json" id="embedding-data">'
        + json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
        + "</script>"
    )
    return markup, static


__all__ = ["COLOURS", "download", "embeddings", "feature_figures", "figure_block"]
