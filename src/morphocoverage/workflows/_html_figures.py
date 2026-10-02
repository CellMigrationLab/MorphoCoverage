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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._wording import percent
from ..embedding import Embedding, embed
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from ..normalize import normalize_to_control
from ..plots_embedding import embedding_figures
from ..plots_pairwise import PAIR_NOTE, PAIRED_NOTE
from ._html_plotly import feature_specs, interactive_caption
from ._populations import population_of
from ._rare_share import with_share
from ._shared import check_workflow_inputs
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


def condition_colours(objects: pd.DataFrame, fovs: pd.DataFrame, control: Any) -> dict:
    """Every condition's colour, by its place among all of them (the control first), the
    same in every interactive figure."""
    values = [
        v for table in (fovs, objects) if CONDITION_COLUMN in table for v in table[CONDITION_COLUMN]
    ]
    names = sorted_labels([v for v in values if pd.notna(v)], control)
    return {name: COLOURS[i % len(COLOURS)] for i, name in enumerate(names)}


def feature_figures(
    result: Report, objects, fovs, folder: Path
) -> tuple[list[str], list[dict[str, Any]]]:
    """The readout and best-ranked features: each figure's block (interactive, with its
    static PNG and PDF) and the specs the page's script draws (``_html_plotly``)."""
    setup = result.setup
    drawn = plots(objects, fovs, setup, ranking=result.discovery)
    views = _views(setup)
    figures_by_view = {views[0][0]: drawn}
    for key, view in views[1:]:
        try:
            figures_by_view[key] = plots(objects, fovs, view, features=list(drawn))
        except ValueError:
            views = views[:1]  # a control mean at or below 0: the declared values only
    discovery = result.discovery
    top = discovery.top(len(discovery.ranking)) if discovery is not None else []
    blocks, names, explained = [], {}, False
    for feature, parts in drawn.items():
        for index, figure in enumerate(parts, start=1):
            name = f"{feature}_{index}" if len(parts) > 1 else str(feature)
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            names[f"{feature}#{index}"] = name
            rank = f"best-ranked feature {top.index(feature) + 1}" if feature in top else ""
            role = rank or "a declared feature"
            if feature == result.setup.readout:
                role = "the readout"
            if len(parts) > 1:
                role += f", part {index} of {len(parts)}"
            uri = _save(figure, folder, name)
            label = f"{feature}: {role}"
            pairs = getattr(figure, "pairwise", None)
            paired = pairs is not None and len(pairs) > 0
            pair = (
                '<div class="pair-row">'
                + "".join(
                    f'<div class="panel"><p class="panel-title"><b>{letter}</b> {title}</p>'
                    f'<div class="pl-pair" data-panel="{letter.lower()}"></div></div>'
                    for letter, title in (
                        ("B", "d · randomization test"),
                        ("C", "paired t-test, repeat means"),
                    )
                )
                + "</div>"
                if paired
                else ""
            )
            letter_a = '<p class="panel-title"><b>A</b></p>' if paired else ""
            caption = f'<span class="caption">{html.escape(interactive_caption(figure))}</span>'
            # Panels B and C read the same in every figure: explained under the first.
            legend = (
                f"<p>(A) {caption}</p>"
                + (
                    "<p>(B) and (C): as in the first figure above.</p>"
                    if explained
                    else f"<p>(B) {html.escape(PAIR_NOTE)}</p><p>(C) {html.escape(PAIRED_NOTE)}</p>"
                )
                if paired
                else f"<p>{caption}</p>"
            )
            explained = explained or paired
            switch = "".join(
                f'<button type="button" data-key="values" data-value="{key}" '
                f'aria-pressed="{str(key == views[0][0]).lower()}">{key.capitalize()}</button>'
                for key, _ in views
            )
            switch = (
                f'<span class="seg" role="group" aria-label="Values">{switch}</span>'
                if len(views) > 1
                else ""
            )
            blocks.append(
                f'<div class="feature" data-spec="{html.escape(name)}">'
                f'<div class="feature-head"><h4>{html.escape(label)}</h4><span class="tools">'
                f'{switch}<button type="button" class="dl full">Enlarge</button></span></div>'
                f'<div class="panel">{letter_a}'
                f'<div class="pl-plot" id="pl-{html.escape(name)}"></div></div>{pair}'
                f'<div class="figure-legend"><p><strong>Legend:</strong></p>{legend}</div>'
                f'<p class="note sampled"></p><p class="tools">'
                f"{download(f'figures/{name}.pdf', 'PDF')} {download(f'figures/{name}.png', 'PNG')}"
                "</p><details><summary>The static figure (as in the PDF)</summary>"
                f'<figure><img src="{uri}" alt="{html.escape(label)}" loading="lazy"></figure>'
                "</details></div>"
            )
    checked, checked_fovs, _ = check_workflow_inputs(objects, fovs, setup)
    control = (setup.control or {}).get(CONDITION_COLUMN)
    colours = condition_colours(objects, fovs, control)
    by_view = {}
    for key, view in views:
        prepared = with_share(view.prepared(checked, checked_fovs), view)
        by_view[key] = feature_specs(
            figures_by_view[key], prepared, checked_fovs, view, colours, names
        )
    first = views[0][0]
    specs = [
        {
            "id": spec["id"],
            "feature": spec["feature"],
            "current": first,
            "views": {key: by_view[key][i] for key in by_view},
        }
        for i, spec in enumerate(by_view[first])
    ]
    return blocks, specs


def _views(setup: Any) -> list[tuple[str, Any]]:
    """The values the feature figures can show: those the setup declares first, and, when a
    control condition is declared, the others (raw or normalised to each repeat's control)."""
    declared = "normalised" if setup.normalize is not None else "raw"
    views = [(declared, setup)]
    if CONDITION_COLUMN not in (setup.control or {}):
        return views
    other = replace(setup, normalize=None if setup.normalize else "control_mean")
    return [*views, ("raw" if declared == "normalised" else "normalised", other)]


# The most features a map can be coloured by and a selection described with.
MAP_FEATURES = 60


def _extras(points: pd.DataFrame, features: list[str], populations: list[str], label: str | None):
    """Per point: its population (an index into ``populations``, -1 none), whether it carries
    the label, and its standardised value of each feature (2 decimals)."""
    extra: dict[str, Any] = {}
    if populations and "population" in points:
        index = {name: i for i, name in enumerate(populations)}
        extra["pop"] = [index.get(v, -1) for v in points["population"]]
    if label and label in points:
        extra["lab"] = [int(bool(v)) if pd.notna(v) else 0 for v in points[label]]
    extra["z"] = {
        f: [round(float(v), 2) for v in points[f"z:{f}"]] for f in features if f"z:{f}" in points
    }
    return extra


def _set(
    embedding: Embedding,
    method: str,
    order: dict,
    values: str,
    extras: tuple[list[str], list[str], str | None] = ([], [], None),
) -> dict[str, Any] | None:
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
        **_extras(points, *extras),
    }


def _colour_menu(data: dict[str, Any]) -> str:
    """The "Colour by" menu: condition, population (when found) or a feature."""
    options = ['<option value="condition">condition</option>']
    if data["populations"]:
        options.append('<option value="population">population</option>')
    options += [
        f'<option value="{html.escape(f)}">{html.escape(f)}</option>' for f in data["features"]
    ]
    return (
        '<label class="colour-by-label">Colour by <select class="colour-by">'
        + "".join(options)
        + "</select></label>"
    )


def embeddings(result: Report, objects, fovs, folder: Path) -> tuple[str, list[str]]:
    """The interactive PCA and UMAP (its data and its markup), and the static figures."""
    setup = result.setup
    top = result.discovery.top() if result.discovery is not None else ()
    # The populations of question 10 and the rare label ride on the points (owner, 2026-10-02).
    found = population_of(result.results.get("populations"), objects)
    if found is not None:
        objects = objects.assign(population=found)
    populations = list(result.results["populations"].populations.names) if found is not None else []
    label = setup.rare_label if setup.rare_label in objects else None
    carry = [c for c in ("population", label) if c]
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
            carry=carry,
            random_state=setup.random_state,
            **setup.common,
        )
        for values, table in tables.items()
    }
    ranked = list(top) + [f for f in setup.columns if f not in set(top)]
    features = [f for f in ranked if f in next(iter(embeddings.values())).settings["features"]][
        :MAP_FEATURES
    ]
    first = next(iter(embeddings.values())).points
    control_value = (control or {}).get(CONDITION_COLUMN)
    names = (
        sorted_labels(first[CONDITION_COLUMN], control_value)
        if CONDITION_COLUMN in first
        else ["all"]
    )
    order = {name: i for i, name in enumerate(names)}
    shared = condition_colours(objects, fovs, control_value)
    colours = [shared.get(name, COLOURS[i % len(COLOURS)]) for i, name in enumerate(names)]
    counts = (
        [int((first[CONDITION_COLUMN] == name).sum()) for name in names]
        if CONDITION_COLUMN in first
        else [len(first)]
    )
    sets = {
        values: {
            method: _set(embedding, short, order, values, (features, populations, label))
            for method, short in (("PCA", "pc"), ("UMAP", "umap"))
        }
        for values, embedding in embeddings.items()
    }
    data = {
        "conditions": [str(n) for n in names],
        "colours": colours,
        "counts": counts,
        "methods": ["PCA", "UMAP"],
        "features": features,
        "populations": [
            {"name": r.population, "share": round(float(r.share), 4), "defined": r.defined_by}
            for r in result.results["populations"].populations.table.itertuples()
        ]
        if found is not None
        else [],
        "label": label,
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
        + _colour_menu(data)
        + '<button type="button" class="dl full">Enlarge</button>'
        '<span class="note">Click an entry in the legend to hide it, double-click to show '
        "it alone; draw around points (the lasso is on) to see which conditions and "
        "populations they come from and which features set them apart; colour the map by a "
        "feature to see where its high and low values sit; ◯ rings a population.</span></div>"
        '<div class="viz-body"><div class="pl-embed"></div>'
        '<div class="selection" aria-live="polite"></div></div>'
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


__all__ = [
    "COLOURS",
    "condition_colours",
    "download",
    "embeddings",
    "feature_figures",
    "figure_block",
]
