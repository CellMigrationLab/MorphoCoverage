"""The HTML report's figures (private; html_report): each saved as PNG and PDF and
embedded, and the data and markup of the interactive PCA and UMAP.

- feature_figures: the discovery's best-ranked features, the readout when none ranks
  (plots), each with the CSV of what it draws (``_html_data``).
- embeddings: the PCA and UMAP of every feature, raw and normalised to the control, as
  data for the page's script (with the static figures beside): every condition, the control
  included, in a colour of COLOURS; past 12 conditions the control and the hits in colour,
  the others grey (``figures.highlighted``), with a search to find one.
- figure_block: a figure with its download buttons (``_html_data.download``).
"""

from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._palette import COLOURS, GREY, condition_palette
from .._wording import percent
from ..embedding import Embedding, embed
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from ..normalize import control_mean_problem, normalize_to_control
from ..plots_embedding import embedding_figures
from ..plots_pairwise import PAIR_NOTE, PAIRED_NOTE, PANEL_KEYS, PANEL_TITLES
from ..sampling_unit.populations import MAX_COMPONENTS, VARIANCE
from ._contrasts import left_out_sentence, normalisation_problem
from ._html_data import data_buttons, download, embedding_data, feature_data, json_island
from ._html_plotly import feature_specs, interactive_caption
from ._populations import population_of
from ._rare_share import with_share
from ._shared import check_workflow_inputs
from .figures import colour_note, highlighted, plots
from .retrospective import Report

# The interactive figures use the static figures' palette (``_palette.condition_palette``,
# the control included); past its 12 colours the control and the hits keep theirs and the
# others are grey, and the legend and hover name each point.


# The widest a static PNG is drawn, in pixels: a split figure of 12 conditions is 27 inches
# wide, which at 150 dpi made PNGs of 4113 pixels; past this width the dpi is lowered.
PNG_PIXELS = 2400


def _save(figure: Any, folder: Path, name: str) -> str:
    """``name``.png and .pdf in ``folder``; the PNG's path as the page refers to it (given
    its data when the page carries it, ``_html_data.inline_images``)."""
    width = max(figure.get_size_inches())
    figure.savefig(folder / f"{name}.png", dpi=min(150, PNG_PIXELS / width))
    figure.savefig(folder / f"{name}.pdf")
    return f"figures/{name}.png"


def figure_block(uri: str, name: str, caption: str) -> str:
    """A static figure with its caption and its PDF and PNG download links."""
    return (
        f'<figure><img src="{uri}" data-file="{uri}" alt="{html.escape(caption)}" '
        'loading="lazy">'
        f"<figcaption>{html.escape(caption)} · "
        f"{download(f'figures/{name}.pdf', 'PDF')} {download(f'figures/{name}.png', 'PNG')}"
        "</figcaption></figure>"
    )


def condition_colours(
    objects: pd.DataFrame, fovs: pd.DataFrame, control: Any, highlight: Any = ()
) -> dict:
    """Every condition's colour, by its place among all of them (the control first), the
    same in every interactive figure; past 12, the ``highlight`` ones in colour, the rest grey
    (a split feature figure colours its own, ``plots.plots_of_data``)."""
    values = [
        v for table in (fovs, objects) if CONDITION_COLUMN in table for v in table[CONDITION_COLUMN]
    ]
    names = sorted_labels([v for v in values if pd.notna(v)], control)
    return condition_palette(names, control, highlight)


def file_stem(name: Any, taken: set[str]) -> str:
    """``name`` as a file name and page id (letters, digits, ``-_.``), made unique among
    ``taken`` with a counter (``a b`` and ``a_b`` are two features, two files); added to
    ``taken``."""
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)) or "feature"
    unique, n = stem, 1
    while unique in taken:
        n += 1
        unique = f"{stem}_{n}"
    taken.add(unique)
    return unique


def feature_figures(
    result: Report,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    folder: Path,
    legends: dict[str, str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The best-ranked features: each figure's block (interactive, with its static PNG and
    PDF and its data as CSV, and the feature's text for a figure legend folded, ``legends``
    as markup by feature) and the specs the page's script draws (``_html_plotly``)."""
    legends = legends or {}
    setup = result.setup
    drawn = plots(objects, fovs, setup, ranking=result.discovery)
    checked, checked_fovs, _ = check_workflow_inputs(objects, fovs, setup)
    views = _views(setup, checked, checked_fovs)
    figures_by_view = {views[0][0]: drawn}
    for key, view in views[1:]:
        figures_by_view[key] = plots(objects, fovs, view, features=list(drawn))
    discovery = result.discovery
    top = discovery.top(len(discovery.ranking)) if discovery is not None else []
    prepared = {key: with_share(view.prepared(checked, checked_fovs), view) for key, view in views}
    blocks, names, explained = [], {}, False
    taken: set[str] = set()
    for feature, parts in drawn.items():
        stem = file_stem(feature, taken)
        # The data each figure draws, to draw it again outside (owner, 2026-10-02).
        data = feature_data(
            stem,
            feature,
            prepared,
            {key: figures_by_view[key][feature] for key in prepared if key in figures_by_view},
            checked_fovs,
            dict(views),
            folder,
        )
        for index, figure in enumerate(parts, start=1):
            name = file_stem(f"{stem}_{index}", taken) if len(parts) > 1 else stem
            names[f"{feature}#{index}"] = name
            rank = f"best-ranked feature {top.index(feature) + 1}" if feature in top else ""
            role = rank or "a declared feature"
            if feature == result.setup.readout:
                role = f"{rank}, the readout" if rank else "the readout"
            if len(parts) > 1:
                role += f", part {index} of {len(parts)}"
            uri = _save(figure, folder, name)
            label = f"{feature}: {role}"
            pairs = getattr(figure, "pairwise", None)
            paired = pairs is not None and len(pairs) > 0
            pair = (
                '<div class="pair-row">'
                + "".join(
                    f'<div class="panel"><p class="panel-title"><b>{letter}</b> {title}'
                    f'<span class="panel-key">{html.escape(PANEL_KEYS[letter])}</span></p>'
                    f'<div class="pl-pair" data-panel="{letter.lower()}"></div></div>'
                    for letter, title in PANEL_TITLES.items()
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
                + (
                    "<details><summary>For a figure legend</summary>"
                    f'<p class="legend-box">{legends[feature]}</p></details>'
                    if feature in legends and index == 1
                    else ""
                )
                + f'<p class="note sampled"></p><p class="tools">'
                f"{download(f'figures/{name}.pdf', 'PDF')} {download(f'figures/{name}.png', 'PNG')}"
                f" {data_buttons(data)}"
                "</p><details><summary>The static figure (as in the PDF)</summary>"
                f'<figure><img src="{uri}" data-file="{uri}" alt="{html.escape(label)}" '
                'loading="lazy"></figure>'
                "</details></div>"
            )
    control = (setup.figure_control or {}).get(CONDITION_COLUMN)
    colours = condition_colours(objects, fovs, control, highlighted(result.discovery)[0])
    by_view = {}
    for key, view in views:
        by_view[key] = feature_specs(
            figures_by_view[key], prepared[key], checked_fovs, view, colours, names
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


def _views(setup: Any, objects: pd.DataFrame, fovs: pd.DataFrame) -> list[tuple[str, Any]]:
    """The values the feature figures can show: those the setup declares first, and, when a
    control condition is declared, the others (raw or normalised to each repeat's control,
    unless ``control_mean_problem`` says why not: ``objects`` and ``fovs`` are checked, as
    ``plots`` prepares them)."""
    declared = "normalised" if setup.normalize is not None else "raw"
    views = [(declared, setup)]
    if CONDITION_COLUMN not in (setup.figure_control or {}):
        return views
    if declared == "raw" and _unnormalisable(setup, objects, fovs) is not None:
        return views  # the declared values only
    other = replace(setup, normalize=None if setup.normalize else "control_mean")
    return [*views, ("raw" if declared == "normalised" else "normalised", other)]


def _unnormalisable(setup: Any, objects: pd.DataFrame, fovs: pd.DataFrame) -> str | None:
    """Why ``setup.prepared`` cannot normalise these checked tables to the control (a repeat
    without control objects, a control mean zero, negative or undefined), or ``None``; any
    other error raises."""
    if setup.own_references is not None:  # several controls: each to its own
        return normalisation_problem(setup, objects, fovs)
    return control_mean_problem(
        objects,
        fovs,
        feature_cols=list(setup.columns),
        control=setup.control,
        repeat_col=setup.repeat_col(fovs),
        **setup.common,
    )


# The most features a map can be coloured by and a selection described with.
MAP_FEATURES = 60


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
    extra["z"] = {
        f: [round(float(v), 2) for v in points[f"z:{f}"]] for f in features if f"z:{f}" in points
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
        drawn = "the loadings of the best-ranked features"
        if shown.empty:
            size = np.hypot(loadings["pc1"], loadings["pc2"])
            shown = loadings.loc[size.sort_values(ascending=False).index[:5]]
            drawn = f"the {len(shown)} largest loadings, as no feature ranks"
        reach = 0.8 * float(np.nanmax(np.hypot(points[x], points[y])) or 1.0)
        longest = float(np.hypot(shown["pc1"], shown["pc2"]).max()) or 1.0
        arrows = [
            [r.feature, round(r.pc1 / longest * reach, 3), round(r.pc2 / longest * reach, 3)]
            for r in shown.itertuples()
        ]
        head = (
            f"PCA of {n} standardised features ({values} values): PC1 {percent(first)} and "
            f"PC2 {percent(second)} of the variance{held}; arrows: {drawn}."
        )
        caption = "\n".join([head, *_loading_lines(loadings, embedding.explained)])
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


def _colour_menu(data: dict[str, Any], n_features: int) -> str:
    """The "Colour by" menu: condition, population (when found) or a feature, and how many of
    the ``n_features`` it offers when it cannot offer them all (``MAP_FEATURES``)."""
    options = ['<option value="condition">condition</option>']
    if data["populations"]:
        options.append('<option value="population">population</option>')
    options += [
        f'<option value="{html.escape(f)}">{html.escape(f)}</option>' for f in data["features"]
    ]
    shown = len(data["features"])
    limit = (
        f'<span class="note">Colour by and the selection use the {shown} best-ranked of '
        f"{n_features} features.</span>"
        if shown < n_features
        else ""
    )
    return (
        '<label class="colour-by-label">Colour by <select class="colour-by">'
        + "".join(options)
        + "</select></label>"
        + limit
    )


def _find(names: list[str], colours: list[str]) -> str:
    """Past 12 conditions, a search box to pick one and see it highlighted on the map."""
    if GREY not in colours:
        return ""
    options = "".join(f'<option value="{html.escape(n)}"></option>' for n in names)
    return (
        '<label class="find-label">Find a condition <input type="search" class="find" '
        'list="embedding-conditions" placeholder="name" aria-label="Find a condition">'
        f'</label><datalist id="embedding-conditions">{options}</datalist>'
    )


def embeddings(
    result: Report, objects: pd.DataFrame, fovs: pd.DataFrame, folder: Path
) -> tuple[str, list[str]]:
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
    control = setup.figure_control
    unnormalisable = None
    if control is not None and CONDITION_COLUMN in control:
        if setup.own_references is None:
            arguments = {"feature_cols": list(setup.columns), "control": control, **setup.common}
            unnormalisable = control_mean_problem(objects, fovs, **arguments)
            if unnormalisable is None:
                tables["normalised"] = normalize_to_control(objects, fovs, **arguments)
        else:  # several controls: each condition to its own
            unnormalisable = normalisation_problem(setup, objects, fovs)
            if unnormalisable is None:
                normalised = replace(setup, normalize="control_mean")
                tables["normalised"] = normalised.prepared(objects, fovs)
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
    # Past 12 conditions, the control and the hits in colour (owner, 2026-10-05).
    hits, words = highlighted(result.discovery)
    palette = condition_palette(names, control_value, hits)
    colours = [palette[name] for name in names]
    note = colour_note(palette, words)
    counts = (
        [int((first[CONDITION_COLUMN] == name).sum()) for name in names]
        if CONDITION_COLUMN in first
        else [len(first)]
    )
    # The populations counted the PCs of the declared values: said on their set only.
    declared = "normalised" if setup.normalize is not None else "raw"
    held = _held(result)
    sets = {
        values: {
            method: _set(
                embedding,
                short,
                order,
                values,
                (features, populations, label),
                held if values == declared else "",
            )
            for method, short in (("PCA", "pc"), ("UMAP", "umap"))
        }
        for values, embedding in embeddings.items()
    }
    data = {
        "conditions": [str(n) for n in names],
        "colours": colours,
        "grey": GREY,
        "counts": counts,
        "methods": ["PCA", "UMAP"],
        "features": features,
        "populations": [
            {
                "name": r.population,
                "share": round(float(r.share), 4),
                "defined": r.defined_by,
                "kind": r.kind,
            }
            for r in result.results["populations"].populations.table.itertuples()
        ]
        if found is not None
        else [],
        "label": label,
        "values": list(sets),
        "sets": sets,
    }
    data_files = embedding_data(embeddings, folder)
    static = []
    for values, embedding in embeddings.items():
        drawn = embedding_figures(embedding, control=control, highlight=hits)
        for method, figure in drawn.items():
            name = f"{method}_{values}"
            caption = f"{method.upper()}, {values}" + (f". {note}" if note else "")
            static.append(figure_block(_save(figure, folder, name), name, caption))
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
        + _colour_menu(data, len(setup.columns))
        + _find(data["conditions"], colours)
        + '<button type="button" class="dl full">Enlarge</button>'
        '<span class="note">Click an entry in the legend to hide it, double-click to show '
        "it alone; draw around points (the lasso is on) to see which conditions and "
        "populations they come from and which features set them apart; colour the map by a "
        "feature to see where its high and low values sit; ◯ rings a population.</span></div>"
        '<div class="viz-body"><div class="pl-embed"></div>'
        '<div class="selection" aria-live="polite"></div></div>'
        '<p class="caption note" style="white-space: pre-line"></p>'
        + (f'<p class="note">{html.escape(note)}</p>' if note else "")
        + f'<p class="tools">Data behind the maps: {data_buttons(data_files)}</p>'
        '<p class="note">Both describe, they do not test: a separation is a pattern to read '
        "with the tests between conditions. On raw values a separation can be a batch between "
        + (
            f"repeats, which the normalised values ({_divided(setup, fovs)}) take out.</p></div>"
            if unnormalisable is None
            else "repeats; values normalised to each repeat's control mean would take it out, "
            f"but they are not shown: {html.escape(unnormalisable)}.</p></div>"
        )
        + '<script type="application/json" id="embedding-data">'
        + json_island(data)
        + "</script>"
    )
    return markup, static


def _divided(setup: Any, fovs: pd.DataFrame) -> str:
    """How the normalised values are divided; with several controls, each condition by its
    own control, the conditions left out named once (owner, 2026-10-05)."""
    if setup.own_references is None:
        return "each divided by its repeat's control mean"
    left = left_out_sentence(setup, fovs)
    return (
        "each condition divided by its own control's mean in its repeat, the raw values being "
        f"where levels across experiments are read{'; ' + left if left else ''}"
    )


__all__ = [
    "COLOURS",
    "condition_colours",
    "embeddings",
    "feature_figures",
    "figure_block",
]
