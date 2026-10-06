"""The report page's maps (private; html_report): the interactive PCA and UMAP of every
feature space, as data for the page's script, with the static figures beside.

- embeddings: the PCA and UMAP of every feature, raw and normalised to the control, every
  condition, the control included, in a colour of COLOURS; past 12 conditions the control
  and the hits in colour, the others grey (``figures.highlighted``), with a search to find
  one. With other feature tables (owner, 2026-10-05), one map per table and the weighted
  combined map first (``_feature_space.space_names``), a selector between them; an other
  table is never divided by a control, so its map is the same in both values. Under each
  map, how far the conditions separate in it (``separation``), and a comparison of the
  tables under the maps.
"""

from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._palette import GREY, condition_palette
from .._wording import percent
from ..embedding import MIN_FEATURES, Embedding, embed
from ..feature_tables import COMBINED
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from ..normalize import control_mean_problem, normalize_to_control
from ..plots_embedding import embedding_figures
from ..sampling_unit.populations import MAX_COMPONENTS, VARIANCE
from ._contrasts import left_out_sentence, normalisation_problem
from ._feature_space import (
    analysed_tables,
    centred_tables,
    centring_line,
    features_words,
    separation_in_report,
    space_names,
)
from ._html_data import data_buttons, embedding_data, json_island, write_csv
from ._html_figures import figure_block, file_stem, save
from ._html_map_text import (
    NEIGHBOURS,
    VALUE_WORDS,
    centred_note,
    combined_words,
    comparison_table,
    share_lines,
)
from ._population_spaces import found_words
from ._populations import population_of
from .figures import colour_note, highlighted
from .retrospective import Report

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
    n = f"{len(settings['features'])} standardised features"
    shares = settings.get("table_shares")
    if shares:  # the combined space of several feature tables (owner, 2026-10-06)
        n = combined_words(settings["table_columns"])
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
        labels_xy = (f"PC1 ({percent(first)})", f"PC2 ({percent(second)})")
    else:
        arrows = []
        caption = (
            f"UMAP of {n} ({values} values), "
            f"{settings['umap_points']} observations (n_neighbors {settings['n_neighbors']}, "
            f"min_dist {settings['min_dist']:g}); its axes have no units."
        )
        labels_xy = ("UMAP 1", "UMAP 2")
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


def _tables(
    setup: Any, objects: pd.DataFrame, fovs: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """The values the maps show (raw, normalised to the control when they can be, with
    ``normalize`` the other tables centred there as the populations read them, audit 8, and
    the tables centred), and why the normalised ones cannot, or ``None``; any other error
    raises."""
    tables = {"raw": objects}
    control = setup.figure_control
    unnormalisable = None
    # The explainable features only are divided by a control: none in an embedding-only report.
    if control is not None and CONDITION_COLUMN in control and setup.columns:
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
    centred = centred_tables(objects, fovs, setup)  # the other tables only (owner, 2026-10-06)
    if centred is not None:
        tables["centred"] = centred
    if "normalised" in tables:  # with normalize, the tables centred, as the populations read
        tables["normalised"] = analysed_tables(tables["normalised"], fovs, setup)
    return tables, unnormalisable


def _maps(
    setup: Any,
    tables: dict[str, pd.DataFrame],
    fovs: pd.DataFrame,
    top: Any,
    carry: list[str],
) -> dict[str, dict[str, Embedding]]:
    """By values, each feature space's map (``space_names``; one with at least
    ``MIN_FEATURES`` features). An other table is never divided by a control: its map is
    computed once; the explainable features' standardised values ride on every map, for
    colouring it."""
    spaces = space_names(setup)
    several = len(spaces) > 1
    own = set(setup.feature_tables or {})
    # Without explainable features the combined space is the tables' alone: centred, it is
    # the populations' combined space (audit 8, seed 15).
    centred = own | ({COMBINED.capitalize()} if not setup.columns else set())
    maps: dict[str, dict[str, Embedding]] = {}
    for values, table in tables.items():
        maps[values] = {}
        for name, (columns, blocks) in spaces.items():
            if len(columns) < MIN_FEATURES or (values == "centred" and name not in centred):
                continue
            if name in own and name in maps["raw"] and values != "centred":
                maps[values][name] = maps["raw"][name]
                continue
            maps[values][name] = embed(
                table,
                fovs,
                features=columns,
                top=top,
                carry=carry,
                random_state=setup.random_state,
                blocks=blocks,
                standardised=list(setup.columns) if several else None,
                **setup.common,
            )
    if not maps.get("centred"):
        maps.pop("centred", None)
    for values, view in maps.items():
        source = maps["raw"] if values == "centred" else view
        first = next(iter(source.values())).points.set_index("observation")
        explained = [c for c in first.columns if c.startswith("z:")]
        for name, embedding in view.items():
            points = embedding.points.drop(columns=explained, errors="ignore")
            if name in own:
                view[name] = replace(
                    embedding, points=points.join(first[explained], on="observation")
                )
    return maps


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
    control = setup.figure_control
    tables, unnormalisable = _tables(setup, objects, fovs)
    maps = _maps(setup, tables, fovs, top, carry)
    ran = "unusual" in result.results
    line = centring_line(objects, fovs, setup, ran) if "centred" in maps else ""
    centred = centred_note(line)
    spaces = list(maps["raw"])
    several = len(spaces) > 1
    # How far the conditions separate in each space, under its map (owner, 2026-10-05).
    own = list(setup.feature_tables or {})
    separations = {
        values: separation_in_report(
            table, fovs, setup, own if values == "centred" else None, raw=objects
        )
        for values, table in tables.items()
        if values in maps
    }
    # One well per condition and repeat: a well difference reads as a condition's (audit 8).
    raw_separation = separations.get("raw")
    wells = raw_separation.well_line() if raw_separation is not None else ""
    neighbours = f"{NEIGHBOURS} {wells}".rstrip()
    ranked = list(top) + [f for f in setup.columns if f not in set(top)]
    first = maps["raw"][spaces[0]]
    features = [f for f in ranked if f"z:{f}" in first.points][:MAP_FEATURES]
    control_value = (control or {}).get(CONDITION_COLUMN)
    points = first.points
    names = (
        sorted_labels(points[CONDITION_COLUMN], control_value)
        if CONDITION_COLUMN in points
        else ["all"]
    )
    order = {name: i for i, name in enumerate(names)}
    # Past 12 conditions, the control and the hits in colour (owner, 2026-10-05).
    hits, words = highlighted(result.discovery)
    palette = condition_palette(names, control_value, hits)
    colours = [palette[name] for name in names]
    note = colour_note(palette, words)
    counts = (
        [int((points[CONDITION_COLUMN] == name).sum()) for name in names]
        if CONDITION_COLUMN in points
        else [len(points)]
    )
    # The populations counted the PCs of the declared values: said on their set only. An
    # embedding-only report has no normalised set: with normalize, its tables are centred.
    declared = "raw"
    if setup.normalize is not None:
        declared = next((v for v in ("normalised", "centred") if v in maps), "raw")
    held = _held(result)

    def built(values: str, space: str) -> dict[str, Any]:
        scored = separations[values]
        line = scored.line(space) if scored is not None else ""
        here = held if values == declared and space == spaces[0] else ""
        extras = (features, populations, label)
        return {
            method: _set(maps[values][space], short, order, values, extras, here, line)
            for method, short in (("PCA", "pc"), ("UMAP", "umap"))
        }

    # An other table's normalised map is its raw one: the page draws the raw set for it, not
    # a second copy (``_html_plotly_script``), which keeps room for its centred view.
    sets = {
        values: {
            space: built(values, space)
            for space in view
            if not (values == "normalised" and space in own)
        }
        if several
        else built(values, spaces[0])
        for values, view in maps.items()
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
                "defined": r.defined_by or found_words(r._asdict()),
                "kind": r.kind,
            }
            for r in result.results["populations"].populations.table.itertuples()
        ]
        if found is not None
        else [],
        "label": label,
        "values": list(sets),
        "sets": sets,
        **({"tables": spaces} if several else {}),
    }
    # Each map once in the files: an other table's is the same in both values; the centred
    # view is in the interactive maps only, which keeps the page within its budget.
    taken: set[str] = set()
    stems = {space: file_stem(space, taken) for space in spaces}
    shown = [
        (values, space, embedding)
        for values, view in maps.items()
        for space, embedding in view.items()
        if values == "raw" or (values == "normalised" and space not in own)
    ]
    data_files = embedding_data(
        {f"{values}, {s}" if several else values: e for values, s, e in shown}, folder, setup
    )
    comparison = comparison_table(separations[declared]) if several else ""
    if separations[declared] is not None:
        path = write_csv(separations[declared].table, folder, "separation.csv")
        data_files.append((path, "Neighbours sharing the condition, per repeat (CSV)"))
    static = []
    for values, space, embedding in shown:
        drawn = embedding_figures(embedding, control=control, highlight=hits)
        key = f"{values}_{stems[space]}" if several else values
        for method, figure in drawn.items():
            name = f"{method}_{key}"
            caption = f"{method.upper()}, {values}" + (f", {space}" if several else "")
            caption += f". {note}" if note else ""
            static.append(figure_block(save(figure, folder, name), name, caption))
    buttons = "".join(
        f'<button type="button" data-key="method" data-value="{m}" '
        f'aria-pressed="{str(i == 0).lower()}">{m}</button>'
        for i, m in enumerate(data["methods"])
    )
    values_buttons = "".join(
        f'<button type="button" data-key="values" data-value="{v}" '
        f'aria-pressed="{str(i == 0).lower()}">{VALUE_WORDS.get(v, v.capitalize())}</button>'
        for i, v in enumerate(data["values"])
    )
    table_buttons = "".join(
        f'<button type="button" data-key="table" data-value="{html.escape(t)}" '
        f'aria-pressed="{str(i == 0).lower()}">{html.escape(t)}</button>'
        for i, t in enumerate(spaces)
    )
    markup = (
        '<div class="viz" data-viz="embedding"><div class="controls">'
        + (
            f'<span class="seg" role="group" aria-label="Feature table">{table_buttons}</span>'
            if several
            else ""
        )
        + f'<span class="seg" role="group" aria-label="Method">{buttons}</span>'
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
        + comparison
        + f'<p class="tools">Data behind the maps: {data_buttons(data_files)}</p>'
        f'<p class="note">{html.escape(neighbours)}</p>'
        + (f'<p class="note">{html.escape(centred)}</p>' if "centred" in maps else "")
        + '<p class="note">Both describe, they do not test: a separation is a pattern to read '
        "with the tests between conditions. On raw values a separation can be a batch between "
        + _batch_words(setup, fovs, unnormalisable, "centred" in maps)
        + '<script type="application/json" id="embedding-data">'
        + json_island(data)
        + "</script>"
    )
    return markup, static


def maps_body(setup: Any, viz: str, static: list[str]) -> str:
    """The maps' section: every feature at once (with other feature tables, a map per table,
    the combined one first), the interactive maps, the static figures folded."""
    words = f"every {setup.unit_word} a point"
    intro = f"All {len(setup.columns)} features at once, {words}."
    if setup.feature_tables and len(space_names(setup)) == 1:  # one table, nothing else
        intro = f"All {features_words(setup)} at once, {words}."
    elif setup.feature_tables:
        intro = (
            f"All {features_words(setup)} ({', '.join(setup.feature_tables)}) at once, "
            f"{words}: the weighted combined map first, then one per table."
        )
    return (
        f"<p>{html.escape(intro)}</p>{viz}"
        "<details><summary>Static PCA and UMAP (PNG and PDF)</summary>"
        f'<div class="figures">{"".join(static)}</div></details>'
    )


def _batch_words(setup: Any, fovs: pd.DataFrame, unnormalisable: str | None, centred: bool) -> str:
    """The end of the maps' note: which view takes a batch between repeats out."""
    if not setup.columns:  # an embedding-only report: the tables are never divided
        # The map's view; the separation by condition compares within each repeat already.
        view = (
            "the centred view takes out of the map"
            if centred
            else "a control would let the centred view take out of the map"
        )
        return f"repeats, which {view}.</p></div>"
    if unnormalisable is None:
        return f"repeats, which the normalised values ({_divided(setup, fovs)}) take out.</p></div>"
    return (
        "repeats; values normalised to each repeat's control mean would take it out, "
        f"but they are not shown: {html.escape(unnormalisable)}.</p></div>"
    )


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


__all__ = ["MAP_FEATURES", "embeddings", "maps_body"]
