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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from .._palette import GREY, condition_palette
from ..embedding import MIN_FEATURES, Embedding, embed
from ..feature_tables import COMBINED
from ..metadata import CONDITION_COLUMN, sorted_labels
from ..normalize import control_mean_problem, normalize_to_control
from ..plots_embedding import embedding_figures
from ._acquisition_gap import gap_maps_line
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
from ._html_map_controls import colour_menu, find_box, values_controls
from ._html_map_sets import _held, _set
from ._html_map_text import (
    METHOD_WORDS,
    NEIGHBOURS,
    ONE_CONDITION,
    VALUE_HELP,
    VIEW_ORDER,
    centred_note,
    comparison_table,
)
from ._population_spaces import found_words
from ._populations import population_of
from .figures import colour_note, highlighted
from .retrospective import Report

# The most features a map can be coloured by and a selection described with.
MAP_FEATURES = 60


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
    _carry_explained(maps, own)
    return maps


def _carry_explained(maps: dict[str, dict[str, Embedding]], own: set[str]) -> None:
    """On each other table's map, the explainable features' standardised values of the
    first map of its values (the raw one's for the centred maps), in place."""
    for values, view in maps.items():
        source = maps["raw"] if values == "centred" else view
        first = next(iter(source.values())).points.set_index("observation")
        explained = [c for c in first.columns if c.startswith("z:")]
        for name, embedding in view.items():
            # fallback: the explained columns are present only in some maps
            points = embedding.points.drop(columns=explained, errors="ignore")
            if name in own:
                view[name] = replace(
                    embedding, points=points.join(first[explained], on="observation")
                )


class _Legend(NamedTuple):
    """The conditions of the maps as the page draws them: their names (the control first),
    their order, the hits kept in colour past 12, the colours, the colour note and how many
    points each has."""

    names: list[Any]
    order: dict[Any, int]
    hits: list[Any]
    colours: list[str]
    note: str
    counts: list[int]


@dataclass(frozen=True)
class _Sets:
    """What every map's data shares: the separations by values, the values whose PCs the
    populations counted (``declared``), the first space, the held clause, the extras riding
    on the points and the conditions' order."""

    separations: dict[str, Any]
    declared: str
    first_space: str
    held: str
    extras: tuple[list[str], list[str], str | None]
    order: dict[Any, int]


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
    centring = _centring_note(result, objects, fovs, "centred" in maps)
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
    neighbours = _neighbours(separations.get("raw"), fovs)
    ranked = list(top) + [f for f in setup.columns if f not in set(top)]
    first = maps["raw"][spaces[0]]
    features = [f for f in ranked if f"z:{f}" in first.points][:MAP_FEATURES]
    legend = _legend(first.points, (control or {}).get(CONDITION_COLUMN), result.discovery)
    declared = _declared(setup, maps)
    shared = _Sets(
        separations,
        declared,
        spaces[0],
        _held(result),
        (features, populations, label),
        legend.order,
    )
    sets = {values: _view_sets(maps, shared, values, own) for values in maps}
    data = _page_data(legend, features, _population_rows(result, found), label, sets, spaces)
    shown, data_files = _files(maps, own, separations[declared], folder, setup)
    comparison = comparison_table(separations[declared]) if several else ""
    static = _static(shown, spaces, control, legend, folder)
    batch = _batch_words(setup, fovs, unnormalisable, "centred" in maps)
    notes = (neighbours, centring, batch)
    markup = _controls(data, spaces, len(setup.columns)) + _body(
        data, legend.note, comparison, data_files, notes
    )
    return markup, static


def _centring_note(result: Report, objects: pd.DataFrame, fovs: pd.DataFrame, centred: bool) -> str:
    """The note under the maps on centring: what the centred view takes out (with it, as a
    paragraph even when empty), else the acquisition gaps' line when there is one."""
    setup = result.setup
    ran = "unusual" in result.results
    line = centring_line(objects, fovs, setup, ran) if centred else ""
    contrast = result.results.get("contrast")
    gaps = contrast.settings.get("acquisition_gaps", []) if contrast is not None else []
    with_centred = " ".join(p for p in (centred_note(line), gap_maps_line(gaps)) if p)
    uncentred = gap_maps_line(gaps, centred=False)
    if centred:
        return f'<p class="note">{html.escape(with_centred)}</p>'
    return f'<p class="note">{html.escape(uncentred)}</p>' if uncentred else ""


def _neighbours(raw_separation: Any, fovs: pd.DataFrame) -> str:
    """The note on the neighbours under the maps, with its line on wells; with one
    condition, that there is nothing to separate."""
    # One well per condition and repeat: a well difference reads as a condition's (audit 8).
    wells = raw_separation.well_line() if raw_separation is not None else ""
    neighbours = f"{NEIGHBOURS} {wells}".rstrip()
    one = CONDITION_COLUMN not in fovs or fovs[CONDITION_COLUMN].nunique() < 2
    if raw_separation is None and one:
        neighbours = ONE_CONDITION  # nothing to separate, said with why (audit 10, C29)
    return neighbours


def _legend(points: pd.DataFrame, control_value: Any, discovery: Any) -> _Legend:
    """The conditions as the maps draw and colour them (``_Legend``)."""
    names = (
        sorted_labels(points[CONDITION_COLUMN], control_value)
        if CONDITION_COLUMN in points
        else ["all"]
    )
    order = {name: i for i, name in enumerate(names)}
    # Past 12 conditions, the control and the hits in colour (owner, 2026-10-05).
    hits, words = highlighted(discovery)
    palette = condition_palette(names, control_value, hits)
    colours = [palette[name] for name in names]
    note = colour_note(palette, words)
    counts = (
        [int((points[CONDITION_COLUMN] == name).sum()) for name in names]
        if CONDITION_COLUMN in points
        else [len(points)]
    )
    return _Legend(names, order, hits, colours, note, counts)


def _declared(setup: Any, maps: dict[str, dict[str, Embedding]]) -> str:
    """The values whose PCs the populations counted, said on their set only. An
    embedding-only report has no normalised set: with normalize, its tables are centred."""
    declared = "raw"
    if setup.normalize is not None:
        declared = next((v for v in ("normalised", "centred") if v in maps), "raw")
    return declared


def _built(maps: dict[str, dict[str, Embedding]], shared: _Sets, values: str, space: str) -> dict:
    """The PCA's and the UMAP's data of one space in one values."""
    scored = shared.separations[values]
    line = scored.line(space) if scored is not None else ""
    here = shared.held if values == shared.declared and space == shared.first_space else ""
    return {
        method: _set(maps[values][space], short, shared.order, values, shared.extras, here, line)
        for method, short in (("PCA", "pc"), ("UMAP", "umap"))
    }


def _view_sets(
    maps: dict[str, dict[str, Embedding]], shared: _Sets, values: str, own: list[str]
) -> dict:
    """The maps' data of one values: by space with several, else the one space's."""
    if len(maps["raw"]) <= 1:
        return _built(maps, shared, values, shared.first_space)
    # An other table's normalised map is its raw one: the page draws the raw set for it, not
    # a second copy (``_html_plotly_script``), which keeps room for its centred view.
    return {
        space: _built(maps, shared, values, space)
        for space in maps[values]
        if not (values == "normalised" and space in own)
    }


def _population_rows(result: Report, found: Any) -> list[dict[str, Any]]:
    """The populations as the page's data names them, none without."""
    if found is None:
        return []
    return [
        {
            "name": r.population,
            "share": round(float(r.share), 4),
            "defined": r.defined_by or found_words(r._asdict()),
            "kind": r.kind,
        }
        for r in result.results["populations"].populations.table.itertuples()
    ]


def _page_data(
    legend: _Legend,
    features: list[str],
    populations: list[dict[str, Any]],
    label: str | None,
    sets: dict[str, Any],
    spaces: list[str],
) -> dict[str, Any]:
    """The maps' data for the page's script."""
    return {
        "conditions": [str(n) for n in legend.names],
        "colours": legend.colours,
        "grey": GREY,
        "counts": legend.counts,
        "methods": ["PCA", "UMAP"],
        "features": features,
        "populations": populations,
        "label": label,
        "values": sorted(sets, key=lambda v: VIEW_ORDER.get(v, len(VIEW_ORDER))),
        "sets": sets,
        **({"tables": spaces} if len(spaces) > 1 else {}),
    }


def _files(
    maps: dict[str, dict[str, Embedding]],
    own: list[str],
    separation: Any,
    folder: Path,
    setup: Any,
) -> tuple[list[tuple[str, str, Embedding]], list[tuple[str, str]]]:
    """The maps shown in the files and the data behind them, written: each map's points and
    the separation's table."""
    several = len(maps["raw"]) > 1
    # Each map once in the files: an other table's is the same in both values; the centred
    # view is in the interactive maps only, which keeps the page within its budget.
    shown = [
        (values, space, embedding)
        for values, view in maps.items()
        for space, embedding in view.items()
        if values == "raw" or (values == "normalised" and space not in own)
    ]
    data_files = embedding_data(
        {f"{values}, {s}" if several else values: e for values, s, e in shown}, folder, setup
    )
    if separation is not None:
        path = write_csv(separation.table, folder, "separation.csv")
        data_files.append((path, "Neighbours sharing the condition, per repeat (CSV)"))
    return shown, data_files


def _static(
    shown: list[tuple[str, str, Embedding]],
    spaces: list[str],
    control: Any,
    legend: _Legend,
    folder: Path,
) -> list[str]:
    """The static PCA and UMAP of each map shown, saved, as figure blocks."""
    several = len(spaces) > 1
    taken: set[str] = set()
    stems = {space: file_stem(space, taken) for space in spaces}
    static = []
    for values, space, embedding in shown:
        drawn = embedding_figures(embedding, control=control, highlight=legend.hits)
        key = f"{values}_{stems[space]}" if several else values
        for method, figure in drawn.items():
            name = f"{method}_{key}"
            caption = f"{method.upper()}, {values}" + (f", {space}" if several else "")
            caption += f". {legend.note}" if legend.note else ""
            static.append(figure_block(save(figure, folder, name), name, caption))
    return static


def _controls(data: dict[str, Any], spaces: list[str], n_features: int) -> str:
    """The maps' controls: the feature table, the method, the values, the colour, the
    search and the enlarge button, with how to use the map."""
    buttons = "".join(
        f'<button type="button" data-key="method" data-value="{m}" '
        f'aria-pressed="{str(i == 0).lower()}">{m}</button>'
        for i, m in enumerate(data["methods"])
    )
    values_buttons, method_menu = values_controls(data["values"])
    table_buttons = "".join(
        f'<button type="button" data-key="table" data-value="{html.escape(t)}" '
        f'aria-pressed="{str(i == 0).lower()}">{html.escape(t)}</button>'
        for i, t in enumerate(spaces)
    )
    return (
        '<div class="viz" data-viz="embedding"><div class="controls">'
        + (
            f'<span class="seg" role="group" aria-label="Feature table">{table_buttons}</span>'
            if len(spaces) > 1
            else ""
        )
        + f'<span class="seg" role="group" aria-label="Method">{buttons}</span>'
        + (
            f'<span class="seg" role="group" aria-label="Values">{values_buttons}</span>'
            + method_menu
            if len(data["values"]) > 1
            else ""
        )
        + colour_menu(data, n_features)
        + find_box(data["conditions"], data["colours"])
        + '<button type="button" class="dl full">Enlarge</button>'
        '<span class="note">Click an entry in the legend to hide it, double-click to show '
        "it alone; draw around points (the lasso is on) to see which conditions and "
        "populations they come from and which features set them apart; colour the map by a "
        "feature to see where its high and low values sit; ◯ rings a population.</span></div>"
    )


def _body(
    data: dict[str, Any],
    note: str,
    comparison: str,
    data_files: list[tuple[str, str]],
    notes: tuple[str, str, str],
) -> str:
    """The maps themselves, their notes (``notes``: the neighbours, the centring paragraph
    and the end of the batch note), the data behind them and the data for the script."""
    neighbours, centring, batch = notes
    return (
        '<div class="viz-body"><div class="pl-embed"></div>'
        '<div class="selection" aria-live="polite"></div></div>'
        '<p class="caption note" style="white-space: pre-line"></p>'
        + (f'<p class="note">{html.escape(note)}</p>' if note else "")
        + (
            '<p class="note"><strong>Normalisation.</strong> '
            + html.escape(" ".join(f"{METHOD_WORDS[v]}: {VALUE_HELP[v]}" for v in data["values"]))
            + "</p>"
            if len(data["values"]) > 1
            else ""
        )
        + comparison
        + f'<p class="tools">Data behind the maps: {data_buttons(data_files)}</p>'
        f'<p class="note">{html.escape(neighbours)}</p>'
        + centring
        + '<p class="note">Both describe, they do not test: a separation is a pattern to read '
        "with the tests between conditions. On raw values a separation can be a batch between "
        + batch
        + '<script type="application/json" id="embedding-data">'
        + json_island(data)
        + "</script>"
    )


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
