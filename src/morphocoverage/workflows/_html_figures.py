"""The HTML report's figures (private; html_report, _html_maps): each saved as PNG and PDF
and embedded (``save``); the maps are in ``_html_maps``.

- feature_figures: the discovery's best-ranked features, the readout when none ranks
  (plots), each with the CSV of what it draws (``_html_data``).
- figure_block: a figure with its download buttons (``_html_data.download``).
"""

from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .._palette import COLOURS, condition_palette
from ..metadata import CONDITION_COLUMN, sorted_labels
from ..normalize import control_mean_problem
from ..plots_pairwise import (
    PAIR_NOTE,
    PAIR_NOTE_SEVERAL,
    PAIRED_NOTE,
    PANEL_KEYS,
    PANEL_TITLES,
)
from ._contrasts import normalisation_problem
from ._html_data import data_buttons, download, feature_data
from ._html_plotly import feature_specs, interactive_caption
from ._rare_share import with_share
from ._shared import check_workflow_inputs
from .figures import highlighted, plots
from .retrospective import Report

# The interactive figures use the static figures' palette (``_palette.condition_palette``,
# the control included); past its 12 colours the control and the hits keep theirs and the
# others are grey, and the legend and hover name each point.


# The widest a static PNG is drawn, in pixels: a split figure of 12 conditions is 27 inches
# wide, which at 150 dpi made PNGs of 4113 pixels; past this width the dpi is lowered.
PNG_PIXELS = 2400


def save(figure: Any, folder: Path, name: str) -> str:
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
    pair_note = PAIR_NOTE if setup.own_references is None else PAIR_NOTE_SEVERAL
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
            uri = save(figure, folder, name)
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
                    else f"<p>(B) {html.escape(pair_note)}</p><p>(C) {html.escape(PAIRED_NOTE)}</p>"
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


__all__ = [
    "COLOURS",
    "condition_colours",
    "feature_figures",
    "figure_block",
    "save",
]
