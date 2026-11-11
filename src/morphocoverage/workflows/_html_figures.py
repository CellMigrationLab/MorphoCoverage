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
    verdict_note,
)
from ._contrasts import normalisation_problem
from ._html_data import data_buttons, download, feature_data
from ._html_plotly import feature_specs, interactive_caption
from ._pairing import paired_inputs
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


def _verdict_paragraph(figure: Any) -> str:
    """Under panel C, each condition's verdict q on the figure's feature (audit 10, owner,
    2026-10-06): panel C's q is adjusted over the figure only."""
    note = verdict_note(getattr(figure, "pairwise_verdicts", None) or {})
    return f"<p>{html.escape(note)}</p>" if note else ""


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
    if setup.embedding_only:  # no feature to plot one at a time (owner, 2026-10-06)
        return [], []
    drawn = plots(objects, fovs, setup, ranking=result.discovery)
    # The views read the tables paired by repeat group, as ``plots`` does (#617).
    paired = paired_inputs(objects, fovs, setup)
    setup = paired.setup
    checked, checked_fovs, _ = check_workflow_inputs(paired.objects, paired.fovs, setup)
    views = _views(setup, checked, checked_fovs)
    figures_by_view = {views[0][0]: drawn}
    for key, view in views[1:]:
        figures_by_view[key] = plots(objects, fovs, view, features=list(drawn))
    discovery = result.discovery
    top = discovery.top(len(discovery.ranking)) if discovery is not None else []
    prepared = {key: with_share(view.prepared(checked, checked_fovs), view) for key, view in views}
    blocks, names, explained = [], {}, False
    pair_note = PAIR_NOTE if setup.own_references is None else PAIR_NOTE_SEVERAL
    switch = _switch(views)
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
            role = _role(feature, top, result.setup.readout, index, len(parts))
            uri = save(figure, folder, name)
            label = f"{feature}: {role}"
            pairs = getattr(figure, "pairwise", None)
            paired = pairs is not None and len(pairs) > 0
            legend = _legend(figure, paired, explained, pair_note)
            explained = explained or paired
            box = legends[feature] if feature in legends and index == 1 else None
            blocks.append(_feature_block(name, label, switch, paired, legend, box, uri, data))
    control = (setup.figure_control or {}).get(CONDITION_COLUMN)
    colours = condition_colours(objects, fovs, control, highlighted(result.discovery)[0])
    specs = _specs(figures_by_view, prepared, checked_fovs, views, colours, names)
    return blocks, specs


def _role(feature: str, top: list[str], readout: Any, index: int, parts: int) -> str:
    """What a figure shows: the feature's rank, the readout, or a declared feature, and its
    part of a split figure."""
    rank = f"best-ranked feature {top.index(feature) + 1}" if feature in top else ""
    role = rank or "a declared feature"
    if feature == readout:
        role = f"{rank}, the readout" if rank else "the readout"
    if parts > 1:
        role += f", part {index} of {parts}"
    return role


def _legend(figure: Any, paired: bool, explained: bool, pair_note: str) -> str:
    """A figure's legend: panel A's caption, and panels B and C's when it has them, which
    read the same in every figure: explained under the first (``explained``)."""
    caption = f'<span class="caption">{html.escape(interactive_caption(figure))}</span>'
    if not paired:
        return f"<p>{caption}</p>"
    return (
        f"<p>(A) {caption}</p>"
        + (
            "<p>(B) and (C): as in the first figure above.</p>"
            if explained
            else f"<p>(B) {html.escape(pair_note)}</p><p>(C) {html.escape(PAIRED_NOTE)}</p>"
        )
        + _verdict_paragraph(figure)
    )


def _switch(views: list[tuple[str, Any]]) -> str:
    """The buttons switching a figure between its values (raw, normalised), when it has
    several."""
    switch = "".join(
        f'<button type="button" data-key="values" data-value="{key}" '
        f'aria-pressed="{str(key == views[0][0]).lower()}">{key.capitalize()}</button>'
        for key, _ in views
    )
    return (
        f'<span class="seg" role="group" aria-label="Values">{switch}</span>'
        if len(views) > 1
        else ""
    )


def _pair_row(paired: bool) -> str:
    """The pairwise panels B and C under a figure that has them."""
    if not paired:
        return ""
    return (
        '<div class="pair-row">'
        + "".join(
            f'<div class="panel"><p class="panel-title"><b>{letter}</b> {title}'
            f'<span class="panel-key">{html.escape(PANEL_KEYS[letter])}</span></p>'
            f'<div class="pl-pair" data-panel="{letter.lower()}"></div></div>'
            for letter, title in PANEL_TITLES.items()
        )
        + "</div>"
    )


def _feature_block(
    name: str,
    label: str,
    switch: str,
    paired: bool,
    legend: str,
    box: str | None,
    uri: str,
    data: Any,
) -> str:
    """One figure's block: its interactive panels, its legend (and the text for a figure
    legend, ``box``, folded), its downloads and its static figure folded."""
    letter_a = '<p class="panel-title"><b>A</b></p>' if paired else ""
    return (
        f'<div class="feature" data-spec="{html.escape(name)}">'
        f'<div class="feature-head"><h4>{html.escape(label)}</h4><span class="tools">'
        f'{switch}<button type="button" class="dl full">Enlarge</button></span></div>'
        f'<div class="panel">{letter_a}'
        f'<div class="pl-plot" id="pl-{html.escape(name)}"></div></div>{_pair_row(paired)}'
        f'<div class="figure-legend"><p><strong>Legend:</strong></p>{legend}</div>'
        + (
            "<details><summary>For a figure legend</summary>"
            f'<p class="legend-box">{box}</p></details>'
            if box is not None
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


def _specs(
    figures_by_view: dict[str, Any],
    prepared: dict[str, Any],
    fovs: pd.DataFrame,
    views: list[tuple[str, Any]],
    colours: dict,
    names: dict[str, str],
) -> list[dict[str, Any]]:
    """The specs the page's script draws, one per figure, with its data in every view."""
    by_view = {}
    for key, view in views:
        by_view[key] = feature_specs(
            figures_by_view[key], prepared[key], fovs, view, colours, names
        )
    first = views[0][0]
    return [
        {
            "id": spec["id"],
            "feature": spec["feature"],
            "current": first,
            "views": {key: by_view[key][i] for key in by_view},
        }
        for i, spec in enumerate(by_view[first])
    ]


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
        repeat_col=setup.pairing_col(fovs),
        **setup.common,
    )


__all__ = [
    "COLOURS",
    "condition_colours",
    "feature_figures",
    "figure_block",
    "save",
]
