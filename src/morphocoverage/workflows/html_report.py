"""The report as one HTML page, with its figures as PNG and PDF beside it.

``html_report(result, objects, fovs, folder)`` writes ``folder/report.html``: a single file
that opens in any browser, offline, and can be emailed or archived as it is. It holds:

- a sidebar with every answer's status, and **At a glance** (each answer's marker, as a chip
  with its word, and its one-line answer, linking to the section);
- **What is in the dataset**, **what the acquisition was asked to preserve**, then every
  question of the report in its short form, the full text folded under "Full detail", and
  "What this dataset cannot tell me": the same words as ``Report.summary``, its aligned
  tables as sortable tables;
- under question 9, the **figures**: the readout and the discovery's best-ranked features
  (``plots``, embedded, each linking to its PDF), and with ``embedding.MIN_FEATURES`` or more
  features an **interactive PCA and UMAP** of all of them (hover a point for its condition,
  repeat and field; click a condition to hide it; switch PCA and UMAP, and raw and
  normalised values when a control condition is declared).

Every figure is also saved in ``folder/figures`` as PNG and PDF (vector, for papers). The
page follows the viewer's light or dark setting, with a switch; it prints with every
section open. Nothing on it is computed anew except the figures: the text is the report's.
"""

from __future__ import annotations

import base64
import html
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._style import Status
from .._wording import percent
from ..embedding import MIN_FEATURES, Embedding, embed
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from ..normalize import normalize_to_control
from ..plots_embedding import embedding_figures
from ..retrospective.report import (
    DECLARATION_TITLE,
    LIMITS_TITLE,
    OVERVIEW_TITLE,
    QUESTIONS,
    SHORT_QUESTIONS,
)
from ._html_render import CSS, SCRIPT, chip, text_html
from .figures import plots
from .retrospective import REPORT_QUESTIONS, Report

REPORT_FILE = "report.html"
# The categorical palette of the interactive figures (validated order); the control is grey.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# The control's grey (matplotlib's "0.6" of the static figures).
GREY_HEX = "#999999"
TITLES = dict(QUESTIONS)


def _save(figure: Any, folder: Path, name: str) -> str:
    """``name``.png and .pdf in ``folder``; the PNG as a data URI for the page."""
    figure.savefig(folder / f"{name}.png", dpi=150)
    figure.savefig(folder / f"{name}.pdf")
    data = base64.b64encode((folder / f"{name}.png").read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _figure(uri: str, name: str, caption: str) -> str:
    return (
        f'<figure><a href="figures/{html.escape(name)}.png" title="Open the full size figure">'
        f'<img src="{uri}" alt="{html.escape(caption)}" loading="lazy"></a>'
        f"<figcaption>{html.escape(caption)} · "
        f'<a href="figures/{html.escape(name)}.pdf">PDF</a> · '
        f'<a href="figures/{html.escape(name)}.png">PNG</a></figcaption></figure>'
    )


def _feature_figures(result: Report, objects, fovs, folder: Path) -> list[str]:
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
            blocks.append(_figure(_save(figure, folder, name), name, f"{feature}: {role}"))
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


def _embeddings(result: Report, objects, fovs, folder: Path) -> tuple[str, list[str]]:
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
    others = iter(COLOURS)
    colours = [GREY_HEX if name == control_value else next(others, "#888888") for name in names]
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
            static.append(_figure(_save(figure, folder, name), name, f"{method.upper()}, {values}"))
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
        + '</div><div class="legend" aria-label="Conditions (click to hide)"></div>'
        '<div class="plot-box"><div class="svg-slot"></div><div class="tooltip"></div></div>'
        '<p class="caption note"></p>'
        '<p class="note">Both describe, they do not test: a separation is a pattern to read with '
        "question 9's tests. On raw values a separation can be a batch between repeats, which "
        "the normalised values (each divided by its repeat's control mean) take out.</p></div>"
        '<script type="application/json" id="embedding-data">'
        + json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
        + "</script>"
    )
    return markup, static


def _section(key: str, title: str, status: Status | None, short: str, full: str | None) -> str:
    body = text_html(short)
    if full and full.strip() != short.strip():
        body += f"<details><summary>Full detail</summary>{text_html(full)}</details>"
    return (
        f'<section class="card" id="{key}"><h2>{chip(status)} {html.escape(title)}</h2>'
        f"{body}</section>"
    )


def html_report(
    result: Report,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    folder: str | Path,
    *,
    title: str = "MorphoCoverage report",
) -> Path:
    """Write ``folder/report.html`` and its figures (see the module); returns the page's path.

    ``objects`` and ``fovs`` are the tables the ``report`` was run on.
    """
    if not isinstance(result, Report):
        raise TypeError("result must be the Report that report() returned")
    folder = Path(folder)
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    doc = result.document
    keys = [k for k in REPORT_QUESTIONS if k in doc.sections]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        feature_blocks = _feature_figures(result, objects, fovs, figures)
        viz, static = ("", [])
        if len(result.setup.columns) >= MIN_FEATURES:
            viz, static = _embeddings(result, objects, fovs, figures)
    nav = ['<nav class="toc" aria-label="Sections"><h2>Report</h2>']
    glance = []
    for key in ["overview", *keys]:
        status = doc.statuses.get(key)
        label = SHORT_QUESTIONS.get(key, key)
        icon = status.marker if status is not None else ""
        nav.append(
            f'<a href="#{key}"><span class="icon" aria-hidden="true">{icon}</span>'
            f"<span>{html.escape(label)}</span></a>"
        )
        if status is not None:
            glance.append(
                f'<a class="glance-row" href="#{key}"><span class="q">{html.escape(label)}'
                f'<br>{chip(status)}</span><span class="a">'
                f"{html.escape(doc.headlines.get(key, ''))}</span></a>"
            )
    nav.append('<a href="#limits"><span class="icon"></span><span>What it cannot tell</span></a>')
    nav.append("</nav>")
    overview_short = doc.short.get("overview") or doc.overview or ""
    subtitle = _first_sentence(overview_short)
    cards = [
        '<section class="card" id="glance"><h2>At a glance</h2><p class="key">A marker says '
        "how far to trust an answer, not what it found.</p>" + "".join(glance) + "</section>"
    ]
    if doc.overview:
        cards.append(
            _section(
                "overview",
                OVERVIEW_TITLE,
                doc.statuses.get("overview"),
                overview_short,
                doc.overview,
            )
        )
    if doc.declaration:
        declared = doc.short.get("declaration") or "\n".join(f"- {d}" for d in doc.declaration)
        cards.append(_section("declaration", DECLARATION_TITLE, None, declared, None))
    for key in keys:
        short = doc.short.get(key, doc.sections[key])
        card = _section(key, TITLES[key], doc.statuses.get(key), short, doc.sections[key])
        if key == "conditions" and (feature_blocks or viz):
            extra = ""
            if feature_blocks:
                extra += (
                    "<h3>The readout and the best-ranked features</h3>"
                    f'<div class="figures">{"".join(feature_blocks)}</div>'
                )
            if viz:
                extra += f"<h3>All {len(result.setup.columns)} features at once</h3>{viz}"
                extra += (
                    "<details><summary>Static PCA and UMAP (PNG and PDF)</summary>"
                    f'<div class="figures">{"".join(static)}</div></details>'
                )
            card = card[: -len("</section>")] + extra + "</section>"
        cards.append(card)
    limits = doc.limits(list(REPORT_QUESTIONS), "short")
    cards.append(
        _section("limits", LIMITS_TITLE, None, limits, doc.limits(list(REPORT_QUESTIONS), "full"))
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f'<div class="layout">{"".join(nav)}<main>'
        '<header class="top"><button type="button" class="theme-toggle">Light / dark</button>'
        f'<h1>{html.escape(title)}</h1><p class="sub">{html.escape(subtitle)}</p>'
        f'<p class="meta">Generated {stamp} by MorphoCoverage. The same answers as the text '
        "report; figures in figures/ as PNG and PDF.</p></header>"
        + "".join(cards)
        + "<footer>"
        + html.escape(_settings(result))
        + f"</footer></main></div><script>{SCRIPT}</script></body></html>"
    )
    path = folder / REPORT_FILE
    path.write_text(page, encoding="utf-8")
    return path


def _first_sentence(text: str) -> str:
    """The first line of ``text`` with its wrapped continuation lines joined again."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = len(lines[0]) - len(lines[0].lstrip())
    parts = [lines[0].strip()]
    for line in lines[1:]:
        if len(line) - len(line.lstrip()) <= first:
            break
        parts.append(line.strip())
    return " ".join(parts)


def _settings(result: Report) -> str:
    text = result.summary().rsplit("\n\n", 1)[-1]
    return text if text.startswith("Settings") else ""


__all__ = ["REPORT_FILE", "html_report"]
