"""The frame of the HTML report's page (private; ``html_report``): the check that the tables
are the report's, the sections shown, the sidebar and "At a glance", the texts carried as
files, and the page around the cards with its scripts.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ._html_data import LINKED, json_island
from ._html_fields import NEW_STATES, has_new_states
from ._html_plan import PLAN_TITLE
from ._html_plotly import plotly_js
from ._html_plotly_script import PLOTLY_SCRIPT
from ._html_render import chip
from ._html_script import SCRIPT
from ._html_sections import ORDER, hidden
from ._html_style import CSS
from .retrospective import Report


def check_counts(result: Report, objects: pd.DataFrame, fovs: pd.DataFrame) -> None:
    """Raise ``ValueError`` when ``objects`` and ``fovs`` do not hold as many rows as the
    tables the report was run on."""
    counts = getattr(result.results.get("overview"), "counts", None)
    if counts is not None and (len(objects), len(fovs)) != (counts["objects"], counts["fields"]):
        raise ValueError(
            f"objects and fovs must be the tables the report was run on ({counts['objects']} "
            f"observations, {counts['fields']} fields), not {len(objects)} rows and {len(fovs)}"
        )


def shown_sections(doc: Any, maps: bool) -> list[str]:
    """The sections on the page, in its order: those the report answers (the data, the maps
    when drawn), less those this design cannot answer (``hidden``)."""
    statuses, headlines = doc.statuses, doc.headlines
    keys = [
        k
        for k in ORDER
        if (k == "overview" and doc.overview) or k in doc.sections or (k == "dataset" and maps)
    ]
    return [k for k in keys if not hidden(statuses.get(k), headlines.get(k, ""))]


def navigation(
    result: Report,
    shown: list[str],
    titles: dict[str, str],
    words: dict[str, str],
    annotate: Any,
) -> tuple[str, str]:
    """The sidebar (every section shown, the new states' card after the fields to check, and
    the closing sections) and the "At a glance" card (each answer's chip, in ``words``, and
    its headline; ``annotate`` links its terms)."""
    doc = result.document
    statuses, headlines = doc.statuses, doc.headlines
    nav = ['<nav class="toc" aria-label="Sections"><h2>Report</h2>']
    glance = []
    for key in shown:
        status = statuses.get(key)
        icon = status.marker if status is not None else ""
        nav.append(
            f'<a href="#{key}"><span class="icon" aria-hidden="true">{icon}</span>'
            f"<span>{html.escape(titles[key])}</span></a>"
        )
        if key == "inspect" and has_new_states(result):  # the new states' card (audit 10)
            nav.append(
                f'<a href="#{NEW_STATES}"><span class="icon"></span><span>New states</span></a>'
            )
        if status is not None:
            glance.append(
                f'<a class="glance-row" href="#{key}"><span class="q">{html.escape(titles[key])}'
                f"<br>{chip(status, key, headlines.get(key, ''), words.get(key))}</span>"
                '<span class="a">'
                f"{html.escape(headlines.get(key, ''))}</span></a>"
            )
    for anchor, label in (
        ("limits", "What it cannot tell"),
        ("plan", PLAN_TITLE),
        ("glossary", "Glossary"),
        ("files", "Files"),
    ):
        nav.append(f'<a href="#{anchor}"><span class="icon"></span><span>{label}</span></a>')
    nav.append("</nav>")
    card = (
        '<section class="card" id="glance"><h2>At a glance</h2><p class="key">Each marker '
        "says what the answer found, in the word beside it. Underlined words show their "
        "meaning, and a click opens them in the glossary, on the side.</p>"
        + annotate("".join(glance))
        + "</section>"
    )
    return "".join(nav), card


def page_texts(result: Report) -> dict[str, str]:
    """The texts the page carries as files: the short and full reports, the feature
    discovery's tests and the contrast table, as CSV, when there are."""
    texts = {
        "report_short.txt": result.summary() + "\n",
        "report_full.txt": result.summary(detail="full") + "\n",
    }
    if result.discovery is not None and len(result.discovery.tests):
        texts["feature_discovery.csv"] = result.discovery.tests.to_csv(index=False)
    if result.setup.contrast_table is not None:  # one row per test (owner, 2026-10-05)
        texts["contrasts.csv"] = result.setup.contrast_table.to_csv(index=False)
    return texts


def page_markup(
    title: str,
    nav: str,
    subtitle: str,
    cards: list[str],
    files: dict[str, Any],
    ends: tuple[str, str, str],
) -> str:
    """The page: its head and style, the sidebar, the cards, the footer and the glossary
    drawer, then the files' data and the scripts (``ends``: the footer's settings, the
    drawer and the plotly scripts)."""
    settings, drawer, plotly = ends
    inside = (
        "every figure and file is inside this page (Files)"
        if all(entry[1] != LINKED for entry in files.values())
        else "the larger files open from the figures folder beside it (Files)"
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f'<div class="layout">{nav}<main>'
        '<header class="top"><button type="button" class="theme-toggle">Light / dark</button>'
        f'<h1>{html.escape(title)}</h1><p class="sub">{html.escape(subtitle)}</p>'
        f'<p class="meta">Generated {stamp} by MorphoCoverage. The same answers as the text '
        f"report; {inside}.</p></header>"
        + "".join(cards)
        + "<footer>"
        + html.escape(settings)
        + "</footer></main></div>"
        + drawer
        + '<script type="application/json" id="files-data">'
        + json_island(files)
        + f"</script><script>{SCRIPT}</script>"
        + plotly
        + "</body></html>"
    )


def plotly_scripts(specs: list[dict], embedding: bool) -> str:
    """plotly.js, the feature figures' data and the script that draws them; nothing when
    the page has no interactive figure."""
    if not specs and not embedding:
        return ""
    data = json_island(specs)
    return (
        f'<script type="application/json" id="feature-data">{data}</script>'
        f"<script>{plotly_js()}</script><script>{PLOTLY_SCRIPT}</script>"
    )


def settings_line(result: Report) -> str:
    """The report's closing settings sentence, for the footer; empty without."""
    text = result.summary().rsplit("\n\n", 1)[-1]
    return text if text.startswith("Settings") else ""
