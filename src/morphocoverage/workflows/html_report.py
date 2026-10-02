"""The report as one HTML page, with its figures as PNG and PDF beside it.

``html_report(result, objects, fovs, folder)`` writes ``folder/report.html``: a single file
that opens in any browser, offline, and can be emailed or archived as it is. It holds:

- a sidebar with every section's status, and **At a glance** (each answer's marker, as a
  chip with its word, and its one-line answer, linking to the section);
- the sections in the order a user acts on them, titled in words (``_html_sections``): the
  data (one row per check, ``_html_cards``), the fields to check and why, with
  ``embedding.MIN_FEATURES`` or more features the whole dataset at once (an **interactive
  PCA and UMAP**, ``_html_figures``), the populations, whether the conditions differ,
  whether enough was imaged (one sentence and a table across the features, then a tab per
  feature and declared target, ``_html_enough``), and the rest; a section this design
  cannot answer is left out (its reason stays in "What this dataset cannot tell me"). Each
  gives what it is for under a "?", its answer first and the rest under "Show more" (the
  words of ``Report.summary``, its aligned tables as sortable tables), and, where an
  analysis follows, **what to do next** with its code (``_html_next``);
- under "Do the conditions differ?", the features significant across repeats per
  condition, a **heatmap** of Cohen's d, the **figures** of the best-ranked features, then
  the tests behind the reading;
- the **glossary** as a drawer on the side, with a search, and the **Files**.

Every figure is also saved in ``folder/figures`` as PNG and PDF (vector, for papers), with
the data it draws as CSV (``_html_data``); a click enlarges it. The page follows the
viewer's light or dark setting, with a switch; it prints with every section open. Nothing on
it is computed anew except the figures: the text is the report's.
"""

from __future__ import annotations

import html
import re
import time
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .._style import Status
from .._wording import listing
from ..embedding import MIN_FEATURES
from ..glossary import GLOSSARY, PHRASES, slug
from ..retrospective.report import LIMITS_TITLE
from ..sampling_unit.legend_text import legend_box
from ..sampling_unit.reading_text import readings_parts
from ..sufficiency_short import rule_tallies
from ._html_cards import (
    check_rows,
    chip_words,
    fields_table,
    flagged_fields,
    help_icon,
    readout_summary,
    significant_summary,
)
from ._html_data import data_buttons, files_card, heatmap_data, json_island, page_files
from ._html_enough import enough_body, enough_help
from ._html_figures import embeddings, feature_figures
from ._html_next import fields_next, populations_next
from ._html_plan import PLAN_TITLE, plan_card
from ._html_plotly import plotly_js
from ._html_plotly_script import PLOTLY_SCRIPT
from ._html_render import Terms, adjusted_p, chip, text_html
from ._html_script import SCRIPT
from ._html_sections import (
    INTROS,
    ORDER,
    TITLES,
    enough_reading,
    heatmap,
    hidden,
    intros_for,
    not_declared,
    unnumbered,
    what_checked,
)
from ._html_style import CSS
from ._html_tests import tests_section
from .retrospective import REPORT_QUESTIONS, Report

REPORT_FILE = "report.html"


def _on_page(term: Any, text: str) -> bool:
    """Whether the page's ``text`` (lower case, without tags) uses ``term``: its name or one
    of its phrases, whole."""
    name = re.sub(r"\s*\(.*\)$", "", term.name)
    words = (name, *PHRASES.get(term.name, ()))
    return any(re.search(rf"(?<![\w-]){re.escape(w.lower())}(?![\w-])", text) for w in words)


def _glossary_drawer(page: str) -> str:
    """The glossary as a drawer on the side (opened by its tab, the sidebar and every
    underlined term), with a search; printed at the end of the page. Only the terms the
    page uses (``page``, its markup) are in it."""
    # The tooltips repeat the glossary's own words: left out.
    page = re.sub(r'<span class="tip"[^>]*>.*?</span>', " ", page, flags=re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page)).lower()
    parts = [
        (
            '<button type="button" class="glossary-tab" aria-controls="glossary">Glossary'
            '</button><aside class="drawer" id="glossary" aria-label="Glossary" '
            'aria-hidden="true"><header><button type="button" class="close" '
            'aria-label="Close the glossary">×</button><h2>Glossary</h2><input type="search" '
            'placeholder="Search the terms" aria-label="Search the glossary"></header>'
            '<div class="drawer-body">'
        )
    ]
    section = None
    for term in (t for t in GLOSSARY if _on_page(t, text)):
        if term.section != section:
            if section is not None:
                parts.append("</dl>")
            section = term.section
            parts.append(f'<h3>{html.escape(section)}</h3><dl class="glossary">')
        refs = "; ".join(
            f'<a href="{html.escape(r.url)}" target="_blank" rel="noopener">'
            f"{html.escape(r.text)}</a>"
            if r.url
            else html.escape(r.text)
            for r in term.references
        )
        parts.append(
            f'<dt id="{slug(term)}">{html.escape(term.name)}</dt><dd>{html.escape(term.meaning)}'
            f'<span class="where">Where: {_unnumbered(html.escape(term.where))}.</span>'
            + (f'<span class="refs">See: {refs}</span>' if refs else "")
            + "</dd>"
        )
    parts.append("</dl></div></aside>")
    return "".join(parts)


_TERMS = Terms()
_INTROS: dict[str, str] = dict(INTROS)
# The chip words the results decide (``chip_words``) and the report's observations, set
# per page like the intros.
_WORDS: dict[str, str] = {}
_UNITS = {"words": "cells"}


def _unnumbered(markup: str) -> str:
    """``unnumbered`` in this page's words for its observations."""
    return unnumbered(markup, _UNITS["words"])


def _html(text: str) -> str:
    """The report's text as HTML, under this page's titles, its terms linked."""
    return _TERMS.annotate(_unnumbered(text_html(text)))


def _section(
    key: str,
    status: Status | None,
    short: str,
    full: str | None,
    answer: str | None,
    *,
    title: str | None = None,
    body_extra: str = "",
    found: str | None = None,
    next_steps: str = "",
) -> str:
    """A card: its title and status (what it is for under its "?"), its answer in a line,
    its short text, its full text folded, and what to do next."""
    head = ""
    if answer:
        head += (
            '<p class="answer"><strong>In short:</strong> '
            f"{_TERMS.annotate(html.escape(answer[:1].upper() + answer[1:]))}.</p>"
        )
    body = body_extra or _html(short)
    if full and full.strip() != short.strip():
        detail = _html(full)
        body += f"<details><summary>Show more</summary>{detail}</details>"
    return _card(
        key,
        title or TITLES.get(key, key),
        chip(status, key, found or answer or "", _WORDS.get(key)),
        head + body + next_steps,
    )


def _card(key: str, title: str, chip_html: str, body: str) -> str:
    """A section: its chip, its title with its "?" help, and its body."""
    return (
        f'<section class="card" id="{key}"><h2>{chip_html} {html.escape(title)}'
        f"{help_icon(_INTROS.get(key))}</h2>{body}</section>"
    )


def _undefined_rules(result: Report) -> list[str]:
    """The stopping rules that did not apply to any group (one per object, on tracks)."""
    sufficiency = (getattr(result, "results", {}) or {}).get("sufficiency")
    tallies = rule_tallies(sufficiency) if sufficiency is not None else []
    return [t["label"] for t in tallies if t["groups"] and t["undefined"] == t["groups"]]


def _data_card(result: Report, doc: Any, status: Status | None, fields_named: bool) -> str:
    """The data: one row per check with its chip, what was not declared in a line; the
    tables, what was checked and the declaration under "Show more"."""
    short = doc.short.get("overview") or doc.overview or ""
    missing = not_declared(result.setup)
    note = (
        f'<p class="note">Not declared, so not checked: {html.escape(listing(missing, 6))}.</p>'
        if missing
        else ""
    )
    lines = "".join(
        f"<li>{_TERMS.annotate(html.escape(line))}</li>"
        for line in what_checked(result.setup, _undefined_rules(result))
    )
    declaration = doc.short.get("declaration") or "\n".join(f"- {d}" for d in doc.declaration)
    more = (
        f"<details><summary>Show more</summary>{_html(short)}<h3>What was checked</h3>"
        f'<ul class="checked">{lines}</ul>'
        + (
            f"<details><summary>The declaration in full</summary>{text_html(declaration)}</details>"
            if declaration
            else ""
        )
        + "</details>"
    )
    rows = check_rows(short, _TERMS.annotate, fields_named)
    return _card("overview", TITLES["overview"], chip(status, "overview", ""), rows + note + more)


def _fields_card(result: Report, status: Status | None, short: str, full: str) -> str:
    """The fields to check, and why; the report's text under "Show more"."""
    rows, identities, key = flagged_fields(result)
    if rows:
        body = (
            "<p>Please check these fields of view, and why:</p>"
            + fields_table(rows, _TERMS.annotate)
            + '<p class="note">None of these is called an error: look at the images to see '
            "whether it is biology, technique or simply a large field.</p>"
        )
    else:
        body = _html(short)
    body += f"<details><summary>Show more</summary>{_html(full)}</details>"
    body += fields_next(identities, key)
    return _card("inspect", TITLES["inspect"], chip(status, "inspect", short), body)


def _dataset_card(result: Report, viz: str, static: list[str]) -> str:
    """Every feature at once: the interactive PCA and UMAP, the static figures folded."""
    body = (
        f"<p>All {len(result.setup.columns)} features at once, every cell a point.</p>{viz}"
        "<details><summary>Static PCA and UMAP (PNG and PDF)</summary>"
        f'<div class="figures">{"".join(static)}</div></details>'
    )
    return _card("dataset", TITLES["dataset"], "", body)


def _condition_extras(result: Report, feature_blocks: list[str], folder: Path) -> str:
    """Question 9's figures: the heatmap (its numbers as CSV), the best-ranked features."""
    conditions = []
    contrast = result.results.get("contrast")
    if contrast is not None and len(contrast.table):
        conditions = list(dict.fromkeys(contrast.table["condition"]))
    extra = ""
    grid = heatmap(result.discovery, conditions)
    if grid:
        files = heatmap_data(result.discovery, folder)
        extra += (
            f"<h3>Every feature against the control</h3>{grid}"
            f'<p class="tools">{data_buttons(files)}</p>'
        )
    if feature_blocks:
        ranked = result.discovery is not None and len(result.discovery.top())
        extra += (
            f"<h3>{'The best-ranked features' if ranked else 'The readout'}</h3>"
            f'<div class="feature-grid">{"".join(feature_blocks)}</div>'
        )
    return extra


def _conditions_card(
    result: Report,
    status: Status | None,
    short: str,
    full: str,
    answer: str | None,
    extras: str,
    next_html: str,
    folder: Path,
) -> str:
    """Question 9: the significant features per condition, the figures, then the tests
    behind them (the readout's reading first), then what to do next."""
    contrast = result.results.get("contrast")
    reading = "\n".join(readings_parts(contrast, indent=2)) if contrast is not None else ""
    summary = significant_summary(result, _TERMS.annotate) or readout_summary(
        result, _TERMS.annotate
    )
    if reading and short.startswith(reading):
        rest = short[len(reading) :].lstrip("\n")
        box = legend_box(contrast, result.setup.unit_words)
        legend = (
            f'<p class="legend-box"><strong>For a figure legend.</strong> '
            f"{_TERMS.annotate(html.escape(box.split(': ', 1)[1]))}</p>"
            if box
            else ""
        )
        readout = (
            f"<details><summary>The readout, {html.escape(result.setup.readout)}, condition by "
            f"condition</summary>{_html(reading)}{legend}</details>"
        )
        body = (summary or _html(reading)) + extras + "<h3>The tests behind the reading</h3>"
        # One kind of test at a time (_html_tabs).
        body += (tests_section(result, rest, _html, folder) if rest.strip() else "") + readout
    else:
        body = summary + _html(short) + extras
    # The summary answers, so the headline is not repeated above it (it stays the chip's).
    shown = None if summary else answer
    return _section(
        "conditions",
        status,
        short,
        full,
        shown,
        body_extra=body,
        found=answer,
        next_steps=next_html,
    )


def html_report(
    result: Report,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    folder: str | Path,
    *,
    title: str = "MorphoCoverage report",
    attach: Sequence[str | Path] = (),
) -> Path:
    """Write ``folder/report.html`` and its figures (see the module); returns the page's path.

    ``objects`` and ``fovs`` are the tables the ``report`` was run on; ``attach`` are more files
    the page carries for download (the design scan, the saved answers).
    """
    if not isinstance(result, Report):
        raise TypeError("result must be the Report that report() returned")
    folder = Path(folder)
    # The "?" help in the report's words for its observations (tracks, patches).
    _INTROS.clear()
    one = result.setup.observation_unit or "object"
    _INTROS.update(intros_for(result.setup.unit_words, one, **enough_help(result)))
    _WORDS.clear()
    _WORDS.update(chip_words(result))
    _UNITS["words"] = result.setup.unit_words
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # Allow for file systems that store modification times to the second.
    started = time.time() - 1
    doc = result.document
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        feature_blocks, specs = feature_figures(result, objects, fovs, figures)
        viz, static = ("", [])
        if len(result.setup.columns) >= MIN_FEATURES:
            viz, static = embeddings(result, objects, fovs, figures)
    statuses, headlines = doc.statuses, doc.headlines
    # The Foundation's title names the report's own observations (tracks, patches).
    titles = {**TITLES, "unit": f"Why count fields, and not only {result.setup.unit_words}?"}
    keys = [
        k
        for k in ORDER
        if (k == "overview" and doc.overview) or k in doc.sections or (k == "dataset" and viz)
    ]
    shown = [k for k in keys if not hidden(statuses.get(k), headlines.get(k, ""))]
    nav = ['<nav class="toc" aria-label="Sections"><h2>Report</h2>']
    glance = []
    for key in shown:
        status = statuses.get(key)
        icon = status.marker if status is not None else ""
        nav.append(
            f'<a href="#{key}"><span class="icon" aria-hidden="true">{icon}</span>'
            f"<span>{html.escape(titles[key])}</span></a>"
        )
        if status is not None:
            glance.append(
                f'<a class="glance-row" href="#{key}"><span class="q">{html.escape(titles[key])}'
                f"<br>{chip(status, key, headlines.get(key, ''), _WORDS.get(key))}</span>"
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
    overview_short = doc.short.get("overview") or doc.overview or ""
    subtitle = _first_sentence(overview_short)
    cards = [
        '<section class="card" id="glance"><h2>At a glance</h2><p class="key">Each marker '
        "says what the answer found, in the word beside it. Underlined words show their "
        "meaning, and a click opens them in the glossary, on the side.</p>"
        + _TERMS.annotate("".join(glance))
        + "</section>"
    ]
    for key in shown:
        if key == "overview":
            named = "inspect" in shown
            cards.append(_data_card(result, doc, statuses.get("overview"), named))
            continue
        if key == "dataset":
            cards.append(_dataset_card(result, viz, static))
            continue
        short = doc.short.get(key, doc.sections[key])
        if key == "inspect":
            cards.append(_fields_card(result, statuses.get(key), short, doc.sections[key]))
            continue
        if key == "conditions":
            extras = _condition_extras(result, feature_blocks, figures)
            cards.append(
                _conditions_card(
                    result,
                    statuses.get(key),
                    short,
                    doc.sections[key],
                    headlines.get(key),
                    extras,
                    "",
                    figures,
                )
            )
            continue
        extra, answer, after = "", headlines.get(key), ""
        # One sentence and a table, then one feature at a time (_html_enough).
        body = enough_body(result, _TERMS.annotate) if key == "enough" else None
        reading = enough_reading(result) if key == "enough" and body is None else []
        if body is not None:
            extra, answer = body, None
        elif reading:
            # The reading answers in sentences, so the headline is not repeated above it.
            lines = "".join(f"<p>{html.escape(line)}</p>" for line in reading)
            extra = _TERMS.annotate(
                f'<div class="reading">{lines}</div><h3>The numbers</h3>'
                + _unnumbered(text_html(short))
            )
            answer = None
        if key == "populations":
            after = populations_next(result, fovs, objects)
        cards.append(
            _section(
                key,
                statuses.get(key),
                short,
                # The card says it all: the full text, the stopping rules, is in the files.
                None if body is not None else doc.sections[key],
                answer,
                body_extra=extra,
                found=headlines.get(key),
                title=titles[key],
                next_steps=after,
            )
        )
    limits = doc.limits(list(REPORT_QUESTIONS), "short")
    cards.append(
        _section(
            "limits",
            None,
            limits,
            doc.limits(list(REPORT_QUESTIONS), "full"),
            None,
            title=LIMITS_TITLE,
        )
    )
    # The next experiment, planned with optimise and live (_html_plan; owner, 2026-10-02).
    cards.append(_card("plan", PLAN_TITLE, "", plan_card(result, objects, fovs, _TERMS.annotate)))
    texts = {
        "report_short.txt": result.summary() + "\n",
        "report_full.txt": result.summary(detail="full") + "\n",
    }
    if result.discovery is not None and len(result.discovery.tests):
        texts["feature_discovery.csv"] = result.discovery.tests.to_csv(index=False)
    files = page_files(folder, texts, [Path(p) for p in attach], started)
    cards.append(files_card(files))
    # The page names the q-value the adjusted p (owner, 2026-10-02).
    cards = [adjusted_p(card) for card in cards]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f'<div class="layout">{"".join(nav)}<main>'
        '<header class="top"><button type="button" class="theme-toggle">Light / dark</button>'
        f'<h1>{html.escape(title)}</h1><p class="sub">{html.escape(subtitle)}</p>'
        f'<p class="meta">Generated {stamp} by MorphoCoverage. The same answers as the text '
        "report; every figure and file is inside this page (Files).</p></header>"
        + "".join(cards)
        + "<footer>"
        + html.escape(_settings(result))
        + "</footer></main></div>"
        + adjusted_p(_glossary_drawer(subtitle + "".join(cards)))
        + '<script type="application/json" id="files-data">'
        + json_island(files)
        + f"</script><script>{SCRIPT}</script>"
        + _plotly_scripts(specs, bool(viz))
        + "</body></html>"
    )
    path = folder / REPORT_FILE
    path.write_text(page, encoding="utf-8")
    return path


def _plotly_scripts(specs: list[dict], embedding: bool) -> str:
    """plotly.js, the feature figures' data and the script that draws them; nothing when
    the page has no interactive figure."""
    if not specs and not embedding:
        return ""
    data = json_island(specs)
    return (
        f'<script type="application/json" id="feature-data">{data}</script>'
        f"<script>{plotly_js()}</script><script>{PLOTLY_SCRIPT}</script>"
    )


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


def zip_results(folder: str | Path) -> Path:
    """Every file of ``folder`` (its subfolders included) in ``folder/<name>.zip``, for
    archiving or sending as one file; a zip already there is replaced, never included."""
    import zipfile

    folder = Path(folder)
    target = folder / f"{folder.name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, Path(folder.name) / path.relative_to(folder))
    return target


__all__ = ["REPORT_FILE", "html_report", "zip_results"]
