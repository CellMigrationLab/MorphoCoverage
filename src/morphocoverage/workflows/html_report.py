"""The report as one HTML page, with its figures as PNG and PDF beside it.

``html_report(result, objects, fovs, folder)`` writes ``folder/report.html``: a single file
that opens in any browser, offline, and can be emailed or archived as it is. It holds:

- a sidebar with every section's status, and **At a glance** (each answer's marker, as a
  chip with its word, and its one-line answer, linking to the section);
- the sections in the order a user acts on them, titled in words (``_html_sections``): the
  data, **what was checked**, the fields to check (quality control), whether the conditions
  differ, whether enough was imaged, and the rest; a section this design cannot answer is
  left out (its reason stays in "What this dataset cannot tell me"). Each says what it is
  for, its answer in a line, its short text, and its full text folded under "Full detail":
  the words of ``Report.summary``, its aligned tables as sortable tables;
- under "Do the conditions differ?", the reading per condition, a **heatmap** of Cohen's d,
  the **figures** (the readout and the best-ranked features) and with
  ``embedding.MIN_FEATURES`` or more features an **interactive PCA and UMAP**
  (``_html_figures``), then the tests behind the reading;
- the **glossary** as a drawer on the side, with a search, and the **Files**.

Every figure is also saved in ``folder/figures`` as PNG and PDF (vector, for papers);
a click enlarges it. The page follows the viewer's light or dark setting, with a switch;
it prints with every section open. Nothing on it is computed anew except the figures: the
text is the report's.
"""

from __future__ import annotations

import base64
import html
import json
import time
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .._style import Status
from ..embedding import MIN_FEATURES
from ..glossary import GLOSSARY, slug
from ..retrospective.report import LIMITS_TITLE
from ..sampling_unit.reading_text import readings_parts
from ._html_figures import download, embeddings, feature_figures
from ._html_plotly import plotly_js
from ._html_plotly_script import PLOTLY_SCRIPT
from ._html_render import Terms, chip, text_html
from ._html_script import SCRIPT
from ._html_sections import (
    INTROS,
    ORDER,
    TITLES,
    enough_reading,
    heatmap,
    hidden,
    unnumbered,
    what_checked,
)
from ._html_style import CSS
from .retrospective import REPORT_QUESTIONS, Report

REPORT_FILE = "report.html"
_TYPES = {
    ".png": "image/png",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
}


def _files(
    folder: Path, texts: dict[str, str], attach: Sequence[Path], since: float
) -> dict[str, list[str]]:
    """Every file the page carries, by the path it is saved under: the figures this run wrote
    in ``folder/figures`` (modified at or after ``since``, so an older run's are left out), the
    ``texts`` written here, and the ``attach`` files."""
    files: dict[str, list[str]] = {}
    for path in sorted((folder / "figures").glob("*")):
        if path.suffix in _TYPES and path.stat().st_mtime >= since:
            data = base64.b64encode(path.read_bytes()).decode()
            files[f"figures/{path.name}"] = [_TYPES[path.suffix], data]
    for name, text in texts.items():
        mime = _TYPES.get(Path(name).suffix, "text/plain")
        files[name] = [mime, base64.b64encode(text.encode("utf-8")).decode()]
    for path in attach:
        path = Path(path)
        if path.exists():
            mime = _TYPES.get(path.suffix, "application/octet-stream")
            files[path.name] = [mime, base64.b64encode(path.read_bytes()).decode()]
    return files


def _files_card(files: dict[str, list[str]]) -> str:
    rows = "".join(
        f'<tr><td>{html.escape(name)}</td><td class="num">'
        f"{len(base64.b64decode(data)) / 1024:.0f} kB</td><td>{download(name, 'Save')}</td></tr>"
        for name, (_, data) in files.items()
    )
    return (
        '<section class="card" id="files"><h2>Files</h2><p class="note">Everything this report '
        "was made with and from is inside this page: save any file with its button, no folder "
        "needed. The same files are in the results folder and its zip.</p>"
        '<div class="table-wrap"><table class="sortable"><thead><tr><th>file</th><th>size</th>'
        f"<th></th></tr></thead><tbody>{rows}</tbody></table></div></section>"
    )


def _glossary_drawer() -> str:
    """The glossary as a drawer on the side (opened by its tab, the sidebar and every
    underlined term), with a search; printed at the end of the page."""
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
    for term in GLOSSARY:
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
            f'<span class="where">Where: {unnumbered(html.escape(term.where))}.</span>'
            + (f'<span class="refs">See: {refs}</span>' if refs else "")
            + "</dd>"
        )
    parts.append("</dl></div></aside>")
    return "".join(parts)


_TERMS = Terms()


def _html(text: str) -> str:
    """The report's text as HTML, under this page's titles, its terms linked."""
    return _TERMS.annotate(unnumbered(text_html(text)))


def _section(
    key: str,
    status: Status | None,
    short: str,
    full: str | None,
    answer: str | None,
    *,
    title: str | None = None,
    body_extra: str = "",
) -> str:
    """A card: its title and status, what it is for, its answer in a line, its short text,
    and its full text folded."""
    intro = INTROS.get(key)
    head = f'<p class="intro">{html.escape(intro)}</p>' if intro else ""
    if answer:
        head += (
            '<p class="answer"><strong>In short:</strong> '
            f"{_TERMS.annotate(html.escape(answer[:1].upper() + answer[1:]))}.</p>"
        )
    body = body_extra or _html(short)
    if full and full.strip() != short.strip():
        detail = _html(full)
        body += f"<details><summary>Full detail</summary>{detail}</details>"
    name = title or TITLES.get(key, key)
    return (
        f'<section class="card" id="{key}"><h2>{chip(status)} {html.escape(name)}</h2>'
        f"{head}{body}</section>"
    )


def _checked_card(result: Report, doc: Any) -> str:
    """What the run checked, in short lines; the declaration's sentences folded."""
    lines = "".join(
        f"<li>{_TERMS.annotate(html.escape(line))}</li>" for line in what_checked(result.setup)
    )
    full = doc.short.get("declaration") or "\n".join(f"- {d}" for d in doc.declaration)
    detail = f"<details><summary>Full detail</summary>{text_html(full)}</details>" if full else ""
    return (
        '<section class="card" id="declaration"><h2>What was checked</h2>'
        f'<ul class="checked">{lines}</ul>{detail}</section>'
    )


def _condition_extras(
    result: Report, feature_blocks: list[str], viz: str, static: list[str]
) -> str:
    """Question 9's figures: the heatmap, the plots, the PCA and UMAP."""
    conditions = []
    contrast = result.results.get("contrast")
    if contrast is not None and len(contrast.table):
        conditions = list(dict.fromkeys(contrast.table["condition"]))
    extra = ""
    grid = heatmap(result.discovery, conditions)
    if grid:
        extra += f"<h3>Every feature against the control</h3>{grid}"
    if feature_blocks:
        extra += (
            "<h3>The readout and the best-ranked features</h3>"
            f'<div class="feature-grid">{"".join(feature_blocks)}</div>'
        )
    if viz:
        extra += f"<h3>All {len(result.setup.columns)} features at once</h3>{viz}"
        extra += (
            "<details><summary>Static PCA and UMAP (PNG and PDF)</summary>"
            f'<div class="figures">{"".join(static)}</div></details>'
        )
    return extra


def _conditions_card(
    result: Report, status: Status | None, short: str, full: str, answer: str | None, extras: str
) -> str:
    """Question 9: the reading per condition, then the figures, then the tests behind them."""
    contrast = result.results.get("contrast")
    reading = "\n".join(readings_parts(contrast, indent=2)) if contrast is not None else ""
    if reading and short.startswith(reading):
        rest = short[len(reading) :].lstrip("\n")
        body = _html(reading) + extras
        if rest.strip():
            body += "<h3>The tests behind the reading</h3>" + _html(rest)
    else:
        body = _html(short) + extras
    return _section("conditions", status, short, full, answer, body_extra=body)


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
    keys = [k for k in ORDER if (k == "overview" and doc.overview) or k in doc.sections]
    shown = [k for k in keys if not hidden(statuses.get(k), headlines.get(k, ""))]
    nav = ['<nav class="toc" aria-label="Sections"><h2>Report</h2>']
    glance = []
    for key in shown:
        status = statuses.get(key)
        icon = status.marker if status is not None else ""
        nav.append(
            f'<a href="#{key}"><span class="icon" aria-hidden="true">{icon}</span>'
            f"<span>{html.escape(TITLES[key])}</span></a>"
        )
        if key == "overview":
            nav.append(
                '<a href="#declaration"><span class="icon"></span><span>What was checked</span></a>'
            )
        if status is not None:
            glance.append(
                f'<a class="glance-row" href="#{key}"><span class="q">{html.escape(TITLES[key])}'
                f'<br>{chip(status)}</span><span class="a">'
                f"{html.escape(headlines.get(key, ''))}</span></a>"
            )
    for anchor, label in (
        ("limits", "What it cannot tell"),
        ("glossary", "Glossary"),
        ("files", "Files"),
    ):
        nav.append(f'<a href="#{anchor}"><span class="icon"></span><span>{label}</span></a>')
    nav.append("</nav>")
    overview_short = doc.short.get("overview") or doc.overview or ""
    subtitle = _first_sentence(overview_short)
    cards = [
        '<section class="card" id="glance"><h2>At a glance</h2><p class="key">A marker says '
        "how far to trust an answer, not what it found. Underlined words show their meaning, "
        "and a click opens them in the glossary, on the side.</p>"
        + _TERMS.annotate("".join(glance))
        + "</section>"
    ]
    for key in shown:
        if key == "overview":
            cards.append(
                _section(
                    "overview",
                    statuses.get("overview"),
                    overview_short,
                    doc.overview,
                    headlines.get("overview"),
                )
            )
            cards.append(_checked_card(result, doc))
            continue
        short = doc.short.get(key, doc.sections[key])
        if key == "conditions":
            extras = _condition_extras(result, feature_blocks, viz, static)
            cards.append(
                _conditions_card(
                    result, statuses.get(key), short, doc.sections[key], headlines.get(key), extras
                )
            )
            continue
        extra = ""
        extra, answer = "", headlines.get(key)
        reading = enough_reading(result) if key == "enough" else []
        if reading:
            # The reading answers in sentences, so the headline is not repeated above it.
            lines = "".join(f"<p>{html.escape(line)}</p>" for line in reading)
            extra = _TERMS.annotate(
                f'<div class="reading">{lines}</div><h3>The numbers</h3>'
                + unnumbered(text_html(short))
            )
            answer = None
        cards.append(
            _section(key, statuses.get(key), short, doc.sections[key], answer, body_extra=extra)
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
    texts = {
        "report_short.txt": result.summary() + "\n",
        "report_full.txt": result.summary(detail="full") + "\n",
    }
    if result.discovery is not None and len(result.discovery.tests):
        texts["feature_discovery.csv"] = result.discovery.tests.to_csv(index=False)
    files = _files(folder, texts, [Path(p) for p in attach], started)
    cards.append(_files_card(files))
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
        + _glossary_drawer()
        + '<script type="application/json" id="files-data">'
        + json.dumps(files, separators=(",", ":"))
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
    data = json.dumps(specs, separators=(",", ":")).replace("</", "<\\/")
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
