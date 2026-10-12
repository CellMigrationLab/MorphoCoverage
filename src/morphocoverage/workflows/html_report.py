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
  condition, a **heatmap** of Cohen's d (the features no condition makes a candidate named
  under it), the **figures** of the best-ranked features, each with its text for a figure
  legend folded, then the tests behind the reading, one kind a tab, and under "Show more"
  the section's assumptions only (the readings, the ranking, the plan and the tables within
  single repeats stay in the text report);
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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .._style import Status
from .._wording import listing
from ..embedding import MIN_FEATURES
from ..glossary import GLOSSARY, PHRASES, slug
from ..retrospective.report import LIMITS_TITLE
from ..sampling_unit.across_text import across_parts
from ..sampling_unit.reading_text import readings_parts
from ..sufficiency_short import rule_tallies
from ._exploratory import exploratory_html
from ._feature_space import attached, space
from ._html_cards import (
    check_rows,
    chip_words,
    help_icon,
    readout_summary,
    significant_summary,
)
from ._html_data import (
    LINKED,
    across_data,
    data_buttons,
    files_card,
    heatmap_data,
    inline_images,
    json_island,
    page_files,
)
from ._html_enough import enough_body, enough_help
from ._html_fields import NEW_STATES, NEW_STATES_HELP, fields_cards, has_new_states
from ._html_figures import feature_figures
from ._html_maps import embeddings, maps_body
from ._html_next import populations_next
from ._html_plan import PLAN_TITLE, plan_card
from ._html_plotly import plotly_js
from ._html_plotly_script import PLOTLY_SCRIPT
from ._html_render import Terms, adjusted_p, chip, first_sentence, in_units, text_html
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
    titles_for,
    unnumbered,
    what_checked,
)
from ._html_style import CSS
from ._html_tests import figure_legends, page_assumptions, tests_section
from ._source import verify_source
from ._whole_table import whole_summary_html, whole_table_html, whole_table_lines
from .retrospective import REPORT_QUESTIONS, Report

REPORT_FILE = "report.html"


def _on_page(term: Any, text: str) -> bool:
    """Whether the page's ``text`` (lower case, without tags) uses ``term``: its name or one
    of its phrases, whole."""
    name = re.sub(r"\s*\(.*\)$", "", term.name)
    words = (name, *PHRASES.get(term.name, ()))
    words = (*words, *(_words(w) for w in words))
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
        name, meaning, where = (_words(t) for t in (term.name, term.meaning, term.where))
        parts.append(
            f'<dt id="{slug(term)}">{html.escape(name)}</dt><dd>{html.escape(meaning)}'
            f'<span class="where">Where: {_unnumbered(html.escape(where))}.</span>'
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
_UNITS = {"words": "cells", "one": "cell"}


def _words(text: str) -> str:
    """``text``, written of cells, in this page's words for its observations."""
    return in_units(text, _UNITS["words"], _UNITS["one"])


def _unnumbered(markup: str) -> str:
    """``unnumbered`` in this page's words for its observations."""
    return unnumbered(markup, _UNITS["words"], _UNITS["one"])


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
    """The fields to check, and why, and the new states apart (``_html_fields``)."""
    chip_html, title = chip(status, "inspect", short), TITLES["inspect"]
    return fields_cards(result, chip_html, _html(short), _html(full), _card, _TERMS.annotate, title)


def _dataset_card(result: Report, viz: str, static: list[str]) -> str:
    """Every feature at once: the interactive PCA and UMAP, the static figures folded."""
    return _card("dataset", _words(TITLES["dataset"]), "", maps_body(result.setup, viz, static))


def _condition_extras(result: Report, feature_blocks: list[str], folder: Path) -> str:
    """Question 9's figures: the heatmap (its numbers as CSV), the best-ranked features."""
    conditions = []
    contrast = result.results.get("contrast")
    if contrast is not None and len(contrast.table):
        conditions = list(dict.fromkeys(contrast.table["condition"]))
    extra = ""
    grid = heatmap(result.discovery, conditions, result.setup.unit_word)
    if grid:
        files = heatmap_data(result.discovery, folder)
        # The features no condition makes a candidate, under the heatmap (owner, 2026-10-02).
        unranked = "".join(
            f'<p class="note">{html.escape(line)}</p>' for line in result.discovery._unranked(8)
        )
        extra += (
            f"<h3>Every feature against the control</h3>{grid}{unranked}"
            f'<p class="tools">{data_buttons(files)}</p>'
        )
    across = across_parts(contrast, indent=0) if contrast is not None else []
    if across:  # several controls: the secondary rows, the contrast table (owner, 2026-10-05)
        buttons = data_buttons(across_data(contrast, folder, result.setup))
        extra += "<h3>Compared across experiments</h3>" + _html("\n".join(across))
        extra += f'<p class="tools">{buttons}</p>' if buttons else ""
    # Every other table tested whole (owner, 2026-10-06), its data as CSV.
    extra += whole_table_html(result.results.get("whole_table"), folder, _html, data_buttons)
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
    folder: Path,
) -> str:
    """Question 9: the significant features per condition, the figures (each with its
    legend), then the tests behind them, the assumptions under "Show more"; planning the
    next experiment is the page's last section (_html_plan). The readings, the ranking and
    the plan are not repeated here (owner, 2026-10-02): the tabs and the figures hold their
    numbers, and the text report every one."""
    contrast = result.results.get("contrast")
    reading = "\n".join(readings_parts(contrast, indent=2)) if contrast is not None else ""
    summary = significant_summary(result, _TERMS.annotate) or readout_summary(
        result, _TERMS.annotate
    )
    whole = result.results.get("whole_table")
    if summary and whole is not None and len(whole.tests):
        summary += whole_summary_html(whole)
    rest = short[len(reading) :].lstrip("\n") if reading and short.startswith(reading) else short
    whole = result.results.get("whole_table")
    if whole is not None and len(whole.tests):  # its own block, among the extras
        rest = rest.replace("\n".join(whole_table_lines(whole)), "")
    # Without a summary (no reading by verdict), the readings answer.
    body = (summary or _html(reading)) + extras
    if rest.strip():
        # One kind of test at a time (_html_tabs).
        body += "<h3>The tests behind the reading</h3>" + tests_section(result, rest, _html, folder)
    # The summary answers, so the headline is not repeated above it (it stays the chip's).
    shown = None if summary else answer
    return _section(
        "conditions",
        status,
        "",
        page_assumptions(full, contrast),
        shown,
        body_extra=body,
        found=answer,
    )


def html_report(
    result: Report,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    folder: str | Path,
    *,
    title: str = "MorphoCoverage report",
    attach: Sequence[str | Path] = (),
    feature_tables: Mapping[str, pd.DataFrame | tuple[pd.DataFrame, str]] | None = None,
    check_source: bool = True,
) -> Path:
    """Write ``folder/report.html`` and its figures (see the module); returns the page's path.

    ``objects`` and ``fovs`` are the tables the ``report`` was run on; ``attach`` are more files
    the page carries for download (the design scan, the saved answers): one not found is named
    in the Files card, and two of one name are both carried (``_html_data.page_files``).
    ``feature_tables`` are the feature tables given to ``report`` (with their tags), attached
    again here. ``objects`` and ``fovs`` must be the very tables the report was run on (their
    digest is recorded on the report): the page never mixes its answers with figures or
    downloads from other data. ``check_source=False`` skips the check for a report that records
    no source (one made before sources were recorded).
    """
    if not isinstance(result, Report):
        raise TypeError("result must be the Report that report() returned")
    if check_source:
        verify_source(result.source, objects, fovs)
    # The figures are drawn from these tables, the answers were computed from the report's.
    counts = getattr(result.results.get("overview"), "counts", None)
    if counts is not None and (len(objects), len(fovs)) != (counts["objects"], counts["fields"]):
        raise ValueError(
            f"objects and fovs must be the tables the report was run on ({counts['objects']} "
            f"observations, {counts['fields']} fields), not {len(objects)} rows and {len(fovs)}"
        )
    objects = attached(objects, fovs, result.setup, feature_tables)[0]
    folder = Path(folder)
    # The "?" help in the report's words for its observations (tracks, patches).
    _INTROS.clear()
    units, one = result.setup.unit_words, result.setup.unit_word
    _INTROS.update(intros_for(units, one, **enough_help(result)), **{NEW_STATES: NEW_STATES_HELP})
    _WORDS.clear()
    _WORDS.update(chip_words(result))
    _UNITS.update(words=units, one=one)
    _TERMS.use(units, one)
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # Allow for file systems that store modification times to the second.
    started = time.time() - 1
    doc = result.document
    with warnings.catch_warnings():
        warnings.filterwarnings(  # fallback: the data check is said once, in its own section
            "ignore", message="check_tables found", category=UserWarning
        )
        legends = figure_legends(result, objects, fovs)
        legends = {f: _TERMS.annotate(html.escape(t)) for f, t in legends.items()}
        feature_blocks, specs = feature_figures(result, objects, fovs, figures, legends)
        viz, static = ("", [])
        if len(space(result.setup)[0]) >= MIN_FEATURES:
            viz, static = embeddings(result, objects, fovs, figures)
    statuses, headlines = doc.statuses, doc.headlines
    # The titles name the report's own observations (tracks, patches).
    titles = titles_for(units, one)
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
        if key == "inspect" and has_new_states(result):  # the new states' card (audit 10)
            nav.append(
                f'<a href="#{NEW_STATES}"><span class="icon"></span><span>New states</span></a>'
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
    subtitle = first_sentence(overview_short)
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
                    figures,
                )
            )
            continue
        extra, answer, after = "", headlines.get(key), ""
        # One sentence and a table, then one feature at a time (_html_enough).
        body = enough_body(result, _TERMS.annotate) if key == "enough" else None
        reading = enough_reading(result) if key == "enough" and body is None else []
        if body is not None:  # the section's text is not printed: its exploratory line is
            extra, answer = body + exploratory_html(result, key), None
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
    # The page leaves out the sections it cannot answer: their reasons stay here.
    limits = doc.limits(list(REPORT_QUESTIONS), "short", pointers=False)
    cards.append(
        _section(
            "limits",
            None,
            limits,
            doc.limits(list(REPORT_QUESTIONS), "full", pointers=False),
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
    if result.setup.contrast_table is not None:  # one row per test (owner, 2026-10-05)
        texts["contrasts.csv"] = result.setup.contrast_table.to_csv(index=False)
    files, images = page_files(folder, texts, [Path(p) for p in attach], started)
    cards.append(files_card(files, attach))
    # The page names the q-value the adjusted p (owner, 2026-10-02); then each PNG it
    # carries is given its data, once (past the budget it stays a link, _html_data).
    cards = [inline_images(adjusted_p(card, units, one), images) for card in cards]
    inside = (
        "every figure and file is inside this page (Files)"
        if all(entry[1] != LINKED for entry in files.values())
        else "the larger files open from the figures folder beside it (Files)"
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
        f"report; {inside}.</p></header>"
        + "".join(cards)
        + "<footer>"
        + html.escape(_settings(result))
        + "</footer></main></div>"
        + adjusted_p(_glossary_drawer(subtitle + "".join(cards)), units, one)
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


def _settings(result: Report) -> str:
    text = result.summary().rsplit("\n\n", 1)[-1]
    return text if text.startswith("Settings") else ""


def zip_results(folder: str | Path, *, since: float | None = None) -> Path:
    """Every file of ``folder`` (its subfolders included) in ``folder/<name>.zip``, for
    archiving or sending as one file; that zip, when already there, is replaced, never
    included, and any other zip (an attachment) is included. With
    ``since`` (a ``time.time()``), the figures modified before it are an earlier run's,
    left out as the page leaves them out."""
    import zipfile

    folder = Path(folder)
    target = folder / f"{folder.name}.zip"
    figures = folder / "figures"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path == target:
                continue
            if since is not None and path.parent == figures and path.stat().st_mtime < since:
                continue
            archive.write(path, Path(folder.name) / path.relative_to(folder))
    return target


__all__ = ["REPORT_FILE", "html_report", "zip_results"]
