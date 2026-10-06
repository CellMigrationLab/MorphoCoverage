"""The short cards of the HTML report (private; ``html_report``), each its answer first and
the rest under "Show more" (owner, 2026-10-02):

- ``help_icon``: what a section is for, beside its title, shown on hover or a tap.
- ``check_rows``: "The data" as one row per check (the data checks, the integrity), each
  with its chip and its finding in a few words.
- ``flagged_fields``: the fields to check and why, the integrity's, the acquisition
  anomalies' and the inspection's but its new states, which ``new_state_fields`` lists apart
  (owner, 2026-10-06, audit 10).
- ``significant_summary``: "Do the conditions differ?" in bullets, condition by condition:
  the features significant across repeats; with fewer than 3 biological repeats, the
  candidates of the exploratory cell-level screen, said as such; ``readout_summary`` with
  one feature, the conditions by their verdict on it.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .._style import Status, group_label, span
from .._wording import change, count, listing, p_value
from ..retrospective.inspection_text import _fields, _keys
from ..sampling_unit._experiments import not_compared
from ..sampling_unit.acquisition import difference_words
from ..sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS
from ..sampling_unit.reading_text import VERDICTS, condition_readings, unpaired_conditions
from ..schema import observation_words
from ._distribution import split_words
from ._enough_features import NOT_YET
from ._populations import only_labelled

_MARKERS = {status.marker: status for status in Status}
_MARKER_LINE = re.compile(r"^\s*(✅|⚠️|❌|➖)\s+(.*)$")
# The chip of a check: its kind (the page's colours) and its word.
_CHECK_WORDS = {
    Status.OK: ("ok", "OK"),
    Status.CAUTION: ("caution", "Please check"),
    Status.PROBLEM: ("problem", "Please check"),
    Status.NOT_ASSESSED: ("na", "Partly checked"),
}
# The features named per condition in the summary; the rest are counted.
NAMED = 6


def help_icon(text: str | None) -> str:
    """A "?" beside a title that shows ``text`` on hover, focus or a tap."""
    if not text:
        return ""
    words = html.escape(text)
    return (
        f'<span class="help" tabindex="0" role="note" aria-label="{words}" '
        f'data-help="{words}">?</span>'
    )


def check_chip(status: Status) -> str:
    """The chip of one check of "The data": its marker and word ("OK", "Please check",
    "Partly checked")."""
    kind, word = _CHECK_WORDS[status]
    icon = f'<span aria-hidden="true">{status.marker}</span>'
    return f'<span class="chip {kind}">{icon} {word}</span>'


def _marker_lines(text: str) -> list[tuple[Status, str]]:
    """The lines of ``text`` that open with a marker, their wrapped continuations joined."""
    found: list[tuple[Status | None, str]] = []
    for line in text.splitlines():
        match = _MARKER_LINE.match(line)
        if match:
            found.append((_MARKERS[match.group(1)], match.group(2).strip()))
        elif found and line.startswith("   ") and line.strip():
            status, words = found[-1]
            found[-1] = (status, f"{words} {line.strip()}")
        elif line.strip():
            found.append((None, ""))  # a line without a marker ends the one before
    return [(s, w) for s, w in found if s is not None]


def check_rows(text: str, annotate: Callable[[str], str], fields_named: bool) -> str:
    """One row per check of "The data": its chip, its name and its finding in a few words
    (the rest of the sentence is under "Show more")."""
    rows = []
    for status, words in _marker_lines(text):
        name, _, rest = words.partition(": ")
        short = re.split(r"; |: ", rest, maxsplit=1)[0].rstrip(".") if rest else ""
        if (
            fields_named
            and status in (Status.PROBLEM, Status.CAUTION)
            and ("integrity" in name.lower() or "acquisition anomalies" in name.lower())
        ):
            short += " (listed under “Fields to check”)"
        rows.append(
            f"<li>{check_chip(status)} <strong>{html.escape(name)}</strong>"
            + (f" {annotate(html.escape(short))}" if short else "")
            + "</li>"
        )
    return f'<ul class="check-rows">{"".join(rows)}</ul>' if rows else ""


def flagged_fields(result: Any) -> tuple[list[tuple[str, str]], list[tuple[Any, ...]], list[str]]:
    """The fields to check, quality control: ``(field, why)`` rows (the integrity's first,
    then the acquisition anomalies, then the inspection's reasons but its new states), the
    fields' identities (their values of the key) and the key's columns. The new states are
    exploratory and listed apart (``new_state_fields``), never among the fields to remove."""
    results = getattr(result, "results", {}) or {}
    rows: list[tuple[str, str]] = []
    identities: list[tuple[Any, ...]] = []
    key: list[str] = []
    integrity = results.get("integrity")
    named: list[str] = []
    if integrity is not None and len(integrity.findings):
        labels = integrity.settings.get("field_labels", {})
        for finding in integrity.findings.itertuples():
            # Of a pair of copies one is the original: the code lists the other (the copy).
            named.append(str(finding.other_field))
            field = labels.get(finding.field, finding.field)
            other = labels.get(finding.other_field, finding.other_field)
            same = isinstance(other, str) and other and other != field
            # The first clause of the detail; the rest is in the full text.
            detail = re.sub(r"\s*\([^)]*\)", "", str(finding.detail).split(": ")[0])
            why = f"{finding.check}: {detail}" + (f" (with {other})" if same else "")
            rows.append((str(field), why))
    inspection = results.get("inspection")
    anomalies = results.get("acquisition")
    if anomalies is not None and anomalies.found:
        key = list(anomalies.settings["fov_key"])
        for row in anomalies.fields.to_dict("records"):
            side = "above" if row["z"] > 0 else "below"
            why = (
                f"far from its condition and repeat: {row['feature']}, "
                f"{abs(row['z']):.1f} SDs {side}, {difference_words(row)}"
            )
            rows.append((str(row["field"]), why))
            identities.append(tuple(row[c] for c in key))
    if inspection is not None and not inspection.fovs.empty:
        groups, fov_key, _ = _keys(inspection)
        key = [*groups, *fov_key]
        for group, name, own in _fields(inspection):
            own = own[own["reason"] != "new state"]
            if own.empty:
                continue
            label = " · ".join(p for p in (group_label(group) if groups else "", name) if p)
            rows.append((label, "; ".join(own["detail"])))
            # Flagged only for cells of the declared rare label: biology, never an artefact.
            if own["detail"].astype(str).str.contains("(all labelled", regex=False).all():
                continue
            first = own.iloc[0]
            identities.append(tuple(first[c] for c in key))
    columns = integrity.settings.get("field_columns", []) if integrity is not None else []
    key = key or list(columns)
    if named and list(columns) == key:
        # The integrity's fields first, as in the table: "condition=a, biological_repeat=1".
        found = [_field_values(name, key) for name in dict.fromkeys(named)]
        identities = [f for f in found if f is not None] + identities
    return rows, identities, key


def new_state_fields(result: Any) -> tuple[list[tuple[str, str]], list[str]]:
    """The fields holding a new state, ``(field, why)``, and why a group lists none (the
    definition not informative at its size): exploratory, to look at, never to remove
    (owner, 2026-10-06, audit 10)."""
    inspection = (getattr(result, "results", {}) or {}).get("inspection")
    if inspection is None:
        return [], []
    rows = []
    if not inspection.fovs.empty:
        groups, _, _ = _keys(inspection)
        for group, name, own in _fields(inspection):
            own = own[own["reason"] == "new state"]
            if len(own):
                label = " · ".join(p for p in (group_label(group) if groups else "", name) if p)
                rows.append((label, "; ".join(own["detail"])))
    reasons = list(
        dict.fromkeys(e["reason"] for e in inspection.settings.get("uninformative_novelty", []))
    )
    return rows, reasons


def _field_values(name: str, key: list[str]) -> tuple[str, ...] | None:
    """An integrity finding's field (``col=value, col=value``) as its values of ``key``."""
    pairs = dict(part.split("=", 1) for part in re.split(r", (?=[\w ]+=)", name) if "=" in part)
    return tuple(pairs[c] for c in key) if all(c in pairs for c in key) else None


def fields_table(rows: list[tuple[str, str]], annotate: Callable[[str], str]) -> str:
    """The fields to check, one row per field with why, sortable."""
    body = "".join(
        f"<tr><td>{html.escape(field)}</td><td>{annotate(html.escape(why))}</td></tr>"
        for field, why in rows
    )
    return (
        '<div class="table-wrap"><table class="sortable fields"><thead><tr><th>field</th><th>why'
        f"</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def _direction(row: Any) -> str:
    return "higher" if float(row.d) > 0 else "lower"


def _named(rows: Any, words: Callable[[Any], str]) -> str:
    """One sub-bullet per feature, the first ``NAMED``; the rest counted. With explainable
    tables (a ``table`` column), the features listed under their table (owner, 2026-10-06)."""
    first = rows.head(NAMED)
    if "table" in first:
        shown = [
            f"<li>{html.escape(str(table))}<ul>"
            + "".join(f"<li>{html.escape(words(r))}</li>" for r in mine.itertuples())
            + "</ul></li>"
            for table, mine in first.groupby("table", sort=False)
        ]
    else:
        shown = [f"<li>{html.escape(words(r))}</li>" for r in first.itertuples()]
    more = len(rows) - len(first)
    if more > 0:
        shown.append(f"<li>and {more} more (the heatmap below)</li>")
    return f"<ul>{''.join(shown)}</ul>"


def _change(row: Any) -> str:
    """The change against the control, as the paired test compares it (``×1.25``)."""
    value = getattr(row, "change", np.nan)
    return change(value, getattr(row, "paired_values", "")) if pd.notna(value) else ""


def _significant(row: Any, unit: str) -> str:
    q = f"adjusted p {p_value(row.q_paired)}" if np.isfinite(row.q_paired) else ""
    size = ", ".join(x for x in (_change(row), q) if x)
    # The direction and its count from the repeats, as the test across repeats reads them.
    word = "higher" if row.higher >= row.lower else "lower"
    return (
        f"{row.feature}: {word} in {max(row.lower, row.higher)} of {row.repeats} "
        f"repeats ({size}; {unit}-level d {row.d:+.2f})"
    )


def _candidate(row: Any, unit: str) -> str:
    q = p_value(row.q_cells)
    size = f"{_change(row)}, " if _change(row) else ""
    return f"{row.feature}: {_direction(row)} ({size}{unit}-level d {row.d:+.2f}, adjusted p {q})"


def _condition_words(
    condition: Any, rows: Any, beyond: list[str], unit: str, split: str = ""
) -> tuple[str, str]:
    """A condition's words and its named features (markup), for its bullet (``unit``: the
    word for one observation)."""
    rows = rows.assign(_size=rows["d"].abs()).sort_values("_size", ascending=False)
    repeats = int(rows["repeats"].max())
    listed = ""
    confirmed = rows[rows["confirmed"].fillna(False).astype(bool)]
    candidates = rows[rows["passes"].fillna(False).astype(bool)]
    if repeats >= MIN_REPEATS:
        if len(confirmed):
            words = count(len(confirmed), "feature")
            listed = _named(confirmed, lambda row: _significant(row, unit))
        else:
            # The trend line from TREND_REPEATS repeats only (owner, 2026-10-06, audit 8).
            consistent = candidates[
                (
                    (candidates["lower"] == candidates["repeats"])
                    | (candidates["higher"] == candidates["repeats"])
                )
                & (candidates["repeats"] >= TREND_REPEATS)
            ]
            words = "no feature significant across repeats"
            if len(consistent):
                named = [f"{r.feature} {_direction(r)}" for r in consistent.itertuples()]
                trend = listing(named, 4)
                words += f"; the same direction in every repeat, not significant: {trend}"
    else:
        same = candidates[
            (candidates["lower"] == candidates["repeats"])
            | (candidates["higher"] == candidates["repeats"])
        ]
        kept = same if repeats > 1 else candidates
        go = "goes" if len(kept) == 1 else "go"
        both = f" that {go} the same way in both repeats" if repeats == 2 else ""
        words = f"{count(len(kept), 'candidate')}{both}" if len(kept) else _quiet(unit)[1]
        listed = _named(kept, lambda row: _candidate(row, unit)) if len(kept) else ""
    if beyond:
        words += f"; beyond the mean, the distribution differs for {listing(beyond, 4)}"
    if split:
        words += f"; {split}"
    return words, listed


def _quiet(unit: str) -> tuple[str, str]:
    """The words of a condition with nothing to say beyond them, one of the quiet ones
    (``unit``: the word for one observation)."""
    return "no feature significant across repeats", f"no candidate in the {unit}-level screen"


# From this many quiet conditions on, they share one line (stress test, 2026-10-05: 150
# bullets saying the same).
QUIET_FOLDED = 4


def significant_summary(result: Any, annotate: Callable[[str], str]) -> str:
    """The bullets of "Do the conditions differ?" (see the module); empty without a feature
    discovery."""
    discovery = getattr(result, "discovery", None)
    tests = getattr(discovery, "tests", None)
    if tests is None or not len(tests):
        return ""
    distribution = (getattr(result, "results", {}) or {}).get("distribution")
    differing = distribution.differing() if distribution is not None else None
    controls = list(dict.fromkeys(tests["control_condition"].astype(str)))
    unit = observation_words(getattr(discovery, "settings", {}))[0]
    # Each condition's own number of repeats: one count when they share it, else the range.
    held = sorted({int(n) for n in tests.groupby("condition", sort=False)["repeats"].max()})
    repeats = held[-1]
    if repeats >= MIN_REPEATS:
        across = (
            f"the {held[0]} biological repeats"
            if len(held) == 1
            else f"the biological repeats, {span(held)} per condition"
        )
        head = (
            f"Compared with {listing(controls, 8)}, the features significant across {across} "
            "(the paired t-test on the repeat means, adjusted p ≤ 0.05):"
        )
    else:
        have = (
            count(held[0], "biological repeat")
            if len(held) == 1
            else f"{span(held)} biological repeats"
        )
        head = (
            f"Compared with {listing(controls, 8)}: with {have}, "
            f"no feature can be called significant (the test across repeats needs "
            f"{MIN_REPEATS}). The candidates of the {unit}-level screen, exploratory (every "
            f"{unit} an independent draw, so it asks whether these wells differ, not whether the "
            "difference repeats):"
        )
    bullets, quiet = [], {}
    for condition, rows in tests.groupby("condition", sort=False):
        beyond = []
        if differing is not None and len(differing):
            mine = differing[differing["condition"] == condition]
            beyond = [str(f) for f in mine["feature"]]
        split = split_words(distribution, condition)
        words, listed = _condition_words(condition, rows, beyond, unit, split)
        if words in _quiet(unit) and not listed:
            quiet.setdefault(words, []).append(str(condition))
        name = f"<strong>{html.escape(str(condition))}</strong>"
        bullets.append((words, f"<li>{name}: {html.escape(words)}{listed}</li>"))
    folded = {w for w, names in quiet.items() if len(names) >= QUIET_FOLDED}
    bullets = [b for w, b in bullets if w not in folded] + [
        f"<li><strong>{count(len(quiet[w]), 'condition')}</strong>: {html.escape(w)} "
        f"({html.escape(listing(quiet[w], 6))})</li>"
        for w in quiet
        if w in folded
    ]
    unpaired = [str(c) for c in getattr(discovery, "settings", {}).get("unpaired", [])]
    return (
        f'<div class="summary"><p>{annotate(html.escape(head))}</p>'
        f'<ul class="significant">{annotate("".join(bullets))}</ul>'
        + _not_compared(unpaired, listing(controls, 8), annotate)
        + "</div>"
    )


def _not_compared(unpaired: list[str], control: str, annotate: Callable[[str], str]) -> str:
    """The conditions no biological repeat holds beside the control, once (empty without)."""
    if not unpaired:
        return ""
    return f"<p>{annotate(html.escape(not_compared(unpaired, control)))}</p>"


def readout_summary(result: Any, annotate: Callable[[str], str]) -> str:
    """With one feature (no feature discovery): the conditions by verdict on the readout,
    each with its direction; empty without a contrast across repeats."""
    contrast = (getattr(result, "results", {}) or {}).get("contrast")
    readings = condition_readings(contrast) if contrast is not None else None
    if readings is None or readings.empty:
        return ""
    controls = list(dict.fromkeys(readings["control_condition"].astype(str)))
    unpaired = _not_compared(unpaired_conditions(contrast), listing(controls, 8), annotate)
    bullets = []
    for verdict in VERDICTS:
        rows = readings[readings["verdict"] == verdict]
        if rows.empty:
            continue
        by_direction = [
            f"{listing([str(c) for c in group['condition']], 8)} ({direction})"
            for direction, group in rows.groupby("direction", sort=False)
        ]
        words = "; ".join(by_direction)
        bullets.append(
            f"<li><strong>{html.escape(verdict[:1].upper() + verdict[1:])}</strong>: "
            f"{html.escape(words)}</li>"
        )
    head = f"{result.setup.readout}, compared with {listing(controls, 8)}, condition by condition:"
    return (
        f'<div class="summary"><p>{annotate(html.escape(head))}</p>'
        f'<ul class="significant">{annotate("".join(bullets))}</ul>{unpaired}</div>'
    )


def chip_words(result: Any) -> dict[str, str]:
    """The chip words the results decide, by question, where the headline's words could
    mislead: "Not yet" when no feature has enough fields to compare with the control, "The
    declared label" when the only group found is the rare label's."""
    words = {}
    features = (getattr(result, "results", {}) or {}).get("enough_features")
    table = getattr(features, "table", None)
    if table is not None and len(table) and (table["verdict"] == NOT_YET).all():
        words["enough"] = "Not yet"
    # The only group is the declared label's, compared as its share: nothing enriched here.
    if only_labelled((getattr(result, "results", {}) or {}).get("populations")):
        words["populations"] = "The declared label"
    return words


__all__ = [
    "check_chip",
    "check_rows",
    "chip_words",
    "fields_table",
    "flagged_fields",
    "help_icon",
    "new_state_fields",
    "readout_summary",
    "significant_summary",
]
