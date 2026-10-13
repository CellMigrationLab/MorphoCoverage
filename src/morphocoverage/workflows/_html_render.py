"""The HTML report's pieces (private; ``html_report``): the page's style and script, the
report's plain text as HTML, and the status chips.

The text of every answer is the same as ``Report.summary``: ``text_html`` turns its lines
into paragraphs (a wrapped sentence joined again), status lines (the marker as a chip with
its word), list items, muted assumptions and tables (lines whose columns are aligned by two
or more spaces become a sortable table). Nothing is reworded.
"""

from __future__ import annotations

import html
import re

from .._style import MARKER_LINE, MARKERS, Status, indent_width, without_provenance
from ..glossary import GLOSSARY, PHRASES, Term, slug

STATUS_WORDS = {
    Status.OK: ("ok", "Answered"),
    Status.CAUTION: ("caution", "Caution"),
    Status.PROBLEM: ("problem", "Check the data first"),
    Status.NOT_ASSESSED: ("na", "Not assessed"),
}
_COLUMNS = re.compile(r"\S {2,}\S")
_SPLIT = re.compile(r" {2,}")
_NUMBER = re.compile(r"^[−+-]?[<>≤≥]?\d[\d.,]*%?$")


def in_units(text: str, units: str, one: str) -> str:
    """``text``, written of cells, in the words of the report's observations (``units``,
    ``one``: ``patches``, ``patch``), capitals kept; "cell-level" names the test for them
    (``patch-level``)."""

    def word(match: re.Match) -> str:
        found = match.group(0)
        written = units if found.lower() == "cells" else one
        return written[:1].upper() + written[1:] if found[:1].isupper() else written

    return re.sub(r"\b[Cc]ells?\b", word, text)


class Terms:
    """The glossary's phrases in the report's text, each linked to its entry with its meaning
    as a tooltip: only the first mention of a term in each section (``annotate``), never inside
    a link or a table, so the text stays readable."""

    def __init__(self, units: str = "cells", one: str = "cell") -> None:
        self.use(units, one)

    def use(self, units: str, one: str) -> None:
        """Link the phrases as the page writes them for its observations (``in_units``:
        "patch-level test" as well as "cell-level test")."""
        self.units, self.one = units, one
        by_name = {term.name: term for term in GLOSSARY}
        self.lookup: dict[str, Term] = {}
        for name, phrases in PHRASES.items():
            for phrase in phrases:
                for written in {phrase, in_units(phrase, units, one)}:
                    self.lookup[html.escape(written).lower()] = by_name[name]
        alternatives = sorted(self.lookup, key=len, reverse=True)
        self.pattern = re.compile(
            r"(?<![\w-])(" + "|".join(re.escape(a) for a in alternatives) + r")(?![\w-])",
            re.IGNORECASE,
        )

    def _link(self, match: re.Match, seen: set[str]) -> str:
        term = self.lookup[match.group(1).lower()]
        if term.name in seen:
            return match.group(0)
        seen.add(term.name)
        return (
            f'<a class="term" href="#{slug(term)}">{match.group(0)}'
            f'<span class="tip" role="tooltip">'
            f"{html.escape(in_units(term.meaning, self.units, self.one))}</span></a>"
        )

    def annotate(self, markup: str) -> str:
        """``markup`` with the first mention of each term linked (its text, not its tags)."""
        seen: set[str] = set()
        out, in_link, in_table = [], 0, 0
        for piece in re.split(r"(<[^>]+>)", markup):
            if piece.startswith("<"):
                if re.match(r"<a[\s>]", piece):
                    in_link += 1
                elif piece.startswith("</a"):
                    in_link = max(0, in_link - 1)
                elif re.match(r"<(table|pre|select)[\s>]", piece):
                    in_table += 1
                elif re.match(r"</(table|pre|select)>", piece):
                    in_table = max(0, in_table - 1)
                out.append(piece)
            # A table scrolls sideways, which would clip a tooltip; its words are explained
            # around it. Code and a list's options are never linked.
            elif in_link or in_table or not piece.strip():
                out.append(piece)
            else:
                out.append(self.pattern.sub(lambda m: self._link(m, seen), piece))
        return "".join(out)


# What each answer found, per question and status (owner, 2026-10-01: the marker says the
# result, not how far to trust it); a question not listed keeps the status's own word.
RESULT_WORDS = {
    ("overview", Status.OK): "Clean",
    ("inspect", Status.OK): "Nothing flagged",
    ("inspect", Status.CAUTION): "Fields to inspect",
    ("conditions", Status.OK): "Differs",
    ("conditions", Status.CAUTION): "Trend only",
    # Population enrichment is exploratory: ⚠️, never "Enriched" (owner, 2026-10-06, audit 10).
    ("populations", Status.OK): "Groups found",
    ("populations", Status.CAUTION): "Trend only",
    ("enough", Status.OK): "Enough",
    ("enough", Status.CAUTION): "Partly",
}


# The page calls the Benjamini-Hochberg q the adjusted p (owner, 2026-10-02): the text
# report's wording, rewritten in the page's text (never inside its tags or scripts).
_ADJUSTED = (
    (re.compile(r"\bq-values\b"), "adjusted p-values"),
    (re.compile(r"\(q\)"), "(adjusted p)"),
    (re.compile(r"\bq repeat means\b"), "adjusted p, repeat means"),
    (re.compile(r"\bq: Benjamini"), "adjusted p: Benjamini"),
    (re.compile(r"Benjamini-Hochberg q\b(?!-)"), "Benjamini-Hochberg adjusted p"),
    (re.compile(r"\b(normalised|raw|and) q\b(?!-)"), r"\1 adjusted p"),
    (re.compile(r"\bq(?= (?:≤|<|&lt;|=|\d))"), "adjusted p"),
    # "q: the version ...", "p uncorrected, q corrected": a q before a colon or a verb.
    (re.compile(r"(?<![\w-])q(?=:)"), "adjusted p"),
    (re.compile(r"(?<![\w-])q (corrected|uncorrected)\b"), r"the adjusted p \1"),
)
_TEXT_NODE = re.compile(r">([^<]+)<")


_SCRIPT = re.compile(r"(<script\b.*?</script>)", re.DOTALL)


# The screen on every observation: the text report names it for the observations
# ("patch-level screen", the discovery table's "q patch level"); the page's own labels that
# are written of cells (the glossary, the tab "Cell-level screen") are named here for them
# too (owner, 2026-10-02, "cells could be the default"). A unit's own word never matches, so
# nothing is converted twice.
_SCREEN = re.compile(r"\b([Cc])ell([- ])level\b")


def adjusted_p(markup: str, units: str = "cells", one: str = "cell") -> str:
    """``markup`` with every q (a Benjamini-Hochberg q-value) in its text called the
    adjusted p, and the cell-level screen named for the report's observations (``units``,
    ``one``); its scripts (the figures' data) left as they are."""

    def screen(match: re.Match) -> str:
        word = one.capitalize() if match.group(1) == "C" else one
        return f"{word}{match.group(2)}level"

    # The discovery table's column, "q nucleus level" (as the text report writes it).
    column = re.compile(rf"\bq {re.escape(one)} level\b")

    def fix(match: re.Match) -> str:
        text = column.sub(f"adjusted p, {units}", match.group(1))
        for pattern, words in _ADJUSTED:
            text = pattern.sub(words, text)
        return f">{_SCREEN.sub(screen, text)}<"

    parts = _SCRIPT.split(markup)
    return "".join(
        part if part.startswith("<script") else _TEXT_NODE.sub(fix, part) for part in parts
    )


def chip(status: Status | None, key: str = "", answer: str = "", word: str | None = None) -> str:
    """The status as a coloured chip with its icon and the word for what the answer found
    (never colour alone): ``word`` when the results give it (``_html_cards.chip_words``),
    else per question (``key``) and, where the ``answer`` decides it, the answer."""
    if status is None:
        return ""
    kind, default = STATUS_WORDS[status]
    if word is None:
        word = _found_word(status, key, answer, default)
    icon = f'<span aria-hidden="true">{status.marker}</span>'
    return f'<span class="chip {kind}">{icon} {word}</span>'


def _found_word(status: Status, key: str, answer: str, default: str) -> str:
    """The chip's word for what the answer to question ``key`` found: the question's word for
    the status, or the answer's own (``_answer_word``)."""
    word = RESULT_WORDS.get((key, status), default)
    if key == "enough" and status is Status.CAUTION and "shown in 0 of" in answer:
        word = "Not yet"
    few = re.search(r"\bn = ([12])\b", answer) if key in ("conditions", "populations") else None
    if few:
        return f"No conclusion (n = {few.group(1)})"
    return _answer_word(status, key, answer) or word


def _answer_word(status: Status, key: str, answer: str) -> str | None:
    """The word the answer itself decides for the conditions and the populations, or None
    when the question's word stands."""
    if key == "conditions" and answer.startswith("nothing compared"):
        return "Nothing compared"  # audit 10 (F12): never "No consistent difference"
    if key == "populations" and answer.startswith("no group:"):
        return "No group"
    if key == "conditions" and status is Status.OK and answer.startswith("beyond the mean"):
        return "Differs beyond the mean"
    if key == "populations" and status is Status.CAUTION:
        return _population_caution(answer)
    if key == "conditions" and status is Status.CAUTION:
        return _trend(answer)
    return None


def _population_caution(answer: str) -> str:
    """The word for a population answer with a caution."""
    if "(exploratory;" in answer:
        return "More common (exploratory)"  # never "Enriched" (owner, 2026-10-06, audit 10)
    if "not a subpopulation" in answer:
        return "A condition's shift"
    return _trend(answer)


def _trend(answer: str) -> str:
    """A difference with a caution: a trend when it keeps one direction, else none."""
    consistent = "same direction" in answer or "in every repeat" in answer
    return "Trend only" if consistent else "No consistent difference"


def _table(rows: list[str]) -> str:
    cells = [_SPLIT.split(row.strip()) for row in rows]
    width = len(cells[0])
    if width < 2 or any(len(row) != width for row in cells):
        return "<pre>" + html.escape("\n".join(r.strip() for r in rows)) + "</pre>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cells[0])
    body = []
    for row in cells[1:]:
        tds = "".join(
            f'<td class="{"num" if _NUMBER.match(c) else ""}">{html.escape(c)}</td>' for c in row
        )
        body.append(f"<tr>{tds}</tr>")
    return (
        '<div class="table-wrap"><table class="sortable"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _paragraph(text: str, level: int) -> str:
    style = f' style="margin-left:{min(level, 4) * 1.1:.1f}em"' if level else ""
    marker = MARKER_LINE.match(text)
    if marker:
        status = MARKERS[marker.group(1)]
        kind, word = STATUS_WORDS[status]
        # Inline, the sentence says what the marker found; the icon keeps its word for readers
        # that cannot see it.
        icon = f'<span class="mark {kind}" role="img" aria-label="{word}">{status.marker}</span>'
        return f'<p class="status-line"{style}>{icon} {html.escape(marker.group(2))}</p>'
    if text.startswith("- "):
        return f'<p class="item"{style}>{html.escape(text[2:])}</p>'
    if text.startswith("Assumptions:"):
        body = html.escape(text[len("Assumptions:") :].strip())
        return f'<p class="assumptions"{style}><strong>Assumptions</strong> {body}</p>'
    return f"<p{style}>{html.escape(text)}</p>"


def _starts_own(lines: list[str], index: int, hang: int | None) -> bool:
    """Whether the line at ``index``, at the paragraph's continuation indent ``hang``, is
    rather an item of its own: the line after it hangs deeper, as its own continuation."""
    if hang is None or indent_width(lines[index]) != hang or index + 1 >= len(lines):
        return False
    after = lines[index + 1]
    return bool(after.strip()) and indent_width(after) > hang and not _COLUMNS.search(after.strip())


def text_html(text: str) -> str:
    """The report's text as HTML (see the module), without the rules' provenance, which only
    the full text report keeps (owner, 2026-10-06)."""
    lines = without_provenance(text).splitlines()
    out: list[str] = []
    i = 0
    base = min((indent_width(line) for line in lines if line.strip()), default=0)
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _COLUMNS.search(line.strip()):
            block = _column_block(lines, i)
            if len(block) >= 2:
                out.append(_table(block))
                i += len(block)
                continue
        parts, i = _joined(lines, i)
        out.append(_paragraph(" ".join(parts), (indent_width(line) - base) // 2))
        i += 1
    return "\n".join(out)


def _column_block(lines: list[str], start: int) -> list[str]:
    """The line at ``start`` and the non-blank lines after it whose columns are aligned."""
    end = start + 1
    while end < len(lines) and lines[end].strip() and _COLUMNS.search(lines[end].strip()):
        end += 1
    return lines[start:end]


def _joined(lines: list[str], i: int) -> tuple[list[str], int]:
    """The paragraph starting at line ``i``: its stripped lines (a wrapped sentence's
    continuation lines with it) and the index of its last line."""
    first = indent_width(lines[i])
    parts = [lines[i].strip()]
    hang = None
    while _continues(lines, i + 1, first, hang):
        i += 1
        hang = indent_width(lines[i]) if hang is None else hang
        parts.append(lines[i].strip())
    return parts, i


def _continues(lines: list[str], index: int, first: int, hang: int | None) -> bool:
    """Whether the line at ``index`` continues the paragraph whose first line is indented by
    ``first``: deeper, and not a status line, an item, a table row or an item of its own."""
    if index >= len(lines):
        return False
    stripped = lines[index].strip()
    return (
        bool(stripped)
        and indent_width(lines[index]) > first
        and not MARKER_LINE.match(stripped)
        and not stripped.startswith("- ")
        and not _COLUMNS.search(stripped)
        and not _starts_own(lines, index, hang)
    )


def first_sentence(text: str) -> str:
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


__all__ = [
    "STATUS_WORDS",
    "Terms",
    "adjusted_p",
    "chip",
    "first_sentence",
    "in_units",
    "text_html",
]
