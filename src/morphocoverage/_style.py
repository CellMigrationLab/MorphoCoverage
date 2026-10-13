"""Shared style of the text output (private): status markers, labels, tables, wrapping.

The project owner's decisions for readable reports (2026-09-28): every summary
reads like a colleague's note, opens each answer with a status marker, names
rules and groups in plain words, puts per-group numbers in aligned tables and
states a caveat that repeats across rules or groups once, at the end of its
section. Every text output formats through these helpers, so the style is the
same everywhere; the numbers inside go through ``_wording``.

- ``OK``, ``CAUTION``, ``PROBLEM``, ``NOT_ASSESSED``: the markers, and
  ``Status`` with ``worst``. ✅ answered, and good for the user's goal; ⚠️
  answered, but look at this (met in some groups only, a caution applies, an
  estimate is uncertain); ❌ a problem that changes how the answers are read
  or acted on; ➖ not assessed.
- ``headline``: ``"⚠️ 2. Did I image enough? — ..."``, a marker, a title and
  its one-line answer.
- ``group_label`` (``"MOCK · 1A · B6"``) and ``group_columns`` (the columns,
  named once per section).
- ``rule_label``: a stopping rule in words (``"mean-precision rule, per nucleus, 5%"``),
  from a strategy or from a row of a result table.
- ``table``: rows as aligned plain text; ``wrap``: text wrapped to 100
  characters under an indent.
- Caveats (cautions about these data and standing assumptions) are printed by
  ``_caveats`` (``caveat_lines``), with these helpers.
- ``check_detail``: the ``detail`` argument of every ``summary``
  (``"short"`` or ``"full"``).
"""

from __future__ import annotations

import math
import re
import textwrap
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

import pandas as pd

from ._rule_names import _named
from ._wording import column_word, number, percent, plural
from .metadata import MORPHOBRICKS_GROUP_COLUMNS, REPEAT_COLUMN

OK = "✅"
CAUTION = "⚠️"
PROBLEM = "❌"
NOT_ASSESSED = "➖"

DETAILS = ("short", "full")
WIDTH = 100
DASH = "—"
SEPARATOR = " · "
# CellPatcher's value for a folder level an experiment did not have.
SINGLE_LEVEL = "single"
_KEPT = "\u00a0·\u00a0"  # the separator while wrapping: textwrap never breaks a no-break space


class Status(Enum):
    """The status of one answer; its ``marker`` opens the answer's line."""

    NOT_ASSESSED = NOT_ASSESSED
    OK = OK
    CAUTION = CAUTION
    PROBLEM = PROBLEM

    @property
    def marker(self) -> str:
        """The emoji marker: ✅, ⚠️, ❌ or ➖."""
        return self.value

    @property
    def rank(self) -> int:
        """How much the status asks of the reader: not assessed 0, fine 1, look 2, problem 3."""
        return _RANKS[self]


_RANKS = {Status.NOT_ASSESSED: 0, Status.OK: 1, Status.CAUTION: 2, Status.PROBLEM: 3}
# Each marker's status, to read an answer's line back.
MARKERS = {status.marker: status for status in Status}
# A line that opens with a marker, after any indent: the marker, then the words.
MARKER_LINE = re.compile(rf"^\s*({'|'.join(map(re.escape, MARKERS))})\s+(.*)$")


def worst(statuses: Iterable[Status]) -> Status:
    """The status that asks most of the reader; ``NOT_ASSESSED`` when there is none.

    A part that was not assessed does not lower an assessed one: ✅ and ➖ give
    ✅ (the headline says what was not assessed), ⚠️ and ✅ give ⚠️.
    """
    found = [Status(status) for status in statuses]
    return max(found, key=lambda status: status.rank) if found else Status.NOT_ASSESSED


def check_detail(detail: str) -> str:
    """``detail`` if it is ``"short"`` or ``"full"``; a ``ValueError`` naming both otherwise."""
    if detail not in DETAILS:
        raise ValueError(f"detail must be one of {DETAILS}, not {detail!r}")
    return detail


def headline(status: Status, title: str, text: str | None = None) -> str:
    """``"✅ title"`` or ``"✅ title — text"``: an answer's line, its marker first."""
    line = f"{Status(status).marker} {title}"
    return f"{line} {DASH} {text}" if text else line


def _value(value: Any) -> str:
    """A group value as text; a missing one as ``(none)``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(none)"
    return str(value)


def group_label(group: Mapping[str, Any]) -> str:
    """``"MOCK · repeat 1A · B6"``: the group's values, the biological repeat named as one;
    a level the folders did not have (``"single"``, CellPatcher's word for it) is left out.
    ``"all fields (pooled)"`` for no group."""
    parts = [
        f"repeat {_value(value)}" if column == REPEAT_COLUMN else _value(value)
        for column, value in group.items()
        if not (value == SINGLE_LEVEL and column in MORPHOBRICKS_GROUP_COLUMNS)
    ]
    return SEPARATOR.join(parts) or "all fields (pooled)"


def group_columns(columns: Sequence[str]) -> str:
    """``"condition · biological repeat · technical repeat"``: the columns a group label
    stands for, as words."""
    return SEPARATOR.join(column_word(c) for c in columns) or "all fields pooled"


def among(k: int, n: int, word: str = "group") -> str:
    """``"all 12 groups"``, ``"7 of 12 groups"``, ``"none of 12 groups"``; ``"both groups"``
    for 2 of 2; ``"the group"`` or ``"no group"`` when there is one."""
    if n == 1:
        return f"the {word}" if k else f"no {word}"
    words = plural(word)
    if k == n:
        return f"both {words}" if n == 2 else f"all {n} {words}"
    return f"none of {n} {words}" if k == 0 else f"{k} of {n} {words}"


def span(values: Iterable[float], kind: str = "number") -> str:
    """The lowest and the highest of ``values``: ``8`` or ``3 to 5``; with
    ``kind="percent"``, ``4%`` or ``0 to 20%``, the unit said once (through ``_wording``, so
    the guard on inline percentages holds)."""
    found = sorted(float(value) for value in values)
    if not found:
        raise ValueError("span needs at least one value")
    write = percent if kind == "percent" else number
    low, high = write(found[0]), write(found[-1])
    if kind == "percent" and low != high and low.endswith("%") and high.endswith("%"):
        low = low[:-1]
    return low_to_high(low, high)


def low_to_high(low: Any, high: Any) -> str:
    """``3 to 5``, the ends written as given, or one of them alone when they read the same: a
    range whose ends are equal is one number (owner, 2026-10-12)."""
    low, high = str(low), str(high)
    return low if low == high else f"{low} to {high}"


# --- Rule names ---------------------------------------------------------------

_THRESHOLD = re.compile(r"([\w:.]+) (<=|>=) (-?[0-9.]+(?:e-?[0-9]+)?)")


def _policy_thresholds(policies: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, float]]:
    """``{quantity: (operator, value)}`` from the parameters of decision policies."""
    thresholds = {}
    for settings in policies:
        if "quantity" in settings and "threshold" in settings:
            thresholds[settings["quantity"]] = ("", float(settings["threshold"]))
        elif "effort" in settings and "target" in settings:
            thresholds[settings["effort"]] = (">=", float(settings["target"]))
    return thresholds


def _is_row(rule: Any) -> bool:
    """Whether ``rule`` is a row of a result table (a mapping or a pandas Series), not a
    strategy (a Series answers ``row.policies`` like a strategy's attribute)."""
    return isinstance(rule, Mapping | pd.Series)


def _rule_parts(rule: Any) -> tuple[str, dict[str, tuple[str, float]], dict[str, Any]]:
    """The rule's name, its thresholds ``{quantity: (operator, value)}`` and its parameters.

    From a strategy; or from a row: its ``criterion`` (or ``strategy``) name and
    its ``settings`` (or ``policies``) text, or its ``policies`` as a
    configuration's list of policies (``{"parameters": {...}}``), with the
    ``parameters`` it has.
    """
    if not _is_row(rule):
        thresholds = _policy_thresholds(policy.parameters for policy in rule.policies)
        return str(rule.name), thresholds, dict(rule.parameters)
    row = dict(rule)
    name = row.get("criterion") or row.get("strategy") or ""
    settings = row.get("settings") or row.get("policies") or ""
    parameters = row.get("parameters")
    parameters = dict(parameters) if isinstance(parameters, Mapping) else {}
    if isinstance(settings, list | tuple):
        policies = [p.get("parameters", {}) for p in settings if isinstance(p, Mapping)]
        return str(name), _policy_thresholds(policies), parameters
    thresholds = {
        quantity: (operator, float(value))
        for quantity, operator, value in _THRESHOLD.findall(str(settings))
    }
    return str(name), thresholds, parameters


def rule_label(rule: Any, unit: str | None = None, *, detail: str = "short") -> str:
    """A stopping rule in words: ``"mean-precision rule, per nucleus, 5%"``, ``"fixed 8 fields"``.

    ``rule`` is a strategy, or a row of a result table: its ``rule`` entry when
    it has one, else its ``criterion`` (or ``strategy``) name and its
    ``settings`` (or ``policies``) text, such as ``averaging_cv <= 0.05``.
    ``unit`` is the observation unit (``nucleus``; ``object`` when ``None``).
    Every default rule family has a name (the fixed field, object and area
    counts, the mean-precision rule per object and per field with the point or upper-95
    SD and a pilot, the mean-precision plan, the JCS spread and plateau, the FOV
    bootstrap, the support-novelty quantities, the rare-state rules and the
    incidence coverage); any other rule keeps the strategy's own name.
    ``detail="full"`` adds the strategy's name, settings and config id.
    """
    check_detail(detail)
    unit = unit or "object"
    name, thresholds, parameters = _rule_parts(rule)
    given = dict(rule).get("rule") if _is_row(rule) else None
    label = given if isinstance(given, str) and given else None
    label = label or _named(name, thresholds, parameters, unit) or name
    if detail == "short":
        return label
    if not _is_row(rule):
        return f"{label} ({name}, config {rule.config_id})"
    row = dict(rule)
    settings = row.get("settings") or row.get("policies")
    settings = settings if isinstance(settings, str) else None
    config = row.get("config_id")
    extra = "; ".join(
        part
        for part in (f"{name}: {settings}" if settings else name, config and f"config {config}")
        if part
    )
    return f"{label} ({extra})" if extra != label else label


# A rule's provenance as the full report prints it: ``[fixed_fov (target_fovs=8)]`` in the
# declaration, ``(scalar_precision_fov_upper95: averaging_cv <= 0.05; config 8d1f...)`` beside a
# rule (``rule_label(detail="full")``).
_PROVENANCE = re.compile(
    r" \[[a-z][a-z0-9]*_[a-z0-9_]*(?:\s+\(.*?\))?\]"
    r"| \([a-z][a-z0-9]*_[a-z0-9_]*(?::[^()]*?)?[;,]\s*config [0-9a-f]{16}\)"
    r"| \([a-z][a-z0-9]*_[a-z0-9_]*:[^()]*?(?:<=|>=|==|<|>)[^()]*?\)",
    re.DOTALL,
)


def without_provenance(text: str) -> str:
    """``text`` without the rules' provenance brackets, which the full text report keeps and
    the short report and the page never show (owner, 2026-10-06)."""
    return _PROVENANCE.sub("", text)


# --- Layout -------------------------------------------------------------------


def indent_width(line: str) -> int:
    """The spaces ``line`` starts with."""
    return len(line) - len(line.lstrip(" "))


def indented(text: str) -> str:
    """``text`` with every non-empty line indented by two spaces."""
    return "\n".join(f"  {line}" if line else line for line in text.splitlines())


def width(text: str) -> int:
    """The columns ``text`` takes in a terminal: wide characters and emoji count two."""
    total = 0
    for position, character in enumerate(text):
        if character == "️" or unicodedata.combining(character):
            continue  # a variation selector or a combining mark takes no column
        wide = unicodedata.east_asian_width(character) in ("W", "F")
        emoji = text[position + 1 : position + 2] == "️"
        total += 2 if wide or emoji else 1
    return total


def _cell(value: Any) -> str:
    """A table cell as text; a missing value as a dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return str(value)


def table(
    rows: Iterable[Sequence[Any] | Mapping[str, Any]],
    columns: Sequence[str],
    *,
    indent: int | str = 2,
    align: str | None = None,
) -> str:
    """Rows as aligned plain text under a header of ``columns``, two spaces between columns.

    A row is a sequence of cells or a mapping from column name to cell;
    missing values print as a dash. ``align`` has one letter per column,
    ``l`` (the default) or ``r``; wide characters (emoji) count two columns.
    """
    prefix = " " * indent if isinstance(indent, int) else indent
    cells = [[_cell(value) for value in columns]]
    for row in rows:
        values = [row.get(column) for column in columns] if isinstance(row, Mapping) else row
        if len(values) != len(columns):
            raise ValueError(f"a row has {len(values)} cells for {len(columns)} columns")
        cells.append([_cell(value) for value in values])
    sides = (align or "l" * len(columns)).ljust(len(columns), "l")
    widths = [max(width(line[i]) for line in cells) for i in range(len(columns))]
    lines = []
    for line in cells:
        padded = []
        for i, text in enumerate(line):
            gap = " " * (widths[i] - width(text))
            padded.append(gap + text if sides[i] == "r" else text + gap)
        lines.append((prefix + "  ".join(padded)).rstrip())
    return "\n".join(lines)


def wrap(text: str, indent: int | str = 0, *, hang: int | str | None = None) -> str:
    """``text`` wrapped to 100 characters, each line under ``indent``, continuations under
    ``hang`` (by default as deep as ``indent``); words are never broken, blank lines kept."""
    first = " " * indent if isinstance(indent, int) else indent
    if hang is None:
        later = " " * width(first)
    else:
        later = " " * hang if isinstance(hang, int) else hang
    lines = []
    for line in text.splitlines() or [""]:
        if not line.strip():
            lines.append("")
            continue
        # An emoji in the indent takes two columns but counts one character; a group label
        # ("MOCK · 1A · B6") is never broken, its separators held by no-break spaces. A marker
        # opening the text is moved into the indent, so its two columns are counted too (no
        # line one column too wide, audit 10).
        start = first
        marker = next(
            (m for m in (OK, CAUTION, PROBLEM, NOT_ASSESSED) if line.startswith(f"{m} ")), ""
        )
        if marker:
            start, line = f"{first}{marker} ", line[len(marker) + 1 :]
        filled = textwrap.fill(
            line.replace(SEPARATOR, _KEPT),
            WIDTH - (width(start) - len(start)),
            initial_indent=start,
            subsequent_indent=later,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.append(filled.replace(_KEPT, SEPARATOR))
    return "\n".join(lines)


# A number in running text (signed, 2.3e-05 as one): not the tail of a word or of "68-88%".
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def merge_numbers(texts: Sequence[str]) -> str:
    """One sentence for sentences that differ only in their numbers, each number as a range.

    ``["rho -0.43, p = 0.023", "rho -0.55, p = 0.0023"]`` gives
    ``"rho -0.55 to -0.43, p = 0.0023 to 0.023"``: a number the sentences share
    stays as it is. Sentences with different words raise ``ValueError``
    (group them first, for example by ``number_pattern``).
    """
    texts = list(dict.fromkeys(texts))
    if not texts:
        raise ValueError("no sentence to merge")
    patterns = {number_pattern(text) for text in texts}
    if len(patterns) > 1:
        raise ValueError("the sentences differ in more than their numbers")
    words = _NUMBER.split(texts[0])
    columns = list(zip(*(_NUMBER.findall(text) for text in texts), strict=True))
    merged = [words[0]]
    for values, rest in zip(columns, words[1:], strict=True):
        if len(set(values)) == 1:
            merged.append(values[0])
        else:
            low = min(values, key=float)
            high = max(values, key=float)
            merged.append(f"{low} to {high}")
        merged.append(rest)
    return "".join(merged)


def varying_numbers(texts: Sequence[str]) -> list[str]:
    """For sentences alike but for their numbers, each one's numbers that differ, named by the
    words before them: ``"Spearman rho -0.43, p = 0.023"`` (empty when none differs).

    The per-group side of ``merge_numbers``: the merged sentence is stated once,
    and each group keeps its own numbers.
    """
    if len({number_pattern(text) for text in texts}) > 1:
        raise ValueError("the sentences differ in more than their numbers")
    if not texts:
        return []
    words = _NUMBER.split(texts[0])
    numbers = [_NUMBER.findall(text) for text in texts]
    varying = [i for i, column in enumerate(zip(*numbers, strict=True)) if len(set(column)) > 1]
    labels = [" ".join(re.findall(r"[\w=<>≤≥'-]+", words[i])[-2:]) for i in varying]
    return [
        ", ".join(f"{label} {values[i]}".strip() for label, i in zip(labels, varying, strict=True))
        for values in numbers
    ]


_TABLE_GAP = re.compile(r"\S {2,}\S")  # a table row: columns apart by two spaces or more


def number_pattern(text: str) -> str:
    """``text`` with every number replaced by ``#``: sentences alike but for their numbers."""
    return _NUMBER.sub("#", text)


__all__ = [
    "CAUTION",
    "DASH",
    "DETAILS",
    "MARKERS",
    "MARKER_LINE",
    "NOT_ASSESSED",
    "OK",
    "PROBLEM",
    "SEPARATOR",
    "Status",
    "among",
    "check_detail",
    "group_columns",
    "group_label",
    "headline",
    "indent_width",
    "indented",
    "low_to_high",
    "merge_numbers",
    "number_pattern",
    "rule_label",
    "span",
    "table",
    "varying_numbers",
    "width",
    "worst",
    "wrap",
]
