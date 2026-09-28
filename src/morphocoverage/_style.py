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
- ``rule_label``: a stopping rule in words (``"JCB rule, per nucleus, 5%"``),
  from a strategy or from a row of a result table.
- ``table``: rows as aligned plain text; ``wrap``: text wrapped to 100
  characters under an indent; ``caution_lines``: caveats stated once each, as
  ⚠️ lines.
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

from ._wording import count, number, percent, plural, rate

OK = "✅"
CAUTION = "⚠️"
PROBLEM = "❌"
NOT_ASSESSED = "➖"

DETAILS = ("short", "full")
WIDTH = 100
DASH = "—"
SEPARATOR = " · "


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
    """``"MOCK · 1A · B6"``: the group's values; ``"all fields (pooled)"`` for no group."""
    return SEPARATOR.join(_value(value) for value in group.values()) or "all fields (pooled)"


def group_columns(columns: Sequence[str]) -> str:
    """``"condition · repeat · technical_repeat"``: the columns a group label stands for."""
    return SEPARATOR.join(columns) or "all fields pooled"


def among(k: int, n: int, word: str = "group") -> str:
    """``"all 12 groups"``, ``"7 of 12 groups"``, ``"none of 12 groups"``; ``"the group"``
    or ``"no group"`` when there is one."""
    if n == 1:
        return f"the {word}" if k else f"no {word}"
    words = plural(word)
    if k == n:
        return f"all {n} {words}"
    return f"none of {n} {words}" if k == 0 else f"{k} of {n} {words}"


def span(values: Iterable[float], kind: str = "number") -> str:
    """The lowest and the highest of ``values``: ``8`` or ``3-5``; with ``kind="percent"``,
    ``4%`` or ``0-20%`` (through ``_wording``, so the guard on inline percentages holds)."""
    found = sorted(float(value) for value in values)
    if not found:
        raise ValueError("span needs at least one value")
    write = percent if kind == "percent" else number
    low, high = write(found[0]), write(found[-1])
    if low == high:
        return low
    if kind == "percent" and low.endswith("%") and high.endswith("%"):
        return f"{low[:-1]}-{high}"
    return f"{low} to {high}" if found[0] < 0 else f"{low}-{high}"


# --- Rule names ---------------------------------------------------------------

_THRESHOLD = re.compile(r"([\w:.]+) (<=|>=) (-?[0-9.]+(?:e-?[0-9]+)?)")
_BOOTSTRAP = {
    "centroid": "mean",
    "quantiles": "quantiles",
    "covariance": "covariance",
    "eigenspectrum": "eigenspectrum",
    "subspace": "subspace",
}
# The support-novelty quantities (strategies.support_novelty.QUANTITIES), by name here so
# that this module imports no strategy; tests/test_style_foundation.py checks they agree.
_NOVELTY = {
    "lofo_novel_fov_upper": "nothing new ({})",
    "lofo_novel_fov_fraction": "new fields ({}, point estimate)",
    "lofo_novel_fraction": "new {units} ({})",
    "lofo_mean_fov_novel_fraction": "new {units} per field ({})",
}
# The incidence-coverage quantities (strategies.incidence_coverage.QUANTITIES): the
# strategy's name is its quantity.
_INCIDENCE = {
    "incidence_coverage": "state coverage ({})",
    "chao2_completeness": "Chao2 completeness ({})",
    "expected_new_states_next_fov": "expected new states ({})",
    "prob_new_state_next_fov": "new state in the next field ({})",
    "good_turing_new_state_estimate": "Good-Turing new state ({})",
    "lofo_new_state_fov_fraction": "fields with a new state ({})",
    "new_state_fov_upper": "no new labelled state ({})",
}
_PROBABILITY_KINDS = {*_NOVELTY, *(k for k in _INCIDENCE if k != "expected_new_states_next_fov")}


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


def _threshold(
    thresholds: Mapping[str, tuple[str, float]], parameters: Mapping[str, Any], *keys: str
) -> float | None:
    """The first of ``keys`` among the thresholds, then among the parameters."""
    for key in keys:
        if key in thresholds:
            return thresholds[key][1]
    for key in keys:
        value = parameters.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _bound(value: float | None, probability: bool, below: bool = True) -> str:
    """``≤5%`` (a probability), ``≥95%``, ``≤0.3`` (a number in the readout's units)."""
    if value is None:
        return "its threshold"
    sign = "≤" if below else "≥"
    return f"{sign}{rate(value) if probability else f'{value:g}'}"


def _per(unit_name: str, words: str) -> str:
    """``per nucleus`` for the object unit, ``per field`` for the FOV unit."""
    return "per field" if unit_name == "fov" else f"per {words}"


def _named(name: str, thresholds: Mapping, parameters: Mapping, unit: str) -> str | None:
    """The readable name of a known rule family, or ``None``."""
    units = plural(unit)
    if name == "fixed_fov":
        target = _threshold(thresholds, parameters, "n_fovs", "target_fovs")
        return None if target is None else f"fixed {count(target, 'field')}"
    if name in ("fixed_objects", "fixed_observations"):
        keys = ("n_objects", "target_objects", "n_observations", "target_observations")
        target = _threshold(thresholds, parameters, *keys)
        return None if target is None else f"fixed {count(target, unit, units)}"
    if name == "fixed_area":
        target = _threshold(thresholds, parameters, "imaged_area", "target_area")
        return None if target is None else f"fixed imaged area {target:g}"
    if name.startswith("scalar_precision_planned"):
        plan = parameters.get("plan") or {}
        needed = plan.get("required_objects") if isinstance(plan, Mapping) else None
        return f"JCB plan, {count(needed, unit, units)}" if needed else "JCB planned count"
    if name.startswith("scalar_precision_"):
        parts = name.split("_")[2:]
        if not parts or parts[0] not in ("object", "fov"):
            return None
        target = _threshold(
            thresholds, parameters, "averaging_cv", "pilot_projected_averaging_cv", "target_cv"
        )
        label = f"JCB rule, {_per(parts[0], unit)}, {'?' if target is None else rate(target)}"
        if "upper95" not in parts:
            label += ", point SD"
        if "pilot" in parts:
            pilot = parameters.get("pilot_fovs")
            label += f", pilot of {count(pilot, 'field')}" if pilot else ", from a pilot"
        return label
    if name.startswith("descriptor_spread_"):
        resampling = name.removeprefix("descriptor_spread_")
        if resampling not in ("object", "fov"):
            return None
        tolerance = _threshold(
            thresholds, parameters, "max_location_spread", "max_scale_spread", "tolerance"
        )
        return f"JCS spread, {_per(resampling, unit)}, {_bound(tolerance, False)}"
    if name == "descriptor_plateau":
        tolerance = _threshold(
            thresholds, parameters, "max_location_range", "max_scale_range", "tolerance"
        )
        window = parameters.get("window")
        over = f" over {count(window, 'field')}" if window else ""
        return f"JCS plateau, {_bound(tolerance, False)}{over}"
    if name.startswith("fov_bootstrap_"):
        summary = name.removeprefix("fov_bootstrap_")
        if summary not in _BOOTSTRAP:
            return None
        keys = [q for q in thresholds if q.startswith("fov_bootstrap_")]
        tolerance = _threshold(thresholds, parameters, *keys, "tolerance")
        measure = "angle" if summary == "subspace" else "SE"
        return f"field bootstrap of the {_BOOTSTRAP[summary]}, {measure} {_bound(tolerance, False)}"
    if name == "support_novelty":
        quantity = parameters.get("quantity") or next(
            (q for q in thresholds if q in _NOVELTY), None
        )
        if quantity not in _NOVELTY:
            return None
        bound = _bound(_threshold(thresholds, parameters, quantity, "tolerance"), True)
        return _NOVELTY[quantity].format(bound, units=units)
    if name == "rare_incidence":
        risk = _threshold(thresholds, parameters, "rare_incidence_risk", "max_risk")
        return f"rare-state risk {_bound(risk, True)}"
    if name == "fov_incidence_plan":
        return "rare-state field plan"
    if name in _INCIDENCE:
        higher = name in ("incidence_coverage", "chao2_completeness")
        target = _threshold(thresholds, parameters, name, "target")
        return _INCIDENCE[name].format(_bound(target, name in _PROBABILITY_KINDS, not higher))
    return None


def rule_label(rule: Any, unit: str | None = None, *, detail: str = "short") -> str:
    """A stopping rule in words: ``"JCB rule, per nucleus, 5%"``, ``"fixed 8 fields"``.

    ``rule`` is a strategy, or a row of a result table: its ``rule`` entry when
    it has one, else its ``criterion`` (or ``strategy``) name and its
    ``settings`` (or ``policies``) text, such as ``averaging_cv <= 0.05``.
    ``unit`` is the observation unit (``nucleus``; ``object`` when ``None``).
    Every default rule family has a name (the fixed field, object and area
    counts, the JCB rule per object and per field with the point or upper-95
    SD and a pilot, the JCB plan, the JCS spread and plateau, the FOV
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


# --- Layout -------------------------------------------------------------------


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
        # An emoji in the indent takes two columns but counts one character.
        lines.append(
            textwrap.fill(
                line,
                WIDTH - (width(first) - len(first)),
                initial_indent=first,
                subsequent_indent=later,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


# A number in running text: not the tail of a word or of a range such as "68-88%".
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


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


def number_pattern(text: str) -> str:
    """``text`` with every number replaced by ``#``: sentences alike but for their numbers."""
    return _NUMBER.sub("#", text)


def once(items: Iterable[str]) -> list[str]:
    """The items without repeats, in first-seen order (blank items dropped)."""
    return [item for item in dict.fromkeys(item.strip() for item in items) if item]


def caution_lines(cautions: Iterable[str], indent: int | str = 2) -> list[str]:
    """Each caveat once, as a wrapped ``⚠️`` line, its first letter capitalised."""
    prefix = " " * indent if isinstance(indent, int) else indent
    lines = []
    for text in once(cautions):
        sentence = text[0].upper() + text[1:]
        sentence = sentence if sentence.endswith((".", "?", "!", ")")) else f"{sentence}."
        lines.append(wrap(sentence, f"{prefix}{CAUTION} ", hang=prefix + "   "))
    return lines


__all__ = [
    "CAUTION",
    "DASH",
    "DETAILS",
    "NOT_ASSESSED",
    "OK",
    "PROBLEM",
    "SEPARATOR",
    "Status",
    "among",
    "caution_lines",
    "check_detail",
    "group_columns",
    "group_label",
    "headline",
    "merge_numbers",
    "number_pattern",
    "once",
    "rule_label",
    "span",
    "table",
    "varying_numbers",
    "width",
    "worst",
    "wrap",
]
