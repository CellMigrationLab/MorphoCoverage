"""Question 2 with a control: do the fields acquired resolve the difference from the control?

When a control is declared and other conditions are compared with it, "Did I
image enough?" first asks what the experiment was built for: whether, in each
repeat, the fields acquired pin down the difference from the control
(``sampling_unit.contrast_sampling``, computed with the condition contrast).
The precision of each group's own mean (the stopping rules) follows as a
secondary line.

- ``resolved_headline``: the "At a glance" words and status: ✅ when every
  difference is resolved or too small to settle with a practical number of
  fields; ⚠️ when some would be resolved with more fields (the advice says how
  many a side). Differences too small to settle are described, not flagged.
- ``resolved_parts``: the short section: one sentence, the contrasts not
  resolved in a table (all contrasts when there are few), and the notes.
- ``resolved_full``: every contrast with its interval.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._caveats import assumption
from .._style import Status, table, wrap
from .._wording import count, percent
from ..metadata import REPEAT_COLUMN
from ..sampling_unit.contrast_sampling import LEVEL, PRACTICAL
from ..schema import unit_plural

# Contrasts listed in full in the short form; beyond, only those not resolved.
SHORT_ROWS = 8


def _table(contrast: Any) -> pd.DataFrame:
    """The contrasts with a resolution answer (an interval was computed)."""
    table_ = getattr(contrast, "table", pd.DataFrame())
    if table_.empty or "d_low" not in table_:
        return pd.DataFrame()
    return table_[table_["d_low"].notna()]


def has_resolution(contrast: Any) -> bool:
    """Whether question 2 can answer from the contrast (some interval was computed)."""
    return contrast is not None and not _table(contrast).empty


def _needing(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[~rows["resolved"] & ~rows["too_small"]]


def resolved_headline(contrast: Any) -> tuple[Status, str]:
    """The headline and status of question 2 from the contrast (see the module)."""
    rows = _table(contrast)
    resolved = int(rows["resolved"].sum())
    needing = _needing(rows)
    small = int(rows["too_small"].sum())
    text = f"the difference from the control is resolved in {resolved} of " + count(
        len(rows), "contrast"
    )
    if len(needing):
        fields = needing["fields_to_resolve"].astype(float)
        low, high = int(fields.min()), int(fields.max())
        about = f"{low}" if low == high else f"{low}-{high}"
        text += f"; {len(needing)} would need about {about} fields a side"
    if small:
        text += f"; {small} too small to settle with more fields here"
    return (Status.CAUTION if len(needing) else Status.OK), text


def _name(row: dict[str, Any]) -> str:
    name = f"{row['condition']} vs {row['control_condition']}"
    repeat = row.get(REPEAT_COLUMN)
    return name if repeat is None or pd.isna(repeat) else f"{name} · {repeat}"


def _answer(row: dict[str, Any]) -> str:
    if row["resolved"]:
        return "resolved"
    if row["too_small"]:
        return "too small to settle here"
    return f"about {int(row['fields_to_resolve'])} fields a side"


def _rows(rows: pd.DataFrame) -> list[list[str]]:
    return [
        [
            _name(row),
            f"{int(row['fields'])} vs {int(row['control_fields'])}",
            f"{row['d_objects']:.2f} ({row['d_low']:.2f} to {row['d_high']:.2f})",
            _answer(row),
        ]
        for row in rows.to_dict("records")
    ]


def _columns(contrast: Any) -> list[str]:
    words = unit_plural(contrast.settings["observation_unit"])
    return ["contrast", "fields", f"d {words} ({percent(LEVEL)} interval)", "enough?"]


def _notes(contrast: Any) -> list[str]:
    words = unit_plural(contrast.settings["observation_unit"])
    return [
        assumption(
            f"Resolved: the {percent(LEVEL)} interval of d, from the fields resampled as they were "
            "acquired (technical repeats, then their fields, with replacement), excludes 0. "
            f"The fields a side to resolve a difference are projected from these fields (the "
            f"interval narrowing as 1/sqrt(fields)); beyond {PRACTICAL} times the fields "
            f"acquired a difference is called too small to settle here. This asks whether the "
            f"fields pin down the difference between these {words}; whether it holds across "
            "biological repeats is question 9."
        )
    ]


def resolved_parts(contrast: Any, indent: int = 2) -> tuple[list[str], list[str]]:
    """The short form: a sentence, a table, and the notes (see the module)."""
    rows = _table(contrast)
    lines = [
        wrap(
            f"Enough to resolve the difference from the control? {contrast.settings['feature']}, "
            "each condition against the control of the same repeat:",
            indent,
            hang=indent + 2,
        )
    ]
    shown = rows if len(rows) <= SHORT_ROWS else rows[~rows["resolved"]]
    if len(shown):
        lines.append(table(_rows(shown), _columns(contrast), indent=indent + 2))
    if len(shown) < len(rows):
        hidden = len(rows) - len(shown)
        lines.append(
            wrap(
                f"The other {count(hidden, 'contrast')} are resolved (every one in the full "
                "report).",
                indent,
                hang=indent + 2,
            )
        )
    return lines, _notes(contrast)


def resolved_full(contrast: Any) -> str:
    """Every contrast with its interval and answer, then the notes."""
    rows = _table(contrast)
    head = (
        f"Enough to resolve the difference from the control? {contrast.settings['feature']}, "
        f"each condition against the control of the same repeat, {percent(LEVEL)} interval of d "
        "from the fields resampled as acquired:"
    )
    body = table(_rows(rows), _columns(contrast), indent=2)
    return "\n".join([wrap(head, 0, hang=2), body, *(f"  {n}" for n in _notes(contrast))])


def precision_words(tallies: list[dict[str, Any]], word: str) -> str | None:
    """The precision of each mean, as a secondary line: the first data rule's tally."""
    data = [t for t in tallies if t["data_rule"]]
    if not data:
        return None
    first = data[0]
    return f"{first['label']}: met in {first['met']} of {count(first['groups'], word)}"


__all__ = [
    "SHORT_ROWS",
    "has_resolution",
    "precision_words",
    "resolved_full",
    "resolved_headline",
    "resolved_parts",
]
