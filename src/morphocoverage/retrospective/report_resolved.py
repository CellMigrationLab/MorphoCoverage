"""Question 2 with a control: do the fields acquired resolve the difference from the control?

When a control is declared and other conditions are compared with it, "Did I
image enough?" first asks what the experiment was built for: whether, in each
repeat, the fields acquired pin down the difference from the control
(``sampling_unit.contrast_sampling``, computed with the condition contrast).
The precision of each group's own mean (the stopping rules) follows as a
secondary line.

- ``resolved_headline``: the "At a glance" words and status: ✅ when every
  difference is resolved; ⚠️ when any is not (owner, 2026-10-01): some would be
  resolved with more fields (the advice says how many a side), are too small to
  settle with a practical number of fields, or have too few fields for an interval.
- ``resolved_parts``: the short section: one sentence, the contrasts not
  resolved in a table (all contrasts when there are few), and the notes.
- ``resolved_full``: every contrast with its interval.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._caveats import assumption
from .._style import SINGLE_LEVEL, Status, table, wrap
from .._wording import count, percent
from ..metadata import REPEAT_COLUMN
from ..sampling_unit.contrast_sampling import LEVEL, PRACTICAL
from ..sampling_unit.repeat_plan_text import field_words
from ..schema import observation_words

# Contrasts listed in full in the short form; beyond, only those not resolved.
SHORT_ROWS = 8


def _table(contrast: Any) -> pd.DataFrame:
    """Every comparison with the control of its repeat (fields on both sides), those without
    an interval (too few fields) included, so they stay in the count; empty when no
    comparison has an interval."""
    table_ = getattr(contrast, "table", pd.DataFrame())
    if table_.empty or "d_low" not in table_ or table_["d_low"].isna().all():
        return pd.DataFrame()
    if {"fields", "control_fields"} <= set(table_):
        return table_[(table_["fields"] > 0) & (table_["control_fields"] > 0)]
    return table_


def has_resolution(contrast: Any) -> bool:
    """Whether question 2 can answer from the contrast (some interval was computed)."""
    return contrast is not None and not _table(contrast).empty


def _needing(rows: pd.DataFrame) -> pd.DataFrame:
    """The comparisons with an interval that more fields would resolve."""
    return rows[~rows["resolved"] & ~rows["too_small"] & rows["d_low"].notna()]


def _unassessed(rows: pd.DataFrame) -> int:
    """How many comparisons have no interval (too few fields to resample)."""
    return int(rows["d_low"].isna().sum())


def resolved_headline(contrast: Any) -> tuple[Status, str]:
    """The headline and status of question 2 from the contrast (see the module)."""
    rows = _table(contrast)
    resolved = int(rows["resolved"].sum())
    needing = _needing(rows)
    small = int(rows["too_small"].sum())
    _, fields_word = field_words(contrast)
    text = f"to compare with the control, within single repeats: shown in {resolved} of " + count(
        len(rows), "comparison"
    )
    if len(needing):
        fields = needing["fields_to_resolve"].astype(float)
        low, high = int(fields.min()), int(fields.max())
        about = f"{low}" if low == high else f"{low}-{high}"
        text += f"; {len(needing)} would need about {about} {fields_word} a side"
    if small:
        text += f"; {small} too small to show with more {fields_word}"
    if _unassessed(rows):
        text += f"; {_unassessed(rows)} not assessed (too few {fields_word} for an interval)"
    # The marker says what was found: every comparison shown is enough, any other partly or
    # not yet (owner, 2026-10-01).
    return (Status.OK if resolved == len(rows) else Status.CAUTION), text


def _name(row: dict[str, Any]) -> str:
    name = f"{row['condition']} vs {row['control_condition']}"
    repeat = row.get(REPEAT_COLUMN)
    # CellPatcher's "single" names a folder level the data did not have: no repeat to name.
    if repeat is None or pd.isna(repeat) or str(repeat) == SINGLE_LEVEL:
        return name
    return f"{name}, repeat {repeat}"


def _answer(row: dict[str, Any], fields: str = "fields") -> str:
    if pd.isna(row["d_low"]):
        return f"not assessed: too few {fields}"
    if row["resolved"]:
        return "yes (shown)"
    if row["too_small"]:
        return f"no: too small to show with more {fields}"
    return f"no: about {int(row['fields_to_resolve'])} {fields} a side"


def _interval(row: dict[str, Any]) -> str:
    if pd.isna(row["d_low"]):
        return "no interval" if pd.isna(row["d_objects"]) else f"{row['d_objects']:+.2f}"
    return f"{row['d_objects']:+.2f} ({row['d_low']:+.2f} to {row['d_high']:+.2f})"


def _rows(rows: pd.DataFrame, fields: str = "fields") -> list[list[str]]:
    return [
        [
            _name(row),
            f"{int(row['fields'])} vs {int(row['control_fields'])}",
            _interval(row),
            _answer(row, fields),
        ]
        for row in rows.to_dict("records")
    ]


def _columns(contrast: Any) -> list[str]:
    words = observation_words(contrast.settings)[1]
    _, fields = field_words(contrast)
    return [
        "comparison",
        f"{fields} (condition vs control)",
        # d is computed on the observations, whatever the first column counts (fields).
        f"d on {words} ({percent(LEVEL)} interval)",
        "enough in this repeat?",
    ]


def _notes(contrast: Any) -> list[str]:
    _, fields = field_words(contrast)
    return [
        assumption(
            f"Shown: the {percent(LEVEL)} interval of d, from resampling the technical repeats "
            f"and their fields as acquired, excludes 0. The {fields} a side it would take are "
            f"projected from these (the interval narrowing as 1/sqrt({fields})); beyond "
            f"{PRACTICAL} times the {fields} acquired a difference is called too small to show."
        )
    ]


def resolved_parts(
    contrast: Any, indent: int = 2, *, every: bool = False, where: bool = True
) -> tuple[list[str], list[str]]:
    """The short form: a sentence, a table, and the notes (see the module); ``every`` lists
    every comparison, and ``where=False`` leaves out that the full report has them all (the
    HTML page says so once, in its help)."""
    rows = _table(contrast)
    lines = [
        wrap(
            "Within each repeat, does the difference from the control show? "
            f"{contrast.settings['feature']}, each condition against the control of the same "
            "repeat:",
            indent,
            hang=indent + 2,
        )
    ]
    shown = rows if every or len(rows) <= SHORT_ROWS else rows[~rows["resolved"]]
    if len(shown):
        lines.append(
            table(_rows(shown, field_words(contrast)[1]), _columns(contrast), indent=indent + 2)
        )
    if len(shown) < len(rows):
        hidden = len(rows) - len(shown)
        full = " (every one in the full report)" if where else ""
        lines.append(
            wrap(
                f"The other {count(hidden, 'comparison')} show it{full}.",
                indent,
                hang=indent + 2,
            )
        )
    return lines, _notes(contrast)


def resolved_full(contrast: Any) -> str:
    """Every contrast with its interval and answer, then the notes."""
    rows = _table(contrast)
    head = (
        "Within each repeat, does the difference from the control show? "
        f"{contrast.settings['feature']}, each condition against the control of the same "
        f"repeat, {percent(LEVEL)} interval of d "
        "from the fields resampled as acquired:"
    )
    body = table(_rows(rows, field_words(contrast)[1]), _columns(contrast), indent=2)
    return "\n".join([wrap(head, 0, hang=2), body, *(f"  {n}" for n in _notes(contrast))])


def precision_words(tallies: list[dict[str, Any]], word: str) -> str | None:
    """The precision of each mean, as a secondary line: the first data rule's tally, the
    rule named (the workflows add "default" when it is theirs, ``_enough_features``)."""
    data = [t for t in tallies if t["data_rule"] and t["undefined"] < t["groups"]]
    if not data:
        return None
    first = data[0]
    return (
        f"each {word}'s own mean by the {first['label']}: met in {first['met']} of "
        f"{first['groups']}, a different goal from the comparison"
    )


__all__ = [
    "SHORT_ROWS",
    "has_resolution",
    "precision_words",
    "resolved_full",
    "resolved_headline",
    "resolved_parts",
]
