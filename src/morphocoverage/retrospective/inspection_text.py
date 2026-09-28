"""The words of "Which fields should I look at?" (question 6), short and full.

``inspection_summary(result, detail)`` writes an ``InspectionResult``:

- **short** (the default): one line with how many fields to reopen, in how
  many groups and for which reasons, a table of the fields (group, field,
  why; the first ``SHORT_FIELDS``, then how many more), the reminder that
  none of them is called an error, and each reason that could not be judged
  somewhere once, with the groups it concerns, as a ⚠️ line;
- **full**: the analysis-group columns named once, then per group every field
  with its image file and stage position and every reason with its numbers,
  and every group where a reason could not be judged, each with its own
  reason.

``inspection_parts(result, indent)`` gives the short form as ``(lines,
cautions)`` so the report can print the caveats of question 6 once, at the
end of its section.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._caveats import caveat_lines
from .._style import check_detail, group_columns, group_label, table, width, wrap
from .._wording import count, listing
from .._wording import fields as fields_text

# Fields listed in the short form; the full form lists every one.
SHORT_FIELDS = 8
_NOT_AN_ERROR = (
    "None of these is called an error: look at the images to see whether it is biology, "
    "technique or simply a large field."
)
# Each reason in a few words, for the short form's first line.
_REASON_WORDS = {
    "new state": "a state no other field shows",
    "influence": "sways a declared summary",
    "count": "an unusual count",
    "neighbours": "unlike its stage neighbours",
}
_SKIPPED = (
    (
        "skipped_summaries",
        "The influence on the {summary} was not checked in {groups}: it {reason}",
    ),
    ("skipped_neighbours", "Stage neighbours were not compared in {groups}: {reason}"),
    ("skipped_novelty", "New states were not looked for in {groups}: {reason}"),
)


def _keys(result: Any) -> tuple[list[str], list[str], list[str]]:
    """The group columns, the field's own key columns, and the traceability columns."""
    groups = list(result.group_cols)
    fov_key = [c for c in result.settings.get("fov_key", []) if c not in groups]
    trace = [c for c in result.settings.get("trace", []) if c not in fov_key]
    return groups, fov_key, trace


def _fields(result: Any) -> list[tuple[dict[str, Any], str, pd.DataFrame]]:
    """``(group, field name, rows)`` per listed field, in the order of the table."""
    groups, fov_key, _ = _keys(result)
    frame = result.fovs
    found = []
    blocks = frame.groupby(groups, sort=False, dropna=False) if groups else [((), frame)]
    for key, rows in blocks:
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(groups, key, strict=True))
        by_field = rows.groupby(fov_key, sort=False, dropna=False) if fov_key else [((), rows)]
        for _, own in by_field:
            first = own.iloc[0]
            name = " / ".join(str(first[c]) for c in fov_key) or "the field"
            found.append((group, name, own))
    return found


def _skipped(result: Any, *, full: bool) -> list[str]:
    """Where a reason could not be judged: per group (full), or once per reason with the
    groups it concerns (short)."""
    sentences = []
    for key, template in _SKIPPED:
        entries = result.settings.get(key, [])
        if full:
            for entry in entries:
                label = entry.get("label", entry["group"])
                sentences.append(template.format(**{**entry, "groups": label}))
            continue
        merged: dict[tuple[str, str], list[str]] = {}
        for entry in entries:
            what = (entry.get("summary", ""), entry["reason"])
            merged.setdefault(what, []).append(entry.get("label", entry["group"]))
        for (summary, reason), labels in merged.items():
            groups = (
                labels[0] if len(labels) == 1 else f"{len(labels)} groups ({listing(labels, 3)})"
            )
            sentences.append(template.format(summary=summary, reason=reason, groups=groups))
    return sentences


def _empty(result: Any) -> str:
    """The answer when no field is listed: nothing judged, or nothing stands out."""
    checked = result.settings.get("checked")
    if checked is not None and not checked:
        # Nothing was judged: saying that no field stands out would read as a clean bill.
        return (
            "No field was checked: no reason was turned on (summaries, novelty, "
            "count_ratio or stage_cols), or none could be judged in any group."
        )
    which = f" ({', '.join(checked)} checked)" if checked else ""
    return f"No field stands out under the declared thresholds{which}."


def reason_counts(result: Any) -> list[tuple[str, int]]:
    """``(reason, fields)``: how many listed fields each reason names, in the order of
    ``inspection.REASONS``."""
    groups, fov_key, _ = _keys(result)
    frame = result.fovs
    if frame.empty:
        return []
    named = frame.drop_duplicates([*groups, *fov_key, "reason"])
    counts = named["reason"].value_counts()
    order = [r for r in _REASON_WORDS if r in counts.index]
    order += [r for r in counts.index if r not in order]
    return [(reason, int(counts[reason])) for reason in order]


def tally(result: Any) -> tuple[int, int]:
    """How many fields are listed, and in how many groups."""
    listed = _fields(result) if not result.fovs.empty else []
    return len(listed), len({tuple(map(str, group.values())) for group, _, _ in listed})


def inspection_parts(result: Any, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form of question 6 under ``indent``, and its caveats (see the module)."""
    cautions = _skipped(result, full=False)
    if result.fovs.empty:
        return [wrap(_empty(result), indent, hang=indent + 2)], cautions
    groups, _, _ = _keys(result)
    listed = _fields(result)
    _, group_count = tally(result)
    reasons = "; ".join(
        f"{_REASON_WORDS.get(reason, reason)} ({count(n, 'field')})"
        for reason, n in reason_counts(result)
    )
    where = f" ({group_columns(groups)})" if groups else ""
    reopen = f"{count(len(listed), 'field')} to reopen in {count(group_count, 'group')}"
    head = f"{reopen}{where}: {reasons}."
    lines = [wrap(head, indent, hang=indent + 2)]
    rows = [
        [group_label(group), name, "; ".join(own["detail"])]
        for group, name, own in listed[:SHORT_FIELDS]
    ]
    columns = ["group", "field", "why"] if groups else ["field", "why"]
    rows = rows if groups else [row[1:] for row in rows]
    shown = table(rows, columns, indent=indent + 2)
    if max(width(line) for line in shown.splitlines()) <= 100:
        lines.append(shown)
    else:
        # Long reasons (an influence, a neighbour) do not fit a table row: one line each.
        for row in rows:
            lines.append(wrap(f"{' · '.join(row[:-1])}: {row[-1]}.", indent + 2, hang=indent + 4))
    if len(listed) > SHORT_FIELDS:
        more = len(listed) - SHORT_FIELDS
        lines.append(" " * (indent + 2) + f"and {fields_text(more)} more (detail='full').")
    lines.append(wrap(_NOT_AN_ERROR, indent, hang=indent + 2))
    return lines, cautions


def _full(result: Any) -> str:
    """Every field and reason with its numbers, per group, and every reason not judged."""
    groups, _, trace = _keys(result)
    lines = []
    if groups:
        lines.append(f"Analysis groups: {group_columns(groups)}.")
    if result.fovs.empty:
        lines.append(wrap(_empty(result), 0, hang=2))
    else:
        current = None
        for group, name, own in _fields(result):
            label = group_label(group) if groups else "All fields"
            if label != current:
                current = label
                lines.append(f"{label}: fields to reopen before trusting the result")
            first = own.iloc[0]
            extra = ", ".join(f"{c} {first[c]}" for c in trace)
            reasons = "; ".join(own["detail"])
            lines.append(wrap(f"{name}{f' ({extra})' if extra else ''}: {reasons}.", 2, hang=4))
        lines.append(wrap(_NOT_AN_ERROR, 0, hang=2))
    lines += caveat_lines(_skipped(result, full=True), indent=0, detail="full")
    return "\n".join(lines)


def inspection_summary(result: Any, detail: str = "short") -> str:
    """ "Fields to reopen": a table of the fields (short) or every reason per group (full)."""
    if check_detail(detail) == "full":
        return _full(result)
    lines, cautions = inspection_parts(result)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0)])


__all__ = ["SHORT_FIELDS", "inspection_parts", "inspection_summary", "reason_counts", "tally"]
