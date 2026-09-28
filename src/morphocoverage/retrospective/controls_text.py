"""The representation control in words (``control_summary``), split from ``controls``.

``ControlResult.summary(detail=...)`` prints through this module:

- **short** (``control_parts``): one lead-in, then one table: what was judged
  (the positive-control fields, reference fields with planted positive-control
  objects, the reference fields' false alarms), each after ``fovs_seen``
  reference fields and as ``inspect_fovs`` judges a field (against the other
  reference fields of its analysis group); the positive control by stratum in
  one line; each caveat once, as ⚠️ lines at the end;
- **full**: the same table with the counts behind every rate (recognized of
  judged), then one row per stratum (``pair_cols``) with its every rate, and
  each caveat once.

Groups and strata print by their values (``_style.group_label``), their
columns named once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._style import caution_lines, check_detail, group_columns, group_label, table, wrap
from .._wording import count
from ._text import percent, unit_plural

if TYPE_CHECKING:
    from .controls import ControlResult

# The rows of the table: the kind after ``fovs_seen`` fields and its leave-one-field-out twin.
_KINDS = (
    ("positive", "positive_lofo"),
    ("planted", "planted_lofo"),
    ("reference", "reference_lofo"),
)
_UNJUDGED = "some fields could not be judged (missing feature values or too few fields)"


def _selected(selector: Mapping[str, Any] | None) -> str:
    """``YAP`` for ``{"condition": ["YAP"]}``; empty when nothing was declared."""
    if not selector:
        return ""
    parts = []
    for value in selector.values():
        values = value if isinstance(value, list | tuple | set) else [value]
        parts.append("/".join(str(v) for v in values))
    return " · ".join(parts)


def _stratum(entry: Mapping[str, Any], pair_cols: tuple[str, ...]) -> str:
    """A stratum by its values (``2B``); ``all fields (pooled)`` without pairing."""
    values = {c: entry[c] for c in pair_cols if c in entry}
    return group_label(values) if values else "all fields (pooled)"


def named_stratum(entry: Mapping[str, Any], pair_cols: tuple[str, ...]) -> str:
    """``repeat 2B`` (one pairing column, named), else the stratum's values."""
    if len(pair_cols) == 1 and pair_cols[0] in entry:
        return f"{pair_cols[0]} {entry[pair_cols[0]]}"
    return _stratum(entry, pair_cols)


def _rate(row: Any, full: bool, what: str = "checks") -> str | None:
    """``100% of 1680 checks`` (``1680 of 1680 checks: 100%`` in full), or ``no check``."""
    if row is None:
        return None
    total = int(row["candidates"])
    if pd.isna(row["rate"]) or not total:
        return f"no {what[:-1]} could be judged"
    text = (
        f"{int(row['judged_new'])} of {total} {what}: {percent(row['rate'])}"
        if full
        else f"{percent(row['rate'])} of {total} {what}"
    )
    unjudged = int(row["not_judged"])
    return f"{text}; {unjudged} not judged" if unjudged else text


def _names(result: ControlResult, cells: str) -> dict[str, str]:
    control = result.settings.get("control") or {}
    positive, reference = _selected(control.get("positive")), _selected(control.get("reference"))
    planted = control.get("planted_objects")
    reference_fields = f"{reference} fields" if reference else "reference fields"
    return {
        "positive": f"{positive} fields (positive control)" if positive else "positive fields",
        "planted": f"{reference_fields} with {planted} {positive or 'positive'} {cells} planted",
        "reference": f"{reference_fields} (false alarms)",
    }


def _rows(result: ControlResult, full: bool) -> tuple[list[list[str | None]], bool]:
    """The table's rows: what was judged, after ``fovs_seen`` fields and as inspect_fovs
    judges; and whether any field could not be judged."""
    rates = result.pooled.set_index("candidate")
    names = _names(result, unit_plural(result.settings.get("observation_unit", "object")))
    rows, unjudged = [], False
    for kind, lofo in _KINDS:
        seen = rates.loc[kind] if kind in rates.index else None
        within = rates.loc[lofo] if lofo in rates.index else None
        if kind == "planted" and seen is None and within is None:
            continue
        after = _rate(seen, full)
        if kind == "reference" and (seen is None or not int(seen["candidates"])):
            why = (
                "no reference field could be judged"
                if result.status == "positive only" and seen is not None
                else "no reference control was declared"
            )
            after = f"not assessed ({why})"
        unjudged |= any(r is not None and int(r["not_judged"]) for r in (seen, within))
        own = _rate(within, full, "fields" if kind == "reference" else "checks")
        rows.append([names[kind], after, own])
    return rows, unjudged


def _cautions(result: ControlResult, cells: str, unjudged: bool) -> list[str]:
    rates = result.pooled.set_index("candidate")
    seen = result.settings["fovs_seen"]
    pair_cols = tuple(result.pair_cols)
    cautions = [_UNJUDGED] if unjudged else []
    for gap in result.settings.get("planted_skipped", []):
        cautions.append(
            f"planted {cells}: not measured in {named_stratum(gap, pair_cols)}: "
            f"{gap['reason'].replace('objects', cells)}"
        )
    for skipped in result.settings.get("skipped_strata", []):
        cautions.append(
            f"not judged after {seen} reference fields in {named_stratum(skipped, pair_cols)}: "
            f"{skipped['reason']}"
        )
    cautions.append(
        "this holds for this definition of new and these features, on these controls; "
        "a phenotype unlike the control may be seen better or worse: a whole-field "
        "phenotype says little about a small or sparse state in an ordinary field"
        + (" (the planted rates measure that)" if "planted" in rates.index else "")
    )
    return cautions


def _empty_line(result: ControlResult, cells: str) -> str | None:
    empty = int(result.settings.get("positive_empty", 0) or 0)
    if not empty:
        return None
    return (
        f"{count(empty, 'positive-control field')} without {cells} "
        f"{'was' if empty == 1 else 'were'} not judged: an empty field cannot show the "
        "phenotype, so it is not a miss."
    )


def _by_stratum(result: ControlResult) -> str | None:
    """``By repeat, positive-control fields recognized: 1A 100%, 2A 100%.``"""
    table_rows = result.table
    if len(table_rows["stratum"].unique()) < 2:
        return None
    pair_cols = tuple(result.pair_cols)
    by = table_rows[table_rows["candidate"] == "positive"]
    parts = []
    for row in by.to_dict("records"):
        judged = "not judged" if pd.isna(row["rate"]) else percent(row["rate"])
        parts.append(f"{_stratum(row, pair_cols)} {judged}")
    return f"By {group_columns(pair_cols)}, positive-control fields recognized: {', '.join(parts)}."


def _not_assessed(result: ControlResult) -> str:
    skipped = result.settings.get("skipped_strata", [])
    declared = result.settings.get("reason") != "none"
    pair_cols = tuple(result.pair_cols)
    why = (
        "a control was declared, but no stratum could be judged ("
        + "; ".join(f"{named_stratum(s, pair_cols)}: {s['reason']}" for s in skipped)
        + ")"
        if declared and skipped
        else "a control was declared, but no check could be made (missing values?)"
        if declared
        else "no control was declared"
    )
    return (
        f"Representation sensitivity: not assessed: {why}. A 'nothing new' answer says "
        "nothing about phenotypes these features and this definition of new cannot see."
    )


def _lead(result: ControlResult) -> str:
    control = result.settings.get("control") or {}
    positive, reference = _selected(control.get("positive")), _selected(control.get("reference"))
    against = f"{positive} against {reference}" if positive and reference else positive
    paired = f", within each {group_columns(result.pair_cols)}" if result.pair_cols else ""
    declared = f" ({against}{paired})" if against else ""
    return f"Can these features see the declared phenotype{declared}?"


def _columns(result: ControlResult) -> list[str]:
    seen = result.settings["fovs_seen"]
    return ["recognized as different", f"after {seen} reference fields", "as inspect_fovs judges"]


def _method_line(result: ControlResult, cells: str) -> str:
    groups = result.settings.get("lofo_group_cols")
    within = f"its analysis group ({group_columns(groups)})" if groups else "its stratum"
    return (
        f"After {result.settings['fovs_seen']} reference fields with {cells}: as a stopping "
        "rule judges the next field. As inspect_fovs judges: each field against the other "
        f"reference fields of {within}."
    )


def control_parts(result: ControlResult, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form (see the module): ``(lines, cautions)``, lines under ``indent``."""
    cells = unit_plural(result.settings.get("observation_unit", "object"))
    if result.status == "not assessed":
        return [wrap(_not_assessed(result), indent, hang=indent + 2)], []
    rows, unjudged = _rows(result, full=False)
    lines = [wrap(_lead(result), indent, hang=indent + 2)]
    lines.append(table(rows, _columns(result), indent=indent + 2))
    lines.append(wrap(_method_line(result, cells), indent, hang=indent + 2))
    for line in (_by_stratum(result), _empty_line(result, cells)):
        if line:
            lines.append(wrap(line, indent, hang=indent + 2))
    return lines, _cautions(result, cells, unjudged)


def _strata_table(result: ControlResult, indent: int = 0) -> str | None:
    """Two rows per stratum (after ``fovs_seen`` fields, as inspect_fovs judges): every rate."""
    frame = result.table
    if frame.empty:
        return None
    pair_cols = tuple(result.pair_cols)
    present = set(frame["candidate"])
    kinds = [
        k for k in ("positive", "planted", "reference") if k in present or f"{k}_lofo" in present
    ]
    labels = {"positive": "positive", "planted": "planted", "reference": "false alarms"}
    columns = [group_columns(pair_cols) or "stratum", "judged"] + [labels[k] for k in kinds]
    seen = f"after {result.settings['fovs_seen']} fields"
    cells = []
    for stratum, own in frame.groupby("stratum", sort=False):
        by = own.set_index("candidate")
        name = _stratum(own.iloc[0], pair_cols) if pair_cols else stratum
        for how, suffix in ((seen, ""), ("as inspect_fovs", "_lofo")):
            cell = [name if how == seen else "", how]
            for kind in kinds:
                row = by.loc[f"{kind}{suffix}"] if f"{kind}{suffix}" in by.index else None
                judged = row is not None and int(row["candidates"]) and not pd.isna(row["rate"])
                cell.append(
                    f"{percent(row['rate'])} ({int(row['judged_new'])}/{int(row['candidates'])})"
                    if judged
                    else "not judged"
                )
            cells.append(cell)
    return table(cells, columns, indent=indent + 2)


def control_full(result: ControlResult, indent: int = 0) -> str:
    """Every number of the control: pooled counts and rates, then each stratum's rates."""
    cells = unit_plural(result.settings.get("observation_unit", "object"))
    if result.status == "not assessed":
        return wrap(_not_assessed(result), indent, hang=indent + 2)
    rows, unjudged = _rows(result, full=True)
    lines = [
        wrap(_lead(result), indent, hang=indent + 2),
        table(rows, _columns(result), indent=indent + 2),
    ]
    lines.append(wrap(_method_line(result, cells), indent, hang=indent + 2))
    strata = _strata_table(result, indent)
    if strata:
        per = f"Per {group_columns(result.pair_cols) or 'stratum'} (recognized/judged):"
        lines.append(wrap(per, indent))
        lines.append(strata)
    empty = _empty_line(result, cells)
    if empty:
        lines.append(wrap(empty, indent, hang=indent + 2))
    return "\n".join([*lines, *caution_lines(_cautions(result, cells, unjudged), indent=indent)])


def control_summary(result: ControlResult, detail: str = "short") -> str:
    """ "Can these features see the declared phenotype?" in words, short or full."""
    if check_detail(detail) == "full":
        return control_full(result)
    lines, cautions = control_parts(result)
    return "\n".join([*lines, *caution_lines(cautions, indent=0)])


__all__ = ["control_full", "control_parts", "control_summary", "named_stratum"]
