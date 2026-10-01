"""The words of "Could I have sampled more efficiently?" (question 8), short and full.

``efficiency_summary(result, detail)`` writes an ``EfficiencyResult`` (medians
over hold-outs and orders, for all the criteria together):

- **short** (the default): one table with a row per order kind (recorded,
  random, spread out) and the range over the groups of the fields and stage
  travel each needed until reliably enough, the groups where the orders
  differ by more than one field in a second table, how many groups show no
  difference (a null result, not a preference), and each caveat once;
- **full**: one table with a row per group and order kind with every median,
  share and count, the groups where the orders do not differ named once, and
  the groups the replay skipped.

``order_parts(result, indent)`` gives the short form as ``(lines, cautions)``.
``group_comparison(result)`` tells, per group, whether the orders differ by
more than one field, do not, or cannot be compared (some order not reliably
enough in half or more of its runs), and whether the recorded order needed
more fields than another; ``report_status.order_status`` reads it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._caveats import caveat_lines
from .._style import among, check_detail, group_columns, group_label, span, table, wrap
from .._wording import count, duration, listing, number, percent
from .._wording import fields as fields_text
from ._text import order_name

_NULL = "a null result, not a preference for any order"
_TRAVEL = (
    "Medians over hold-outs and orders; stage travel is the straight-line path between "
    "successive fields."
)
# Rows of groups where the orders differ, in the short form.
SHORT_GROUPS = 8


def _distance(value: float, unit: str) -> str:
    if unit == "µm" and value >= 1000:
        return f"{value / 1000:.1f} mm"
    return f"{value:.0f} {unit}"


def _missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _available(row: Any) -> str:
    """``the 12 fields``, or ``the 12 to 24 fields`` when the streams differ in length."""
    longest = row.get("longest_stream_fovs", row["stream_fovs"])
    longest = row["stream_fovs"] if _missing(longest) else int(longest)
    shortest = int(row["stream_fovs"])
    return f"the {shortest if longest == shortest else f'{shortest} to {longest}'} fields"


def _judged(result: Any) -> str:
    """What the realized points are for: one criterion, or every criterion together."""
    if len(result.settings.get("labels", {})) <= 1:
        return "for this criterion"
    return "for all the criteria together"


def _groups(result: Any) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    keys = list(result.group_cols)
    frame = result.table
    parts = frame.groupby(keys, sort=False, dropna=False) if keys else [((), frame)]
    found = []
    for key, rows in parts:
        key = key if isinstance(key, tuple) else (key,)
        found.append((dict(zip(keys, key, strict=True)), rows))
    return found


def group_comparison(result: Any) -> list[dict[str, Any]]:
    """Per group: its ``group``, its ``medians`` by order kind (``None``: not reliably
    enough in half or more of the runs), ``verdict`` (``"differ"``, ``"same"`` or
    ``"open"``), ``gap`` (the fields between the slowest and the fastest reached order) and
    ``recorded_slower`` (the recorded order needed more than one field more than another,
    or never got there while another did)."""
    found = []
    for group, rows in _groups(result):
        medians = {
            row["order"]: None if _missing(row["median_fovs"]) else float(row["median_fovs"])
            for row in rows.to_dict("records")
            if int(row.get("unjudged", 0) or 0) < int(row["runs"])
        }
        reached = [v for v in medians.values() if v is not None]
        gap = max(reached) - min(reached) if reached else 0.0
        if gap > 1:
            verdict = "differ"
        elif len(medians) > 1 and len(reached) == len(medians):
            verdict = "same"
        else:
            verdict = "open"
        recorded = medians.get("recorded", 0.0)
        others = [v for k, v in medians.items() if k != "recorded" and v is not None]
        slower = "recorded" in medians and bool(others)
        slower = slower and (recorded is None or recorded - min(others) > 1)
        found.append(
            {
                "group": group,
                "medians": medians,
                "verdict": verdict,
                "gap": gap,
                "recorded_slower": slower,
            }
        )
    return found


def _order_rows(result: Any, unit: str) -> list[list[str]]:
    """Per order kind, the range over groups of its fields, travel, time and misses."""
    rows = []
    frame = result.table
    timed = "median_time" in frame and frame["median_time"].notna().any()
    for order in dict.fromkeys(frame["order"]):
        own = frame[frame["order"] == order]
        own = own[own["unjudged"].fillna(0) < own["runs"]] if "unjudged" in own else own
        reached = own[own["median_fovs"].notna()]
        never = len(own) - len(reached)
        fields = span(reached["median_fovs"]) if len(reached) else "never"
        if never and len(reached):
            fields += f" (never in {count(never, 'group')})"
        travel = reached["median_travel"].dropna()
        distance = "–"
        if len(travel):
            low, high = float(travel.min()), float(travel.max())
            distance = _distance(low, unit)
            if high > low:
                distance += f" to {_distance(high, unit)}"
        missed = own[own["reached"].fillna(1) < 1]
        misses = span(1 - missed["reached"].astype(float), "percent") if len(missed) else "–"
        row = [order_name(order), fields, distance]
        if timed:
            times = reached["median_time"].dropna()
            row.append(
                "–"
                if not len(times)
                else f"{duration(times.min())}"
                + (f" to {duration(times.max())}" if times.max() > times.min() else "")
            )
        rows.append([*row, misses])
    return rows


def _unjudged_caution(result: Any) -> list[str]:
    frame = result.table
    if "unjudged" not in frame or not frame["unjudged"].fillna(0).any():
        return []
    left = int(frame["unjudged"].fillna(0).sum())
    runs = int(frame["runs"].sum())
    return [
        (
            f"{left} of {count(runs, 'run')} could not be judged at any field count (the "
            "outcome was undefined) and are left out of the medians"
        )
    ]


def order_parts(result: Any, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form of question 8 under ``indent``, and its caveats (see the module)."""
    keys = list(result.group_cols)
    unit = result.settings.get("stage_unit", "µm")
    compared = group_comparison(result)
    n = len(compared)
    where = f" ({group_columns(keys)})" if keys else ""
    head = (
        f"Fields until reliably enough {_judged(result)}, median over hold-outs and orders, "
        f"in {count(n, 'group')}{where}:"
    )
    frame = result.table
    timed = "median_time" in frame and frame["median_time"].notna().any()
    columns = ["order", "fields", "stage travel", *(["time"] if timed else []), "never enough"]
    lines = [wrap(head, indent, hang=indent + 2)]
    if len(frame):
        lines.append(table(_order_rows(result, unit), columns, indent=indent + 2))
    differ = [c for c in compared if c["verdict"] == "differ"]
    same = [c for c in compared if c["verdict"] == "same"]
    open_ = [c for c in compared if c["verdict"] == "open"]
    if differ:
        orders = list(dict.fromkeys(frame["order"]))
        lines.append(
            " " * indent + f"The orders differ by more than one field in {among(len(differ), n)}:"
        )
        rows = [
            [
                group_label(c["group"]),
                *[
                    "–"
                    if o not in c["medians"]
                    else ("never" if c["medians"][o] is None else number(c["medians"][o]))
                    for o in orders
                ],
            ]
            for c in differ[:SHORT_GROUPS]
        ]
        names = ["group", *(order_name(o) for o in orders)]
        lines.append(table(rows, names, indent=indent + 2))
        if len(differ) > SHORT_GROUPS:
            more = count(len(differ) - SHORT_GROUPS, "group")
            lines.append(" " * (indent + 2) + f"and {more} more (report_full.txt).")
    if same:
        text = f"In {among(len(same), n)} the orders do not differ by more than one field: {_NULL}."
        lines.append(wrap(text, indent, hang=indent + 2))
    if open_:
        labels = [group_label(c["group"]) for c in open_]
        text = (
            f"Not compared in {count(len(open_), 'group')} ({listing(labels, 3)}): some order "
            "was not reliably enough in half or more of its runs."
        )
        lines.append(wrap(text, indent, hang=indent + 2))
    lines += [wrap(line, indent, hang=indent + 2) for line in _skipped(result)]
    lines.append(wrap(_TRAVEL, indent, hang=indent + 2))
    return lines, _unjudged_caution(result)


def _skipped(result: Any) -> list[str]:
    """``<group>: skipped, <reason>.`` for every group the replay could not run."""
    skipped = result.skipped
    if skipped is None or skipped.empty or "reason" not in skipped.columns:
        return []
    shown = [k for k in result.group_cols if k in skipped.columns]
    return [
        f"{group_label({k: row[k] for k in shown})}: skipped, {row['reason']}."
        for _, row in skipped.iterrows()
    ]


def _cells(row: dict[str, Any], unit: str) -> tuple[str, str, str, str, str]:
    """A table row's enough-after, stage travel, time, never-within and left-out cells."""
    runs, unjudged = int(row["runs"]), int(row.get("unjudged", 0) or 0)
    left_out = f"{unjudged} of {count(runs, 'run')}" if unjudged else "–"
    if unjudged == runs:
        return "not judged (undefined in every run)", "–", "–", "–", left_out
    never = (
        f"{percent(1 - row['reached'])} of runs, within {_available(row).removeprefix('the ')}"
        if row["reached"] < 1
        else "–"
    )
    if _missing(row["median_fovs"]):
        return "not in half or more of runs", "–", "–", never, left_out
    travel = "–" if _missing(row["median_travel"]) else _distance(row["median_travel"], unit)
    time = "–" if _missing(row.get("median_time")) else f"about {duration(row['median_time'])}"
    return fields_text(row["median_fovs"]), travel, time, never, left_out


def _full(result: Any) -> str:
    """Every median, share and count, one row per group and order kind (see the module)."""
    keys = list(result.group_cols)
    unit = result.settings.get("stage_unit", "µm")
    frame = result.table
    lines = []
    if keys:
        lines.append(f"Analysis groups: {group_columns(keys)}.")
    lines.append(
        wrap(
            f"Could the same positions have been visited in a better order? Reliably enough "
            f"{_judged(result)}, per group and order:",
            0,
            hang=2,
        )
    )
    timed = "median_time" in frame and frame["median_time"].notna().any()
    rows = []
    for group, own in _groups(result):
        for row in own.to_dict("records"):
            enough, travel, time, never, left_out = _cells(row, unit)
            cells = [group_label(group), order_name(row["order"]), enough]
            cells += [travel, *([time] if timed else []), never, left_out]
            rows.append(cells)
    columns = ["group", "order", "reliably enough after", "stage travel"]
    columns += [*(["time"] if timed else []), "never enough", "left out"]
    if rows:
        lines.append(table(rows, columns))
    compared = group_comparison(result)
    same = [group_label(c["group"]) for c in compared if c["verdict"] == "same"]
    if same:
        lines.append(
            wrap(
                f"The orders do not differ by more than one field {_judged(result)} in "
                f"{listing(same, len(same))}: {_NULL}.",
                0,
                hang=2,
            )
        )
    differ = [c for c in compared if c["verdict"] == "differ"]
    if differ:
        named = listing(
            [f"{group_label(c['group'])} ({fields_text(c['gap'])} apart)" for c in differ],
            len(differ),
        )
        lines.append(wrap(f"The orders differ by more than one field in {named}.", 0, hang=2))
    lines.append(_TRAVEL)
    lines += [wrap(line, 0, hang=2) for line in _skipped(result)]
    return "\n".join(lines)


def efficiency_summary(result: Any, detail: str = "short") -> str:
    """ "Your raster needed 28 fields and 7 mm; spreading them out ...": a table per order
    kind (short) or per group and order kind (full)."""
    if check_detail(detail) == "full":
        return _full(result)
    lines, cautions = order_parts(result)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0)])


__all__ = ["efficiency_summary", "group_comparison", "order_parts"]
