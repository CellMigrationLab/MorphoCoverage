"""The adequacy points in words: "what did the extra fields buy?", short and full.

The wording of ``adequacy_points`` (``adequacy.py`` holds the computation and
re-exports ``adequacy_summary`` and the point helpers).

- ``detail="short"`` (the default): across the analysis groups, one table of
  when everything wanted was first, and reliably, preserved in held-back
  fields, a row per order kind (the medians over hold-outs and orders, as the
  range over groups); one table of when each rule would have said stop, a
  row per rule by its readable name and a column per order kind; one table
  per criterion when several are declared; then every caveat once.
- ``detail="full"``: per group, the sentences of every order kind with their
  numbers, and each rule's stops in a table (readable name, strategy and
  config id).

A median over runs counts a run that never got there as beyond its own
stream (``_text.median_beyond``); a tail is an observed field count, never
interpolated (``_tail_value``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._caveats import caveat_lines
from .._style import check_detail, group_columns, group_label, span, table, wrap
from .._wording import listing
from ._text import median_beyond, number, order_phrase, rows_of, rule_names, skipped_labels

if TYPE_CHECKING:
    from .adequacy import AdequacyResult

ALL_CRITERIA = "all criteria"
RELIABLY = (
    "'Reliably' means every later field count, up to the last available field, also preserved "
    "it; each hold-out is a different set of held-back fields."
)
# The order kinds as the short tables name them, in the order the summaries give them.
_ORDER_COLUMNS = {"dispersed": "spread out", "permuted": "random", "recorded": "recorded"}
_ORDER_ROWS = {
    "dispersed": "the same positions spread out",
    "permuted": "random orders",
    "recorded": "the recorded order",
}


def _median_point(values: pd.Series, streams: pd.Series) -> str:
    """The median over runs as "field 12"; a run that never reached it counts as beyond its own
    stream, so the median is "not reached" when half or more of the runs never reached it.
    """
    median = median_beyond(values)
    if median is None:
        return (
            f"not within the fields available (up to {int(streams.max())}) in half or more of runs"
        )
    return f"field {number(median)}"


def _tail_value(values: pd.Series) -> float | None:
    """The 90th percentile over runs as an observed field count, or ``None`` when more than
    10% of runs never got there (they count as beyond their stream)."""
    filled = values.astype("Float64").to_numpy(dtype=float, na_value=np.inf)
    tail = float(np.quantile(filled, 0.9, method="inverted_cdf"))
    return tail if np.isfinite(tail) else None


def _tail_point(values: pd.Series, streams: pd.Series) -> str:
    """The 90th percentile over runs, the number a protocol has to serve.

    Taken as an observed field count (the smallest one reached by at least
    90% of runs), never interpolated between runs, so "by field k in 90% of
    runs" is true of the runs; a run that never reached it counts as beyond
    its stream.
    """
    tail = _tail_value(values)
    if tail is None:
        limit = int(streams.max())
        return f"not within the fields available (up to {limit}) in more than 10% of runs"
    return f"by field {int(tail)} in 90% of runs"


def _reached(point: str, reached: str, missed: str) -> str:
    """``reached`` followed by the point, or ``missed`` in its place when it was not reached:
    "first preserved at field 3", "not preserved within the fields available (up to 11) ..."."""
    if point.startswith("not"):
        return f"{missed} {point.removeprefix('not ')}"
    return f"{reached} {point}"


def _unjudged(rows: pd.DataFrame) -> pd.Series:
    """Runs in which no prefix could be judged (every outcome undefined)."""
    if "judged_fovs" not in rows.columns:
        return pd.Series(False, index=rows.index)
    return rows["judged_fovs"].fillna(0).astype(int).eq(0)


def _tail_unjudged(rows: pd.DataFrame) -> pd.Series:
    """Judged runs whose last fields could not be judged, so 'reliably' was not confirmed."""
    if "last_judged" not in rows.columns:
        return pd.Series(False, index=rows.index)
    last = rows["last_judged"].astype("Float64")
    return (last.notna() & (last < rows["stream_fovs"].astype(float))).fillna(False).astype(bool)


def _realized_sentence(rows: pd.DataFrame) -> str:
    """When everything wanted was first, and reliably, preserved over the judged runs."""
    unjudged = _unjudged(rows)
    if unjudged.all():
        return (
            "judged on held-back fields, what you wanted could not be judged at any field count "
            "(the outcome was undefined in every run)."
        )
    rows = rows[~unjudged]
    stream = rows["stream_fovs"]
    first = _median_point(rows["first_realized"], stream)
    lasting = _median_point(rows["sustained_realized"], stream)
    tail = _tail_point(rows["sustained_realized"], stream)
    if first.startswith("not"):
        text = f"everything you wanted was not preserved {first.removeprefix('not ')}"
    else:
        text = (
            f"everything you wanted was first preserved at {first} and "
            f"{_reached(lasting, 'reliably from', 'not reliably')}"
        )
    skipped = (
        f"; {int(unjudged.sum())} of {len(unjudged)} runs could not be judged and are left out"
        if unjudged.any()
        else ""
    )
    cut = _tail_unjudged(rows)
    if cut.any():
        skipped += (
            f"; in {int(cut.sum())} of {len(cut)} runs the last fields could not be judged, "
            "so it was not confirmed reliably there"
        )
    return f"judged on held-back fields, {text} (medians; {tail}{skipped})."


def _groups(result: AdequacyResult) -> list[dict[str, Any]]:
    """Every group the points hold, apparent or realized, in order."""
    keys = list(result.group_cols)
    if not keys:
        return [{}]
    frames = [result.apparent[keys], result.realized[keys]]
    found = pd.concat(frames).drop_duplicates()
    return [{k: row[k] for k in keys} for _, row in found.iterrows()]


def _orders(frame: pd.DataFrame) -> list[str]:
    """The order kinds of a table, spread out, random, recorded first, then any other."""
    kinds = list(dict.fromkeys(frame["order"])) if "order" in frame else []
    known = [k for k in _ORDER_COLUMNS if k in kinds]
    return known + sorted(k for k in kinds if k not in _ORDER_COLUMNS)


def _cell(medians: Sequence[float | None]) -> str:
    """The range of the groups' medians: ``1-3``; ``1-3; never in 2`` or ``never``."""
    reached = [m for m in medians if m is not None]
    missed = len(medians) - len(reached)
    if not reached:
        return "never" if medians else "–"
    text = span(reached)
    return f"{text}; never in {missed}" if missed else text


def _realized_rows(result: AdequacyResult, criterion: str) -> pd.DataFrame:
    realized = result.realized
    return realized[realized["criterion"] == criterion]


def adequacy_short_parts(result: AdequacyResult, indent: int = 2) -> tuple[list[str], list[str]]:
    """The short form's lines under ``indent`` and its caveats (the module), for a report."""
    pad = " " * indent
    groups = _groups(result)
    total = len(groups)
    labels = result.settings.get("labels", {})
    over = f", the range over {total} groups" if total > 1 else ""
    lines, cautions = [], []
    if result.group_cols:
        lines.append(f"{pad}Analysis groups: {group_columns(result.group_cols)}.")
    wanted = listing(list(labels.values()), limit=3) or "what you wanted"
    everything = _realized_rows(result, ALL_CRITERIA)
    if len(everything):
        lines.append(
            wrap(
                f"Judged on held-back fields ({wanted}): the field from which everything wanted "
                f"was preserved (medians over hold-outs and orders{over}):",
                indent,
                hang=indent + 2,
            )
        )
        rows = []
        for order in _orders(everything):
            own = everything[everything["order"] == order]
            per_group = [rows_of(own, group) for group in groups]
            per_group = [g[~_unjudged(g)] for g in per_group if len(g)]
            per_group = [g for g in per_group if len(g)]
            rows.append(
                [
                    _ORDER_ROWS.get(order, order),
                    _cell([median_beyond(g["first_realized"]) for g in per_group]),
                    _cell([median_beyond(g["sustained_realized"]) for g in per_group]),
                    _cell([_tail_value(g["sustained_realized"]) for g in per_group]),
                ]
            )
        columns = ["order", "first preserved", "reliably from", "90% of runs by"]
        lines.append(table(rows, columns, indent=indent + 2))
        cautions += _realized_cautions(everything, groups, total)
    if len(labels) > 1:
        lines += _criteria_table(result, groups, labels, indent)
    lines += _rules_table(result, groups, indent, over)
    if len(everything) or len(result.apparent):
        lines.append(wrap(RELIABLY, indent, hang=indent + 2))
    lines += [f"{pad}{text}" for text in skipped_labels(result.skipped, list(result.group_cols))]
    return lines, cautions


def _named(groups: Sequence[Mapping[str, Any]], total: int) -> str:
    labels = [group_label(g) for g in groups]
    if total == 1:
        return ""
    return f"in {len(labels)} of {total} groups ({listing(labels, limit=3)}), "


def _realized_cautions(
    everything: pd.DataFrame, groups: Sequence[Mapping[str, Any]], total: int
) -> list[str]:
    """Runs left out as not judged, and runs whose last fields could not confirm 'reliably'."""
    unjudged, cut, runs_out, runs_cut, runs = [], [], 0, 0, 0
    for group in groups:
        own = rows_of(everything, group)
        if not len(own):
            continue
        out = _unjudged(own)
        runs += len(own)
        if out.any():
            unjudged.append(group)
            runs_out += int(out.sum())
        tail = _tail_unjudged(own[~out])
        if tail.any():
            cut.append(group)
            runs_cut += int(tail.sum())
    cautions = []
    if unjudged:
        cautions.append(
            f"{_named(unjudged, total)}{runs_out} of {runs} runs could not be judged at any "
            "field count and are left out"
        )
    if cut:
        cautions.append(
            f"{_named(cut, total)}in {runs_cut} of {runs} runs the last fields could not be "
            "judged, so 'reliably' was not confirmed there"
        )
    return cautions


def _criteria_table(
    result: AdequacyResult,
    groups: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    indent: int,
) -> list[str]:
    """Each criterion on its own: reliably preserved from (the range of the groups' medians)."""
    orders = _orders(result.realized)
    rows = []
    for criterion, label in labels.items():
        own = _realized_rows(result, criterion)
        if not len(own):
            continue
        cells = [label]
        for order in orders:
            per_group = [rows_of(own[own["order"] == order], group) for group in groups]
            per_group = [g[~_unjudged(g)] for g in per_group if len(g)]
            judged = [g for g in per_group if len(g)]
            cells.append(
                _cell([median_beyond(g["sustained_realized"]) for g in judged])
                if judged
                else "not judged"
            )
        rows.append(cells)
    if not rows:
        return []
    head = wrap("Each target on its own, reliably preserved from:", indent, hang=indent + 2)
    columns = ["to preserve", *(_ORDER_COLUMNS.get(o, o) for o in orders)]
    return [head, table(rows, columns, indent=indent + 2)]


def _rules_table(
    result: AdequacyResult, groups: Sequence[Mapping[str, Any]], indent: int, over: str
) -> list[str]:
    """When each rule would have first said stop (and kept saying it), a column per order."""
    apparent = result.apparent
    if not len(apparent):
        return []
    names = rule_names(result.settings)
    orders = _orders(apparent)
    rows = []
    for config_id, own in apparent.groupby("config_id", sort=False):
        cells = [names.get(config_id, own["strategy"].iloc[0])]
        for order in orders:
            per_group = [rows_of(own[own["order"] == order], group) for group in groups]
            per_group = [g for g in per_group if len(g)]
            first = _cell([median_beyond(g["first_apparent"]) for g in per_group])
            kept = _cell([median_beyond(g["sustained_apparent"]) for g in per_group])
            cells.append(first if kept == first else f"{first} (kept from {kept})")
        rows.append(cells)
    head = wrap(
        f"When each rule would have first said stop (median field{over}; 'kept from': the field "
        "from which it kept saying stop):",
        indent,
        hang=indent + 2,
    )
    columns = ["rule", *(_ORDER_COLUMNS.get(o, o) for o in orders)]
    return [head, table(rows, columns, indent=indent + 2)]


def adequacy_full_parts(result: AdequacyResult, indent: int = 0) -> tuple[list[str], list[str]]:
    """The full form's lines under ``indent``: every sentence per group (the module)."""
    pad = " " * indent
    keys = list(result.group_cols)
    labels = result.settings.get("labels", {})
    names = rule_names(result.settings)
    lines = [f"{pad}Analysis groups: {group_columns(keys)}."] if keys else []
    full = rule_names(result.settings, detail="full")
    if full:
        lines.append(wrap(f"Rules: {'; '.join(full.values())}.", indent, hang=indent + 2))
    for group in _groups(result):
        apparent, realized = rows_of(result.apparent, group), rows_of(result.realized, group)
        if lines:
            lines.append("")
        head = f"{group_label(group)}: " if keys else ""
        lines.append(f"{pad}{head}when was it enough? (medians over hold-outs and orders)")
        for order, rows in realized[realized["criterion"] == ALL_CRITERIA].groupby("order"):
            sentence = f"{order_phrase(order, capital=True)}: {_realized_sentence(rows)}"
            lines.append(wrap(sentence, indent + 2, hang=indent + 4))
            if len(labels) > 1:
                lines += _criterion_lines(realized, order, labels, indent + 4)
        lines += _full_rules(apparent, names, indent + 2)
    if len(result.apparent) or len(result.realized):
        lines.append(wrap(RELIABLY, indent, hang=indent + 2))
    skipped = skipped_labels(result.skipped, keys)
    lines += [f"{pad}{text}" for text in skipped]
    return lines, []


def _stop_cell(own: pd.DataFrame) -> str:
    """``field 3``, ``field 3 (kept from field 5)``, or ``never (up to 18)``: one rule's stops."""
    first = _median_point(own["first_apparent"], own["stream_fovs"])
    if first.startswith("not"):
        return f"never (up to {int(own['stream_fovs'].max())})"
    lasting = _median_point(own["sustained_apparent"], own["stream_fovs"])
    if lasting.startswith("not"):
        lasting = f"never (up to {int(own['stream_fovs'].max())})"
    return first if lasting == first else f"{first} (kept from {lasting})"


def _full_rules(apparent: pd.DataFrame, names: Mapping[str, str], indent: int) -> list[str]:
    """One group's rules: when each would have first said stop, a column per order kind."""
    if not len(apparent):
        return []
    orders = _orders(apparent)
    rows = []
    for config_id, own in apparent.groupby("config_id", sort=False):
        cells = [names.get(config_id, own["strategy"].iloc[0])]
        for order in orders:
            chosen = own[own["order"] == order]
            cells.append(_stop_cell(chosen) if len(chosen) else "–")
        rows.append(cells)
    head = (
        "When each rule would have first said stop (medians; 'kept from': from which field it "
        "kept saying stop; never: not within the fields available in half or more of runs):"
    )
    columns = ["rule", *(_ORDER_COLUMNS.get(o, o) for o in orders)]
    return [wrap(head, indent, hang=indent + 2), table(rows, columns, indent=indent + 2)]


def _criterion_lines(
    realized: pd.DataFrame, order: str, labels: Mapping[str, str], indent: int
) -> list[str]:
    """Each criterion's own reliably-preserved point in one order kind."""
    lines = []
    chosen = realized[(realized["order"] == order) & (realized["criterion"] != ALL_CRITERIA)]
    for criterion, own in chosen.groupby("criterion", sort=False):
        label = labels.get(criterion, criterion)
        judged = own[~_unjudged(own)]
        if judged.empty:
            lines.append(wrap(f"{label}: could not be judged in any run.", indent, hang=indent + 2))
            continue
        point = _median_point(judged["sustained_realized"], judged["stream_fovs"])
        reached = _reached(point, "reliably from", "not reliably")
        lines.append(wrap(f"{label}: {reached}.", indent, hang=indent + 2))
    return lines


def adequacy_summary(result: AdequacyResult, detail: str = "short") -> str:
    """When rules would have stopped and when held-back fields confirmed it (the module)."""
    if check_detail(detail) == "full":
        lines, cautions = adequacy_full_parts(result)
    else:
        lines, cautions = adequacy_short_parts(result, indent=0)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0, detail=detail)])


__all__ = [
    "ALL_CRITERIA",
    "RELIABLY",
    "adequacy_full_parts",
    "adequacy_short_parts",
    "adequacy_summary",
]
