"""The sampling reliability in words: "how many fields do you need?", short and full.

The wording of ``sampling_reliability`` (``reliability.py`` holds the
computation). ``reliability_summary`` is re-exported there.

- ``detail="short"`` (the default): one table across the analysis groups,
  a row per listed field count and a column per criterion, each cell the
  share of random acquisitions of that many fields that were enough, as the
  range over the groups; the last row is what all available fields together
  reached (the ceiling). What the curves rest on in one line, then every
  caveat once as a ⚠️ line.
- ``detail="full"``: per group and criterion, a table of every share with its
  Monte Carlo standard error and its spread across hold-outs, the ceiling
  and what the curve rests on in sentences, and the caveats every curve
  shares once at the end.

With fewer distinct hold-outs than splits (few blocks), shares and spreads
are said over splits, not hold-outs, since several splits held back the
same fields. With unequal streams, the ceiling says the range of fields
the whole streams had.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._caveats import assumption, caveat_lines
from .._style import (
    among,
    check_detail,
    group_columns,
    group_label,
    merge_numbers,
    number_pattern,
    span,
    table,
    width,
    wrap,
)
from .._wording import count, listing
from ..sampling_unit._group_text import present
from ._text import percent, skipped_labels

if TYPE_CHECKING:
    from .reliability import ReliabilityResult

RANDOM_DRAWS = assumption("fields are treated as random draws, as random-position sampling does")
CONSERVATIVE = assumption(
    "the shares are conservative by construction: the held-back fields are a sample too, and "
    "their own error counts against the drawn fields, so a field count read from this curve "
    "errs high"
)
_CAP = "the held-back fields are themselves a sample, so the curve cannot be read above this"


def _ceiling_of(
    result: ReliabilityResult, group: Mapping[str, Any], criterion: str
) -> pd.DataFrame:
    """The ceiling row of one group and criterion (empty when there is none)."""
    match = result.ceiling
    if match.empty:
        return match
    match = match[match["criterion"] == criterion]
    for column, value in group.items():
        match = match[match[column] == value]
    return match


def _ceiling_value(match: pd.DataFrame, column: str) -> int | None:
    """An integer column of the ceiling row, or ``None`` when absent (a rebuilt result)."""
    if match.empty or column not in match.columns or pd.isna(match[column].iloc[0]):
        return None
    return int(match[column].iloc[0])


def _row_cells(row: Any, stream: int, n_splits: int, over_word: str, per: str) -> list[str]:
    """``fields | enough in | Monte Carlo SE | ± across | note`` of one listed field count."""
    if row.splits == 0:
        return [str(row.fovs), "–", "–", "–", f"more than the {stream} fields available"]
    if not present(row.preserved):
        undefined = f"could not be judged ({count(row.undefined_draws, 'draw')} undefined)"
        return [str(row.fovs), "–", "–", "–", undefined]
    se = getattr(row, "preserved_mc_se", None)
    spread = row.preserved_split_sd
    notes = []
    if row.fovs == stream:
        # Every draw is the same acquisition: this point is the ceiling itself.
        se, spread = None, None
        notes.append(f"all fields: one acquisition per {per}")
    if n_splits and row.defined_splits < n_splits:
        notes.append(f"judged on {row.defined_splits} of {n_splits} {over_word}")
    if row.undefined_draws:
        total = row.undefined_draws + row.defined_draws
        notes.append(f"{percent(row.undefined_draws / total)} of draws not judged")
    return [
        str(row.fovs),
        percent(row.preserved),
        percent(se) if present(se) else "–",
        percent(spread) if present(spread) else "–",
        "; ".join(notes),
    ]


def _curves(result: ReliabilityResult) -> list[dict[str, Any]]:
    """Every group and criterion: its rows, its ceiling and what it rests on, in words."""
    keys = list(result.group_cols)
    labels = result.settings.get("labels", {})
    n_splits = int(result.settings.get("n_splits", 0))
    curves = []
    for key, rows in result.curve.groupby([*keys, "criterion"], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(keys, key[: len(keys)], strict=True))
        criterion = key[-1]
        rows = rows.sort_values("fovs")
        stream = int(rows["stream_fovs"].max())
        match = _ceiling_of(result, group, criterion)
        splits = n_splits or int(rows["splits"].max())
        distinct = _distinct_holdouts(match)
        repeated = distinct is not None and distinct < splits
        # A share over splits that repeat hold-outs is not a share of distinct hold-outs.
        per, over_word = ("split", "splits") if repeated else ("hold-out", "hold-outs")
        shortest = _ceiling_value(match, "shortest_stream_fovs")
        unequal = shortest is not None and shortest < stream
        ceiling, ceiling_text, missing_text = None, None, None
        if len(match) and match["whole_stream_preserved"].notna().iloc[0]:
            row = match.iloc[0]
            ceiling = float(row["whole_stream_preserved"])
            defined = int(row["defined_splits"])
            judged = (
                f" of the {defined} that could judge it" if defined < int(row["splits"]) else ""
            )
            longest = int(row["stream_fovs"])
            fields_span = f"{shortest} to {longest}" if unequal else f"{longest}"
            ceiling_text = (
                f"All {fields_span} available fields together were enough in "
                f"{percent(ceiling)} of {over_word}{judged}: {_CAP}."
            )
        elif result.ceiling.empty:
            # A result rebuilt without its ceiling table (for example from a saved curve)
            # would otherwise read as if every field count could reach 100%.
            missing_text = (
                f"How often all {stream} available fields together were enough was not "
                "computed: the held-back fields are themselves a sample, so a flat curve may "
                "be capped by them rather than by the fields."
            )
        warning = None
        if repeated:
            warning = (
                f"the {splits} splits repeat hold-outs (too few fields or blocks to hold back a "
                "different set each time): splits of the same hold-out differ only by the "
                "random draws, so the spread across splits is partly draw noise, and the curve "
                f"rests on {count(distinct, 'reference')}, not {splits}"
            )
        partial = bool(
            ((rows["splits"] > 0) & rows["preserved"].isna()).any()
            or rows["undefined_draws"].fillna(0).gt(0).any()
            or (n_splits and (rows["defined_splits"] < n_splits)[rows["splits"] > 0].any())
        )
        curves.append(
            {
                "group": group,
                "criterion": criterion,
                "label": labels.get(criterion, criterion),
                "rows": rows,
                "stream": stream,
                "shortest": shortest,
                "ceiling": ceiling,
                "ceiling_text": ceiling_text,
                "missing_text": missing_text,
                "basis": _basis_line(rows, match, splits, distinct, stream, shortest),
                "warning": warning,
                "partial": partial,
                "cells": [
                    _row_cells(row, stream, n_splits, over_word, per)
                    for row in rows.itertuples(index=False)
                ],
                "over_word": over_word,
            }
        )
    return curves


def reliability_summary(result: ReliabilityResult, detail: str = "short") -> str:
    """ "How many fields do you need?": short, across groups, or full, per group (the module)."""
    lines, cautions = (
        reliability_full_parts(result)
        if check_detail(detail) == "full"
        else reliability_short_parts(result, indent=0)
    )
    return "\n".join([*lines, *caveat_lines(cautions, indent=0, detail=detail)])


def _groups_named(groups: Sequence[Mapping[str, Any]], total: int) -> str:
    """``in 2 of 12 groups (A · 1 and B · 2)``, or ``in all 12 groups``."""
    labels = list(dict.fromkeys(group_label(group) for group in groups))
    if len(labels) == total:
        return f"in {among(total, total)}" if total > 1 else ""
    return f"in {among(len(labels), total)} ({listing(labels, limit=3)})"


def _merged(texts: Sequence[str]) -> list[str]:
    """Sentences alike but for their numbers as one each (``_style.merge_numbers``)."""
    patterns: dict[str, list[str]] = {}
    for text in texts:
        patterns.setdefault(number_pattern(text), []).append(text)
    return [merge_numbers(alike) for alike in patterns.values()]


def _cell(values: Sequence[float], judged: int, total: int) -> str:
    """The range of shares over the groups that judged it: ``90-99%``, ``100% (3 of 12)``."""
    if not values:
        return "–"
    text = span(values, "percent")
    return text if judged == total else f"{text} ({judged} of {total})"


def reliability_short_parts(
    result: ReliabilityResult, indent: int = 2
) -> tuple[list[str], list[str]]:
    """The short form's lines under ``indent`` and its caveats (the module), for a report."""
    curves = _curves(result)
    pad = " " * indent
    groups = list(dict.fromkeys(tuple(c["group"].items()) for c in curves))
    total = len(groups)
    lines = []
    if result.group_cols:
        lines.append(f"{pad}Analysis groups: {group_columns(result.group_cols)}.")
    criteria = list(dict.fromkeys(c["criterion"] for c in curves))
    if curves:
        over = f" (the range over {count(total, 'group')})" if total > 1 else ""
        lines.append(
            wrap(
                f"How often that many random fields were enough, judged on held-back fields{over}:",
                indent,
                hang=indent + 2,
            )
        )
        lines += _share_table(curves, criteria, total, indent + 2)
        lines += [wrap(text, indent, hang=indent + 2) for text in _merged(_bases(curves))]
    cautions = _short_cautions(curves, total)
    lines += [f"{pad}{text}" for text in skipped_labels(result.skipped, list(result.group_cols))]
    return lines, cautions


def _bases(curves: Sequence[Mapping[str, Any]]) -> list[str]:
    """What each curve rests on, without the caveats every curve shares (stated once)."""
    return [c["basis"] for c in curves if c["basis"].startswith("Judged")]


def _share_table(
    curves: Sequence[Mapping[str, Any]], criteria: Sequence[str], total: int, indent: int
) -> list[str]:
    """Rows: the listed field counts, then all fields; columns: the criteria."""
    labels = {c["criterion"]: c["label"] for c in curves}
    counts = sorted({int(f) for c in curves for f in c["rows"]["fovs"]})
    ceilings = any(c["ceiling"] is not None for c in curves)
    rows, beyond = [], []
    for fovs in counts:
        at = [(c, c["rows"][c["rows"]["fovs"] == fovs]) for c in curves]
        # A count every group reached with all its fields is the ceiling row below.
        if ceilings and all(len(r) and int(r["fovs"].iloc[0]) == c["stream"] for c, r in at):
            continue
        if all(len(r) and int(r["splits"].iloc[0]) == 0 for _, r in at):
            beyond.append(fovs)  # beyond every group's fields: said below the table
            continue
        cells = [str(fovs)]
        for criterion in criteria:
            own = [(c, r) for c, r in at if c["criterion"] == criterion and len(r)]
            values = [
                float(r["preserved"].iloc[0])
                for _, r in own
                if int(r["splits"].iloc[0]) > 0 and present(r["preserved"].iloc[0])
            ]
            cells.append(_cell(values, len(values), total))
        rows.append(cells)
    if ceilings:
        streams = [c["stream"] for c in curves] + [c["shortest"] or c["stream"] for c in curves]
        cells = [f"all {span(streams)}"]
        for criterion in criteria:
            values = [
                c["ceiling"]
                for c in curves
                if c["criterion"] == criterion and c["ceiling"] is not None
            ]
            cells.append(_cell(values, len(values), total))
        rows.append(cells)
    columns = ["fields", *(labels[c] for c in criteria)]
    text = table(rows, columns, indent=indent, align="l" + "r" * len(criteria))
    after = []
    if beyond:
        most = max(c["stream"] for c in curves)
        where = "in every group" if total > 1 else "in the group"
        after.append(
            wrap(
                f"{_counts_text(beyond)}: more than the {most} fields available {where}.",
                indent,
                hang=indent + 2,
            )
        )
    if max(width(line) for line in text.splitlines()) <= 100 or len(criteria) == 1:
        return [text, *after]
    # Too wide with the criteria named in the header: number them, and name them once.
    numbered = [f"({i})" for i in range(1, len(criteria) + 1)]
    legend = "; ".join(f"{n} {labels[c]}" for n, c in zip(numbered, criteria, strict=True))
    text = table(rows, ["fields", *numbered], indent=indent, align="l" + "r" * len(criteria))
    return [wrap(f"{legend}.", indent, hang=indent + 2), text, *after]


def _counts_text(counts: Sequence[int]) -> str:
    """``50 fields``, ``40 and 50 fields``."""
    word = "field" if list(counts) == [1] else "fields"
    return f"{listing([str(c) for c in counts])} {word}"


def _short_cautions(curves: Sequence[Mapping[str, Any]], total: int) -> list[str]:
    """Every caveat of the curves once: ceilings, repeated hold-outs, what could not be judged."""
    cautions = []
    judged = [c for c in curves if c["basis"].startswith("Judged")]
    if any(c["ceiling"] is not None for c in curves):
        cautions.append(
            assumption(f"the last row, all available fields together, is the ceiling: {_CAP}")
        )
    cautions += _merged([c["missing_text"] for c in curves if c["missing_text"]])
    for text in _merged([c["warning"] for c in curves if c["warning"]]):
        named = _groups_named([c["group"] for c in curves if c["warning"]], total)
        cautions.append(f"{named}, {text}" if named else text)
    partial = [c["group"] for c in curves if c["partial"]]
    if partial:
        cautions.append(
            f"not every hold-out or draw could be judged {_groups_named(partial, total)}"
            ": the shares are over those that could (the full form says how many)"
        )
    unjudged: dict[str, list[Mapping[str, Any]]] = {}
    for curve in curves:
        if not curve["basis"].startswith("Judged"):
            unjudged.setdefault(curve["label"], []).append(curve)
    for label, missed in unjudged.items():
        # One line per target, the groups named, not one line per group.
        if len(missed) == 1 or not all(c["group"] for c in missed):
            for curve in missed:
                where_text = group_label(curve["group"]) if curve["group"] else ""
                prefix = f"{where_text}, {label}: " if where_text else f"{label}: "
                cautions.append(prefix + curve["basis"].strip())
            continue
        named = _groups_named([c["group"] for c in missed], total)
        cautions.append(
            f"{label}: could not be judged against any of the held-back fields {named}, so "
            "nothing is said there at any field count"
        )
    if judged:
        cautions += [RANDOM_DRAWS, CONSERVATIVE]
    return cautions


def reliability_full_parts(
    result: ReliabilityResult, indent: int = 0
) -> tuple[list[str], list[str]]:
    """The full form's lines under ``indent`` and its shared caveats (the module)."""
    pad = " " * indent
    curves = _curves(result)
    lines = []
    if result.group_cols:
        lines.append(f"{pad}Analysis groups: {group_columns(result.group_cols)}.")
    for curve in curves:
        head = f"{group_label(curve['group'])}, " if curve["group"] else ""
        lines.append("")
        lines.append(
            wrap(
                f"{head}how many fields do you need for {curve['label']}?", indent, hang=indent + 2
            )
        )
        columns = [
            "fields",
            "enough in",
            "Monte Carlo SE",
            f"± across {curve['over_word']}",
            "note",
        ]
        lines.append(table(curve["cells"], columns, indent=indent + 2, align="rrrrl"))
        for text in (curve["ceiling_text"], curve["missing_text"], curve["basis"].strip()):
            if text:
                lines.append(wrap(text, indent + 2, hang=indent + 4))
        if curve["warning"]:
            lines += caveat_lines([f"Warning: {curve['warning']}"], indent + 2, detail="full")
    skipped = skipped_labels(result.skipped, list(result.group_cols))
    if skipped:
        lines.append("")
        lines += [f"{pad}{text}" for text in skipped]
    judged = any(c["basis"].startswith("Judged") for c in curves)
    if lines and not lines[0]:
        lines = lines[1:]  # no header line: the first block needs no blank line before it
    return lines, [RANDOM_DRAWS, CONSERVATIVE] if judged else []


def _basis_line(
    rows: pd.DataFrame,
    match: pd.DataFrame,
    splits: int,
    distinct: int | None,
    stream: int,
    shortest: int | None,
) -> str:
    """What the curve of one group and criterion rests on: held-back fields and splits."""
    judged = rows[rows["splits"] > 0]
    if len(judged):
        smallest = int(judged["reservoir_fovs"].min())
    else:
        # Every listed count was beyond the streams: the reservoirs still held fields.
        smallest = _ceiling_value(match, "reservoir_fovs")
    over = (
        f"{count(splits, 'hold-out')}"
        if distinct is None
        else f"{count(splits, 'split')} over {count(distinct, 'distinct hold-out')}"
    )
    beyond = f"{stream} fields" + (
        f" ({shortest} in some splits)" if shortest is not None and shortest < stream else ""
    )
    held = "" if smallest is None else f" {smallest} or more held-back fields in"
    anything = rows["preserved"].notna().any() or (
        len(match) and match["whole_stream_preserved"].notna().iloc[0]
    )
    if not anything:
        amount = "fields" if smallest is None else f"{smallest} or more fields"
        return (
            f"Held back {amount} in {over}; the criterion could not be judged against any "
            "of them, so nothing is said at any field count."
        )
    return f"Judged on{held or ''} {over}; nothing is said beyond {beyond}."


def _distinct_holdouts(ceiling: pd.DataFrame) -> int | None:
    """How many distinct hold-outs the group's splits used, when the ceiling table says."""
    if ceiling.empty or "distinct_holdouts" not in ceiling.columns:
        return None
    value = ceiling["distinct_holdouts"].iloc[0]
    return None if pd.isna(value) else int(value)


__all__ = [
    "CONSERVATIVE",
    "RANDOM_DRAWS",
    "reliability_full_parts",
    "reliability_short_parts",
    "reliability_summary",
]
