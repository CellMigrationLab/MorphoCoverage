"""Protocol-level reliability in words: "for experiments like this one", short and full.

The wording of ``protocol_reliability`` (``protocol.py`` holds the
computation). ``protocol_summary`` and ``FEW_UNITS`` are re-exported there.

- ``detail="short"`` (the default): one table, a row per criterion (and
  stratum): the fewest listed fields from which every sample that could be
  judged was served (``served_from``), the fields each sample needed
  (median and range), and the leave-one-unit-out check; then, per
  criterion, what the table cannot say (samples capped by their held-back
  fields, samples needing more fields than could be judged, units not
  checked), and every caveat once.
- ``detail="full"``: per stratum and criterion, a table of every listed field
  count (samples served, per independent unit, the plausible range), the
  samples named, the check, and the caveats once at the end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._caveats import assumption, caveat_lines
from .._style import check_detail, group_label, merge_numbers, number_pattern, span, table, wrap
from .._wording import count, number
from ._text import column_word, fields, median_beyond, percent, plural, rows_of, sample_word

if TYPE_CHECKING:
    from .protocol import ProtocolResult

FEW_UNITS = 5


def served_from(rows: pd.DataFrame) -> int | None:
    """The fewest listed fields from which every judged sample was served at every larger
    count (``adequate == eligible`` wherever ``eligible > 0``), or ``None``."""
    judged = rows[rows["eligible"] > 0].sort_values("fovs")
    found = None
    for row in judged.itertuples(index=False):
        if row.adequate == row.eligible:
            found = int(row.fovs) if found is None else found
        else:
            found = None
    return found


def _words(result: ProtocolResult) -> dict[str, str]:
    hierarchy = result.settings.get("hierarchy", {})
    # The column's own name, readable: "plate", "technical repeat".
    unit = column_word(result.settings.get("independent_unit", "unit"))
    sample = sample_word(hierarchy)
    return {
        "unit": unit,
        "units": plural(unit),
        "sample": sample,
        "samples": plural(sample),
        "threshold": percent(result.settings.get("threshold", 0.9)),
    }


def _named(rows: pd.DataFrame, columns: Sequence[str], extra: Any = None) -> str:
    """``B6, 1A (more than 18); ...``: samples by their column values."""
    return "; ".join(
        group_label({c: r[c] for c in columns}) + (extra(r) if extra else "")
        for _, r in rows.iterrows()
    )


def protocol_answers(result: ProtocolResult) -> list[dict[str, Any]]:
    """Every stratum and criterion: its table rows, its sentences and its short cells.

    Each answer holds the ``stratum``, the criterion's ``label``, the number of
    ``samples`` and independent ``units``, ``served`` (``served_from``), and,
    when some samples could not say, ``unreached``, ``short_of_fields`` (the
    samples needing more fields than could be judged) and ``capped`` (those
    capped by their held-back fields); ``served_cell`` words it.
    """
    labels = result.settings.get("labels", {})
    w = _words(result)
    blocks = []
    keys = [*result.strata, "criterion"]
    for key, rows in result.table.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        where = dict(zip(keys, key, strict=True))
        stratum = {k: where[k] for k in result.strata}
        n_samples = int(rows["eligible"].max() + rows["not_evaluated"].min())
        n_units = int(rows["all_units"].max())
        block = {
            "stratum": stratum,
            "label": labels.get(where["criterion"], where["criterion"]),
            "samples": n_samples,
            "units": n_units,
            "cells": _cells(rows, rows_of(result.per_unit, where), w),
            "served": served_from(rows),
            "notes": [],
            "short_notes": [],
            "needed": "–",
            "check": "not checked",
        }
        partly = int(rows["partly_judged"].max()) if "partly_judged" in rows else 0
        if partly:
            block["short_notes"].append(
                f"at some field counts {partly} judged on only some of the hold-outs (the others "
                "could not judge it)"
            )
        _difficulty(block, rows_of(result.difficulty, where), result.sample_cols, w)
        _validation(block, rows_of(result.validation, where), n_units, w)
        blocks.append(block)
    return blocks


def _cells(rows: pd.DataFrame, units: pd.DataFrame, w: Mapping[str, str]) -> list[list[str]]:
    """``fields | enough for | per unit | plausible range | note`` of every listed count."""
    cells = []
    for row in rows.sort_values("fovs").itertuples(index=False):
        beyond = int(getattr(row, "beyond_stream", 0) or 0)
        undefined = int(row.not_evaluated) - beyond
        if row.eligible == 0:
            cells.append(
                [str(row.fovs), "–", "–", "–", _none_judged(w["sample"], beyond, undefined)]
            )
            continue
        per = units[units["fovs"] == row.fovs]
        by_unit = ", ".join(f"{u.adequate} of {u.eligible}" for u in per.itertuples())
        notes = []
        if beyond:
            notes.append(f"{beyond} had fewer fields than that")
        if undefined:
            notes.append(f"{undefined} could not be judged at this depth")
        partly = getattr(row, "partly_judged", 0)
        if partly:
            notes.append(
                f"{partly} judged on only some of the hold-outs (the others could not judge the "
                "criterion)"
            )
        interval = (
            "–"
            if row.share_low is None or pd.isna(row.share_low)
            else f"{percent(row.share_low)} to {percent(row.share_high)}"
        )
        cells.append(
            [
                str(row.fovs),
                f"{row.adequate} of {row.eligible} {w['samples']}",
                by_unit,
                interval,
                "; ".join(notes),
            ]
        )
    return cells


def _difficulty(
    block: dict[str, Any], hard: pd.DataFrame, sample_cols: Sequence[str], w: Mapping[str, str]
) -> None:
    """The fields each sample needed, and the samples that could not say, into ``block``."""
    if not len(hard):
        return
    never = hard[hard["max_fovs"].isna()]
    needs_more = hard[hard["fields_needed"].isna() & hard["max_fovs"].notna()]
    # A sample whose held-back fields were not enough even with every field is
    # capped by its reference, not short of fields; it is named apart.
    capped = hard["ceiling_below_threshold"].fillna(False).astype(bool)
    short = needs_more[~capped.loc[needs_more.index]]
    needed, cell = _needed(hard, short, w["sample"], w["samples"])
    block["notes"] += needed
    block["needed"] = cell
    if len(short):
        named = _named(short, sample_cols, lambda r: f" (more than {r['max_fovs']})")
        block["notes"].append(f"Needs more fields than could be judged: {named}.")
        block["short_notes"].append(
            f"{count(len(short), w['sample'], w['samples'])} needed more fields than could be "
            f"judged ({named})"
        )
    reference_capped = hard[capped]
    if len(reference_capped):
        ceilings = [float(c) for c in reference_capped["ceiling"]]
        named = _named(
            reference_capped,
            sample_cols,
            lambda r: f" (all fields enough in {percent(float(r['ceiling']))} of hold-outs)",
        )
        why = (
            f"even every available field together was enough in fewer than {w['threshold']} of "
            "hold-outs: the held-back fields are themselves a sample and cap what can be shown, "
            "so these are not counted as needing more fields"
        )
        block["notes"].append(
            f"For {len(reference_capped)} of {len(hard)} {w['samples']} {why}: {named}."
        )
        reached = span(ceilings, "percent")
        # The reason is said once under the table (protocol_short_parts).
        block["short_notes"].append(
            f"for {len(reference_capped)} of {len(hard)} {w['samples']} every available field "
            f"together was enough in fewer than {w['threshold']} of hold-outs ({reached})"
        )
        block["capped"] = len(reference_capped)
    if len(never):
        named = _named(never, sample_cols)
        block["notes"].append(f"Could not be judged at any depth: {named}.")
        block["short_notes"].append(f"could not be judged at any depth: {named}")
    block["unreached"] = len(short) + len(reference_capped) + len(never)
    block["never"] = len(never)
    block["short_of_fields"] = len(short)


def _validation(
    block: dict[str, Any], check: pd.DataFrame, n_units: int, w: Mapping[str, str]
) -> None:
    """The leave-one-unit-out check into ``block``: its sentences and its short cell."""
    if not len(check):
        block["check"] = "–"
        return
    chosen = check.dropna(subset=["chosen_fovs"])
    # A left-out unit whose samples have fewer fields than the chosen number
    # was not checked: it is named, never counted as "0 of 0".
    unchecked = chosen[chosen["eligible"] == 0]
    checked = chosen[chosen["eligible"] > 0]
    cell = []
    if len(checked):
        served, total = int(checked["adequate"].sum()), int(checked["eligible"].sum())
        # The number is chosen anew for each unit left out: one count, or their range.
        counts = checked["chosen_fovs"].astype(int)
        same = counts.nunique() == 1
        chosen_at = (
            fields(int(counts.iloc[0]))
            if same
            else f"{span(counts)} fields, by the {w['unit']} left out"
        )
        block["notes"].append(
            f"Choosing the number on the other {w['units']} ({chosen_at}) and checking it on "
            f"the one left out: enough for {served} of {total} {w['samples']}."
        )
        cell.append(f"{served} of {total}" + ("" if same else f" (at {span(counts)} fields)"))
    if len(unchecked):
        named = "; ".join(
            f"{r['left_out']} (chosen {int(r['chosen_fovs'])})" for _, r in unchecked.iterrows()
        )
        block["notes"].append(
            f"Not checked on {count(len(unchecked), w['unit'], w['units'])} whose "
            f"{w['samples']} had fewer fields than the number chosen on the others: {named}."
        )
        block["short_notes"].append(
            f"not checked on {count(len(unchecked), w['unit'], w['units'])} with fewer fields "
            f"than the number chosen on the others ({named})"
        )
        cell.append(f"{count(len(unchecked), w['unit'], w['units'])} not checked")
    unchosen = len(check) - len(chosen)
    if unchosen and n_units < 2:
        text = (
            f"With a single {w['unit']}, no number can be chosen on other {w['units']} and "
            "checked on it."
        )
        block["notes"].append(text)
        block["short_notes"].append(text[:1].lower() + text[1:-1])
    elif unchosen:
        text = (
            f"For {count(unchosen, w['unit'], w['units'])}, the others reached no number "
            f"that served enough {w['samples']}, so nothing was checked there."
        )
        block["notes"].append(text)
        # The reason is said once under the table (protocol_short_parts).
        block["short_notes"].append(f"not checked on {count(unchosen, w['unit'], w['units'])}")
        block["unchosen"] = unchosen
        if len(chosen):
            cell.append(f"{count(unchosen, w['unit'], w['units'])} not checked")
    block["check"] = "; ".join(cell) or "not checked"


def _few(blocks: Sequence[Mapping[str, Any]], w: Mapping[str, str]) -> list[str]:
    """The few-units note, once: ``With 3 repeats the field count is a guide: ...``. Few
    repeats is a usual design, so it is a standing note on how to read the count, not a
    caution about these data."""
    texts = [
        (
            f"With 1 {w['unit']} the field count rests on that {w['unit']} alone: it covers the "
            f"variation between its fields, not how other {w['units']} or {w['samples']} "
            "differ, so check it on the next experiment before fixing a protocol."
        )
        if b["units"] == 1
        else (
            f"With {count(b['units'], w['unit'], w['units'])} the field count is a guide: it "
            f"rests on these {w['units']}, so check it on the next experiment before fixing a "
            "protocol."
        )
        for b in blocks
        if b["units"] < FEW_UNITS
    ]
    patterns: dict[str, list[str]] = {}
    for text in texts:
        patterns.setdefault(number_pattern(text), []).append(text)
    return [assumption(merge_numbers(alike)) for alike in patterns.values()]


def served_cell(block: Mapping[str, Any]) -> str:
    """``2 fields``, or why no count served every sample: ``not shown`` (capped by the
    held-back fields), ``not judged`` (at no depth), ``not reached``."""
    if block["served"] is not None:
        return fields(block["served"])
    if block.get("unreached") and not block.get("short_of_fields"):
        if block.get("capped"):
            return "not shown"
        if block.get("never"):
            return "not judged"
    return "not reached"


def protocol_short_parts(result: ProtocolResult, indent: int = 2) -> tuple[list[str], list[str]]:
    """The short form's lines under ``indent`` and its caveats (the module), for a report."""
    w = _words(result)
    blocks = protocol_answers(result)
    if not blocks:
        return [], []
    samples = span([b["samples"] for b in blocks])
    units = span([b["units"] for b in blocks])
    unit_words = w["unit"] if units == "1" else w["units"]
    sample_words = w["sample"] if samples == "1" else w["samples"]
    pooled = "" if result.strata else ", over all samples"
    lines = [
        wrap(
            f"Fields per {w['sample']}, for experiments like this one{pooled} ({samples} "
            f"{sample_words} "
            f"in {units} {unit_words}; a {w['sample']} counts when at least {w['threshold']} of "
            "random acquisitions were enough):",
            indent,
            hang=indent + 2,
        )
    ]
    strata = list(result.strata)
    columns = [
        *(["stratum"] if strata else []),
        "to preserve",
        "all served from",
        "needed",
        "left-out check",
    ]
    rows = [
        [
            *([group_label(b["stratum"])] if strata else []),
            b["label"],
            served_cell(b),
            b["needed"],
            b["check"],
        ]
        for b in blocks
    ]
    lines.append(table(rows, columns, indent=indent + 2))
    lines.append(
        wrap(
            f"All served from: every {w['sample']} that could be judged was served at that count "
            f"and every larger one. Needed: the median (range) of the fields each needed. "
            f"Left-out check: the number chosen on the other {w['units']}, tried on the one left "
            "out.",
            indent,
            hang=indent + 2,
        )
    )
    lines += _short_notes(blocks, bool(strata), indent)
    lines += [wrap(text, indent, hang=indent + 2) for text in _short_reasons(blocks, w)]
    return lines, _few(blocks, w)


def _short_notes(blocks: Sequence[Mapping[str, Any]], strata: bool, indent: int) -> list[str]:
    """Each stratum and target's notes, or the same note for every one said once."""
    lines = []
    noted = [block for block in blocks if block["short_notes"]]
    shared = {"; ".join(block["short_notes"]) for block in noted}
    if len(noted) == len(blocks) > 1 and len(shared) == 1:
        # The same note for every stratum and target: said once.
        text = f"For every {'stratum and ' if strata else ''}target: {shared.pop()}."
        lines.append(wrap(text, indent, hang=indent + 2))
        noted = []
    for block in noted:
        where = group_label(block["stratum"]) + ", " if strata else ""
        text = f"{where}{block['label']}: {'; '.join(block['short_notes'])}."
        lines.append(wrap(text[:1].upper() + text[1:], indent, hang=indent + 2))
    return lines


def _short_reasons(blocks: Sequence[Mapping[str, Any]], w: Mapping[str, str]) -> list[str]:
    """Why some samples are not counted as needing more fields, and why a check was not
    made, each once under the table."""
    reasons = []
    if any(block.get("capped") for block in blocks):
        reasons.append(
            f"Every available field enough in fewer than {w['threshold']} of hold-outs: the "
            f"held-back fields are themselves a sample and cap what can be shown, so those "
            f"{w['samples']} are not counted as needing more fields."
        )
    if any(block.get("unchosen") for block in blocks):
        reasons.append(
            f"Not checked: on the other {w['units']} no number of fields served enough "
            f"{w['samples']} to be tried on the one left out."
        )
    return reasons


def protocol_full_parts(result: ProtocolResult, indent: int = 0) -> tuple[list[str], list[str]]:
    """The full form's lines under ``indent`` and its caveats (the module)."""
    w = _words(result)
    blocks = protocol_answers(result)
    lines: list[str] = []
    for block in blocks:
        stratum = group_label(block["stratum"]) if block["stratum"] else "all samples"
        header = (
            f"{stratum}: fields per {w['sample']} for {block['label']} "
            f"({count(block['samples'], w['sample'], w['samples'])} in "
            f"{count(block['units'], w['unit'], w['units'])}; a {w['sample']} counts when at "
            f"least {w['threshold']} of random acquisitions were enough)"
        )
        if lines:
            lines.append("")
        lines.append(wrap(header, indent, hang=indent + 2))
        columns = ["fields", "enough for", f"per {w['unit']}", "plausible range", "note"]
        cells = block["cells"]
        if all(cell[3] == "–" for cell in cells):
            # No interval (a single unit): no column that would read as one.
            columns.pop(3)
            cells = [cell[:3] + cell[4:] for cell in cells]
        lines.append(table(cells, columns, indent=indent + 2, align="rlll"))
        lines += [wrap(note, indent + 2, hang=indent + 4) for note in block["notes"]]
    return lines, _few(blocks, w)


def protocol_summary(result: ProtocolResult, detail: str = "short") -> str:
    """ "For experiments like this one": short, one table, or full, per criterion (the module)."""
    if check_detail(detail) == "full":
        lines, cautions = protocol_full_parts(result)
    else:
        lines, cautions = protocol_short_parts(result, indent=0)
    if not lines:
        # Every sample skipped upstream (or none given): say so, never an empty answer.
        return "Not assessed: no sample had a reliability curve to count."
    return "\n".join([*lines, *caveat_lines(cautions, indent=0, detail=detail)])


def _none_judged(sample: str, beyond: int, undefined: int) -> str:
    """Why no sample could be judged at one depth: too few fields, or an undefined criterion."""
    if not undefined:
        return f"no {sample} had that many fields to judge"
    if not beyond:
        return (
            f"no {sample} could be judged at this depth (the criterion could not be judged "
            "against their held-back fields)"
        )
    return (
        f"no {sample} could be judged at this depth ({beyond} had fewer fields than that; "
        f"for {undefined} the criterion could not be judged against their held-back fields)"
    )


def _needed(
    hard: pd.DataFrame, short: pd.DataFrame, sample: str, samples_word: str
) -> tuple[list[str], str]:
    """The median fields needed per sample, a sample that never got there counting as more,
    as a sentence and as the short table's cell.

    Over the samples that got there (``fields_needed``) and those that need
    more fields than could be judged (``short``): these count as beyond every
    listed count (``_text.median_beyond``), so the median is not that of the
    easy samples only. Samples capped by their reference or never judged are
    left out, as the summary says separately.
    """
    known = hard["fields_needed"].dropna().astype(int)
    if not len(known) and not len(short):
        return [], "–"
    pool = [*known.tolist(), *([None] * len(short))]
    median = median_beyond(pool)
    spread = f"range {known.min()} to {known.max()}" if len(known) else ""
    if median is None:
        reached = f"{len(known)} of {count(len(pool), sample, samples_word)}"
        line = (
            f"Fields needed per {sample}: median not reached within the listed counts "
            f"(only {reached} got there{'; ' + spread if spread else ''})."
        )
        return [line], f"not reached ({len(known)} of {len(pool)} got there)"
    others = f"; {len(short)} needing more counted as more" if len(short) else ""
    cell_range = span(known.tolist()) if len(known) else ""
    more = f"; {len(short)} more" if len(short) else ""
    if cell_range == number(median) and not more:
        cell_range = ""  # every sample needed the same number: the range adds nothing
    cell = f"{number(median)} ({cell_range}{more})" if cell_range or more else number(median)
    return [f"Fields needed per {sample}: median {number(median)}, {spread}{others}."], cell


__all__ = [
    "FEW_UNITS",
    "protocol_answers",
    "protocol_full_parts",
    "protocol_short_parts",
    "protocol_summary",
    "served_cell",
    "served_from",
]
