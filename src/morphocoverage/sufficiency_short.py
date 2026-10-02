"""The short form of the sufficiency report: the rules in one table, the novelty in a few lines.

``SufficiencyReport.summary()`` (``detail="short"``) states, across the
analysis groups at once:

- **Did I image enough?** One row per rule, by its readable name
  (``_style.rule_label``): in how many groups its own threshold was met by
  the acquired fields, and after how many fields it was first met in the
  recorded order (the range over groups) and in the random orders (the
  median of the groups' medians, with the lowest 10% point to the highest 90%
  point of any group). With rules that differ between groups, one table per
  group.
- **Would one more field show something new?** The chance and its nominal
  bound, the states not yet seen, the states seen in few fields and the
  fields still needed, each as one line over the groups (ranges, and the
  groups that differ by name).
- **Every caveat once**, as a ⚠️ line at the end: the rules' own cautions
  (a caution alike in several groups but for its numbers becomes one line
  with each number as a range, ``_style.merge_numbers``), what every stop
  rests on (the order check or the random-sample assumption), and what every
  novelty bound assumes.

``rule_tallies`` gives the per-rule counts the report's status reads
(``retrospective.report_status``). The full form, every number group by
group, is ``sufficiency_full``; the sentences per group and rule are the
``answer`` columns (``sufficiency_text``).
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ._caveats import assumption, caveat_lines, is_assumption
from ._style import (
    among,
    group_columns,
    group_label,
    merge_numbers,
    number_pattern,
    rule_label,
    span,
    table,
    wrap,
)
from ._wording import count, listing, number, percent, plural, rows_of
from .order_check import order_caveat
from .schema import unit_plural

_MET = ("yes", "target reached")


def _group(row: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def _present(value: Any) -> bool:
    """Whether a table cell holds a value (not None, NaN or <NA>)."""
    return (
        value is not None
        and value is not pd.NA
        and not (isinstance(value, float) and pd.isna(value))
    )


def _span(values: Sequence[float], format_one: Any = number) -> str:
    """``8`` or ``3-5``; ``4%`` or ``0-20%`` for ``percent`` (``_style.span``)."""
    return span(values, "percent" if format_one is percent else "number")


def _more(values: Sequence[float]) -> str:
    """``1 more field``, ``31-96 more fields``."""
    text = _span(values)
    return f"{text} more field" if text == "1" else f"{text} more fields"


def _warning_list(row: Mapping[str, Any]) -> list[str]:
    """A rule's cautions one by one (tables written before ``warning_list`` join them)."""
    listed = row.get("warning_list")
    if isinstance(listed, list | tuple):
        return [text if isinstance(text, str) else str(text) for text in listed if text]
    joined = row.get("warnings")
    return [joined] if isinstance(joined, str) and joined else []


def _data_warnings(row: Mapping[str, Any]) -> list[str]:
    """A rule's cautions about these data: its warnings but the standing assumptions it
    states beside every value (``_style.Assumption``), which never raise a status."""
    return [text for text in _warning_list(row) if not is_assumption(text)]


def _ever_met(stops: pd.DataFrame, row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    """Whether any replayed order of this group met this rule's criterion."""
    if stops.empty:
        return False
    own = rows_of(stops, {**_group(row, keys), "config_id": row["config_id"]})
    return bool((own["orders"] > own["never_stopped"]).any())


def rule_tallies(report: Any) -> list[dict[str, Any]]:
    """Per rule (first-seen order): its label and in how many groups it was met, undefined,
    met and then lost, carried a caution of its own or was flagged by the order check.

    ``data_rule`` is false for a fixed amount of effort (fields, objects, area),
    whose "met" says its target was reached, not that the data were enough.
    """
    enough, final, keys = report.enough, report.final, list(report.group_cols)
    if enough.empty:
        return []
    unit = getattr(report, "observation_unit", None)
    tallies = []
    for config, rows in enough.groupby("config_id", sort=False):
        own = final[final["config_id"].eq(config)] if len(final) else final
        effort = bool(own["effort_rule"].any()) if "effort_rule" in own else False
        first = rows.iloc[0]
        met = rows["sampled_enough"].isin(_MET)
        warned = flagged = lost = 0
        for _, row in own.iterrows():
            stopped = row["verdict"] == "stop"
            warned += bool(stopped and _data_warnings(row))
            flagged += bool(stopped and _flagged(row))
            lost += bool(row["verdict"] == "continue" and _ever_met(report.stops, row, keys))
        tallies.append(
            {
                "config_id": config,
                "label": rule_label(first, unit),
                "data_rule": not effort,
                "groups": len(rows),
                "met": int(met.sum()),
                "undefined": int(rows["sampled_enough"].eq("undefined").sum()),
                "lost": lost,
                "warned": warned,
                "flagged": flagged,
            }
        )
    return tallies


def _flagged(row: Mapping[str, Any]) -> bool:
    """Whether the order check flagged the group's fields for this rule's stop."""
    flagged = row.get("order_flagged")
    if isinstance(flagged, bool):
        return flagged
    caution = row.get("order_caution")
    return isinstance(caution, str) and caution.startswith("caution:")


def _stops_cell(stops: pd.DataFrame, config: str, kind: str, groups: int) -> str | None:
    """Where one kind of order first met the rule, over the groups; ``None`` if not replayed."""
    rows = stops[stops["config_id"].eq(config) & stops["order_kind"].eq(kind)]
    if rows.empty:
        return None
    met = rows["orders"] - rows["never_stopped"]
    reached = rows[met > 0]
    if reached.empty:
        return "never"
    if kind == "recorded":
        text = _span(reached["stop_fov_median"])
        return text if len(reached) == groups else f"{text} ({len(reached)} of {groups})"
    median = statistics.median(reached["stop_fov_median"].astype(float))
    text = (
        f"{number(median)} ({number(reached['stop_fov_p10'].min())}-"
        f"{number(reached['stop_fov_p90'].max())})"
    )
    total, stopped = int(rows["orders"].sum()), int(met.sum())
    if stopped == total:
        return text
    # Most orders never met it: said first, the fields only for those that did (audit).
    that = "the order" if stopped == 1 else f"the {stopped}"
    return f"never in {total - stopped} of {total} orders; {text} in {that} that met it"


def _rules_table(report: Any, tallies: list[dict[str, Any]], indent: int) -> list[str]:
    """The rules across groups: met in, first met after (recorded; random orders)."""
    groups = max(t["groups"] for t in tallies)
    kinds = [k for k in ("recorded", "permuted") if k in set(report.stops.get("order_kind", []))]
    header = [
        "rule",
        "met with the fields acquired",
        *(("fields when first met",) if "recorded" in kinds else ()),
    ]
    if "permuted" in kinds:
        header.append("fields needed in random orders: median (10-90%)")
    rows = []
    for tally in tallies:
        if tally["groups"] == 1:
            one = report.enough[report.enough["config_id"].eq(tally["config_id"])]
            met = str(one["sampled_enough"].iloc[0])
        elif tally["undefined"] == tally["groups"]:
            met = "not defined here"
        else:
            met = f"{tally['met']} of {tally['groups']}" + (
                f" ({tally['undefined']} not defined)" if tally["undefined"] else ""
            )
        cells = [tally["label"], met]
        cells += [_stops_cell(report.stops, tally["config_id"], k, tally["groups"]) for k in kinds]
        rows.append(cells)
    where = "the group" if groups == 1 else f"{groups} groups"
    pad = " " * indent
    lines = [f"{pad}Each rule by its own threshold (not a held-out check), in {where}:"]
    lines.append(table(rows, header, indent=indent + 2))
    if kinds:
        spreads = []
        if groups > 1 and "recorded" in kinds:
            spreads.append("recorded: the range over groups")
        if groups > 1 and "permuted" in kinds:
            spreads.append(
                "random: the median of the groups' medians, 10-90% from the lowest to the "
                "highest group"
            )
        first = (
            ["Fields when first met: in the order acquired, when it first said enough"]
            if "recorded" in kinds
            else ["Fields when first met in random orders"]
        )
        text = "; ".join([*first, *spreads])
        lines.append(wrap(f"{text}.", indent, hang=indent + 2))
    return lines


def _labels(rows: pd.DataFrame, keys: Sequence[str], limit: int = 3) -> str:
    """The groups of ``rows`` by their labels, ``listing`` style."""
    return listing([group_label(_group(row, keys)) for _, row in rows.iterrows()], limit=limit)


def _novelty_lines(report: Any, target: float | None) -> tuple[list[str], list[str]]:
    """The next-field answers over the groups, and the caveats they rest on."""
    frame, keys = report.next_fov, list(report.group_cols)
    if frame.empty:
        return [], []
    groups = len(frame)
    lines, cautions = [], []
    estimable = frame[frame["chance_new"].map(_present)]
    missing = frame[~frame["chance_new"].map(_present)]
    if len(missing):
        reasons = listing(sorted({str(r) for r in missing["reason"]}))
        lines.append(f"Not estimable in {among(len(missing), groups)}: {reasons}.")
    if estimable.empty:
        return lines, cautions
    confidence = percent(float(estimable["confidence"].iloc[0]))
    bound = _span(estimable["chance_new_at_most"], percent)
    new = estimable[estimable["fovs_with_new"].astype(int) > 0]
    if new.empty:
        where = "" if groups == 1 else f", in {among(len(estimable), groups)}"
        lines.append(
            f"One more field: no field showed a state counted as new{where}; nominal "
            f"{confidence} bound for the next field {bound}."
        )
    elif groups == 1:
        shown = count(int(new["fovs_with_new"].iloc[0]), "field")
        lines.append(
            f"One more field: {shown} showed a state counted as new; chance "
            f"{percent(new['chance_new'].max())}, nominal {confidence} bound {bound}."
        )
    else:
        named = listing(
            [
                f"{group_label(_group(row, keys))} ({count(int(row['fovs_with_new']), 'field')})"
                for _, row in new.iterrows()
            ],
            limit=3,
        )
        lines.append(
            f"One more field: fields with a state counted as new in "
            f"{among(len(new), len(estimable))}: {named}; chance up to "
            f"{percent(new['chance_new'].max())}, nominal {confidence} bound {bound}."
        )
    unseen = _span(estimable["unseen_state_max_frequency"], percent)
    lines.append(
        f"Not yet seen: a state no field has shown is in at most {unseen} of fields "
        f"({confidence} confidence); rarer states can remain."
    )
    shares = estimable.get("unseen_state_max_technical_share")
    technical = estimable[shares.map(_present)] if shares is not None else estimable.iloc[:0]
    if len(technical):
        share = _span(technical["unseen_state_max_technical_share"], percent)
        lines.append(
            f"Pooled technical repeats ({among(len(technical), groups)}): a state none of them "
            f"showed is in at most {share} of technical repeats ({confidence} confidence)."
        )
        cautions.append(
            "fields of one technical repeat are not independent draws: where a group pools "
            "technical repeats, the field bound is optimistic"
        )
    lines += _few_lines(estimable, keys, confidence)
    if target is not None:
        # The fields still needed are projections: the full form gives them, the short form
        # points there (they read as a plan, not as what these fields showed).
        lines.append(
            "How many more fields would rule out unseen states, or bring the next-field bound "
            f"below {percent(target)}, is in the full report (report_full.txt)."
        )
        cautions.append(
            assumption(
                "every novelty bound holds at a number of fields fixed in advance, not at the "
                "first field where it passes: stopping there is premature more often than "
                f"{percent(1 - float(estimable['confidence'].iloc[0]))} of the time"
            )
        )
    cautions[:0] = [
        assumption(
            "the next-field bound is approximate: the leave-one-field-out outcomes share their "
            "reference fields"
        ),
        assumption(
            "the novelty bounds treat fields as independent, representative draws: neighbouring "
            "fields acquired in a raster over patchy states make them too low"
        ),
    ]
    m = estimable["min_novel_observations"].dropna()
    words = unit_plural(getattr(report, "observation_unit", None) or "object")
    if len(m) and int(m.max()) > 1:
        cautions.append(
            assumption(
                f"a state carried by fewer than {int(m.max())} similar {words} per field is "
                "never counted, however many fields hold it"
            )
        )
    empty = estimable[estimable["fovs_without_observations"].fillna(0).astype(int) > 0]
    if len(empty):
        cautions.append(
            f"in {among(len(empty), groups)} the field counts include fields without "
            "observations, which cannot show a state"
        )
    return lines, cautions


def _few_lines(frame: pd.DataFrame, keys: Sequence[str], confidence: str) -> list[str]:
    """States seen in few fields: which groups hold one, and what seeing one again takes."""
    if "few_fovs" not in frame or not frame["few_fovs"].map(_present).any():
        return []
    rows = frame[frame["fovs_with_state_in_few"].map(_present)]
    few = int(frame["few_fovs"].dropna().iloc[0])
    lines = []
    unjudged = len(frame) - len(rows)
    if unjudged:
        lines.append(f"Seen in few fields: not estimable in {among(unjudged, len(frame))}.")
    if rows.empty:
        return lines
    fewest = int(rows["fewest_recurrent_fovs"].min())
    span = f"{fewest}" if fewest == few else f"{fewest} to {few}"
    holding = rows[rows["fovs_with_state_in_few"].astype(int) > 0]
    single = len(rows) == 1
    if holding.empty:
        who = "no field holds" if single else "no group holds"
        lines.append(f"Seen in few fields: {who} a state seen in only {span} fields.")
        return lines
    who = (
        f"{count(int(holding['fovs_with_state_in_few'].iloc[0]), 'field')} hold"
        if single
        else f"{among(len(holding), len(rows))} hold"
    )
    where = "" if single else f" ({_labels(holding, keys)})"
    lines.append(
        f"Seen in few fields: {who} a state seen in only {span} fields{where}; look at those "
        "fields before calling the sample settled."
    )
    return lines


def _rule_cautions(
    report: Any, tallies: list[dict[str, Any]], name_groups: bool = False
) -> list[str]:
    """Each rule's cautions once over the groups, then what every stop rests on;
    ``name_groups`` names the groups a caution holds in when it is not all of them."""
    final, keys = report.final, list(report.group_cols)
    if final.empty:
        return []
    unit = getattr(report, "observation_unit", None) or "object"
    cautions, stops = [], []
    for tally in tallies:
        config, label = tally["config_id"], tally["label"]
        own = final[final["config_id"].eq(config)]
        groups = len(own)
        if not tally["data_rule"]:
            cautions.append(
                assumption(
                    f"{label}: a fixed amount of effort reaching its target says nothing about "
                    "what the data show"
                )
            )
        if own["assumes_spatial_uniformity"].astype(bool).any():
            cautions.append(
                assumption(
                    f"{label} treats {plural(unit)} as independent draws: it assumes a "
                    "spatially uniform sample"
                )
            )
        found: dict[str, list[tuple[str, str]]] = {}
        for _, row in own.iterrows():
            for text in _warning_list(row):
                where = group_label(_group(row, keys))
                found.setdefault(number_pattern(text), []).append((where, text))
        for pairs in found.values():
            where = among(len(pairs), groups)
            if name_groups and len(pairs) < groups:
                where += f" ({listing([w for w, _ in pairs], limit=5)})"
            merged = f"{label}, in {where}: {merge_numbers([t for _, t in pairs])}"
            standing = all(is_assumption(t) for _, t in pairs)
            cautions.append(assumption(merged) if standing else merged)
        undefined = own[own["verdict"].isna()]["undefined_reason"].dropna().astype(str)
        waiting = own[own["verdict"].eq("continue")]["insufficient_reason"].dropna().astype(str)
        for reasons, what in ((waiting, "cannot be judged yet"), (undefined, "is undefined")):
            reasons = [r for r in reasons if r]
            if reasons:
                cautions.append(
                    f"{label} {what} in {among(len(reasons), groups)}: "
                    f"{listing(sorted(set(reasons)))}"
                )
        if tally["lost"]:
            cautions.append(
                f"{label} was met and then lost as more fields arrived, in "
                f"{among(tally['lost'], groups)}"
            )
        for _, row in own.iterrows():
            text = row.get("order_caution")
            if (
                isinstance(text, str)
                and text
                and (row["verdict"] == "stop" or _ever_met(report.stops, row, keys))
            ):
                stops.append((group_label(_group(row, keys)), text))
    return cautions + _stop_cautions(stops)


def _stop_cautions(stops: list[tuple[str, str]]) -> list[str]:
    """What the stops rest on, once per kind of sentence, with the groups it concerns."""
    groups = list(dict.fromkeys(label for label, _ in stops))
    by_kind: dict[str, list[tuple[str, str]]] = {}
    for label, text in stops:
        by_kind.setdefault(number_pattern(text), []).append((label, text))
    lines = []
    for pairs in by_kind.values():
        text = merge_numbers([t for _, t in pairs])
        mine = list(dict.fromkeys(label for label, _ in pairs))
        if text.startswith("caution: "):
            flagged = text.removeprefix("caution: ")
            lines.append(f"order check in {listing(mine, limit=3)}: {flagged}")
            continue
        caveat = order_caveat(text)
        where = "" if len(mine) == len(groups) else f" in {listing(mine, 3)}"
        if text.startswith("treats"):
            caveat = caveat.replace("so every stop", f"so every stop{where}", 1)
        elif where:
            caveat = type(caveat)(f"{listing(mine, 3)}: {caveat}")
        lines.append(caveat)
    return lines


def short_parts(
    report: Any, target: float | None = None, indent: int = 0
) -> tuple[list[str], list[str]]:
    """The short form's lines (wrapped under ``indent``) and its caveats, apart, so a report
    can put the caveats of several analyses together at the end of one section."""
    keys = list(report.group_cols)
    target = target if target is not None else getattr(report, "novelty_target", None)
    pad = " " * indent
    lines = [f"{pad}Analysis groups: {group_columns(keys)}."] if keys else []
    tallies = rule_tallies(report)
    if tallies:
        lines += _rules_table(report, tallies, indent)
    novelty, novelty_cautions = _novelty_lines(report, target)
    lines += [wrap(line, indent, hang=indent + 2) for line in novelty]
    definition = _new_in_short(report)
    if definition and novelty:
        lines.append(wrap(definition, indent, hang=indent + 2))
    return lines, [*_rule_cautions(report, tallies), *novelty_cautions]


def _new_in_short(report: Any) -> str | None:
    """What counts as new, in one plain line: its exact definition is the full form's."""
    if not report.novelty_definition:
        return None
    frame = report.next_fov
    m = frame["min_novel_observations"].dropna() if "min_novel_observations" in frame else []
    words = unit_plural(getattr(report, "observation_unit", None) or "object")
    least = f"at least {int(max(m))} similar {words}" if len(m) else f"similar {words}"
    return f"'New': {least} that no other field shows; the exact definition is in the full report."


def short_summary(report: Any, target: float | None = None) -> str:
    """The short form of a ``SufficiencyReport`` (see the module)."""
    lines, cautions = short_parts(report, target)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0)])


__all__ = ["rule_tallies", "short_parts", "short_summary"]
