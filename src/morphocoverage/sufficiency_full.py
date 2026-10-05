"""The full form of the sufficiency report: every number, group by group, each caveat once.

``SufficiencyReport.summary(detail="full")`` keeps every number of the
sentences in the ``answer`` columns of ``enough`` and ``next_fov`` (which
keep those sentences), laid out so that nothing repeats:

- the rules across groups, as in the short form (``sufficiency_short``);
- per analysis group (``MOCK · 1A · B6``, the columns named once): a table of
  the rules, by their readable names with their config ids (whether each was
  met by its own threshold, where it was first met in the recorded order and
  in the random orders, with the Monte Carlo SE of the median and the
  10-90% range, and the fields imaged beyond that point), the numbers of
  each caution that differ between groups, and the next-field answers with
  their numbers in a few lines;
- every caveat once, as ⚠️ lines at the end: the rules' cautions with each
  number as a range over the groups, the groups named when a caution holds
  in some only, what every stop rests on, and what every novelty bound
  assumes.

Shared-private: reads the helpers of ``sufficiency_short``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ._caveats import assumption, caveat_lines
from ._style import (
    group_columns,
    group_label,
    number_pattern,
    rule_label,
    table,
    varying_numbers,
    wrap,
)
from ._wording import count, listing, number, percent, rows_of
from .bounds import EXTRA_TRIALS_LIMIT
from .sufficiency_short import (
    _group,
    _novelty_lines,
    _present,
    _rule_cautions,
    _rules_table,
    _warning_list,
    rule_tallies,
)


def _stop_cell(stops: pd.DataFrame, row: Mapping[str, Any], kind: str) -> str | None:
    """Where one kind of order first met this group's rule; ``None`` if not replayed."""
    rows = stops[stops["config_id"].eq(row["config_id"]) & stops["order_kind"].eq(kind)]
    if rows.empty:
        return None
    stop = rows.iloc[0]
    orders, met = int(stop["orders"]), int(stop["orders"]) - int(stop["never_stopped"])
    if not met:
        return "never"
    median = number(stop["stop_fov_median"])
    if kind == "recorded" and orders == 1:
        return median
    se = stop.get("stop_fov_median_mc_se")
    se = "" if se is None or pd.isna(se) else f"{number(se)}; "
    text = f"{median} ({se}{number(stop['stop_fov_p10'])}-{number(stop['stop_fov_p90'])})"
    return text if met == orders else f"{text}; {met} of {orders} orders"


def _rule_names(report: Any) -> str:
    """Each readable rule name once with its strategy, settings and config id."""
    unit = getattr(report, "observation_unit", None)
    rows = report.enough.drop_duplicates("config_id")
    names = [rule_label(row, unit, detail="full") for _, row in rows.iterrows()]
    return wrap(f"Rules: {'; '.join(names)}.", 0, hang=2)


def _group_table(report: Any, group: Mapping[str, Any], indent: int) -> list[str]:
    """One group's rules: enough, first met after, fields beyond."""
    if report.enough.empty:
        return []
    enough = rows_of(report.enough, group) if len(group) else report.enough
    stops = rows_of(report.stops, group) if len(group) and len(report.stops) else report.stops
    if enough.empty:
        return []
    kinds = [k for k in ("recorded", "permuted") if k in set(stops.get("order_kind", []))]
    header = ["rule", "enough", *(("recorded",) if "recorded" in kinds else ())]
    if "permuted" in kinds:
        orders = int(stops[stops["order_kind"].eq("permuted")]["orders"].max())
        header.append(f"{count(orders, 'random order')}: median (MC SE; 10-90%)")
    header.append("beyond")
    unit = getattr(report, "observation_unit", None)
    rows = []
    for _, row in enough.iterrows():
        cells = [rule_label(row, unit), row["sampled_enough"]]
        cells += [_stop_cell(stops, row, kind) for kind in kinds]
        beyond = row.get("fovs_beyond_enough")
        cells.append(number(beyond) if _present(beyond) else None)
        rows.append(cells)
    return [table(rows, header, indent=indent)]


def _group_notes(
    report: Any,
    group: Mapping[str, Any],
    varying: Mapping[tuple, str],
    shared: Mapping[str, str],
) -> list[str]:
    """One group's cautions by their numbers, its reasons (those ``shared`` by every group,
    said once above the groups, left out), and its flagged order check."""
    if report.final.empty:
        return []
    final = rows_of(report.final, group) if len(group) else report.final
    unit = getattr(report, "observation_unit", None)
    notes, orders = [], {}
    for _, row in final.iterrows():
        label = rule_label(row, unit)
        numbers = [
            varying.get((row["config_id"], group_label(group), text)) for text in _warning_list(row)
        ]
        numbers = [n for n in numbers if n]
        if numbers:
            notes.append(f"{label}: {'; '.join(numbers)}")
        reason = row.get("undefined_reason")
        if pd.isna(row["verdict"]) and isinstance(reason, str) and shared.get(label) != reason:
            notes.append(f"{label}: undefined ({reason})")
        reason = row.get("insufficient_reason")
        if row["verdict"] == "continue" and isinstance(reason, str) and reason:
            notes.append(f"{label}: cannot be judged yet ({reason})")
        caution = row.get("order_caution")
        # The order check speaks of a stop: only for rules that stopped, said once for all.
        if (
            isinstance(caution, str)
            and caution.startswith("caution: ")
            and row["verdict"] == "stop"
        ):
            orders.setdefault(caution.removeprefix("caution: "), []).append(label)
    for caution, labels in orders.items():
        notes.append(f"the order check, for {listing(labels, 4)}: {caution}")
    return notes


def _shared_undefined(report: Any) -> dict[str, str]:
    """Per rule label, the reason it is undefined when that reason is the same in every
    group: said once, not in each group."""
    final = report.final
    if final.empty or "undefined_reason" not in final:
        return {}
    unit = getattr(report, "observation_unit", None)
    labels = final.apply(lambda row: rule_label(row, unit), axis=1)
    shared = {}
    for label, rows in final.groupby(labels, sort=False):
        reasons = rows["undefined_reason"]
        if rows["verdict"].isna().all() and reasons.notna().all() and reasons.nunique() == 1:
            shared[str(label)] = str(reasons.iloc[0])
    return shared


def _varying(report: Any) -> dict[tuple, str]:
    """Each rule's caution numbers that differ between groups, by rule, group and caution."""
    final, keys = report.final, list(report.group_cols)
    found: dict[tuple, list[tuple[str, str]]] = {}
    for _, row in final.iterrows():
        for text in _warning_list(row):
            where = group_label(_group(row, keys))
            found.setdefault((row["config_id"], number_pattern(text)), []).append((where, text))
    varying = {}
    for (config, _), pairs in found.items():
        for (where, text), numbers in zip(
            pairs, varying_numbers([t for _, t in pairs]), strict=True
        ):
            varying[(config, where, text)] = numbers
    return varying


def _next_field_lines(row: Mapping[str, Any], target: float | None) -> list[str]:
    """One group's next-field answers, every number of its ``answer`` sentences kept."""
    if not _present(row["chance_new"]):
        return [f"One more field: not estimable ({row['reason']})."]
    fields, confidence = int(row["fovs_imaged"]), percent(row["confidence"])
    empty = int(row.get("fovs_without_observations", 0) or 0)
    without = f" ({count(empty, 'field')} without observations)" if empty else ""
    chance = (
        f"One more field: chance {percent(row['chance_new'])} (nominal {confidence} bound "
        f"{percent(row['chance_new_at_most'])}); {int(row['fovs_with_new'])} of "
        f"{count(fields, 'field')}{without} showed a state counted as new."
    )
    lines = [chance]
    successes = -math.log(1 - float(row["confidence"]))
    unseen = (
        f"Not yet seen: at most {percent(row['unseen_state_max_frequency'])} of fields "
        f"({confidence} confidence, about {successes:.2g} / {fields})"
    )
    technical, share = row.get("technical_repeats"), row.get("unseen_state_max_technical_share")
    if _present(technical) and int(technical) > 1 and _present(share):
        unseen += (
            f"; over its {int(technical)} technical repeats, at most {percent(share)} of them "
            f"({int(technical)} independent technical repeats)"
        )
    lines.append(f"{unseen}.")
    lines += _few_line(row, confidence)
    if target is not None:
        lines.append(_needed_line(row, target))
    return lines


def _few_line(row: Mapping[str, Any], confidence: str) -> list[str]:
    """States seen in few fields, with the planning numbers when there are some."""
    few = row.get("few_fovs")
    if not _present(few):
        return []
    seen = row.get("fovs_with_state_in_few")
    if not _present(seen):
        return [f"Seen in few fields: not estimable ({row['few_reason']})."]
    fewest = int(row["fewest_recurrent_fovs"])
    span = f"{fewest}" if fewest == int(few) else f"{fewest} to {int(few)}"
    text = (
        f"Seen in few fields: {int(seen)} of {count(int(row['fovs_imaged']), 'field')} hold a "
        f"state seen in {span} fields"
    )
    if int(seen):
        text += (
            f"; together {percent(row['few_state_incidence'])} of fields ({confidence} interval "
            f"{percent(row['few_state_incidence_low'])}-{percent(row['few_state_incidence_high'])})"
            f": to see it once with {confidence} probability, about "
            f"{int(row['fovs_to_see_again'])} independent representative fields, up to "
            f"{int(row['fovs_to_see_again_low_end'])} at the low end of the interval"
        )
        if int(seen) >= 2 * fewest and _present(row.get("fovs_to_see_fewest")):
            text += (
                f"; they may hold several states, each rarer: one seen in only {fewest} fields "
                f"needs about {int(row['fovs_to_see_fewest'])}"
            )
    return [f"{text}."]


def _needed_line(row: Mapping[str, Any], target: float) -> str:
    """The fields still needed to rule out unseen states and to bring the bound below target."""
    goal = percent(target)
    extra = row.get("extra_fovs_to_rule_out_target")
    rule_out = (
        f"unseen states in more than {goal} of fields already ruled out"
        if _present(extra) and int(extra) == 0
        else f"at least {count(int(extra), 'more field')} with nothing new to rule out unseen "
        f"states in more than {goal} of fields"
        if _present(extra)
        else f"more than {count(EXTRA_TRIALS_LIMIT, 'more field')} with nothing new to rule out "
        f"unseen states in more than {goal} of fields"
    )
    if float(row["chance_new_at_most"]) <= target:
        bound = f"the next-field bound is already below {goal}"
    else:
        rate = row.get("extra_fovs_at_current_rate")
        at_rate = (
            f"{number(rate)} at the current rate of new fields"
            if _present(rate)
            else "never, at the current rate of new fields"
        )
        best = row.get("extra_fovs_if_nothing_new")
        least = (
            f"at least {count(int(best), 'more field')}"
            if _present(best)
            else f"more than {count(EXTRA_TRIALS_LIMIT, 'more field')}"
        )
        bound = (
            f"to bring the next-field bound below {goal}, {least} if none shows "
            f"anything new ({at_rate})"
        )
    return f"Fields still needed: {'; '.join(p for p in (rule_out, bound) if p)}."


def _check_line(report: Any, group: Mapping[str, Any]) -> list[str]:
    """The shuffle check of the prediction with the most fields, when it was run."""
    calibrated = report.calibration
    if calibrated is None or calibrated.empty:
        return []
    rows = rows_of(calibrated, group) if len(group) else calibrated
    rows = rows[rows["draws"] > 0]
    if rows.empty:
        return []
    best = rows.sort_values("fovs_imaged").iloc[-1]
    check = (
        f"Check on this dataset: from {int(best['fovs_imaged'])} random fields, predicted "
        f"{percent(best['predicted'])}, actual {percent(best['actual'])}; the bound held in "
        f"{percent(best['bound_held'])} of {int(best['draws'])} draws."
    )
    return [check]


def _full_cautions(report: Any, target: float | None) -> list[str]:
    """What the per-group numbers rest on, once each (beyond the short form's caveats)."""
    cautions = []
    frame = report.next_fov
    if frame.empty:
        return cautions
    cautions.append(
        assumption("a state below the not-yet-seen bound can remain: rarer states can remain")
    )
    if "few_fovs" in frame and frame["few_fovs"].map(_present).any():
        cautions += [
            assumption(
                "a state seen in few fields (fields counted as new are not among them) is not "
                "new to the next field but is not established either: it counts as recurrent, "
                "so look at those fields before calling the sample settled"
            ),
            assumption("fields that cluster a state need more fields than the planning numbers"),
        ]
    if report.calibration is not None and not report.calibration.empty:
        cautions.append(
            assumption(
                "the check on this dataset draws random subsets, which break the acquisition "
                "order and the neighbours, so it cannot reveal neighbour correlation"
            )
        )
    return cautions


def full_summary(report: Any, target: float | None = None) -> str:
    """The full form of a ``SufficiencyReport`` (see the module)."""
    keys = list(report.group_cols)
    target = target if target is not None else getattr(report, "novelty_target", None)
    tallies = rule_tallies(report)
    shared = _shared_undefined(report) if tallies else {}
    blocks = []
    head = [f"Analysis groups: {group_columns(keys)}."] if keys else []
    if tallies:
        head += _rules_table(report, tallies, 0)
        head.append(_rule_names(report))
        head += [
            wrap(f"{label} is undefined in every group: {reason}.", 0, hang=2)
            for label, reason in shared.items()
        ]
        head.append(
            wrap(
                "Per group below: whether each rule was met by its own threshold, where it was "
                "first met, and the fields imaged beyond that point (beyond: a median over the "
                "random orders, or the recorded one without them; only where the acquired "
                "fields are enough).",
                0,
                hang=2,
            )
        )
    if head:
        blocks.append("\n".join(head))
    tables = [t[keys] for t in (report.enough, report.next_fov) if not t.empty]
    groups = pd.concat(tables).drop_duplicates() if tables and keys else None
    rows = [()] * bool(tables) if groups is None else groups.itertuples(index=False, name=None)
    varying = _varying(report) if len(report.final) else {}
    for values in rows:
        group = dict(zip(keys, values, strict=True))
        acquired = rows_of(report.enough, group) if len(report.enough) and keys else report.enough
        fields = int(acquired["fovs_acquired"].iloc[0]) if len(acquired) else None
        title = group_label(group) + (f" — {count(fields, 'field')} acquired" if fields else "")
        lines = [f"{title}:", *_group_table(report, group, 2)]
        notes = _group_notes(report, group, varying, shared)
        if notes:
            lines.append(wrap(f"Cautions here: {'; '.join(notes)}.", 2, hang=4))
        outlook = rows_of(report.next_fov, group) if len(report.next_fov) and keys else None
        outlook = report.next_fov if outlook is None else outlook
        if len(outlook):
            answers = _next_field_lines(outlook.iloc[0], target) + _check_line(report, group)
            lines += [wrap(line, 2, hang=4) for line in answers]
        blocks.append("\n".join(lines))
    if report.novelty_definition and not report.next_fov.empty:
        blocks.append(wrap(f"'New' means {report.novelty_definition}.", 0, hang=2))
    _, novelty_cautions = _novelty_lines(report, target)
    cautions = [
        *_full_cautions(report, target),
        *_rule_cautions(report, tallies, name_groups=True),
        *novelty_cautions,
    ]
    blocks.append("\n".join(caveat_lines(cautions, indent=0, detail="full")))
    return "\n\n".join(block for block in blocks if block)


__all__ = ["full_summary"]
