"""The Foundation's checks in words (private): field variability and the object-count rules.

``FOVVariability``, ``ObjectRuleCheck`` and ``DescriptorSpreadCheck`` print
through this module, short and full (``summary(detail=...)``):

- **short**: one sentence with the answer over the analysis groups, then one
  aligned table (the groups' numbers for the field variability; one row per
  target or descriptor, with ranges over the groups, for the rules), and each
  caveat once at the end;
- **full**: every number of each group in one table, the columns named once,
  and each caveat once.

``*_parts(result, indent)`` return ``(lines, cautions)`` so that the report
can pool the caveats of the three checks at the end of its Foundation.
Shared-private: used by ``variability``, ``object_rules`` and
``retrospective.report_sampling``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .._caveats import caveat_lines
from .._style import among, group_columns, group_label, span, table, wrap
from .._wording import count, listing, number, percent, rate
from ..schema import unit_plural


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value)


def _group(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    return group_label({key: row[key] for key in keys})


def _one(value: float) -> str:
    """A ratio with one decimal, as the sentences wrote it (``1.0``, ``2.4``)."""
    return f"{float(value):.1f}"


def _ones(values: Sequence[float]) -> str:
    """The lowest and highest ratio with one decimal: ``1.0-3.4``, or ``1.2`` when alike."""
    low, high = _one(min(values)), _one(max(values))
    return low if low == high else f"{low}-{high}"


def _held(k: int, n: int) -> str:
    """``12 of 12`` (a table cell), ``yes`` or ``no`` for one group."""
    return ("yes" if k else "no") if n == 1 else f"{k} of {n}"


def _per_field(value: float) -> str:
    """Observations per field as read: ``12``, ``0.4``, and never ``0`` when some exist."""
    written = number(value)
    return f"{value:.1g}" if written == "0" and value > 0 else written


def _by_reason(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], what: str) -> list[str]:
    """``not estimated in MOCK · 1A · B6 and LATS1 · 1A · I23: <reason>``, one per reason."""
    reasons: dict[str, list[str]] = {}
    for row in rows:
        reasons.setdefault(str(row["status"]), []).append(_group(row, keys))
    return [
        f"{what} in {listing(list(dict.fromkeys(groups)), limit=3)}: {reason}"
        for reason, groups in reasons.items()
    ]


def join(
    lines: Sequence[str], cautions: Sequence[str], indent: int = 0, detail: str = "short"
) -> str:
    """The lines, then each caveat once under ``indent`` (``_caveats.caveat_lines``)."""
    return "\n".join([*lines, *caveat_lines(cautions, indent=indent, detail=detail)])


# --- Field variability --------------------------------------------------------


def _variability_rows(result: Any) -> tuple[list[dict], list[dict]]:
    rows = result.groups.to_dict("records")
    return rows, [row for row in rows if row["status"] == "ok"]


def _variability_cautions(result: Any, rows: list[dict], words: str) -> list[str]:
    keys = list(result.group_cols)
    cautions = _by_reason([r for r in rows if r["status"] != "ok"], keys, "not estimated")
    missing = [r for r in rows if int(r.get("missing_values") or 0)]
    if missing:
        numbers = [int(r["missing_values"]) for r in missing]
        where = f" per group in {among(len(missing), len(rows))}" if len(rows) > 1 else ""
        cautions.append(f"{words} with no value were left out: {span(numbers)}{where}")
    flat = [r for r in rows if r["status"] == "ok" and not _present(r["unusual_fov_share"])]
    if flat:
        cautions.append(
            f"the {words} do not vary within any field in "
            f"{listing([_group(r, keys) for r in flat], limit=3)}, so its fields cannot be "
            f"compared with random {words}"
        )
    return cautions


def variability_parts(result: Any, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form of ``fov_variability``: one answer sentence, one table of the groups."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows, ok = _variability_rows(result)
    feature = result.settings["feature"]
    lines = []
    if ok:
        effects = [float(r["design_effect"]) for r in ok]
        ratios = [float(r["error_ratio"]) for r in ok]
        median = float(pd.Series(effects).median())
        answer = (
            f"How variable are the fields ({feature})? A mean from whole fields has "
            f"{_ones(effects)} times the variance of one from as many random {words} (median "
            f"{_one(median)}, over {count(len(ok), 'group')}), so SD / sqrt(n) understates "
            f"its error up to {_one(max(ratios))}-fold."
        )
    else:
        answer = f"How variable are the fields ({feature})? Not estimated in any group."
    lines.append(wrap(answer, indent, hang=indent + 2))
    if ok:
        cells = [
            [
                _group(r, keys),
                int(r["fovs"]),
                int(r["observations"]),
                percent(r["icc"]),
                _one(r["design_effect"]),
                percent(r["unusual_fov_share"]) if _present(r["unusual_fov_share"]) else None,
            ]
            for r in ok
        ]
        columns = ["group", "fields", words, "between fields", "design effect"]
        columns.append("fields that differ")
        lines.append(table(cells, columns, indent=indent + 2, align="lrrrrr"))
        groups = f"Groups: {group_columns(keys)}. " if keys else ""
        lines.append(
            wrap(
                f"{groups}Between fields: the share of the variance that lies between fields. "
                f"Fields that differ: more than random {words} would let them (about 5% expected).",
                indent,
                hang=indent + 2,
            )
        )
    return lines, _variability_cautions(result, rows, words)


def variability_full(result: Any, indent: int = 0) -> str:
    """Every number of ``fov_variability``, one row per group."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows, _ = _variability_rows(result)
    head = (
        f"How variable are the fields? Feature {result.settings['feature']}; analysis "
        f"groups: {group_columns(keys)}."
    )
    lines = [wrap(head, indent, hang=indent + 2)]
    missing = any(int(r.get("missing_values") or 0) for r in rows)
    columns = ["group", "fields", words, "per field"]
    columns += ["no value, left out"] if missing else []
    columns += ["between", "design effect", "error ratio", "differ"]
    cells = []
    for r in rows:
        cell = [_group(r, keys), int(r["fovs"]), int(r["observations"])]
        cell.append(_per_field(r["observations_per_fov"]))
        cell += [int(r.get("missing_values") or 0)] if missing else []
        if r["status"] == "ok":
            share = r["unusual_fov_share"]
            cell += [percent(r["icc"]), _one(r["design_effect"]), _one(r["error_ratio"])]
            cell.append(percent(share) if _present(share) else None)
        else:
            cell += [None, None, None, None]
        cells.append(cell)
    lines.append(table(cells, columns, indent=indent + 2, align="l" + "r" * (len(columns) - 1)))
    lines.append(
        wrap(
            "Between: the share of the variance that lies between fields. Design effect: how "
            "many times the variance of a mean from whole fields is that of one from as many "
            f"random {words}; error ratio: how many times SD / sqrt(n) understates its error "
            "(the square root). Differ: the share of fields whose mean differs more than random "
            f"{words} would let it (about 5% expected).",
            indent,
            hang=indent + 2,
        )
    )
    return join(lines, _variability_cautions(result, rows, words), indent, "full")


# --- The JCB object-count rule ----------------------------------------------------


def _selector(selector: Mapping[str, Any]) -> str:
    """``MOCK`` for ``{"condition": ["MOCK"]}``: the values a selector accepts."""
    parts = []
    for value in selector.values():
        values = value if isinstance(value, list | tuple | set) else [value]
        parts.append("/".join(str(v) for v in values))
    return " · ".join(parts)


def rule_reference(settings: Mapping[str, Any], words: str) -> str:
    """The reference mean of the JCB check, in words."""
    control = settings.get("control")
    if not control:
        return str(settings["reference"])
    repeat = settings.get("repeat_col")
    within = f" of the same {repeat}" if repeat else ", every repeat pooled"
    return f"the mean of all {_selector(control)} {words}{within}"


def _rule_cautions(result: Any, rows: list[dict]) -> list[str]:
    cautions = []
    shift = result.settings.get("reference_shift")
    if shift:
        cautions.append(shift)
    skipped = [r for r in rows if r["status"] != "ok"]
    reasons: dict[str, list[str]] = {}
    for row in skipped:
        where = f"{_group(row, result.group_cols)} ({rate(row['target'])})"
        reasons.setdefault(str(row["status"]), []).append(where)
    for reason, where in reasons.items():
        cautions.append(f"JCB check in {listing(where, limit=3)}: {reason}")
    return cautions


def object_rule_parts(result: Any, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form of ``object_rule_check``: one row per target, ranges over the groups."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows = result.table.to_dict("records")
    ok = [r for r in rows if r["status"] == "ok"]
    groups = len({_group(r, keys) for r in rows})
    reference = rule_reference(result.settings, words)
    lines = [
        wrap(
            f"Would the JCB count rule hold with whole fields? The {words} it asks for, taken as "
            f"whole fields in random order, against {reference}:",
            indent,
            hang=indent + 2,
        )
    ]
    cells, missed = [], []
    for target in result.settings["targets"]:
        own = [r for r in ok if float(r["target"]) == float(target)]
        if not own:
            continue
        held = [r for r in own if r["meets_target"]]
        misses = [_group(r, keys) for r in own if not r["meets_target"]]
        missed += [(rate(target), misses)] if misses else []
        ratios = [float(r["error_ratio"]) for r in own if _present(r["error_ratio"])]
        predicted = span((float(r["predicted_error"]) for r in own), "percent")
        real = span((float(r["real_error"]) for r in own), "percent")
        times = f" ({_ones(ratios)}x)" if ratios else ""
        fields = span(int(r["fov_rule_acquired"]) for r in own)
        upper = span(int(r["fov_rule_acquired_upper95"]) for r in own)
        cells.append(
            [
                rate(target),
                _held(len(held), len(own)),
                span(int(r["rule_objects"]) for r in own),
                span(float(r["fovs_used"]) for r in own),
                f"{predicted} → {real}{times}",
                f"{fields} ({upper})",
            ]
        )
    if cells:
        columns = ["target", "held in", f"{words} asked", "fields taken"]
        columns += ["error: SD/sqrt(n) → real", "as fields (upper-95 SD)"]
        lines.append(table(cells, columns, indent=indent + 2, align="llrrll"))
        ranges = "Ranges are over the groups; as" if groups > 1 else "As"
        lines.append(
            wrap(
                f"{ranges} fields: the fields to acquire when the rule counts fields, not {words}.",
                indent,
                hang=indent + 2,
            )
        )
    if missed:
        where = "; ".join(f"{target} in {listing(groups, limit=3)}" for target, groups in missed)
        text = f"The rule misses its target with whole fields: {where}."
        lines.append(wrap(text, indent, hang=indent + 2))
    return lines, _rule_cautions(result, rows)


def object_rule_full(result: Any, indent: int = 0) -> str:
    """Every number of ``object_rule_check``, one row per group and target."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows = result.table.to_dict("records")
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        wrap(
            f"Would the JCB object-count rule hold with whole fields? Feature "
            f"{result.settings['feature']}; reference mean: "
            f"{rule_reference(result.settings, words)}; analysis groups: {group_columns(keys)}.",
            indent,
            hang=indent + 2,
        )
    ]
    partial = any(int(r["fovs_with_observations"]) != int(r["fovs"]) for r in ok)
    cells = []
    for r in ok:
        ratio = r["error_ratio"]
        times = f" ({_one(ratio)}x)" if _present(ratio) else ""
        verdict = "met" if r["meets_target"] else "missed"
        cell = [_group(r, keys), f"{rate(r['target'])} {verdict}", int(r["rule_objects"])]
        cell += [_one(r["fovs_used"])]
        cell += [f"{percent(r['predicted_error'])} → {percent(r['real_error'])}{times}"]
        cell += [f"{int(r['fov_rule_fovs'])} ({int(r['fov_rule_fovs_upper95'])})"]
        if partial:
            cell += [f"{int(r['fovs_with_observations'])} of {int(r['fovs'])}"]
            cell += [f"{int(r['fov_rule_acquired'])} ({int(r['fov_rule_acquired_upper95'])})"]
        else:
            cell += [int(r["fovs"])]
        cells.append(cell)
    if cells:
        columns = ["group", "target", f"{words} asked", "fields taken", "SD/sqrt(n) → real"]
        columns += ["as fields"]
        columns += [f"fields with {words}", "to acquire"] if partial else ["of"]
        lines.append(table(cells, columns, indent=indent + 2, align="llrrlrrr"[: len(columns)]))
        scaled = (
            f", counted in fields with {words} ('to acquire': scaled to the acquired fields)"
            if partial
            else ", of the acquired fields ('of')"
        )
        lines.append(
            wrap(
                f"Target: met or missed by the real error. {words.capitalize()} asked: the "
                f"rule's number of {words}; fields taken: the mean number of whole fields drawn "
                "at random until that many; SD/sqrt(n) → real: the rule's predicted error of "
                "their mean, then its real error (the SD of that mean over the draws), both as "
                "shares of the reference mean, and real over predicted. As fields: the fields "
                "the rule asks for with the field as the unit, from the point SD (in brackets: "
                f"from the upper-95 SD){scaled}.",
                indent,
                hang=indent + 2,
            )
        )
    return join(lines, _rule_cautions(result, rows), indent, "full")


# --- The JCS descriptor spread -------------------------------------------------


def _match(row: Mapping[str, Any]) -> str:
    """``from 40`` (the smallest sample matching the random-object spread), or not within."""
    match = row["objects_to_match"]
    if _present(match):
        return f"from {int(match)}"
    return f"not by {int(row['largest_size_tried'])}"


def _ratio(row: Mapping[str, Any]) -> str:
    ratio = row["ratio"]
    return f"{_one(ratio)}x" if _present(ratio) else "no ratio"


def _spread_cautions(result: Any, words: str) -> list[str]:
    cautions = []
    rows = result.at_threshold
    if len(rows) and rows["ratio"].isna().any():
        cautions.append("no ratio: the random-object samples never varied in that descriptor")
    threshold = result.settings["threshold"]
    for skipped in result.settings.get("skipped_groups", []):
        where = skipped.get("label") or skipped["group"]
        cautions.append(
            f"JCS check in {where}: not checked: {skipped['observations']} {words}, fewer than "
            f"the threshold of {threshold}"
        )
    return cautions


def _spread_head(result: Any, words: str) -> str:
    return (
        "Do descriptors stabilise as fast with whole fields? Feature "
        f"{result.settings['feature']}, spread over {result.settings['n_draws']} samples "
        f"of {result.settings['threshold']} {words}: whole fields against random {words}"
    )


def descriptor_spread_parts(result: Any, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form of ``descriptor_spread_check``: one row per descriptor over the groups."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows = result.at_threshold
    lines = [wrap(f"{_spread_head(result, words)}:", indent, hang=indent + 2)]
    cells = []
    for descriptor, own in rows.groupby("descriptor", sort=False) if len(rows) else []:
        ratios = own["ratio"].dropna().astype(float)
        matched = own["objects_to_match"].dropna().astype(int)
        never = own[own["objects_to_match"].isna()].to_dict("records")
        groups = len(own)
        cells.append(
            [
                descriptor,
                f"{_ones(list(ratios))}x" if len(ratios) else "no ratio",
                _held(len(matched), groups),
                span(matched) if len(matched) else None,
                listing([_group(r, keys) for r in never], limit=2) if never else None,
            ]
        )
    if cells:
        columns = ["descriptor", "spread ratio", "matched in", "from", "never matched in"]
        lines.append(table(cells, columns, indent=indent + 2, align="lrlrl"))
        lines.append(
            wrap(
                f"From: the smallest sample of {words} from which whole-field samples stay "
                f"within the random-{words} spread at the threshold (the range over groups).",
                indent,
                hang=indent + 2,
            )
        )
    return lines, _spread_cautions(result, words)


def descriptor_spread_full(result: Any, indent: int = 0) -> str:
    """Every number of ``descriptor_spread_check``: one row per group, one column per
    descriptor (the spread ratio, and from how many objects whole fields match)."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows = result.at_threshold
    lines = [
        wrap(
            f"{_spread_head(result, words)}; analysis groups: {group_columns(keys)}. Each cell: "
            f"the spread ratio (whole fields over random {words}) and, in brackets, the smallest "
            f"sample of {words} from which whole fields stay within the random-{words} spread "
            "(not by: never, up to the largest sample tried).",
            indent,
            hang=indent + 2,
        )
    ]
    descriptors = list(dict.fromkeys(rows["descriptor"])) if len(rows) else []
    cells = []
    parts = rows.groupby(keys, sort=False) if keys and len(rows) else [((), rows)]
    for _, own in parts if len(rows) else []:
        records = own.to_dict("records")
        by_name = {r["descriptor"]: r for r in records}
        cell = [_group(records[0], keys)]
        cell += [
            f"{_ratio(by_name[d])} ({_match(by_name[d])})" if d in by_name else None
            for d in descriptors
        ]
        cells.append(cell)
    if cells:
        lines.append(table(cells, ["group", *descriptors], indent=indent + 2))
    return join(lines, _spread_cautions(result, words), indent, "full")


__all__ = [
    "descriptor_spread_full",
    "descriptor_spread_parts",
    "join",
    "object_rule_full",
    "object_rule_parts",
    "rule_reference",
    "variability_full",
    "variability_parts",
]
