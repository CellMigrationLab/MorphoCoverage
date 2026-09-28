"""Unusual objects and state clustering in words (private), short and full.

``UnusualObjects.summary`` and ``FOVClustering.summary`` print through this
module (``detail="short"`` by default, ``"full"`` for every number):

- **unusual objects**: the definition in one sentence; short, the groups in
  one table (scored, unusual, share, how well refits agree); full, the same
  with the threshold and its calibration; the unstable labels and the
  single-reference calibration each once as a ⚠️ line.
- **clustering**: one answer sentence per state (in how many groups where it
  was seen it is clustered in a few fields), then one table of the groups
  where it was seen (its objects, the fields holding it against random
  objects, the permutation p, and the fields to see it against a count of
  objects); full adds the fields holding at least ``min_objects_per_fov``,
  the reference groups' floor and the fields left out. The technical repeats
  holding each state in one line (short) or table (full), and the assumption
  of independent technical repeats once.

``*_parts(result, indent)`` return ``(lines, cautions)`` for the report,
which pools the caveats of question 3. Shared-private: used by ``unusual``,
``clustering`` and ``retrospective.report_sampling``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .._caveats import assumption
from .._style import among, group_columns, group_label, span, table, wrap
from .._wording import count, listing, percent
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural
from ._outliers import LEAVE_ONE_FIELD_OUT

# Seed-check agreement below which the labels are called unstable (as in fiberfeatures).
SEED_AGREEMENT_WARNING = 0.9


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value)


def _group(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    return group_label({key: row[key] for key in keys})


def _selector(selector: Mapping[str, Any]) -> str:
    parts = []
    for value in selector.values():
        values = value if isinstance(value, list | tuple | set) else [value]
        parts.append("/".join(str(v) for v in values))
    return " · ".join(parts)


# --- Unusual objects -------------------------------------------------------------


def _unusual_head(result: Any, words: str) -> str:
    settings = result.settings
    return (
        f"Unusual {words}: farther from the reference ({_selector(settings['reference'])}) "
        f"than all but {percent(settings['quantile'])} of reference {words} are from the other "
        f"reference groups ({len(settings['feature_cols'])} standardized features, "
        f"{settings['n_neighbors']} nearest neighbours)."
    )


def _unusual_cautions(result: Any, words: str) -> list[str]:
    keys = list(result.group_cols)
    rows = result.groups.to_dict("records")
    ok = [r for r in rows if r["status"] == "ok"]
    cautions = []
    reasons: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] != "ok":
            reasons.setdefault(str(row["status"]), []).append(_group(row, keys))
    for reason, groups in reasons.items():
        cautions.append(f"unusual {words} not scored in {listing(groups, limit=3)}: {reason}")
    agreements = [r for r in ok if _present(r["seed_agreement"])]
    unstable = [r for r in agreements if r["seed_agreement"] < SEED_AGREEMENT_WARNING]
    if unstable:
        shares = [float(r["seed_agreement"]) for r in unstable]
        named = listing([_group(r, keys) for r in unstable], limit=3)
        cautions.append(
            f"refits with other reference samples agree on only {span(shares, 'percent')} of the "
            f"unusual {words} in {among(len(unstable), len(agreements))} ({named}; below "
            f"{percent(SEED_AGREEMENT_WARNING)}): treat those labels as unstable"
        )
    single = [r for r in ok if r.get("calibration") == LEAVE_ONE_FIELD_OUT]
    if single:
        cautions.append(
            "with a single reference group the threshold comes from its fields scored against "
            f"its other fields (in {listing([_group(r, keys) for r in single])}), so a shift "
            "between that group and the one scored raises the share"
        )
    return cautions


def unusual_parts(
    result: Any, indent: int = 0, *, with_table: bool = True
) -> tuple[list[str], list[str]]:
    """The short form of ``unusual_objects``: the definition, the groups in one table."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    ok = [r for r in result.groups.to_dict("records") if r["status"] == "ok"]
    lines = [wrap(_unusual_head(result, words), indent, hang=indent + 2)]
    if ok and with_table:
        cells = [
            [
                _group(r, keys),
                int(r["scored"]),
                int(r["unusual"]),
                percent(r["unusual_share"]),
                percent(r["seed_agreement"]) if _present(r["seed_agreement"]) else None,
            ]
            for r in ok
        ]
        columns = ["group", f"{words} scored", "unusual", "share", "refits agree"]
        lines.append(table(cells, columns, indent=indent + 2, align="lrrrr"))
    elif ok:
        shares = [float(r["unusual_share"]) for r in ok]
        lines.append(
            wrap(
                f"{span(shares, 'percent')} of the {words} of each group are unusual "
                f"({span(int(r['unusual']) for r in ok)} {words}, over {count(len(ok), 'group')}).",
                indent,
                hang=indent + 2,
            )
        )
    return lines, _unusual_cautions(result, words)


def unusual_full(result: Any, indent: int = 0) -> str:
    """Every number of ``unusual_objects``, one row per group."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    rows = result.groups.to_dict("records")
    lines = [
        wrap(
            f"{_unusual_head(result, words)} Groups: {group_columns(keys)}.",
            indent,
            hang=indent + 2,
        )
    ]
    cells = []
    for r in rows:
        if r["status"] != "ok":
            continue
        agreement = r["seed_agreement"]
        cells.append(
            [
                _group(r, keys),
                int(r["scored"]),
                int(r["unusual"]),
                percent(r["unusual_share"]),
                percent(agreement) if _present(agreement) else None,
                f"{float(r['threshold']):.3g}" if _present(r.get("threshold")) else None,
                "single reference"
                if r.get("calibration") == LEAVE_ONE_FIELD_OUT
                else r.get("calibration"),
            ]
        )
    if cells:
        columns = ["group", f"{words} scored", "unusual", "share", "refits agree"]
        columns += ["threshold", "calibrated"]
        lines.append(table(cells, columns, indent=indent + 2, align="lrrrrrl"))
    return "\n".join(lines)


# --- Clustering ------------------------------------------------------------------


def _needed(row: Mapping[str, Any]) -> str | None:
    """``7 (up to 13)``: the fields to see the state, from its field incidence."""
    if not _present(row["fovs_needed"]):
        return None
    upper = row["fovs_needed_upper"]
    bound = f" (up to {int(upper)})" if _present(upper) else ""
    return f"{int(row['fovs_needed'])}{bound}"


def _seen(result: Any, block: pd.DataFrame) -> list[dict]:
    return [r for r in block.to_dict("records") if r["labelled"] and r["state_objects"]]


def _clustering_cautions(result: Any, words: str) -> list[str]:
    keys = list(result.group_cols)
    rows = result.table.to_dict("records")
    cautions = []
    unlabelled = [r for r in rows if r["labelled"] and int(r["fovs_unlabelled"])]
    if unlabelled:
        numbers = [int(r["fovs_unlabelled"]) for r in unlabelled]
        cautions.append(
            f"fields whose {words} carry no label were left out of the fields assessed: "
            f"{span(numbers)} per group in {listing([_group(r, keys) for r in unlabelled], 3)}"
        )
    m = int(result.settings.get("min_objects_per_fov", 1))
    none = [r for r in rows if r["state_objects"] and not _present(r["fovs_needed"])]
    for state in dict.fromkeys(r["state"] for r in none):
        where = listing([_group(r, keys) for r in none if r["state"] == state], 3)
        cautions.append(
            f"no field holds at least {m} {state_words(result, state, words)} in {where}, so no "
            "field count is given there"
        )
    if _technical(result):
        cautions.append(
            assumption(
                "the field counts assume fields from independent technical repeats: a state "
                "held by few technical repeats is clustered in their fields, so seeing it needs "
                "fields from more technical repeats than the count says (no two-level plan is "
                "made)"
            )
        )
    return cautions


def _state_answer(result: Any, state: Any, block: pd.DataFrame, words: str) -> str:
    keys = list(result.group_cols)
    seen = _seen(result, block)
    clustered = [r for r in seen if r["p_clustered"] <= 0.05]
    name = state_words(result, state, words)
    text = f"Are the {name} spread over the fields? "
    if not seen:
        text += "Not seen in any group."
    else:
        text += (
            f"Clustered in a few fields (p ≤ 0.05) in {among(len(clustered), len(seen))} where seen"
        )
        text += f" ({listing([_group(r, keys) for r in clustered], 3)})." if clustered else "."
    unseen = [r for r in block.to_dict("records") if r["labelled"] and not r["state_objects"]]
    if unseen:
        where = [
            f"{_group(r, keys)} ({int(r['labelled'])} labelled {words} in "
            f"{count(int(r['fovs']), 'field')})"
            for r in unseen
        ]
        text += f" Not seen in {listing(where, 3)}."
    unlooked = [r for r in block.to_dict("records") if not r["labelled"]]
    if unlooked:
        where = [
            f"{_group(r, keys)} (none of the {words} of its "
            f"{count(int(r['fovs_acquired']), 'field')} is labelled)"
            for r in unlooked
        ]
        text += f" Not looked for in {listing(where, 3)}."
    return text


def state_words(result: Any, state: Any, words: str) -> str:
    """``unusual nuclei``: the state and the unit, the unit once (``unusual cells``)."""
    name = result.state_name(state)
    return name if name.endswith(words) else f"{name} {words}"


def _state_columns(result: Any, words: str, full: bool) -> list[str]:
    m = int(result.settings.get("min_objects_per_fov", 1))
    k = int(result.settings.get("min_fov_occurrences", 1))
    chance = percent(result.settings["detection_probability"])
    columns = ["group"]
    columns += [words, "in fields", "if random", "ratio" if full else None, "p"]
    columns += [f"holding ≥{m}"] if m > 1 else []
    reference = "reference_fovs" in result.table.columns
    columns += ["reference holding"] if reference else []
    to_see = f"to see in ≥{k}, {chance}" if k > 1 else f"to see, {chance}"
    columns += [to_see, "if random"]
    return [c for c in columns if c is not None]


def _state_cells(result: Any, row: Mapping[str, Any], keys: Sequence[str], full: bool) -> list:
    m = int(result.settings.get("min_objects_per_fov", 1))
    cell = [_group(row, keys)]
    cell += [f"{int(row['state_objects'])} ({percent(row['prevalence'])})"]
    cell += [f"{int(row['fovs_with_state'])} of {int(row['fovs'])}"]
    cell += [f"{float(row['expected_fovs_if_random']):.1f}"]
    cell += [f"{float(row['clustering_ratio']):.1f}"] if full else []
    cell += [f"{float(row['p_clustered']):.2g}"]
    if m > 1:
        holding = int(row.get("fovs_holding_state", row["fovs_with_state"]))
        cell += [f"{holding} of {int(row['fovs'])}"]
    if "reference_fovs" in result.table.columns:
        total = row.get("reference_fovs")
        if _present(total) and int(total):
            held = int(row["reference_fovs_holding"])
            cell += [f"{held} of {int(total)} ({percent(held / int(total))})"]
        else:
            cell += [None]
    cell += [_needed(row)]
    random = row["fovs_needed_if_random"]
    cell += [int(random) if _present(random) and _present(row["fovs_needed"]) else None]
    return cell


def _technical(result: Any) -> list[dict]:
    """Per condition and repeat and state: the technical repeats holding it (P6-5)."""
    if "technical_repeats" not in result.table.columns:
        return []
    rows = result.table.dropna(subset=["technical_repeats"])
    keys = [c for c in result.group_cols if c != TECHNICAL_REPEAT_COLUMN]
    found = []
    for _, block in rows.groupby([*keys, "state"] if keys else ["state"], sort=False):
        if not (block["state_objects"].fillna(0) > 0).any():
            continue
        first = block.iloc[0]
        found.append(
            {
                "where": _group(first, keys) if keys else "all groups",
                "state": result.state_name(first["state"]),
                "held": int(block["technical_repeats_holding"].sum()),
                "total": int(block["technical_repeats"].sum()),
            }
        )
    return found


def _technical_lines(result: Any, indent: int, full: bool) -> list[str]:
    found = _technical(result)
    if not found:
        return []
    keys = [c for c in result.group_cols if c != TECHNICAL_REPEAT_COLUMN]
    if not full and all(f["held"] == f["total"] for f in found):
        text = (
            f"Technical repeats: every technical repeat holds it, in "
            f"{among(len(found), len(found), group_columns(keys) or 'group')}."
            if len({f["state"] for f in found}) == 1
            else "Technical repeats: every technical repeat holds each state."
        )
        return [wrap(text, indent, hang=indent + 2)]
    cells = [
        [f["where"], f["state"], f"{f['held']} of {count(f['total'], 'technical repeat')}"]
        for f in found
    ]
    columns = [group_columns(keys) or "groups", "state", "held by"]
    return [
        wrap("Technical repeats holding each state:", indent),
        table(cells, columns, indent=indent + 2),
    ]


def clustering_parts(
    result: Any, indent: int = 0, *, full: bool = False
) -> tuple[list[str], list[str]]:
    """The state clustering: per state one answer sentence and one table of the groups where
    it was seen; the technical repeats holding it. ``(lines, cautions)``, under ``indent``."""
    words = unit_plural(result.settings["observation_unit"])
    keys = list(result.group_cols)
    lines = []
    for state, block in result.table.groupby("state", sort=False):
        lines.append(wrap(_state_answer(result, state, block, words), indent, hang=indent + 2))
        seen = _seen(result, block)
        if seen:
            cells = [_state_cells(result, row, keys, full) for row in seen]
            columns = _state_columns(result, words, full)
            lines.append(table(cells, columns, indent=indent + 2, align="l" + "r" * 20))
    if lines:
        lines.append(
            wrap(
                f"In fields: the fields holding the state; if random: the fields as many random "
                f"{words} would fill; p: how often shuffled labels reach as few fields. To see: "
                f"the fields to see it from its field incidence (up to: from the low end of its "
                f"interval), against the fields a count of random {words} asks for.",
                indent,
                hang=indent + 2,
            )
        )
    lines += _technical_lines(result, indent, full)
    return lines, _clustering_cautions(result, words)


__all__ = [
    "SEED_AGREEMENT_WARNING",
    "clustering_parts",
    "state_words",
    "unusual_full",
    "unusual_parts",
]
