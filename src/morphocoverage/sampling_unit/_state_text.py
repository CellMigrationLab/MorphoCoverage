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
  objects, the permutation p and its q, and the fields to see it against a
  count of objects); a group is clustered at ``q <= CLUSTER_ALPHA``,
  Benjamini-Hochberg over every group and state of the analysis
  (``clustered``, owner, 2026-10-05); full adds the fields holding at least ``min_objects_per_fov``,
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
from ..schema import observation_words
from ._group_text import distinct_labels
from ._outliers import LEAVE_ONE_FIELD_OUT

# Seed-check agreement below which the labels are called unstable (as in fiberfeatures).
SEED_AGREEMENT_WARNING = 0.9
# The q at or below which a state is clustered in a group's fields (Benjamini-Hochberg).
CLUSTER_ALPHA = 0.05


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


def reference_words(settings: Mapping[str, Any]) -> str:
    """The reference of the unusual observations: ``the reference siCTRL``, or with several
    controls ``each condition's own control`` (owner, 2026-10-05)."""
    if settings.get("own"):
        return "each condition's own control"
    return f"the reference {_selector(settings['reference'])}"


def unusual_words(result: Any) -> str:
    """The unusual observations' plural word: ``images`` when each field holds one
    observation (``settings["words"]``, plan v2), else the observation unit's."""
    return result.settings.get("words") or observation_words(result.settings)[1]


def _unusual_head(result: Any, words: str) -> str:
    settings = result.settings
    # Biological unusualness, on values relative to each own control (plan v2), is exploratory.
    label = f"Unusual {words} (exploratory)" if settings.get("exploratory") else f"Unusual {words}"
    return (
        f"{label}: farther from {reference_words(settings)} "
        f"than all but {percent(settings['quantile'])} of reference {words} are from the other "
        f"reference groups ({_space(settings)}, "
        f"{settings['n_neighbors']} nearest neighbours)"
        + (
            ", each group against the reference of its own sub-experiment (the conditions "
            "that share its repeats and control)."
            if settings.get("within_experiment")
            else "."
        )
    )


def _space(settings: Any) -> str:
    """The space unusual observations were judged in: ``12 standardized features``, or with
    other feature tables, which (owner, 2026-10-06: the explainable features, or without them
    the tables' weighted combined space)."""
    space = settings.get("space")
    standardized = f"{count(len(settings['feature_cols']), 'standardized feature')}"
    return f"on {space}, standardized" if space else standardized


def single_feature(result: Any) -> bool:
    """Whether the unusual observations read one feature: they then restate its shift from
    the control (owner, 2026-10-06, audit 10: the share flagged followed the mean shift,
    Spearman 0.82, and 88% sat on its low side)."""
    return len(result.settings.get("feature_cols", ())) == 1


def _restated(result: Any, words: str) -> str:
    """The sentence in place of the shares when one feature is read."""
    feature = result.settings["feature_cols"][0]
    return (
        f"On one feature, {feature}, unusual {words} restate its shift from the control: not "
        "informative beyond the mean shift in question 9, so no share is given."
    )


def _agreement(row: Mapping[str, Any]) -> str | None:
    """``93%``, or with no unusual observation ``– (none unusual; refits flag 2)``."""
    if _present(row.get("seed_agreement")):
        return percent(row["seed_agreement"])
    refits = row.get("refit_unusual")
    if row.get("unusual") == 0 and _present(refits):
        return f"– (none unusual; refits flag {int(refits)})" if refits else "– (none unusual)"
    return None


def scored_shares(result: Any) -> list[float]:
    """The unusual share of each scored group that is not a control (all scored groups when
    every one is a control)."""
    ok = result.groups[result.groups["status"] == "ok"]
    tested = ok[~ok["reference_group"].astype(bool)] if "reference_group" in ok else ok
    return [float(v) for v in (tested if len(tested) else ok)["unusual_share"]]


def scored_label(result: Any) -> str:
    """``each group that is not a control``, or ``each scored group`` when every scored group
    is a control (then the shares are theirs)."""
    ok = result.groups[result.groups["status"] == "ok"]
    if "reference_group" in ok and (~ok["reference_group"].astype(bool)).any():
        return "each group that is not a control"
    return "each scored group"


def background_share(result: Any) -> float | None:
    """The share of the controls' own observations flagged by the same rule, ``None`` when
    the controls are not scored or the result declares none (audit A5)."""
    groups = result.groups
    if "reference_group" not in groups:
        return None
    ok = groups[groups["reference_group"].astype(bool) & (groups["status"] == "ok")]
    scored = int(ok["scored"].sum())
    return int(ok["unusual"].sum()) / scored if scored else None


def background(result: Any, words: str) -> str:
    """The controls' own share flagged by the same rule, the background every share reads
    against, always said (owner, 2026-10-06, audit 10: the threshold flagged 16-21% of the
    controls' own images), or why the controls are not scored."""
    groups = result.groups
    if "reference_group" not in groups:
        return ""
    controls = groups[groups["reference_group"].astype(bool)]
    ok = controls[controls["status"] == "ok"]
    if not len(ok):
        reasons = listing(list(dict.fromkeys(str(r) for r in controls["status"])), 2)
        why = f": {reasons}" if reasons else ""
        return f"The controls' own {words} are not scored{why}, so no share has a background."
    flagged, scored = int(ok["unusual"].sum()), int(ok["scored"].sum())
    return (
        f"Background: the controls' own {words}, scored the same way against the other "
        f"control groups, are {percent(flagged / scored)} unusual, {flagged} of {scored}; "
        "a condition's share reads against it."
    )


def _blind_spot(words: str) -> Any:
    """What the reference cannot show: a state the control already holds (``words``: the
    observations, plural)."""
    return assumption(
        "A state the reference already holds, at any share, is not unusual: only its share "
        f"can change, and the populations (“Are there groups of {words}?”) compare each "
        "condition's share of such a state with its control's."
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
    # One feature: the labels restate the mean shift, so their stability is not at issue.
    agreements = [] if single_feature(result) else [r for r in ok if _present(r["seed_agreement"])]
    unstable = [r for r in agreements if r["seed_agreement"] < SEED_AGREEMENT_WARNING]
    if unstable:
        shares = [float(r["seed_agreement"]) for r in unstable]
        named = listing([_group(r, keys) for r in unstable], limit=3)
        cautions.append(
            f"refits with other reference samples agree on only {span(shares, 'percent')} of the "
            f"unusual {words} in {among(len(unstable), len(agreements))} ({named}; below "
            f"{percent(SEED_AGREEMENT_WARNING)}): treat those labels as unstable"
        )
    failing = [r for r in ok if _present(r.get("seed_failures")) and r["seed_failures"]]
    if failing:
        failed = sum(int(r["seed_failures"]) for r in failing)
        named = listing([_group(r, keys) for r in failing], limit=3)
        cautions.append(
            f"{count(failed, 'refit')} with other reference samples failed in {named}, so "
            f"those labels may depend on the sample: {failing[0]['seed_failure']}"
        )
    single = [r for r in ok if r.get("calibration") == LEAVE_ONE_FIELD_OUT]
    if single:
        # The reference named as the reference, the groups scored as scored (audit 10, A24).
        refs = [g for r in single for g in r.get("reference_groups") or []]
        named = listing(distinct_labels(refs, list(refs[0]))) if refs else "the reference"
        cautions.append(
            f"against a single reference group, {named}, the threshold comes from its fields "
            "scored against its other fields, so a shift between it and "
            f"{listing([_group(r, keys) for r in single])} raises the share"
        )
    return cautions


def unusual_parts(
    result: Any, indent: int = 0, *, with_table: bool = True
) -> tuple[list[str], list[str]]:
    """The short form of ``unusual_objects``: the definition, the groups in one table."""
    words = unusual_words(result)
    keys = list(result.group_cols)
    ok = [r for r in result.groups.to_dict("records") if r["status"] == "ok"]
    lines = [wrap(_unusual_head(result, words), indent, hang=indent + 2)]
    if ok and single_feature(result):
        lines.append(wrap(_restated(result, words), indent, hang=indent + 2))
    elif ok and with_table:
        cells = [
            [
                _group(r, keys),
                int(r["scored"]),
                int(r["unusual"]),
                percent(r["unusual_share"]),
                _agreement(r),
            ]
            for r in ok
        ]
        columns = ["group", f"{words} scored", "unusual", "share", "refits agree"]
        lines.append(table(cells, columns, indent=indent + 2, align="lrrrr"))
    elif ok:
        lines.append(
            wrap(
                f"{span(scored_shares(result), 'percent')} of the {words} of "
                f"{scored_label(result)} are unusual "
                f"({span(int(r['unusual']) for r in ok)} {words}, over "
                f"{count(len(ok), 'group')}).",
                indent,
                hang=indent + 2,
            )
        )
    if not single_feature(result) and background(result, words):
        lines.append(wrap(background(result, words), indent, hang=indent + 2))
    observed = observation_words(result.settings)[1]
    return lines, [*_unusual_cautions(result, words), _blind_spot(observed)]


def unusual_full(result: Any, indent: int = 0) -> str:
    """Every number of ``unusual_objects``, one row per group."""
    words = unusual_words(result)
    keys = list(result.group_cols)
    rows = result.groups.to_dict("records")
    lines = [
        wrap(
            f"{_unusual_head(result, words)} Groups: {group_columns(keys)}.",
            indent,
            hang=indent + 2,
        )
    ]
    if single_feature(result):
        return "\n".join([*lines, wrap(_restated(result, words), indent, hang=indent + 2)])
    cells = []
    for r in rows:
        if r["status"] != "ok":
            continue
        cells.append(
            [
                _group(r, keys),
                int(r["scored"]),
                int(r["unusual"]),
                percent(r["unusual_share"]),
                _agreement(r),
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
    if background(result, words):
        lines.append(wrap(background(result, words), indent, hang=indent + 2))
    return "\n".join(lines)


# --- Clustering ------------------------------------------------------------------


def _needed(row: Mapping[str, Any]) -> str | None:
    """``7 (up to 13)``: the fields to see the state, from its field incidence."""
    if not _present(row["fovs_needed"]):
        return None
    upper = row["fovs_needed_upper"]
    bound = f" (up to {int(upper)})" if _present(upper) else ""
    return f"{int(row['fovs_needed'])}{bound}"


def clustered(rows: pd.DataFrame) -> pd.Series:
    """Which rows of a clustering table are clustered: ``q_clustered <= CLUSTER_ALPHA``, the
    permutation p corrected over every group and state of the analysis."""
    return pd.to_numeric(rows["q_clustered"], errors="coerce") <= CLUSTER_ALPHA


def _family(result: Any) -> int:
    """The tests the q-values are corrected over: the groups and states with a p."""
    return int(pd.to_numeric(result.table["p_clustered"], errors="coerce").notna().sum())


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


def tested(rows: pd.DataFrame) -> pd.Series:
    """Which rows of a clustering table were tested: their smallest possible p can reach 0.05
    (audit 10; a table from before has every row with a p tested)."""
    return pd.to_numeric(rows["p_clustered"], errors="coerce").notna()


def _state_answer(result: Any, state: Any, block: pd.DataFrame, words: str) -> str:
    keys = list(result.group_cols)
    seen = _seen(result, block)
    run = [r for r in seen if _present(r["p_clustered"])]
    hits = [r for r in run if float(r["q_clustered"]) <= CLUSTER_ALPHA]
    name = state_words(result, state, words)
    text = f"Are the {name} spread over the fields? "
    if not seen:
        text += "Not seen in any group."
    elif run:
        tests = count(_family(result), "test")
        text += (
            f"Clustered in a few fields (q ≤ {CLUSTER_ALPHA:g} after Benjamini-Hochberg over the "
            f"{tests} of every group and state) in {among(len(hits), len(run))} where tested"
        )
        text += f" ({listing([_group(r, keys) for r in hits], 3)})." if hits else "."
        text += "" if hits else " No clustering shown."
    if len(run) < len(seen):
        left = [_group(r, keys) for r in seen if not _present(r["p_clustered"])]
        text += (
            f" Not tested in {listing(left, 3)}: too few {name} for a p that can reach "
            f"{CLUSTER_ALPHA:g}, the tests that can be corrected over."
        )
    unseen = [r for r in block.to_dict("records") if r["labelled"] and not r["state_objects"]]
    if unseen:
        where = [
            f"{_group(r, keys)} ({int(r['labelled'])} {words} scored in "
            f"{count(int(r['fovs']), 'field')}, none {result.state_name(state)})"
            for r in unseen
        ]
        text += f" Not seen in {listing(where, 3)}."
    unlooked = [r for r in block.to_dict("records") if not r["labelled"]]
    if unlooked:
        where = [
            f"{_group(r, keys)} (none of the {words} of its "
            f"{count(int(r['fovs_acquired']), 'field')} scored)"
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
    columns += [words, "in fields", "if random", "ratio" if full else None, "p", "q"]
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
    run = _present(row["p_clustered"])
    cell += [f"{float(row[k]):.2g}" if run else "–" for k in ("p_clustered", "q_clustered")]
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
    # Holding: a field of the technical repeat with at least m of the state (audit 10, C10).
    m = int(result.settings.get("min_objects_per_fov", 1))
    holds = f"a field with at least {m} of it" if m > 1 else "a field with one of it"
    if not full and all(f["held"] == f["total"] for f in found):
        text = (
            f"Technical repeats: every technical repeat holds it ({holds}), in "
            f"{among(len(found), len(found), group_columns(keys) or 'group')}."
            if len({f["state"] for f in found}) == 1
            else f"Technical repeats: every technical repeat holds each state ({holds})."
        )
        return [wrap(text, indent, hang=indent + 2)]
    cells = [
        [f["where"], f["state"], f"{f['held']} of {count(f['total'], 'technical repeat')}"]
        for f in found
    ]
    columns = [group_columns(keys) or "groups", "state", "held by"]
    return [
        wrap(f"Technical repeats holding each state ({holds}):", indent),
        table(cells, columns, indent=indent + 2),
    ]


def clustering_parts(
    result: Any, indent: int = 0, *, full: bool = False
) -> tuple[list[str], list[str]]:
    """The state clustering: per state one answer sentence and one table of the groups where
    it was seen; the technical repeats holding it. ``(lines, cautions)``, under ``indent``."""
    words = observation_words(result.settings)[1]
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
                f"{words} would fill; p: how often shuffled labels reach as few fields; q: that "
                f"p corrected over every group and state tested (Benjamini-Hochberg; – not "
                f"tested: its smallest possible p cannot reach {CLUSTER_ALPHA:g}). To see: "
                f"the fields to see it from its field incidence (up to: from the low end of its "
                f"interval), against the fields a count of random {words} asks for.",
                indent,
                hang=indent + 2,
            )
        )
    lines += _technical_lines(result, indent, full)
    return lines, _clustering_cautions(result, words)


__all__ = [
    "CLUSTER_ALPHA",
    "SEED_AGREEMENT_WARNING",
    "background",
    "clustered",
    "clustering_parts",
    "scored_shares",
    "single_feature",
    "state_words",
    "tested",
    "unusual_full",
    "unusual_parts",
]
