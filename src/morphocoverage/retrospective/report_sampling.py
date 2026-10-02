"""The Foundation, question 1 and question 3 of the microscopist's report, short and full.

- **Foundation** ("why count fields, and not only objects?"): the field
  variability, the mean-precision rule and the JCS descriptor spread with whole fields
  (``sampling_unit``), each as one answer and one table (short) or every
  number of each group (full); the caveats of the three once, at the end.
- **1** ("can these features detect what I care about?"): the representation
  control, one table of its rates (``controls_text``).
- **3** ("what is limiting my sampling?"): short, one line per measured
  quantity of the sampling profile (``report_profile``), the labelled states
  over the fields in one table (``fov_clustering``) and the unusual objects in
  one line, each caveat once; full, every statement of the profile with its
  source, the unusual objects and the clustering group by group.

The helpers ``rule_held``, ``skipped_strata``, ``clustered_states``,
``largest_sources`` and ``unstable_labels`` give ``report_status`` the counts
its headlines read.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import caveat_lines, once
from .._style import among, group_label, span, wrap
from .._wording import count, listing, percent
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..sampling_unit._state_text import (
    SEED_AGREEMENT_WARNING,
    clustering_parts,
    state_words,
    unusual_full,
    unusual_parts,
)
from ..sampling_unit._unit_text import (
    descriptor_spread_full,
    descriptor_spread_parts,
    object_rule_full,
    object_rule_parts,
    variability_full,
    variability_parts,
)
from ._text import unit_plural
from .controls_text import control_full, control_parts, named_stratum
from .effort_text import level_words, share_caveat, unit_word

_LARGEST = "largest source of uncertainty of the mean"


def _joined(
    lines: Sequence[str], cautions: Sequence[str], indent: int, detail: str = "short"
) -> str:
    return "\n".join([*lines, *caveat_lines(once(cautions), indent=indent, detail=detail)])


# --- Foundation ------------------------------------------------------------------


def unit_short(variability: Any, object_rules: Any, descriptor_spread: Any, indent: int = 2) -> str:
    """The Foundation, short: each check's answer and table, then the caveats once."""
    lines: list[str] = []
    cautions: list[str] = []
    for result, parts in (
        (variability, variability_parts),
        (object_rules, object_rule_parts),
        (descriptor_spread, descriptor_spread_parts),
    ):
        if result is not None:
            own, own_cautions = parts(result, indent)
            lines += own
            cautions += own_cautions
    return _joined(lines, cautions, indent)


def unit_full(variability: Any, object_rules: Any, descriptor_spread: Any, indent: int = 2) -> str:
    """The Foundation, full: each check's every number, one block each, under ``indent``."""
    blocks = [
        full(result, indent)
        for result, full in (
            (variability, variability_full),
            (object_rules, object_rule_full),
            (descriptor_spread, descriptor_spread_full),
        )
        if result is not None
    ]
    return "\n\n".join(blocks)


def rule_held(object_rules: Any) -> tuple[int, int]:
    """In how many groups the mean-precision rule held with whole fields at every target, of
    how many."""
    table = object_rules.table
    checked = table[table["status"] == "ok"]
    if not len(checked):
        return 0, 0
    keys = list(object_rules.group_cols)
    if not keys:
        return int(bool(checked["meets_target"].all())), 1
    held = checked.groupby(keys, sort=False)["meets_target"].all()
    return int(held.sum()), len(held)


# --- 1. Controls -----------------------------------------------------------------


def controls_short(controls: Any, indent: int = 2) -> str:
    """Question 1, short: the control's rates in one table, its caveats once."""
    lines, cautions = control_parts(controls, indent)
    return _joined(lines, cautions, indent)


def controls_full(controls: Any, indent: int = 2) -> str:
    """Question 1, full: every count behind the control's rates, under ``indent``."""
    return control_full(controls, indent)


def skipped_strata(controls: Any) -> str:
    """``not judged in biological repeat 2B (no positive-control field with nuclei)``."""
    pair_cols = tuple(controls.pair_cols)
    reasons: dict[str, list[str]] = {}
    for entry in controls.settings.get("skipped_strata", []):
        reasons.setdefault(entry["reason"], []).append(named_stratum(entry, pair_cols))
    return "; ".join(
        f"not judged in {listing(where, limit=2)} ({reason})" for reason, where in reasons.items()
    )


# --- 3. What is limiting my sampling? -----------------------------------------------


def clustered_states(clustering: Sequence[Any]) -> tuple[list[str], int, int]:
    """The clustered states in words, and in how many groups where seen they cluster."""
    names, clustered, seen = [], 0, 0
    for result in clustering:
        if result is None:
            continue
        rows = result.table[result.table["state_objects"] > 0]
        words = unit_plural(result.settings.get("observation_unit", "object"))
        for state, block in rows.groupby("state", sort=False):
            hits = int((block["p_clustered"] <= 0.05).sum())
            if hits:
                word = "technical repeat" if TECHNICAL_REPEAT_COLUMN in result.group_cols else ""
                names.append(
                    f"{state_words(result, state, words)} clustered in {hits} of "
                    f"{count(len(block), word or 'group')}"
                )
            clustered += hits
            seen += len(block)
    return names, clustered, seen


def largest_sources(profile: pd.DataFrame) -> tuple[str | None, bool]:
    """Where most of the variance of the mean lies over the strata (from the profile's
    statements), and whether that is not established in some stratum."""
    rows = profile[profile["item"] == _LARGEST] if len(profile) else profile
    if not len(rows):
        return None, False
    pattern = re.compile(r"of the mean is (between [^;(]+)")
    found = [m.group(1).strip() for m in map(pattern.search, rows["statement"]) if m]
    uncertain = bool(rows["statement"].str.contains("not established").any())
    if not found:
        return None, uncertain
    level, times = Counter(found).most_common(1)[0]
    # The strata by their column (``condition=MOCK``: conditions), when there is one.
    columns = {str(w).split("=")[0] for w in rows["where"] if "=" in str(w) and "," not in str(w)}
    word = columns.pop() if len(columns) == 1 else "stratum"
    where = among(times, len(rows), word)
    return f"most variance {level} in {where}", uncertain


def unstable_labels(unusual: Any) -> str | None:
    """``7 of 12 groups`` where refits agree on fewer than 90% of the unusual labels."""
    if unusual is None or "seed_agreement" not in unusual.groups:
        return None
    agreement = pd.to_numeric(unusual.groups["seed_agreement"], errors="coerce").dropna()
    low = int((agreement < SEED_AGREEMENT_WARNING).sum())
    return among(low, len(agreement)) if low else None


def _variance_line(effort: Any, observations: str) -> tuple[str | None, list[str]]:
    """``Largest share of the variance of the mean: between samples in LATS1, YAP and ARAP2
    (79-99%); ...``, by level over the strata, and the caveat of a share not established."""
    if effort is None or effort.components.empty:
        return None, []
    hierarchy = effort.settings.get("hierarchy", {})
    words = level_words(hierarchy, observations)
    strata = list(effort.settings.get("strata") or [])
    levels: dict[str, list[tuple[str, float]]] = {}
    cautions = []
    for _, block in effort.components.groupby("stratum", sort=False):
        top = block.dropna(subset=["share"]).sort_values("share", ascending=False)
        if not len(top):
            continue
        first = top.iloc[0]
        where = group_label({c: first[c] for c in strata if c in first}) if strata else "pooled"
        level = words.get(first["level"], first["level"])
        levels.setdefault(level, []).append((where, float(first["share"])))
        caveat = share_caveat(first, int(first["units"]), unit_word(hierarchy, str(first["unit"])))
        if caveat:
            cautions.append(f"the largest share of the variance of the mean is {caveat[2:]}")
    if not levels:
        return None, []
    entries = [
        f"{level} in {listing([w for w, _ in found])} ({span([v for _, v in found], 'percent')})"
        for level, found in levels.items()
    ]
    return f"Largest share of the variance of the mean: {'; '.join(entries)}.", cautions


def _fields_line(profile: pd.DataFrame, variability: Any, observations: str) -> list[str]:
    """How much more a mean from whole fields varies, over the groups (or per stratum)."""
    if variability is not None:
        ok = variability.groups[variability.groups["status"] == "ok"]
        if not len(ok):
            return []
        effects = ok["design_effect"].astype(float)
        shares = pd.to_numeric(ok["unusual_fov_share"], errors="coerce").dropna()
        differ = (
            f"; {percent(shares.median())} of fields differ more than random {observations} "
            "would (5% expected)"
            if len(shares)
            else ""
        )
        spread = f"{effects.min():.1f}-{effects.max():.1f} over {count(len(ok), 'group')}"
        text = (
            f"Fields: a mean from whole fields varies {effects.median():.1f} times as much as one "
            f"from as many random {observations} (median; {spread}){differ}."
        )
        return [text]
    rows = profile[profile["item"] == "field-to-field heterogeneity"] if len(profile) else profile
    return [f"Fields ({row.where}): {row.statement}." for row in rows.itertuples()]


def states_lines(clustering: Any, unusual: Any, words: str) -> list[str]:
    """The labelled states over the fields, one line per state: where they cluster."""
    if clustering is None:
        return []
    keys = list(clustering.group_cols)
    share = ""
    if unusual is not None:
        ok = unusual.groups[unusual.groups["status"] == "ok"]
        if len(ok):
            shares = span([float(v) for v in ok["unusual_share"]], "percent")
            settings = unusual.settings
            reference = "/".join(str(v) for v in dict(settings["reference"]).values())
            share = (
                f" (farther from the {reference} reference than all but "
                f"{percent(settings['quantile'])} of its {words}; {shares} of each group)"
            )
    lines = []
    for state, block in clustering.table.groupby("state", sort=False):
        seen = block[block["state_objects"] > 0]
        if not len(seen):
            continue
        hits = seen[seen["p_clustered"] <= 0.05]
        named = [group_label({k: r[k] for k in keys}) for r in hits.to_dict("records")]
        where = f" ({listing(named, limit=3)})" if named else ""
        text = (
            f"clustered in a few fields (p ≤ 0.05) in {len(hits)} of the {len(seen)} groups "
            f"where seen{where}"
        )
        factor = (
            pd.to_numeric(hits["fovs_needed"], errors="coerce")
            / pd.to_numeric(hits["fovs_needed_if_random"], errors="coerce")
        ).dropna()
        if len(factor):
            text += (
                f"; there, seeing them takes about {factor.median():.1f} times the fields a "
                f"count of {words} predicts"
            )
        # The True state of the unusual labels is the unusual objects the share describes.
        own = share if isinstance(state, bool | np.bool_) and state else ""
        name = state_words(clustering, state, words)
        lines.append(f"{name[:1].upper()}{name[1:]}{own}: {text}.")
    return lines


def _short_statement(row: Any) -> str:
    """A profile statement without the clause that repeats its method (full keeps it)."""
    text = str(row.statement)
    if row.source == "inspect_fovs" and ": the declared reference's" in text:
        text = text.split(": the declared reference's")[0] + ")"
    return text


def profile_short(
    profile: pd.DataFrame,
    *,
    variability: Any = None,
    effort: Any = None,
    unusual: Any = None,
    clustering: Any = None,
    frame: str | None = None,
    indent: int = 2,
) -> str:
    """Question 3, short (see the module): one line per measured quantity, each caveat once."""
    units = [
        r.settings.get("observation_unit")
        for r in (variability, unusual, clustering, effort)
        if r is not None and r.settings.get("observation_unit")
    ]
    observations = unit_plural(units[0] if units else "object")
    texts = _fields_line(profile, variability, observations)
    texts += states_lines(clustering, unusual, observations)
    variance, cautions = _variance_line(effort, observations)
    texts += [variance] if variance else []
    for row in profile.itertuples() if len(profile) else []:
        if row.source == "redundancy_distance":
            texts.append(f"Neighbouring fields: {row.statement}.")
        elif row.source == "inspect_fovs":
            statement = _short_statement(row)
            texts.append(f"{statement[:1].upper()}{statement[1:]}.")
    if len(profile) and (profile["source"] == "representation_control").any():
        texts.append("How well the declared control is seen: question 1.")
    lines = [wrap(f"- {text}", indent, hang=indent + 2) for text in texts]
    if clustering is not None:
        cautions += clustering_parts(clustering, indent)[1]
    if unusual is not None:
        cautions += unusual_parts(unusual, indent)[1]
    if frame:
        cautions.append(frame)
    return _joined(lines, cautions, indent)


def profile_full(
    profile: pd.DataFrame,
    unusual: Any = None,
    clustering: Any = None,
    frame: str | None = None,
    indent: int = 2,
) -> str:
    """Question 3, full: every statement of the sampling profile with where it holds and its
    source, then the unusual objects and the states over the fields group by group."""
    blocks = []
    if len(profile):
        lines = [
            wrap(
                "Measured quantities (the sampling profile), each with where it holds and its "
                "source:",
                indent,
                hang=indent + 2,
            )
        ]
        for row in profile.itertuples():
            text = f"- {row.statement} [{row.where}; {row.source}]"
            lines.append(wrap(text, indent, hang=indent + 2))
        blocks.append("\n".join(lines))
    if unusual is not None:
        _, cautions = unusual_parts(unusual)
        text = unusual_full(unusual, indent)
        blocks.append("\n".join([text, *caveat_lines(cautions, indent=indent, detail="full")]))
    if clustering is not None:
        lines, cautions = clustering_parts(clustering, indent, full=True)
        blocks.append(_joined(lines, cautions, indent, "full"))
    if frame:
        blocks.append(wrap(frame, indent, hang=indent + 2))
    return "\n\n".join(blocks)


def _unit(*results: Any) -> str:
    """The observation unit the results declare (``object`` when none does)."""
    for result in results:
        if isinstance(result, Sequence):
            found = _unit(*result)
            if found != "object":
                return found
        elif result is not None and result.settings.get("observation_unit"):
            return str(result.settings["observation_unit"])
    return "object"


def _seen_states(clustering: Sequence[Any], words: str) -> str:
    """The states seen, in words: ``unusual cells``, ``mitotic cells and unusual cells``."""
    names = [
        state_words(result, state, words)
        for result in clustering
        if result is not None
        for state in result.table.loc[result.table["state_objects"] > 0, "state"].unique()
    ]
    return listing(list(dict.fromkeys(names)), 3) or "labelled states"


def profile_answer(profile: pd.DataFrame, clustering: Sequence[Any], unusual: Any) -> list[str]:
    """The parts of question 3's headline (``report_status.profile_status``)."""
    names, clustered, seen = clustered_states(clustering)
    parts = []
    fields = profile[profile["item"] == "fields against random objects"] if len(profile) else []
    if len(fields):
        words = unit_plural(_unit(clustering, unusual))
        effect = float(fields["value"].iloc[0])
        parts.append(f"whole fields {effect:.1f}x as variable as random {words}")
    parts += names
    if seen and not clustered:
        words = unit_plural(_unit(clustering, unusual))
        where = "the one group seen" if seen == 1 else f"all {seen} groups seen"
        parts.append(f"{_seen_states(clustering, words)} spread like random {words} in {where}")
    # The main answer only: whether the largest share is established and whether the
    # unusual labels are stable are cautions of the section (and raise its status).
    largest, _ = largest_sources(profile)
    if largest:
        parts.append(largest)
    return parts


__all__ = [
    "clustered_states",
    "controls_full",
    "controls_short",
    "largest_sources",
    "profile_answer",
    "profile_full",
    "profile_short",
    "rule_held",
    "skipped_strata",
    "states_lines",
    "unit_full",
    "unit_short",
    "unstable_labels",
]
