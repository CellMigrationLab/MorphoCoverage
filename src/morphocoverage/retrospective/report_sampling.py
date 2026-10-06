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

The helpers ``rule_held``, ``rule_unchecked``, ``spread_headline``,
``skipped_strata``, ``clustered_states``, ``largest_sources`` and
``unstable_labels`` give ``report_status`` the counts its headlines read.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import caveat_lines, once
from .._control import control_name
from .._style import among, group_label, span, wrap
from .._wording import count, listing, percent
from ..metadata import TECHNICAL_REPEAT_COLUMN
from ..sampling_unit._state_text import (
    CLUSTER_ALPHA,
    SEED_AGREEMENT_WARNING,
    clustering_parts,
    scored_shares,
    single_feature,
    state_words,
    tested,
    unusual_full,
    unusual_parts,
)
from ..sampling_unit._state_text import clustered as clustered_rows
from ..sampling_unit._unit_text import (
    descriptor_spread_full,
    descriptor_spread_parts,
    object_rule_full,
    object_rule_parts,
    one_per_field_rows,
    variability_full,
    variability_parts,
)
from ..schema import observation_words
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


def rule_unchecked(object_rules: Any) -> tuple[int, str]:
    """The groups where the mean-precision rule could not be checked at any target, and why
    in a few words: counted apart in the Foundation's headline, neither held nor missed."""
    table = object_rules.table
    keys = list(object_rules.group_cols)
    if not len(table):
        return 0, ""
    ok = table["status"] == "ok"
    if keys:
        by = [table[k] for k in keys]
        checked = ok.groupby(by, sort=False, dropna=False).transform("any").astype(bool)
    else:
        checked = pd.Series(bool(ok.any()), index=table.index)
    unchecked = table[~checked]
    groups = len(unchecked[keys].drop_duplicates()) if keys else int(len(unchecked) > 0)
    return groups, "; ".join(once([_few_words(str(text)) for text in unchecked["status"]]))


def _few_words(reason: str) -> str:
    """``not applicable`` for ``not applicable: the reference mean is 0, ...``."""
    return reason.split(":")[0].split(", so ")[0]


def spread_headline(descriptor_spread: Any, word: str) -> tuple[bool, str]:
    """The Foundation's descriptor-spread phrase, and whether whole fields never settled in
    some group; groups too small to check are counted apart (#272, owner, 2026-10-05)."""
    rows = descriptor_spread.at_threshold
    keys = list(descriptor_spread.group_cols)
    if len(rows) and keys:
        total = len(rows[keys].drop_duplicates())
        missed = len(rows[rows["objects_to_match"].isna()][keys].drop_duplicates())
    else:
        total, missed = (
            int(len(rows) > 0),
            int(len(rows) > 0 and rows["objects_to_match"].isna().any()),
        )
    skipped = len(descriptor_spread.settings.get("skipped_groups", []))
    words = observation_words(descriptor_spread.settings)[1]
    why = f"fewer than {descriptor_spread.settings.get('threshold')} {words}"
    if missed:
        never = (
            among(missed, total, word)
            if not skipped
            else f"{missed} of the {count(total, word)} assessed"
        )
        apart = f"; {skipped} more not assessed ({why})" if skipped else ""
        return True, f"descriptors from whole fields never settled in {never}{apart}"
    if skipped:
        return (
            False,
            f"descriptor spread not assessed in {among(skipped, total + skipped, word)} ({why})",
        )
    return False, ""


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
    """The clustered states in words, and in how many groups where tested they cluster (seen,
    with a smallest possible p that can reach 0.05; audit 10)."""
    names, clustered, seen = [], 0, 0
    for result in clustering:
        if result is None:
            continue
        rows = result.table[(result.table["state_objects"] > 0) & tested(result.table)]
        words = observation_words(result.settings)[1]
        for state, block in rows.groupby("state", sort=False):
            hits = int(clustered_rows(block).sum())
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
        if not len(ok):  # one observation per field: the fields are the observations
            rows = one_per_field_rows(variability, observations)
            return [f"Fields: {row['statement']}." for row in rows]
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


def _unusual_share(unusual: Any, words: str) -> str:
    """`` (farther from the siCTRL reference than all but 1% of its cells; 0-9% of each
    scored group)``, empty when no group was scored."""
    ok = unusual.groups[unusual.groups["status"] == "ok"] if unusual is not None else []
    if not len(ok):
        return ""
    if single_feature(unusual):  # it restates the mean shift (audit 10)
        return " (on one feature: not informative beyond the mean shift in question 9)"
    shares = span(scored_shares(unusual), "percent")
    settings = unusual.settings
    # Each against its own control with several; names joined by "/", never a tuple.
    reference = (
        "each against its own control, farther from it"
        if settings.get("own")
        else f"farther from the {control_name(settings['reference'])} reference"
    )
    exploratory = "exploratory, " if settings.get("exploratory") else ""
    return (
        f" ({exploratory}{reference} than all but {percent(settings['quantile'])} of its "
        f"{words}; {shares} of each scored group)"
    )


def states_lines(clustering: Any, unusual: Any, words: str) -> list[str]:
    """The labelled states over the fields, one line per state: where they cluster; without
    clustering (one observation per field), the unusual observations' share alone."""
    if clustering is None:
        # One word for the observations: "unusual images ... all but 1% of its images" (A22).
        named = unusual.settings.get("words", words) if unusual is not None else words
        share = _unusual_share(unusual, named)
        return [f"Unusual {named}{share}."] if share else []
    keys = list(clustering.group_cols)
    share = _unusual_share(unusual, words)
    lines = []
    for state, block in clustering.table.groupby("state", sort=False):
        seen = block[block["state_objects"] > 0]
        if not len(seen):
            continue
        hits = seen[clustered_rows(seen)]
        named = [group_label({k: r[k] for k in keys}) for r in hits.to_dict("records")]
        where = f" ({listing(named, limit=3)})" if named else ""
        run = int(tested(seen).sum())
        text = (
            f"clustered in a few fields (q ≤ {CLUSTER_ALPHA:g}) in {len(hits)} of the "
            f"{run} groups where tested{where}"
            if run
            else "not tested: too few in any group for a p that can reach 0.05"
        )
        if run and len(seen) > run:
            text += f", {len(seen) - run} more where seen too few to reach 0.05"
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
    for method in (": the declared reference's", ": this definition of new"):
        if row.source == "inspect_fovs" and method in text:
            text = text.split(method)[0]
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
        observation_words(r.settings)[1]
        for r in (variability, unusual, clustering, effort)
        if r is not None and r.settings.get("observation_unit")
    ]
    observations = units[0] if units else observation_words({})[1]
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
                "Measured quantities (the sampling profile), each with where it holds:",
                indent,
                hang=indent + 2,
            )
        ]
        for row in profile.itertuples():
            # Where it holds, never the internal function's name (audit 10, A22).
            text = f"- {row.statement} [{row.where}]"
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
    """The observations the results declare, plural (``cells`` when none does)."""
    for result in results:
        if isinstance(result, Sequence):
            found = _unit(*result)
            if found != observation_words({})[1]:
                return found
        elif result is not None and result.settings.get("observation_unit"):
            return observation_words(result.settings)[1]
    return observation_words({})[1]


def _seen_states(clustering: Sequence[Any], words: str) -> str:
    """The states seen, in words: ``unusual cells``, ``mitotic cells and unusual cells``."""
    names = [
        state_words(result, state, words)
        for result in clustering
        if result is not None
        for state in result.table.loc[result.table["state_objects"] > 0, "state"].unique()
    ]
    return listing(list(dict.fromkeys(names)), 3) or "labelled states"


def unusual_alone(unusual: Any, clustering: Sequence[Any] = ()) -> str:
    """``unusual images 0-8% of each scored group (exploratory)`` when unusual observations were
    scored without a clustering in fields (one observation per field), else empty."""
    if unusual is None or any(c is not None for c in clustering):
        return ""
    ok = unusual.groups[unusual.groups["status"] == "ok"]
    if not len(ok) or single_feature(unusual):
        return ""
    shares = span(scored_shares(unusual), "percent")
    words = unusual.settings.get("words") or observation_words(unusual.settings)[1]
    return f"unusual {words} {shares} of each scored group (exploratory)"


def profile_answer(
    profile: pd.DataFrame, clustering: Sequence[Any], unusual: Any, variability: Any = None
) -> list[str]:
    """The parts of question 3's headline (``report_status.profile_status``), in the words of
    the observations the results declare (``variability`` too: never "random cells" in a
    report on images, audit 10, C18)."""
    names, clustered, seen = clustered_states(clustering)
    parts = []
    fields = profile[profile["item"] == "fields against random objects"] if len(profile) else []
    if len(fields):
        words = _unit(clustering, unusual, variability)
        effect = float(fields["value"].iloc[0])
        # One observation per field: fields and observations coincide, no ratio to give (B21).
        single = str(fields["statement"].iloc[0]).startswith("each field holds one of")
        parts.append(
            f"one of the {words} per field"
            if single
            else f"whole fields {effect:.1f}x as variable as random {words}"
        )
    parts += [alone] if (alone := unusual_alone(unusual, clustering)) else []
    parts += names
    if seen and not clustered:
        # Not "spread like random": a test that finds nothing shows no clustering (audit 10).
        words = _unit(clustering, unusual, variability)
        where = "the one group tested" if seen == 1 else f"the {seen} groups tested"
        parts.append(f"no clustering shown for {_seen_states(clustering, words)} in {where}")
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
    "rule_unchecked",
    "skipped_strata",
    "spread_headline",
    "states_lines",
    "unit_full",
    "unit_short",
    "unstable_labels",
    "unusual_alone",
]
