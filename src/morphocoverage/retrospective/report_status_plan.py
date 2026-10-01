"""The statuses of the report's planning answers, questions 4, 5 and 7 (see ``report_status``).

``report_status`` documents every question's rule and re-exports these; they
live apart only to keep each module under 600 lines.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._style import Status, among
from .._wording import count, fields, listing, number, percent, plural
from ._text import column_word, sample_word
from .adequacy import ALL_CRITERIA
from .adequacy_text import _tail_value, _unjudged
from .effort_text import LEANING_SHARE
from .protocol_text import protocol_answers, served_cell

# Fewer independent units than this make a protocol's field count uncertain.
FEW_UNITS = 5
# The share of random acquisitions that must be enough for a per-group curve to count as
# reached (a provisional preset, as protocol_reliability's threshold).
RELIABLE_SHARE = 0.9

Answer = tuple[Status, str]


def _reliable_from(rows: pd.DataFrame) -> int | None:
    """The fewest listed fields from which a curve stays at ``RELIABLE_SHARE`` or above."""
    found = None
    for row in rows.sort_values("fovs").itertuples(index=False):
        if getattr(row, "splits", 1) == 0 or pd.isna(row.preserved):
            continue
        if float(row.preserved) >= RELIABLE_SHARE - 1e-9:
            found = int(row.fovs) if found is None else found
        else:
            found = None
    return found


def _reliability_status(reliability: Any) -> Answer:
    """Question 4 from the per-group curves alone: never ✅, as nothing was checked across
    samples; ❌ when no criterion reached ``RELIABLE_SHARE`` in every group."""
    curve, keys = reliability.curve, list(reliability.group_cols)
    labels = reliability.settings.get("labels", {})
    if curve.empty:
        return Status.NOT_ASSESSED, "not assessed"
    reached, missed = [], []
    for criterion, rows in curve.groupby("criterion", sort=False):
        parts = rows.groupby(keys, sort=False, dropna=False) if keys else [((), rows)]
        points = [_reliable_from(own) for _, own in parts]
        name = labels.get(criterion, criterion)
        if all(point is not None for point in points):
            reached.append(max(points))
        else:
            missed.append(name)
    total = len(reached) + len(missed)
    groups = len(curve[keys].drop_duplicates()) if keys else 1
    where = f"in {among(groups, groups)}" if groups > 1 else "in the group"
    share = percent(RELIABLE_SHARE)
    if not reached:
        return Status.PROBLEM, f"{share} of acquisitions enough {where} for no target"
    text = (
        f"{fields(max(reached))}: {share} of acquisitions enough {where} for "
        f"{among(len(reached), total, 'target')}"
    )
    if missed:
        text += f"; not for {listing(missed, limit=2)}"
    return Status.CAUTION, f"{text}; not checked across samples"


def next_time_status(reliability: Any, protocol: Any) -> Answer:
    """4. How many fields should I image next time? (see the module)."""
    if protocol is None:
        if reliability is None:
            return Status.NOT_ASSESSED, "not assessed"
        return _reliability_status(reliability)
    answers = protocol_answers(protocol)
    if not answers:
        return Status.NOT_ASSESSED, "not assessed"
    words = protocol.settings.get("hierarchy", {})
    sample = sample_word(words)
    unit = column_word(protocol.settings.get("independent_unit", "unit"))
    by_criterion: dict[str, list[dict[str, Any]]] = {}
    for answer in answers:
        by_criterion.setdefault(answer["label"], []).append(answer)
    reached, shown, unjudged, short = [], [], [], []
    for label, own in by_criterion.items():
        served = [answer["served"] for answer in own]
        cells = {served_cell(a) for a in own if a["served"] is None}
        if all(point is not None for point in served):
            reached.append(max(served))
        elif "not reached" in cells:
            short.append(label)
        elif "not shown" in cells:
            shown.append(label)  # capped by the held-back fields, never short of fields
        else:
            unjudged.append(label)
    units = min(answer["units"] for answer in answers)
    total = len(by_criterion)
    if not reached and short:
        status = Status.PROBLEM
    elif len(reached) < total:
        # Few biological repeats is the usual design, stated in the line, not a caution.
        status = Status.CAUTION
    else:
        status = Status.OK
    if reached:
        text = (
            f"{fields(max(reached))} served every {sample} for "
            f"{among(len(reached), total, 'target')}"
        )
    else:
        text = f"no field count served every {sample}"
    # The targets that were not served, counted: their names are in the section's table.
    for missed, what in (
        (short, "not reached"),
        (shown, "not shown (held-back fields cap it)"),
        (unjudged, "not judged"),
    ):
        if missed:
            text += f"; {among(len(missed), total, 'target')} {what}"
    if units < FEW_UNITS:
        text += f"; from {count(units, unit, plural(unit))}"
    return status, text


def bought_status(adequacy: Any) -> Answer:
    """5. What did the extra fields buy? (see the module)."""
    if adequacy is None:
        return Status.NOT_ASSESSED, "not assessed"
    rows = adequacy.realized[adequacy.realized["criterion"] == ALL_CRITERIA]
    rows = rows[~_unjudged(rows)]
    if rows.empty:
        return Status.NOT_ASSESSED, "not assessed"
    sustained = rows["sustained_realized"].dropna().astype(float)
    if sustained.empty:
        return Status.CAUTION, "held-back fields never reliably confirmed everything wanted"
    head = (
        f"held-back fields confirmed everything wanted from field {number(sustained.median())} "
        "(median)"
    )
    tail = _tail_value(rows["sustained_realized"])
    if len(sustained) < len(rows):
        return Status.CAUTION, f"{head}; never in {percent(1 - len(sustained) / len(rows))} of runs"
    return Status.OK, f"{head}; by field {number(tail)} in 90% of runs"


def effort_status(effort: Any) -> Answer:
    """7. More fields or more samples? (see the module)."""
    if effort is None or effort.comparisons.empty:
        return Status.NOT_ASSESSED, "not assessed"
    strata_cols = list(effort.settings.get("strata", []))
    word = " · ".join(column_word(c) for c in strata_cols) or "stratum"
    sample = sample_word(effort.settings.get("hierarchy", {}))
    strata = effort.comparisons.groupby("stratum", sort=False)["resolved"].any()
    resolved, total = int(strata.sum()), len(strata)
    top = _top_levels(effort)
    between = sum(level == effort.settings.get("hierarchy", {}).get("sample") for level in top)
    most = (
        f"; most variance between {plural(sample)} in {among(between, len(top), word)}"
        if top
        else ""
    )
    if resolved == total:
        return Status.OK, f"a better split of the same fields found in {among(total, total, word)}"
    units = sorted({int(u) for u in effort.comparisons["units"]})
    few = count(units[0], sample) if len(units) == 1 else f"{units[0]}-{count(units[-1], sample)}"
    where = "" if resolved == 0 else f" in {among(total - resolved, total, word)}"
    if units[-1] < FEW_UNITS:
        # Few samples is the design: answered as far as it goes, with the estimates' reading.
        leaning_strata = _leaning_strata(effort, effort.settings.get("hierarchy", {}).get("sample"))
        leaning = (
            f"; the estimates lean to more {plural(sample)} in {among(leaning_strata, total, word)}"
            if leaning_strata
            else ""
        )
        return Status.OK, f"not established{where} with {few} per {word}{most}{leaning}"
    return Status.CAUTION, f"cannot tell{where} with {few} per {word}{most}"


def _leaning_strata(effort: Any, level: Any) -> int:
    """The strata whose largest variance share (at least ``LEANING_SHARE``) is ``level``."""
    shares = effort.components.dropna(subset=["share"])
    if shares.empty:
        return 0
    ranked = shares.sort_values("share", ascending=False).groupby("stratum", sort=False).head(1)
    return int(((ranked["level"] == level) & (ranked["share"] >= LEANING_SHARE)).sum())


def _top_levels(effort: Any) -> list[str]:
    """Each stratum's level with the largest share of the variance of the mean."""
    shares = effort.components.dropna(subset=["share"])
    if shares.empty:
        return []
    ranked = shares.sort_values("share", ascending=False).groupby("stratum", sort=False)
    return [str(rows["level"].iloc[0]) for _, rows in ranked]


__all__ = ["FEW_UNITS", "RELIABLE_SHARE", "bought_status", "effort_status", "next_time_status"]
