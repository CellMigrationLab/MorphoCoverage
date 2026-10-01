"""The effort projection in words: level names, caveats and the per-stratum summary.

Kept apart from the estimation (``effort``) so that each stays readable. When
the sample column is the MorphoBricks ``technical_repeat`` (a well, a dish)
the sentences call it a technical repeat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._caveats import assumption, caveat_lines
from .._style import (
    check_detail,
    group_columns,
    group_label,
    merge_numbers,
    number_pattern,
    table,
    wrap,
)
from .._wording import count, listing, percent
from ._text import column_word, sample_word, unit_plural, unit_word
from .protocol import FEW_UNITS

if TYPE_CHECKING:
    from .effort import EffortResult

OBSERVATION = "observation"
# Why the repeat level was left out of a stratum (``repeat_dropped_reason``).
SINGLE_REPEAT = "single repeat"
ONE_SAMPLE_PER_REPEAT = "one sample per repeat"
# Acquired and balanced standard errors closer than this are not called an imbalance cost.
_IMBALANCE_SHOWN = 0.01


def level_words(hierarchy: Mapping[str, Any], observations: str) -> dict[str | None, str]:
    """Where a variance component sits, by level column, in words."""
    sample = sample_word(hierarchy)
    return {
        hierarchy.get("fov"): f"between fields within a {sample}",
        hierarchy.get("sample"): f"between {sample}s",
        hierarchy.get("biological_repeat"): "between repeats",
        OBSERVATION: f"between {observations} within fields",
    }


def _no_interval(component: Mapping[str, Any]) -> str:
    """``no interval, as 152 of 500 resamples could not be estimated``."""
    failed, n_boot = component.get("boot_failed"), component.get("n_boot")
    if failed is None or n_boot is None or pd.isna(failed) or pd.isna(n_boot):
        return "no interval"
    return f"no interval, as {int(failed)} of {int(n_boot)} resamples could not be estimated"


def share_caveat(component: Mapping[str, Any], units: int, unit_word: str) -> str:
    """Why the largest variance share is not established, or an empty string.

    A share is a point estimate from the truncated components. With fewer
    than ``FEW_UNITS`` independent units, when the bootstrap interval of the
    largest component includes zero, or when there is no interval at all (too
    many resamples could not be estimated), the same data are consistent with
    that level contributing nothing (a null simulation with 3 samples and no
    between-sample variance attributed most of the variance to the samples
    in 4 of 20 runs), so the sentence says so.
    """
    low = component.get("low")
    missing = low is None or pd.isna(low)
    includes_zero = not missing and float(low) <= 0
    if units >= FEW_UNITS and not includes_zero and not missing:
        return ""
    reasons = []
    if units < FEW_UNITS:
        reasons.append(f"only {count(units, unit_word)}")
    if missing:
        reasons.append(_no_interval(component))
    elif includes_zero:
        reasons.append("its interval includes zero")
    return f"; not established: {', '.join(reasons)}"


def _repeat_line(first: Mapping[str, Any], repeat: str, samples: str, observations: str) -> str:
    """Why the repeat is not a variance level in this stratum, or an empty string."""
    reason = first.get("repeat_dropped_reason")
    if not bool(first["repeat_dropped"]):
        return ""
    acquired = first.get("repeats_acquired")
    if reason == ONE_SAMPLE_PER_REPEAT:
        return (
            f"  Every {repeat} holds a single {samples} here, so "
            f"the repeat is not a variance level in this stratum (its variance is counted "
            f"with the {samples}'s) and the {samples} is the independent unit."
        )
    if acquired is not None and not pd.isna(acquired) and int(acquired) > 1:
        return (
            f"  Only one of the {int(acquired)} {repeat}s here holds {observations}, so the "
            f"variance between repeats cannot be estimated: the components describe that "
            f"{repeat} only, and the {samples} is the independent unit."
        )
    return (
        f"  There is a single {repeat} here, so the variance between repeats cannot be "
        f"estimated: the components describe this {repeat} only, and the {samples} is "
        "the independent unit."
    )


def _unequal(first: Mapping[str, Any], samples: str) -> str:
    """Which acquired units differ in size: ``repeats``, ``technical repeats``, or both.

    Only the sample and repeat sizes separate the acquired from the balanced
    standard error (the field sizes are kept in both, through the cluster
    size): a level differs when its weight ``sum N_u^2 / N^2`` exceeds
    ``1 / units``.
    """
    parts = []

    def differs(weight: object, units: object) -> bool:
        if weight is None or units is None or pd.isna(weight) or pd.isna(units):
            return False
        return float(weight) * float(units) > 1 + 1e-9

    if not bool(first.get("repeat_dropped", True)) and differs(
        first.get("repeat_weight"), first.get("repeats")
    ):
        parts.append("repeats")
    if differs(first.get("sample_weight"), first.get("samples")):
        parts.append(f"{samples}s")
    return " and ".join(parts) or f"{samples}s"


def _imbalance_line(
    projections: pd.DataFrame, first: Mapping[str, Any], samples: str, observations: str
) -> str:
    """The cost of the acquired imbalance: the acquired against the balanced SE, or ''."""
    current = projections[projections["current"]].set_index("design")
    if not {"acquired", "balanced"} <= set(current.index):
        return ""
    acquired, balanced = float(current.loc["acquired", "se"]), float(current.loc["balanced", "se"])
    if not balanced > 0 or abs(acquired / balanced - 1) < _IMBALANCE_SHOWN:
        return ""
    direction = "larger" if acquired > balanced else "smaller"
    unequal = _unequal(first, samples)
    return (
        f"  The acquired {unequal} differ in size: the standard error of the mean is "
        f"{acquired:.3g} as acquired and would be {balanced:.3g} with the same {samples}s, "
        f"fields and {observations} spread evenly ({percent(abs(acquired / balanced - 1))} "
        f"{direction} as acquired); the comparisons below are with the balanced version."
    )


def _allocation_line(
    compared: pd.DataFrame, first: Mapping[str, Any], units: int, unit: str, samples: str
) -> str:
    """The resolved best allocation at the same total, or why the data cannot tell."""
    resolved = compared[compared["resolved"]]
    if len(compared) and len(resolved):
        best = resolved.sort_values("ratio").iloc[0]
        change = 1 - best["ratio"]
        direction = "smaller" if change > 0 else "larger"
        return (
            f"  At the same total number of fields, {best['samples']} {samples}s of about "
            f"{best['fovs_per_sample']:.3g} fields would give a {percent(abs(change))} "
            f"{direction} standard error than the acquired {best['current_samples']} "
            f"{samples}s of about {best['current_fovs_per_sample']:.3g}, made balanced."
        )
    if units < FEW_UNITS:
        reason = f"only {count(units, unit)}"
    elif not len(compared):
        reason = "no other allocation was compared"
    elif not compared["same_total"].any():
        reason = "no allocation compared has the acquired total of fields"
    elif compared.loc[compared["same_total"], "ratio_low"].isna().all():
        reason = _no_interval(first)
    else:
        reason = "the intervals of the alternatives at the same total include no change"
    return f"  More fields or more {samples}s: cannot tell from these data ({reason})."


def _counts_lines(first: Mapping[str, Any], samples: str, observations: str) -> list[str]:
    """Missing values, empty fields, samples without observations, failed resamples."""
    lines = []
    missing = int(first["missing_values"])
    if missing:
        lines.append(f"  {missing} {observations} without a value were left out.")
    fovs = int(first["fovs"])
    empty = fovs - int(first["fovs_with_observations"])
    if empty:
        lines.append(
            f"  {empty} of {fovs} acquired fields had no {observations}; projections assume "
            "the same share would be empty again."
        )
    acquired = first.get("samples_acquired")
    if acquired is not None and not pd.isna(acquired) and int(acquired) > int(first["samples"]):
        without = int(acquired) - int(first["samples"])
        lines.append(
            f"  {without} of {int(acquired)} {samples}s held no {observations} with a value: "
            f"their fields count as acquired effort, not as {samples}s."
        )
    failed = first.get("boot_failed")
    if failed and not pd.isna(failed) and not pd.isna(first.get("low")):
        lines.append(
            f"  {count(int(failed), 'resample')} of {int(first['n_boot'])} could not be "
            "estimated (their levels could not be separated) and are left out of the intervals."
        )
    return lines


def _stratum_group(result: EffortResult, rows: pd.DataFrame) -> dict[str, Any]:
    """The stratum's columns and values (``{"condition": "MOCK"}``), empty when pooled."""
    first = rows.iloc[0]
    return {c: first[c] for c in result.settings.get("strata", []) if c in rows.columns}


def _label(result: EffortResult, rows: pd.DataFrame) -> str:
    """The stratum as the summaries name it: ``MOCK``, or ``all samples`` when pooled."""
    group = _stratum_group(result, rows)
    return group_label(group) if group else "all samples"


def _words(result: EffortResult) -> dict[str, Any]:
    hierarchy = result.settings.get("hierarchy", {})
    observations = unit_plural(result.settings.get("observation_unit", "object"))
    return {
        "feature": result.settings.get("feature", "the measurement"),
        "hierarchy": hierarchy,
        "observations": observations,
        "levels": level_words(hierarchy, observations),
        "samples": sample_word(hierarchy),
        # The repeat level's column in words (``biological repeat``, ``plate``).
        "repeat": column_word(hierarchy.get("biological_repeat")),
    }


def _stratum_lines(result: EffortResult, stratum: str, rows: pd.DataFrame) -> list[str]:
    """One stratum in sentences, as the full summary gives it."""
    w = _words(result)
    samples = w["samples"]
    lines = [
        f"{_label(result, rows)}: more fields or more {samples}s, for the mean {w['feature']}?"
    ]
    first = rows.iloc[0]
    shares = rows.dropna(subset=["share"]).sort_values("share", ascending=False)
    units = int(first["units"])
    unit = unit_word(w["hierarchy"], str(first["unit"]))
    dropped = _repeat_line(first, w["repeat"], samples, w["observations"])
    if dropped:
        lines.append(dropped)
    if len(shares):
        top = shares.iloc[0]
        # "Most" only for more than half: the largest of three shares can be 40%.
        lead = "Most of" if top["share"] > 0.5 else "The largest part of"
        lines.append(
            f"  {lead} the uncertainty of the mean is "
            f"{w['levels'].get(top['level'], top['level'])} ({percent(top['share'])} of its "
            f"variance with the acquired design{share_caveat(top, units, unit)})."
        )
    projections = result.projections[result.projections["stratum"] == stratum]
    imbalance = _imbalance_line(projections, first, samples, w["observations"])
    if imbalance:
        lines.append(imbalance)
    compared = result.comparisons[result.comparisons["stratum"] == stratum]
    lines.append(_allocation_line(compared, first, units, unit, samples))
    lines += _counts_lines(first, samples, w["observations"])
    return lines


def _comparison_table(compared: pd.DataFrame, samples: str) -> str | None:
    """The allocations compared with the acquired one, balanced: one row each."""
    if not len(compared):
        return None
    rows = []
    for row in compared.itertuples(index=False):
        interval = "–" if pd.isna(row.ratio_low) else f"{row.ratio_low:.3g}-{row.ratio_high:.3g}"
        rows.append(
            [
                row.samples,
                f"{row.fovs_per_sample:.3g}",
                f"{row.ratio:.3g}",
                interval,
                "yes" if bool(row.same_total) else "no",
                "yes" if bool(row.resolved) else "no",
            ]
        )
    columns = [f"{samples}s", "fields each", "SE ratio", "95% interval", "same total", "resolved"]
    return table(rows, columns, indent=4, align="rrrrll")


def effort_summary(result: EffortResult, detail: str = "short") -> str:
    """ "More fields or more samples?": short, one table over the strata, or full, per stratum."""
    if check_detail(detail) == "full":
        lines, cautions = effort_full_parts(result)
    else:
        lines, cautions = effort_short_parts(result, indent=0)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0, detail=detail)])


def effort_full_parts(result: EffortResult, indent: int = 0) -> tuple[list[str], list[str]]:
    """Every stratum's sentences, with the allocations compared in a table, under ``indent``."""
    pad = " " * indent
    samples = _words(result)["samples"]
    blocks = []
    for stratum, rows in result.components.groupby("stratum", sort=False):
        lines = _stratum_lines(result, stratum, rows)
        compared = result.comparisons[result.comparisons["stratum"] == stratum]
        shown = _comparison_table(compared, samples)
        if shown:
            lines.append(
                "  Allocations compared: the standard error against the acquired design made "
                "balanced (resolved: its 95% interval excludes no change)."
            )
            lines.append(shown)
        blocks.append(lines)
    for row in result.skipped.itertuples(index=False):
        blocks.append([f"{_skipped_label(result, row)}: {_not_estimable(result, row)}"])
    lines = []
    for block in blocks:
        if lines:
            lines.append("")
        lines += [_rewrap(line, pad) for line in block]
    return lines, []


def _rewrap(line: str, pad: str) -> str:
    """A sentence of the summary wrapped at its own indent (tables kept as they are)."""
    if "\n" in line:
        return "\n".join(pad + part for part in line.splitlines())
    depth = len(line) - len(line.lstrip())
    return wrap(line.strip(), pad + " " * depth, hang=len(pad) + depth + 2)


def _skipped_label(result: EffortResult, row: Any) -> str:
    group = {c: getattr(row, c) for c in result.settings.get("strata", []) if hasattr(row, c)}
    return group_label(group) if group else str(row.stratum)


def _not_estimable(result: EffortResult, row: Any) -> str:
    w = _words(result)
    return (
        f"more fields or more {w['samples']}s, for the mean {w['feature']}? "
        f"Not estimable: {row.reason}."
    )


# A level holding at least this share of the variance of the mean makes the estimates lean to
# more of that level (more samples) or more fields, before the split can be established.
LEANING_SHARE = 0.75
FEW_NOTE = (
    "with few independent units the shares are point estimates (in a simulation with 3 "
    "samples and no variance between them, most variance was put between samples in 4 of 20 "
    "runs), so a leaning is a reading, not a result"
)


def _leaning(top: Mapping[str, Any], hierarchy: Mapping[str, Any], samples: str) -> str:
    """``leans to more samples`` (or fields) when one level holds ``LEANING_SHARE`` of the
    variance of the mean, else ''."""
    if pd.isna(top.get("share")) or float(top["share"]) < LEANING_SHARE:
        return ""
    if top["level"] == hierarchy.get("sample"):
        return f"leans to more {samples}s (not established)"
    if top["level"] == hierarchy.get("fov"):
        return "leans to more fields (not established)"
    return ""


def _allocation_cell(
    compared: pd.DataFrame, first: Mapping[str, Any], units: int, unit: str, samples: str
) -> tuple[str, str | None]:
    """The short table's answer, and why the data cannot tell (``None`` when they can)."""
    resolved = compared[compared["resolved"]]
    if len(compared) and len(resolved):
        best = resolved.sort_values("ratio").iloc[0]
        change = 1 - best["ratio"]
        direction = "smaller" if change > 0 else "larger"
        answer = (
            f"{best['samples']} {samples}s of about {best['fovs_per_sample']:.3g} fields: "
            f"{percent(abs(change))} {direction} SE"
        )
        return answer, None
    reason = _allocation_line(compared, first, units, unit, samples)
    return "cannot tell", reason.split("cannot tell from these data (", 1)[1][:-2]


def effort_short_parts(result: EffortResult, indent: int = 2) -> tuple[list[str], list[str]]:
    """One table over the strata, the skipped strata, and every caveat once (see ``effort``)."""
    w = _words(result)
    samples = w["samples"]
    pad = " " * indent
    rows, notes, reasons, shares = [], {}, {}, {}
    strata = result.components.groupby("stratum", sort=False)
    labels = []
    for stratum, own in strata:
        label = _label(result, own)
        labels.append(label)
        first = own.iloc[0]
        units = int(first["units"])
        unit = unit_word(w["hierarchy"], str(first["unit"]))
        ranked = own.dropna(subset=["share"]).sort_values("share", ascending=False)
        largest = "–"
        if len(ranked):
            top = ranked.iloc[0]
            place = w["levels"].get(top["level"], top["level"])
            largest = f"{percent(top['share'])} {place}"
            caveat = share_caveat(top, units, unit).removeprefix("; not established: ")
            if caveat:
                shares.setdefault(caveat, []).append(label)
        compared = result.comparisons[result.comparisons["stratum"] == stratum]
        answer, reason = _allocation_cell(compared, first, units, unit, samples)
        if reason and units < FEW_UNITS and len(ranked):
            # Too few units to establish it, but the estimates still read one way: say how.
            answer = _leaning(ranked.iloc[0], w["hierarchy"], samples) or answer
        if reason:
            reasons.setdefault(reason, []).append(label)
        rows.append([label, count(units, unit), largest, answer])
        projections = result.projections[result.projections["stratum"] == stratum]
        extra = [
            _repeat_line(first, w["repeat"], samples, w["observations"]),
            _imbalance_line(projections, first, samples, w["observations"]),
            *_counts_lines(first, samples, w["observations"]),
        ]
        for text in extra:
            if text:
                notes.setdefault(number_pattern(text.strip()), []).append((label, text.strip()))
    lines, cautions = [], []
    total = len(labels)
    if rows:
        lines.append(
            wrap(
                f"For the mean {w['feature']}: where most of the uncertainty of the mean sits "
                f"(its share of the variance with the acquired design), and whether more fields "
                f"or more {samples}s would help at the same total of fields:",
                indent,
                hang=indent + 2,
            )
        )
        strata_cols = [c for c in result.settings.get("strata", [])]
        head = group_columns(strata_cols) if strata_cols else "stratum"
        columns = [head, "independent units", "largest share", f"more fields or {samples}s?"]
        lines.append(table(rows, columns, indent=indent + 2))
    for reasons_of, lead in (
        (shares, "the largest share is not established"),
        (reasons, f"cannot tell whether more fields or more {samples}s would help"),
    ):
        for reason, where in reasons_of.items():
            text = f"{lead}{_where(where, total)}: {reason}"
            # Few independent units is the design, not a fault of these data: a note, not a
            # caution (the owner's rule: three repeats are a usual design).
            few = reason.startswith("only ")
            cautions.append(assumption(f"{text}; {FEW_NOTE}") if few else text)
    for alike in notes.values():
        texts = [text for _, text in alike]
        merged = merge_numbers(texts) if len(texts) > 1 else texts[0]
        cautions.append(f"{_where([label for label, _ in alike], total, lead=False)}{merged}")
    for row in result.skipped.itertuples(index=False):
        lines.append(
            wrap(
                f"{_skipped_label(result, row)}: {_not_estimable(result, row)}",
                pad,
                hang=indent + 2,
            )
        )
    return lines, cautions


def _where(labels: Sequence[str], total: int, *, lead: bool = True) -> str:
    """`` in MOCK and YAP`` (or ``In MOCK and YAP: ``), nothing when it holds everywhere."""
    if total <= 1 or len(labels) == total:
        return ""
    named = listing(list(labels), limit=3)
    return f" in {named}" if lead else f"In {named}: "


__all__ = [
    "OBSERVATION",
    "ONE_SAMPLE_PER_REPEAT",
    "SINGLE_REPEAT",
    "effort_summary",
    "level_words",
    "sample_word",
    "share_caveat",
    "unit_word",
]
