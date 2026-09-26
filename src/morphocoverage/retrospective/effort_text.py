"""The effort projection in words: level names, caveats and the per-stratum summary.

Kept apart from the estimation (``effort``) so that each stays readable. When
the sample column is the MorphoBricks ``technical_repeat`` (a well, a dish)
the sentences call it a technical repeat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._wording import count, percent
from ._text import sample_word, unit_plural, unit_word
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
        reasons.append(f"only {units} {unit_word}{'s' if units != 1 else ''}")
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


def _imbalance_line(projections: pd.DataFrame, samples: str, observations: str) -> str:
    """The cost of the acquired imbalance: the acquired against the balanced SE, or ''."""
    current = projections[projections["current"]].set_index("design")
    if not {"acquired", "balanced"} <= set(current.index):
        return ""
    acquired, balanced = float(current.loc["acquired", "se"]), float(current.loc["balanced", "se"])
    if not balanced > 0 or abs(acquired / balanced - 1) < _IMBALANCE_SHOWN:
        return ""
    direction = "larger" if acquired > balanced else "smaller"
    return (
        f"  The acquired {samples}s differ in size: the standard error of the mean is "
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
            f"{best['fovs_per_sample']:.3g} fields would give a {100 * abs(change):.0f}% "
            f"{direction} standard error than the acquired {best['current_samples']} "
            f"{samples}s of about {best['current_fovs_per_sample']:.3g}, made balanced."
        )
    if units < FEW_UNITS:
        reason = f"only {units} {unit}s"
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
            "the same share would again."
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


def effort_summary(result: EffortResult) -> str:
    """ "More fields or more samples?" in sentences, per stratum."""
    feature = result.settings.get("feature", "the measurement")
    hierarchy = result.settings.get("hierarchy", {})
    observations = unit_plural(result.settings.get("observation_unit", "object"))
    words = level_words(hierarchy, observations)
    samples = sample_word(hierarchy)
    repeat = hierarchy.get("biological_repeat")
    blocks = []
    for stratum, rows in result.components.groupby("stratum", sort=False):
        lines = [f"{stratum}: more fields or more {samples}s, for the mean {feature}?"]
        first = rows.iloc[0]
        shares = rows.dropna(subset=["share"]).sort_values("share", ascending=False)
        units = int(first["units"])
        unit = unit_word(hierarchy, str(first["unit"]))
        dropped = _repeat_line(first, repeat, samples, observations)
        if dropped:
            lines.append(dropped)
        if len(shares):
            top = shares.iloc[0]
            # "Most" only for more than half: the largest of three shares can be 40%.
            lead = "Most of" if top["share"] > 0.5 else "The largest part of"
            lines.append(
                f"  {lead} the uncertainty of the mean is "
                f"{words.get(top['level'], top['level'])} ({100 * top['share']:.0f}% of its "
                f"variance with the acquired design{share_caveat(top, units, unit)})."
            )
        projections = result.projections[result.projections["stratum"] == stratum]
        imbalance = _imbalance_line(projections, samples, observations)
        if imbalance:
            lines.append(imbalance)
        compared = result.comparisons[result.comparisons["stratum"] == stratum]
        lines.append(_allocation_line(compared, first, units, unit, samples))
        normalized = result.settings.get("normalized")
        if normalized:
            lines.append(
                f"  Values are normalised ({normalized}): variance shared by a whole repeat "
                "is removed, so the between-repeat share is lower than in raw values."
            )
        lines += _counts_lines(first, samples, observations)
        blocks.append("\n".join(lines))
    for row in result.skipped.itertuples(index=False):
        blocks.append(
            f"{row.stratum}: more fields or more {samples}s, for the mean {feature}? "
            f"Not estimable: {row.reason}."
        )
    return "\n\n".join(blocks)


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
