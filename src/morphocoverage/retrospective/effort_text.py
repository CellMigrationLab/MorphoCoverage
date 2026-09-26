"""The effort projection in words: level names, caveats and the per-stratum summary.

Kept apart from the estimation (``effort``) so that each stays readable. When
the sample column is the MorphoBricks ``technical_repeat`` (a well, a dish)
the sentences call it a technical repeat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from ._text import sample_word, unit_plural, unit_word
from .protocol import FEW_UNITS

if TYPE_CHECKING:
    from .effort import EffortResult

OBSERVATION = "observation"


def level_words(hierarchy: Mapping[str, Any], observations: str) -> dict[str | None, str]:
    """Where a variance component sits, by level column, in words."""
    sample = sample_word(hierarchy)
    return {
        hierarchy.get("fov"): f"between fields within a {sample}",
        hierarchy.get("sample"): f"between {sample}s",
        hierarchy.get("biological_repeat"): "between repeats",
        OBSERVATION: f"between {observations} within fields",
    }


def share_caveat(component: Mapping[str, Any], units: int, unit_word: str) -> str:
    """Why the largest variance share is not established, or an empty string.

    A share is a point estimate from the truncated components. With fewer
    than ``FEW_UNITS`` independent units, or when the bootstrap interval of
    the largest component includes zero, the same data are consistent with
    that level contributing nothing (a null simulation with 3 samples and no
    between-sample variance attributed most of the variance to the samples
    in 4 of 20 runs), so the sentence says so.
    """
    low = component.get("low")
    includes_zero = low is not None and not pd.isna(low) and float(low) <= 0
    if units >= FEW_UNITS and not includes_zero:
        return ""
    reasons = []
    if units < FEW_UNITS:
        reasons.append(f"only {units} {unit_word}{'s' if units != 1 else ''}")
    if includes_zero:
        reasons.append("its interval includes zero")
    return f"; not established: {', '.join(reasons)}"


def effort_summary(result: EffortResult) -> str:
    """ "More fields or more samples?" in sentences, per stratum."""
    feature = result.settings.get("feature", "the measurement")
    hierarchy = result.settings.get("hierarchy", {})
    observations = unit_plural(result.settings.get("observation_unit", "object"))
    words = level_words(hierarchy, observations)
    samples = sample_word(hierarchy)
    blocks = []
    for stratum, rows in result.components.groupby("stratum", sort=False):
        lines = [f"{stratum}: more fields or more {samples}s, for the mean {feature}?"]
        shares = rows.dropna(subset=["share"]).sort_values("share", ascending=False)
        units = int(rows["units"].iloc[0])
        unit = unit_word(hierarchy, str(rows["unit"].iloc[0]))
        repeat = hierarchy.get("biological_repeat")
        if bool(rows["repeat_dropped"].iloc[0]) and rows["repeats"].iloc[0] == 1:
            lines.append(
                f"  There is a single {repeat} here, so the variance between repeats cannot be "
                f"estimated: the components describe this {repeat} only, and the {samples} is "
                "the independent unit."
            )
        elif bool(rows["repeat_dropped"].iloc[0]):
            lines.append(
                f"  Every {repeat} holds a single {samples} here, so "
                f"the repeat is not a variance level in this stratum (its variance is counted "
                f"with the {samples}'s) and the {samples} is the independent unit."
            )
        if len(shares):
            top = shares.iloc[0]
            lines.append(
                f"  Most of the uncertainty of the mean is {words.get(top['level'], top['level'])} "
                f"({100 * top['share']:.0f}% of its variance with the acquired design"
                f"{share_caveat(top, units, unit)})."
            )
        compared = result.comparisons[result.comparisons["stratum"] == stratum]
        resolved = compared[compared["resolved"]]
        if len(compared) and len(resolved):
            best = resolved.sort_values("ratio").iloc[0]
            change = 1 - best["ratio"]
            direction = "smaller" if change > 0 else "larger"
            lines.append(
                f"  At the same total number of fields, {best['samples']} {samples}s of about "
                f"{best['fovs_per_sample']:.3g} fields would give a {100 * abs(change):.0f}% "
                f"{direction} standard error than the acquired {best['current_samples']} "
                f"{samples}s of about {best['current_fovs_per_sample']:.3g}."
            )
        else:
            if units < FEW_UNITS:
                reason = f"only {units} {unit}s"
            elif not len(compared):
                reason = "no other allocation was compared"
            elif not compared["same_total"].any():
                reason = "no allocation compared has the acquired total of fields"
            else:
                reason = "the intervals of the alternatives at the same total include no change"

            lines.append(
                f"  More fields or more {samples}s: cannot tell from these data ({reason})."
            )
        missing = int(rows["missing_values"].iloc[0])
        if missing:
            lines.append(f"  {missing} {observations} without a value were left out.")
        empty = int(rows["fovs"].iloc[0]) - int(rows["fovs_with_observations"].iloc[0])
        if empty:
            lines.append(
                f"  {empty} of {int(rows['fovs'].iloc[0])} acquired fields had no "
                f"{observations}; projections assume the same share would again."
            )
        blocks.append("\n".join(lines))
    for row in result.skipped.itertuples(index=False):
        blocks.append(
            f"{row.stratum}: more fields or more {samples}s, for the mean {feature}? "
            f"Not estimable: {row.reason}."
        )
    return "\n\n".join(blocks)


__all__ = [
    "OBSERVATION",
    "effort_summary",
    "level_words",
    "sample_word",
    "share_caveat",
    "unit_word",
]
