"""The representation control in words (``control_summary``), split from ``controls``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from ._text import percent, unit_plural

if TYPE_CHECKING:
    from .controls import ControlResult


def _unjudged(row: Any) -> str:
    """``; 6 could not be judged (...)`` when some candidate fields could not be judged."""
    count = int(row["not_judged"])
    if not count:
        return ""
    return f"; {count} could not be judged (missing feature values or too few fields)"


def _rate(value: float, checks: int) -> str:
    """``100% of 120 checks``, or ``no check could be made`` when nothing was judged."""
    if pd.isna(value) or not checks:
        return "no check could be made"
    return f"{percent(value)} of {checks} checks"


def _fields_rate(row: Any) -> str:
    """``3% of 60 fields``, or ``no field could be judged``."""
    total = int(row["candidates"])
    if pd.isna(row["rate"]) or not total:
        return "no field could be judged"
    return f"{percent(row['rate'])} of {total} fields"


def _leave_one_out_line(rates: pd.DataFrame, groups: Sequence[str] | None) -> str:
    """The rates judged within each analysis group, as inspect_fovs judges, or an empty string."""
    parts = []
    if "reference_lofo" in rates.index:
        parts.append(
            f"reference fields were called different in {_fields_rate(rates.loc['reference_lofo'])}"
            " (the false-alarm rate of inspect_fovs)"
        )
    if "positive_lofo" in rates.index:
        positive = rates.loc["positive_lofo"]
        parts.append(
            "positive-control fields, each judged against the reference fields of each group, "
            f"in {_rate(positive['rate'], int(positive['candidates']))}"
        )
    if not parts:
        return ""
    within = f"each analysis group ({', '.join(groups)})" if groups else "each stratum"
    return (
        f"  Against the other reference fields of {within}, as inspect_fovs judges a field: "
        f"{'; '.join(parts)}."
    )


def control_summary(result: ControlResult) -> str:
    """ "Can these features see the declared phenotype?" in sentences."""
    skipped = result.settings.get("skipped_strata", [])
    cells = unit_plural(result.settings.get("observation_unit", "object"))
    if result.status == "not assessed":
        declared = result.settings.get("reason") != "none"
        why = (
            "a control was declared, but no stratum could be judged ("
            + "; ".join(f"{s['stratum']}: {s['reason']}" for s in skipped)
            + ")"
            if declared and skipped
            else "a control was declared, but no check could be made (missing values?)"
            if declared
            else "no control was declared"
        )
        return (
            f"Representation sensitivity: not assessed: {why}. A 'nothing new' answer says "
            "nothing about phenotypes these features cannot see."
        )
    rates = result.pooled.set_index("candidate")
    seen = result.settings["fovs_seen"]
    heading = f"(after {seen} reference fields with {cells})"
    lines = [f"Can these features see the declared phenotype? {heading}"]
    positive = rates.loc["positive"]
    lines.append(
        f"  Fields of the declared positive control were recognized as different in "
        f"{_rate(positive['rate'], int(positive['candidates']))}{_unjudged(positive)}."
    )
    for gap in result.settings.get("planted_skipped", []):
        lines.append(
            f"  Planted {cells}: not measured in {gap['stratum']}: "
            f"{gap['reason'].replace('objects', cells)}."
        )
    if "planted" in rates.index:
        planted = rates.loc["planted"]
        planted_n = result.settings["control"]["planted_objects"]
        lines.append(
            f"  Ordinary reference fields with {planted_n} positive-control {cells} added were "
            f"recognized in {_rate(planted['rate'], int(planted['candidates']))}"
            f"{_unjudged(planted)}."
        )
    if "reference" in rates.index and rates.loc["reference", "candidates"] > 0:
        reference = rates.loc["reference"]
        lines.append(
            f"  Reference fields were called different in "
            f"{_rate(reference['rate'], int(reference['candidates']))} (false alarms)"
            f"{_unjudged(reference)}."
        )
    elif result.status == "positive only" and "reference" in rates.index:
        lines.append("  False-alarm rate: not assessed (no reference field could be judged).")
    else:
        lines.append("  False-alarm rate: not assessed (no reference control was declared).")
    lofo = _leave_one_out_line(rates, result.settings.get("lofo_group_cols"))
    if lofo:
        lines.append(lofo)
    if len(result.table["stratum"].unique()) > 1:
        by = result.table[result.table["candidate"] == "positive"]
        lines.append(
            "  Positive control per "
            + "/".join(result.pair_cols)
            + ": "
            + ", ".join(
                f"{r.stratum}: {'not judged' if pd.isna(r.rate) else percent(r.rate)}"
                for r in by.itertuples()
            )
            + "."
        )
    if skipped:
        lines.append(
            f"  Not judged after {seen} reference fields: "
            + "; ".join(f"{s['stratum']} ({s['reason']})" for s in skipped)
            + "."
        )
    lines.append(
        "  This holds for this definition of new and these features, on these controls; "
        "a phenotype unlike the control may be seen better or worse."
    )
    return "\n".join(lines)


__all__ = ["control_summary"]
