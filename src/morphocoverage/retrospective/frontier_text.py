"""The frontier score in words (``frontier_summary``), kept apart from its computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ._text import fields, percent
from .effort import MIN_BOOT_SUCCESS
from .protocol_text import FEW_UNITS

if TYPE_CHECKING:
    from .frontier import FrontierResult

BASELINE = "fixed fields"
# Units from which the widened interval of the fields saved came close to its level in
# simulation (0.64 at 5 units, 0.80 at 12, 0.95 at 20 for a nominal 95%).
_MANY_UNITS = 20


def _setting(value: object) -> str:
    """A family setting as given: ``0.05``, ``12`` or a label such as ``strict``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def _number(value: float) -> str:
    """One decimal, without a negative zero."""
    value = round(float(value), 1)
    return f"{0.0 if value == 0 else value:.1f}"


def _lapse(row: object) -> str | None:
    """``met this risk at 2 (risk 0%) but not from there on: ...``, or None if it never met it."""
    met = getattr(row, "met_setting", None)
    if met is None or pd.isna(met):
        return None
    return (
        f"met this risk at {_setting(met)} (risk {percent(row.met_risk)}) but "
        f"not from there on: at {_setting(row.exceeded_setting)}, with more fields, the risk "
        f"was {percent(row.exceeded_risk)}, so no setting is taken as safe"
    )


def _not_chosen(row: object) -> str:
    """A family with no safe setting: met the risk once but not from there on, or never."""
    lapse = _lapse(row)
    if lapse is None:
        return f"  {row.family}: never got there at any setting tried."
    return f"  {row.family}: {lapse}."


def _unscored(rows: pd.DataFrame, base: object) -> list[str]:
    """A risk level the fixed count never met lastingly: why, then each family on its own."""
    lapse = _lapse(base)
    if lapse is not None:
        first = f"  a fixed number of fields {lapse}; no rule is scored against it at this level."
    else:
        first = (
            "  a fixed number of fields never got there within the fields available, "
            "so no rule can be scored against it at this level."
        )
    lines = [first]
    for row in rows[rows["family"] != BASELINE].itertuples(index=False):
        if row.effort is None or pd.isna(row.effort):
            lines.append(_not_chosen(row))
        else:
            lines.append(
                f"  {row.family} at {_setting(row.setting)}: {fields(row.effort)} on average "
                f"(risk {percent(row.risk)}), with no fixed count to compare."
            )
    return lines


def _interval(row: object, saved: float, units: int | None = None) -> str:
    """The interval of the fields saved in the verdict's direction, with its notes, or ''.

    No interval when fewer than ``MIN_BOOT_SUCCESS`` of the resamples reached
    the risk for both (the others are a selected subset), nor with fewer than
    ``FEW_UNITS`` independent units, below which no resampling of the units
    gives an interval near its level. The interval is widened for the number
    of units and still approximate, and says so.
    """
    if units is not None and units < FEW_UNITS:
        return ""
    reached = getattr(row, "saved_reached", None)
    reached = None if reached is None or pd.isna(reached) else float(reached)
    if reached is not None and reached < MIN_BOOT_SUCCESS:
        return (
            f", no interval: only {percent(reached)} of the resamples reached this risk for "
            f"both (at least {percent(MIN_BOOT_SUCCESS)} are needed)"
        )
    if row.saved_low is None or pd.isna(row.saved_low):
        return ""
    # The interval reads in the verdict's direction: fields saved, or fields it costs.
    if saved < -0.05:
        interval = (
            f", approximate {percent(0.95)} interval {_number(-row.saved_high)} to "
            f"{_number(-row.saved_low)}"
        )
    else:
        of = "" if saved > 0.05 else " of the fields saved"
        interval = (
            f", approximate {percent(0.95)} interval{of} {_number(row.saved_low)} to "
            f"{_number(row.saved_high)}"
        )
    notes = []
    # A saving (or cost) whose interval includes no change is not resolved.
    if abs(saved) > 0.05 and row.saved_low <= 0 <= row.saved_high:
        notes.append("not resolved")
    if reached is not None and reached < 1:
        notes.append(
            f"from the {percent(reached)} of the resamples in which both reached this risk"
        )
    return interval + (f" ({'; '.join(notes)})" if notes else "")


def frontier_summary(result: FrontierResult) -> str:
    """ "At most 5% too-early stops: a fixed count needs ..., this rule saves ..." in sentences."""
    label = result.settings.get("label", result.settings.get("criterion"))
    units = result.settings.get("units", 0)
    blocks = []
    for alpha, rows in result.scores.groupby("alpha", sort=True):
        lines = [f"For {label}, stopping too early in at most {percent(alpha)} of acquisitions:"]
        base = rows[rows["family"] == BASELINE].iloc[0]
        if base["effort"] is None or pd.isna(base["effort"]):
            # The families are still listed, with their own effort and risk.
            baseline = next(rows[rows["family"] == BASELINE].itertuples(index=False))
            lines += _unscored(rows, baseline)
            blocks.append("\n".join(lines))
            continue
        # Fields saved subtract mean efforts; when streams shorter than the fixed count
        # stop early, the baseline's mean effort is below its setting, and the text says so.
        mean_effort = (
            ""
            if abs(float(base["effort"]) - float(base["setting"])) < 0.05
            else f", {_number(base['effort'])} on average as some acquisitions had fewer"
        )
        lines.append(
            f"  a fixed number of fields needed {fields(int(base['setting']))}{mean_effort} "
            f"(risk {percent(base['risk'])})."
        )
        for row in rows[rows["family"] != BASELINE].itertuples(index=False):
            if row.effort is None or pd.isna(row.effort):
                lines.append(_not_chosen(row))
                continue
            saved = row.fields_saved
            verdict = (
                f"saves {_number(saved)} fields"
                if saved > 0.05
                else f"costs {_number(-saved)} more fields"
                if saved < -0.05
                else "does as well as the fixed count, no better"
            )
            interval = _interval(row, saved, units)
            adaptive = (
                ""
                if row.adaptivity is None or pd.isna(row.adaptivity)
                else f"; its stops follow the fields each acquisition needed with Spearman "
                f"{row.adaptivity:.2f}"
            )
            lines.append(
                f"  {row.family} at {_setting(row.setting)}: {fields(row.effort)} on average "
                f"(risk {percent(row.risk)}); it {verdict}{interval}{adaptive}."
            )
        unit_cols = result.settings.get("unit_cols") or []
        named = f" ({', '.join(unit_cols)})" if unit_cols else ""
        if units >= FEW_UNITS:
            resampled = (
                f"Intervals resample the {units} units{named}, treated as independent, "
                "widened for their number (approximate"
                + (
                    ""
                    if units >= _MANY_UNITS
                    else f": with fewer than {_MANY_UNITS} units "
                    "they cover the truth less often than stated"
                )
                + ")"
            )
        elif units >= 2:
            resampled = (
                f"No interval: only {units} units{named} to resample, and at least "
                f"{FEW_UNITS} are needed for one near its level"
            )
        else:
            resampled = f"No interval: there is a single unit{named} to resample"

        lines.append(
            f"  {resampled}; hold-outs and orders are not independent experiments. Risk is "
            "judged on held-back fields."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["BASELINE", "frontier_summary"]
