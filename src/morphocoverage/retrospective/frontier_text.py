"""The frontier score in words (``frontier_summary``), kept apart from its computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ._text import fields, percent

if TYPE_CHECKING:
    from .frontier import FrontierResult

BASELINE = "fixed fields"


def _setting(value: object) -> str:
    """A family setting as given: ``0.05``, ``12`` or a label such as ``strict``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def _number(value: float) -> str:
    """One decimal, without a negative zero."""
    value = round(float(value), 1)
    return f"{0.0 if value == 0 else value:.1f}"


def frontier_summary(result: FrontierResult) -> str:
    """ "At most 5% too-early stops: a fixed count needs ..., this rule saves ..." in sentences."""
    label = result.settings.get("label", result.settings.get("criterion"))
    units = result.settings.get("units", 0)
    blocks = []
    for alpha, rows in result.scores.groupby("alpha", sort=True):
        lines = [f"For {label}, stopping too early in at most {percent(alpha)} of acquisitions:"]
        base = rows[rows["family"] == BASELINE].iloc[0]
        if base["effort"] is None or pd.isna(base["effort"]):
            lines.append(
                "  a fixed number of fields never got there within the fields available, "
                "so no rule can be scored against it at this level."
            )
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
                lines.append(f"  {row.family}: never got there at any setting tried.")
                continue
            saved = row.fields_saved
            verdict = (
                f"saves {_number(saved)} fields"
                if saved > 0.05
                else f"costs {_number(-saved)} more fields"
                if saved < -0.05
                else "does as well as the fixed count, no better"
            )
            # The interval reads in the verdict's direction: fields saved, or fields it costs.
            if row.saved_low is None or pd.isna(row.saved_low):
                interval = ""
            elif saved < -0.05:
                interval = (
                    f", {percent(0.95)} interval {_number(-row.saved_high)} to "
                    f"{_number(-row.saved_low)}"
                )
            else:
                of = "" if saved > 0.05 else " of the fields saved"
                interval = (
                    f", {percent(0.95)} interval{of} {_number(row.saved_low)} to "
                    f"{_number(row.saved_high)}"
                )
            reached = getattr(row, "saved_reached", None)
            if interval and reached is not None and not pd.isna(reached) and reached < 1:
                interval += (
                    f" (from the {percent(reached)} of the resamples in which both reached "
                    "this risk)"
                )
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
        resampled = (
            f"Intervals resample the {units} units{named}, treated as independent"
            if units >= 2
            else f"No interval: there is a single unit{named} to resample"
        )
        lines.append(
            f"  {resampled}; hold-outs and orders are not independent experiments. Risk is "
            "judged on held-back fields."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["BASELINE", "frontier_summary"]
