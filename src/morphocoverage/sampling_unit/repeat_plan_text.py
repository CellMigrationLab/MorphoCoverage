"""The words of "Planning the next experiment" (``repeat_plan``), part of question 9.

One sentence says what is planned (a paired test across biological repeats,
its power and threshold); one table gives, per condition, the per-repeat d,
the variation between repeats and the repeats needed with the fields acquired
and with twice as many; the notes say what the numbers rest on. With one
repeat, one sentence per the conditions: the variation between repeats needs
a second repeat, and question 2 gives the fields within a repeat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption
from .._style import table, wrap
from .._wording import count, listing, percent
from ..schema import unit_plural
from .repeat_plan import ALPHA, MAX_REPEATS, POWER


def _plan(contrast: Any) -> pd.DataFrame:
    return getattr(contrast, "plan", pd.DataFrame())


def _repeats_cell(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"more than {MAX_REPEATS}" if np.isinf(value) else str(int(value))


def plan_parts(contrast: Any, indent: int = 2) -> tuple[list[str], list[str]]:
    """The plan in a sentence and a table, and its notes (none of them a caution)."""
    plan = _plan(contrast)
    if plan.empty:
        return [], []
    words = unit_plural(contrast.settings["observation_unit"])
    planned = plan[plan["status"] == "ok"]
    single = plan[plan["status"] != "ok"]
    lines: list[str] = []
    notes: list[str] = []
    if len(planned):
        corrected = ALPHA / len(planned)
        lines.append(
            wrap(
                f"Planning the next experiment: the biological repeats a paired test across "
                f"repeats needs for {percent(POWER)} power at p ≤ {ALPHA:g}"
                + (
                    f" (Bonferroni over {count(len(planned), 'condition')}: {corrected:.2g})"
                    if len(planned) > 1
                    else ""
                )
                + ", with the fields a side acquired and with twice as many:",
                indent,
                hang=indent + 2,
            )
        )
        rows = [
            [
                f"{row['condition']} vs {row['control_condition']}",
                f"{row['d_mean']:.2f}",
                f"{row['between_sd']:.2f}",
                _repeats_cell(row["repeats_needed"]),
                _repeats_cell(row["repeats_needed_twice_fields"]),
            ]
            for row in planned.to_dict("records")
        ]
        columns = [
            "condition",
            f"d {words}, mean",
            "SD between repeats",
            "repeats needed",
            "with twice the fields",
        ]
        lines.append(table(rows, columns, indent=indent + 2))
        n = int(planned["repeats"].min())
        notes.append(
            assumption(
                f"The plan rests on {count(n, 'biological repeat')}: the variation between "
                "repeats (the per-repeat d's spread beyond their sampling error, which more "
                "fields cannot reduce) is estimated from them, so the repeats needed are a guide; "
                "the fields a side scale the sampling error within a repeat as 1/sqrt(fields)."
            )
        )
    if len(single):
        lines.append(
            wrap(
                f"Planning the next experiment: {listing(list(single['condition']), 3)} "
                f"{'was' if len(single) == 1 else 'were'} imaged in one biological repeat, so "
                "the variation between repeats is not yet known; a second repeat gives it, and "
                "the number of repeats to plan for. Within a repeat, question 2 gives the fields "
                "that would resolve each difference.",
                indent,
                hang=indent + 2,
            )
        )
    return lines, notes


__all__ = ["plan_parts"]
