"""The words of "Planning the next experiment" (``repeat_plan``), part of question 9.

At the design the user ran: per condition, the fields a side each repeat
needs to resolve the difference and what limits it (the sampling within a
repeat, which more fields reduce, or the variation between repeats, which
they do not). The full form adds the repeats a paired test would need, per
condition. With one repeat, one sentence says the variation between repeats
needs a second repeat. None of it is a caution: it is advice.
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

_LIMITS = {
    "within repeats": "sampling (more fields help)",
    "between repeats": "repeats differ (more fields help little)",
    "": "unknown with one repeat",
}


def _plan(contrast: Any) -> pd.DataFrame:
    return getattr(contrast, "plan", pd.DataFrame())


def _repeats_cell(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"more than {MAX_REPEATS}" if np.isinf(value) else str(int(value))


def _fields_cell(value: float) -> str:
    if pd.isna(value):
        return "–"
    return "too small to settle" if np.isinf(value) else f"about {int(value)}"


def plan_parts(contrast: Any, indent: int = 2, full: bool = False) -> tuple[list[str], list[str]]:
    """The plan in a sentence and a table, and its notes (none of them a caution)."""
    plan = _plan(contrast)
    if plan.empty:
        return [], []
    words = unit_plural(contrast.settings["observation_unit"])
    n = int(plan["repeats"].max())
    lines = [
        wrap(
            f"Planning the next experiment, with {count(n, 'biological repeat')} as here: the "
            "fields a side each repeat needs to resolve the difference from the control, and what "
            "limits it"
            + (
                f" (full report: the repeats a paired test needs for {percent(POWER)} power at "
                f"p ≤ {ALPHA:g}, per condition)"
                if full
                else ""
            )
            + ":",
            indent,
            hang=indent + 2,
        )
    ]
    rows = []
    for row in plan.to_dict("records"):
        cells = [
            f"{row['condition']} vs {row['control_condition']}",
            f"{row['d_mean']:.2f}",
            _fields_cell(row["fields_per_repeat"]),
            _LIMITS.get(row["limited_by"], row["limited_by"]),
        ]
        if full:
            cells += [
                _repeats_cell(row["repeats_needed"]),
                _repeats_cell(row["repeats_needed_twice_fields"]),
            ]
        rows.append(cells)
    columns = ["condition", f"d {words}, mean", "fields a side per repeat", "limited by"]
    if full:
        columns += ["repeats needed", "with twice the fields"]
    lines.append(table(rows, columns, indent=indent + 2))
    single = plan[plan["status"] != "ok"]
    if len(single):
        lines.append(
            wrap(
                f"{listing(list(single['condition']), 3)} "
                f"{'was' if len(single) == 1 else 'were'} imaged in one biological repeat: a "
                "second repeat shows how much the difference varies between repeats.",
                indent,
                hang=indent + 2,
            )
        )
    notes = [
        assumption(
            "The fields a side per repeat are the most any repeat needs, projected from its "
            "interval of d (narrowing as 1/sqrt(fields)); limited by: whether the repeats' d "
            "differ more than their sampling error, which more fields cannot reduce. Both are "
            "estimated from the repeats here, so the plan is a guide."
        )
    ]
    return lines, notes


__all__ = ["plan_parts"]
