"""The words of "Planning the next experiment" (``repeat_plan``), part of question 9.

At the design the user ran: per condition, the fields a side each repeat
needs to resolve the difference and what limits it (the sampling within a
repeat, which more fields reduce, or the variation between repeats, which
they do not), led by the repeats a paired test needs at the report's correction.
When every field holds one observation, the advice says cells (the unit), not
fields. With one repeat, one sentence says the variation between repeats
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
from .contrast_sampling import PRACTICAL
from .repeat_plan import ALPHA, MAX_REPEATS, POWER

_LIMITS = {
    "within repeats": "sampling within a repeat: more {fields} help",
    "between repeats": "the repeats differ: add repeats, not {fields}",
    "": "unknown with one repeat",
}


def field_words(contrast: Any) -> tuple[str, str]:
    """``(field, fields)``, or the observation unit's words when every field holds one
    observation (a field is then one cell, and advice in fields would read as images)."""
    table = getattr(contrast, "table", pd.DataFrame())
    unit = contrast.settings.get("observation_unit") or "observation"
    single = (
        len(table) > 0
        and {"fields", "observations", "control_fields", "control_observations"} <= set(table)
        and (table["fields"] == table["observations"]).all()
        and (table["control_fields"] == table["control_observations"]).all()
    )
    return (unit, unit_plural(unit)) if single else ("field", "fields")


def _plan(contrast: Any) -> pd.DataFrame:
    return getattr(contrast, "plan", pd.DataFrame())


def _repeats_cell(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"more than {MAX_REPEATS}" if np.isinf(value) else str(int(value))


def _fields_cell(value: float, fields: str = "fields") -> str:
    if pd.isna(value):
        return "–"
    return f"more than {PRACTICAL} times the {fields}" if np.isinf(value) else f"about {int(value)}"


def plan_parts(contrast: Any, indent: int = 2, full: bool = False) -> tuple[list[str], list[str]]:
    """The plan in a sentence and a table, and its notes (none of them a caution)."""
    plan = _plan(contrast)
    if plan.empty:
        return [], []
    words = unit_plural(contrast.settings["observation_unit"])
    _, fields = field_words(contrast)
    n = int(plan["repeats"].max())
    lines = [
        wrap(
            f"Planning the next experiment: the biological repeats a paired t-test on the repeat "
            f"means needs for {percent(POWER)} power, at the correction used above, with the "
            f"{fields} per repeat as now and with twice as many; then the {fields} a side each of "
            f"{count(n, 'repeat')} needs to show the difference on its own, and what limits it:",
            indent,
            hang=indent + 2,
        )
    ]
    rows = []
    for row in plan.to_dict("records"):
        small = np.isinf(row["fields_per_repeat"]) if pd.notna(row["fields_per_repeat"]) else False
        limit = _LIMITS.get(row["limited_by"], row["limited_by"]).format(fields=fields)
        if small:
            limit = f"a small difference: add repeats, not {fields}"
        rows.append(
            [
                f"{row['condition']} vs {row['control_condition']}",
                f"{row['d_mean']:+.2f}",
                _repeats_cell(row["repeats_needed"]),
                _repeats_cell(row["repeats_needed_twice_fields"]),
                _fields_cell(row["fields_per_repeat"], fields),
                limit,
            ]
        )
    columns = [
        "condition",
        f"mean d per repeat ({words})",
        "repeats needed",
        f"with twice the {fields}",
        f"{fields} a side per repeat",
        "limited by",
    ]
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
            f"Repeats needed: for a paired t-test at p ≤ {ALPHA:g}/m, m the conditions of the "
            "sub-experiment (what Benjamini-Hochberg asks at most). All projected from the "
            "repeats here, so the plan is a guide."
        )
    ]
    if full:
        notes.append(
            assumption(
                f"The {fields} a side per repeat are the most any repeat needs, projected from "
                "its interval of d (narrowing as 1/sqrt of the number); limited by: whether the "
                "repeats' d differ more than their sampling error, which more "
                f"{fields} cannot reduce."
            )
        )
        notes.append(
            assumption(
                f"Most repeats searched: {MAX_REPEATS}; per condition, its own experiment's m."
            )
        )
    return lines, notes


__all__ = ["field_words", "plan_parts"]
