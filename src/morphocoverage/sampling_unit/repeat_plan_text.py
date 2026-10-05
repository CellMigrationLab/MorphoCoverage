"""The words of "Planning the next experiment" (``repeat_plan``), part of question 9.

At the design the user ran: per condition, the repeats a paired test needs at the
report's correction, with the fields per repeat as now and twice as many, then the fields a
side each repeat needs to resolve the difference on its own. When every field holds one
observation, the advice says cells (the unit), not fields. With one repeat, one sentence
says the variation between repeats needs a second repeat. None of it is a caution: it is
advice. It names no limiting factor (owner, 2026-10-02): whether the repeats differ more
than their sampling error is not something 3 repeats can tell (docs/questions.md, P10).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption
from .._control import versus
from .._style import table, wrap
from .._wording import change, count, listing, percent
from ..schema import observation_words, unit_plural
from .contrast_sampling import MORE_WELLS, PRACTICAL
from .repeat_plan import ALPHA, MAX_REPEATS, POWER


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


def _fields_cell(value: float, fields: str = "fields", wells: bool = False) -> str:
    if pd.isna(value):
        return "–"
    if np.isinf(value) and wells:
        # More fields will not resolve it (owner, 2026-10-05).
        return MORE_WELLS
    return f"more than {PRACTICAL} times the {fields}" if np.isinf(value) else f"about {int(value)}"


def _between_wells(contrast: Any, row: dict[str, Any]) -> bool:
    """Whether some repeat of the plan's condition is one more fields will not resolve, the
    wells differing more than the difference (``contrast_sampling``)."""
    table = getattr(contrast, "table", pd.DataFrame())
    if "between_wells" not in table:
        return False
    mine = (table["condition"] == row["condition"]) & (
        table["control_condition"] == row["control_condition"]
    )
    return bool(table.loc[mine, "between_wells"].fillna(False).astype(bool).any())


def change_cell(row: Any) -> str:
    """The change the plan powers, as the paired t-test sees it (``_wording.change``): a fold
    change of the means, else the difference of the means; ``–`` for a plan on Cohen's d."""
    if row.get("planned_on") not in ("log ratio", "difference"):
        return "–"
    return change(
        row.get("change", np.nan), "normalised" if row["planned_on"] == "log ratio" else "raw"
    )


def plan_parts(contrast: Any, indent: int = 2, full: bool = False) -> tuple[list[str], list[str]]:
    """The plan in a sentence and a table, and its notes (none of them a caution)."""
    plan = _plan(contrast)
    if plan.empty:
        return [], []
    words = observation_words(contrast.settings)[1]
    _, fields = field_words(contrast)
    feature = contrast.settings.get("feature", "the readout")
    held = sorted({int(n) for n in plan["repeats"]})
    # Each condition's own repeats: "each of 3 repeats", "the one repeat", or "each repeat".
    each = (
        "the one repeat"
        if held == [1]
        else f"each of {count(held[0], 'repeat')}"
        if len(held) == 1
        else "each repeat"
    )
    lines = [
        wrap(
            f"Planning the next experiment on {feature} (for a new experiment, if the "
            "difference and the variation seen here recur; not what the current result "
            "needs): the biological repeats a paired t-test on the repeat means needs for "
            f"{percent(POWER)} power, corrected over the conditions of its sub-experiment (a "
            f"follow-up on {feature} alone), with the "
            f"{fields} per repeat as now and with twice as many; then the {fields} a side "
            f"{each} needs to show the difference on its own:",
            indent,
            hang=indent + 2,
        )
    ]
    rows = []
    for row in plan.to_dict("records"):
        rows.append(
            [
                versus(row["condition"], row["control_condition"]),
                change_cell(row),
                f"{row['d_mean']:+.2f}",
                _repeats_cell(row["repeats_needed"]),
                _repeats_cell(row["repeats_needed_twice_fields"]),
                _fields_cell(row["fields_per_repeat"], fields, _between_wells(contrast, row)),
            ]
        )
    columns = [
        "condition",
        "change across repeats",
        f"mean d per repeat ({words})",
        "repeats needed",
        f"with twice the {fields}",
        f"{fields} a side per repeat",
    ]
    lines.append(table(rows, columns, indent=indent + 2))
    single = plan[plan["status"].str.startswith("one biological repeat")]
    # Imaged in several repeats, but fewer than two with an interval of d (``repeat_plan``).
    unestimated = plan[(plan["status"] != "ok") & ~plan.index.isin(single.index)]
    if len(unestimated):
        lines.append(
            wrap(
                f"{listing(list(unestimated['condition']), 3)}: fewer than two of the repeats "
                f"have two {fields} a side or more, so the variation between repeats is not "
                "estimated.",
                indent,
                hang=indent + 2,
            )
        )
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
            "sub-experiment (what Benjamini-Hochberg asks at most), on the values that test "
            "compares: each repeat's change (its mean over its control's, a fold change; the "
            "difference of the means where a mean is not positive; the table gives their "
            "geometric mean, as the verdict does), its spread "
            "between the repeats and its sampling error within one. All projected from the "
            "repeats here, so the plan is a guide."
        )
    ]
    if full:
        notes.append(
            assumption(
                f"The {fields} a side per repeat are the most any repeat needs, projected from "
                "the part of its interval of d within wells (narrowing as 1/sqrt of the number); "
                f"the part between wells stays, and when it alone is as wide as the difference, "
                f"more {fields} will not resolve it: {MORE_WELLS}."
            )
        )
        notes.append(
            assumption(
                f"Most repeats searched: {MAX_REPEATS}; per condition, its own experiment's m."
            )
        )
    return lines, notes


__all__ = ["change_cell", "field_words", "plan_parts"]
