"""A difference from the control in "Plan the next experiment" (private; ``_html_plan``): the
repeats each condition not yet significant across repeats would need, per feature, as rows
and as the page's table.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd

from .._identity import IdentityKey, identity_key
from .._wording import change
from ..sampling_unit.repeat_plan import MAX_REPEATS
from ..sampling_unit.repeat_plan_text import field_words

# A plan with no estimable answer, in the table's "repeats needed" (#90).
NOT_ASSESSABLE = "not assessable"
# A row of the table: feature, condition, change seen, repeats needed, with twice the fields,
# and the fields' word.
PlanRow = tuple[str, str, str, str, str, str]


def _plan_words(value: Any) -> str:
    if pd.isna(value):
        return "–"
    return f"more than {MAX_REPEATS}" if not np.isfinite(float(value)) else str(int(value))


def _repeat_conditions(discovery: Any) -> set[IdentityKey]:
    tests = discovery.tests["condition"] if discovery is not None else []
    return {identity_key(c) for c in tests}


def _plan_rows(
    plans: list[tuple[str, Any]],
    pilots: set[IdentityKey],
    done: set[tuple[IdentityKey, IdentityKey]],
) -> tuple[list[PlanRow], list[str]]:
    """One row per feature and condition with a plan, the pilots and the pairs ``done``
    (significant across repeats) left out, and the features of those done, in order. A plan
    with no estimable answer stays, said not assessable with its reason (#90)."""
    rows: list[PlanRow] = []
    shown: list[str] = []
    for feature, contrast in plans:
        plan = getattr(contrast, "plan", None)
        if plan is None or plan.empty:
            continue
        unit = field_words(contrast)[1]
        for _, row in plan.iterrows():
            if identity_key(row["condition"]) in pilots:
                continue
            if (identity_key(feature), identity_key(row["condition"])) in done:
                shown.append(feature)
                continue
            unplanned = pd.isna(row["repeats_needed"])
            rows.append(
                (
                    feature,
                    str(row["condition"]),
                    _moved(row),
                    _unplanned(row) if unplanned else _plan_words(row["repeats_needed"]),
                    "–" if unplanned else _plan_words(row["repeats_needed_twice_fields"]),
                    unit,
                )
            )
    return rows, shown


def _unplanned(row: pd.Series) -> str:
    """``not assessable: <the plan's status>``: why a plan gives no repeats needed (a zero or
    missing variance, ``repeat_plan.DEGENERATE``, ``MISSING_VALUE``), never left out."""
    status = row.get("status")
    reason = status if isinstance(status, str) and status and status != "ok" else ""
    return f"{NOT_ASSESSABLE}: {reason}" if reason else NOT_ASSESSABLE


def _moved(row: pd.Series) -> str:
    """The change seen that the plan is made on, normalised for a log ratio; "–" without."""
    planned = str(row["planned_on"])
    return (
        change(float(row["change"]), "normalised" if planned == "log ratio" else "raw")
        if planned in ("log ratio", "difference") and pd.notna(row["change"])
        else "–"
    )


def _plan_table(rows: list[PlanRow]) -> str:
    """The rows as the page's sortable table."""
    unit = rows[0][5]
    head = (
        "<tr><th>feature</th><th>condition</th><th>change seen</th><th>repeats needed</th>"
        f"<th>with twice the {html.escape(unit)} per repeat</th></tr>"
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row[:5]) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="sortable"><thead>{head}</thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
