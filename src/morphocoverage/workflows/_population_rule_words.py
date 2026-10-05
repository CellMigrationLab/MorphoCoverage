"""A population group's feature rule in words (private; ``_populations``, ``_html_next``,
``_html_plan``): the rule shown is fitted on every cell, and its numbers describe the
procedure refitted with whole fields held out (``sampling_unit._population_rule``; owner,
2026-10-05).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._wording import percent


def rule_cell(row: Any) -> str:
    """``AR > 3.18 and MinFeret ≤ 17.7 (91%, 97%)`` for the table, ``(91%, 97%, cells held
    out)`` for a group in one field; ``–`` without a rule."""
    if not row.rule:
        return "–"
    if pd.isna(row.rule_recall):
        return row.rule
    cells = ", cells held out" if row.rule_held_out == "cells" else ""
    return f"{row.rule} ({percent(row.rule_recall)}, {percent(row.rule_precision)}{cells})"


def rule_words(row: Any, units: str) -> str:
    """``picked by AR > 3.18 and MinFeret ≤ 17.7, fitted on all the cells; the procedure,
    refitted with whole fields held out, finds 91% of the group's cells in them, 97% of those
    it picks being in it``, or empty without a rule (``units``: the observations' plural
    word). The numbers describe the procedure, not the thresholds shown (owner, 2026-10-05)."""
    if not row["rule"]:
        return ""
    if pd.isna(row["rule_recall"]):
        return f"picked by {row['rule']}"
    held = (
        f"refitted with {units} held out (the group is in one field), finds "
        f"{percent(row['rule_recall'])} of its {units}"
        if row.get("rule_held_out") == "cells"
        else "refitted with whole fields held out, finds "
        f"{percent(row['rule_recall'])} of the group's {units} in them"
    )
    return (
        f"picked by {row['rule']}, fitted on all the {units}; the procedure, {held}, "
        f"{percent(row['rule_precision'])} of those it picks being in it"
    )


__all__ = ["rule_cell", "rule_words"]
