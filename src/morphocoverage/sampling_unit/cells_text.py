"""The words of the cell-level test (Plot&Stats), part of question 9 (``contrast_cells``).

One sentence on the balance check (whether one repeat weighs more than the
others, and whether the control comes from repeats a condition was not
imaged in), one table with a row per condition: Cohen's d on the
observations and its randomization p as provided, the same balanced over the
repeats when the data are unbalanced, and the chance the test gives
p <= 0.05 without an effect at the variation between repeats seen here. A
repeat too small to balance to is named instead of balanced. The full form
adds the observations per repeat each balanced draw takes. ``cells_headline``
gives the "At a glance" words.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption
from .._style import table, wrap
from .._wording import column_word, count, listing, percent
from ..schema import unit_plural
from .contrast_cells import ALPHA, BALANCED_DRAWS, IMBALANCE, SPARSE, TYPICAL_BETWEEN


def _cells(contrast: Any) -> pd.DataFrame:
    return getattr(contrast, "cells", pd.DataFrame())


def _test(d: float, p: float, corrected: float) -> str:
    if pd.isna(d):
        return "–"
    return f"{d:.2f}" if pd.isna(p) else f"{d:.2f} ({corrected:.2g})"


def _chance(row: dict[str, Any]) -> str:
    low, high = row["chance_low"], row["chance_high"]
    if pd.isna(low):
        return "–"
    if np.isclose(low, high):
        return f"about {percent(low)}"
    return f"{percent(low)}–{percent(high)}"


def _balance(contrast: Any, cells: pd.DataFrame) -> str:
    """The balance check in a sentence."""
    word = column_word(contrast.settings["repeat_col"])
    words = unit_plural(contrast.settings["observation_unit"])
    unbalanced = cells[cells["unbalanced"]]
    if unbalanced.empty:
        return (
            f"Balance: every {word} holds {words} within {percent(IMBALANCE - 1)} of the others, "
            "and each condition is compared with the control of its own repeats, so the test "
            "is run on the data as provided."
        )
    reasons = []
    uneven = unbalanced[unbalanced["spread"] > IMBALANCE]
    if len(uneven):
        reasons.append(
            f"one {word} holds more than {percent(IMBALANCE - 1)} more {words} than another "
            f"(up to {uneven['spread'].max():.1f} times)"
        )
    others = unbalanced[unbalanced["other_control_repeats"] > 0]
    if len(others):
        reasons.append(
            f"the control is pooled from {count(int(others['other_control_repeats'].max()), word)} "
            "the condition was not imaged in"
        )
    who = (
        "every condition"
        if len(unbalanced) == len(cells)
        else f"{listing(list(unbalanced['condition']), 3)}"
    )
    return (
        f"Balance: for {who}, {' and '.join(reasons)}, so the test is given as provided and "
        f"balanced (each condition against the control of its own {word}s, the same number of "
        f"{words} from each {word}; the median d and p over {BALANCED_DRAWS} draws)."
    )


def _sparse(contrast: Any, cells: pd.DataFrame) -> list[str]:
    """One sentence per condition with a repeat too small to balance to."""
    word = column_word(contrast.settings["repeat_col"])
    words = unit_plural(contrast.settings["observation_unit"])
    lines = []
    for row in cells[cells["sparse_repeats"].map(len) > 0].to_dict("records"):
        names = listing([str(r) for r in row["sparse_repeats"]], 3)
        lines.append(
            f"{row['condition']}: {word} {names} holds very few {words} (under "
            f"{percent(SPARSE)} of the largest {word}); balancing down to it would leave little "
            "of the others, so it is not balanced: worth checking that repeat first."
        )
    return lines


def cells_parts(contrast: Any, indent: int = 2, full: bool = False) -> tuple[list[str], list[str]]:
    """The balance check, the table of the cell-level test and its note (empty without it)."""
    cells = _cells(contrast)
    if cells.empty or contrast.settings.get("repeat_col") is None:
        return [], []
    words = unit_plural(contrast.settings["observation_unit"])
    balanced = bool(cells["p_balanced"].notna().any())
    lines = [
        wrap(
            f"Cell-level test (as in Plot&Stats): Cohen's d on every "
            f"{contrast.settings['observation_unit']} against the "
            f"control, the randomization test shuffling the labels over the {words}:",
            indent,
            hang=indent + 2,
        ),
        # One repeat compared with its own control has no balance to check.
        *(
            [wrap(_balance(contrast, cells), indent + 2, hang=indent + 4)]
            if cells["repeats"].max() > 1 or cells["unbalanced"].any()
            else []
        ),
        *(wrap(line, indent + 2, hang=indent + 4) for line in _sparse(contrast, cells)),
    ]
    rows = []
    for row in cells.to_dict("records"):
        values = [
            f"{row['condition']} vs {row['control_condition']}",
            _test(row["d_cells"], row["p_cells"], row["p_cells_bonferroni"]),
        ]
        if balanced:
            values.append(_test(row["d_balanced"], row["p_balanced"], row["p_balanced_bonferroni"]))
            if full:
                values.append(
                    "–"
                    if pd.isna(row["p_balanced"])
                    else f"{int(row['balanced_cells'])} / {int(row['balanced_control_cells'])}"
                )
        values.append(_chance(row))
        rows.append(values)
    columns = ["condition", "d as provided (p Bonf.)"]
    if balanced:
        columns.append("d balanced (p Bonf.)")
        if full:
            columns.append(f"{words} per repeat a side")
    columns.append(f"p ≤ {ALPHA:g} without an effect")
    lines.append(table(rows, columns, indent=indent + 2))
    single = bool(cells["between_sd"].isna().any())
    note = assumption(
        f"The cell-level test treats the {words} as independent draws: it asks whether these "
        f"wells differ, and it responds with one or two repeats. When the difference varies "
        f"between repeats, it gives p ≤ {ALPHA:g} without an effect in about "
        f"2·P(Z > 1.96/√(1 + n·s²/2)) of experiments (n {words} per well, s the SD between "
        "repeats in observation SDs, as simulated in data/simulations/test_choice.py)"
        + (
            f"; with one repeat, s is taken as {TYPICAL_BETWEEN[0]:g}–{TYPICAL_BETWEEN[1]:g}"
            if single
            else ""
        )
        + ". Whether the effect repeats is what the paired test across repeats asks."
    )
    return lines, [note]


def cells_headline(contrast: Any) -> str | None:
    """``cell-level test: 11 of 12 at Bonferroni p ≤ 0.05 (balanced: 11 of 12)``, or ``None``."""
    cells = _cells(contrast)
    if cells.empty or contrast.settings.get("repeat_col") is None:
        return None
    tested = cells[cells["p_cells"].notna()]
    text = (
        f"cell-level test: {int((tested['p_cells_bonferroni'] <= ALPHA).sum())} of "
        f"{len(tested)} at Bonferroni p ≤ {ALPHA:g}"
    )
    balanced = cells[cells["p_balanced"].notna()]
    if len(balanced):
        text += (
            f" (balanced over the repeats: "
            f"{int((balanced['p_balanced_bonferroni'] <= ALPHA).sum())} of {len(balanced)})"
        )
    return text


__all__ = ["cells_headline", "cells_parts"]
