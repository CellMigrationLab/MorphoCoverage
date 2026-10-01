"""The words of the cell-level test (Plot&Stats), part of question 9 (``contrast_cells``).

One sentence on the balance check (whether one repeat weighs more than the
others, and whether the control comes from repeats a condition was not
imaged in), then a table with a row per condition: Cohen's d on the
observations and its randomization p, on raw and on normalised values, and
the chance the test gives p <= 0.05 without an effect at the variation
between repeats seen here. When the data are unbalanced, a second table gives
the same balanced over the repeats; a repeat too small to balance to is named
instead. The full form adds the observations per repeat each balanced draw
takes. The notes state the assumptions and limitations of the test, of the
normalisation and of the balancing. ``cells_headline`` gives the "At a glance"
words.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption
from .._style import table, wrap
from .._wording import column_word, count, listing, percent
from ..schema import unit_plural
from .contrast_cells import (
    ALPHA,
    BALANCED_DRAWS,
    IMBALANCE,
    MIN_BALANCED,
    SPARSE,
    TYPICAL_BETWEEN,
)


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
    """One sentence per repeat too small to balance to, naming the conditions it concerns."""
    word = column_word(contrast.settings["repeat_col"])
    words = unit_plural(contrast.settings["observation_unit"])
    flagged = cells[cells["sparse_repeats"].map(len) > 0]
    groups: dict[tuple[tuple[str, ...], int], list[str]] = {}
    for row in flagged.to_dict("records"):
        key = (tuple(str(r) for r in row["sparse_repeats"]), int(row["balanced_repeats"]))
        groups.setdefault(key, []).append(str(row["condition"]))
    lines = []
    for (names, kept), conditions in groups.items():
        rest = (
            f"the balanced test balances the other {count(kept, word)}"
            if kept
            else "no other repeat is left to balance"
        )
        lines.append(
            f"{listing(conditions, 3)}: {word} {listing(list(names), 3)} holds very few {words} "
            f"(under {percent(SPARSE)} of the largest {word}, or fewer than {MIN_BALANCED}), "
            "worth checking first. Balancing down to it would leave little of the others, so it "
            f"is kept in the test as provided and left out of the balanced one: {rest}."
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
    as_provided = [
        [
            f"{row['condition']} vs {row['control_condition']}",
            _test(row["d_cells"], row["p_cells"], row["p_cells_bonferroni"]),
            _test(
                row["d_cells_normalised"],
                row["p_cells_normalised"],
                row["p_cells_normalised_bonferroni"],
            ),
            _chance(row),
        ]
        for row in cells.to_dict("records")
    ]
    columns = ["condition", "raw: d (p Bonf.)", "normalised", f"p ≤ {ALPHA:g} without an effect"]
    if balanced:
        lines.append(wrap("As provided:", indent + 2, hang=indent + 4))
    lines.append(table(as_provided, columns, indent=indent + 2))
    if balanced:
        lines.append(wrap("Balanced over the repeats:", indent + 2, hang=indent + 4))
        rows = []
        for row in cells[cells["p_balanced"].notna()].to_dict("records"):
            values = [
                f"{row['condition']} vs {row['control_condition']}",
                _test(row["d_balanced"], row["p_balanced"], row["p_balanced_bonferroni"]),
                _test(
                    row["d_balanced_normalised"],
                    row["p_balanced_normalised"],
                    row["p_balanced_normalised_bonferroni"],
                ),
            ]
            if full:
                values.append(
                    f"{int(row['balanced_cells'])} / {int(row['balanced_control_cells'])}"
                )
            rows.append(values)
        names = ["condition", "raw: d (p Bonf.)", "normalised"]
        names += [f"{words} per repeat, condition / control"] if full else []
        lines.append(table(rows, names, indent=indent + 2))
    return lines, _notes(contrast, cells, balanced)


def _notes(contrast: Any, cells: pd.DataFrame, balanced: bool) -> list[str]:
    """The assumptions and limitations of the cell-level test, normalisation and balancing."""
    words = unit_plural(contrast.settings["observation_unit"])
    single = bool(cells["between_sd"].isna().any())
    notes = [
        assumption(
            f"Cell-level test (Cohen's d, randomization over the {words}): it assumes the "
            f"{words} are independent draws, so it asks whether these wells differ, not whether "
            "the effect repeats, and it responds with one or two repeats. Limitation: when the "
            f"difference varies between repeats it gives p ≤ {ALPHA:g} without an effect in "
            f"about 2·P(Z > 1.96/√(1 + n·s²/2)) of experiments (n {words} per well, s the SD "
            "between repeats in observation SDs; data/simulations/test_choice.py), the "
            "'without an effect' column"
            + (
                f" (with one repeat s is unknown and taken as "
                f"{TYPICAL_BETWEEN[0]:g}–{TYPICAL_BETWEEN[1]:g})"
                if single
                else ""
            )
            + f"; more {words} per well raise it, and a stricter cutoff does not remove it "
            "(22% at p ≤ 0.01 with 30 per well and s = 0.5)."
        ),
        assumption(
            "Normalised: each value divided by the mean of its own repeat's control, which "
            "removes a scale both wells of a repeat share (staining, imaging) and pairs every "
            "condition with its own repeat's control. It assumes a positive feature on a ratio "
            "scale (not computed when a control mean is 0 or below); it does not remove the part "
            "of the effect that varies between repeats, so the cell-level chance still applies."
        ),
    ]
    if balanced:
        notes.append(
            assumption(
                f"Balanced: the median d and median p over {BALANCED_DRAWS} draws taking the same "
                f"number of {words} from each repeat; it describes the typical draw and is not "
                "the p of one test. It removes the weight of the repeats holding more, not the "
                "independence assumption of the cell-level test."
            )
        )
    return notes


def _passing(cells: pd.DataFrame, column: str) -> str:
    tested = cells[cells[column].notna()]
    return f"{int((tested[f'{column}_bonferroni'] <= ALPHA).sum())}"


def cells_headline(contrast: Any) -> str | None:
    """``cell-level test, raw / normalised: 9 / 10 of 12 at Bonferroni p ≤ 0.05 (balanced
    over the repeats: 10 / 11)``, or ``None``."""
    cells = _cells(contrast)
    if cells.empty or contrast.settings.get("repeat_col") is None:
        return None
    text = (
        f"cell-level test, raw / normalised: {_passing(cells, 'p_cells')} / "
        f"{_passing(cells, 'p_cells_normalised')} of {len(cells)} at Bonferroni p ≤ {ALPHA:g}"
    )
    if cells["p_balanced"].notna().any():
        text += (
            f" (balanced over the repeats: {_passing(cells, 'p_balanced')} / "
            f"{_passing(cells, 'p_balanced_normalised')})"
        )
    return text


__all__ = ["cells_headline", "cells_parts"]
