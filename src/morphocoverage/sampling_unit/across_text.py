"""The secondary rows of the contrast table in words: "Compared across experiments" (owner,
2026-10-05).

After the main verdicts (each condition against its own control), question 9 gives the
secondary rows of ``contrast_across``: each condition normalised to its own control against
a condition of another experiment with the same treatment, and the controls against the
master control on the raw values, with the master's repeats and the one sentence that raw
comparisons across experiments assume they were imaged and processed alike. Each family is
corrected on its own; the rows held in too few repeats are named once as not compared, and
the rows whose values fell back to differences (a mean that is not positive) are named. With
controls apart in intensity (``settings["acquisition_gaps"]``, ``workflows._acquisition_gap``)
the rows that join two of them are named under their table (owner, 2026-10-05).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._control import ACROSS, MASTER_RAW
from .._style import CAUTION, table, wrap
from .._wording import change, column_word, listing, p_value
from .contrast_across import NORMALISED, RAW
from .contrast_repeats import MIN_REPEATS
from .repeats_text import _direction


def _rows(block: pd.DataFrame) -> list[list[str]]:
    rows = []
    for row in block.to_dict("records"):
        # A fold change where every mean is positive, else the difference.
        folded = pd.notna(row["fold_change"])
        moved = row["fold_change"] if folded else row["effect"]
        tested = pd.notna(row["p_paired_t"])
        rows.append(
            [
                row["comparison"],
                _direction(row),
                change(moved, NORMALISED if folded else RAW),
                f"{p_value(row['p_paired_t'])} ({p_value(row['q_paired_t'])})" if tested else "–",
            ]
        )
    return rows


def _gap_note(block: pd.DataFrame, gaps: list[dict], indent: int) -> list[str]:
    """Under a table of secondary rows: those that join two controls apart in intensity."""
    named, joined = [], []
    sides = ["condition", "reference", "condition_control", "reference_control"]
    for row in block.to_dict("records"):
        held = {row[c] for c in sides if pd.notna(row.get(c))}
        found = [" vs ".join(map(str, g["controls"])) for g in gaps if set(g["controls"]) <= held]
        if found:
            named.append(row["comparison"])
            joined += found
    if not named:
        return []
    one = len(named) == 1
    text = (
        f"{listing(named, 3)}: {'joins' if one else 'join'} controls apart in intensity "
        f"({listing(list(dict.fromkeys(joined)), 3)}), so {'it' if one else 'they'} may show "
        "acquisition rather than biology."
    )
    return [wrap(text, f"{' ' * indent}{CAUTION} ", hang=" " * (indent + 3))]


def _master(contrast: Any, word: str) -> str:
    """``VEC_SiNC was imaged in 5 of the 6 biological repeats (not R5).``"""
    found = contrast.settings.get("master_repeats")
    if not found:
        return ""
    master, held, every = found
    if len(held) == len(every):
        return f"The master control {master} was imaged in every {word}."
    missing = listing([str(r) for r in every if r not in held], 6)
    return (
        f"The master control {master} was imaged in {len(held)} of the {len(every)} {word}s, "
        f"not in {missing}."
    )


def contrast_lines(contrast: Any, indent: int = 2) -> list[str]:
    """The contrast table the tests ran, one row per test (saved as ``contrasts.csv``);
    empty with a single control."""
    rows = contrast.settings.get("contrasts")
    if rows is None or not len(rows):
        return []
    head = wrap(
        "The contrasts tested, one row per test (contrasts.csv): the main verdicts "
        "(own_control), then the secondary ones, each family corrected on its own:",
        indent,
        hang=indent + 2,
    )
    cells = [[r["kind"], r["label"], r["family"]] for r in rows.to_dict("records")]
    return [head, table(cells, ["kind", "contrast", "family"], indent=indent + 2)]


def across_parts(contrast: Any, indent: int = 2) -> list[str]:
    """``Compared across experiments`` and its tables, then the contrast table (empty with a
    single control)."""
    across = getattr(contrast, "across", None)
    if across is None or not len(across):
        return contrast_lines(contrast, indent)
    word = column_word(contrast.settings.get("repeat_col")) or "repeat"
    compared = across[across["status"] == "ok"]
    gaps = contrast.settings.get("acquisition_gaps", [])
    lines = [
        wrap(
            "Compared across experiments, secondary to the verdicts above: the paired t-test on "
            f"the {word} means, one value per {word}, each family corrected on its own:",
            indent,
            hang=indent + 2,
        )
    ]
    columns = ["comparison", f"by {word}", "change", "paired t: p (q)"]
    normalised = compared[compared["kind"] == ACROSS]
    if len(normalised):
        lines.append(
            wrap(
                "The same treatment in two experiments, each condition normalised to its own "
                f"control: in every {word} holding all four, the log ratio of each to its "
                "control, the one minus the other.",
                indent + 2,
                hang=indent + 4,
            )
        )
        lines.append(table(_rows(normalised), columns, indent=indent + 4))
        lines += _gap_note(normalised, gaps, indent + 4)
    raw = compared[compared["kind"] == MASTER_RAW]
    if len(raw) or (across["kind"] == MASTER_RAW).any():
        lines.append(
            wrap(
                f"{_master(contrast, word)} The controls against it on the raw values, in the "
                f"{word}s holding both: raw comparisons across experiments assume they were "
                "imaged and processed alike.",
                indent + 2,
                hang=indent + 4,
            )
        )
        if len(raw):
            lines.append(table(_rows(raw), columns, indent=indent + 4))
            lines += _gap_note(raw, gaps, indent + 4)
    fell = compared[(compared["kind"] == ACROSS) & (compared["values"] == RAW)]
    if len(fell):
        lines.append(
            wrap(
                f"{listing(list(fell['comparison']), 3)}: a mean is not positive, so the "
                "differences of the raw differences are compared instead of log ratios.",
                indent + 2,
                hang=indent + 4,
            )
        )
    left = across[across["status"] != "ok"]
    if len(left):
        lines.append(
            wrap(
                f"Not compared: {listing(list(left['comparison']), 4)}: fewer than "
                f"{MIN_REPEATS} {word}s hold every condition of the comparison.",
                indent + 2,
                hang=indent + 4,
            )
        )
    return lines + contrast_lines(contrast, indent)


__all__ = ["across_parts", "contrast_lines"]
