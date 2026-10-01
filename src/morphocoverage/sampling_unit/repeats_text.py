"""The words of the comparison across biological repeats (question 9, ``contrast_repeats``).

One table with a row per condition: the direction against the control of the
same repeat in every repeat, the mean and range of the per-repeat Cohen's d,
and from ``MIN_REPEATS`` repeats the paired t-test on the repeat means, raw and
normalised (log ratio to the control), with the sign-flip test in the full form
where it can reach 0.05. ``repeat_cautions`` gives its notes: what each test
assumes, what it costs in power, and why a test the repeats cannot support is
left out rather than flagged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._caveats import assumption
from .._wording import column_word, count, listing
from ..schema import unit_plural
from .contrast_repeats import MIN_REPEATS


def _range(values: pd.Series) -> str:
    """``-1.52 to 1.42``, or ``1.42`` for one value: statistics to two decimals."""
    low, high = float(values.min()), float(values.max())
    return f"{low:.2f}" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f} to {high:.2f}"


def _repeats(contrast: Any) -> pd.DataFrame:
    """The across-repeat table, empty for a contrast made before it existed."""
    return getattr(contrast, "repeats", pd.DataFrame())


def _direction(row: dict[str, Any]) -> str:
    """``lower in 3 of 3``, ``higher in 3 of 3`` or ``2 lower, 1 higher``."""
    n, lower, higher = int(row["repeats"]), int(row["lower"]), int(row["higher"])
    if lower == n:
        return f"lower in {n} of {n}"
    if higher == n:
        return f"higher in {n} of {n}"
    same = n - lower - higher
    return f"{lower} lower, {higher} higher" + (f", {same} equal" if same else "")


def _p_cell(p: float, corrected: float) -> str:
    return "–" if pd.isna(p) else f"{p:.2g} ({corrected:.2g})"


def _repeat_rows(contrast: Any, full: bool = False) -> tuple[list[list[str]], list[str]]:
    """One row per condition across the biological repeats, and the columns (the sign-flip
    test in the full form only)."""
    words = unit_plural(contrast.settings["observation_unit"])
    repeats = _repeats(contrast)
    flip = full and bool(repeats["p_sign_flip"].notna().any())
    tested = bool(repeats["p_paired_t"].notna().any())
    rows = []
    for row in repeats.to_dict("records"):
        d = (
            "–"
            if pd.isna(row["d_mean"])
            else f"{row['d_mean']:.2f} ({_range(pd.Series([row['d_min'], row['d_max']]))})"
        )
        cells = [f"{row['condition']} vs {row['control_condition']}", _direction(row), d]
        if tested:
            cells.append(_p_cell(row["p_paired_t"], row["p_paired_t_bonferroni"]))
            cells.append(
                _p_cell(row["p_paired_t_normalised"], row["p_paired_t_normalised_bonferroni"])
            )
        if flip:
            cells.append(_p_cell(row["p_sign_flip"], row["p_sign_flip_bonferroni"]))
        rows.append(cells)
    columns = ["condition", "vs control, by repeat", f"d {words}: mean (range)"]
    columns += [
        *(["paired t p (Bonf.)", "normalised"] if tested else []),
        *(["sign-flip p (Bonf.)"] if flip else []),
    ]
    return rows, columns


def _repeat_head(contrast: Any) -> str:
    repeats = _repeats(contrast)
    word = column_word(contrast.settings["repeat_col"])
    words = unit_plural(contrast.settings["observation_unit"])
    n = int(repeats["repeats"].max())
    return (
        f"Across the {count(n, word)} (each repeat's value is the mean of its {words}, paired "
        "with the control of the same repeat):"
    )


def repeat_cautions(contrast: Any, full: bool = True) -> list[str]:
    """The notes of the across-repeat table, once each (see ``contrast_repeats``): which
    tests ran and what they assume, in a sentence (``full=False``, the short report) or with
    the simulated power (``full=True``). None is a caution: the tests the repeats cannot
    support are left out, not flagged."""
    repeats = _repeats(contrast)
    if repeats.empty:
        return []
    word = column_word(contrast.settings["repeat_col"])
    notes = []
    untested = repeats[repeats["status"] != "ok"]
    if len(untested):
        n = int(untested["repeats"].max())
        notes.append(
            assumption(
                f"With {count(n, word)} holding both sides, "
                f"{listing(list(untested['condition']), 3)} "
                f"{'is' if len(untested) == 1 else 'are'} described by direction and effect "
                f"size; the paired test across repeats needs {MIN_REPEATS}."
            )
        )
    if repeats["p_paired_t"].notna().any() and not full:
        notes.append(
            assumption(
                f"Across {word}s, the paired t-test on the repeat means (as in SuperPlots) "
                f"treats each {word} as one replicate: it keeps false positives at 5%, but with "
                f"3 {word}s it needs a large effect (the simulated power is in the full report "
                "and docs/condition_tests.md); Bonferroni-corrected over the conditions."
            )
        )
    elif repeats["p_paired_t"].notna().any():
        flip = repeats["p_sign_flip"].notna().any()
        notes.append(
            assumption(
                f"Across {word}s, the paired t-test on the repeat means (as in SuperPlots) "
                f"treats each {word} as one replicate and assumes the differences (raw) or the "
                "log ratios to the control (normalised) are normally distributed and "
                f"independent between {word}s. It keeps false positives at 5% when the effect "
                f"varies between {word}s, which the cell-level test does not; the cost is power: "
                f"with 3 {word}s and 30 cells a well, d 1.5 is found in 98, 90 and 63% of "
                f"experiments and d 1.0 in 85, 67 and 39% as {word}s differ by 0, 0.25 and 0.5 "
                "cell SD (data/simulations/test_choice.py)"
                + (
                    "; the exact sign-flip test (full report) assumes only that each "
                    "difference is as likely positive as negative"
                    if flip
                    else ""
                )
                + "; Bonferroni-corrected over the conditions."
            )
        )
    return notes


__all__ = ["repeat_cautions"]
