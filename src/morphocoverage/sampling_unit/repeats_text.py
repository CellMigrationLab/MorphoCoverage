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
from .._style import span
from .._wording import column_word, count, listing, p_value
from ..schema import observation_words
from ._experiments import family_words
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
    return "–" if pd.isna(p) else f"{p_value(p)} ({p_value(corrected)})"


def _repeat_rows(contrast: Any, full: bool = False) -> tuple[list[list[str]], list[str]]:
    """One row per condition across the biological repeats, and the columns (the sign-flip
    test in the full form only)."""
    words = observation_words(contrast.settings)[1]
    repeats = _repeats(contrast)
    flip = full and bool(repeats["p_sign_flip"].notna().any())
    tested = bool(repeats["p_paired_t"].notna().any())
    rows = []
    for row in repeats.to_dict("records"):
        d = (
            "–"
            if pd.isna(row["d_mean"])
            else f"{row['d_mean']:+.2f} ({_range(pd.Series([row['d_min'], row['d_max']]))})"
        )
        cells = [f"{row['condition']} vs {row['control_condition']}", _direction(row), d]
        if tested:
            cells.append(_p_cell(row["p_paired_t"], row["q_paired_t"]))
            cells.append(_p_cell(row["p_paired_t_normalised"], row["q_paired_t_normalised"]))
        if flip:
            cells.append(_p_cell(row["p_sign_flip"], row["q_sign_flip"]))
        rows.append(cells)
    columns = ["condition", "vs control, by repeat", f"d {words} per repeat: mean (range)"]
    columns += [
        *(["paired t, raw: p (q)", "paired t, normalised: p (q)"] if tested else []),
        *(["sign-flip: p (q)"] if flip else []),
    ]
    return rows, columns


def _repeat_head(contrast: Any) -> str:
    repeats = _repeats(contrast)
    word = column_word(contrast.settings["repeat_col"])
    words = observation_words(contrast.settings)[1]
    held = sorted({int(n) for n in repeats["repeats"]})
    # Each condition's own number of repeats: one count when they share it, else the range.
    across = (
        f"the {count(held[0], word)}"
        if len(held) == 1
        else f"the {word}s, {span(held)} per condition"
    )
    return (
        f"Across {across} (each repeat's value is the mean of its {words}, paired with the "
        "control of the same repeat):"
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
        held = [int(n) for n in untested["repeats"]]
        names = listing([str(c) for c in untested["condition"]], 3)
        # Each condition's own number of repeats holding both sides (the range beyond three).
        numbers = (
            count(held[0], word)
            if len(set(held)) == 1
            else f"{listing([str(n) for n in held])} {word}s"
            if len(held) <= 3
            else f"{span(held)} {word}s"
        )
        notes.append(
            assumption(
                f"{names}, with {numbers} holding both sides, "
                f"{'is' if len(untested) == 1 else 'are'} described by direction and effect "
                f"size; the paired test across repeats needs {MIN_REPEATS}."
            )
        )
    if repeats["p_paired_t"].notna().any() and not full:
        # As ``reading_text._paired``: the normalised q, the raw q where there is none.
        raw = repeats["p_paired_t"].notna() & repeats["q_paired_t_normalised"].isna()
        used = (
            "the raw q: a repeat mean is not positive, so there is no ratio to the control"
            if raw[repeats["p_paired_t"].notna()].all()
            else "the normalised q"
            + (
                f", the raw q for {listing([str(c) for c in repeats.loc[raw, 'condition']], 3)}"
                " where a repeat mean is not positive"
                if raw.any()
                else ""
            )
        )
        notes.append(
            assumption(
                f"The paired t-test on the repeat means treats each {word} as one replicate: "
                f"it keeps false positives at 5%, but with 3 {word}s only large, consistent "
                f"effects reach q ≤ 0.05. q: Benjamini-Hochberg {family_words(contrast.settings)}; "
                f"'significant across repeats' uses {used}."
            )
        )
    elif repeats["p_paired_t"].notna().any():
        unit = observation_words(contrast.settings)[0]
        flip = repeats["p_sign_flip"].notna().any()
        notes.append(
            assumption(
                f"Across {word}s, the paired t-test on the repeat means (as in SuperPlots) "
                f"treats each {word} as one replicate and assumes the differences (raw) or the "
                "log ratios to the control (normalised) are normally distributed and "
                f"independent between {word}s. It keeps false positives at 5% when the effect "
                f"varies between {word}s, which the {unit}-level test does not; the cost is "
                "power: "
                f"with 3 {word}s and 30 cells a well, d 1.5 is found in 98, 90 and 63% of "
                f"experiments and d 1.0 in 85, 67 and 39% as {word}s differ by 0, 0.25 and 0.5 "
                "cell SD (data/simulations/test_choice.py)"
                + (
                    "; the exact sign-flip test (full report) assumes only that each "
                    "difference is as likely positive as negative"
                    if flip
                    else ""
                )
                + ". p uncorrected, q corrected (Benjamini-Hochberg within each "
                "sub-experiment: the conditions sharing their repeats and control)."
            )
        )
    return notes


__all__ = ["repeat_cautions"]
