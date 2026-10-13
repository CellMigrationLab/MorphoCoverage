"""The cell-level screen in a condition's reading (``reading_text``): the version of the test
trusted most (balanced and normalised, balanced, normalised, raw), its corrected and
uncorrected p, and how often it gives p <= 0.05 without an effect, in words.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._wording import p_value, percent
from .contrast_cells import ALPHA


def _cell_version(row: pd.Series) -> tuple[str, float, float, float, float, float]:
    """``(label, d, q, chance low, chance high, p)`` of the most trustworthy cell-level test
    computed (balanced and normalised, balanced, normalised, raw): its corrected and its
    uncorrected p."""
    chance = (row["chance_low"], row["chance_high"])
    balanced = (row.get("chance_balanced_low", np.nan), row.get("chance_balanced_high", np.nan))
    versions = [
        ("balanced, normalised", "d_balanced_normalised", "p_balanced_normalised", balanced),
        ("balanced", "d_balanced", "p_balanced", balanced),
        ("normalised", "d_cells_normalised", "p_cells_normalised", chance),
        ("raw", "d_cells", "p_cells", chance),
    ]
    # As provided, a control pooled from repeats the condition was not imaged in gets no
    # number (cells_text): only the balanced test, against its own repeats' control.
    pooled = row.get("other_control_repeats", 0) > 0
    for label, d, p, limits in versions[:2] if pooled else versions:
        if pd.notna(row.get(p)):
            q = float(row[p.replace("p_", "q_", 1)])
            return label, float(row[d]), q, *map(float, limits), float(row[p])
    return "", np.nan, np.nan, np.nan, np.nan, np.nan


def chance_words(low: float, high: float) -> str:
    """How often the cell-level test gives p <= 0.05 without an effect, in words."""
    if pd.isna(low):
        return ""
    if np.isclose(low, high):
        return f"about {percent(low)}"
    return f"{percent(low)}–{percent(high)}"


def _chance_sentence(chance: str, single: bool = False) -> str:
    """How often the cell-level test gives p <= alpha with no real effect; with one repeat
    the variation between repeats is not seen, so the range is for typical variation."""
    if single:
        return (
            f"one repeat cannot show how much repeats vary: at typical variation, with no real "
            f"effect this test gives p ≤ {ALPHA:g} in {chance} of experiments"
        )
    return f"with no real effect it gives p ≤ {ALPHA:g} in {chance} of experiments like this one"


def _why_raw(row: pd.Series, label: str, units: str) -> str:
    """What the raw pooled test does that the version read does not (``units``: the
    observations' plural word)."""
    if row["other_control_repeats"] > 0:
        return "it pools the control of repeats this condition was not imaged in"
    if label.startswith("balanced"):
        return f"it lets the repeat with most {units} weigh more"
    return "it keeps a scale shared within each repeat"


def _cells(row: pd.Series | None, words: tuple[str, str], single: bool = False) -> str:
    """The cell-level screen in words, with its chance of p <= alpha without an effect
    (``words``: the observations' singular and plural words)."""
    if row is None:
        return ""
    label, _, q, low, high, _p_value = _cell_version(row)
    if pd.isna(q):
        return ""
    chance = chance_words(low, high)
    text = f"{words[0]}-level screen (exploratory) q {p_value(q)} ({label}" + (
        f"; {_chance_sentence(chance, single)})" if chance else ")"
    )
    # A withheld test as provided has no q (contrast_cells): its p says whether to explain.
    raw = row["p_cells"] if row["other_control_repeats"] > 0 else row["q_cells"]
    if label != "raw" and q > ALPHA and pd.notna(raw) and raw <= ALPHA:
        text += f"; the raw pooled test is not read here: {_why_raw(row, label, words[1])}"
    return text
