"""Shared wording of the summaries (private): percentages, counts, fields and groups.

Every summary formats numbers through these helpers, so the same number is
written the same way everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def percent(value: float) -> str:
    """A share as a short percentage: ``37%``, ``0.3%``, ``99.5%``; ``undefined`` for NaN.

    Whole percentages round half up (12.5% reads ``13%``). A share that is not
    0 or 1 never reads ``0%`` or ``100%``: below 1% and above 99% it keeps the
    digits that tell it apart (1 in 5000 is ``0.02%``, 199 in 200 ``99.5%``).
    """
    value = float(value)
    if math.isnan(value):
        return "undefined"
    if value in (0.0, 1.0):
        return f"{100 * value:.0f}%"
    share = 100 * value
    if 0 < share < 1:
        return f"{share:.1g}%"
    if 99 < share < 100:
        missing = round(100 - share, 9)  # 99.9 must not read 99.90 from rounding error
        digits = max(1, -math.floor(math.log10(missing)))
        return f"{share:.{digits}f}%"
    return f"{math.floor(share + 0.5):.0f}%"


def percent_fine(value: float) -> str:
    """A share with one decimal below 10% (``6.8%``, ``1.8%``), else as ``percent``: for small
    shares set side by side, which whole percentages would read alike."""
    value = float(value)
    if not math.isnan(value) and 0.01 <= value < 0.1:
        return f"{100 * value:.1f}%"
    return percent(value)


def rate(value: float) -> str:
    """A declared rate as it was set: ``2.5%``, ``5%``, ``95%``, ``0.1%`` (never rounded).

    For settings (a tolerance, a confidence level, a target prevalence), which
    ``percent`` would round (2.5% would read ``3%``). Measured shares go
    through ``percent``.
    """
    return f"{100 * float(value):.10g}%"


def p_value(value: float) -> str:
    """A p- or q-value in a sentence: ``< 0.001``, two significant digits below 0.01, four
    decimals from 0.045 to 0.055 (so a value either side of 0.05 never reads 0.050), three
    below 0.1, else two;
    ``–`` for NaN. The figures' cells write it the same way (``plots_pairwise.format_p``)."""
    value = float(value)
    if math.isnan(value):
        return "–"
    if value < 0.001:
        return "< 0.001"
    if 0.045 <= value < 0.055:
        # Either side of 0.05 at four decimals, so a value that misses never reads 0.050.
        return f"{value:.4f}"
    return f"{value:.2g}" if value < 0.01 else f"{value:.3f}" if value < 0.1 else f"{value:.2f}"


def change(value: float, values: str = "normalised") -> str:
    """The change against the control the paired t-test compares: a fold change of the means
    (``×1.25``; three decimals within 10% of no change and five digits within 1%, so a tiny
    change never reads ``×1.00``), or with raw ``values`` the difference of the means (``+3.2``);
    ``–`` for NaN."""
    value = float(value)
    if math.isnan(value):
        return "–"
    if values != "normalised":
        return f"{value:+.3g}"
    off = abs(value - 1)
    return f"×{value:.2f}" if off >= 0.1 else f"×{value:.3f}" if off >= 0.01 else f"×{value:.5g}"


def number(value: float) -> str:
    """A count or median as a person reads it: ``7``, ``7.5``, ``1234567``; one decimal at most."""
    value = round(float(value), 1)
    return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"


def count(n: float, word: str, plural: str | None = None) -> str:
    """``1 field``, ``3 fields``, ``2.5 fields`` (``plural`` defaults to ``plural(word)``)."""
    many = plural if plural is not None else _plural(word)
    written = number(n)  # singular when the number as written is 1 (1.04 reads "1 field")
    return f"{written} {word if written == '1' else many}"


# Plurals that do not follow the English rules below, for the words this package prints.
_PLURALS = {
    "nucleus": "nuclei",
    "fov_summary": "field summaries",
    "field summary": "field summaries",
    "index": "indices",
    "stratum": "strata",
}


def plural(word: str) -> str:
    """The plural of a unit's name: ``nuclei``, ``patches``, ``colonies``, ``wells``.

    Known irregular words first, then English rules. A word already ending in
    ``s`` is taken as plural (``organoids``), since observation labels this
    package does not know are kept as the caller wrote them.
    """
    if word in _PLURALS:
        return _PLURALS[word]
    if word.endswith("s"):
        return word
    if word.endswith(("x", "z", "ch", "sh")):
        return f"{word}es"
    if len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    return f"{word}s"


_plural = plural


def listing(items: Sequence[object], limit: int = 5) -> str:
    """``a, b and c``, or ``a, b, c, d, e and 3 more`` beyond ``limit`` items."""
    names = [str(item) for item in items]
    if len(names) > limit:
        return f"{', '.join(names[:limit])} and {len(names) - limit} more"
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def fields(n: float) -> str:
    """``1 field``, ``12 fields``, ``2.5 fields``."""
    return count(n, "field")


def duration(seconds: float) -> str:
    """``45 s``, ``12 min``, ``1.5 h``: never ``0 min`` for a short time."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{number(seconds / 3600)} h"


def ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``11th``, ``22nd``."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def shortened(name: object, most: int = 30) -> str:
    """``name`` as a label of at most ``most`` characters, cut in the middle with an ellipsis
    (its start and its end kept); the full name goes where the label is hovered and in the
    CSV."""
    text = str(name)
    if len(text) <= most:
        return text
    # Cut in the middle: names of a series often differ only at their end (siRNA lots, wells).
    head = (most - 1) * 3 // 5
    tail = most - 1 - head
    return text[:head].rstrip() + "…" + text[len(text) - tail :].lstrip()


def column_word(column: object) -> str:
    """A column's own name as a word: ``plate``, ``biological repeat`` (underscores as spaces)."""
    return str(column).replace("_", " ")


def where(group: Mapping[str, Any]) -> str:
    """``condition=MOCK, biological_repeat=1A``, or a pooled label for an empty group."""
    return ", ".join(f"{k}={v}" for k, v in group.items()) or "all FOVs (pooled)"


def rows_of(frame: pd.DataFrame, group: Mapping[str, Any]) -> pd.DataFrame:
    """The rows of ``frame`` belonging to ``group`` (a missing value matches a missing one)."""
    mask = pd.Series(True, index=frame.index)
    for column, value in group.items():
        missing = value is None or (isinstance(value, float) and np.isnan(value))
        mask &= frame[column].isna() if missing else frame[column].eq(value)
    return frame.loc[mask]


__all__ = [
    "change",
    "column_word",
    "count",
    "duration",
    "fields",
    "listing",
    "number",
    "ordinal",
    "p_value",
    "percent",
    "percent_fine",
    "plural",
    "rate",
    "rows_of",
    "shortened",
    "where",
]
