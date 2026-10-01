"""Question 9 in a sentence per condition: what the tests add up to (``condition_readings``).

The contrast gives several answers (the direction in every repeat, the paired
t-test across repeats, the cell-level test, raw and normalised, balanced when
the repeats are not, and the agreement of independent reagents); this module
reads them together, condition by condition, so the reader does not have to.
Each condition gets one verdict:

- **confirmed**: the paired test across repeats gives Bonferroni p <= 0.05
  (on normalised values when they exist, else raw), with the condition on one
  side of the control in every repeat;
- **same direction, not confirmed**: on one side of the control in every repeat
  (two or more), and the paired test does not confirm it or cannot run;
- **no consistent direction**: lower in some repeats and higher in others;
- **one repeat**: imaged in a single repeat, so only the cell-level test speaks.

Then the evidence: the direction and mean per-repeat d, the paired test's p,
the cell-level test's p in its most trustworthy version (balanced and
normalised when the repeats are unbalanced, else normalised, else raw) with
its false-positive chance here (how often it gives p <= 0.05 without an
effect at the variation between repeats seen), and, with a ``target``
column, whether the target's reagents agree. When the raw pooled test is
significant and that version is not, the sentence says what the raw test does
differently (pools the control of repeats the condition was not imaged in, lets
the repeat with most cells weigh more, or keeps a scale shared within each
repeat), as a fact about the tests, not a cause. No new statistic is computed: every number is the
contrast's (``contrast_repeats``, ``contrast_cells``, ``reagent_agreement``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._style import wrap
from .._wording import column_word, count, listing, percent
from .contrast_cells import ALPHA
from .contrast_repeats import MIN_REPEATS

CONFIRMED = "confirmed across repeats"
CONSISTENT = "same direction in every repeat, not confirmed"
MIXED = "no consistent direction"
ONE_REPEAT = "one repeat"
VERDICTS = (CONFIRMED, CONSISTENT, MIXED, ONE_REPEAT)


def _by_condition(frame: Any) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    return frame.drop_duplicates("condition").set_index("condition")


def _direction_rows(contrast: Any) -> pd.DataFrame:
    """The across-repeat rows, or with one repeat their equivalent from the per-repeat table."""
    across = _by_condition(getattr(contrast, "repeats", None))
    if len(across):
        return across
    table = contrast.table[contrast.table["mean"].notna() & contrast.table["control_mean"].notna()]
    rows = []
    for condition, block in table.groupby("condition", sort=False):
        difference = block["mean"] - block["control_mean"]
        rows.append(
            {
                "condition": condition,
                "control_condition": block["control_condition"].iloc[0],
                "repeats": len(block),
                "lower": int((difference < 0).sum()),
                "higher": int((difference > 0).sum()),
                "d_mean": float(block["d_objects"].mean()),
                "p_paired_t_bonferroni": np.nan,
                "p_paired_t_normalised_bonferroni": np.nan,
            }
        )
    return _by_condition(pd.DataFrame(rows))


def _cell_version(row: pd.Series) -> tuple[str, float, float, float, float]:
    """``(label, d, Bonferroni p, chance low, chance high)`` of the most trustworthy cell-level
    test computed: balanced and normalised, balanced, normalised, raw."""
    chance = (row["chance_low"], row["chance_high"])
    balanced = (row.get("chance_balanced_low", np.nan), row.get("chance_balanced_high", np.nan))
    for label, d, p, limits in (
        ("balanced, normalised", "d_balanced_normalised", "p_balanced_normalised", balanced),
        ("balanced", "d_balanced", "p_balanced", balanced),
        ("normalised", "d_cells_normalised", "p_cells_normalised", chance),
        ("raw", "d_cells", "p_cells", chance),
    ):
        if pd.notna(row.get(p)):
            return label, float(row[d]), float(row[f"{p}_bonferroni"]), *map(float, limits)
    return "", np.nan, np.nan, np.nan, np.nan


def _chance_words(low: float, high: float) -> str:
    if pd.isna(low):
        return ""
    if np.isclose(low, high):
        return f"about {percent(low)}"
    return f"{percent(low)}–{percent(high)}"


def _why_raw(row: pd.Series, label: str) -> str:
    """What the raw pooled test does that the version read does not."""
    if row["other_control_repeats"] > 0:
        return "it pools the control of repeats this condition was not imaged in"
    if label.startswith("balanced"):
        return "it lets the repeat with most cells weigh more"
    return "it keeps a scale shared within each repeat"


def _paired(row: pd.Series, n: int, word: str) -> tuple[bool, str]:
    """``(confirmed, words)`` of the paired test across repeats."""
    p = row["p_paired_t_normalised_bonferroni"]
    which = "normalised"
    if pd.isna(p):
        p, which = row["p_paired_t_bonferroni"], "raw"
    if pd.isna(p):
        return False, f"the paired test needs {count(MIN_REPEATS, word)}"
    confirmed = bool(p <= ALPHA)
    words = f"paired test p {p:.2g}" + ("" if which == "normalised" else " (raw)")
    raw = row["p_paired_t_bonferroni"]
    if which == "normalised" and pd.notna(raw) and (raw <= ALPHA) != confirmed:
        words += f", raw values p {raw:.2g}"
    return confirmed, words


def _cells(row: pd.Series | None) -> str:
    """The cell-level test in words, with its chance of p <= alpha without an effect."""
    if row is None:
        return ""
    label, _, p, low, high = _cell_version(row)
    if pd.isna(p):
        return ""
    chance = _chance_words(low, high)
    text = f"cell-level p {p:.2g} ({label}" + (
        f"; false-positive chance {chance} here)" if chance else ")"
    )
    raw = row["p_cells_bonferroni"]
    if label != "raw" and p > ALPHA and pd.notna(raw) and raw <= ALPHA:
        text += f"; the raw pooled test gives p {raw:.2g}, but {_why_raw(row, label)}"
    return text


def _reagents(contrast: Any, condition: Any, word: str) -> str:
    targets = getattr(contrast, "targets", None)
    if targets is None or len(targets) == 0:
        return ""
    for row in targets.to_dict("records"):
        if condition in row["reagents"]:
            others = [r for r in row["reagents"] if r != condition]
            verb = "agrees" if len(others) == 1 else "agree"
            return (
                f"{listing(others, 2)} ({row['target']}) {verb} in {row['agree']} of "
                f"{count(int(row['repeats']), word)}"
            )
    return ""


def condition_readings(contrast: Any) -> pd.DataFrame:
    """One row per condition: ``condition``, ``control_condition``, ``verdict`` (one of
    ``VERDICTS``), ``direction`` (lower, higher or mixed) and ``reading``, the sentence."""
    rows = _direction_rows(contrast)
    columns = ["condition", "control_condition", "verdict", "direction", "reading"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    cells = _by_condition(getattr(contrast, "cells", None))
    word = column_word(contrast.settings.get("repeat_col"))
    out = []
    for condition, row in rows.iterrows():
        n, lower, higher = int(row["repeats"]), int(row["lower"]), int(row["higher"])
        direction = "lower" if lower == n else "higher" if higher == n else "mixed"
        d = f"d {row['d_mean']:.2f}" if pd.notna(row["d_mean"]) else ""
        if n == 1:
            verdict, confirmed = ONE_REPEAT, False
            parts = [f"{direction} than the control in its one {word}" + (f" ({d})" if d else "")]
        else:
            confirmed, paired = _paired(row, n, word)
            if direction == "mixed":
                verdict = MIXED
                parts = [f"lower in {lower} and higher in {higher} of {n} {word}s"]
            else:
                verdict = CONFIRMED if confirmed else CONSISTENT
                parts = [f"{direction} in {n} of {n} {word}s" + (f" ({d})" if d else "")]
            parts.append(paired)
        cell = _cells(cells.loc[condition] if condition in cells.index else None)
        parts += [cell] if cell else []
        reagents = _reagents(contrast, condition, word)
        parts += [reagents] if reagents else []
        out.append(
            {
                "condition": condition,
                "control_condition": row["control_condition"],
                "verdict": verdict,
                "direction": direction,
                "reading": f"{condition} — {verdict}: " + "; ".join(parts) + ".",
            }
        )
    return pd.DataFrame(out, columns=columns)


def readings_parts(contrast: Any, indent: int = 2) -> list[str]:
    """The readings under a one-line heading (empty without a contrast to read)."""
    readings = condition_readings(contrast)
    if readings.empty:
        return []
    control = listing(sorted(set(map(str, readings["control_condition"]))), 3)
    head = (
        f"Reading, condition by condition, against {control} (paired test: Bonferroni, on "
        "normalised values; cell-level: the version below that is least biased by the "
        "repeats):"
    )
    return [wrap(head, indent, hang=indent + 2)] + [
        wrap(f"- {text}", indent + 2, hang=indent + 4) for text in readings["reading"]
    ]


def readings_headline(contrast: Any) -> str | None:
    """The verdicts in a few words: ``2 conditions across 3 biological repeats: same direction
    in every repeat, not confirmed: siCCT8#5 and siCCT8#8 (lower); no consistent direction:
    siCCT7#3 and siCCT7#6``, or ``None`` without a contrast."""
    readings = condition_readings(contrast)
    if readings.empty:
        return None
    word = column_word(contrast.settings.get("repeat_col"))
    n_repeats = int(_direction_rows(contrast)["repeats"].max())
    groups = []
    for verdict in VERDICTS:
        block = readings[readings["verdict"] == verdict]
        if block.empty:
            continue
        names = []
        for direction in ("lower", "higher", "mixed"):
            these = list(block.loc[block["direction"] == direction, "condition"].astype(str))
            if these:
                tail = "" if direction == "mixed" else f" ({direction})"
                names.append(f"{listing(these, 4)}{tail}")
        groups.append(f"{verdict}: {'; '.join(names)}")
    text = f"{count(len(readings), 'condition')} across {count(n_repeats, word)}: "
    text += "; ".join(groups)
    if n_repeats < MIN_REPEATS:
        text += _cells_headline(contrast)
    return text


def _cells_headline(contrast: Any) -> str:
    """Without the paired test the cell-level test is the one that responds: its count and
    false-positive chance, for the headline."""
    cells = _by_condition(getattr(contrast, "cells", None))
    versions = [_cell_version(row) for _, row in cells.iterrows()]
    tested = [v for v in versions if pd.notna(v[2])]
    if not tested:
        return ""
    lows = [v[3] for v in tested if pd.notna(v[3])]
    highs = [v[4] for v in tested if pd.notna(v[4])]
    chance = _chance_words(min(lows), max(highs)) if lows and highs else ""
    passing = sum(v[2] <= ALPHA for v in tested)
    return f"; cell-level test: {passing} of {len(tested)} at Bonferroni p ≤ {ALPHA:g}" + (
        f" (false-positive chance {chance} here)" if chance else ""
    )


__all__ = [
    "CONFIRMED",
    "CONSISTENT",
    "MIXED",
    "ONE_REPEAT",
    "VERDICTS",
    "condition_readings",
    "readings_headline",
    "readings_parts",
]
