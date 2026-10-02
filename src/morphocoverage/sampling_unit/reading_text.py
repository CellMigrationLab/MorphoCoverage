"""Question 9 in a sentence per condition: what the tests add up to (``condition_readings``).

The contrast gives several answers (the direction in every repeat, the paired
t-test across repeats, the cell-level test, raw and normalised, balanced when
the repeats are not, and the agreement of independent reagents); this module
reads them together, condition by condition, so the reader does not have to.
Each condition gets one verdict:

- **significant across repeats**: the paired test across repeats gives q <= 0.05 (Benjamini-Hochberg
  within the condition's sub-experiment, ``_experiments``; on normalised values when they
  exist, else raw), with the condition on one side of the control in every repeat;
- **same direction, not significant**: on one side of the control in every repeat
  (two or more), and the paired test does not reach q <= 0.05 or cannot run (a
  consistent trend, not evidence of no effect);
- **no consistent direction**: lower in some repeats and higher in others;
- **one repeat**: imaged in a single repeat, so only the cell-level test speaks.

Then the evidence: the direction, the mean per-repeat Cohen's d and the change in the
feature's own terms (the geometric mean of the per-repeat ratios to the control, as the
normalised paired test reads them), the paired test's p and q,
the cell-level test's p in its most trustworthy version (balanced and
normalised when the repeats are unbalanced, else normalised, else raw) with
how often it gives p <= 0.05 without an effect at the variation between repeats seen
(said as such: it is not the chance that this result is false), and, with a ``target``
column, whether the target's reagents agree. When the raw pooled test is
significant and that version is not, the sentence says what the raw test does
differently (pools the control of repeats the condition was not imaged in, lets
the repeat with most cells weigh more, or keeps a scale shared within each
repeat), as a fact about the tests, not a cause, and gives no number for it (owner,
2026-10-01: a q beside "not valid" is still read as the result). No new test is run:
every p is the contrast's (``contrast_repeats``, ``contrast_cells``, ``reagent_agreement``),
and the change in the feature's own terms comes from its per-repeat means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._style import wrap
from .._wording import column_word, count, listing, p_value, percent
from ._experiments import family_words
from .contrast_cells import ALPHA
from .contrast_repeats import MIN_REPEATS

CONFIRMED = "significant across repeats"
CONSISTENT = "same direction in every repeat, not significant"
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
                "q_paired_t": np.nan,
                "q_paired_t_normalised": np.nan,
            }
        )
    return _by_condition(pd.DataFrame(rows))


def _cell_version(row: pd.Series) -> tuple[str, float, float, float, float, float]:
    """``(label, d, q, chance low, chance high, p)`` of the most trustworthy cell-level test
    computed (balanced and normalised, balanced, normalised, raw): its corrected and its
    uncorrected p."""
    chance = (row["chance_low"], row["chance_high"])
    balanced = (row.get("chance_balanced_low", np.nan), row.get("chance_balanced_high", np.nan))
    for label, d, p, limits in (
        ("balanced, normalised", "d_balanced_normalised", "p_balanced_normalised", balanced),
        ("balanced", "d_balanced", "p_balanced", balanced),
        ("normalised", "d_cells_normalised", "p_cells_normalised", chance),
        ("raw", "d_cells", "p_cells", chance),
    ):
        if pd.notna(row.get(p)):
            q = float(row[p.replace("p_", "q_", 1)])
            return label, float(row[d]), q, *map(float, limits), float(row[p])
    return "", np.nan, np.nan, np.nan, np.nan, np.nan


def _chance_words(low: float, high: float) -> str:
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


def _why_raw(row: pd.Series, label: str) -> str:
    """What the raw pooled test does that the version read does not."""
    if row["other_control_repeats"] > 0:
        return "it pools the control of repeats this condition was not imaged in"
    if label.startswith("balanced"):
        return "it lets the repeat with most cells weigh more"
    return "it keeps a scale shared within each repeat"


def _paired(row: pd.Series, n: int, word: str) -> tuple[bool, str]:
    """``(confirmed, words)`` of the paired test across repeats: its p and its q."""
    q, p, which = row["q_paired_t_normalised"], row.get("p_paired_t_normalised"), "normalised"
    if pd.isna(q):
        q, p, which = row["q_paired_t"], row.get("p_paired_t"), "raw"
    if pd.isna(q):
        return False, f"the paired test needs {count(MIN_REPEATS, word)}"
    confirmed = bool(q <= ALPHA)
    uncorrected = f"p {p_value(p)}, " if pd.notna(p) else ""
    words = f"paired t-test on the {n} repeat means {uncorrected}q {p_value(q)}"
    words += "" if which == "normalised" else " (raw values)"
    raw = row["q_paired_t"]
    if which == "normalised" and pd.notna(raw) and (raw <= ALPHA) != confirmed:
        words += f"; on raw values q {p_value(raw)}"
    return confirmed, words


def _cells(row: pd.Series | None, single: bool = False) -> str:
    """The cell-level screen in words, with its chance of p <= alpha without an effect."""
    if row is None:
        return ""
    label, _, q, low, high, _p_value = _cell_version(row)
    if pd.isna(q):
        return ""
    chance = _chance_words(low, high)
    text = f"cell-level screen (exploratory) q {p_value(q)} ({label}" + (
        f"; {_chance_sentence(chance, single)})" if chance else ")"
    )
    raw = row["q_cells"]
    if label != "raw" and q > ALPHA and pd.notna(raw) and raw <= ALPHA:
        text += f"; the raw pooled test is not read here: {_why_raw(row, label)}"
    return text


def change_words(contrast: Any, condition: Any) -> str:
    """The change in the feature's own terms: the geometric mean of the per-repeat ratios of
    the condition's mean to its control's (what the normalised paired test reads)."""
    table = getattr(contrast, "table", None)
    if table is None or "mean" not in table:
        return ""
    rows = table[table["condition"] == condition]
    ratio = (rows["mean"] / rows["control_mean"]).to_numpy(dtype=float)
    if not len(ratio) or not (np.isfinite(ratio).all() and (ratio > 0).all()):
        return ""
    fold = float(np.exp(np.mean(np.log(ratio))))
    side = "lower" if fold < 1 else "higher"
    return f"mean {percent(abs(fold - 1))} {side} than the control"


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
        d = f"Cohen's d per repeat, mean {row['d_mean']:+.2f}" if pd.notna(row["d_mean"]) else ""
        change = change_words(contrast, condition)
        d = ", ".join(x for x in (d, change) if x)
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
        cell = _cells(cells.loc[condition] if condition in cells.index else None, n == 1)
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
    unit = contrast.settings["observation_unit"]
    head = (
        f"Reading, condition by condition, against {control}. Significant across repeats: the "
        "paired t-test on the biological repeat means (as SuperPlots; values normalised to "
        f"each repeat's control) gives q ≤ 0.05 after Benjamini-Hochberg "
        f"{family_words(contrast.settings, own=True)}: this is the result to report. Same "
        "direction, not "
        "significant: on one side of the control in every repeat without reaching q ≤ 0.05, "
        "a consistent trend and not evidence of no effect (with 3 repeats only large, "
        "consistent effects reach it). Beside it, the cell-level screen (as Plot&Stats, every "
        f"{unit} an independent draw) is exploratory: it finds candidates, not effects that "
        "repeat:"
    )
    design = _design(contrast, control)
    return [wrap(design + head, indent, hang=indent + 2)] + [
        wrap(f"- {text}", indent + 2, hang=indent + 4) for text in readings["reading"]
    ]


def _design(contrast: Any, control: str) -> str:
    """The design in one sentence: the sub-experiments when the conditions were imaged in
    separate biological repeats, each with its own control; empty with one."""
    across = getattr(contrast, "repeats", None)
    if across is None or "experiment" not in across or across["experiment"].nunique() < 2:
        return ""
    word = column_word(contrast.settings.get("repeat_col"))
    groups = [
        listing(list(block["condition"].astype(str)), 3)
        for _, block in across.groupby("experiment", sort=False)
    ]
    sizes = sorted({int(n) for n in across["repeats"]})
    held = count(sizes[0], word) if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]} {word}s"
    return (
        f"Design: {len(groups)} sub-experiments, each with its own {control} in its own "
        f"{held} ({'; '.join(groups[:6])}{'; ...' if len(groups) > 6 else ''}). "
    )


def readings_headline(contrast: Any) -> str | None:
    """The verdicts in a few words, the readout first: ``Area, 2 conditions across 3
    biological repeats: same direction in every repeat, not significant: siCCT8#5 and
    siCCT8#8 (lower)``; with one repeat ``no conclusion across repeats (n = 1)``; ``None``
    without a contrast."""
    readings = condition_readings(contrast)
    if readings.empty:
        return None
    word = column_word(contrast.settings.get("repeat_col"))
    feature = contrast.settings["feature"]
    n_repeats = int(_direction_rows(contrast)["repeats"].max())
    table = getattr(contrast, "table", pd.DataFrame())
    repeat_col = contrast.settings.get("repeat_col")
    total = int(table[repeat_col].nunique()) if repeat_col in table else n_repeats
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
                names.append(f"{listing(these, 8)}{tail}")
        if verdict == ONE_REPEAT:
            groups.append("; ".join(names))
        else:
            groups.append(f"{verdict}: {'; '.join(names)}")
    if n_repeats == 1:
        text = (
            f"{feature}, {count(len(readings), 'condition')} in one {word} (n = 1, so no "
            "conclusion across repeats): "
        )
    else:
        text = f"{feature}, {count(len(readings), 'condition')} " + (
            f"across {count(n_repeats, word)}: "
            if total <= n_repeats
            else f"against the control, each in {count(n_repeats, word)} ({total} in all): "
        )
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
    single = int(_direction_rows(contrast)["repeats"].max()) == 1
    return f"; exploratory cell-level screen: {passing} of {len(tested)} at q ≤ {ALPHA:g}" + (
        f" ({_chance_sentence(chance, single)})" if chance else ""
    )


__all__ = [
    "CONFIRMED",
    "CONSISTENT",
    "MIXED",
    "ONE_REPEAT",
    "VERDICTS",
    "change_words",
    "condition_readings",
    "readings_headline",
    "readings_parts",
]
