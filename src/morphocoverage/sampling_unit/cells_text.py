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
from .._control import versus
from .._identity import IdentityKey, display, identity_key
from .._style import span, table, wrap
from .._wording import column_word, count, features_counted, listing, p_value, percent
from ..schema import observation_words
from ._experiments import control_only, family_noun
from .contrast_cells import (
    ALPHA,
    BALANCED_DRAWS,
    IMBALANCE,
    MIN_BALANCED,
    SPARSE,
    TYPICAL_BETWEEN,
)
from .reading_text import chance_words


def _cells(contrast: Any) -> pd.DataFrame:
    return getattr(contrast, "cells", pd.DataFrame())


def _unit(contrast: Any) -> str:
    """The word for one observation (``nucleus``; ``cell`` when none was declared)."""
    return observation_words(contrast.settings)[0]


def _test(d: float, p: float, corrected: float) -> str:
    """``d, p (q)``: the effect, its uncorrected and its corrected p."""
    if pd.isna(d):
        return "–"
    return f"{d:+.2f}" if pd.isna(p) else f"{d:+.2f}, {p_value(p)} ({p_value(corrected)})"


def _chance(row: dict[str, Any], prefix: str = "chance") -> str:
    return chance_words(row[f"{prefix}_low"], row[f"{prefix}_high"]) or "–"


def _which(rows: pd.DataFrame, cells: pd.DataFrame) -> str:
    """The conditions a reason holds for: ``every condition`` or their names."""
    # Each reason names the conditions it holds for (audit, 2026-10-02).
    return (
        "every condition"
        if len(rows) == len(cells)
        else listing([str(c) for c in rows["condition"]], 3)
    )


def _pooled_from(others: pd.DataFrame, cells: pd.DataFrame, word: str) -> str:
    """The repeats a pooled control comes from that its condition was not imaged in."""
    # Each condition's own number, in the order named (their range beyond three).
    held_out = [int(n) for n in others["other_control_repeats"]]
    if len(set(held_out)) == 1:
        return f"{count(held_out[0], word)} the condition was not imaged in"
    named = len(held_out) <= 3 and len(others) != len(cells)
    numbers = listing([str(n) for n in held_out]) if named else None
    return f"{word}s the condition was not imaged in " + (
        f"({numbers} of them)" if numbers else f"({span(held_out)} of them)"
    )


def _shifted(others: pd.DataFrame, cells: pd.DataFrame, word: str) -> str:
    """The sentence on control means that differ between repeats; empty when they do not."""
    shift = others["control_shift"].max() if len(others) and "control_shift" in others else np.nan
    # A pooled control mixes these levels: the raw test as provided then compares a condition
    # with other repeats' level.
    return (
        f" The control means differ up to {shift:.1f}x between {word}s, so a raw test pooling "
        "them would compare each condition with other repeats' level: read the normalised"
        + (" or balanced" if cells["p_balanced"].notna().any() else "")
        + " columns."
        if pd.notna(shift) and shift > IMBALANCE
        else ""
    )


def _also(cells: pd.DataFrame, word: str, words: str) -> str:
    """Which tests are given: balanced too, or as provided only, and why."""
    # "Also given balanced" only when some condition got that test (audit 10, C6).
    return (
        f"so the test is also given balanced (each "
        f"condition against the control of its own {word}s, the same number of {words} from "
        f"each {word}; the median d over {BALANCED_DRAWS} draws, labels shuffled within each "
        f"{word} for p), and as provided only "
        if cells["p_balanced"].notna().any()
        else f"no balanced test could be drawn, every {word} holding fewer than {MIN_BALANCED} "
        f"{words}, so the test is given as provided only "
        if (cells["sparse_repeats"].map(len) >= cells["repeats"]).all()
        else "no balanced test could be drawn, so the test is given as provided only "
    )


def _balance(contrast: Any, cells: pd.DataFrame) -> str:
    """The balance check in a sentence."""
    word = column_word(contrast.settings["repeat_col"])
    words = observation_words(contrast.settings)[1]
    unbalanced = cells[cells["unbalanced"]]
    if unbalanced.empty:
        return (
            f"Balance: every {word} holds {words} within {percent(IMBALANCE - 1)} of the others, "
            "and each condition is compared with the control of its own repeats, so the test "
            "is run on the data as provided." + _control_only(contrast, cells)
        )
    reasons = []
    uneven = unbalanced[unbalanced["spread"] > IMBALANCE]
    if len(uneven):
        reasons.append(
            f"for {_which(uneven, cells)}, one {word} holds more than {percent(IMBALANCE - 1)} "
            f"more {words} than another on one side (up to {uneven['spread'].max():.1f} times)"
        )
    others = unbalanced[unbalanced["other_control_repeats"] > 0]
    if len(others):
        reasons.append(
            f"for {_which(others, cells)}, the control as provided is pooled from "
            f"{_pooled_from(others, cells, word)}"
        )
    return (
        f"Balance: {'; and '.join(reasons)}; {_also(cells, word, words)}"
        f"where the control comes from the condition's own {word}s."
        + _shifted(others, cells, word)
        + _control_only(contrast, cells)
    )


def _control_only(contrast: Any, cells: pd.DataFrame) -> str:
    """The repeats holding only the control, which no comparison pools; empty without."""
    repeats = [str(r) for r in contrast.settings.get("control_only_repeats", [])]
    if not repeats:
        return ""
    word = column_word(contrast.settings["repeat_col"])
    control = listing(sorted(set(map(str, cells["control_condition"]))), 2)
    return " " + control_only(repeats, control, word)


def _sparse(contrast: Any, cells: pd.DataFrame) -> list[str]:
    """One sentence per repeat too small to balance to, naming the conditions it concerns."""
    word = column_word(contrast.settings["repeat_col"])
    words = observation_words(contrast.settings)[1]
    flagged = cells[cells["sparse_repeats"].map(len) > 0]
    groups: dict[tuple, list[IdentityKey]] = {}
    for row in flagged.to_dict("records"):
        left = int(row["repeats"]) - len(row["sparse_repeats"])
        provided = not row["other_control_repeats"] > 0
        key = (
            tuple(identity_key(r) for r in row["sparse_repeats"]),
            int(row["balanced_repeats"]),
            left,
            provided,
        )
        groups.setdefault(key, []).append(identity_key(row["condition"]))
    lines = []
    repeats = display(list(dict.fromkeys(k for key in groups for k in key[0])))
    conditions_shown = display(list(dict.fromkeys(k for own in groups.values() for k in own)))
    for (keys, kept, left, provided), identities in groups.items():
        names = tuple(repeats[k] for k in keys)
        conditions = [conditions_shown[k] for k in identities]
        # One repeat alone is not balanced over the repeats: no balanced test, the test as
        # provided is read (audit 10, owner, 2026-10-06).
        if not kept and not left:  # every repeat few: said once, in the balance sentence (C6)
            continue
        rest = (
            f"the balanced test balances the other {count(kept, word)}"
            if kept
            else f"one {word} alone would be left, so there is no balanced test"
            + (" and the test as provided is read" if provided else "")
            if left == 1
            else f"no other {word} is left to balance"
        )
        one = len(names) == 1
        lines.append(
            f"{listing(conditions, 3)}: {word if one else f'{word}s'} {listing(list(names), 3)} "
            f"{'holds' if one else 'hold'} very few {words} "
            f"(under {percent(SPARSE)} of the median {word}, or fewer than {MIN_BALANCED}), "
            f"worth checking first. Balancing down to {'it' if one else 'them'} would leave "
            f"little of the others, so {'it is' if one else 'they are'} kept in the test as "
            f"provided and left out of the balanced one: {rest}."
        )
    return lines


def balance_lines(contrast: Any) -> list[str]:
    """The balance check in a sentence and the repeats too small to balance to, unwrapped
    (the text report's and the page's cell-level screen); empty for one repeat compared with
    its own control, which has no balance to check, or without the cell-level test."""
    cells = _cells(contrast)
    if cells.empty or contrast.settings.get("repeat_col") is None:
        return []
    balance = (
        [_balance(contrast, cells)]
        if cells["repeats"].max() > 1 or cells["unbalanced"].any()
        else []
    )
    return balance + _sparse(contrast, cells)


def _as_provided(cells: pd.DataFrame) -> list[list[str]]:
    """The rows of the test as provided, for the conditions compared with their own repeats'
    control only."""
    return [
        [
            versus(row["condition"], row["control_condition"]),
            _test(row["d_cells"], row["p_cells"], row["q_cells"]),
            _test(
                row["d_cells_normalised"],
                row["p_cells_normalised"],
                row["q_cells_normalised"],
            ),
            _chance(row),
        ]
        for row in cells.to_dict("records")
        if row["other_control_repeats"] == 0
    ]


def _pooled_note(contrast: Any, cells: pd.DataFrame, pooled: list[str], indent: int) -> str:
    """Why the conditions whose control pools other repeats get no test as provided, and which
    of them get no test at all."""
    word = column_word(contrast.settings["repeat_col"])
    # Neither test for a condition: said, with why (audit 10, C6).
    none = set(cells.loc[cells["p_balanced"].isna(), "condition"].astype(str))
    neither = [c for c in pooled if c in none]
    return wrap(
        f"Not given as provided for {listing(pooled, 3)}: "
        f"{'its' if len(pooled) == 1 else 'their'} control would include "
        f"{word}s the condition was not imaged in"
        + (
            f"; so no {_unit(contrast)}-level test for {listing(neither, 3)}, which "
            f"{'has' if len(neither) == 1 else 'have'} no balanced test either."
            if neither
            else "."
        ),
        indent,
        hang=indent + 2,
    )


def _balanced_table(cells: pd.DataFrame, words: str, full: bool, indent: int) -> list[str]:
    """The test balanced over the repeats: its title and table, the observations per repeat of
    each draw in the full form."""
    rows = []
    for row in cells[cells["p_balanced"].notna()].to_dict("records"):
        values = [
            versus(row["condition"], row["control_condition"]),
            _test(row["d_balanced"], row["p_balanced"], row["q_balanced"]),
            _test(
                row["d_balanced_normalised"],
                row["p_balanced_normalised"],
                row["q_balanced_normalised"],
            ),
        ]
        if full:
            values.append(f"{int(row['balanced_cells'])} / {int(row['balanced_control_cells'])}")
        values.append(_chance(row, "chance_balanced"))
        rows.append(values)
    names = ["condition", "raw: d, p (q)", "normalised: d, p (q)"]
    names += [f"{words} per repeat, condition / control"] if full else []
    names.append(f"gives p ≤ {ALPHA:g} with no effect")
    return [
        wrap("Balanced over the repeats:", indent, hang=indent + 2),
        table(rows, names, indent=indent),
    ]


def _family_note(settings: dict[str, Any], indent: int) -> list[str]:
    """What q is corrected over, with several features (empty with one)."""
    features = settings.get("family_features")
    if not (features and features > 1):
        return []
    # The version the reading uses shares the family of every feature (_families).
    return [
        wrap(
            "q: the version the reading uses (balanced and normalised when given, else "
            "normalised) is corrected over the "
            f"{features_counted(int(features), settings.get('family_share'), ', ')} and "
            "their conditions "
            f"of each {family_noun(settings)}, as the feature ranking; the other columns over "
            "this "
            "feature's conditions only.",
            indent,
            hang=indent + 2,
        )
    ]


def cells_parts(contrast: Any, indent: int = 2, full: bool = False) -> tuple[list[str], list[str]]:
    """The balance check, the table of the cell-level test and its note (empty without it)."""
    cells = _cells(contrast)
    if cells.empty or contrast.settings.get("repeat_col") is None:
        return [], []
    words = observation_words(contrast.settings)[1]
    balanced = bool(cells["p_balanced"].notna().any())
    lines = [
        wrap(
            f"{_unit(contrast).capitalize()}-level screen (exploratory, as in Plot&Stats): "
            f"Cohen's d on every {_unit(contrast)} against the "
            f"control, the randomization test shuffling the labels over the {words}:",
            indent,
            hang=indent + 2,
        ),
        *(wrap(line, indent + 2, hang=indent + 4) for line in balance_lines(contrast)),
    ]
    # A test that pools the control of repeats the condition was not imaged in gets no
    # number (the page leaves it out too): its d reads the batch between repeats.
    pooled = [
        str(r["condition"]) for r in cells.to_dict("records") if r["other_control_repeats"] > 0
    ]
    as_provided = _as_provided(cells)
    columns = [
        "condition",
        "raw: d, p (q)",
        "normalised: d, p (q)",
        f"gives p ≤ {ALPHA:g} with no effect",
    ]
    if balanced and as_provided:
        lines.append(wrap("As provided:", indent + 2, hang=indent + 4))
    if as_provided:
        lines.append(table(as_provided, columns, indent=indent + 2))
    if pooled:
        lines.append(_pooled_note(contrast, cells, pooled, indent + 2))
    if balanced:
        lines += _balanced_table(cells, words, full, indent + 2)
    lines += _family_note(contrast.settings, indent + 2)
    return lines, (_notes if full else _short_notes)(contrast, cells, balanced)


def _short_notes(contrast: Any, cells: pd.DataFrame, balanced: bool) -> list[str]:
    """The same assumptions in a sentence each, for the short report (the full report and
    docs/report/conditions.md give the formula and the simulations)."""
    words = observation_words(contrast.settings)[1]
    text = (
        f"The {_unit(contrast)}-level test treats the {words} as independent, so it asks "
        "whether these wells differ, not whether the effect repeats. Normalised: each value "
        "divided by its own repeat's control mean"
    )
    text += (
        f". Balanced: the median d over {BALANCED_DRAWS} draws with as many {words} from each "
        "repeat, its p from labels shuffled within each repeat."
        if balanced
        else "."
    )
    return [assumption(text)]


def _notes(contrast: Any, cells: pd.DataFrame, balanced: bool) -> list[str]:
    """The assumptions and limitations of the cell-level test, normalisation and balancing."""
    words = observation_words(contrast.settings)[1]
    single = bool(cells["between_sd"].isna().any())
    notes = [
        assumption(
            f"{_unit(contrast).capitalize()}-level test (Cohen's d, randomization over the "
            f"{words}): it assumes the "
            f"{words} are independent draws, so it asks whether these wells differ, not whether "
            "the effect repeats, and it responds with one or two repeats. Limitation: when the "
            f"difference varies between repeats it gives p ≤ {ALPHA:g} without an effect in "
            f"about 2·P(Z > 1.96/√(1 + n·s²/2)) of experiments (n {words} per well, each repeat "
            "weighted by its share when they differ; s the SD between repeats in observation "
            "SDs; data/simulations/test_choice.py and unbalanced.py), the "
            "'with no effect' column"
            + (
                f" (with one repeat s is unknown and taken as "
                f"{TYPICAL_BETWEEN[0]:g} to {TYPICAL_BETWEEN[1]:g})"
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
            "of the effect that varies between repeats, so the "
            f"{_unit(contrast)}-level chance still applies."
        ),
    ]
    if balanced:
        notes.append(
            assumption(
                f"Balanced: one permutation test against the control of the condition's own "
                f"repeats, d the median over {BALANCED_DRAWS} draws taking the same number of "
                f"{words} from each repeat, p from the condition and control labels shuffled "
                "within each repeat, the same statistic on the same draws for each shuffle. In "
                "data/simulations/unbalanced.py, with one repeat holding 2 to 8 times the cells "
                "of the others, the test as provided gives 22 to 46% false positives where "
                "equal repeats give 16% (s = 0.25), and the balanced test 15 to 19%; with the "
                "control "
                "pooled from repeats the condition was not imaged in, the raw test as provided "
                "gives 46% even when the repeats agree, the balanced test about 5%. Balancing "
                "removes the weight of the repeats holding more, not the independence assumption."
            )
        )
    return notes


def _passing(cells: pd.DataFrame, column: str) -> str:
    tested = cells[cells[column].notna()]
    return f"{int((tested[column.replace('p_', 'q_', 1)] <= ALPHA).sum())}"


def cells_headline(contrast: Any) -> str | None:
    """``cell-level test, raw / normalised: 9 / 10 of 12 at q ≤ 0.05 (balanced
    over the repeats: 10 / 11)``, or ``None``."""
    cells = _cells(contrast)
    if cells.empty or contrast.settings.get("repeat_col") is None:
        return None
    text = (
        f"{_unit(contrast)}-level test, raw / normalised: {_passing(cells, 'p_cells')} / "
        f"{_passing(cells, 'p_cells_normalised')} of {len(cells)} at q ≤ {ALPHA:g}"
    )
    if cells["p_balanced"].notna().any():
        text += (
            f" (balanced over the repeats: {_passing(cells, 'p_balanced')} / "
            f"{_passing(cells, 'p_balanced_normalised')})"
        )
    return text


__all__ = ["balance_lines", "cells_headline", "cells_parts"]
