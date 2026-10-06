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

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .._control import ACROSS, MASTER_RAW, OWN, comparison_label
from .._style import CAUTION, table, wrap
from .._wording import change, column_word, count, listing, p_value
from ._experiments import ALPHA
from .contrast_across import NORMALISED, RAW
from .contrast_repeats import MIN_REPEATS
from .repeats_text import _direction

# Every secondary row of every feature, saved by the HTML report (owner, 2026-10-05).
ACROSS_CSV = "tests_across_experiments.csv"


def _repeat_names(row: Mapping[str, Any]) -> str:
    """The repeats a row compared, named: ``R1, R2, R3, R4 and R6``."""
    held = row.get("held")
    return listing([str(r) for r in held], 8) if isinstance(held, list) and held else "–"


def _larger(row: Mapping[str, Any]) -> str:
    """Which of an interaction row's two effects is larger in how many repeats:
    ``VEC_SiCol6a1/VEC_SiNC in 5 of 5``, or each side's count."""
    first = f"{row['condition']}/{row['condition_control']}"
    second = f"{row['reference']}/{row['reference_control']}"
    n, lower, higher = int(row["repeats"]), int(row["lower"]), int(row["higher"])
    if higher == n:
        return f"{first} in {n} of {n}"
    if lower == n:
        return f"{second} in {n} of {n}"
    same = n - lower - higher
    return f"{first} in {higher}, {second} in {lower}" + (f", equal in {same}" if same else "")


def _rows(block: pd.DataFrame, feature: bool = False) -> list[list[str]]:
    rows = []
    for row in block.to_dict("records"):
        # A fold change on the feature's log scale, else the difference.
        folded = pd.notna(row["fold_change"])
        moved = row["fold_change"] if folded else row["effect"]
        tested = pd.notna(row["p_paired_t"])
        rows.append(
            [
                *([row["feature"]] if feature else []),
                row["comparison"],
                _larger(row) if row["kind"] == ACROSS else _direction(row),
                change(moved, NORMALISED if folded else RAW),
                f"{p_value(row['p_paired_t'])} ({p_value(row['q_paired_t'])})" if tested else "–",
                _repeat_names(row),
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


def own_gap_lines(contrast: Any, indent: int = 2) -> list[str]:
    """Beside the main verdicts: those that join two controls apart in intensity, a control
    against its parent (owner, 2026-10-05); empty without."""
    gaps = contrast.settings.get("acquisition_gaps", []) if contrast is not None else []
    pairs = contrast.settings.get("comparisons") or [] if contrast is not None else []
    if not gaps or not pairs:
        return []
    block = pd.DataFrame(
        [{"comparison": comparison_label(c, r), "condition": c, "reference": r} for c, r in pairs]
    )
    return _gap_note(block, gaps, indent)


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
    every = contrast.settings.get("across_features")
    features = every["feature"].nunique() if every is not None and len(every) else 1
    over = f" over the {features} features" if features > 1 else ""
    head = wrap(
        "The contrasts tested, one row per test (contrasts.csv): the main verdicts, "
        f"own_control, corrected within each sub-experiment{over}; then the secondary ones, "
        f"each family corrected on its own{over}:",
        indent,
        hang=indent + 2,
    )
    cells = [
        [r["kind"], r["label"], "sub-experiment" if r["kind"] == OWN else r["family"]]
        for r in rows.to_dict("records")
    ]
    return [head, table(cells, ["kind", "contrast", "corrected within"], indent=indent + 2)]


def _others(contrast: Any, word: str, indent: int) -> list[str]:
    """The other features' secondary rows with q at most ``ALPHA``, then a count of them
    all (empty with one feature)."""
    every = contrast.settings.get("across_features")
    readout = contrast.settings["feature"]
    if every is None or every["feature"].nunique() < 2:
        return []
    others = every[(every["feature"] != readout) & (every["status"] == "ok")]
    hits = others[others["q_paired_t"] <= ALPHA]
    lines = []
    if len(hits):
        lines.append(wrap(f"The other features' rows at q ≤ {ALPHA:g}:", indent, hang=indent + 2))
        columns = ["feature", "comparison", f"by {word}", "change", "paired t: p (q)", f"{word}s"]
        lines.append(table(_rows(hits, feature=True), columns, indent=indent + 2))
        lines += _gap_note(hits, contrast.settings.get("acquisition_gaps", []), indent + 2)
    n_features = every["feature"].nunique() - 1
    owner = f"other {n_features} features'" if n_features > 1 else "other feature's"
    found = "none is" if not len(hits) else "1 is" if len(hits) == 1 else f"{len(hits)} are"
    lines.append(
        wrap(
            f"Of the {owner} {count(len(others), 'row')} "
            f"compared, {found} significant at q ≤ {ALPHA:g}; every row is in "
            f"{ACROSS_CSV}.",
            indent,
            hang=indent + 2,
        )
    )
    return lines


def secondary_headline(contrast: Any) -> str:
    """``across experiments: 1 of 12 significant`` when a secondary row of any feature has
    q at most ``ALPHA``; empty otherwise."""
    every = contrast.settings.get("across_features") if contrast is not None else None
    if every is None:
        every = getattr(contrast, "across", None)
    if every is None or not len(every):
        return ""
    compared = every[every["status"] == "ok"]
    hits = int((compared["q_paired_t"] <= ALPHA).sum())
    return f"across experiments: {hits} of {len(compared)} significant" if hits else ""


def across_parts(contrast: Any, indent: int = 2) -> list[str]:
    """``Compared across experiments`` and its tables, the readout's rows first, then the
    other features' significant rows and their count, then the contrast table (empty with a
    single control)."""
    across = getattr(contrast, "across", None)
    if across is None or not len(across):
        return contrast_lines(contrast, indent)
    word = column_word(contrast.settings.get("repeat_col")) or "repeat"
    compared = across[across["status"] == "ok"]
    gaps = contrast.settings.get("acquisition_gaps", [])
    every = contrast.settings.get("across_features")
    features = every["feature"].nunique() if every is not None and len(every) else 1
    readout = contrast.settings["feature"]
    family = "each family corrected on its own" + (
        f", over its rows of the {features} features" if features > 1 else ""
    )
    lead = f"; first {readout}, the readout:" if features > 1 else ":"
    lines = [
        wrap(
            "Compared across experiments, secondary to the verdicts above: the paired t-test on "
            f"the {word} means, one value per {word}, {family}{lead}",
            indent,
            hang=indent + 2,
        )
    ]
    columns = ["comparison", f"by {word}", "change", "paired t: p (q)", f"{word}s"]
    normalised = compared[compared["kind"] == ACROSS]
    if len(normalised):
        lines.append(
            wrap(
                "The same treatment in two experiments, each condition normalised to its own "
                f"control: in every {word} holding all four, the log ratio of each to its "
                "control, the one minus the other; the larger ratio is named.",
                indent + 2,
                hang=indent + 4,
            )
        )
        larger = ["comparison", f"larger ratio, by {word}", *columns[2:]]
        lines.append(table(_rows(normalised), larger, indent=indent + 4))
        lines += _gap_note(normalised, gaps, indent + 4)
    raw = compared[compared["kind"] == MASTER_RAW]
    if len(raw) or (across["kind"] == MASTER_RAW).any():
        lines.append(
            wrap(
                f"{_master(contrast, word)} The controls against it on the raw values, not "
                f"normalised, in the {word}s holding both, on each feature's scale: the log "
                "ratio of the control's mean to the master's, else their difference; raw "
                "comparisons across experiments assume they were imaged and processed alike.",
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
                f"{listing(list(fell['comparison']), 3)}: a value is not positive, so the "
                "differences of the raw differences are compared instead of log ratios.",
                indent + 2,
                hang=indent + 4,
            )
        )
    fell = compared[(compared["kind"] == MASTER_RAW) & (compared["values"] == RAW)]
    if len(fell):
        lines.append(
            wrap(
                f"{listing(list(fell['comparison']), 3)}: a value is not positive, so the "
                "raw differences are compared instead of log ratios.",
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
    lines += _others(contrast, word, indent + 2)
    return lines + contrast_lines(contrast, indent)


__all__ = [
    "ACROSS_CSV",
    "across_parts",
    "contrast_lines",
    "own_gap_lines",
    "secondary_headline",
]
