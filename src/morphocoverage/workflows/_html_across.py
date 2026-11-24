"""The table across repeats on the HTML page (private; ``_html_tests``, ``html_report``): one
row per feature and condition, the direction in the repeats, the change, the per-repeat d, p,
the adjusted p and the verdict, saved as ``tests_across_repeats.csv``.

With exactly 3 biological repeats (the minimum for a test) each row also shows the three
repeat contrasts and the t-based 95% interval of the effect, and the caption says once that
this is very limited biological replication (plan v2, owner, 2026-10-06); the CSV always
carries them (``repeat_contrasts``, ``ci_low``, ``ci_high``).
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._style import low_to_high
from .._wording import change, contrasts_words, p_value
from ..sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS, untested_reason
from ..sampling_unit.discovery import declared_scales
from ..sampling_unit.reading_text import (
    CONFIRMED,
    CONSISTENT,
    LIMITED_REPLICATION,
    MIXED,
    NOT_ASSESSED,
    NOT_SIGNIFICANT,
)
from ..statistics import ALPHA
from ._html_data import data_buttons, write_csv

# Above this many rows a table shows its first ones and a button for the rest.
FIRST_ROWS = 20


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]], name: str) -> str:
    """A sortable table, its first ``FIRST_ROWS`` rows and a button above that."""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
    )
    table = (
        f'<div class="table-wrap"><table class="sortable"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
    if len(rows) <= FIRST_ROWS:
        return table
    return (
        f'<div class="long-table">{table}<button type="button" class="show-all" '
        f'data-rows="{FIRST_ROWS}" aria-expanded="false">Show all {len(rows)} {name}</button></div>'
    )


def _direction(lower: int, higher: int, n: int) -> str:
    if n == 0:
        return "–"
    if lower == n:
        return f"lower in {n} of {n}"
    if higher == n:
        return f"higher in {n} of {n}"
    return f"lower in {lower}, higher in {higher} of {n}"


def _verdict(
    q: float,
    lower: int,
    higher: int,
    n: int,
    confirmed: bool | None = None,
    one: str = "cell",
) -> str:
    """The verdict in the reading's words (``reading_text.VERDICTS``): a direction that
    changes between repeats first, then significant across repeats (the discovery's
    ``confirmed``, which also needs the cell-level screen, when given; else q <= 0.05); the
    screen named for the observations, ``one`` (``patch-level``). Enough repeats and no q:
    not assessed, with the reason (``contrast_repeats.untested_reason``), never "not
    significant" (#90)."""
    if n < MIN_REPEATS:
        return f"not tested (needs {MIN_REPEATS} repeats)"
    if not (lower == n or higher == n):
        return MIXED
    if pd.isna(q):
        return f"{NOT_ASSESSED}: {untested_reason(n, np.nan)}"
    significant = pd.notna(q) and q <= ALPHA
    if confirmed if confirmed is not None else significant:
        return CONFIRMED
    if significant:
        return f"adjusted p ≤ 0.05, but not a candidate of the {one}-level screen"
    return CONSISTENT if n >= TREND_REPEATS else NOT_SIGNIFICANT


def _order(tests: pd.DataFrame, discovery: Any) -> pd.DataFrame:
    """Best-ranked features first, then the conditions in their order."""
    rank = {f: i for i, f in enumerate(discovery.ranking["feature"])} if discovery else {}
    conditions = {c: i for i, c in enumerate(dict.fromkeys(tests["condition"]))}
    keyed = tests.assign(
        _f=tests["feature"].map(rank).fillna(len(rank)), _c=tests["condition"].map(conditions)
    )
    return keyed.sort_values(["_f", "_c"], kind="stable").drop(columns=["_f", "_c"])


def across_frame(result: Any) -> pd.DataFrame:
    """The table across repeats, one row per feature and condition (the readout's
    conditions with one feature): the direction in the repeats, the change, the mean and
    range of the per-repeat d, p and adjusted p normalised and raw, the verdict; empty
    without a test across repeats. The figure legends read their numbers from it."""
    discovery, contrast = result.discovery, result.results.get("contrast")
    if discovery is not None and len(discovery.tests):
        frame = _tests_frame(discovery)
    elif contrast is not None and len(contrast.repeats):
        frame = _repeats_frame(contrast, result.setup.readout)
    else:
        return pd.DataFrame()
    one = getattr(result.setup, "unit_word", "cell")
    frame["verdict"] = [
        _verdict(r.adjusted_p, r.lower, r.higher, r.repeats, getattr(r, "confirmed", None), one)
        for r in frame.itertuples()
    ]
    return frame


def _tests_frame(discovery: Any) -> pd.DataFrame:
    """The table's rows from the feature discovery: every feature and condition."""
    tests = _order(discovery.tests, discovery)
    return pd.DataFrame(
        {
            "feature": tests["feature"],
            **({"table": tests["table"]} if "table" in tests else {}),  # owner, 2026-10-06
            "condition": tests["condition"],
            "control": tests["control_condition"],
            "repeats": tests["repeats"],
            "lower": tests["lower"],
            "higher": tests["higher"],
            "values": tests["paired_values"],
            "change": tests["change"],
            "p": tests["p_paired"],
            "adjusted_p": tests["q_paired"],
            "p_raw": tests["p_paired_raw"],
            "adjusted_p_raw": tests["q_paired_raw"],
            "screen_wide_adjusted_p": tests.get("q_screen", np.nan),
            "confirmed": tests["confirmed"].fillna(False).astype(bool),
            "d_per_repeat_mean": tests["d_repeat_mean"],
            "d_per_repeat_min": tests["d_repeat_min"],
            "d_per_repeat_max": tests["d_repeat_max"],
            "repeat_observations_min": tests["repeat_observations_min"],
            "repeat_observations_max": tests["repeat_observations_max"],
            "repeat_contrasts": tests["repeat_contrasts"],
            "ci_low": tests["ci_low"],
            "ci_high": tests["ci_high"],
        }
    )


def _repeats_frame(contrast: Any, readout: str) -> pd.DataFrame:
    """The table's rows of the readout alone (one feature), from its contrast."""
    rep = contrast.repeats
    # One mask for the values, the change and p: normalised on the feature's log scale.
    normalised = rep["fold_change"].notna()
    return pd.DataFrame(
        {
            "feature": readout,
            "condition": rep["condition"],
            "control": rep["control_condition"],
            "repeats": rep["repeats"],
            "lower": rep["lower"],
            "higher": rep["higher"],
            "values": np.where(normalised, "normalised", "raw"),
            "change": rep["fold_change"].where(normalised, rep["mean_difference"]),
            "p": rep["p_paired_t_normalised"].where(normalised, rep["p_paired_t"]),
            "adjusted_p": rep["q_paired_t_normalised"].where(normalised, rep["q_paired_t"]),
            "p_raw": rep["p_paired_t"],
            "adjusted_p_raw": rep["q_paired_t"],
            "screen_wide_adjusted_p": rep.get("q_screen", np.nan),
            "d_per_repeat_mean": rep["d_mean"],
            "d_per_repeat_min": rep["d_min"],
            "d_per_repeat_max": rep["d_max"],
            "repeat_contrasts": rep["repeat_contrasts"],
            "ci_low": rep["ci_low"],
            "ci_high": rep["ci_high"],
        }
    )


def d_per_repeat(row: Any) -> str:
    """``-0.57 (-1.13 to -0.09)``: the mean and range of the per-repeat d of a table row
    (the mean alone for one repeat)."""
    mean, low, high = row.d_per_repeat_mean, row.d_per_repeat_min, row.d_per_repeat_max
    if pd.isna(mean):
        return "–"
    low_text, high_text = f"{low:+.2f}", f"{high:+.2f}"
    if low_text == high_text:
        return f"{mean:+.2f}"
    return f"{mean:+.2f} ({low_to_high(low_text, high_text)})"


def _declared(result: Any) -> str:
    """``; declared in the Setup: velocity, the difference``: the scales the Setup declares
    (plan v2, owner, 2026-10-06), which replace the rule; empty without."""
    return declared_scales(getattr(result.setup, "scales", None))


def _limited(row: Any) -> list[str]:
    """Each repeat's change and the 95% t interval of a row on exactly 3 repeats; dashes on
    another row of the same table."""
    if row.repeats != MIN_REPEATS or not isinstance(row.repeat_contrasts, str):
        return ["–", "–"]
    interval = (
        low_to_high(change(row.ci_low, row.values), change(row.ci_high, row.values))
        if pd.notna(row.ci_low)
        else "–"
    )
    return [contrasts_words(row.repeat_contrasts, row.values), interval]


def _across(result: Any, folder: Path) -> tuple[str, int]:
    """The paired t-test across repeats: the table, its raw values folded, and its row count."""
    frame = across_frame(result)
    if frame.empty:
        return "", 0
    many = frame["feature"].nunique() > 1
    # Exactly 3 repeats: each repeat's change and the 95% t interval (plan v2).
    three = bool((frame["repeats"] == MIN_REPEATS).any())
    rows = []
    for r in frame.itertuples():
        row = [r.feature] if many else []
        row += [r.condition, _direction(r.lower, r.higher, r.repeats), change(r.change, r.values)]
        row += _limited(r) if three else []
        row += [d_per_repeat(r), p_value(r.p), p_value(r.adjusted_p), r.verdict]
        rows.append(row)
    columns = (["feature"] if many else []) + ["condition", "in the repeats", "change"]
    columns += ["change in each repeat", "95% t interval"] if three else []
    columns += ["d per repeat: mean (range)", "p", "adjusted p", "verdict"]
    files = [(write_csv(frame, folder, "tests_across_repeats.csv"), "Data (CSV)")]
    raw_rows = [
        ([r.feature] if many else []) + [r.condition, p_value(r.p_raw), p_value(r.adjusted_p_raw)]
        for r in frame.itertuples()
        if r.values == "normalised" and pd.notna(r.p_raw)
    ]
    raw = (
        "<details><summary>On raw values</summary><p>The same test on the raw differences "
        "between the repeat means (not divided by each repeat's control mean): a shift shared "
        "by a whole repeat, such as a staining batch, stays in it. Shown for comparison; the "
        "verdict uses the normalised values.</p>"
        + _table((["feature"] if many else []) + ["condition", "p", "adjusted p"], raw_rows, "rows")
        + "</details>"
        if raw_rows
        else ""
    )
    caption = (
        "<p>Each condition against the control of the same biological repeats, on the "
        "repeat means divided by that control's mean. Change: the condition's mean over the "
        "control's, a fold change (the geometric mean over the repeats; the difference of the "
        "means for a feature with a value that is not positive, the same scale for every "
        f"condition{html.escape(_declared(result))}), so a significant change can be read for "
        "its size. Significant: adjusted p ≤ 0.05.</p>"
        + (f"<p>{html.escape(LIMITED_REPLICATION)}</p>" if three else "")
    )
    return caption + _table(
        columns, rows, "rows"
    ) + f'<p class="tools">{data_buttons(files)}</p>' + raw, len(rows)


__all__ = ["across_frame", "d_per_repeat"]
