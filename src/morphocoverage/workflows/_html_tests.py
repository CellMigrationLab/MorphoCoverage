"""'The tests behind the reading' on the HTML page (private; ``html_report``; owner,
2026-10-02, after an audit found it hard to read): what each test is, first, then one table
per test, built from the results rather than cut from the text report.

- **The tests, in short**: each test in a paragraph (what it compares, on which unit,
  whether it decides or finds candidates), what Cohen's d is in each table, and what p and
  the adjusted p are.
- **Across repeats** (the result): every feature and condition (or the readout's
  conditions, with one feature): the direction in each repeat, p and the adjusted p
  (Benjamini-Hochberg, one family, ``_families``), and the verdict; the raw values folded.
- **Cell-level screen** (exploratory): d on the cells, p and adjusted p in the version the
  report trusts, and for the readout how often it gives p <= 0.05 with no effect; the
  readout's other versions folded, never a version whose control comes from repeats the
  condition was not imaged in.
- **Beyond the mean** (a fingerprint declared): each statistic's value, p and adjusted p.
- **Reagents** and **Planning**: their paragraphs from the text report.

Each table can be saved as CSV (``_html_data``); every number is also in the full text.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._wording import count, listing, p_value, percent
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.distribution import shift_words, spread_words
from ._families import _CELL_VERSIONS
from ._html_data import data_buttons, write_csv
from ._html_tabs import tabs_html, test_blocks

ALPHA = 0.05
# Above this many rows a table shows its first ones and a button for the rest.
FIRST_ROWS = 20
# The readout's cell-level versions: (label, column suffix, chance columns), the suffixes
# those of the shared family (``_families``).
_VERSIONS = tuple(
    (
        f"{'balanced' if balanced else 'as provided'}, {'normalised' if normalised else 'raw'}",
        _CELL_VERSIONS[(normalised, balanced)],
        ("chance_balanced_low", "chance_balanced_high")
        if balanced
        else ("chance_low", "chance_high"),
    )
    for balanced in (True, False)
    for normalised in (True, False)
)
# The reading's names of the screen's versions (reading_text._cell_version), as here.
_LABELS = {
    "balanced, normalised": "balanced, normalised",
    "balanced": "balanced, raw",
    "normalised": "as provided, normalised",
    "raw": "as provided, raw",
}


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


def _verdict(q: float, lower: int, higher: int, n: int) -> str:
    if n < MIN_REPEATS:
        return f"not tested (needs {MIN_REPEATS} repeats)"
    if pd.notna(q) and q <= ALPHA:
        return "significant"
    if n and (lower == n or higher == n):
        return "same direction, not significant"
    return "no consistent direction"


def _order(tests: pd.DataFrame, discovery: Any) -> pd.DataFrame:
    """Best-ranked features first, then the conditions in their order."""
    rank = {f: i for i, f in enumerate(discovery.ranking["feature"])} if discovery else {}
    conditions = {c: i for i, c in enumerate(dict.fromkeys(tests["condition"]))}
    keyed = tests.assign(
        _f=tests["feature"].map(rank).fillna(len(rank)), _c=tests["condition"].map(conditions)
    )
    return keyed.sort_values(["_f", "_c"], kind="stable").drop(columns=["_f", "_c"])


def _across(result: Any, folder: Path) -> tuple[str, int]:
    """The paired t-test across repeats: the table, its raw values folded; and how many
    rows it has."""
    discovery, contrast = result.discovery, result.results.get("contrast")
    if discovery is not None and len(discovery.tests):
        tests = _order(discovery.tests, discovery)
        frame = pd.DataFrame(
            {
                "feature": tests["feature"],
                "condition": tests["condition"],
                "control": tests["control_condition"],
                "repeats": tests["repeats"],
                "lower": tests["lower"],
                "higher": tests["higher"],
                "values": tests["paired_values"],
                "p": tests["p_paired"],
                "adjusted_p": tests["q_paired"],
                "p_raw": tests["p_paired_raw"],
                "adjusted_p_raw": tests["q_paired_raw"],
            }
        )
    elif contrast is not None and len(contrast.repeats):
        rep = contrast.repeats
        normalised = rep["p_paired_t_normalised"].notna()
        frame = pd.DataFrame(
            {
                "feature": result.setup.readout,
                "condition": rep["condition"],
                "control": rep["control_condition"],
                "repeats": rep["repeats"],
                "lower": rep["lower"],
                "higher": rep["higher"],
                "values": np.where(normalised, "normalised", "raw"),
                "p": rep["p_paired_t_normalised"].where(normalised, rep["p_paired_t"]),
                "adjusted_p": rep["q_paired_t_normalised"].where(normalised, rep["q_paired_t"]),
                "p_raw": rep["p_paired_t"],
                "adjusted_p_raw": rep["q_paired_t"],
                "d_per_repeat_mean": rep["d_mean"],
                "d_per_repeat_min": rep["d_min"],
                "d_per_repeat_max": rep["d_max"],
            }
        )
    else:
        return "", 0
    frame["verdict"] = [
        _verdict(r.adjusted_p, r.lower, r.higher, r.repeats) for r in frame.itertuples()
    ]
    many = frame["feature"].nunique() > 1
    single_d = "d_per_repeat_mean" in frame
    rows = []
    for r in frame.itertuples():
        row = [r.feature] if many else []
        row += [r.condition, _direction(r.lower, r.higher, r.repeats)]
        if single_d:
            low, high = r.d_per_repeat_min, r.d_per_repeat_max
            mean = r.d_per_repeat_mean
            row.append(f"{mean:+.2f} ({low:+.2f} to {high:+.2f})" if pd.notna(mean) else "–")
        row += [p_value(r.p), p_value(r.adjusted_p), r.verdict]
        rows.append(row)
    columns = (["feature"] if many else []) + ["condition", "in the repeats"]
    columns += ["d per repeat: mean (range)"] if single_d else []
    columns += ["p", "adjusted p", "verdict"]
    files = [(write_csv(frame, folder, "tests_across_repeats.csv"), "Data (CSV)")]
    raw_rows = [
        ([r.feature] if many else []) + [r.condition, p_value(r.p_raw), p_value(r.adjusted_p_raw)]
        for r in frame.itertuples()
        if r.values == "normalised"
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
        "repeat means divided by that control's mean. Significant: adjusted p ≤ 0.05.</p>"
    )
    return caption + _table(
        columns, rows, "rows"
    ) + f'<p class="tools">{data_buttons(files)}</p>' + raw, len(rows)


def _chance(row: Any, columns: tuple[str, str]) -> str:
    low, high = (row.get(c, np.nan) for c in columns)
    if pd.isna(low):
        return "–"
    return f"about {percent(low)}" if np.isclose(low, high) else f"{percent(low)}–{percent(high)}"


def _version(normalised: Any, balanced: Any) -> str:
    return f"{'balanced' if balanced else 'as provided'}, {'normalised' if normalised else 'raw'}"


def _cells(result: Any, folder: Path) -> str:
    """The cell-level screen: the version trusted, and the readout's others folded."""
    discovery, contrast = result.discovery, result.results.get("contrast")
    cells = getattr(contrast, "cells", None)
    by_condition = (
        {r["condition"]: r for _, r in cells.iterrows()} if cells is not None and len(cells) else {}
    )
    readout = result.setup.readout
    if discovery is not None and len(discovery.tests):
        tests = _order(discovery.tests, discovery)
        frame = pd.DataFrame(
            {
                "feature": tests["feature"],
                "condition": tests["condition"],
                "cells": tests["observations"],
                "control_cells": tests["control_observations"],
                "version": [
                    _version(n, b)
                    for n, b in zip(tests["normalised"], tests["balanced"], strict=True)
                ],
                "d": tests["d"],
                "p": tests["p_cells"],
                "adjusted_p": tests["q_cells"],
                "candidate": tests["passes"],
            }
        )
    elif by_condition:
        from ..sampling_unit.reading_text import _cell_version

        rows = []
        for condition, row in by_condition.items():
            label, d, q, _low, _high, p = _cell_version(row)
            balanced = label.startswith("balanced")
            # The balanced test draws as many from each repeat: its own counts.
            repeats = row.get("balanced_repeats", np.nan)
            cells = row["balanced_cells"] * repeats if balanced else row["observations"]
            controls = (
                row["balanced_control_cells"] * repeats if balanced else row["control_observations"]
            )
            rows.append(
                {
                    "feature": readout,
                    "condition": condition,
                    "cells": cells,
                    "control_cells": controls,
                    "version": _LABELS.get(label, label),
                    "d": d,
                    "p": p,
                    "adjusted_p": q,
                    "candidate": bool(pd.notna(q) and q <= ALPHA),
                }
            )
        frame = pd.DataFrame(rows)
    else:
        return ""
    chance = []
    for r in frame.itertuples():
        row = by_condition.get(r.condition) if r.feature == readout else None
        columns = next((c for label, _, c in _VERSIONS if label == r.version), None)
        chance.append(_chance(row, columns) if row is not None and columns else "–")
    frame["p_le_0.05_with_no_effect"] = chance
    many = frame["feature"].nunique() > 1
    rows = [
        ([r.feature] if many else [])
        + [
            r.condition,
            f"{int(r.cells)} vs {int(r.control_cells)}" if pd.notna(r.cells) else "–",
            f"{r.d:+.2f}" if pd.notna(r.d) else "–",
            p_value(r.p),
            p_value(r.adjusted_p),
            "yes" if r.candidate else "no",
        ]
        + ([] if many else [c])
        for r, c in zip(frame.itertuples(), chance, strict=True)
    ]
    columns = (["feature"] if many else []) + [
        "condition",
        f"{result.setup.unit_words} (vs control)",
        "d (cells)",
        "p",
        "adjusted p",
        "candidate",
    ]
    columns += [] if many else ["p ≤ 0.05 with no effect"]
    versions = list(dict.fromkeys(frame["version"]))
    caption = (
        f"<p>Every {result.setup.observation_unit or 'observation'} against the control's, "
        f"{listing(versions, 3)}. A candidate: "
        "adjusted p ≤ 0.05.</p>"
    )
    known = [(r.condition, c) for r, c in zip(frame.itertuples(), chance, strict=True) if c != "–"]
    if known and many:
        rates = "; ".join(f"{condition}: {c}" for condition, c in known)
        caption += (
            f"<p>How often this test gives p ≤ 0.05 when there is no effect at all, computed "
            f"for the readout ({html.escape(readout)}): {html.escape(rates)}. With independent "
            f"{result.setup.unit_words} it would be 5%; above it, its p is too small.</p>"
        )
    files = [(write_csv(frame, folder, "tests_cell_level.csv"), "Data (CSV)")]
    body = caption + _table(columns, rows, "rows") + f'<p class="tools">{data_buttons(files)}</p>'
    return body + _other_versions(by_condition, frame, readout, result.setup.unit_words)


def _other_versions(
    by_condition: dict, frame: pd.DataFrame, readout: str, unit: str = "observations"
) -> str:
    """The readout's other versions of the screen, its own conditions as the family."""
    shown = set(
        zip(
            frame.loc[frame["feature"] == readout, "condition"],
            frame.loc[frame["feature"] == readout, "version"],
            strict=True,
        )
    )
    rows, skipped = [], []
    for condition, row in by_condition.items():
        for label, suffix, chance in _VERSIONS:
            if (condition, label) in shown or pd.isna(row.get(f"p_{suffix}")):
                continue
            if label.startswith("as provided") and row.get("other_control_repeats", 0) > 0:
                skipped.append(str(condition))
                continue
            rows.append(
                [
                    condition,
                    label,
                    f"{row[f'd_{suffix}']:+.2f}",
                    p_value(row[f"p_{suffix}"]),
                    p_value(row[f"q_{suffix}"]),
                    _chance(row, chance),
                ]
            )
    if not rows and not skipped:
        return ""
    note = (
        f"<p>Not shown as provided for {listing(sorted(set(skipped)), 4)}: its control "
        "would include repeats the condition was not imaged in.</p>"
        if skipped
        else ""
    )
    columns = ["condition", "version", "d (cells)", "p", "adjusted p", "p ≤ 0.05 with no effect"]
    return (
        f"<details><summary>The other versions, for the readout ({html.escape(readout)})</summary>"
        f"<p>As provided: the {unit} pooled as imaged, so the repeat with most weighs most. "
        "Raw: not divided by each repeat's control mean. Adjusted over the readout's conditions "
        "only.</p>" + note + (_table(columns, rows, "rows") if rows else "") + "</details>"
    )


def _beyond(result: Any, folder: Path) -> str:
    distribution = result.results.get("distribution")
    tests = getattr(distribution, "tests", None)
    if tests is None or not len(tests):
        return ""
    names = {
        "spread": "spread (SD ratio)",
        "q10": "10% quantile",
        "q50": "median",
        "q90": "90% quantile",
    }
    rows = []
    for _, r in tests.iterrows():
        value = spread_words(r) if r["statistic"] == "spread" else shift_words(r).split(" ", 2)[-1]
        rows.append(
            [
                r["feature"],
                r["condition"],
                names.get(r["statistic"], r["statistic"]),
                value,
                _direction(int(r["lower"]), int(r["higher"]), int(r["repeats"])),
                p_value(r["p"]),
                p_value(r["q"]),
            ]
        )
    frame = tests.drop(columns=["values"]).rename(columns={"q": "adjusted_p"})
    files = [(write_csv(frame, folder, "tests_beyond_the_mean.csv"), "Data (CSV)")]
    caption = (
        "<p>Per biological repeat against its own control: the spread (the condition's SD "
        "over the control's) and the 10%, 50% and 90% quantiles (their shift in control SDs), "
        "each tested across the repeats like a mean. Differs: adjusted p ≤ 0.05.</p>"
    )
    columns = [
        "feature",
        "condition",
        "statistic",
        "value: mean (range)",
        "in the repeats",
        "p",
        "adjusted p",
    ]
    return caption + _table(columns, rows, "rows") + f'<p class="tools">{data_buttons(files)}</p>'


def _explainer(result: Any, kinds: Sequence[str], family: str) -> str:
    """The tests, in short: one paragraph per test on the page, then p and adjusted p."""
    unit = result.setup.unit_words
    one = result.setup.observation_unit or "observation"
    paired = (
        f"In each biological repeat, the mean of the condition's {unit} is divided by the "
        "mean of the control of the same repeat; the repeats' ratios are tested against no "
        f"change. Each repeat counts once, so the answer holds across repeats. It needs "
        f"{MIN_REPEATS} repeats, and with 3 only large, consistent effects reach adjusted "
        "p ≤ 0.05: the same direction in every repeat without it is a trend, not evidence "
        "of no effect."
    )
    screen = (
        f"Cohen's d of every {one} against the control's, and a randomization test that "
        f"shuffles the condition labels over them (as Plot&Stats). It treats every {one} as "
        "independent, so when the repeats or wells differ its p is too small: it finds "
        "candidates, not effects that repeat. Balanced: as many from each repeat; "
        "normalised: divided by each repeat's control mean."
    )
    parts = [
        ("Paired t-test on the biological repeat means: decides", paired),
        ("Cell-level screen: exploratory", screen),
    ]
    if "Beyond the mean" in kinds:
        beyond = (
            "The spread and the 10%, 50% and 90% quantiles of each condition against its "
            "control, per repeat, tested across the repeats like the means."
        )
        parts.append(("Beyond the mean", beyond))
    if "Reagents" in kinds:
        reagents = (
            "Whether independent reagents for one target go the same way in every repeat; "
            "described, not tested (they share the control)."
        )
        parts.append(("Reagents of one target", reagents))
    if "Planning" in kinds:
        planning = (
            "For a new experiment: the biological repeats the paired t-test would need for "
            "80% power if the difference and the variation seen here recur, and the "
            f"{unit} a side that would show the difference within one repeat."
        )
        parts.append(("Planning", planning))
    effect = (
        "The difference in means in units of the spread. \u2018d per repeat\u2019 is "
        "computed in each repeat and averaged; the cell-level d, \u2018d (cells)\u2019, on every "
        f"{one} at once, in the version of the screen. They differ when the repeats differ."
    )
    adjusted = (
        "p is each test's own probability of a difference this large with no effect. The "
        "adjusted p (Benjamini-Hochberg, also called the q-value) allows for testing "
        f"{family} at once: read it against 0.05. A randomization p cannot fall below 1 in "
        "the number of shuffles."
    )
    parts += [("Cohen's d", effect), ("p and adjusted p", adjusted)]
    items = "".join(
        f"<dt>{html.escape(name)}</dt><dd>{html.escape(text)}</dd>" for name, text in parts
    )
    return f'<h4>The tests, in short</h4><dl class="tests-explained">{items}</dl>'


def tests_section(result: Any, rest: str, render: Callable[[str], str], folder: Path) -> str:
    """The tests behind the reading (see the module); ``rest`` is the text report's part
    below the readings, for Reagents and Planning and the cautions."""
    _, blocks, tail = test_blocks(rest)
    across, n_rows = _across(result, folder)
    if not across:
        return render(rest)
    items = [("Across repeats", "", across)]
    cells = _cells(result, folder)
    if cells:
        items.append(("Cell-level screen", "", cells))
    beyond = _beyond(result, folder)
    if beyond:
        items.append(("Beyond the mean", "", beyond))
    for name in ("Reagents", "Planning"):
        if name in blocks:
            items.append((name, "", render("\n".join(blocks[name]))))
    discovery = result.discovery
    if discovery is not None and len(discovery.tests):
        features = discovery.tests["feature"].nunique()
        conditions = discovery.tests["condition"].nunique()
        family = f"{count(features, 'feature')} and {count(conditions, 'condition')}"
    else:
        family = count(n_rows, "condition")
    family += " (within each sub-experiment)" if _sub_experiments(result) > 1 else ""
    cautions = [p for p in tail if p.strip().startswith("⚠️")]
    return (
        _explainer(result, [name for name, _, _ in items], family)
        + tabs_html("tests", items, "Tests")
        + (render("\n".join(cautions)) if cautions else "")
    )


def _sub_experiments(result: Any) -> int:
    discovery = result.discovery
    tests = discovery.tests if discovery is not None else None
    if tests is not None and len(tests) and "experiment" in tests:
        return int(tests["experiment"].nunique())
    contrast = result.results.get("contrast")
    repeats = getattr(contrast, "repeats", None)
    return int(repeats["experiment"].nunique()) if repeats is not None and len(repeats) else 1


__all__ = ["tests_section"]
