"""'The tests behind the reading' on the HTML page (private; ``html_report``; owner,
2026-10-02, after an audit found it hard to read): what each test is, first, then one table
per test, built from the results rather than cut from the text report.

- **The tests, in short**: each test in a paragraph (what it compares, on which unit,
  whether it decides or finds candidates), what Cohen's d is in each table, and what p and
  the adjusted p are.
- **Across repeats** (the result): every feature and condition (or the readout's
  conditions, with one feature): the direction in each repeat, the change, the mean and
  range of the per-repeat d, p and the adjusted p (Benjamini-Hochberg, one family,
  ``_families``), and the verdict; the raw values folded. Its rows (``across_frame``) give
  each figure's legend its numbers (``figure_legends``).
- **Cell-level screen** (exploratory): the balance of the repeats, d on the cells, p and
  adjusted p in the version the report trusts, and for the readout how often it gives
  p <= 0.05 with no effect; the readout's other versions folded, never a version whose
  control comes from repeats the condition was not imaged in.
- **Beyond the mean** (a fingerprint declared): each statistic's value, p and adjusted p.
- **Reagents**: its paragraph from the text report.

The plan for the next experiment is not on this card (owner, 2026-10-02); the card's "Show
more" holds the section's assumptions only (``page_assumptions``). Each table can be saved
as CSV (``_html_data``); every number is also in the full text.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import _SENTENCE_END, _assumption_block, assumption, assumption_lines
from .._wording import change, count, listing, p_value, percent
from ..sampling_unit.cells_text import balance_lines
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.distribution import shift_words, spread_words
from ..sampling_unit.legend_text import described_rows, feature_counts, figure_legend
from ..sampling_unit.reading_text import CONFIRMED, CONSISTENT, MIXED
from ..sampling_unit.repeat_plan_text import plan_parts
from ._families import _CELL_VERSIONS
from ._feature_space import with_table
from ._html_data import data_buttons, write_csv
from ._html_tabs import OFF_PAGE, tabs_html, test_blocks

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
_LABELS = {"balanced, normalised": "balanced, normalised", "balanced": "balanced, raw"}
_LABELS |= {"normalised": "as provided, normalised", "raw": "as provided, raw"}


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
    screen named for the observations, ``one`` (``patch-level``)."""
    if n < MIN_REPEATS:
        return f"not tested (needs {MIN_REPEATS} repeats)"
    if not (lower == n or higher == n):
        return MIXED
    significant = pd.notna(q) and q <= ALPHA
    if confirmed if confirmed is not None else significant:
        return CONFIRMED
    if significant:
        return f"adjusted p ≤ 0.05, but not a candidate of the {one}-level screen"
    return CONSISTENT


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
        tests = _order(discovery.tests, discovery)
        frame = pd.DataFrame(
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
                "confirmed": tests["confirmed"].fillna(False).astype(bool),
                "d_per_repeat_mean": tests["d_repeat_mean"],
                "d_per_repeat_min": tests["d_repeat_min"],
                "d_per_repeat_max": tests["d_repeat_max"],
                "repeat_observations_min": tests["repeat_observations_min"],
                "repeat_observations_max": tests["repeat_observations_max"],
            }
        )
    elif contrast is not None and len(contrast.repeats):
        rep = contrast.repeats
        # One mask for the values, the change and p: normalised when every mean is positive.
        normalised = rep["fold_change"].notna()
        frame = pd.DataFrame(
            {
                "feature": result.setup.readout,
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
                "d_per_repeat_mean": rep["d_mean"],
                "d_per_repeat_min": rep["d_min"],
                "d_per_repeat_max": rep["d_max"],
            }
        )
    else:
        return pd.DataFrame()
    one = getattr(result.setup, "unit_word", "cell")
    frame["verdict"] = [
        _verdict(r.adjusted_p, r.lower, r.higher, r.repeats, getattr(r, "confirmed", None), one)
        for r in frame.itertuples()
    ]
    return frame


def d_per_repeat(row: Any) -> str:
    """``-0.57 (-1.13 to -0.09)``: the mean and range of the per-repeat d of a table row
    (the mean alone for one repeat)."""
    mean, low, high = row.d_per_repeat_mean, row.d_per_repeat_min, row.d_per_repeat_max
    if pd.isna(mean):
        return "–"
    return f"{mean:+.2f}" if low == high else f"{mean:+.2f} ({low:+.2f} to {high:+.2f})"


def figure_legends(
    result: Any, objects: pd.DataFrame | None = None, fovs: pd.DataFrame | None = None
) -> dict[str, str]:
    """Each feature's figure legend (``legend_text.figure_legend``): its rows of the table
    across repeats, its observations shown from the tables given (a share: the readout's)."""
    frame = across_frame(result)
    contrast = result.results.get("contrast")
    if contrast is None:
        return {}
    if frame.empty:
        # One repeat and one feature: the readout described, from its per-repeat table.
        frame = described_rows(contrast).assign(feature=result.setup.readout)
    if frame.empty:
        return {}
    shown = {}
    if objects is not None and fovs is not None:
        features = [f for f in dict.fromkeys(frame["feature"]) if f in objects.columns]
        shown = feature_counts(objects, fovs, features, contrast, result.setup.common)
    legends = {}
    for feature, rows in frame.groupby("feature", sort=False):
        text = figure_legend(contrast, rows, result.setup.unit_words, shown.get(feature))
        if text:
            legends[feature] = text
    return legends


def _across(result: Any, folder: Path) -> tuple[str, int]:
    """The paired t-test across repeats: the table, its raw values folded; and how many
    rows it has."""
    frame = across_frame(result)
    if frame.empty:
        return "", 0
    many = frame["feature"].nunique() > 1
    rows = []
    for r in frame.itertuples():
        row = [r.feature] if many else []
        row += [r.condition, _direction(r.lower, r.higher, r.repeats), change(r.change, r.values)]
        row += [d_per_repeat(r), p_value(r.p), p_value(r.adjusted_p), r.verdict]
        rows.append(row)
    columns = (["feature"] if many else []) + ["condition", "in the repeats", "change"]
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
        "means where a mean is not positive), so a significant change can be read for its "
        "size. Significant: adjusted p ≤ 0.05.</p>"
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
                **({"table": tests["table"]} if "table" in tests else {}),
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
        f"d ({result.setup.unit_words})",
        "p",
        "adjusted p",
        "candidate",
    ]
    columns += [] if many else ["p ≤ 0.05 with no effect"]
    versions = list(dict.fromkeys(frame["version"]))
    caption = (
        f"<p>Every {result.setup.unit_word} against the control's, "
        f"{listing(versions, 3)}. A candidate: "
        "adjusted p ≤ 0.05.</p>"
    )
    # The balance of the repeats, once, here (owner, 2026-10-02).
    caption += "".join(f"<p>{html.escape(line)}</p>" for line in balance_lines(contrast))
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
    columns = ["condition", "version", f"d ({unit})", "p", "adjusted p"]
    columns.append("p ≤ 0.05 with no effect")
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
    frame = with_table(tests.drop(columns=["values"]), result.setup)
    frame = frame.rename(columns={"q": "adjusted_p"})
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
    unit, one = result.setup.unit_words, result.setup.unit_word
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
        (f"{one.capitalize()}-level screen: exploratory", screen),
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
    effect = (
        "The difference in means in units of the spread. \u2018d per repeat\u2019 is "
        f"computed in each repeat and averaged; the {one}-level d, \u2018d ({unit})\u2019, on "
        f"every {one} at once, in the version of the screen. They differ when the repeats differ."
    )
    adjusted = (
        "p is each test's own probability of a difference this large with no effect. The "
        f"adjusted p (Benjamini-Hochberg, also called the q-value) allows for testing {family}: "
        "read it against 0.05. A randomization p cannot fall below 1 in the number of shuffles."
    )
    parts += [("Cohen's d", effect), ("p and adjusted p", adjusted)]
    items = "".join(
        f"<dt>{html.escape(name)}</dt><dd>{html.escape(text)}</dd>" for name, text in parts
    )
    return f'<h4>The tests, in short</h4><dl class="tests-explained">{items}</dl>'


def tests_section(result: Any, rest: str, render: Callable[[str], str], folder: Path) -> str:
    """The tests behind the reading (see the module); ``rest`` is the text report's part
    below the readings, for Reagents and the cautions. The plan for the next experiment is
    not on this card (owner, 2026-10-02), nor are the assumptions (under "Show more")."""
    head, blocks, tail = test_blocks(rest)
    cautions = [p for p in tail if p.strip().startswith("⚠️")]
    across, _ = _across(result, folder)
    if not across:
        kept = [p for kind, block in blocks.items() if kind not in OFF_PAGE for p in block]
        return render("\n".join([*head, *kept, *cautions]))
    items = [("Across repeats", "", across)]
    cells = _cells(result, folder)
    if cells:
        items.append((f"{result.setup.unit_word.capitalize()}-level screen", "", cells))
    beyond = _beyond(result, folder)
    if beyond:
        items.append(("Beyond the mean", "", beyond))
    if "Reagents" in blocks:
        items.append(("Reagents", "", render("\n".join(blocks["Reagents"]))))
    return (
        _explainer(result, [name for name, _, _ in items], _family(result))
        + tabs_html("tests", items, "Tests")
        + (render("\n".join(cautions)) if cautions else "")
    )


def _plain(text: str) -> str:
    return " ".join(str(text).split()).rstrip(".").lower()


def page_assumptions(section: str, contrast: Any) -> str:
    """The standing assumptions of the full ``section`` text, each once, as an "Assumptions:"
    list: what the card's "Show more" holds. The plan's are left to the plan (``contrast``'s
    ``plan_parts``), which is not on this card."""
    lines = section.splitlines()
    found: list[str] = []
    for index in range(len(lines)):
        block = _assumption_block(lines, index)
        if block is not None:
            texts = block[1] if block[2] else _SENTENCE_END.split(block[1][0])
            found += [t for t in texts if t not in found]
    plan = [_plain(n) for n in plan_parts(contrast, 0, full=True)[1]] if contrast else []
    kept = [t for t in found if not any(_plain(t).startswith(n) for n in plan)]
    return "\n".join(assumption_lines([assumption(t) for t in kept], 2, detail="full"))


def _family(result: Any) -> str:
    """What the adjusted p allows for: ``10 features and 2 conditions at once``, ``2
    conditions at once in each of 6 sub-experiments``, as the legends count them."""
    discovery, contrast = result.discovery, result.results.get("contrast")
    if discovery is not None and len(discovery.tests):
        tests, features = discovery.tests, count(discovery.tests["feature"].nunique(), "feature")
    else:
        tests, features = getattr(contrast, "repeats", pd.DataFrame()), ""
        if not len(tests):
            return "the conditions at once"
        tested = tests["p_paired_t_normalised"].notna() | tests["p_paired_t"].notna()
        tests = tests[tested] if tested.any() else tests
    sizes = tests.groupby("experiment")["condition"].nunique()
    low, high = int(sizes.min()), int(sizes.max())
    conditions = count(high, "condition") if low == high else f"{low} to {high} conditions"
    what = f"{features} and {conditions}" if features else conditions
    return f"{what} at once" + (
        f" in each of {len(sizes)} sub-experiments" if len(sizes) > 1 else ""
    )


__all__ = ["across_frame", "d_per_repeat", "figure_legends", "page_assumptions", "tests_section"]
