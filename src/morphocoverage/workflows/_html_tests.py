"""'The tests behind the reading' on the HTML page (private; ``html_report``; owner, 2026-10-02,
after an audit found it hard to read): what each test is, first, then one table per test, built
from the results rather than cut from the text report.

- **The tests, in short**: each test in a paragraph (what it compares, on which unit, whether it
  decides or finds candidates), Cohen's d in each table, p and the adjusted p.
- **Across repeats** (the result): every feature and condition (or the readout's conditions,
  with one feature): the direction in each repeat, the change, the mean and range of the
  per-repeat d, p and the adjusted p (Benjamini-Hochberg, one family, ``_families``), the
  verdict, the raw values folded; its rows (``across_frame``) give each figure's legend.
- **Cell-level screen** (exploratory): the balance of the repeats, d on the cells, p and
  adjusted p in the version the report trusts, and for the readout how often it gives p <= 0.05
  with no effect; the readout's other versions folded, never a version whose control comes from
  repeats the condition was not imaged in.
- **Beyond the mean** (a fingerprint declared): each statistic's value, p and adjusted p;
  **Reagents**: its paragraph from the text report.

The plan for the next experiment is not on this card (owner, 2026-10-02); the card's "Show more"
holds the section's assumptions only (``page_assumptions``). Each table can be saved as CSV
(``_html_data``); every number is also in the full text.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import _SENTENCE_END, _assumption_block, assumption, assumption_lines
from .._style import low_to_high
from .._wording import WITHIN_FAMILY, count, listing, p_value, percent
from ..sampling_unit._experiments import family_noun
from ..sampling_unit.cells_text import balance_lines
from ..sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS
from ..sampling_unit.distribution import shift_words, spread_words
from ..sampling_unit.legend_text import described_rows, feature_counts, figure_legend
from ..sampling_unit.repeat_plan_text import plan_parts
from ..statistics import ALPHA
from ._exploratory import exploratory_html
from ._families import _CELL_VERSIONS
from ._feature_space import with_table
from ._html_across import _across, _direction, _order, _table, across_frame, d_per_repeat
from ._html_data import data_buttons, write_csv
from ._html_tabs import OFF_PAGE, tabs_html, test_blocks

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


def _chance(row: Any, columns: tuple[str, str]) -> str:
    low, high = (row.get(c, np.nan) for c in columns)
    if pd.isna(low):
        return "–"
    if np.isclose(low, high):
        return f"about {percent(low)}"
    return low_to_high(percent(low), percent(high))


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
        frame = _discovery_frame(discovery)
    elif by_condition:
        frame = _readout_frame(by_condition, readout)
    else:
        return ""
    chance = _chances(frame, by_condition, readout)
    frame["p_le_0.05_with_no_effect"] = chance
    many = frame["feature"].nunique() > 1
    rows = _cells_rows(frame, chance, many)
    columns = (["feature"] if many else []) + [
        "condition",
        f"{result.setup.unit_words} (vs control)",
        f"d ({result.setup.unit_words})",
        "p",
        "adjusted p",
        "candidate",
    ]
    columns += [] if many else ["p ≤ 0.05 with no effect"]
    caption = _cells_caption(result, frame, chance, many)
    files = [(write_csv(frame, folder, "tests_cell_level.csv"), "Data (CSV)")]
    body = caption + _table(columns, rows, "rows") + f'<p class="tools">{data_buttons(files)}</p>'
    return body + _other_versions(by_condition, frame, readout, result.setup.unit_words)


def _chances(frame: pd.DataFrame, by_condition: dict, readout: str) -> list[str]:
    """Per row, how often the readout's version gives p <= 0.05 with no effect; "–" for
    another feature or without it."""
    chance = []
    for r in frame.itertuples():
        row = by_condition.get(r.condition) if r.feature == readout else None
        columns = next((c for label, _, c in _VERSIONS if label == r.version), None)
        chance.append(_chance(row, columns) if row is not None and columns else "–")
    return chance


def _cells_rows(frame: pd.DataFrame, chance: list[str], many: bool) -> list[list[str]]:
    """The screen's table rows: the feature with ``many``, else how often the test gives
    p <= 0.05 with no effect, last."""
    return [
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


def _discovery_frame(discovery: Any) -> pd.DataFrame:
    """The screen's rows from the feature discovery: every feature and condition."""
    tests = _order(discovery.tests, discovery)
    return pd.DataFrame(
        {
            "feature": tests["feature"],
            **({"table": tests["table"]} if "table" in tests else {}),
            "condition": tests["condition"],
            "cells": tests["observations"],
            "control_cells": tests["control_observations"],
            "version": [
                _version(n, b) for n, b in zip(tests["normalised"], tests["balanced"], strict=True)
            ],
            "d": tests["d"],
            "p": tests["p_cells"],
            "adjusted_p": tests["q_cells"],
            "candidate": tests["passes"],
        }
    )


def _readout_frame(by_condition: dict, readout: str) -> pd.DataFrame:
    """The screen's rows of the readout alone (one feature), in the version the reading
    trusts."""
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
    return pd.DataFrame(rows)


def _cells_caption(result: Any, frame: pd.DataFrame, chance: list[str], many: bool) -> str:
    """What the screen's table holds, the balance of the repeats, and with several features
    how often the test gives p <= 0.05 with no effect, for the readout."""
    readout = result.setup.readout
    versions = list(dict.fromkeys(frame["version"]))
    caption = (
        f"<p>Every {result.setup.unit_word} against the control's, "
        f"{listing(versions, 3)}. A candidate: "
        "adjusted p ≤ 0.05.</p>"
    )
    # The balance of the repeats, once, here (owner, 2026-10-02).
    caption += "".join(
        f"<p>{html.escape(line)}</p>" for line in balance_lines(result.results.get("contrast"))
    )
    known = [(r.condition, c) for r, c in zip(frame.itertuples(), chance, strict=True) if c != "–"]
    if known and many:
        rates = "; ".join(f"{condition}: {c}" for condition, c in known)
        caption += (
            f"<p>How often this test gives p ≤ 0.05 when there is no effect at all, computed "
            f"for the readout ({html.escape(readout)}): {html.escape(rates)}. With independent "
            f"{result.setup.unit_words} it would be 5%; above it, its p is too small.</p>"
        )
    return caption


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
        + exploratory_html(result, "conditions")
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
        f"p ≤ 0.05: from {TREND_REPEATS} repeats, the same direction in every repeat without "
        "it is a trend, not evidence of no effect; with fewer it happens by chance too often "
        "(1 experiment in 4 with 3 repeats)."
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
            "The spread and 10%, 50% and 90% quantiles of each feature the fingerprint reads, per "
            "repeat on as many fields and observations a field on each side, tested like the means."
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
        f"read it against 0.05. {WITHIN_FAMILY} A randomization p cannot fall below 1 in the "
        "number of shuffles."
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
    noun = family_noun(getattr(discovery or contrast, "settings", None), plural=True)
    return f"{what} at once" + (f" in each of {len(sizes)} {noun}" if len(sizes) > 1 else "")


__all__ = ["across_frame", "d_per_repeat", "figure_legends", "page_assumptions", "tests_section"]
