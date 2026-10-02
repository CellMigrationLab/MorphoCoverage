"""Beyond the mean in "Do the conditions differ?" (private; ``retrospective``, ``_document``).

With a phenotypic fingerprint declared, each declared feature's distribution is compared
with the control's (``sampling_unit.distribution``): the spread and the 10%, 50% and 90%
quantiles, per biological repeat, tested across the repeats (owner, 2026-10-02). A
condition whose distribution differs is a difference, so it answers question 9 like a
feature significant across repeats; one that moves the same way in every repeat without
reaching q <= 0.05 is a trend, said as such.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._style import Status, table, wrap
from .._wording import count, listing, p_value
from ..sampling_unit.distribution import (
    DistributionContrast,
    distribution_contrast,
    tests_words,
)

if TYPE_CHECKING:
    from .setup import Setup

HEAD = "Beyond the mean"
_NAMES = {"spread": "spread", "q10": "10% quantile", "q50": "median", "q90": "90% quantile"}
MAX_ROWS = 24


def distribution_by_feature(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> DistributionContrast | None:
    """The comparison of the declared features' distributions, or ``None`` without a
    fingerprint, a control or several conditions."""
    if not setup.fingerprint or setup.control is None:
        return None
    result = distribution_contrast(
        objects, fovs, features=setup.columns, control=setup.control, **setup.common
    )
    return result if result.n_tests else None


def _consistent(rows: pd.DataFrame) -> pd.DataFrame:
    """Statistics on one side of the control in every repeat, not significant."""
    alpha = 0.05
    same = (rows["lower"] == rows["repeats"]) | (rows["higher"] == rows["repeats"])
    return rows[same & (rows["repeats"] > 1) & ~(rows["q"] <= alpha)]


def _named(rows: pd.DataFrame) -> list[str]:
    """``nuclear_ratio (spread, q 0.030)``, the strongest statistic of each feature."""
    best = rows.sort_values("p").drop_duplicates("feature")
    return [f"{r.feature} ({_NAMES[r.statistic]}, q {p_value(r.q)})" for r in best.itertuples()]


def _condition_line(condition: Any, rows: pd.DataFrame) -> str:
    significant = rows[rows["q"] <= 0.05]
    trend = _consistent(rows)
    trend = trend[~trend["feature"].isin(significant["feature"])]
    parts = []
    if len(significant):
        parts.append("differs beyond the mean: " + listing(_named(significant), 6))
    if len(trend):
        parts.append("the same way in every repeat, not significant: " + listing(_named(trend), 6))
    return f"- {condition} — " + ("; ".join(parts) if parts else "no difference beyond the mean")


def _cell(row: pd.Series, statistic: str) -> str:
    q = p_value(row["q"]) if pd.notna(row["q"]) else "–"
    if statistic == "spread":
        return f"{np.exp(row['mean']):.2f}x (q {q})"
    return f"{row['mean']:+.2f} (q {q})"


def distribution_text(result: DistributionContrast, unit: str = "cells") -> str:
    """The paragraph of question 9 (see the module)."""
    tests = result.tests
    repeats = int(tests["repeats"].max())
    intro = wrap(
        f"{HEAD} (the fingerprint is declared): does each feature's distribution differ from "
        f"the control's? Per biological repeat against its own control, on its {unit}: the "
        "spread (the condition's SD over the control's) and the 10%, 50% and 90% quantiles "
        f"(their shift in control SDs), each tested across the {repeats} repeats (t-test on "
        "the per-repeat values; Benjamini-Hochberg within each sub-experiment: "
        f"{tests_words(result)}):",
        2,
        hang=4,
    )
    lines = [intro]
    for condition, rows in tests.groupby("condition", sort=False):
        lines.append(wrap(_condition_line(condition, rows), 4, hang=6))
    grid = tests.pivot_table(
        index=["condition", "feature"], columns="statistic", values="p", aggfunc="first", sort=False
    )
    order = grid.min(axis=1).sort_values().index[:MAX_ROWS]
    keyed = tests.set_index(["condition", "feature", "statistic"])
    rows = [
        [condition, feature, *(_cell(keyed.loc[(condition, feature, s)], s) for s in _NAMES)]
        for condition, feature in order
    ]
    columns = ["condition", "feature", "SD ratio", "10% shift (SD)", "median shift", "90% shift"]
    lines.append(table(rows, columns, indent=4))
    if len(grid) > MAX_ROWS:
        lines.append(f"    and {count(len(grid) - MAX_ROWS, 'more feature')} (report_full.txt)")
    lines.append(
        wrap(
            f"With {repeats} biological repeats only large, consistent changes reach q ≤ 0.05; "
            "the same way in every repeat is a trend, not evidence of no difference.",
            4,
            hang=6,
        )
    )
    return "\n".join(lines)


def distribution_headline(result: DistributionContrast) -> str:
    """``beyond the mean, the distribution differs for drugA (nuclear_ratio)``, or the
    trend."""
    differing = result.differing()
    if len(differing):
        named = [
            f"{c} ({listing(list(rows['feature']), 3)})"
            for c, rows in differing.groupby("condition", sort=False)
        ]
        return "beyond the mean, the distribution differs for " + listing(named, 4)
    trend = _consistent(result.tests)
    if len(trend):
        conditions = listing(list(dict.fromkeys(trend["condition"].astype(str))), 4)
        return f"beyond the mean, the same way in every repeat for {conditions}, not significant"
    return "beyond the mean, no difference in the distributions"


def _insert(text: str, paragraph: str) -> str:
    """``paragraph`` before the section's assumptions (at the end without them)."""
    at = text.rfind("\n  Assumptions")
    if at < 0:
        return f"{text}\n\n{paragraph}" if text else paragraph
    return f"{text[:at]}\n{paragraph}{text[at:]}"


def with_distribution(document: Any, result: DistributionContrast | None, unit: str = "cells"):
    """The report with the distributions compared in question 9: the paragraph in the short
    and full forms, the headline extended, and a difference answering the question."""
    if result is None:
        return document
    paragraph = distribution_text(result, unit)
    short, sections = dict(document.short), dict(document.sections)
    short["conditions"] = _insert(short.get("conditions", ""), paragraph)
    sections["conditions"] = _insert(sections.get("conditions", ""), paragraph)
    headline = document.headlines.get("conditions", "")
    headlines = {**document.headlines, "conditions": f"{headline}; {distribution_headline(result)}"}
    status = document.statuses.get("conditions")
    if len(result.differing()) and status is not Status.PROBLEM:
        status = Status.OK
    statuses = {**document.statuses, "conditions": status}
    return replace(document, short=short, sections=sections, headlines=headlines, statuses=statuses)


__all__ = ["HEAD", "distribution_by_feature", "distribution_headline", "with_distribution"]
