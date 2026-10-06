"""Beyond the mean in "Do the conditions differ?" (private; ``retrospective``, ``_document``).

With a phenotypic fingerprint declared, the distribution of each explainable feature the
fingerprint reads is compared with the control's (``sampling_unit.distribution``): the
spread and the 10%, 50% and 90% quantiles, per biological repeat on field-balanced
subsamples, tested across the repeats (owner, 2026-10-02; 2026-10-06, audit 8). A
fingerprint that reads no explainable feature (one declared on a DINO table) compares none,
and the report says so once (``NOT_RUN``). A
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
from ..sampling_unit.contrast_repeats import MIN_REPEATS, TREND_REPEATS
from ..sampling_unit.distribution import (
    DRAWS,
    DistributionContrast,
    distribution_contrast,
    tests_words,
)
from ..schema import observation_words

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

HEAD = "Beyond the mean"
NOT_RUN = (
    f"{HEAD}: not compared, as the fingerprint reads no explainable feature (the "
    "distributions are compared on the explainable features the fingerprint reads only)."
)
_NAMES = {"spread": "spread", "q10": "10% quantile", "q50": "median", "q90": "90% quantile"}
MAX_ROWS = 24


def fingerprint_features(setup: Setup) -> list[str]:
    """The explainable features the fingerprint reads, in the order of ``setup.columns``."""
    read = set(setup.criterion_columns)
    return [c for c in setup.columns if c in read]


def distribution_by_feature(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> DistributionContrast | None:
    """The comparison of the distributions of the explainable features the fingerprint reads
    (see the module), or ``None`` without a fingerprint, a control, several conditions or
    such a feature."""
    features = fingerprint_features(setup)
    if not setup.fingerprint or not setup.compares or not features:
        return None
    result = distribution_contrast(
        objects,
        fovs,
        features=features,
        random_state=setup.random_state,
        **setup.contrast_arguments,
        **setup.common,
    )
    return result if result.n_tests else None


def _consistent(rows: pd.DataFrame, tested: bool = True) -> pd.DataFrame:
    """Statistics on one side of the control in every repeat, tested and not significant,
    from ``TREND_REPEATS`` repeats, as the per-feature trend (owner, 2026-10-06, audit 8);
    or, with ``tested=False``, not tested: fewer than ``MIN_REPEATS`` usable repeats."""
    alpha = 0.05
    same = (rows["lower"] == rows["repeats"]) | (rows["higher"] == rows["repeats"])
    kept = rows["q"].notna() & (rows["repeats"] >= TREND_REPEATS) if tested else rows["q"].isna()
    return rows[same & (rows["repeats"] > 1) & kept & ~(rows["q"] <= alpha)]


def _named(rows: pd.DataFrame) -> list[str]:
    """``nuclear_ratio (spread, q 0.030)``, the strongest statistic of each feature."""
    best = rows.sort_values("p").drop_duplicates("feature")
    return [f"{r.feature} ({_NAMES[r.statistic]}, q {p_value(r.q)})" for r in best.itertuples()]


def _of_assessed(row: Any) -> str:
    """``in 2 of 2 repeats``, ``in 2 of 2 repeats assessed, 1 repeat not assessed: 3 cells``."""
    if not row.not_assessed:
        return f"in {row.split_repeats} of {row.assessed} repeats"
    return f"in {row.split_repeats} of {row.assessed} repeats assessed, {row.not_assessed}"


def split_words(result: DistributionContrast | None, condition: Any) -> str:
    """``its cells may split into two groups on area (bimodality 0.62, the control 0.41, in 3 of
    3 repeats; see its figure)``, the repeats that cannot be assessed named and left out of the
    count (owner, 2026-10-05); ``a split into two groups is not assessed on area (3 repeats:
    3, 2 and 3 cells)`` when no repeat can be; empty when neither."""
    if result is None:
        return ""
    splits, unassessed = result.splits(), result.not_assessed()
    mine = splits[splits["condition"] == condition]
    blank = unassessed[unassessed["condition"] == condition]
    units = observation_words(result.settings)[1]
    parts = []
    if len(mine):
        named = [
            f"{r.feature} (bimodality coefficient {r.bimodality:.2f}, the control "
            f"{r.control_bimodality:.2f}, {_of_assessed(r)})"
            for r in mine.itertuples()
        ]
        figures = "its figure" if len(mine) == 1 else "their figures"
        parts.append(
            f"its {units} may split into two groups on {listing(named, 4)}, where "
            f"the control's do not (see {figures})"
        )
    if len(blank):
        named = [
            f"{r.feature} ({count(r.repeats, 'repeat')}: {r.reasons})" for r in blank.itertuples()
        ]
        parts.append(f"a split into two groups is not assessed on {listing(named, 4)}")
    return "; ".join(parts)


def _condition_line(condition: Any, rows: pd.DataFrame, result: DistributionContrast) -> str:
    significant = rows[rows["q"] <= 0.05]
    trend = _consistent(rows)
    trend = trend[~trend["feature"].isin(significant["feature"])]
    untested = _consistent(rows, tested=False)
    untested = untested[~untested["feature"].isin([*significant["feature"], *trend["feature"]])]
    parts = []
    split = split_words(result, condition)
    if split:
        parts.append(split)
    if len(significant):
        parts.append("differs beyond the mean: " + listing(_named(significant), 6))
    if len(trend):
        parts.append("the same way in every repeat, not significant: " + listing(_named(trend), 6))
    if len(untested):
        best = untested.drop_duplicates("feature")
        named = [f"{r.feature} ({_NAMES[r.statistic]})" for r in best.itertuples()]
        parts.append(
            f"the same way in every repeat, not tested (fewer than {MIN_REPEATS} repeats): "
            + listing(named, 6)
        )
    return f"- {condition} — " + ("; ".join(parts) if parts else "no difference beyond the mean")


def _cell(row: pd.Series, statistic: str) -> str:
    q = p_value(row["q"]) if pd.notna(row["q"]) else "–"
    if statistic == "spread":
        return f"{np.exp(row['mean']):.2f}x (q {q})"
    return f"{row['mean']:+.2f} (q {q})"


def distribution_text(result: DistributionContrast, unit: str = "cells") -> str:
    """The paragraph of question 9 (see the module)."""
    tests = result.tests
    tested = tests.loc[tests["p"].notna(), "repeats"]
    low, high = int(tested.min()), int(tested.max())
    repeats = str(low) if low == high else f"{low} to {high}"
    features = count(int(tests["feature"].nunique()), "explainable feature")
    draws = result.settings.get("draws", DRAWS)
    intro = wrap(
        f"{HEAD} (the fingerprint is declared): does the distribution of each of the "
        f"{features} the fingerprint reads differ from the control's? Per biological repeat "
        f"against its own control, on field-balanced subsamples of its {unit} (as many fields "
        f"a side, the smaller side's, and as many {unit} a field, the smallest field's, "
        f"averaged over {draws} draws): the "
        "spread (the condition's SD over the control's) and the 10%, 50% and 90% quantiles "
        f"(their shift in control SDs), each tested across the {repeats} repeats (t-test on "
        "the per-repeat values; Benjamini-Hochberg within each sub-experiment: "
        f"{tests_words(result)}):",
        2,
        hang=4,
    )
    lines = [intro]
    for condition, rows in tests.groupby("condition", sort=False):
        lines.append(wrap(_condition_line(condition, rows, result), 4, hang=6))
    # The strongest corrected evidence first (q, then p; untested last), so a cut at
    # MAX_ROWS never drops a smaller q of another sub-experiment for a smaller raw p.
    grid = tests.groupby(["condition", "feature"], sort=False)[["q", "p"]].min()
    order = grid.sort_values(["q", "p"], kind="stable").index[:MAX_ROWS]
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
    untested = _consistent(result.tests, tested=False)
    if len(untested):
        conditions = listing(list(dict.fromkeys(untested["condition"].astype(str))), 4)
        return (
            f"beyond the mean, the same way in every repeat for {conditions}, not tested "
            f"(fewer than {MIN_REPEATS} repeats)"
        )
    return "beyond the mean, no difference in the distributions"


def _insert(text: str, paragraph: str) -> str:
    """``paragraph`` before the section's assumptions (at the end without them)."""
    at = text.rfind("\n  Assumptions")
    if at < 0:
        return f"{text}\n\n{paragraph}" if text else paragraph
    return f"{text[:at]}\n{paragraph}{text[at:]}"


def with_distribution(
    document: MicroscopistReport,
    result: DistributionContrast | None,
    unit: str = "cells",
    setup: Setup | None = None,
) -> MicroscopistReport:
    """The report with the distributions compared in question 9: the paragraph in the short
    and full forms, the headline extended, and a difference answering the question; without
    a result, ``NOT_RUN`` when the fingerprint of ``setup`` reads no explainable feature."""
    short, sections = dict(document.short), dict(document.sections)
    if result is None:
        if setup is None or not (setup.fingerprint and setup.compares and setup.columns):
            return document
        if fingerprint_features(setup):
            return document
        said = wrap(NOT_RUN, 2, hang=4)
        short["conditions"] = _insert(short.get("conditions", ""), said)
        sections["conditions"] = _insert(sections.get("conditions", ""), said)
        return replace(document, short=short, sections=sections)
    paragraph = distribution_text(result, unit)
    short["conditions"] = _insert(short.get("conditions", ""), paragraph)
    sections["conditions"] = _insert(sections.get("conditions", ""), paragraph)
    headline = document.headlines.get("conditions", "")
    status = document.statuses.get("conditions")
    beyond = distribution_headline(result)
    if len(result.differing()) and status is not Status.PROBLEM:
        if status is not Status.OK:
            # Only the distribution differs: said first, so the ✅ reads with its reason.
            headline = f"{beyond}; {headline}"
            beyond = ""
        status = Status.OK
    joined = f"{headline}; {beyond}" if beyond else headline
    headlines = {**document.headlines, "conditions": joined}
    statuses = {**document.statuses, "conditions": status}
    return replace(document, short=short, sections=sections, headlines=headlines, statuses=statuses)


__all__ = [
    "HEAD",
    "NOT_RUN",
    "distribution_by_feature",
    "distribution_headline",
    "fingerprint_features",
    "split_words",
    "with_distribution",
]
