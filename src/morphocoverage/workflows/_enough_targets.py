"""The declared rare and fingerprint targets in "Did I image enough?" (private; ``retrospective``,
``_document``).

The report answers the targets declared beside the mean (owner, 2026-10-02), from the
fields acquired, per analysis group (a well):

- **the rare phenotype** (``Setup(rare=..., rare_label=...)``): whether the labelled state was
  seen in at least ``rare.min_fov_occurrences`` of the fields acquired, and the fields that
  would see it there with ``rare.detection_probability`` at the field incidence seen
  (``fov_clustering``'s plan), so no declared field incidence is needed;
- **the fingerprint** (each ``OutcomeCriterion`` of ``Setup(fingerprint=...)`` whose error is
  lower-is-better): the error of the group's fingerprint with the fields acquired, estimated
  from random halves of its fields (two halves of ``m`` fields differ by about
  ``sqrt(2 n / m)`` times the error of all ``n``), against the tolerance; and the fields the
  error would fall within it, as 1/sqrt(fields). A half the error is not defined on is counted
  (``undefined_splits`` of ``splits``); above ``UNDEFINED_TOLERANCE`` (10%) of the halves the
  group is not judged, since the surviving halves need not be like the rest (owner,
  2026-10-05), and below it the error comes from the others and the text says their share.
  When ``optimise``'s held-back fields would
  themselves err by more than the tolerance, the report says so once: its held-back judgement
  cannot confirm the target, and this estimate is the guide.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._seeds import group_rng
from .._style import Status, span, worst, wrap
from .._wording import count, percent, plural, rate
from ..evaluator import FovSubsets, iter_analysis_groups
from ..schema import validate_input_tables

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

SPLITS = 20
MIN_FIELDS = 4
# Largest share of the random halves without a defined error that is left out.
UNDEFINED_TOLERANCE = 0.10
# The share of a group's fields optimise holds back (reservoir_fraction), as the page's
# "What to do next" runs it (``_html_next``) and docs/workflows.md shows it.
HELD_BACK = 0.35


def fingerprint_precision(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> pd.DataFrame | None:
    """Per group and fingerprint criterion: the fields, the estimated error and whether it is
    within the tolerance (see the module); ``None`` without a fingerprint."""
    criteria = [c for c in setup.fingerprint if c.direction == "lower_is_better"]
    if not criteria:
        return None
    data = validate_input_tables(
        setup.compared_values(objects, fovs),
        fovs=fovs,
        feature_cols=setup.judged_columns,
        **setup.common,
    )
    rows = []
    for group, group_data in iter_analysis_groups(data):
        n = len(group_data.fovs)
        half = n // 2
        subsets = FovSubsets(group_data) if n >= MIN_FIELDS else None
        rng = group_rng(setup.random_state, group)
        for criterion in criteria:
            errors = []
            splits = SPLITS if subsets is not None else 0
            for _ in range(splits):
                order = rng.permutation(n)
                value = criterion.metric.evaluate(
                    subsets.take(order[half : 2 * half]), subsets.take(order[:half])
                )
                if value is not None and np.isfinite(value):
                    errors.append(value)
            undefined = splits - len(errors)
            estimate = (
                float(np.sqrt(np.mean(np.square(errors))) * np.sqrt(half / (2 * n)))
                if errors and undefined <= UNDEFINED_TOLERANCE * splits
                else np.nan
            )
            tolerance = float(criterion.threshold)
            rows.append(
                {
                    **group,
                    "criterion": criterion.description or criterion.metric.name,
                    "tolerance": tolerance,
                    "fields": n,
                    "splits": splits,
                    "undefined_splits": undefined,
                    "error": estimate,
                    "met": bool(estimate <= tolerance) if np.isfinite(estimate) else None,
                    "fields_needed": (
                        int(np.ceil(n * (estimate / tolerance) ** 2))
                        if np.isfinite(estimate) and tolerance > 0
                        else None
                    ),
                    "held_back_error": (
                        estimate * np.sqrt(n / max(1, int(np.ceil(n * HELD_BACK))))
                        if np.isfinite(estimate)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else None


def _number(value: float) -> str:
    return f"{value:.2f}"


def _unjudged_reasons(missing: pd.DataFrame, fields: str) -> str:
    """Why the groups of ``missing`` (rows without ``met``) were not judged: one reason, or each
    with its count."""
    few = missing["fields"] < MIN_FIELDS
    reasons = []
    if few.any():
        reasons.append((f"fewer than {MIN_FIELDS} {fields}", int(few.sum())))
    if not few.all():
        shares = missing.loc[~few, "undefined_splits"] / missing.loc[~few, "splits"]
        reason = (
            f"no defined error on {span(shares, 'percent')} of the random halves, more than "
            f"{percent(UNDEFINED_TOLERANCE)}"
        )
        reasons.append((reason, int((~few).sum())))
    if len(reasons) == 1:
        return reasons[0][0]
    return "; ".join(f"{n} with {reason}" for reason, n in reasons)


def not_judged(rows: pd.DataFrame, word: str, fields: str) -> str:
    """That the fingerprint could not be judged in some groups of ``rows`` (one criterion),
    and why, so a tally over the judged groups does not read as every group; empty otherwise."""
    missing = rows[rows["met"].isna()]
    if not len(missing):
        return ""
    return f"Not judged in {count(len(missing), word)}: {_unjudged_reasons(missing, fields)}."


def partly_undefined(judged: pd.DataFrame, word: str) -> str:
    """The share of the random halves without a defined error where some were left out of a
    judged estimate (at most ``UNDEFINED_TOLERANCE``); empty when every half was defined."""
    some = judged[judged["undefined_splits"] > 0]
    if not len(some):
        return ""
    shares = some["undefined_splits"] / some["splits"]
    return (
        f"In {count(len(some), word)}, {span(shares, 'percent')} of the random halves have no "
        "defined error; the error there is from the others."
    )


def held_back_note(judged: pd.DataFrame, word: str, fields: str) -> str:
    """That optimise's held-back fields cannot confirm the fingerprint target, when their own
    error exceeds the tolerance in every group; empty otherwise. Said in the text report and,
    on the page, by the plan of the next experiment, not by "Did I image enough?"."""
    if not len(judged) or not (judged["held_back_error"] > judged["tolerance"]).all():
        return ""
    return (
        f"Optimise judges it on held-back {fields}: holding back {rate(HELD_BACK)} of a "
        f"{word}'s {fields}, their own error (about "
        f"{_number(judged['held_back_error'].median())}) exceeds the tolerance, so it cannot "
        "confirm this target; this estimate is the guide."
    )


def fingerprint_lines(
    table: pd.DataFrame, word: str, fields: str, held_back: bool = True
) -> list[str]:
    """One paragraph per fingerprint criterion (see the module); ``held_back=False`` leaves
    out what optimise's held-back fields can confirm (``held_back_note``)."""
    lines = []
    for criterion, rows in table.groupby("criterion", sort=False):
        judged = rows[rows["met"].notna()]
        if not len(judged):
            text = (
                f"The fingerprint, {criterion}: not judged in any {word}: "
                f"{_unjudged_reasons(rows, fields)}."
            )
            lines.append(wrap(text, 2, hang=4))
            continue
        met = int(judged["met"].sum())
        errors = judged["error"]
        tolerance = _number(judged["tolerance"].iloc[0])
        text = (
            f"The fingerprint, {criterion}: its error with the {fields} acquired, estimated "
            f"from random halves of each {word}'s {fields}, is {_number(errors.min())}-"
            f"{_number(errors.max())} (median {_number(errors.median())}; the tolerance "
            f"{tolerance}): within it in {met} of {count(len(judged), word)}"
        )
        if met == len(judged):
            text += ": enough."
        else:
            needed = judged.loc[~judged["met"].astype(bool), "fields_needed"].astype(float)
            text += (
                f"; about {span(needed)} {fields} per {word} would bring it there "
                f"({span(judged['fields'].astype(float))} acquired)."
            )
        note = held_back_note(judged, word, fields) if held_back else ""
        said = (text, partly_undefined(judged, word), not_judged(rows, word, fields), note)
        lines.append(wrap(" ".join(part for part in said if part), 2, hang=4))
    return lines


def rare_lines(clustering: Any, setup: Setup, word: str, field: str) -> list[str]:
    """The rare phenotype seen in enough fields, per condition (see the module)."""
    fields = plural(field)
    rows = clustering.table
    k = setup.rare.min_fov_occurrences
    seen = rows["fovs_with_state"] >= k
    parts = []
    for condition, block in rows.assign(seen=seen).groupby("condition", sort=False):
        needed = block["fovs_needed"].astype(float).dropna()
        plan = f", about {span(needed)} {fields}" if len(needed) else ", not seen"
        parts.append(f"{condition}: {int(block['seen'].sum())} of {count(len(block), word)}{plan}")
    text = (
        f"The rare phenotype, {setup.rare.phenotype}, to be seen in at least "
        f"{count(k, field)}: seen in at least {k} of the {fields} acquired in "
        f"{int(seen.sum())} of {count(len(rows), word)}. By condition, the {word}s where it "
        f"was, and the {fields} that would see it in {k} with "
        f"{rate(setup.rare.detection_probability)} probability at the share of {fields} "
        f"holding it here: {'; '.join(parts)}."
    )
    return [wrap(text, 2, hang=4)]


def targets_status(
    clustering: Any, precision: pd.DataFrame | None, setup: Setup
) -> tuple[list[Status], list[str]]:
    """The status and headline parts of the targets: every group meets them, or not."""
    statuses, parts = [], []
    if clustering is not None and setup.rare is not None:
        rows = clustering.table
        seen = int((rows["fovs_with_state"] >= setup.rare.min_fov_occurrences).sum())
        parts.append(
            f"{setup.rare_label} seen in at least {setup.rare.min_fov_occurrences} fields in "
            f"{seen} of {len(rows)}"
        )
        statuses.append(Status.OK if seen == len(rows) else Status.CAUTION)
    if precision is not None:
        judged = precision[precision["met"].notna()]
        if not len(judged):
            groups = len(precision) // precision["criterion"].nunique()
            parts.append(f"the fingerprint judged in none of {groups}")
            statuses.append(Status.NOT_ASSESSED)
        else:
            met = int(judged["met"].sum())
            # Groups it could not be judged in are not counted: say the tally is of the judged.
            of = f"{len(judged)} judged" if len(judged) < len(precision) else f"{len(judged)}"
            parts.append(f"the fingerprint within its tolerance in {met} of {of}")
            statuses.append(Status.OK if met == len(judged) else Status.CAUTION)
    return statuses, parts


def with_targets(
    document: MicroscopistReport, results: dict[str, Any], setup: Setup, word: str, field: str
) -> MicroscopistReport:
    """The report with the rare and fingerprint targets answered in question 2."""
    clustering = results.get("rare_clustering") if setup.rare is not None else None
    precision = results.get("fingerprint_precision")
    lines = []
    if clustering is not None and len(clustering.table):
        lines += rare_lines(clustering, setup, word, field)
    if precision is not None:
        lines += fingerprint_lines(precision, word, plural(field))
    if not lines:
        return document
    paragraph = "\n".join(lines)
    short, sections = dict(document.short), dict(document.sections)
    stopping = "\n\n  The stopping rules, as optimise and live run them"
    text = short.get("enough", "")
    at = text.find(stopping)
    short["enough"] = (
        f"{text[:at]}\n{paragraph}{text[at:]}" if at >= 0 else f"{paragraph}\n\n{text}"
    )
    full = sections.get("enough", "")
    first = full.find("\n\n")
    sections["enough"] = (
        f"{full[:first]}\n\n{paragraph}{full[first:]}" if first >= 0 else f"{paragraph}\n\n{full}"
    )
    statuses, parts = targets_status(clustering, precision, setup)
    headline = document.headlines.get("enough", "")
    headlines = {
        **document.headlines,
        "enough": "; ".join(p for p in (headline, *(f"{p} {word}s" for p in parts)) if p),
    }
    current = document.statuses.get("enough")
    status = worst([s for s in (current, *statuses) if s is not None])
    return replace(
        document,
        short=short,
        sections=sections,
        headlines=headlines,
        statuses={**document.statuses, "enough": status},
    )


__all__ = [
    "fingerprint_lines",
    "fingerprint_precision",
    "held_back_note",
    "not_judged",
    "partly_undefined",
    "rare_lines",
    "with_targets",
]
