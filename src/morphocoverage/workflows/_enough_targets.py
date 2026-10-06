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

from .._seeds import group_key, group_rng
from .._style import Status, span, worst, wrap
from .._wording import count, listing, percent, plural, rate
from ..evaluator import FovSubsets, iter_analysis_groups
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..schema import validate_input_tables
from ._fingerprint_space import MIN_FIELDS, fingerprint_space, label, table_of
from ._fingerprint_space import criterion_columns as _columns

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

SPLITS = 20
# Largest share of the random halves without a defined error that is left out.
UNDEFINED_TOLERANCE = 0.10
# The share of a group's fields optimise holds back (reservoir_fraction), as the page's
# "What to do next" runs it (``_html_next``) and docs/report/questions.md shows it.
HELD_BACK = 0.35


def _not_assessable(data: Any, criterion: str, reason: str) -> list[dict[str, Any]]:
    """One row per group for a criterion that is not assessable, with the reason."""
    return [
        {
            **group,
            "criterion": criterion,
            "fields": len(group_data.fovs),
            "splits": 0,
            "undefined_splits": 0,
            "error": np.nan,
            "met": None,
            "not_assessable": reason,
        }
        for group, group_data in iter_analysis_groups(data)
    ]


def fingerprint_precision(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> pd.DataFrame | None:
    """Per group and fingerprint criterion: the fields, the estimated error and whether it is
    within the tolerance (see the module); ``None`` without a fingerprint. With other feature
    tables, each criterion's ``table``; observations missing a value a criterion reads are
    left out of it, counted (``left_out``; owner, 2026-10-06). A wide table's criterion is
    read on its first principal components (``components``, said in ``reading``), or is not
    assessable (``not_assessable``, the reason; ``_fingerprint_space``)."""
    if not any(c.direction == "lower_is_better" for c in setup.fingerprint):
        return None
    values = setup.compared_values(objects, fovs)
    space = fingerprint_space(values, fovs, setup)
    data = validate_input_tables(
        values, fovs=fovs, feature_cols=setup.judged_columns, **setup.common
    )
    pairs = [(o, r) for o, r in space.pairs if o.direction == "lower_is_better"]
    criteria = [r for _, r in pairs if r is not None]
    names = [label(o) for o, r in pairs if r is not None]
    # A criterion whose columns miss values reads the observations holding all of them.
    complete: dict[int, dict[str, Any]] = {}
    # Normalised with several controls, a condition without its own control in a repeat has
    # no value there: left out for that, never as a missing value (audit 8, seed 4).
    uncontrolled: dict[int, dict[Any, int]] = {}
    for i, criterion in enumerate(criteria):
        held = ~values[_columns(criterion)].isna().any(axis=1)
        lacking = held.index[(~held & objects[_columns(criterion)].notna().all(axis=1)).to_numpy()]
        uncontrolled[i] = {}
        if len(lacking):
            lone = validate_input_tables(
                values.loc[lacking], fovs=fovs, feature_cols=setup.judged_columns, **setup.common
            )
            uncontrolled[i] = {group_key(g): len(d.objects) for g, d in iter_analysis_groups(lone)}
        held = ~data.objects[_columns(criterion)].isna().any(axis=1)
        if not held.all():
            kept = validate_input_tables(
                values.loc[held[held].index],
                fovs=fovs,
                feature_cols=setup.judged_columns,
                **setup.common,
            )
            complete[i] = {group_key(g): d for g, d in iter_analysis_groups(kept)}
    rows = []
    for original, read in pairs:
        reading = space.readings.get(label(original))
        if read is None and reading is not None:
            rows += [
                {**row, "table": table_of(_columns(original), setup)}
                for row in _not_assessable(data, label(original), str(reading.reason))
            ]
    for group, group_data in iter_analysis_groups(data):
        n = len(group_data.fovs)
        half = n // 2
        subsets = FovSubsets(group_data) if n >= MIN_FIELDS else None
        rng = group_rng(setup.random_state, group)
        for i, criterion in enumerate(criteria):
            errors = []
            here, left_out = subsets, 0
            if i in complete:
                own = complete[i].get(group_key(group))
                left_out = len(group_data.objects) - (len(own.objects) if own is not None else 0)
                here = FovSubsets(own) if own is not None and subsets is not None else None
            no_control = uncontrolled[i].get(group_key(group), 0)
            left_out -= no_control
            reading = space.readings.get(names[i])
            # Too few observations for the common k: not assessable here (audit 8).
            small = reading is not None and group_key(group) in {k for k, *_ in reading.small}
            here = None if small else here
            splits = SPLITS if here is not None else 0
            for _ in range(splits):
                order = rng.permutation(n)
                value = criterion.metric.evaluate(
                    here.take(order[half : 2 * half]), here.take(order[:half])
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
                    "criterion": names[i],
                    "table": table_of(_columns(criterion), setup),
                    "left_out": left_out,
                    "no_own_control": no_control,
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
                    "components": reading.components if reading is not None else None,
                    "small": small,
                    "reading": (
                        reading.words(setup.unit_word, setup.unit_words) if reading else ""
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
    small = missing.get("small", pd.Series(False, index=missing.index)).eq(True) & ~few
    alone = missing.get("no_own_control", pd.Series(0, index=missing.index)) > 0
    alone &= ~few & ~small & missing["splits"].eq(0)
    rest = ~few & ~small & ~alone
    reasons = []
    if few.any():
        reasons.append((f"fewer than {MIN_FIELDS} {fields}", int(few.sum())))
    if alone.any():
        reasons.append(("no own control in its repeat", int(alone.sum())))
    if small.any():  # the common k needs five times its observations (audit 8)
        reasons.append(("too few observations for the common k", int(small.sum())))
    if rest.any():
        shares = missing.loc[rest, "undefined_splits"] / missing.loc[rest, "splits"]
        reason = (
            f"no defined error on {span(shares, 'percent')} of the random halves, more than "
            f"{percent(UNDEFINED_TOLERANCE)}"
        )
        reasons.append((reason, int(rest.sum())))
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


def fingerprint_title(criterion: str, rows: pd.DataFrame) -> str:
    """``The fingerprint, <criterion>``, or with other feature tables ``The fingerprint of
    DINOv2 (<criterion>)``: each table its own verdict (owner, 2026-10-06); a wide table's
    adds how it is read (``read on its first 4 principal components (k at most ...)``)."""
    table = rows["table"].iloc[0] if "table" in rows else None
    title = (
        f"The fingerprint of {table} ({criterion})"
        if isinstance(table, str)
        else f"The fingerprint, {criterion}"
    )
    reading = rows["reading"].iloc[0] if "reading" in rows else None
    return f"{title}, {reading}" if isinstance(reading, str) and reading else title


def assessable_reason(rows: pd.DataFrame) -> str | None:
    """Why a criterion's fingerprint is not assessable (rows of one criterion), or ``None``."""
    if "not_assessable" not in rows:
        return None
    reasons = rows["not_assessable"].dropna()
    return str(reasons.iloc[0]) if len(reasons) else None


def _left_out(rows: pd.DataFrame, unit: str) -> str:
    """``125 cells with a missing value of its columns left out.``, and apart ``36 cells of
    b1 in R4 left out: no own control.`` (audit 8, seed 4), or empty."""
    left = int(rows["left_out"].sum()) if "left_out" in rows else 0
    said = f"{count(left, unit)} with a missing value of its columns left out." if left else ""
    lacking = rows[rows["no_own_control"] > 0] if "no_own_control" in rows else rows.iloc[:0]
    if not len(lacking):
        return said
    where = [
        f"{r.get(CONDITION_COLUMN, '')} in {r.get(REPEAT_COLUMN, '')}".strip()
        for _, r in lacking.iterrows()
    ]
    number = count(int(lacking["no_own_control"].sum()), unit)
    return f"{said} {number} of {listing(where, 4)} left out: no own control.".lstrip()


def fingerprint_lines(
    table: pd.DataFrame, word: str, fields: str, held_back: bool = True, unit: str = "observation"
) -> list[str]:
    """One paragraph per fingerprint criterion (see the module); ``held_back=False`` leaves
    out what optimise's held-back fields can confirm (``held_back_note``); ``unit`` names an
    observation."""
    lines = []
    for criterion, rows in table.groupby("criterion", sort=False):
        judged = rows[rows["met"].notna()]
        title = fingerprint_title(criterion, rows)
        reason = assessable_reason(rows)
        if reason is not None:  # a wide table and small groups (owner, 2026-10-06)
            lines.append(wrap(f"{title}: not assessable: {reason}.", 2, hang=4))
            continue
        left = _left_out(rows, unit)
        if not len(judged):
            text = f"{title}: not judged in any {word}: {_unjudged_reasons(rows, fields)}."
            lines.append(wrap(f"{text} {left}".rstrip(), 2, hang=4))
            continue
        met = int(judged["met"].sum())
        errors = judged["error"]
        tolerance = _number(judged["tolerance"].iloc[0])
        text = (
            f"{title}: its error with the {fields} acquired, estimated "
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
        said = (text, partly_undefined(judged, word), not_judged(rows, word, fields), note, left)
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
        # With other feature tables, a verdict per table (owner, 2026-10-06).
        tables = precision["table"] if "table" in precision else pd.Series(None, precision.index)
        by_table = tables.notna().all() and tables.nunique() > 1
        for table, rows in precision.groupby(
            tables if by_table else [0] * len(precision), sort=False
        ):
            only = tables.dropna().unique()
            named = table if by_table else (only[0] if len(only) == 1 else None)
            name = f"the fingerprint of {named}" if isinstance(named, str) else "the fingerprint"
            judged = rows[rows["met"].notna()]
            if assessable_reason(rows) is not None:
                parts.append(f"{name} not assessable")
                statuses.append(Status.NOT_ASSESSED)
                continue
            if not len(judged):
                groups = len(rows) // rows["criterion"].nunique()
                parts.append(f"{name} judged in none of {groups}")
                statuses.append(Status.NOT_ASSESSED)
                continue
            met = int(judged["met"].sum())
            # Groups it could not be judged in are not counted: say the tally is of the judged.
            of = f"{len(judged)} judged" if len(judged) < len(rows) else f"{len(judged)}"
            parts.append(f"{name} within its tolerance in {met} of {of}")
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
        lines += fingerprint_lines(precision, word, plural(field), unit=setup.unit_word)
    if not lines:
        return document
    paragraph = "\n".join(lines)
    short, sections = dict(document.short), dict(document.sections)
    inputs = dict(document.inputs)
    if inputs.get("enough") == "not assessed":
        # The targets alone answer it (an embedding-only report, owner 2026-10-06).
        sections["enough"] = short["enough"] = paragraph
        inputs["enough"] = "given"
        statuses = {k: v for k, v in document.statuses.items() if k != "enough"}
        document = replace(
            document, headlines={**document.headlines, "enough": ""}, statuses=statuses
        )
    else:
        stopping = "\n\n  The stopping rules, as optimise and live run them"
        text = short.get("enough", "")
        at = text.find(stopping)
        short["enough"] = (
            f"{text[:at]}\n{paragraph}{text[at:]}" if at >= 0 else f"{paragraph}\n\n{text}"
        )
        full = sections.get("enough", "")
        first = full.find("\n\n")
        sections["enough"] = (
            f"{full[:first]}\n\n{paragraph}{full[first:]}"
            if first >= 0
            else f"{paragraph}\n\n{full}"
        )
    statuses, parts = targets_status(clustering, precision, setup)
    headline = document.headlines.get("enough", "")
    headlines = {
        **document.headlines,
        "enough": "; ".join(
            p
            for p in (headline, *(p if p.endswith("assessable") else f"{p} {word}s" for p in parts))
            if p
        ),
    }
    current = document.statuses.get("enough")
    status = worst([s for s in (current, *statuses) if s is not None])
    return replace(
        document,
        short=short,
        sections=sections,
        inputs=inputs,
        headlines=headlines,
        statuses={**document.statuses, "enough": status},
    )


__all__ = [
    "assessable_reason",
    "fingerprint_lines",
    "fingerprint_precision",
    "fingerprint_title",
    "held_back_note",
    "not_judged",
    "partly_undefined",
    "rare_lines",
    "with_targets",
]
