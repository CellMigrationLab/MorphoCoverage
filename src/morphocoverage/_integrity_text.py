"""The data-integrity checks in words (private): ``IntegrityResult.summary``, short and full.

- short: one marked line (✅ nothing found, ❌ findings, ➖ no check applied),
  what was checked, the findings grouped by check and pair of groups in one
  table, what to do, then each caveat once (what could not be compared, what
  the checks cannot see);
- full: the same header, every finding with its detail (grouped the same
  way), then every caveat once.

Fields and groups print as ``MOCK · 1A · B6 · 002006-1.tif`` (``_style.group_label``); the
columns they stand for are named once. A finding is structure to inspect, not
a judgment, and the text says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from . import _integrity_match as match
from ._caveats import assumption, caveat_lines
from ._style import NOT_ASSESSED, Status, check_detail, group_columns, headline, table, wrap
from ._wording import count, listing

if TYPE_CHECKING:
    from pandas.core.groupby import DataFrameGroupBy

    from .integrity import IntegrityResult

# Lines (full) or rows (short) of findings printed before "and N more".
SUMMARY_LINES = 12
PRECISION = f"to {match.DIGITS} significant digits"
TITLE = "Data integrity"
_WHAT_TO_DO = (
    "Check where these fields came from (a file read or exported twice, a field identifier "
    "reused, a copy in another unit, a merge that repeats rows) before reading the answers."
)


def integrity_status(result: IntegrityResult) -> Status:
    """❌ with a finding, ➖ when no check applied (and nothing was found), ✅ otherwise."""
    if not result.findings.empty:
        return Status.PROBLEM
    checks = result.checks
    if len(checks) and (checks["status"] == "not applicable").all():
        return Status.NOT_ASSESSED
    return Status.OK


def integrity_summary(result: IntegrityResult, detail: str = "short") -> str:
    """The checks in words (see the module); ``detail`` is ``"short"`` or ``"full"``."""
    short = check_detail(detail) == "short"
    settings = result.settings
    status = integrity_status(result)
    shown = Status.NOT_ASSESSED if status is Status.OK and rows_only(result) else status
    lines = [headline(shown, TITLE, _verdict(result, status))]
    lines.append(wrap(_checked(settings, short), 2))
    columns = settings.get("field_columns")
    if not result.findings.empty and columns:
        lines.append(wrap(f"Fields are named {group_columns(columns)}.", 2))
    if not result.findings.empty:
        lines += _finding_table(result) if short else _finding_lines(result)
        lines.append(wrap(_WHAT_TO_DO, 2))
    lines += [wrap(f"{NOT_ASSESSED} {text}", 2, hang=5) for text in _skipped(result)]
    caveats = [*_coverage(settings), _out_of_reach(settings)]
    lines += caveat_lines(caveats, indent=2, detail=detail)
    return "\n".join(lines)


def _verdict(result: IntegrityResult, status: Status) -> str:
    """The headline's answer: what was found, in a few words."""
    if status is Status.PROBLEM:
        involved = result.fields_involved()
        found = count(len(result.findings), "finding")
        fields = count(len(involved), "field")
        return f"{found} in {fields} (structure to inspect, not a judgment)"
    if status is Status.NOT_ASSESSED:
        return "no check could be applied to these features"
    if rows_only(result):
        return "no copied rows; copied fields not checkable here"
    return "nothing found by the exact copy checks"


# The checks that compare whole fields (the others compare rows).
FIELD_COPY_CHECKS = ("copied field", "reordered copy", "scaled copy", "copied small field")


def rows_only(result: IntegrityResult) -> bool:
    """Whether only the row checks compared anything, so "nothing found" would claim more
    than was looked at."""
    checks = result.checks
    if not len(checks) or "compared" not in checks:
        return False
    compared = checks.set_index("check")["compared"]
    return not compared.reindex(list(FIELD_COPY_CHECKS)).fillna(0).gt(0).any()


def _checked(settings: dict[str, Any], short: bool = False) -> str:
    """What was compared: the exact checks, the fields and observations, the features (named
    in full; counted in the short form when more than two)."""
    features = settings.get("feature_cols") or []
    if short and len(features) > 2 and not settings.get("tables"):
        return (
            f"{count(settings.get('n_fovs', 0), 'field')} and "
            f"{settings.get('n_observations', 0)} "
            f"{settings.get('observation_words', 'observations')} checked on "
            f"{count(len(features), 'feature')}, {PRECISION}."
        )
    return (
        f"Exact checks for copied data, {PRECISION}: "
        f"{count(settings.get('n_fovs', 0), 'field')} and "
        f"{settings.get('n_observations', 0)} "
        f"{settings.get('observation_words', 'observations')} checked, on "
        f"{settings.get('features_compared', 'the declared features')}."
    )


def _label(settings: dict[str, Any], name: str, key: str = "field_labels") -> str:
    """A field or group as ``MOCK · 1A · B6``, or its name when no label was recorded."""
    return settings.get(key, {}).get(name, name)


def _groups(findings: pd.DataFrame) -> DataFrameGroupBy:
    """The findings by check and pair of groups (and table, with several), in the order
    found."""
    by = ["check", "group", "other_group", *(["table"] if "table" in findings else [])]
    return findings.groupby(by, sort=False)


def _in(key: tuple[Any, ...]) -> str:
    """`` in DINOv2``: the table or tables a group of findings was found in (owner,
    2026-10-06), empty with the explainable features alone."""
    return f" in {key[3]}" if len(key) > 3 else ""


def _between(settings: dict[str, Any], group: str, other: str) -> str:
    """``within MOCK · 1A · B6`` or ``between MOCK · 1A · B6 and LATS1 · 1A · I23``."""
    first = _label(settings, group, "group_labels")
    if group == other:
        return f"within {first}"
    return f"between {first} and {_label(settings, other, 'group_labels')}"


def _pair(settings: dict[str, Any], row: Any) -> str:
    """The fields of one finding: one field, or both."""
    first = _label(settings, row.field)
    if row.check == "repeated within a field":
        return first
    return f"{first} and {_label(settings, row.other_field)}"


def _capped(lines: list[Any], unit: str) -> tuple[list[Any], str | None]:
    """At most ``SUMMARY_LINES`` items, and the line that counts the others."""
    if len(lines) <= SUMMARY_LINES:
        return lines, None
    more = count(len(lines) - SUMMARY_LINES, unit)
    tail = f"and {more} of findings (IntegrityResult.findings lists every one)"
    return lines[:SUMMARY_LINES], tail


def _finding_table(result: IntegrityResult) -> list[str]:
    """Short: one row per check and pair of groups, then the fields named."""
    settings = result.settings
    rows = []
    for key, found in _groups(result.findings):
        check, group, other = key[:3]
        within = check == "repeated within a field"
        observations = sorted({int(n) for n in found["observations"]})
        rows.append(
            [
                check + _in(key),
                _between(settings, group, other),
                count(len(found), "field" if within else "pair"),
                observations[0]
                if len(observations) == 1
                else f"{observations[0]} to {observations[-1]}",
            ]
        )
    rows, more = _capped(rows, "more row")
    lines = [table(rows, ["check", "where", "fields", "observations"], indent=4, align="lllr")]
    if more:
        lines.append(wrap(more[:1].upper() + more[1:] + ".", 4))
    involved = [_label(settings, name) for name in result.fields_involved()]
    lines.append(wrap(f"Fields involved: {listing(involved, limit=4)}.", 2, hang=4))
    return lines


def _finding_lines(result: IntegrityResult) -> list[str]:
    """Full: one line per finding, or per check and pair of groups when it repeats; capped."""
    settings = result.settings
    lines = []
    for key, rows in _groups(result.findings):
        check, group, other = key[:3]
        named = check + _in(key)
        first = rows.iloc[0]
        observations = count(first.observations, "observation")
        if len(rows) == 1:
            text = f"{named}: {_pair(settings, first)} ({observations}): {first.detail}."
        elif check == "repeated within a field":
            text = (
                f"{named}, {count(len(rows), 'field')} of "
                f"{_label(settings, group, 'group_labels')}, e.g. {_pair(settings, first)} "
                f"({observations}): {first.detail}."
            )
        else:
            text = (
                f"{named}, {count(len(rows), 'pair')} of fields {_between(settings, group, other)}"
                f", e.g. {_pair(settings, first)} ({observations}): {first.detail}."
            )
        lines.append(text)
    lines, more = _capped(lines, "more line")
    if more:
        lines.append(f"{more}.")
    return [wrap(f"- {line}", 2, hang=4) for line in lines]


def _skipped(result: IntegrityResult) -> list[str]:
    """The checks that did not apply, with the reason, one line per reason."""
    skipped = result.checks[result.checks["status"] == "not applicable"]
    lines = []
    for why, rows in skipped.groupby("reason", sort=False):
        named = "any check" if len(rows) == len(result.checks) else listing(list(rows["check"]))
        lines.append(f"Not checked, {named}: {why}.")
    return lines


def _coverage(settings: dict[str, Any]) -> list[str]:
    """What the checks could not compare, counted, so "nothing found" is not read as more."""
    lines = []
    n_fovs, compared = settings.get("n_fovs", 0), settings.get("fields_compared", 0)
    small, uniform = settings.get("fields_too_small", 0), settings.get("fields_too_uniform", 0)
    if compared < n_fovs and settings.get("continuous_features"):
        parts = []
        if small:
            parts.append(
                f"{count(small, 'field')} held fewer than "
                f"{count(settings['min_objects'], 'observation')}"
            )
        if uniform:
            parts.append(
                f"{count(uniform, 'field')} fewer than {settings['min_objects']} distinct values "
                "of every continuous feature, so a coincidence could not be ruled out"
            )
        lines.append(
            f"Whole fields compared: {compared} of {n_fovs}"
            + (f"; {' and '.join(parts)}." if parts else ".")
        )
    recurring = settings.get("recurring_observations", 0)
    if recurring:
        lines.append(
            f"{count(recurring, 'observation')} {'recurs' if recurring == 1 else 'recur'} in "
            f"{match.RECURRING} or more fields (identical low-information objects, such as hot "
            "pixels or empty background) and were set aside, not counted as copied."
        )
    return lines


def _out_of_reach(settings: dict[str, Any]) -> str:
    """What no exact check can see: a standing assumption of every run of the checks."""
    return assumption(
        f"Out of reach of these checks: a copy re-rounded to fewer than {match.DIGITS} "
        f"significant digits, a block of fewer than {settings.get('block_length', 10)} "
        "observations that is not the whole of the smaller field, a copy with feature "
        "columns swapped or a feature (or a value) missing in one copy only, and "
        "observations pasted into two or more other fields "
        "(they read as recurring)."
    )


__all__ = [
    "FIELD_COPY_CHECKS",
    "PRECISION",
    "SUMMARY_LINES",
    "TITLE",
    "integrity_status",
    "integrity_summary",
    "rows_only",
]
