"""Data-integrity checks: structure in the tables that should be inspected before reading them.

These are deterministic checks for the accidents an image-analysis pipeline
makes: a field exported or read twice under another name, the same objects
written again in another order or in other units, rows pasted into another
field, every object of a field written twice by a merge, a field summary
written twice. Observations are compared to 6 significant digits, so a copy
that went through float32 or a six-digit export still matches; a finding
says what was found, not how likely it is by chance, and it never judges
intent: a finding is structure to inspect, not evidence of misconduct.
(Issue #49, after PaperGuard's duplicate detectors; its statistical screens,
such as Benford or terminal-digit tests, do not apply to features extracted
from images and are left out.)

The checks respect the MorphoBricks hierarchy: the unit compared is the
acquired field, and a finding names both fields with their groups.

- **copied field**: two fields hold the same observations, feature for
  feature, in the same order; **reordered copy**: the same, in another order.
- **scaled copy**: one field's observations are the other's under a change of
  units, ``a * x + b`` per feature (``a`` may be negative, ``a`` not 1 or
  ``b`` not 0 for at least one feature), in any order, missing values in the
  same places.
- **repeated block**: a run of consecutive observations (at least
  ``block_length``, or every observation the two share) appears, in table
  order, in two fields; the finding gives the run's length and where it starts.
- **shared observations**: two fields share many observations (at least
  ``block_length``, or half the smaller field), not in one run: a copy with
  observations dropped, added or reordered, or a block pasted in another order.
- **repeated within a field**: a field holds many of its observations more
  than once (a merge with a table that has one row per channel or plane).
- **repeated field summary**: two fields with the same number of
  observations and the same mean of every feature, although their
  observations differ (a summary row written twice, or copied).

Whole fields are compared only when they hold at least ``min_objects``
observations with at least ``min_objects`` distinct values in a continuous
feature (fewer, and any two fields may map onto each other by chance). A
feature is continuous when its distinct values are at least
``min_distinct_share`` of the distinct observations; with none, every check
is ``not applicable``, since quantized or integer features repeat by
themselves. An observation found in three or more fields (identical hot
pixels, empty background patches) is low-information: it is set aside and
counted, not reported as a copy. Exact duplicate rows *including* their
identifiers are reported by ``check_tables``; these checks find copies that
carry another field's name.

The fingerprint of a field is unchanged by the order of its rows, a change of
units or sign per feature, and six-digit rounding; fields with near-equal
fingerprints, found by sorting, are then verified, so the checks take one pass
and a sort over the table, never a comparison of every pair of fields.

What is **not** here, on purpose: near-identical distributions, "too smooth"
data or collapsed between-field variation. They are not deterministic, need
calibration on imaging data, and the variance-component machinery
(``sampling_unit.fov_variability``) already reports the between-field share.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from . import _integrity_match as match
from ._checks import as_count, is_number
from ._wording import count, listing, percent, plural, where
from .data_check import validate_input_tables
from .schema import ValidatedInput, fov_positions

CHECKS = (
    "copied field",
    "reordered copy",
    "scaled copy",
    "repeated block",
    "shared observations",
    "repeated within a field",
    "repeated field summary",
)
FINDING_COLUMNS = (
    "check",
    "field",
    "other_field",
    "observations",
    "detail",
    "group",
    "other_group",
)
CHECK_COLUMNS = ("check", "status", "reason", "compared")
# Lines of findings the summary prints before "and N more".
SUMMARY_LINES = 12
_PRECISION = f"to {match.DIGITS} significant digits"


@dataclass(frozen=True)
class IntegrityResult:
    """What the integrity checks found (``findings``) and what each one could check (``checks``).

    - ``findings``: one row per pair of fields (or per field, for repeats
      within one), with the ``check``, both fields named with their groups
      (``field``, ``other_field``), the ``observations`` involved (the copied
      field's, the run's length, the shared or repeated observations), a
      ``detail``, and the groups alone (``group``, ``other_group``: a well,
      without the field identifier).
    - ``checks``: per check, ``status`` (``checked`` or ``not applicable``),
      the ``reason`` when not applicable, and what was ``compared``: the
      fields for the whole-field checks, the observations for the row checks.
    - ``settings``: the thresholds used, the continuous features, and what was
      set aside (fields too small or too uniform to compare, recurring
      observations).
    """

    findings: pd.DataFrame
    checks: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)

    def fields_involved(self) -> list[str]:
        """Every field named in a finding, in the order of the findings."""
        names = [*self.findings["field"], *self.findings["other_field"]]
        return list(dict.fromkeys(names))

    def summary(self) -> str:
        """The findings in sentences, grouped by pair of groups, framed as structure to inspect."""
        settings = self.settings
        head = (
            "Data integrity (exact checks for copied data, "
            f"{_PRECISION}; structure to inspect, not a judgment): "
            f"{count(settings.get('n_fovs', 0), 'field')} and "
            f"{settings.get('n_observations', 0)} "
            f"{settings.get('observation_words', 'observations')} checked, on "
            f"{settings.get('features_compared', 'the declared features')}."
        )
        lines = [head]
        if self.findings.empty:
            lines.append("  Nothing was found by these checks.")
        lines += _finding_lines(self.findings)
        lines += _coverage_lines(settings)
        skipped = self.checks[self.checks["status"] == "not applicable"]
        for why, rows in skipped.groupby("reason", sort=False):
            named = "any check" if len(rows) == len(self.checks) else listing(list(rows["check"]))
            lines.append(f"  Not checked, {named}: {why}.")
        lines.append(
            f"  Out of reach of these checks: a copy re-rounded to fewer than {match.DIGITS} "
            f"significant digits, a block of fewer than {settings.get('block_length', 10)} "
            "observations, a copy with feature columns swapped or a feature (or a value) "
            "missing in one copy only, and observations pasted into two or more other fields "
            "(they read as recurring)."
        )
        if not self.findings.empty:
            lines.append(
                "  Check where these fields came from (a file read or exported twice, a field "
                "identifier reused, a copy in another unit, a merge that repeats rows) before "
                "reading the answers."
            )
        return "\n".join(lines)


def _finding_lines(findings: pd.DataFrame) -> list[str]:
    """One line per finding, or per check and pair of groups when it repeats; capped."""
    lines = []
    grouped = findings.groupby(["check", "group", "other_group"], sort=False)
    for (check, group, other), rows in grouped:
        first = rows.iloc[0]
        if len(rows) == 1 and check == "repeated within a field":
            lines.append(
                f"  - {check}: {first.field} "
                f"({count(first.observations, 'observation')}): {first.detail}."
            )
            continue
        if len(rows) == 1:
            lines.append(
                f"  - {check}: {first.field} and {first.other_field} "
                f"({count(first.observations, 'observation')}): {first.detail}."
            )
            continue
        if check == "repeated within a field":
            lines.append(
                f"  - {check}, {count(len(rows), 'field')} of {group}, e.g. {first.field} "
                f"({count(first.observations, 'observation')}): {first.detail}."
            )
            continue
        between = f"within {group}" if group == other else f"between {group} and {other}"
        lines.append(
            f"  - {check}, {count(len(rows), 'pair')} of fields {between}, e.g. "
            f"{first.field} and {first.other_field} "
            f"({count(first.observations, 'observation')}): {first.detail}."
        )
    if len(lines) > SUMMARY_LINES:
        more = len(lines) - SUMMARY_LINES
        tail = f"  - and {count(more, 'more line')} of findings (IntegrityResult.findings lists "
        lines = [*lines[:SUMMARY_LINES], tail + "every one)."]
    return lines


def _coverage_lines(settings: dict[str, Any]) -> list[str]:
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
            f"  Whole fields compared: {compared} of {n_fovs}"
            + (f"; {' and '.join(parts)}." if parts else ".")
        )
    recurring = settings.get("recurring_observations", 0)
    if recurring:
        lines.append(
            f"  {count(recurring, 'observation')} {'recurs' if recurring == 1 else 'recur'} in "
            f"{match.RECURRING} or more fields (identical low-information objects, such as hot "
            "pixels or empty background) and were set aside, not counted as copied."
        )
    return lines


def _fields(data: ValidatedInput) -> tuple[match.Rows, list[str], list[str]]:
    """The observations by field, and each field's name and group (its name without the field)."""
    positions = fov_positions(data, data.fovs)
    rows = match.Rows.build(data.feature_matrix(), positions, len(data.fovs))
    records = data.fovs.to_dict("records")
    names = [where({column: record[column] for column in data.fov_key}) for record in records]
    group_key = [column for column in data.fov_key if column != data.fov_col]
    groups = [where({column: record[column] for column in group_key}) for record in records]
    return rows, names, groups


def _distinct_shares(rows: match.Rows, informative: np.ndarray) -> list[float]:
    """Each feature's distinct values as a share of the distinct rows kept (0 with none)."""
    first = np.unique(rows.ids, return_index=True)[1]
    kept = rows.exact[first[informative[rows.ids[first]]]]
    shares = []
    for j in range(kept.shape[1]):
        values = kept[:, j][np.isfinite(kept[:, j])]
        shares.append(len(np.unique(values)) / len(values) if len(values) else 0.0)
    return shares


def _threshold(smaller: int, min_objects: int, block_length: int) -> int:
    """The shared or repeated observations a finding needs: ``block_length``, or half the smaller
    field, and never fewer than ``min_objects``."""
    return max(min_objects, min(block_length, ceil(smaller / 2)))


def integrity_checks(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    min_objects: int = 5,
    block_length: int = 10,
    min_distinct_share: float = 0.5,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> IntegrityResult:
    """Copied fields, reordered or scaled copies, shared or repeated observations and summaries.

    ``feature_cols`` are the features compared (every declared feature is
    best: a copy shows in all of them). ``min_objects`` is the fewest
    observations (and distinct values of a continuous feature) of a field
    compared as a whole, and the fewest shared observations reported (default
    5); ``block_length`` the shared or repeated observations that make a
    finding whatever the field's size (default 10; fewer suffice when they are
    half the smaller field); ``min_distinct_share`` the share of distinct
    values, among the distinct observations, that makes a feature continuous
    (default 0.5). See the module for each check.
    """
    min_objects = as_count(min_objects, "min_objects", 2)
    block_length = as_count(block_length, "block_length", 2)
    if not is_number(min_distinct_share) or not 0 <= min_distinct_share <= 1:
        raise ValueError("min_distinct_share must be a share between 0 and 1")
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(feature_cols),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    rows, names, groups = _fields(data)
    n_fields = len(rows.sizes)
    recurring = rows.recurrence >= match.RECURRING
    shares = _distinct_shares(rows, ~recurring)
    continuous = [j for j, value in enumerate(shares) if value and value >= min_distinct_share]
    findings: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    paired: set[tuple[int, int]] = set()

    def found(check: str, i: int, j: int, n: int, detail: str) -> None:
        paired.add((min(i, j), max(i, j)))
        findings.append(
            {
                "check": check,
                "field": names[i],
                "other_field": names[j],
                "observations": int(n),
                "detail": detail,
                "group": groups[i],
                "other_group": groups[j],
            }
        )

    def record(check: str, compared: int, why: str | None = None) -> None:
        status = "checked" if why is None else "not applicable"
        checks.append({"check": check, "status": status, "reason": why, "compared": compared})

    # Informative rows: not recurring, with a value in a continuous feature.
    has_value = np.isfinite(rows.exact[:, continuous]).any(axis=1) if continuous else None
    informative = (
        ~recurring[rows.ids] & has_value if has_value is not None else np.zeros(len(rows.ids), bool)
    )
    distinct = np.zeros(n_fields, dtype=np.int64)
    for j in continuous:
        distinct = np.maximum(
            distinct, match.distinct_per_field(rows.exact[:, j], rows.field, n_fields)
        )
    large = rows.sizes >= min_objects
    eligible = large & (distinct >= min_objects)
    reason = None
    if not continuous:
        why = "quantized or integer features repeat by themselves"
        if len(rows.ids) and recurring[rows.ids].all():
            why += f"; every observation recurs in {match.RECURRING} or more fields"
        reason = (
            "no observation to compare"
            if not len(rows.ids)
            else "no feature has distinct values for at least "
            f"{percent(min_distinct_share)} of the distinct observations, so a copy cannot be "
            f"told from a coincidence ({why})"
        )
        eligible[:] = False
    whole = [int(i) for i in np.flatnonzero(eligible)]
    _whole_field_checks(rows, whole, found)
    for check in ("copied field", "reordered copy", "scaled copy"):
        record(check, len(whole), reason)
    kept = int(informative.sum())
    _shared_rows(rows, informative, paired, found, min_objects, block_length)
    record("repeated block", kept, reason)
    record("shared observations", kept, reason)
    _within_field(rows, informative, names, groups, findings, min_objects, block_length)
    record("repeated within a field", kept, reason)
    summaries = _summaries(rows, whole, paired, found)
    record("repeated field summary", summaries, reason)
    settings = {
        "feature_cols": list(data.feature_cols),
        "min_objects": min_objects,
        "block_length": block_length,
        "min_distinct_share": float(min_distinct_share),
        "continuous_features": [data.feature_cols[j] for j in continuous],
        "distinct_share": float(max(shares, default=0.0)),
        "digits": match.DIGITS,
        "n_fovs": len(data.fovs),
        "n_observations": data.n_observations,
        "fields_compared": len(whole),
        "fields_too_small": int((~large).sum()) if continuous else 0,
        "fields_too_uniform": int((large & ~eligible).sum()) if continuous else 0,
        "recurring_observations": int(recurring[rows.ids].sum()),
        "observation_words": plural(data.observation_unit),
        "features_compared": listing(list(data.feature_cols)),
    }
    return IntegrityResult(
        pd.DataFrame(findings, columns=list(FINDING_COLUMNS)),
        pd.DataFrame(checks, columns=list(CHECK_COLUMNS)),
        settings,
    )


def _whole_field_checks(rows: match.Rows, whole: list[int], found: Any) -> None:
    """Copied, reordered and scaled copies among the ``whole`` fields (one sort, then verified).

    Fields with exactly the same rounded rows are compared with the first of
    them only, so twenty identical fields make nineteen findings, not 190.
    """
    representatives = [i for i in whole if rows.copy_of[i] == i]
    pairs = [(int(rows.copy_of[i]), i) for i in whole if rows.copy_of[i] != i]
    if len(representatives) > 1:
        pairs += match.candidates(representatives, rows, match.field_moments(rows))
    for i, j in pairs:
        verdict = match.unit_change(rows.values[rows.of(i)], rows.values[rows.of(j)])
        if verdict is None:
            continue
        changed, same_order = verdict
        n = int(rows.sizes[i])
        order = "in the same order" if same_order else "in another order"
        if changed:
            found(
                "scaled copy",
                i,
                j,
                n,
                f"the same observations {order}, each feature scaled or shifted "
                f"({_PRECISION}; a change of units?)",
            )
        else:
            check = "copied field" if same_order else "reordered copy"
            found(check, i, j, n, f"the same observations {order} ({_PRECISION})")


def _shared_rows(
    rows: match.Rows,
    informative: np.ndarray,
    paired: set[tuple[int, int]],
    found: Any,
    min_objects: int,
    block_length: int,
) -> None:
    """Pairs of fields sharing many informative rows: repeated blocks and shared observations.

    Each informative row occurs in at most two distinct fields (identical
    fields count once), so counting the pairs it links is one pass.
    """
    n_fields = len(rows.sizes)
    owner = rows.copy_of[rows.field]
    code = np.unique(rows.ids[informative] * n_fields + owner[informative])
    row_id, holder = code // n_fields, code % n_fields
    starts = np.flatnonzero(np.r_[True, row_id[1:] != row_id[:-1]])
    two = np.flatnonzero(np.diff(np.r_[starts, len(row_id)]) == 2)
    if not len(two):
        return
    first, second = holder[starts[two]], holder[starts[two] + 1]
    pair_code, shared = np.unique(first * n_fields + second, return_counts=True)
    per_field = np.bincount(holder, minlength=n_fields)  # distinct informative rows
    masked = np.where(informative, rows.ids, -1 - np.arange(len(rows.ids)))
    for code_ij, n_shared in zip(pair_code.tolist(), shared.tolist(), strict=True):
        i, j = divmod(code_ij, n_fields)
        if (i, j) in paired or not (rows.sizes[i] and rows.sizes[j]):
            continue
        smaller = int(min(per_field[i], per_field[j]))
        if n_shared < _threshold(smaller, min_objects, block_length):
            continue
        length, start_i, start_j = match.longest_run(masked[rows.of(i)], masked[rows.of(j)])
        if length >= block_length or length == n_shared:
            detail = (
                f"{length} consecutive observations are the same ({_PRECISION}), from "
                f"observation {start_i + 1} of the first field and {start_j + 1} of the second"
            )
            if n_shared > length:
                detail += f"; {n_shared} observations are shared in all"
            found("repeated block", i, j, length, detail)
        else:
            found(
                "shared observations",
                i,
                j,
                n_shared,
                f"{n_shared} of the first field's {int(rows.sizes[i])} observations appear among "
                f"the second's {int(rows.sizes[j])} ({_PRECISION}), not in one run in the same "
                "order: a copy with observations dropped, added or reordered, or a block pasted "
                "in another order",
            )


def _within_field(
    rows: match.Rows,
    informative: np.ndarray,
    names: list[str],
    groups: list[str],
    findings: list[dict[str, Any]],
    min_objects: int,
    block_length: int,
) -> None:
    """Fields holding many of their informative observations more than once."""
    n_fields = len(rows.sizes)
    code, times = np.unique(
        rows.field[informative] * (int(rows.ids.max(initial=0)) + 1) + rows.ids[informative],
        return_counts=True,
    )
    holder = code // (int(rows.ids.max(initial=0)) + 1)
    repeated = np.bincount(holder[times > 1], minlength=n_fields)
    distinct = np.bincount(holder, minlength=n_fields)
    masked = np.where(informative, rows.ids, -1 - np.arange(len(rows.ids)))
    for i in np.flatnonzero(repeated):
        if repeated[i] < _threshold(int(distinct[i]), min_objects, block_length):
            continue
        length, start, again = match.repeated_run(masked[rows.of(i)])
        n = int(repeated[i])
        detail = (
            f"{n} of its {int(rows.sizes[i])} observations appear more than once ({_PRECISION})"
        )
        if length >= min(block_length, n):
            detail = (
                f"the field holds its observations {start + 1}-{start + length} again from "
                f"observation {again + 1} ({_PRECISION})"
                + (f"; {n} observations repeat in all" if n > length else "")
                + ": a table appended to itself, or merged with one that has a row per channel "
                "or plane"
            )
        findings.append(
            {
                "check": "repeated within a field",
                "field": names[i],
                "other_field": names[i],
                "observations": n,
                "detail": detail,
                "group": groups[i],
                "other_group": groups[i],
            }
        )


def _summaries(rows: match.Rows, whole: list[int], paired: set[tuple[int, int]], found: Any) -> int:
    """Pairs of fields with the same count and means but other observations; the fields compared."""
    buckets: dict[tuple, list[int]] = {}
    for i in whole:
        values = rows.values[rows.of(i)]
        present = ~np.isnan(values)
        with np.errstate(invalid="ignore"):
            means = np.where(present, values, 0.0).sum(axis=0) / np.maximum(present.sum(axis=0), 1)
        # Twelve significant digits: the same values summed in another order differ
        # only in the last bits of their mean.
        key = (len(values), present.sum(axis=0).tobytes(), *(f"{m:.12g}" for m in means))
        buckets.setdefault(key, []).append(i)
    for members in buckets.values():
        for a, i in enumerate(members):
            for j in members[a + 1 :]:
                if (i, j) in paired or rows.copy_of[i] == rows.copy_of[j]:
                    continue
                # Fields with the same rounded rows share one ``copy_of``, so these differ.
                found(
                    "repeated field summary",
                    i,
                    j,
                    int(rows.sizes[i]),
                    "the same number of observations and the same mean of every feature, "
                    "although the observations differ",
                )
    return len(whole)


__all__ = ["CHECKS", "IntegrityResult", "integrity_checks"]
