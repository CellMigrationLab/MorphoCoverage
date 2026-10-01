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
- **copied small field**: every observation of a field too small for the
  checks above (fewer than ``min_objects``, down to one cell per field) is also
  in another field, each with values in at least ``DISTINCTIVE_FEATURES``
  continuous features, so a coincidence to 6 significant digits is not
  plausible: a small field exported twice, or a cell's row pasted into another
  field.
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
from ._integrity_text import PRECISION, integrity_status, integrity_summary
from ._style import Status, group_label
from ._wording import listing, percent, plural, where
from .data_check import validate_input_tables
from .schema import ValidatedInput, fov_positions

CHECKS = (
    "copied field",
    "reordered copy",
    "scaled copy",
    "repeated block",
    "shared observations",
    "copied small field",
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
_PRECISION = PRECISION
# The continuous features an observation needs values in for one shared observation to count as
# copied: two distinct cells agreeing in three continuous measurements to 6 significant digits
# is not a coincidence, where agreeing in one can be.
DISTINCTIVE_FEATURES = 3


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

    @property
    def status(self) -> Status:
        """❌ with a finding, ➖ when no check applied to these features, ✅ otherwise."""
        return integrity_status(self)

    def summary(self, detail: str = "short") -> str:
        """``"short"``: one marked line, what was checked, the findings grouped by check and
        pair of groups in a table, each caveat once; ``"full"``: every finding with its detail.
        Framed as structure to inspect, not a judgment."""
        return integrity_summary(self, detail)


def _fields(data: ValidatedInput) -> tuple[match.Rows, list[str], list[str], dict[str, list]]:
    """The observations by field, each field's name and group (its name without the field),
    and both as the summary prints them (``MOCK · 1A · B6``), by name."""
    positions = fov_positions(data, data.fovs)
    rows = match.Rows.build(data.feature_matrix(), positions, len(data.fovs))
    records = data.fovs.to_dict("records")
    group_key = [column for column in data.fov_key if column != data.fov_col]
    names, groups, labels = [], [], {"field_labels": [], "group_labels": []}
    for record in records:
        field_values = {column: record[column] for column in data.fov_key}
        group_values = {column: record[column] for column in group_key}
        names.append(where(field_values))
        groups.append(where(group_values))
        labels["field_labels"].append(group_label(field_values))
        labels["group_labels"].append(group_label(group_values))
    return rows, names, groups, labels


def _labels(
    findings: list[dict[str, Any]], names: list[str], groups: list[str], labels: dict[str, list]
) -> dict[str, dict[str, str]]:
    """The printed labels of the fields and groups the findings name, by name."""
    named = {name for f in findings for name in (f["field"], f["other_field"])}
    grouped = {group for f in findings for group in (f["group"], f["other_group"])}
    fields_map = {n: label for n, label in zip(names, labels["field_labels"], strict=True)}
    groups_map = {g: label for g, label in zip(groups, labels["group_labels"], strict=True)}
    return {
        "field_labels": {name: fields_map[name] for name in names if name in named},
        "group_labels": {group: groups_map[group] for group in grouped},
    }


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
    5), unless they are every observation of the smaller field and each is
    distinctive (``DISTINCTIVE_FEATURES``); ``block_length`` the shared or
    repeated observations that make a finding whatever the field's size
    (default 10; fewer suffice when they are half the smaller field);
    ``min_distinct_share`` the share of distinct values, among the distinct
    observations, that makes a feature continuous (default 0.5). See the module for each check.
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
    rows, names, groups, labels = _fields(data)
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
    distinctive = np.zeros(len(rows.ids), bool)
    if continuous:
        present = np.isfinite(rows.exact[:, continuous]).sum(axis=1)
        distinctive = informative & (present >= DISTINCTIVE_FEATURES)
    _identical_small(rows, eligible, distinctive, found)
    _shared_rows(rows, informative, distinctive, paired, found, min_objects, block_length)
    record("repeated block", kept, reason)
    record("shared observations", kept, reason)
    small_reason = reason
    if reason is None and len(continuous) < DISTINCTIVE_FEATURES:
        small_reason = (
            f"fewer than {DISTINCTIVE_FEATURES} continuous features "
            f"({len(continuous)}), so one shared observation can be a coincidence"
        )
    record("copied small field", int(distinctive.sum()), small_reason)
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
        "field_columns": list(data.fov_key),
        **_labels(findings, names, groups, labels),
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


def _pair_counts(
    rows: match.Rows, mask: np.ndarray, n_fields: int
) -> tuple[dict[int, int], np.ndarray]:
    """For the rows in ``mask``: how many each pair of fields shares (by ``i * n_fields + j``;
    a row found in exactly two distinct fields links them), and each field's distinct rows."""
    owner = rows.copy_of[rows.field]
    code = np.unique(rows.ids[mask] * n_fields + owner[mask])
    row_id, holder = code // n_fields, code % n_fields
    per_field = np.bincount(holder, minlength=n_fields)
    starts = np.flatnonzero(np.r_[True, row_id[1:] != row_id[:-1]])
    two = np.flatnonzero(np.diff(np.r_[starts, len(row_id)]) == 2)
    if not len(two):
        return {}, per_field
    first, second = holder[starts[two]], holder[starts[two] + 1]
    pair_code, shared = np.unique(first * n_fields + second, return_counts=True)
    return dict(zip(pair_code.tolist(), shared.tolist(), strict=True)), per_field


def _shared_rows(
    rows: match.Rows,
    informative: np.ndarray,
    distinctive: np.ndarray,
    paired: set[tuple[int, int]],
    found: Any,
    min_objects: int,
    block_length: int,
) -> None:
    """Pairs of fields sharing many informative rows (repeated blocks and shared observations),
    or every row of the smaller field when each is distinctive (copied small fields).

    Each informative row occurs in at most two distinct fields (identical
    fields count once), so counting the pairs it links is one pass.
    """
    n_fields = len(rows.sizes)
    shared_pairs, per_field = _pair_counts(rows, informative, n_fields)
    distinct_pairs, _ = _pair_counts(rows, distinctive, n_fields)
    masked = np.where(informative, rows.ids, -1 - np.arange(len(rows.ids)))
    for code_ij, n_shared in shared_pairs.items():
        i, j = divmod(code_ij, n_fields)
        if (i, j) in paired or not (rows.sizes[i] and rows.sizes[j]):
            continue
        smaller = int(min(per_field[i], per_field[j]))
        if n_shared < _threshold(smaller, min_objects, block_length):
            if n_shared == smaller and distinct_pairs.get(code_ij, 0) == n_shared:
                found("copied small field", i, j, n_shared, _small_copy(rows, i, j, n_shared))
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


def _identical_small(
    rows: match.Rows, eligible: np.ndarray, distinctive: np.ndarray, found: Any
) -> None:
    """Fields too small for the whole-field checks whose rounded rows are another field's,
    every row distinctive (identical fields share ``copy_of``, so the row checks see one).

    As for single rows, rows written in ``RECURRING`` or more fields are low-information
    (identical background or hot-pixel objects) and set aside, so only pairs are reported.
    """
    members = np.bincount(rows.copy_of, minlength=len(rows.sizes))
    for i in np.flatnonzero(~eligible & (rows.copy_of != np.arange(len(rows.sizes)))):
        first, i = int(rows.copy_of[i]), int(i)
        if members[first] >= match.RECURRING:
            continue
        n = int(rows.sizes[i])
        if n and distinctive[rows.of(i)].all():
            found("copied small field", first, i, n, _small_copy(rows, first, i, n))


def _small_copy(rows: match.Rows, i: int, j: int, n_shared: int) -> str:
    """The detail of a copied small field: which field is inside the other."""
    small, other = ("first", "second") if rows.sizes[i] <= rows.sizes[j] else ("second", "first")
    if n_shared == 1 and rows.sizes[i] == rows.sizes[j] == 1:
        what = "the two fields hold the same single observation"
    elif n_shared == 1:
        what = f"the {small} field's only observation is also in the {other}"
    else:
        what = f"all {n_shared} observations of the {small} field are also in the {other}"
    return (
        f"{what} ({_PRECISION}, in at least {DISTINCTIVE_FEATURES} continuous features): "
        "a field too small for the whole-field checks, exported twice or pasted"
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


__all__ = ["CHECKS", "DISTINCTIVE_FEATURES", "IntegrityResult", "integrity_checks"]
