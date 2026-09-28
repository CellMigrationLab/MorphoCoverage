"""Data-integrity checks: structure in the tables that should be inspected before reading them.

These are deterministic checks for the accidents an image-analysis pipeline
makes: a field exported or read twice under another name, the same objects
written again in another order or in other units, a block of rows pasted into
another field, a field summary written twice. Each is exact (up to
floating-point rounding), so it says what was found, not how likely it is by
chance, and it never judges intent: a finding is structure to inspect, not
evidence of misconduct. (Issue #49, after PaperGuard's duplicate detectors;
its statistical screens, such as Benford or terminal-digit tests, do not
apply to features extracted from images and are left out.)

The checks respect the MorphoBricks hierarchy: the unit compared is the
acquired field, and a finding names both fields with their groups.

- **copied field**: two fields hold the same observations, feature for
  feature, in the same order; **reordered copy**: the same, in another order.
- **scaled copy**: one field's observations are the other's under a change of
  units, ``a * x + b`` per feature (``a`` not 1 or ``b`` not 0 for at least
  one feature), matched row for row.
- **repeated block**: ``block_length`` consecutive observations, in table
  order, appear in two fields (or twice in one field).
- **repeated field summary**: two fields with the same number of
  observations and the same mean of every feature, although their
  observations differ (a summary row written twice, or copied).

A check is ``not applicable`` when its answer would be chance: when fewer
than ``min_distinct_share`` of the observations are distinct (quantized or
integer features repeat by themselves), the copy and block checks cannot tell
a copy from a coincidence; a field summary made only of integers (counts)
likewise. Fields with fewer than ``min_objects`` observations are not
compared as copies. Exact duplicate rows *including* their identifiers are
reported by ``check_tables``; these checks find copies that carry another
field's name.

What is **not** here, on purpose: near-identical distributions, "too smooth"
data or collapsed between-field variation. They are not deterministic, need
calibration on imaging data, and the variance-component machinery
(``sampling_unit.fov_variability``) already reports the between-field share.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._checks import as_count, is_number
from ._wording import count, listing, percent, plural, where
from .data_check import validate_input_tables
from .schema import ValidatedInput, fov_positions
from .statistics import RELATIVE_TOLERANCE

CHECKS = (
    "copied field",
    "reordered copy",
    "scaled copy",
    "repeated block",
    "repeated field summary",
)
FINDING_COLUMNS = ("check", "field", "other_field", "observations", "detail")
CHECK_COLUMNS = ("check", "status", "reason", "compared")


@dataclass(frozen=True)
class IntegrityResult:
    """What the integrity checks found (``findings``) and what each one could check (``checks``).

    - ``findings``: one row per pair of fields, with the ``check``, both
      fields named with their groups (``field``, ``other_field``), the number
      of ``observations`` involved and a ``detail``.
    - ``checks``: per check, ``status`` (``checked`` or ``not applicable``),
      the ``reason`` when not applicable, and what was ``compared``: the fields
      for the copy and summary checks, the pairs of fields of equal size for
      scaled copies, the windows of consecutive observations for blocks.
    - ``settings``: the thresholds used and the share of distinct observations.
    """

    findings: pd.DataFrame
    checks: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The findings in sentences, framed as structure to inspect."""
        head = (
            "Data integrity (exact checks for copied data; structure to inspect, not a "
            f"judgment): {count(self.settings.get('n_fovs', 0), 'field')} and "
            f"{self.settings.get('n_observations', 0)} "
            f"{self.settings.get('observation_words', 'observations')} checked, on "
            f"{self.settings.get('features_compared', 'the declared features')}."
        )
        lines = [head]
        if self.findings.empty:
            lines.append("  Nothing was found: no field repeats another, in any order or units.")
        for row in self.findings.itertuples():
            lines.append(
                f"  - {row.check}: {row.field} and {row.other_field} "
                f"({count(row.observations, 'observation')}): {row.detail}."
            )
        skipped = self.checks[self.checks["status"] == "not applicable"]
        for row in skipped.itertuples():
            lines.append(f"  Not checked, {row.check}: {row.reason}.")
        if not self.findings.empty:
            lines.append(
                "  Check where these fields came from (a file read or exported twice, a field "
                "identifier reused, a copy in another unit) before reading the answers."
            )
        return "\n".join(lines)


def _fields(data: ValidatedInput) -> tuple[list[np.ndarray], list[str]]:
    """Each acquired field's feature rows, in table order, and its name with its group."""
    matrix = data.feature_matrix()
    positions = fov_positions(data, data.fovs)
    order = np.argsort(positions, kind="stable")
    counts = np.bincount(positions, minlength=len(data.fovs))
    rows = np.split(matrix[order], np.cumsum(counts)[:-1])
    names = [
        where({column: record[column] for column in data.fov_key})
        for record in data.fovs.to_dict("records")
    ]
    return rows, names


def _digest(rows: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(rows).tobytes()).digest()


def _canonical(rows: np.ndarray) -> np.ndarray:
    """The rows in a canonical order (lexicographic), so a reordered copy matches."""
    if len(rows) == 0:
        return rows
    return rows[np.lexsort(rows.T[::-1])]


def _affine(first: np.ndarray, second: np.ndarray) -> tuple[bool, bool]:
    """Whether ``second`` is ``a * first + b`` per column, row for row, and whether it differs.

    Rows are aligned by sorting both on the first column that varies (in
    reverse for a negative slope). Constant columns must map to constants.
    """
    varying = np.flatnonzero(np.ptp(first, axis=0) > 0)
    if not len(varying) or np.isnan(first).any() or np.isnan(second).any():
        return False, False
    key = varying[0]
    for sign in (1, -1):
        a_rows = first[np.argsort(first[:, key], kind="stable")]
        b_rows = second[np.argsort(sign * second[:, key], kind="stable")]
        changed, ok = False, True
        for j in range(first.shape[1]):
            x, y = a_rows[:, j], b_rows[:, j]
            scale = max(float(np.abs(y).max()), float(np.abs(x).max()), 1e-300)
            if np.ptp(x) == 0:
                ok = np.ptp(y) <= RELATIVE_TOLERANCE * scale
                changed |= bool(ok and abs(y[0] - x[0]) > RELATIVE_TOLERANCE * scale)
            else:
                slope, intercept = np.polyfit(x, y, 1)
                residual = float(np.abs(y - (slope * x + intercept)).max())
                ok = residual <= 1e-9 * scale and slope * sign > 0
                changed |= bool(
                    abs(slope - 1) > 1e-9 or abs(intercept) > RELATIVE_TOLERANCE * scale
                )
            if not ok:
                break
        if ok:
            return True, changed
    return False, False


def _pairs_of(buckets: dict[Any, list[int]]) -> list[tuple[int, int]]:
    return [
        (members[i], members[j])
        for members in buckets.values()
        for i in range(len(members))
        for j in range(i + 1, len(members))
    ]


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
    """Copied fields, reordered or scaled copies, repeated blocks and summaries (see the module).

    ``feature_cols`` are the features compared (every declared feature is
    best: a copy shows in all of them). ``min_objects`` is the fewest
    observations of a field compared as a copy (default 5); ``block_length``
    the run of consecutive observations a repeated block must span (default
    10); ``min_distinct_share`` the share of distinct observations below
    which the copy and block checks are not applicable (default 0.5).
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
    rows, names = _fields(data)
    matrix = data.feature_matrix()
    distinct = len(np.unique(matrix, axis=0)) / len(matrix) if len(matrix) else 1.0
    quantized = distinct < min_distinct_share
    reason = (
        f"only {percent(distinct)} of the observations are distinct (quantized or integer "
        "features repeat by themselves), so a copy cannot be told from a coincidence"
    )
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
            }
        )

    def record(check: str, compared: int, why: str | None = None) -> None:
        status = "checked" if why is None else "not applicable"
        checks.append({"check": check, "status": status, "reason": why, "compared": compared})

    large = [i for i, r in enumerate(rows) if len(r) >= min_objects]
    # Copies, in the same order or another: fields with the same canonical rows.
    by_rows: dict[bytes, list[int]] = defaultdict(list)
    for i in large:
        by_rows[_digest(_canonical(rows[i])) + bytes(str(rows[i].shape), "ascii")].append(i)
    copy_pairs = _pairs_of(by_rows)
    if quantized:
        record("copied field", 0, reason)
        record("reordered copy", 0, reason)
    else:
        for i, j in copy_pairs:
            if not np.array_equal(_canonical(rows[i]), _canonical(rows[j]), equal_nan=True):
                continue
            if np.array_equal(rows[i], rows[j], equal_nan=True):
                found("copied field", i, j, len(rows[i]), "the same observations in the same order")
            else:
                found(
                    "reordered copy",
                    i,
                    j,
                    len(rows[i]),
                    "the same observations in another order",
                )
        record("copied field", len(large))
        record("reordered copy", len(large))
    # Scaled copies: same number of observations, an exact change of units.
    by_size: dict[int, list[int]] = defaultdict(list)
    for i in large:
        by_size[len(rows[i])].append(i)
    size_pairs = [p for p in _pairs_of(by_size) if p not in paired]
    if quantized:
        record("scaled copy", 0, reason)
    else:
        for i, j in size_pairs:
            same, changed = _affine(rows[i], rows[j])
            if same and changed:
                found(
                    "scaled copy",
                    i,
                    j,
                    len(rows[i]),
                    "the same observations, each feature scaled or shifted (a change of units?)",
                )
        record("scaled copy", len(size_pairs))
    # Repeated blocks of consecutive observations, in table order.
    if quantized:
        record("repeated block", 0, reason)
    else:
        windows: dict[bytes, tuple[int, int]] = {}
        blocks: dict[tuple[int, int], tuple[int, int, int]] = {}
        compared = 0
        for i, field_rows in enumerate(rows):
            if len(field_rows) < block_length:
                continue
            row_ids = [_digest(r) for r in field_rows]
            for start in range(len(field_rows) - block_length + 1):
                window = row_ids[start : start + block_length]
                if len(set(window)) < 2:
                    continue  # a run of one repeated row says nothing
                compared += 1
                key = hashlib.sha1(b"".join(window)).digest()
                if key not in windows:
                    windows[key] = (i, start)
                    continue
                other, other_start = windows[key]
                if other == i and abs(start - other_start) < block_length:
                    continue  # the same rows, overlapping
                pair = (min(i, other), max(i, other))
                if pair not in paired and pair not in blocks:
                    starts = (other_start, start) if other <= i else (start, other_start)
                    blocks[pair] = starts
        for (a, b), (start_a, start_b) in blocks.items():
            found(
                "repeated block",
                a,
                b,
                block_length,
                f"{block_length} consecutive observations are the same, from observation "
                f"{start_a + 1} of the first field and {start_b + 1} of the second",
            )
        record("repeated block", compared)
    # Repeated field summaries: same count and same means, different observations.
    summaries: dict[tuple, list[int]] = defaultdict(list)
    integer_only = 0
    for i, field_rows in enumerate(rows):
        if not len(field_rows) or np.isnan(field_rows).any():
            continue
        means = field_rows.mean(axis=0)
        if np.all(np.equal(np.mod(field_rows, 1), 0)):
            integer_only += 1
            continue  # counts collide by themselves
        # Twelve significant digits: the same values summed in another order differ
        # only in the last bits of their mean.
        summaries[(len(field_rows), *(f"{m:.12g}" for m in means))].append(i)
    summary_pairs = [p for p in _pairs_of(summaries) if p not in paired]
    for i, j in summary_pairs:
        found(
            "repeated field summary",
            i,
            j,
            len(rows[i]),
            "the same number of observations and the same mean of every feature, although "
            "the observations differ",
        )
    why = (
        f"{count(integer_only, 'field')} {'holds' if integer_only == 1 else 'hold'} only "
        "integer values (counts), whose summaries repeat by themselves"
        if integer_only and not summaries
        else None
    )
    record("repeated field summary", sum(len(v) for v in summaries.values()), why)
    unit_words = plural(data.observation_unit)
    settings = {
        "feature_cols": list(data.feature_cols),
        "min_objects": min_objects,
        "block_length": block_length,
        "min_distinct_share": float(min_distinct_share),
        "distinct_share": float(distinct),
        "n_fovs": len(data.fovs),
        "n_observations": data.n_observations,
        "observation_words": unit_words,
        "features_compared": listing(list(data.feature_cols)),
    }
    return IntegrityResult(
        pd.DataFrame(findings, columns=list(FINDING_COLUMNS)),
        pd.DataFrame(checks, columns=list(CHECK_COLUMNS)),
        settings,
    )


__all__ = ["CHECKS", "IntegrityResult", "integrity_checks"]
