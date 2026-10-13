"""What the integrity checks compare, and the log of what they find (private).

``_selection`` picks the continuous features, the informative rows and the fields large
and varied enough to compare as a whole (``_Selection``); ``_Log`` records each finding
between two fields and each check run or not applicable, for ``integrity.integrity_checks``.

Shared-private: ``_Log``, ``_Selection`` and ``_selection``, used by ``integrity``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from . import _integrity_match as match
from ._wording import percent


def _distinct_shares(rows: match.Rows, informative: np.ndarray) -> list[float]:
    """Each feature's distinct values as a share of the distinct rows kept (0 with none)."""
    first = np.unique(rows.ids, return_index=True)[1]
    kept = rows.exact[first[informative[rows.ids[first]]]]
    shares = []
    for j in range(kept.shape[1]):
        values = kept[:, j][np.isfinite(kept[:, j])]
        shares.append(len(np.unique(values)) / len(values) if len(values) else 0.0)
    return shares


class _Log:
    """The findings and the checks of one ``integrity_checks`` run, as each check records
    them, and the pairs of fields already found."""

    def __init__(self, names: list[str], groups: list[str]) -> None:
        self.names = names
        self.groups = groups
        self.findings: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.paired: set[tuple[int, int]] = set()

    def found(self, check: str, i: int, j: int, n: int, detail: str) -> None:
        """A finding of ``check`` between fields ``i`` and ``j``, on ``n`` observations."""
        self.paired.add((min(i, j), max(i, j)))
        self.findings.append(
            {
                "check": check,
                "field": self.names[i],
                "other_field": self.names[j],
                "observations": int(n),
                "detail": detail,
                "group": self.groups[i],
                "other_group": self.groups[j],
            }
        )

    def record(self, check: str, compared: int, why: str | None = None) -> None:
        """A check run on ``compared`` fields or observations, or not applicable (``why``)."""
        status = "checked" if why is None else "not applicable"
        self.checks.append({"check": check, "status": status, "reason": why, "compared": compared})


class _Selection(NamedTuple):
    """What the checks compare: the recurring rows, each feature's distinct share, the
    continuous features, the informative rows, the fields large enough and those compared
    as a whole, and why no check applies (``None`` when they do)."""

    recurring: np.ndarray
    shares: list[float]
    continuous: list[int]
    informative: np.ndarray
    large: np.ndarray
    eligible: np.ndarray
    reason: str | None


def _selection(rows: match.Rows, min_objects: int, min_distinct_share: float) -> _Selection:
    """The features, rows and fields the checks compare (see ``_Selection``)."""
    n_fields = len(rows.sizes)
    recurring = rows.recurrence >= match.RECURRING
    shares = _distinct_shares(rows, ~recurring)
    continuous = [j for j, value in enumerate(shares) if value and value >= min_distinct_share]
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
    return _Selection(recurring, shares, continuous, informative, large, eligible, reason)
