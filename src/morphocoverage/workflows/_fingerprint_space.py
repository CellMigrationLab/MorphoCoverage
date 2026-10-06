"""A feature table's fingerprint read on its first principal components (private;
``_enough_targets``, ``retrospective``).

A fingerprint criterion compares a group's whole distribution over a table's columns, a
covariance or quantiles of every column. On a wide table, a 384-column embedding, a group of
a few images cannot estimate that: the error of its fingerprint reads noise (errors up to
192,282 and "1e13 images per group" on the ECM tables). The rule (owner, 2026-10-06):

- an other feature table's fingerprint reads its first ``k`` principal components, fitted
  on every observation of the table holding all its values, robustly standardised as the
  maps are (``embedding.robust_scale``: median and MAD, the SD where the MAD is under a
  quarter of it); the components keep their variance, so the first weighs most
  (``multivariate_outcomes.ComponentsError``, the inner metric not standardised again);
  where held-back fields judge them (``optimise``), they are refitted in each split on the
  fields not held back (``ComponentsError.trained``; owner, 2026-10-06, plan v2);
- one common ``k`` for every group: at most a fifth (``SHARE``) of the observations of the
  median group the fingerprint judges (an analysis group of at least ``MIN_FIELDS`` fields,
  the observations holding every value of the table), and at most the table's columns
  (owner, 2026-10-06, audit 8; before, of the smallest group, so a pilot condition of a few
  cells set k for every other group and flipped their verdicts);
- a judged group holding fewer than ``SHARE`` times ``k`` observations is not assessable
  and named, never lowering ``k`` (``Reading.small``);
- below ``MIN_COMPONENTS`` (2) components for the median group the fingerprint is not
  assessable in any group, and says why; so it is when no group holds ``MIN_FIELDS`` fields,
  for a table's fingerprint: ``optimise`` then says it is not assessable instead of judging
  the full width (audit 8, seed 9);
- the explainable features keep today's form when they number at most that fifth, or when
  no group holds ``MIN_FIELDS`` fields, and follow the same rule when they are wider.

The criteria are rewritten in a copy of the ``Setup`` (``FingerprintSpace.setup``), so the
report and ``optimise`` judge the same fingerprint; the line of each says ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._seeds import group_key
from .._wording import count, listing
from ..embedding import robust_components
from ..evaluator import iter_analysis_groups
from ..multivariate_outcomes import ComponentsError
from ..schema import validate_input_tables

if TYPE_CHECKING:
    from ..retrospective.criteria import OutcomeCriterion
    from .setup import Setup

# k is at most this fraction (one in SHARE) of the median judged group's observations.
SHARE = 5
# Fewer components than this: the fingerprint of the table is not assessable.
MIN_COMPONENTS = 2
# The fewest fields of a group whose fingerprint is judged (``_enough_targets``).
MIN_FIELDS = 4


def label(criterion: OutcomeCriterion) -> str:
    """The criterion as the report names it: its description, else its metric's name."""
    return criterion.description or criterion.metric.name


def criterion_columns(criterion: OutcomeCriterion) -> list[str]:
    """The feature columns a criterion's metric reads, from its provenance."""
    parameters = criterion.metric.parameters
    named = [*parameters.get("features", []), parameters.get("feature")]
    return [c for c in dict.fromkeys(named) if isinstance(c, str)]


def table_of(columns: list[str], setup: Setup) -> str | None:
    """The feature table ``columns`` belong to, as the text names it (``the explainable
    features``, ``DINOv2``), when there are other tables; ``None`` without them or when the
    columns span several (owner, 2026-10-06: a fingerprint per table)."""
    if not setup.feature_tables:
        return None
    if set(columns) <= set(setup.columns):
        return "the explainable features"
    return next((n for n, c in setup.feature_tables.items() if set(columns) <= set(c)), None)


@dataclass(frozen=True)
class Reading:
    """How one criterion is read: ``components`` (``k``, ``None`` in today's form),
    ``columns`` its table's, ``median`` the observations of the median judged group,
    ``reason`` why it is not assessable (``None`` when it is), ``small`` the judged groups
    holding fewer than ``SHARE`` times ``k`` observations, each its key, name and size, not
    assessable there."""

    components: int | None
    columns: int
    median: int | None
    reason: str | None = None
    small: tuple[tuple[tuple[Any, ...], str, int], ...] = ()

    def words(self, unit: str, units: str) -> str:
        """``read on its first 4 principal components (k at most a fifth of the median
        group's 23 images and at most its 384 columns; not assessable in pilot R1, holding
        fewer than 5 k, 20 images)``, or empty in today's form."""
        if self.components is None or self.median is None:
            return ""
        small = ""
        if self.small:
            named = listing([name for _, name, _ in self.small], 4)
            hold = "holding" if len(self.small) == 1 else "each holding"
            small = (
                f"; not assessable in {named}, {hold} fewer than {SHARE} k, "
                f"{count(SHARE * self.components, unit, units)}"
            )
        return (
            f"read on its first {self.components} principal components (k at most a fifth "
            f"of the median group's {count(self.median, unit, units)} and at most its "
            f"{count(self.columns, 'column')}{small})"
        )


@dataclass(frozen=True)
class FingerprintSpace:
    """The ``setup`` with each assessable criterion as it is read (on components or as
    declared) and the not assessable left out; ``readings`` by criterion label; ``pairs``
    each declared criterion with the one read (``None``: not assessable)."""

    setup: Setup
    readings: dict[str, Reading] = field(default_factory=dict)
    pairs: tuple[tuple[OutcomeCriterion, OutcomeCriterion | None], ...] = ()

    @property
    def not_assessable(self) -> dict[str, str]:
        """Why each criterion left out is not assessable, by label."""
        return {k: r.reason for k, r in self.readings.items() if r.reason is not None}

    def small_groups(self, unit: str, units: str) -> list[str]:
        """``The fingerprint, <criterion>, is not assessable in pilot R1: fewer than 5 k,
        200 cells.``: the groups too small for the common k, one sentence per criterion."""
        return [
            f"The fingerprint, {name}, is not assessable in "
            f"{listing([n for _, n, _ in r.small], 4)}: fewer than {SHARE} k, "
            f"{count(SHARE * int(r.components or 0), unit, units)}."
            for name, r in self.readings.items()
            if r.small
        ]


def _judged(
    data: Any, columns: list[str], min_fields: int
) -> list[tuple[Any, str, int, tuple[tuple[str, str], ...]]]:
    """Each analysis group of at least ``min_fields`` fields: its key, its name, the
    observations holding every value of ``columns`` and its ``(column, value)`` pairs."""
    sizes = []
    for group, members in iter_analysis_groups(data):
        if len(members.fovs) >= min_fields:
            held = ~members.objects[columns].isna().any(axis=1)
            name = " ".join(str(value) for value in group.values())
            pairs = tuple((str(c), str(v)) for c, v in group.items())
            sizes.append((group_key(group), name, int(held.sum()), pairs))
    return sizes


def _read(
    criterion: OutcomeCriterion, data: Any, setup: Setup, min_fields: int
) -> tuple[OutcomeCriterion | None, Reading | None]:
    """The criterion as it is read, and how (the criterion and ``None``: as declared;
    ``None`` and the reading: not assessable)."""
    columns = criterion_columns(criterion)
    metric = criterion.metric
    explainable = set(columns) <= set(setup.columns)
    other = not explainable and table_of(columns, setup) is not None
    # A single-column metric, or columns spanning several tables, are read as declared.
    if not (other or explainable) or not hasattr(metric, "family"):
        return criterion, None
    judged = _judged(data, columns, min_fields)
    if not judged:
        if explainable:
            return criterion, None
        reason = (
            f"no group holds {min_fields} fields, so no group sets the number of its principal "
            "components"
        )
        return None, Reading(None, len(columns), None, reason)
    median = int(np.median([item[2] for item in judged]))
    cap = median // SHARE
    if explainable and len(columns) <= cap:
        return criterion, None
    k = min(cap, len(columns))
    if k < MIN_COMPONENTS:
        reason = (
            f"its median group holds {count(median, setup.unit_word, setup.unit_words)}, "
            f"and {MIN_COMPONENTS} principal components need groups of at least "
            f"{MIN_COMPONENTS * SHARE} (k at most a fifth of the median group)"
        )
        return None, Reading(None, len(columns), median, reason)
    small = tuple(item[:3] for item in judged if item[2] < SHARE * k)
    excluded = tuple(item[3] for item in judged if item[2] < SHARE * k)
    complete = data.objects[columns].dropna()
    centre, scale, loadings = robust_components(complete.to_numpy(dtype=float), k)
    prefix = table_of(columns, setup) if other else "explainable"
    names = tuple(f"{prefix} PC{i + 1}" for i in range(k))
    inner = replace(metric, features=names, standardize=False, weights=None)
    # The groups too small for k are not judged, in the report and in optimise (audit 8).
    read = ComponentsError(
        inner, tuple(columns), tuple(centre), tuple(scale), tuple(loadings), excluded
    )
    return replace(criterion, metric=read), Reading(k, len(columns), median, small=small)


def fingerprint_space(
    values: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, min_fields: int = MIN_FIELDS
) -> FingerprintSpace:
    """Each fingerprint criterion of ``setup`` as it is read on ``values`` (see the module)."""
    if not setup.fingerprint:
        return FingerprintSpace(setup)
    data = validate_input_tables(
        values, fovs=fovs, feature_cols=setup.judged_columns, **setup.common
    )
    pairs, readings = [], {}
    for criterion in setup.fingerprint:
        read, reading = _read(criterion, data, setup, min_fields)
        if reading is not None:
            readings[label(criterion)] = reading
        pairs.append((criterion, read))
    if not readings:
        return FingerprintSpace(setup, {}, tuple(pairs))
    kept = tuple(read for _, read in pairs if read is not None)
    return FingerprintSpace(replace(setup, fingerprint=kept), readings, tuple(pairs))


__all__ = [
    "MIN_COMPONENTS",
    "MIN_FIELDS",
    "SHARE",
    "FingerprintSpace",
    "Reading",
    "criterion_columns",
    "fingerprint_space",
    "label",
    "table_of",
]
