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
- ``k`` is at most a fifth (``SHARE``) of the observations of the smallest group the
  fingerprint judges (an analysis group of at least ``MIN_FIELDS`` fields, the observations
  holding every value of the table), and at most the table's columns;
- below ``MIN_COMPONENTS`` (2) components the fingerprint is not assessable, and says why;
- the explainable features keep today's form when they number at most that fifth, and
  follow the same rule when they are wider.

The criteria are rewritten in a copy of the ``Setup`` (``FingerprintSpace.setup``), so the
report and ``optimise`` judge the same fingerprint; the line of each says ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._wording import count
from ..embedding import robust_scale
from ..evaluator import iter_analysis_groups
from ..multivariate_outcomes import ComponentsError
from ..schema import validate_input_tables

if TYPE_CHECKING:
    from ..retrospective.criteria import OutcomeCriterion
    from .setup import Setup

# k is at most this fraction (one in SHARE) of the smallest judged group's observations.
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
    ``columns`` its table's, ``smallest`` the observations of the smallest judged group,
    ``reason`` why it is not assessable (``None`` when it is)."""

    components: int | None
    columns: int
    smallest: int | None
    reason: str | None = None

    def words(self, unit: str, units: str) -> str:
        """``read on its first 4 principal components (k at most a fifth of the smallest
        group's 23 images and at most its 384 columns)``, or empty in today's form."""
        if self.components is None or self.smallest is None:
            return ""
        return (
            f"read on its first {self.components} principal components (k at most a fifth "
            f"of the smallest group's {count(self.smallest, unit, units)} and at most its "
            f"{count(self.columns, 'column')})"
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


def _smallest(data: Any, columns: list[str], min_fields: int) -> int | None:
    """The observations holding every value of ``columns`` in the smallest analysis group of
    at least ``min_fields`` fields; ``None`` when no group has that many."""
    sizes = []
    for _, group in iter_analysis_groups(data):
        if len(group.fovs) >= min_fields:
            held = ~group.objects[columns].isna().any(axis=1)
            sizes.append(int(held.sum()))
    return min(sizes) if sizes else None


def _components(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, scale and the first ``k`` loadings of the robustly standardised ``matrix``
    (complete rows), the centre folding in the standardised columns' means; a column with
    one value gets scale 1 and no weight."""
    centre, scale = robust_scale(matrix)
    varies = scale > 0
    scale = np.where(varies, scale, 1.0)
    z = (matrix - centre) / scale
    means = z.mean(axis=0)
    _, _, vt = np.linalg.svd((z - means)[:, varies], full_matrices=False)
    loadings = np.zeros((k, matrix.shape[1]))
    loadings[:, varies] = vt[:k]
    # One sign for one data set: the largest loading of each component is positive.
    for row in loadings:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return centre + means * scale, scale, loadings


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
    smallest = _smallest(data, columns, min_fields) if hasattr(metric, "family") else None
    if not (other or explainable) or smallest is None:
        return criterion, None
    cap = smallest // SHARE
    if explainable and len(columns) <= cap:
        return criterion, None
    k = min(cap, len(columns))
    if k < MIN_COMPONENTS:
        reason = (
            f"its smallest group holds {count(smallest, setup.unit_word, setup.unit_words)}, "
            f"and {MIN_COMPONENTS} principal components need groups of at least "
            f"{MIN_COMPONENTS * SHARE} (k at most a fifth of the smallest group)"
        )
        return None, Reading(None, len(columns), smallest, reason)
    complete = data.objects[columns].dropna()
    centre, scale, loadings = _components(complete.to_numpy(dtype=float), k)
    prefix = table_of(columns, setup) if other else "explainable"
    names = tuple(f"{prefix} PC{i + 1}" for i in range(k))
    inner = replace(metric, features=names, standardize=False, weights=None)
    read = ComponentsError(inner, tuple(columns), tuple(centre), tuple(scale), tuple(loadings))
    return replace(criterion, metric=read), Reading(k, len(columns), smallest)


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
