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

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._identity import identity_key
from .._seeds import group_key
from .._style import group_label
from .._wording import count, listing
from ..embedding import robust_components
from ..evaluator import iter_analysis_groups
from ..multivariate_outcomes import ComponentsError, CovarianceError
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


# A criterion's label that is its metric's own name (``covariance_error:d0,d1 <= 0.2``, no
# description given): code, so the report's prose does not repeat it (audit 10, C22).
_CODE_NAME = re.compile(r"^[a-z_]+:\S")


def named(what: str, criterion: str) -> str:
    """``The covariance fingerprint, <description>,`` or ``The covariance fingerprint`` when
    the criterion's label is its metric's name (``_CODE_NAME``)."""
    return f"The {what}" if _CODE_NAME.match(criterion) else f"The {what}, {criterion},"


def plain(criterion: OutcomeCriterion) -> str:
    """The criterion in words: its description, else ``the covariance of d0, d1, d2 and 5
    more within 0.2`` (never its metric's code name; audit 10, C22)."""
    if criterion.description:
        return criterion.description
    columns = criterion_columns(criterion)
    what = "the covariance" if noun(criterion) == "covariance fingerprint" else "the error"
    of = f" of {listing(columns, 3)}" if columns else ""
    return f"{what}{of} within {criterion.threshold:g}"


def fingerprint_line(setup: Setup, unassessed: Sequence[str], short: bool) -> str:
    """The fingerprint declared, named as the report names it ("Covariance fingerprint" for
    covariance criteria), and "not assessable here" when the report could judge none of them,
    never a promise that optimise judges it (audit 10, owner, 2026-10-06, B20 and C22)."""
    nouns = {noun(c) for c in setup.fingerprint}
    title = (
        "Covariance fingerprint"
        if nouns == {"covariance fingerprint"}
        else "Phenotypic fingerprint"
    )
    named = "; ".join(plain(c) for c in setup.fingerprint)
    if unassessed and all(label(c) in unassessed for c in setup.fingerprint):
        return f'{title}: {named} (not assessable here, see "Did I image enough?").'
    if short:
        return f"{title}: {named} (judged on held-back fields by optimise)."
    return f"{title}: {named}; judged on held-back fields."


def noun(criterion: OutcomeCriterion) -> str:
    """``covariance fingerprint`` for a covariance criterion, on components or as declared, as
    a covariance is not the whole distribution (plan v2, owner, 2026-10-06); ``fingerprint``
    for any other."""
    metric = criterion.metric
    inner = metric.metric if isinstance(metric, ComponentsError) else metric
    return "covariance fingerprint" if isinstance(inner, CovarianceError) else "fingerprint"


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
    # The names of the small groups that hold enough observations but too few with a value:
    # no own control in their repeat, or a value missing (audit 10, C28).
    unvalued: tuple[str, ...] = ()

    def words(self, unit: str, units: str) -> str:
        """``read on its first 4 principal components (k at most a fifth of the median
        group's 23 images and at most its 384 columns; not assessable in pilot R1, holding
        fewer than 5 k, 20 images)``, or empty in today's form."""
        if self.components is None or self.median is None:
            return ""
        small = ""
        few = [name for _, name, _ in self.small if name not in self.unvalued]
        if few:
            hold = "holding" if len(few) == 1 else "each holding"
            small = (
                f"; not assessable in {listing(few, 4)}, {hold} fewer than {SHARE} k, "
                f"{count(SHARE * self.components, unit, units)}"
            )
        if self.unvalued:
            their = "its" if len(self.unvalued) == 1 else "their"
            small += (
                f"; not assessable in {listing(list(self.unvalued), 4)}, {their} {units} "
                "without a value here: no own control in the repeat, or a value missing"
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
    each declared criterion with the one read (``None``: not assessable); ``nouns`` how
    each is named, by label (``noun``)."""

    setup: Setup
    readings: dict[str, Reading] = field(default_factory=dict)
    pairs: tuple[tuple[OutcomeCriterion, OutcomeCriterion | None], ...] = ()
    nouns: dict[str, str] = field(default_factory=dict)

    @property
    def not_assessable(self) -> dict[str, str]:
        """Why each criterion left out is not assessable, by label."""
        return {k: r.reason for k, r in self.readings.items() if r.reason is not None}

    def unread(self) -> list[str]:
        """``The covariance fingerprint, <criterion>, is not assessable: <why>.``, each."""
        return [
            f"{named(self.nouns.get(c, 'fingerprint'), c)} is not assessable: {r}."
            for c, r in self.not_assessable.items()
        ]

    def small_groups(self, unit: str, units: str) -> list[str]:
        """``The fingerprint, <criterion>, is not assessable in pilot R1: fewer than 5 k,
        200 cells.``: the groups too small for the common k, one sentence per criterion."""
        said = []
        for name, r in self.readings.items():
            what = named(self.nouns.get(name, "fingerprint"), name)
            few = [n for _, n, _ in r.small if n not in r.unvalued]
            if few:
                said.append(
                    f"{what} is not assessable in {listing(few, 4)}: fewer than {SHARE} k, "
                    f"{count(SHARE * int(r.components or 0), unit, units)}."
                )
            if r.unvalued:  # enough observations, too few with a value (C28)
                said.append(
                    f"{what} is not assessable in {listing(list(r.unvalued), 4)}: {units} "
                    "without a value here, no own control in the repeat or a value missing."
                )
        return said


def _judged(
    data: Any, columns: list[str], min_fields: int
) -> list[tuple[Any, str, int, tuple[tuple[str, str], ...], int, tuple[str, ...]]]:
    """Each analysis group of at least ``min_fields`` fields: its key, its name, the
    observations holding every value of ``columns``, its ``(column, value)`` pairs, its
    observations and its values' types."""
    sizes = []
    for group, members in iter_analysis_groups(data):
        if len(members.fovs) >= min_fields:
            held = ~members.objects[columns].isna().any(axis=1)
            name = group_label(group)  # "b1 · repeat R3 · W07" (audit 10, C28)
            pairs = tuple((str(c), str(v)) for c, v in group.items())
            kinds = tuple(identity_key(v)[1] for v in group.values())  # 1 is not "1" (#420)
            sizes.append((group_key(group), name, int(held.sum()), pairs, len(held), kinds))
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
    unvalued = tuple(item[1] for item in judged if item[2] < SHARE * k <= item[4])
    excluded = tuple(item[3] for item in judged if item[2] < SHARE * k)
    excluded_types = tuple(item[5] for item in judged if item[2] < SHARE * k)
    complete = data.objects[columns].dropna()
    centre, scale, loadings = robust_components(complete.to_numpy(dtype=float), k)
    prefix = table_of(columns, setup) if other else "explainable"
    names = tuple(f"{prefix} PC{i + 1}" for i in range(k))
    inner = replace(metric, features=names, standardize=False, weights=None)
    # The groups too small for k are not judged, in the report and in optimise (audit 8).
    read = ComponentsError(
        inner,
        tuple(columns),
        tuple(centre),
        tuple(scale),
        tuple(loadings),
        excluded,
        excluded_types,
    )
    reading = Reading(k, len(columns), median, small=small, unvalued=unvalued)
    return replace(criterion, metric=read), reading


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
    nouns = {label(c): noun(c) for c in setup.fingerprint}
    if not readings:
        return FingerprintSpace(setup, {}, tuple(pairs), nouns)
    kept = tuple(read for _, read in pairs if read is not None)
    return FingerprintSpace(_keeping(setup, kept), readings, tuple(pairs), nouns)


def _keeping(setup: Setup, kept: tuple[OutcomeCriterion, ...]) -> Setup:
    """``setup`` with the fingerprint ``kept``. When nothing is kept and the fingerprint was
    the only target (``tolerances=()``, no rare target), the copy judges no target, which a
    Setup refuses as a declaration; it is set on a copy, not validated again, and the report
    and ``optimise`` say that no target can be judged."""
    if kept or setup.tolerances or setup.rare is not None:
        return replace(setup, fingerprint=kept)
    judged_nothing = copy.copy(setup)
    object.__setattr__(judged_nothing, "fingerprint", kept)
    return judged_nothing


__all__ = [
    "MIN_COMPONENTS",
    "MIN_FIELDS",
    "SHARE",
    "FingerprintSpace",
    "Reading",
    "criterion_columns",
    "fingerprint_line",
    "fingerprint_space",
    "label",
    "named",
    "noun",
    "plain",
    "table_of",
]
