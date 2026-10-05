"""The simplest rule on the features that picks out a group of cells (private; ``populations``;
owner, 2026-10-02).

A group found by clustering is a candidate the user validates on the images; once it is
validated, it is studied as a rare phenotype, which needs a label on every cell. The label
is a rule a person can read and reuse: at most ``DEPTH`` thresholds on the declared features,
in their own units, from a decision tree (scikit-learn, its classes balanced so a small
group is not ignored) of depth ``DEPTH``: of the paths from its root, the one that best
tells the group apart (F1), the shorter when two do as well.
The rule shown is fitted on every cell. How well the procedure that fits it does is measured
by fitting it again with whole fields held out (owner, 2026-10-05): ``FOLDS`` folds of fields
(scikit-learn's ``StratifiedGroupKFold`` on the field key, seeded), fewer when the group is in
fewer fields, each fold's rule judged on the cells of its held-out fields: ``recall``, the
share of the group's cells it picks, and ``precision``, the share of the cells it picks that
are in the group. These describe the procedure, not the exact thresholds shown. A group in
one field has no field to hold out: its cells are then held out instead (``FOLDS``-fold,
stratified, seeded), and ``held_out`` says which (``"fields"``, ``"cells"``; empty when the
group is too small to hold out).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEPTH = 2
FOLDS = 5
# The fewest cells a leaf of the tree holds, as a share of the group.
LEAF_SHARE = 0.1


@dataclass(frozen=True)
class Rule:
    """``terms``: ``(feature, ">" or "<=", threshold)``, all to hold, fitted on every cell;
    ``recall`` and ``precision`` of the procedure refitted with ``held_out`` (``"fields"`` or
    ``"cells"``) held out (NaN, ``held_out`` empty, when the group is too small to hold out)."""

    terms: tuple[tuple[str, str, float], ...]
    recall: float
    precision: float
    held_out: str = ""

    def words(self) -> str:
        """``AR > 2.91 and Round <= 0.41``."""
        return " and ".join(f"{f} {'>' if op == '>' else '≤'} {t:.3g}" for f, op, t in self.terms)

    def code(self, table: str = "objects") -> str:
        """The rule as pandas code on ``table``: ``(objects["AR"] > 2.91) & ...``."""
        return " & ".join(f'({table}["{f}"] {op} {t:.6g})' for f, op, t in self.terms)

    def apply(self, values: np.ndarray, features: Sequence[str]) -> np.ndarray:
        """Which rows of ``values`` (columns ``features``) the rule picks."""
        picked = np.ones(len(values), dtype=bool)
        for feature, op, threshold in self.terms:
            column = values[:, list(features).index(feature)]
            picked &= column > threshold if op == ">" else column <= threshold
        return picked


def _fit(values: np.ndarray, member: np.ndarray, features: Sequence[str], seed: int) -> tuple:
    """The terms of the best rule from a balanced tree: of every path from its root (one
    threshold or two), the one that best tells the group apart on these cells (F1), the
    shorter when two do as well."""
    from sklearn.tree import DecisionTreeClassifier

    leaf = max(2, int(np.ceil(LEAF_SHARE * member.sum())))
    tree = DecisionTreeClassifier(
        max_depth=DEPTH, class_weight="balanced", min_samples_leaf=leaf, random_state=seed
    ).fit(values, member)
    inner = tree.tree_
    paths: dict[int, tuple] = {0: ()}
    for node in range(inner.node_count):
        for child, side in ((inner.children_left[node], "<="), (inner.children_right[node], ">")):
            if child >= 0:
                term = (features[inner.feature[node]], side, float(inner.threshold[node]))
                paths[int(child)] = (*paths[node], term)
    best, score = (), 0.0
    for terms in sorted(paths.values(), key=len):
        if not terms:
            continue
        picked = Rule(terms, np.nan, np.nan).apply(values, features)
        hits = int((picked & member).sum())
        if not hits:
            continue
        f1 = 2 * hits / (int(picked.sum()) + int(member.sum()))
        if f1 > score + 1e-9:
            best, score = terms, f1
    return best


def _folds(
    member: np.ndarray, groups: np.ndarray | None, seed: int
) -> tuple[list[np.ndarray], str]:
    """The rows of each held-out fold, and what is held out: whole fields when the group is
    in two or more (``FOLDS`` folds, fewer with fewer fields), else cells."""
    if groups is not None:
        codes = pd.factorize(pd.Series(groups, dtype=object))[0]
        fields = len(np.unique(codes[member]))
        if fields >= 2:
            from sklearn.model_selection import StratifiedGroupKFold

            splits = StratifiedGroupKFold(
                n_splits=min(FOLDS, fields), shuffle=True, random_state=seed
            ).split(np.zeros(len(member)), member, codes)
            return [test for _, test in splits], "fields"
    rng = np.random.default_rng(seed)
    folds = np.empty(len(member), dtype=int)
    for side in (True, False):
        rows = np.flatnonzero(member == side)
        folds[rows] = rng.permutation(len(rows)) % FOLDS
    return [np.flatnonzero(folds == fold) for fold in range(FOLDS)], "cells"


def population_rule(
    values: np.ndarray,
    member: np.ndarray,
    features: Sequence[str],
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> Rule:
    """The rule for ``member`` on ``values`` (one column per feature, in their own units),
    fitted on every row, and how well its procedure does refitted with whole fields of
    ``groups`` (each row's field) held out, or rows without ``groups`` (see the module)."""
    features = list(features)
    terms = _fit(values, member, features, seed)
    if not terms:
        return Rule((), np.nan, np.nan)
    if member.sum() < 2 * FOLDS:
        return Rule(terms, np.nan, np.nan)
    folds, held_out = _folds(member, groups, seed)
    picked = np.zeros(len(values), dtype=bool)
    for test in folds:
        train = np.ones(len(values), dtype=bool)
        train[test] = False
        fitted = _fit(values[train], member[train], features, seed)
        if fitted:
            picked[test] = Rule(fitted, np.nan, np.nan).apply(values[test], features)
    hits = int((picked & member).sum())
    recall = hits / int(member.sum())
    precision = hits / int(picked.sum()) if picked.any() else 0.0
    return Rule(terms, float(recall), float(precision), held_out)


__all__ = ["DEPTH", "FOLDS", "Rule", "population_rule"]
