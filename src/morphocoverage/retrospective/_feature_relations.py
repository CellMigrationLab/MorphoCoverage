"""Declared features that repeat one measurement, for "The data" (private; ``_value_checks``;
owner, 2026-10-05).

On the owner's ECM data ``Porosity X (mean)`` and ``Porosity Y (mean)`` were copies of
``Porosity (mean)``, itself exactly 1 - ``HDM``: one measurement was counted four times in
every Benjamini-Hochberg family, in the heatmap and in the discovery. So the declared
features of the observation table (the explainable ones; the other feature tables, whose
dimensions are expected to correlate, are not checked) are compared pair by pair, on the
rows where both are finite (at least ``MIN_ROWS``):

- **copy**: the two columns are equal, up to ``TOLERANCE`` of the column's range;
- **derived**: one is an exact affine function of the other, ``y = a x + b`` (``1 - x``, a
  change of units), the largest residual within ``TOLERANCE`` of ``y``'s range;
- **near copy**: neither, but ``|r| >= NEAR_R`` (Pearson).

A feature that is constant (``statistics.spread_is_zero``) is left out: the few-values
check (``_value_checks.few_values``) and every analysis already say so. Related pairs are
joined into groups, each said once as a caution (⚠️) naming what to keep. Nothing is
dropped or merged and no statistic changes: the user re-declares the features.

With several controls the same check also runs on each experiment's rows (owner,
2026-10-05; on the ECM data Porosity (mean) = 1 - HDM held in the untransfected experiment
only). An experiment is a control, the conditions whose own control it is, and its parent
when one is linked (``workflows._contrasts.experiments``). A relation found in some
experiments but not on all rows is said once, naming the conditions where it holds, since
the experiments may then have been processed differently; one that holds on all rows is
said once as above, never again per experiment. An experiment with fewer than
``MIN_ROWS`` rows where both values are finite is not checked for that pair.

One pass of matrix products gives every pair's correlation on its shared rows, so hundreds
of features cost a correlation matrix; only the pairs that pass the screen are fitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._wording import listing
from ..metadata import CONDITION_COLUMN
from ..schema import ValidatedInput, fov_positions
from ..statistics import spread_is_zero

# The |Pearson r| from which two features that are not exact copies are near copies.
NEAR_R = 0.999
# The largest residual, as a share of the column's range, of an exact relation: rounding to
# six significant digits stays well within it, a near copy at r = 0.999 does not.
TOLERANCE = 1e-4
# The fewest rows with both values finite on which a pair is compared.
MIN_ROWS = 10
NAME = "Duplicated or derived features"
# Groups said one by one; the rest are counted in one more sentence.
_GROUPS = 6
# Relations written out within one group; the rest are counted.
_RELATIONS = 4
_MINUS = "−"


def _pair_correlations(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each pair's Pearson r on the rows where both are finite, and that row count."""
    finite = np.isfinite(values)
    # Centred on each column's own mean first, so large offsets lose no precision.
    centred = values - np.nanmean(values, axis=0)
    filled = np.where(finite, centred, 0.0)
    mask = finite.astype(float)
    rows = mask.T @ mask
    sums = filled.T @ mask  # [i, j]: the sum of column i over the rows j is finite
    squares = (filled**2).T @ mask
    with np.errstate(divide="ignore", invalid="ignore"):
        products = filled.T @ filled - sums * sums.T / rows
        spread = squares - sums**2 / rows
        r = products / np.sqrt(spread * spread.T)
    return r, rows


def _relation(x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    """How ``y`` follows ``x`` on their shared finite rows (see the module), ``None`` when
    it does not."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    span_x, span_y = float(np.ptp(x)), float(np.ptp(y))
    if len(x) < MIN_ROWS or span_x == 0 or span_y == 0:
        return None
    if float(np.abs(y - x).max()) <= TOLERANCE * span_y:
        return {"kind": "copy"}
    slope, intercept = np.polyfit(x, y, 1)
    if float(np.abs(y - (slope * x + intercept)).max()) <= TOLERANCE * span_y:
        # An intercept within rounding of zero is a pure change of scale.
        zero = abs(intercept) <= TOLERANCE * span_y
        return {"kind": "derived", "a": float(slope), "b": 0.0 if zero else float(intercept)}
    r = float(np.corrcoef(x, y)[0, 1])
    return {"kind": "near", "r": r} if abs(r) >= NEAR_R else None


def _related(values: np.ndarray) -> list[tuple[int, int, dict[str, Any]]]:
    """Every related pair of columns of ``values`` as ``(x, y, relation)``, ``x < y``."""
    varied = []
    for index in range(values.shape[1]):
        column = values[:, index]
        column = column[np.isfinite(column)]
        if len(column) >= MIN_ROWS and not spread_is_zero(float(column.std()), column):
            varied.append(index)
    if len(varied) < 2:
        return []
    r, rows = _pair_correlations(values[:, varied])
    # The screen leaves room for rounding; each pair it passes is fitted exactly.
    first, second = np.nonzero(np.triu((np.abs(r) >= NEAR_R - 1e-9) & (rows >= MIN_ROWS), 1))
    relations = []
    for i, j in zip(first.tolist(), second.tolist()):
        x, y = varied[i], varied[j]
        found = _relation(values[:, x], values[:, y])
        if found is not None:
            relations.append((x, y, found))
    return relations


def _groups(n: int, relations: list[tuple[int, int, Any]]) -> dict[int, list[int]]:
    """The columns joined by ``relations``, by their first column: ``{first: [columns]}``."""
    parent = list(range(n))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for x, y, _ in relations:
        parent[max(root(x), root(y))] = min(root(x), root(y))
    joined = {root(x) for x, _, _ in relations}
    groups: dict[int, list[int]] = {}
    for index in range(n):
        if root(index) in joined:
            groups.setdefault(root(index), []).append(index)
    return dict(sorted(groups.items()))


def related_features(data: ValidatedInput) -> list[dict[str, Any]]:
    """The groups of declared features that repeat one measurement on all rows, in declared
    order: ``{"features": [...], "relations": [(x, y, relation), ...]}``, each relation as
    ``_relation`` gives it, ``y`` declared after ``x`` (see the module)."""
    names = list(data.feature_cols)
    if len(names) < 2:
        return []
    relations = _related(data.feature_matrix())
    groups = _groups(len(names), relations)
    return [
        {
            "features": [names[i] for i in members],
            "relations": [(names[x], names[y], f) for x, y, f in relations if x in members],
        }
        for members in groups.values()
    ]


def _conditions(data: ValidatedInput, condition_col: str) -> np.ndarray | None:
    """Each observation's condition, ``None`` without the column."""
    if condition_col not in data.fovs.columns:
        return None
    positions = fov_positions(data, data.fovs)
    labels = data.fovs[condition_col].to_numpy()[np.maximum(positions, 0)]
    return np.where(positions >= 0, labels, None)


def experiment_relations(
    data: ValidatedInput,
    experiments: Mapping[Any, Sequence[Any]],
    condition_col: str = CONDITION_COLUMN,
) -> list[dict[str, Any]]:
    """The relations that hold within some experiments but not on all rows, grouped by the
    experiments they hold in: ``{"experiments": [...], "conditions": [...], "relations":
    [(x, y, relation), ...], "everywhere": ...}``, each relation as the first of those
    experiments gives it; ``everywhere`` when it holds in every experiment, with other
    coefficients in each."""
    names = list(data.feature_cols)
    labels = _conditions(data, condition_col)
    if len(names) < 2 or labels is None or len(experiments) < 2:
        return []
    values = data.feature_matrix()
    pooled = {(x, y) for x, y, _ in _related(values)}
    found: dict[tuple[int, int], tuple[dict[str, Any], list[Any]]] = {}
    for experiment, conditions in experiments.items():
        rows = np.isin(labels, list(conditions))
        for x, y, relation in _related(values[rows]):
            if (x, y) not in pooled:
                found.setdefault((x, y), (relation, []))[1].append(experiment)
    by_place: dict[tuple[Any, ...], list[tuple[str, str, dict[str, Any]]]] = {}
    for (x, y), (relation, where) in sorted(found.items()):
        by_place.setdefault(tuple(where), []).append((names[x], names[y], relation))
    return [
        {
            "experiments": list(where),
            "conditions": list(dict.fromkeys(c for e in where for c in experiments[e])),
            "relations": relations,
            "everywhere": len(where) == len(experiments),
        }
        for where, relations in by_place.items()
    ]


def _number(value: float) -> str:
    """A coefficient: ``1000``, ``0.001``, ``2.54``."""
    return f"{abs(value):.4g}"


def _affine(x: str, relation: dict[str, Any]) -> str:
    """``1 − x``, ``1000 × x``, ``2 × x + 3``, ``−x``."""
    a, b = relation["a"], relation["b"]
    term = x if _number(a) == "1" else f"{_number(a)} × {x}"
    if b == 0:
        return term if a > 0 else f"{_MINUS}{term}"
    if a < 0:
        sign = "" if b > 0 else _MINUS
        return f"{sign}{_number(b)} {_MINUS} {term}"
    return f"{term} {'+' if b > 0 else _MINUS} {_number(b)}"


def _r(value: float) -> str:
    """``0.9995`` or ``−0.9993``, never rounded up to 1."""
    digits = np.floor(abs(value) * 1e4) / 1e4
    return f"{'' if value > 0 else _MINUS}{digits:.4f}"


def _times(n: int) -> str:
    """``twice``, ``3 times``."""
    return "twice" if n == 2 else f"{n} times"


def group_sentence(group: dict[str, Any]) -> str:
    """One group in words: its copies, then how the rest follow, then what to do."""
    copies: dict[str, list[str]] = {}
    copied: dict[str, str] = {}
    for x, y, found in group["relations"]:
        if found["kind"] == "copy" and y not in copied:
            original = copied.get(x, x)
            copied[y] = original
            copies.setdefault(original, []).append(y)
    clauses = []
    for original, names in copies.items():
        verb = "are copies" if len(names) > 1 else "is a copy"
        clauses.append(f"{listing(names, 4)} {verb} of {original}")
    # The other relations, exact ones first, each joining features not yet joined.
    joined = {name: copied.get(name, name) for name in group["features"]}
    others = sorted(
        (item for item in group["relations"] if item[2]["kind"] != "copy"),
        key=lambda item: item[2]["kind"] == "near",
    )
    for x, y, found in others:
        if joined[x] == joined[y]:
            continue
        old, new = joined[y], joined[x]
        joined = {name: new if mine == old else mine for name, mine in joined.items()}
        if found["kind"] == "derived":
            clauses.append(f"{y} = {_affine(x, found)}")
        else:
            clauses.append(f"{x} and {y} correlate at r = {_r(found['r'])}")
    if len(clauses) > _RELATIONS:
        clauses = [*clauses[:_RELATIONS], f"{len(clauses) - _RELATIONS} more relations"]
    exact = all(found["kind"] != "near" for _, _, found in group["relations"])
    what = "one measurement" if exact else "close to one measurement,"
    n = len(group["features"])
    return (
        f"{NAME}: {_joined(clauses)}: {what} counted {_times(n)} in each test family; "
        "keep one of them."
    )


def _joined(clauses: list[str]) -> str:
    """``a``, ``a, and b``, ``a, b, and c``: the clauses hold "and" of their own."""
    if len(clauses) == 1:
        return clauses[0]
    return f"{', '.join(clauses[:-1])}, and {clauses[-1]}"


def _clause(x: str, y: str, relation: dict[str, Any]) -> str:
    """One relation in words: ``y is a copy of x``, ``y = 1 − x``, ``x and y correlate``."""
    if relation["kind"] == "copy":
        return f"{y} is a copy of {x}"
    if relation["kind"] == "derived":
        return f"{y} = {_affine(x, relation)}"
    return f"{x} and {y} correlate at r = {_r(relation['r'])}"


def _any_coefficients(x: str, y: str, relation: dict[str, Any]) -> str:
    """A relation whose coefficients differ between experiments, in words."""
    if relation["kind"] == "near":
        return f"{x} and {y} correlate at |r| ≥ {NEAR_R:g}"
    return f"{y} = a × {x} + b"


def place_sentence(place: dict[str, Any], pooled: list[dict[str, Any]]) -> str | None:
    """Relations that hold in some experiments only, or in each with other coefficients, in
    words (``None`` when every one joins features already joined on all rows, ``pooled``)."""
    joined = {name: group["features"][0] for group in pooled for name in group["features"]}
    clauses, roots = [], set()
    for x, y, relation in sorted(place["relations"], key=lambda item: item[2]["kind"] == "near"):
        old, new = joined.get(y, y), joined.get(x, x)
        if old == new:
            continue
        joined = {name: new if mine == old else mine for name, mine in joined.items()}
        joined.setdefault(y, new)
        joined.setdefault(x, new)
        roots = {new if root == old else root for root in roots} | {new}
        words = _any_coefficients if place["everywhere"] else _clause
        clauses.append(words(x, y, relation))
    if not clauses:
        return None
    if len(clauses) > _RELATIONS:
        clauses = [*clauses[:_RELATIONS], f"{len(clauses) - _RELATIONS} more relations"]
    if place["everywhere"]:
        # One measurement within each experiment, each scaled its own way: counted again in
        # each experiment's test family, and the experiments were processed differently.
        n = sum(root in roots for root in joined.values())
        exact = all(found["kind"] != "near" for _, _, found in place["relations"])
        what = "one measurement" if exact else "close to one measurement,"
        return (
            f"{NAME}: within each experiment, {_joined(clauses)}, a and b differing between "
            f"experiments: {what} counted {_times(n)} in each test family; keep one of them. "
            "The experiments may have been processed differently."
        )
    return (
        f"{NAME}: in {listing(place['conditions'], 6)} only, {_joined(clauses)}; elsewhere "
        "they are not one measurement, so the experiments may have been processed differently."
    )


def relation_checks(
    data: ValidatedInput,
    experiments: Mapping[Any, Sequence[Any]] | None = None,
    condition_col: str = CONDITION_COLUMN,
) -> tuple[str, ...]:
    """One sentence per group of related features on all rows, then, with several
    ``experiments`` (each control's conditions, ``workflows._contrasts.experiments``), one
    per set of experiments a relation holds in alone (see the module); empty when none."""
    groups = related_features(data)
    sentences = [group_sentence(group) for group in groups[:_GROUPS]]
    rest = groups[_GROUPS:]
    if rest:
        firsts = [group["features"][0] for group in rest]
        sentences.append(
            f"{NAME}: {len(rest)} more groups, starting with {listing(firsts, 4)}: each "
            "repeats one measurement in each test family; keep one feature of each."
        )
    if experiments:
        places = experiment_relations(data, experiments, condition_col)
        said = [place_sentence(place, groups) for place in places[:_GROUPS]]
        sentences += [sentence for sentence in said if sentence]
    return tuple(sentences)


__all__ = [
    "MIN_ROWS",
    "NAME",
    "NEAR_R",
    "TOLERANCE",
    "experiment_relations",
    "group_sentence",
    "place_sentence",
    "related_features",
    "relation_checks",
]
