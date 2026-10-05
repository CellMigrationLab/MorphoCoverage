"""Unusual objects: which leave-out keeps the false-alarm rate at the share asked for?

``sampling_unit.unusual_objects`` scores an observation by its mean distance to the ``K``
nearest reference observations, after standardizing on the reference. The threshold is the
``1 - quantile`` quantile of the reference observations' scores, each scored with its own
reference group left out (with a single reference group, its own field), so a tested
observation is scored with one reference group (field) left out too. Issues #99 and #286: the
calibration leaves a group out in proportion to its observations (each observation leaves out
its own group), while tested observations left a group out uniformly, and with a single
reference group they were scored against all of it, a denser pool. This compares, under the
null (the tested group drawn like the reference groups, a group of its own), the share of
tested observations called unusual for:

- ``current``: the calibration as above; tested observations leave a reference group out
  drawn uniformly, or with a single reference group none (scored against every field);
- ``size_weighted``: the same calibration; tested observations leave out a reference group
  (with a single reference group, a field) drawn in proportion to its observations, the
  weighting the calibration has;
- ``uniform``: the calibration's scores weighted so every reference group (field) weighs the
  same, the threshold their weighted quantile (``weighted_threshold``, ``np.quantile`` when the
  units are of one size); tested observations leave out a group (field) drawn uniformly;
- ``package``: ``morphocoverage.sampling_unit._outliers.score`` itself on the same data.

Data: two features; an observation is ``shift_g + field_f + noise`` with ``field_f ~ N(0,
FIELD_SD)`` per field and ``noise ~ N(0, 1)``; with a repeat shift, ``shift_g ~ N(0, 0.5)`` per
group, the tested group drawing its own. Designs (``DESIGNS``): reference fields per group, and
``CELLS`` observations per field or a uniform draw from 5 to 80 ("unequal"); the tested group
has ``TESTED_FIELDS`` fields drawn the same way. Each setting is ``N_REPLICATES`` datasets; the
rate is the mean over them of the tested group's share flagged, ``se`` the largest standard
error of the four.

Results: ``results/unusual_calibration.csv``. Usage, from the repository root::

    PYTHONPATH=src python data/simulations/unusual_calibration.py [workers]
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from morphocoverage.sampling_unit._outliers import score
from morphocoverage.transforms import Standardize

K = 5
FIELD_SD = 0.3
CELLS = 40
TESTED_FIELDS = 10
N_REPLICATES = 400
QUANTILES = (0.05, 0.01)
SHIFTS = (0.0, 0.5)
SCHEMES = ("current", "size_weighted", "uniform", "package")
DESIGNS = {
    "3 groups, equal": ((10, 10, 10), False),
    "3 groups, unequal": ((3, 10, 30), False),
    "4 groups, unequal": ((2, 4, 8, 16), False),
    "6 groups, equal": ((10,) * 6, False),
    "6 groups, unequal": ((2, 3, 5, 10, 20, 40), False),
    "3 groups, unequal fields": ((10, 10, 10), True),
    "1 group, 10 fields": ((10,), False),
    "1 group, 4 fields": ((4,), False),
    "1 group, 10 unequal fields": ((10,), True),
}
RESULTS = Path(__file__).resolve().parent / "results"


def one_group(rng: np.random.Generator, fields: int, unequal: bool, shift: float) -> list:
    """The fields of one group, each an array of observations (rows) by 2 features."""
    offset = rng.normal(0, shift, 2) if shift else np.zeros(2)
    sizes = rng.integers(5, 81, fields) if unequal else np.full(fields, CELLS)
    return [offset + rng.normal(0, FIELD_SD, 2) + rng.normal(0, 1, (n, 2)) for n in sizes]


def left_out_scores(
    ref: np.ndarray, units: np.ndarray, tested: np.ndarray, left_out: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Reference scores with their own unit left out; tested scores with ``left_out`` (one unit
    index per tested observation) left out, or against every unit when it is None."""
    reference_scores = np.empty(len(ref))
    tested_scores = np.empty(len(tested))
    for unit in np.unique(units):
        inside = units == unit
        tree = cKDTree(ref[~inside])
        reference_scores[inside] = tree.query(ref[inside], k=K)[0].mean(axis=1)
        if left_out is not None and (left_out == unit).any():
            chosen = left_out == unit
            tested_scores[chosen] = tree.query(tested[chosen], k=K)[0].mean(axis=1)
    if left_out is None:
        tested_scores = cKDTree(ref).query(tested, k=K)[0].mean(axis=1)
    return reference_scores, tested_scores


def weighted_threshold(scores: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """The ``1 - quantile`` quantile of ``scores`` weighted by ``weights``: ``np.quantile``'s
    linear interpolation, each score at ``(S_i - w_i) / (S_n - w_n)`` with ``S`` the cumulative
    weights (``(i - 1) / (n - 1)``, ``np.quantile`` itself, when the weights are equal)."""
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    positions = (cumulative - weights[order]) / (cumulative[-1] - weights[order][-1])
    return float(np.interp(1 - quantile, positions, scores[order]))


def one_dataset(rng: np.random.Generator, design: tuple, shift: float, seed: int) -> dict:
    """The share of the tested group flagged by each scheme, at each quantile."""
    fields, unequal = design
    groups = [one_group(rng, n, unequal, shift) for n in fields]
    tested = np.concatenate(one_group(rng, TESTED_FIELDS, unequal, shift))
    reference = np.concatenate([np.concatenate(g) for g in groups])
    group_ids = np.concatenate([np.full(sum(map(len, g)), i) for i, g in enumerate(groups)])
    fov_ids = np.concatenate(
        [np.full(len(f), j) for j, f in enumerate(f for g in groups for f in g)]
    )
    state, _ = Standardize().fit_matrix(reference, ("a", "b"))
    ref = Standardize().apply_matrix(state, reference)
    points = Standardize().apply_matrix(state, tested)
    units = group_ids if len(groups) > 1 else fov_ids
    sizes = np.bincount(units).astype(float)
    uniform_draw = rng.integers(0, len(sizes), len(points))
    sized_draw = rng.choice(len(sizes), size=len(points), p=sizes / sizes.sum())
    current = left_out_scores(ref, units, points, uniform_draw if len(groups) > 1 else None)
    sized = left_out_scores(ref, units, points, sized_draw)
    uniform = left_out_scores(ref, units, points, uniform_draw)
    out = {}
    for quantile in QUANTILES:
        flagged = {
            "current": current[1] > np.quantile(current[0], 1 - quantile),
            "size_weighted": sized[1] > np.quantile(sized[0], 1 - quantile),
            "uniform": uniform[1] > weighted_threshold(uniform[0], 1 / sizes[units], quantile),
        }
        scored = score(
            reference, fov_ids, group_ids, tested, K, quantile, np.random.default_rng(seed)
        )
        flagged["package"] = scored[0] > scored[1]
        for scheme, calls in flagged.items():
            out[(quantile, scheme)] = float(np.mean(calls))
    return out


def run(setting: tuple) -> list[dict]:
    """Every replicate of one design and shift: one row per quantile."""
    name, shift, seed = setting
    rng = np.random.default_rng(seed)
    shares = [one_dataset(rng, DESIGNS[name], shift, seed * 1000 + i) for i in range(N_REPLICATES)]
    fields, unequal = DESIGNS[name]
    rows = []
    for quantile in QUANTILES:
        values = {s: np.array([x[(quantile, s)] for x in shares]) for s in SCHEMES}
        rows.append(
            {
                "design": name,
                "fields": "/".join(map(str, fields)),
                "cells": "5-80" if unequal else str(CELLS),
                "shift": shift,
                "quantile": quantile,
                **{s: float(v.mean()) for s, v in values.items()},
                "se": max(float(v.std(ddof=1) / np.sqrt(len(v))) for v in values.values()),
            }
        )
    return rows


def main() -> None:
    """Every design and shift; the table is written and printed."""
    settings = [
        (name, shift, 9900 + i)
        for i, (name, shift) in enumerate((n, s) for n in DESIGNS for s in SHIFTS)
    ]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        rows = [row for result in pool.map(run, settings) for row in result]
    table = pd.DataFrame(rows).sort_values(
        ["quantile", "shift"], ascending=[False, True], kind="stable"
    )
    table = table.round(4).reset_index(drop=True)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "unusual_calibration.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
