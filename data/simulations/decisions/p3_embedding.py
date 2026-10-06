"""P3, whole-embedding inference (docs/common/validation.md, module P3).

Aim: choose the whole-table test (mean shift and, separately, distribution) and the
centring of embedding tables, after the hard gate of rotation invariance: a global
orthogonal change of the embedding's basis must change no p-value and move centred values
only by the same rotation. Data: ``embedding`` on ``design``; with the environment variable
``DECISIONS_REAL_EMBEDDING`` naming a CSV of real control observations (columns
``biological_repeat``, ``position_id`` and the embedding's ``e*`` columns), two
semi-synthetic points are added (real values, half the fields of each repeat relabelled and
shifted; ``embedding.semi_synthetic``).

Questions and methods:

- ``mean``: ``whole_table`` (ours: the package's whole-table repeat verdict,
  ``sampling_unit.whole_table``), ``loo_unscaled`` (its leave-one-repeat-out projection test
  on unscaled field means), ``u_normal`` and ``u_sign_flip`` (the paired high-dimensional
  U-statistic), ``pc_hotelling``, ``shrinkage_hotelling`` and ``min_p`` (non-invariant
  negative comparator);
- ``distribution``: ``permanova``, ``energy``, ``mmd`` (labels of fields permuted within
  repeats);
- ``centring``: ``coordinate_median`` (ours), ``centroid``, ``geometric_median``,
  ``vector_model``, ``location_scale`` and ``harmony`` (references): the error of the
  condition's pooled shift after centring (``biology_error``, in within-group SDs) and the
  repeats' remaining spread (``batch_residual``).

Each method's invariance (``invariant``) is checked on the first ``INVARIANCE_CHECKS``
replicates of every point: the method on the data rotated by a random orthogonal matrix.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN
from morphocoverage.sampling_unit.whole_table import _table_rows, repeat_test

from .candidates import centring, hd_tests
from .design import FIELD_COLUMN, design_from
from .embedding import EmbeddingData, embedding_from, generate, random_rotation, semi_synthetic
from .runner import Method, Module, Point, Rule, main

TREATED, CONTROL = "treated", "control"
INVARIANCE_CHECKS = 5
REAL = "DECISIONS_REAL_EMBEDDING"


@lru_cache(maxsize=2)
def _real(path: str) -> tuple[np.ndarray, pd.DataFrame]:
    table = pd.read_csv(path)
    columns = [c for c in table.columns if c.startswith("e")]
    return table[columns].to_numpy(dtype=float), table[[REPEAT_COLUMN, FIELD_COLUMN]]


def simulate(params: dict[str, Any], rng: np.random.Generator) -> EmbeddingData:
    """One dataset of a point (semi-synthetic when ``params["real"]``)."""
    if params.get("real"):
        values, meta = _real(os.environ[REAL])
        shifted, labelled, vector = semi_synthetic(values, meta, params.get("shift", 0.0), rng)
        columns = tuple(f"e{j}" for j in range(values.shape[1]))
        objects = pd.concat([labelled, pd.DataFrame(shifted, columns=list(columns))], axis=1)
        fovs = labelled.drop_duplicates(FIELD_COLUMN)
        truth = {
            "shift": vector,
            "shift_norm": float(np.linalg.norm(vector)),
            "mean_null": not np.any(vector),
            "distribution_null": not np.any(vector),
        }
        return EmbeddingData(objects, fovs, columns, truth, None)
    return generate(design_from(params), embedding_from(params), rng)


def _arrays(data: EmbeddingData, values: np.ndarray | None = None) -> dict[str, np.ndarray]:
    frame = data.objects
    return {
        "values": frame[list(data.columns)].to_numpy(dtype=float) if values is None else values,
        "condition": frame[CONDITION_COLUMN].astype(str).to_numpy(),
        "repeat": frame[REPEAT_COLUMN].astype(str).to_numpy(),
        "field": frame[FIELD_COLUMN].astype(str).to_numpy(),
    }


def _with_invariance(test: Any) -> Any:
    """A test returning a p, wrapped to check it on rotated data in the first replicates."""

    def run(data: EmbeddingData, context: dict[str, Any]) -> dict[str, float]:
        arrays = _arrays(data)
        seed = int(context["rng"].integers(2**32))
        p = test(arrays, context, np.random.default_rng(seed))
        out = {"p": p}
        if context["replicate"] < INVARIANCE_CHECKS:
            rotation = random_rotation(arrays["values"].shape[1], context["rng"])
            rotated = {**arrays, "values": arrays["values"] @ rotation}
            again = test(rotated, context, np.random.default_rng(seed))
            same = (np.isnan(p) and np.isnan(again)) or abs(p - again) <= 1e-6 * max(1.0, abs(p))
            out["invariant"] = float(same)
        return out

    run.__name__ = test.__name__
    return run


def _deltas(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    means, condition, repeat = hd_tests.field_means(
        arrays["values"], arrays["condition"], arrays["repeat"], arrays["field"]
    )
    delta, _ = hd_tests.deltas(means, condition, repeat, TREATED, CONTROL)
    return delta, hd_tests.residuals(means, condition, repeat)


def whole_table(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """Ours: the package's repeat-level verdict (columns scaled per comparison)."""
    codes = np.unique(arrays["field"], return_inverse=True)[1]
    meta = pd.DataFrame(
        {CONDITION_COLUMN: arrays["condition"], REPEAT_COLUMN: arrays["repeat"], "field": codes}
    )
    rows = _table_rows("table", arrays["values"], meta, [(TREATED, CONTROL)], 9, 0)
    return float(rows[0].get("p_repeats", np.nan))


def loo_unscaled(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """The package's projection test on unscaled field means."""
    return float(repeat_test(_deltas(arrays)[0])[3])


def u_normal(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """The paired U-statistic, normal p."""
    return hd_tests.u_statistic(_deltas(arrays)[0])["p"]


def u_sign_flip(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """The paired U-statistic, exact sign-flip p."""
    return hd_tests.sign_flip_u(_deltas(arrays)[0])


def pc_hotelling(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """Hotelling on the first n - 2 principal components."""
    return hd_tests.pc_hotelling(*_deltas(arrays))


def shrinkage_hotelling(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """Ledoit-Wolf Hotelling, sign-flip p."""
    return hd_tests.shrinkage_hotelling(*_deltas(arrays))


def min_p(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
    """Per-coordinate minimum p, Bonferroni (non-invariant comparator)."""
    return hd_tests.min_p(_deltas(arrays)[0])


def _permutation(kind: str) -> Any:
    def test(arrays: dict[str, np.ndarray], context: dict[str, Any], rng: Any) -> float:
        means, condition, repeat = hd_tests.field_means(
            arrays["values"], arrays["condition"], arrays["repeat"], arrays["field"]
        )
        keep = np.isin(condition, [TREATED, CONTROL])
        side = (condition[keep] == TREATED).astype(float)
        permutations = 19 if context["smoke"] else hd_tests.PERMUTATIONS
        return hd_tests.permutation_test(means[keep], side, repeat[keep], kind, rng, permutations)

    test.__name__ = kind
    return test


def _centring(name: str) -> Any:
    """A centring as a method: the biology's error and the remaining batch (see the module)."""
    function = centring.CENTRINGS[name]

    def run(data: EmbeddingData, context: dict[str, Any]) -> dict[str, float]:
        arrays = _arrays(data)
        values, condition, repeat = arrays["values"], arrays["condition"], arrays["repeat"]
        centred = function(values, condition, repeat)
        within = float(
            np.sqrt(np.square(values - _group_means(values, condition, repeat)).sum(1).mean())
        )
        shift = centred[condition == TREATED].mean(axis=0) - centred[condition == CONTROL].mean(
            axis=0
        )
        truth = data.truth["shift"]
        out = {
            "biology_error": float(np.linalg.norm(shift - truth) / within),
            "batch_residual": _batch(centred, condition, repeat) / within**2,
        }
        if context["replicate"] < INVARIANCE_CHECKS:
            rotation = random_rotation(values.shape[1], context["rng"])
            again = function(values @ rotation, condition, repeat)
            out["invariant"] = float(np.allclose(again, centred @ rotation, atol=1e-8 * within))
        return out

    run.__name__ = name
    return run


def _group_means(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    keys = np.char.add(condition, "\x1f" + repeat)
    for key in np.unique(keys):
        out[keys == key] = values[keys == key].mean(axis=0)
    return out


def _batch(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> float:
    """The mean over conditions of the summed variance of their repeat centroids."""
    spreads = []
    for c in np.unique(condition):
        centres = [
            values[(condition == c) & (repeat == r)].mean(axis=0)
            for r in np.unique(repeat[condition == c])
        ]
        if len(centres) > 1:
            spreads.append(float(np.var(np.array(centres), axis=0, ddof=1).sum()))
    return float(np.mean(spreads)) if spreads else np.nan


MEAN = (
    ("whole_table", whole_table, "ours", 1),
    ("loo_unscaled", loo_unscaled, "candidate", 1),
    ("u_normal", u_normal, "candidate", 2),
    ("u_sign_flip", u_sign_flip, "candidate", 2),
    ("pc_hotelling", pc_hotelling, "candidate", 3),
    ("shrinkage_hotelling", shrinkage_hotelling, "candidate", 3),
    ("min_p", min_p, "comparator", 2),
)
METHODS = (
    *(Method(n, _with_invariance(f), role=r, question="mean", simplicity=s) for n, f, r, s in MEAN),
    *(
        Method(k, _with_invariance(_permutation(k)), question="distribution", simplicity=s)
        for k, s in (("permanova", 1), ("energy", 2), ("mmd", 3))
    ),
    *(
        Method(n, _centring(n), role=r, question="centring", simplicity=s, primary="biology_error")
        for n, r, s in (
            ("coordinate_median", "ours", 1),
            ("centroid", "candidate", 1),
            ("geometric_median", "candidate", 2),
            ("vector_model", "candidate", 2),
            ("location_scale", "reference", 3),
            ("harmony", "reference", 4),
        )
    ),
)

BASE: dict[str, Any] = {"dims": 32, "batch": "additive", "batch_size": 1.0}
UNITS = {"cells": {"cells": 20.0}, "image": {"cells": 1.0, "fixed_cells": True, "fields": 6}}


def points() -> tuple[Point, ...]:
    """The design points of P3."""
    out = []
    for n in (3, 4, 6):
        for dims in (32, 128):
            for unit, change in UNITS.items():
                params = {**BASE, "repeats": n, "dims": dims, **change}
                out.append(Point(f"core_null_n{n}_d{dims}_{unit}", "core", params))
    alternatives = {
        "dense": [{"shift": s, "shift_kind": "dense"} for s in (0.3, 0.6)],
        "sparse": [{"shift": 0.6, "shift_kind": "sparse"}],
        "aligned": [{"shift": 0.6, "shift_kind": "aligned"}],
        "covariance": [{"shift": 1.0, "shift_kind": "covariance"}],
    }
    for label, settings in alternatives.items():
        for n in (4, 6):
            for i, change in enumerate(settings):
                params = {**BASE, "repeats": n, **UNITS["cells"], **change}
                null = label == "covariance"  # a null for the mean, an alternative for the rest
                out.append(Point(f"core_{label}_n{n}_{i}", "core", params, null, label))
    for batch in ("additive", "scale", "rotation", "nonlinear"):
        params = {**BASE, "repeats": 4, "batch": batch, "shift": 0.6, **UNITS["cells"]}
        out.append(Point(f"core_batch_{batch}", "core", params, False, f"batch_{batch}"))
    incomplete = {
        **BASE,
        "repeats": 4,
        "conditions": ("control", "treated", "other"),
        "presence": (("control", "treated", "other"),) * 3 + (("control", "treated"),),
        "shift": 0.6,
    }
    out.append(Point("core_batch_incomplete", "core", incomplete, False, "batch_incomplete"))
    stress = {
        "rotation_batch": {"batch": "rotation", "batch_size": 2.0},
        "scale_batch": {"batch": "scale", "batch_size": 2.0},
        "nonlinear_batch": {"batch": "nonlinear", "batch_size": 2.0},
        "unrecorded_wells": {"wells": 2, "wells_recorded": False, "icc_well": 0.15},
        "low_rank": {"dims": 128, "rank": 5},
        "wide": {"dims": 384},
        "unbalanced": {"field_imbalance": 0.8, "cell_dispersion": 1.5},
    }
    for name, change in stress.items():
        out.append(Point(f"stress_{name}", "stress", {**BASE, "repeats": 4, **change}))
    if os.environ.get(REAL):
        out.append(Point("core_null_semi", "core", {"real": True, "shift": 0.0}))
        out.append(Point("core_dense_semi", "core", {"real": True, "shift": 0.6}, False, "dense"))
    return tuple(out)


MEAN_CLASSES = ("dense", "sparse", "aligned")
MODULE = Module(
    name="p3",
    title="whole-embedding inference",
    simulate=simulate,
    points=points(),
    methods=METHODS,
    rules=(
        Rule("mean", classes=MEAN_CLASSES, hard=("invariant",), coverage=False),
        Rule(
            "distribution",
            classes=(*MEAN_CLASSES, "covariance"),
            hard=("invariant",),
            coverage=False,
        ),
        Rule(
            "centring",
            gate="",
            coverage=False,
            hard=("invariant",),
            objective="biology_error",
            maximise=False,
            classes=(
                "batch_additive",
                "batch_scale",
                "batch_rotation",
                "batch_nonlinear",
                "batch_incomplete",
            ),
        ),
    ),
)

if __name__ == "__main__":
    main("decisions.p3_embedding", sys.argv[1:])
