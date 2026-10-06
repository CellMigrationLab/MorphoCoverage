"""P5, multiple testing: the false discovery rate of the per-feature verdicts
(docs/common/validation.md, module P5).

Aim: choose the correction family. P1 fixes the per-feature test (the paired t on the repeat
log ratios), so P5 simulates at that level: for ``conditions`` conditions against one shared
control, ``features`` features split over ``tables`` feature tables and ``repeats``
repeats, each repeat's log ratio ``d = delta + e_condition - e_control``, the noise of one
arm standard normal per feature with an equicorrelation ``correlation`` across features (one
shared factor) and the control's multiplied by ``control_sd`` (a shared control of few
wells); the conditions sharing the control are correlated through it, as in the report. A
share ``nonnull`` of each condition's features (drawn at random) carries an effect of
``effect`` SDs of ``d`` with a random sign, spread over repeats by ``effect_sd``. Two more
families of ``extra_tests`` null tests each stand for the report's other families (the
whole-table tests, the comparisons across experiments), corrected apart as the report does.

Methods: ``bh_all`` (ours: one BH family over every feature and condition of the
sub-experiment), ``bh_per_table``, ``by_all``, ``weighted_bh`` (the readout, feature 0,
weighted ``READOUT_WEIGHT``, fixed in advance) and ``hierarchical`` (Benjamini-Bogomolov over
the tables). Outputs: ``fdp`` (false over rejected, 0 without rejections; its mean is the
FDR, the gate), ``tdp`` (true rejections over non-null tests; the objective), ``fwer`` (any
false rejection), ``discoveries`` and ``report_false`` (false rejections across every family
of the report).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .candidates import multiplicity
from .runner import Method, Module, Point, Rule, main

READOUT_WEIGHT = 4.0
Q = 0.05
DEFAULTS: dict[str, Any] = {
    "repeats": 4,
    "conditions": 3,
    "features": 10,
    "tables": 3,
    "nonnull": 0.0,
    "effect": 2.5,
    "effect_sd": 0.0,
    "correlation": 0.0,
    "control_sd": 1.0,
    "extra_tests": 5,
}


@dataclass(frozen=True)
class Tests:
    """One simulated report: ``p`` and ``nonnull`` (conditions x features), each feature's
    table, and the p-values of the other families."""

    p: np.ndarray
    nonnull: np.ndarray
    tables: np.ndarray
    extra: tuple[np.ndarray, ...]


def _noise(rng: np.random.Generator, shape: tuple[int, ...], correlation: float) -> np.ndarray:
    """Standard normal noise, equicorrelated along the last axis (one shared factor)."""
    shared = rng.standard_normal((*shape[:-1], 1))
    own = rng.standard_normal(shape)
    return np.sqrt(correlation) * shared + np.sqrt(1 - correlation) * own


def simulate(params: dict[str, Any], rng: np.random.Generator) -> Tests:
    """One report's p-values (see the module)."""
    s = {**DEFAULTS, **params}
    n, k, m = int(s["repeats"]), int(s["conditions"]), int(s["features"])
    scale = np.sqrt(1 + s["control_sd"] ** 2)
    nonnull = np.zeros((k, m), dtype=bool)
    carrying = round(s["nonnull"] * m)
    for condition in range(k):
        nonnull[condition, rng.choice(m, size=carrying, replace=False)] = True
    sign = rng.choice((-1.0, 1.0), size=(k, m))
    delta = np.where(nonnull, s["effect"] * scale * sign, 0.0)
    control = s["control_sd"] * _noise(rng, (n, 1, m), s["correlation"])
    arms = _noise(rng, (n, k, m), s["correlation"])
    spread = s["effect_sd"] * scale * rng.standard_normal((n, k, m)) * nonnull
    d = delta + spread + arms - control
    t = d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    tables = np.arange(m) * int(s["tables"]) // m
    extra = tuple(rng.random(int(s["extra_tests"])) for _ in range(2))
    return Tests(p, nonnull, tables, extra)


def _outputs(rejected: np.ndarray, data: Tests) -> dict[str, float]:
    false = int((rejected & ~data.nonnull).sum())
    true = int((rejected & data.nonnull).sum())
    found = false + true
    others = sum(int(multiplicity.bh(e, Q).sum()) for e in data.extra)
    return {
        "fdp": false / found if found else 0.0,
        "tdp": true / data.nonnull.sum() if data.nonnull.any() else np.nan,
        "fwer": float(false > 0),
        "discoveries": float(found),
        "report_false": float(false + others),
    }


def bh_all(data: Tests, context: dict[str, Any]) -> dict[str, float]:
    """Ours: one BH family over every feature and condition."""
    return _outputs(multiplicity.bh(data.p.ravel(), Q).reshape(data.p.shape), data)


def bh_per_table(data: Tests, context: dict[str, Any]) -> dict[str, float]:
    """BH within each table (over its features and every condition)."""
    families = np.broadcast_to(data.tables, data.p.shape).ravel()
    return _outputs(
        multiplicity.per_family(data.p.ravel(), families, Q).reshape(data.p.shape), data
    )


def by_all(data: Tests, context: dict[str, Any]) -> dict[str, float]:
    """Benjamini-Yekutieli over one family (reference)."""
    return _outputs(multiplicity.by(data.p.ravel(), Q).reshape(data.p.shape), data)


def weighted_bh(data: Tests, context: dict[str, Any]) -> dict[str, float]:
    """Weighted BH, the readout weighted ``READOUT_WEIGHT`` (fixed in advance)."""
    weights = np.ones(data.p.shape)
    weights[:, 0] = READOUT_WEIGHT
    rejected = multiplicity.weighted_bh(data.p.ravel(), weights.ravel(), Q)
    return _outputs(rejected.reshape(data.p.shape), data)


def hierarchical(data: Tests, context: dict[str, Any]) -> dict[str, float]:
    """Benjamini-Bogomolov over the tables."""
    families = np.broadcast_to(data.tables, data.p.shape).ravel()
    rejected = multiplicity.hierarchical(data.p.ravel(), families, Q)
    return _outputs(rejected.reshape(data.p.shape), data)


METHODS = (
    Method("bh_all", bh_all, role="ours", simplicity=1, primary="fdp"),
    Method("bh_per_table", bh_per_table, simplicity=1, primary="fdp"),
    Method("by_all", by_all, role="reference", simplicity=1, primary="fdp"),
    Method("weighted_bh", weighted_bh, simplicity=2, primary="fdp"),
    Method("hierarchical", hierarchical, simplicity=3, primary="fdp"),
)


def points() -> tuple[Point, ...]:
    """The design points of P5: every core point is gated (FDR at every share of
    non-nulls); those with non-nulls are also a class of the objective."""
    out = []
    for share in (0.0, 0.2, 0.5, 0.8):
        for features in (10, 100):
            for correlation in (0.0, 0.5):
                params = {"nonnull": share, "features": features, "correlation": correlation}
                name = f"core_pi{int(share * 100)}_m{features}_rho{int(correlation * 10)}"
                label = f"nonnull_{int(share * 100)}" if share else None
                out.append(Point(name, "core", params, True, label))
    stress = {
        "thirty_conditions": {"conditions": 30, "nonnull": 0.2},
        "thousand_features": {"features": 1000, "nonnull": 0.2, "tables": 5},
        "three_repeats": {"repeats": 3, "nonnull": 0.2},
        "near_duplicates": {"correlation": 0.9, "nonnull": 0.2},
        "shared_control_few_wells": {"control_sd": 2.0, "nonnull": 0.2},
        "heterogeneous_effects": {"effect_sd": 1.0, "nonnull": 0.2},
        "ten_tables": {"features": 100, "tables": 10, "nonnull": 0.2},
    }
    out += [Point(f"stress_{name}", "stress", change) for name, change in stress.items()]
    return tuple(out)


MODULE = Module(
    name="p5",
    title="multiple testing",
    simulate=simulate,
    points=points(),
    methods=METHODS,
    rules=(
        Rule(
            "main",
            gate="fdp",
            gate_kind="mean",
            coverage=False,
            objective="tdp",
            gate_alternatives=True,
        ),
    ),
)

if __name__ == "__main__":
    main("decisions.p5_multiple_testing", sys.argv[1:])
