"""P4, sequential mean sufficiency: "Did I image enough?" for a mean, as the full sequential
process the microscope runs (docs/common/validation.md, module P4).

Aim: choose the interval behind the mean-precision stop. One sample (one condition in one
repeat, with its wells) is acquired field by field (``sequential``); at each look (after
``FIRST`` fields, then every ``EVERY``, up to ``MAX_FIELDS``) a rule computes an interval for
the sample's cell-weighted mean and stops when its relative half-width is at most
``TOLERANCE``. Outputs: ``stopped``, ``fields`` (at the stop, else all), ``false_stop`` (a
stop whose estimate misses the sample's true mean by more than the tolerance; the gate),
``covered`` (the truth inside the interval at the stop) and ``relative_error``. ``MAX_FIELDS``
is per well.

Methods: ``nested_bootstrap`` (ours: the package's wells-then-fields bootstrap, widened),
``cluster_se``, ``two_stage_bootstrap``, ``block_bootstrap``, ``bonferroni_looks``,
``spending`` and ``confidence_sequence`` (``candidates.sequential_rules``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from .candidates import sequential_rules as rules
from .design import CONTROL, design_from
from .runner import Method, Module, Point, Rule, main
from .scalar import scalar_from
from .sequential import Sequence, schedule, sequence

TOLERANCE = 0.05
FIRST, EVERY, MAX_FIELDS = 5, 5, 60
N_BOOT = 200
CENTRE: dict[str, Any] = {
    "repeats": 1,
    "conditions": (CONTROL,),
    "fields": MAX_FIELDS,
    "distribution": "lognormal",
    "cv": 0.3,
}


def simulate(params: dict[str, Any], rng: np.random.Generator) -> Sequence:
    """One sample acquired in order."""
    interleave = bool(params.get("interleave", True))
    return sequence(design_from(params), scalar_from(params), rng, interleave)


Interval = Callable[[list[np.ndarray], Sequence, int, dict[str, Any]], tuple[float, float, float]]


def _process(interval: Interval) -> Callable[[Sequence, dict[str, Any]], dict[str, float]]:
    """The sequential process of one interval rule (see the module)."""

    def run(data: Sequence, context: dict[str, Any]) -> dict[str, float]:
        looks = schedule(len(data.order), FIRST, EVERY)
        estimate = low = high = np.nan
        fields, stopped = len(data.order), False
        for index, k in enumerate(looks):
            here = {**context, "look": index, "looks": looks}
            estimate, low, high = interval(data.field_values(k), data, k, here)
            half = max(estimate - low, high - estimate) if np.isfinite(low) else np.inf
            if np.isfinite(half) and estimate > 0 and half <= TOLERANCE * estimate:
                fields, stopped = k, True
                break
        error = abs(estimate - data.truth) / data.truth if np.isfinite(estimate) else np.nan
        return {
            "stopped": float(stopped),
            "fields": float(fields),
            "false_stop": float(stopped and error > TOLERANCE),
            "covered": float(low <= data.truth <= high) if np.isfinite(low) else np.nan,
            "relative_error": error,
        }

    return run


def _boot(context: dict[str, Any]) -> int:
    return 20 if context["smoke"] else N_BOOT


def nested(values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]) -> tuple:
    """Ours: the package's nested bootstrap."""
    wells = data.recorded_wells(k)
    return rules.nested_bootstrap(values, wells, context["rng"], _boot(context))


def cluster(values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]) -> tuple:
    """The ratio-estimator SE, at every look at 95%."""
    return rules.cluster_se(values)


def two_stage(values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]) -> tuple:
    """Wells then fields, with replacement."""
    return rules.two_stage_bootstrap(values, data.recorded_wells(k), context["rng"], _boot(context))


def block(values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]) -> tuple:
    """Moving blocks along the acquisition order."""
    return rules.block_bootstrap(values, context["rng"], _boot(context))


def bonferroni(values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]) -> tuple:
    """The cluster SE at ``1 - alpha / K`` over the planned looks."""
    return rules.cluster_se(values, 1 - context["alpha"] / len(context["looks"]))


def spending(values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]) -> tuple:
    """The cluster SE at the O'Brien-Fleming increment of this look."""
    levels = rules.spending_levels(context["looks"], context["looks"][-1], context["alpha"])
    return rules.cluster_se(values, levels[context["look"]])


def sequence_bound(
    values: list[np.ndarray], data: Sequence, k: int, context: dict[str, Any]
) -> tuple:
    """The normal-mixture confidence sequence."""
    return rules.confidence_sequence(values, context["alpha"])


METHODS = tuple(
    Method(name, _process(function), role=role, simplicity=simplicity, primary="fields")
    for name, function, role, simplicity in (
        ("nested_bootstrap", nested, "ours", 2),
        ("cluster_se", cluster, "candidate", 1),
        ("two_stage_bootstrap", two_stage, "candidate", 2),
        ("block_bootstrap", block, "candidate", 3),
        ("bonferroni_looks", bonferroni, "candidate", 1),
        ("spending", spending, "candidate", 2),
        ("confidence_sequence", sequence_bound, "candidate", 3),
    )
)

DEPARTURES = {
    "centre": {},
    "two_wells": {"wells": 2},
    "cells5": {"cells": 5.0},
    "cells80": {"cells": 80.0},
    "icc_low": {"icc_field": 0.02},
    "icc_high": {"icc_field": 0.3},
    "unbalanced": {"cell_dispersion": 1.0},
    "skewed": {"cv": 0.8},
}
STRESS = {
    "unrecorded_wells": {"wells": 2, "wells_recorded": False, "icc_well": 0.15},
    "autocorrelated": {"autocorrelation": 0.6},
    "informative_size": {"informative_size": 0.8},
    "mnar": {"missing": "mnar", "missing_rate": 0.3},
    "empty_fields": {"empty_fields": 0.2},
    "empty_fields_absent": {"empty_fields": 0.2, "empty_recorded": False},
    "heavy_tails": {"distribution": "t3"},
    "sequential_wells": {"wells": 2, "interleave": False},
}


def points() -> tuple[Point, ...]:
    """The design points of P4: each core point is gated and is its own class."""
    out = [
        Point(f"core_{name}", "core", {**CENTRE, **change}, True, name)
        for name, change in DEPARTURES.items()
    ]
    out += [Point(f"stress_{n}", "stress", {**CENTRE, **c}) for n, c in STRESS.items()]
    return tuple(out)


MODULE = Module(
    name="p4",
    title="sequential mean sufficiency",
    simulate=simulate,
    points=points(),
    methods=METHODS,
    rules=(
        Rule(
            "main",
            gate="false_stop",
            coverage=False,
            objective="fields",
            maximise=False,
            gate_alternatives=True,
            tie=1.0,
        ),
    ),
)

if __name__ == "__main__":
    main("decisions.p4_sequential", sys.argv[1:])
