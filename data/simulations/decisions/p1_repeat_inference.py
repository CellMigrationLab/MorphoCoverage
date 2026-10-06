"""P1, core repeat inference: one condition against its control, across biological repeats
(docs/common/validation.md, module P1).

Aim: choose the test and scale of the per-feature verdict ("significant across repeats")
for the default estimand, the cell-weighted repeat mean (owner, 2026-10-06), and report the
field- and well-balanced estimands beside it. Data: ``scalar`` on ``design`` (one feature,
two conditions). Methods (the package's where it has them):

- ``paired_t_auto`` (ours): the paired t-test on the repeat means, log ratios when every
  value of the feature is positive, else differences (``statistics.log_scale``,
  ``repeat_values``, ``paired_t``), with its t interval;
- ``paired_t_declared``: the same on the scale the data were generated on (a declared scale);
- ``paired_t_difference``: always differences;
- ``fixed_block``: the candidate engine on the per-repeat summaries (equal to the paired
  test in a complete two-condition design, by construction: a check of the engine);
- ``sign_flip`` (reference): the package's exact sign-flip test on the same values;
- ``paired_t_field``, ``paired_t_well``: the field- and well-balanced estimands (their own
  questions);
- ``mixed`` (comparator, statsmodels; at most ``COMPARATOR_REPLICATES`` replicates a point)
  and ``pooled_cells`` (reference, ignores the hierarchy): reported, never selected.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from morphocoverage.sampling_unit.contrast_repeats import sign_flip_test
from morphocoverage.statistics import log_scale, paired_t, repeat_values

from .candidates import fixed_block, repeat_means
from .design import design_from, latin_hypercube
from .runner import Method, Module, Point, Rule, main
from .scalar import ScalarData, generate, scalar_from

FEATURE, TREATED, CONTROL = "x0", "treated", "control"
COMPARATOR_REPLICATES = 1_000
CENTRE: dict[str, Any] = {"distribution": "lognormal", "cv": 0.3}


def simulate(params: dict[str, Any], rng: np.random.Generator) -> ScalarData:
    """One dataset of a point."""
    return generate(design_from(params), scalar_from(params), rng)


def _auto(data: ScalarData) -> bool:
    return log_scale(data.objects[FEATURE].to_numpy(dtype=float))


def _truth(data: ScalarData, estimand: str, log: bool) -> float:
    return data.truth[FEATURE, TREATED, estimand, "ratio" if log else "difference"]


def _paired(data: ScalarData, estimand: str, log: bool, alpha: float) -> dict[str, float]:
    """The paired t on the repeat means of ``estimand``, on the log scale or not."""
    table = repeat_means.summaries(data.objects, FEATURE, estimand)
    wide = table.pivot_table(index="repeat", columns="condition", values="value").dropna()
    if TREATED not in wide or CONTROL not in wide:
        return {}
    _, values, _ = repeat_values(wide[TREATED].to_numpy(), wide[CONTROL].to_numpy(), log)
    out = repeat_means.t_summary(values, alpha)
    out["p"] = paired_t(values)
    out["truth"] = _truth(data, estimand, log)
    out["log"] = float(log)
    return out


def paired_t_auto(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """Ours: the package's rule (see the module)."""
    return _paired(data, "cell", _auto(data), context["alpha"])


def paired_t_declared(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """The scale the feature was generated on, declared."""
    declared = context["params"].get("scale", "ratio") == "ratio"
    return _paired(data, "cell", declared, context["alpha"])


def paired_t_difference(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """Always differences."""
    return _paired(data, "cell", False, context["alpha"])


def paired_t_field(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """Field-balanced repeat means, the automatic scale."""
    return _paired(data, "field", _auto(data), context["alpha"])


def paired_t_well(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """Well-balanced repeat means, the automatic scale."""
    return _paired(data, "well", _auto(data), context["alpha"])


def fixed_block_model(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """The candidate engine, always the model, on the per-repeat summaries."""
    log = _auto(data)
    table = repeat_means.summaries(data.objects, FEATURE, "cell")
    if log:
        table["value"] = np.log(table["value"])
    found = fixed_block.fit_model(table, {TREATED: 1.0, CONTROL: -1.0}, context["alpha"])
    keys = ("estimate", "p", "lo", "hi")
    return {**{k: found[k] for k in keys}, "truth": _truth(data, "cell", log)}


def sign_flip(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """The package's exact sign-flip test on the paired values (reference)."""
    log = _auto(data)
    table = repeat_means.summaries(data.objects, FEATURE, "cell")
    values, _ = repeat_means.paired_values(table, TREATED, CONTROL, log)
    p, floor, _exact = sign_flip_test(values, context["rng"])
    return {"p": p, "floor": floor}


def mixed(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """The random-intercept comparator on field means (statsmodels)."""
    return repeat_means.mixed_model(data.objects, FEATURE, TREATED, CONTROL, _auto(data))


def pooled_cells(data: ScalarData, context: dict[str, Any]) -> dict[str, float]:
    """Welch's t on the cells pooled (reference)."""
    return repeat_means.pooled_cells(data.objects, FEATURE, TREATED, CONTROL)


METHODS = (
    Method("paired_t_auto", paired_t_auto, role="ours", simplicity=1),
    Method("paired_t_declared", paired_t_declared, simplicity=2),
    Method("paired_t_difference", paired_t_difference, simplicity=1),
    Method("fixed_block", fixed_block_model, simplicity=3),
    Method("sign_flip", sign_flip, role="reference", simplicity=4),
    Method("paired_t_field", paired_t_field, question="field", simplicity=1),
    Method("paired_t_well", paired_t_well, question="well", simplicity=1),
    Method(
        "mixed",
        mixed,
        role="comparator",
        question="comparators",
        simplicity=5,
        max_replicates=COMPARATOR_REPLICATES,
    ),
    Method("pooled_cells", pooled_cells, role="reference", question="comparators", simplicity=1),
)

# Grid A: the centre at each number of repeats, then one factor at a time.
DEPARTURES = {
    "centre": {},
    "wells2": {"wells": 2},
    "fields5": {"fields": 5},
    "fields30": {"fields": 30},
    "cells5": {"cells": 5.0},
    "cells80": {"cells": 80.0},
    "icc_low": {"icc_field": 0.02},
    "icc_high": {"icc_field": 0.3},
    "unbalanced": {"field_imbalance": 0.5, "cell_dispersion": 1.0},
    "skewed": {"cv": 0.8},
}
ALTERNATIVES = {
    "multiplicative": [{"scale": "ratio", "effect": e} for e in (0.5, 1.0)],
    "additive": [
        {"distribution": "normal", "scale": "difference", "effect": e} for e in (0.5, 1.0)
    ],
    "heterogeneous": [{"scale": "ratio", "effect": 1.0, "effect_sd": 0.5}],
    "unbalanced": [
        {"scale": "ratio", "effect": 0.75, "field_imbalance": 0.5, "cell_dispersion": 1.0}
    ],
}
STRESS = {
    "extreme_repeat": {"extreme_repeat": 2.0},
    "faulty_well": {"wells": 2, "faulty_well": 3.0},
    "heavy_tails": {"distribution": "t3"},
    "skewed_gamma": {"distribution": "gamma", "cv": 1.0},
    "informative_size": {"informative_size": 0.8},
    "near_zero": {"distribution": "near_zero"},
    "background": {"distribution": "background"},
    "zero_inflated": {"distribution": "zero_inflated", "level": 3.0},
    "bounded": {"distribution": "bounded"},
    "saturated": {"distribution": "saturated"},
    "unrecorded_wells": {"wells": 2, "wells_recorded": False, "icc_well": 0.15},
    "mnar": {"missing": "mnar", "missing_rate": 0.3},
    "severe_imbalance": {"field_imbalance": 1.0, "cell_dispersion": 2.0},
    "confounded_session": {"sessions": "confounded", "gap": 2.0},
    "heteroscedastic": {"heteroscedastic": 2.0},
    "segmentation_failure": {"segmentation_loss": 0.3},
    "qc_exclusion": {"missing": "condition", "missing_rate": 0.3},
    "plates": {"sessions": "plates", "plate_sd": 0.3},
    "cell_number": {"count_effect": 0.5},
    "empty_fields_absent": {"empty_fields": 0.2, "empty_recorded": False},
}
SCREENING = {
    "repeats": (3, 6),
    "wells": (1, 2),
    "fields": (5, 30),
    "cells": (5.0, 80.0),
    "icc_field": (0.02, 0.3),
    "icc_well": (0.0, 0.15),
    "field_imbalance": (0.0, 0.8),
    "cv": (0.2, 0.8),
}


def points() -> tuple[Point, ...]:
    """The design points of P1: core nulls, core alternatives, stress, screening."""
    out = []
    for n in (3, 4, 6):
        for name, change in DEPARTURES.items():
            out.append(Point(f"core_null_n{n}_{name}", "core", {**CENTRE, "repeats": n, **change}))
    for label, settings in ALTERNATIVES.items():
        for n in (3, 4, 6):
            for i, change in enumerate(settings):
                params = {**CENTRE, "repeats": n, **change}
                out.append(Point(f"core_{label}_n{n}_{i}", "core", params, False, label))
    for name, change in STRESS.items():
        out.append(Point(f"stress_{name}", "stress", {**CENTRE, "repeats": 4, **change}))
    for i, change in enumerate(latin_hypercube(SCREENING, 32, seed=101)):
        out.append(Point(f"screen_{i:02d}", "screening", {**CENTRE, **change}))
    return tuple(out)


MODULE = Module(
    name="p1",
    title="core repeat inference",
    simulate=simulate,
    points=points(),
    methods=METHODS,
    rules=(Rule("main"), Rule("field"), Rule("well")),
)

if __name__ == "__main__":
    main("decisions.p1_repeat_inference", sys.argv[1:])
