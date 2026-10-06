"""The discovery's tests (private): each feature and condition, in worker processes.

``feature_discovery`` hands the features out a few at a time (``tested_rows``): each one's
values per condition and repeat are pooled here (``_pooled``), and every condition's
cell-level test, paired test and per-repeat d computed (``_row``). Each test is seeded from
its feature and condition, so its row is the same in any worker and for any ``n_jobs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._parallel import UNITS_PER_WORKER, map_units, worker_pool
from .._seeds import group_rng
from ..statistics import (
    cohen_d,
    contrasts_text,
    log_scale,
    paired_t,
    repeat_effect,
    repeat_values,
)
from ._units import control_label, side_units, sides
from .contrast_cells import trusted_test
from .contrast_repeats import MIN_REPEATS

# Why a condition was not tested, beyond one value everywhere: as provided its control would
# pool other repeats and no balanced test could be formed (contrast_cells.trusted_test), or
# there is no Cohen's d (each side holds one value of its own); UNTESTED for mixed reasons.
POOLED_CONTROL = "its control would include repeats the condition was not imaged in"
NO_SPREAD = "no spread within the condition and the control"


def _pooled(
    all_groups: list,
    described: list,
    contrasts: list,
    feature: str,
    technical_col: str | None,
    scale: str | None = None,
) -> dict[Any, dict[str, Any]]:
    """Per condition (or comparison), its observations and its own control's per repeat,
    every observation of that control (a comparison's reference) per repeat, as
    ``condition_contrast`` pools them, and the feature's scale (``log``: ``scale`` when
    declared, else from all its observations, ``statistics.log_scale``)."""
    pooled: dict[Any, dict[str, Any]] = {}
    values = np.concatenate([data.feature_values(feature) for _, data in all_groups])
    log = log_scale(values, scale, repr(feature))
    every: dict[Any, dict[Any, np.ndarray]] = {}
    pairs = getattr(described, "comparisons", None) or {}
    for repeat, condition in contrasts:
        treated, controls = sides(described, repeat, condition)
        if not controls:
            continue
        t_units, _ = side_units(all_groups, feature, treated, technical_col)
        c_units, _ = side_units(all_groups, feature, controls, technical_col)
        t_values = np.concatenate([f for u in t_units for f in u])
        c_values = np.concatenate([f for u in c_units for f in u])
        if not len(t_values) or not len(c_values):
            continue
        reference = pairs[condition][1] if pairs else None
        every.setdefault(reference, {}).setdefault(repeat, c_values)
        side = pooled.setdefault(
            condition,
            {
                "condition": condition,
                "by_t": {},
                "by_c": {},
                "control_condition": control_label(described, controls),
                "reference": reference,
                "log": log,
            },
        )
        side["by_t"][repeat] = t_values
        side["by_c"][repeat] = c_values
    for side in pooled.values():
        side["every"] = every[side["reference"]]
    return pooled


def _paired(
    by_t: dict, by_c: dict, log: bool | None = None
) -> tuple[float, str, int, int, float, float]:
    """The paired t-test on the repeat means (``statistics.repeat_values``: log ratios when
    ``log``, the feature's scale), how many repeats are lower and higher than the control, the
    test on the raw differences (as ``condition_contrast`` gives it beside, so the readout's
    reading shares this family) and the change it tests (a fold change, else the mean
    difference)."""
    means = np.array([np.mean(by_t[r]) for r in by_t])
    controls = np.array([np.mean(by_c[r]) for r in by_t])
    lower, higher = int((means < controls).sum()), int((means > controls).sum())
    values, compared, change = repeat_values(means, controls, log)
    if len(means) < MIN_REPEATS:
        return np.nan, values, lower, higher, np.nan, change
    return paired_t(compared), values, lower, higher, paired_t(means - controls), change


def _effect(by_t: dict, by_c: dict, log: bool | None) -> dict[str, Any]:
    """Each repeat's contrast on the paired test's scale and the t-based 95% interval of the
    effect (``statistics.repeat_effect``; plan v2, owner, 2026-10-06)."""
    means = np.array([np.mean(by_t[r]) for r in by_t])
    controls = np.array([np.mean(by_c[r]) for r in by_t])
    values, compared, _ = repeat_values(means, controls, log)
    effect = repeat_effect(values, compared)
    return {
        "repeat_contrasts": contrasts_text(list(by_t), effect["contrasts"]),
        "ci_low": effect["ci_low"],
        "ci_high": effect["ci_high"],
    }


def _per_repeat(by_t: dict, by_c: dict) -> dict[str, float]:
    """Cohen's d of the observations in each repeat (``condition_contrast``'s ``d_objects``),
    its mean and range, and the fewest and most observations on a side in one repeat, for
    the page's table across repeats and each figure's legend."""
    d = np.array([cohen_d(by_t[r], by_c[r]) for r in by_t], dtype=float)
    d = d[np.isfinite(d)]
    sizes = [len(by_t[r]) for r in by_t] + [len(by_c[r]) for r in by_t]
    return {
        "d_repeat_mean": float(d.mean()) if len(d) else np.nan,
        "d_repeat_min": float(d.min()) if len(d) else np.nan,
        "d_repeat_max": float(d.max()) if len(d) else np.nan,
        "repeat_observations_min": min(sizes),
        "repeat_observations_max": max(sizes),
    }


def _row(
    feature: str,
    condition: Any,
    side: dict[str, Any],
    n_permutations: int,
    rng: np.random.Generator,
    alpha: float,
) -> tuple[dict[str, Any], Any]:
    """One test's row, and what ``contrast_cells.refined`` reruns when its p is at most
    ``alpha`` (empty otherwise, so only those samples are kept)."""
    by_t, by_c = side["by_t"], side["by_c"]
    p_paired, paired_values, lower, higher, p_raw, change = _paired(by_t, by_c, side["log"])
    row: dict[str, Any] = {
        "feature": feature,
        "condition": condition,
        "control_condition": side["control_condition"],
        "repeats": len(by_t),
        "lower": lower,
        "higher": higher,
        "p_paired": p_paired,
        "paired_values": paired_values,
        "change": change,
        "p_paired_raw": p_raw,
        "status": "ok",
        **_per_repeat(by_t, by_c),
        **_effect(by_t, by_c, side["log"]),
    }
    values = np.concatenate([*by_t.values(), *side["every"].values()])
    if np.ptp(values) == 0:
        return {**row, "d": np.nan, "p_cells": np.nan, "status": "one value everywhere"}, []
    test = trusted_test(by_t, by_c, side["every"], n_permutations, rng)
    retest = test["retest"] if test["p"] <= alpha else []
    if not np.isfinite(test["p"]):
        row["status"] = POOLED_CONTROL if len(side["every"]) > len(by_c) else NO_SPREAD
    return {
        **row,
        "observations": test["observations"],
        "control_observations": test["control_observations"],
        "d": test["d"],
        "p_cells": test["p"],
        "normalised": test["normalised"],
        "balanced": test["balanced"],
        "permutations": test["permutations"] if np.isfinite(test["p"]) else np.nan,
    }, retest


def _feature_tests(
    feature: str,
    pooled: dict[Any, dict[str, Any]],
    n_permutations: int,
    random_state: int,
    alpha: float,
) -> list[tuple[dict[str, Any], Any]]:
    """One feature's ``_row`` per condition, each seeded from its feature and condition, so
    the same in any worker."""
    return [
        _row(
            feature,
            side["condition"],
            side,
            n_permutations,
            group_rng(random_state, {"feature": feature, "condition": side["condition"]}),
            alpha,
        )
        for side in pooled.values()
    ]


def tested_rows(
    data: tuple[list, Any, list, str | None, dict[str, str]],
    features: list[str],
    settings: tuple[int, int, float],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[int, Any], dict[Any, set]]:
    """Every feature's rows, in ``workers`` processes: ``data`` is the groups, their
    description, the contrasts, the technical-repeat column and the declared scales;
    ``settings`` the shuffles, the random state and alpha. Also what ``refined`` may rerun,
    by row, and the repeats each condition is compared in."""
    rows: list[dict[str, Any]] = []
    retests: dict[int, Any] = {}
    held: dict[Any, set] = {}
    with worker_pool(workers) as pool:
        # A few features a worker at a time, so the values of every feature are never held.
        for first in range(0, len(features), workers * UNITS_PER_WORKER):
            jobs = [
                (feature, _pooled(*data[:3], feature, data[3], data[4].get(feature)))
                for feature in features[first : first + workers * UNITS_PER_WORKER]
            ]
            tested = map_units(_feature_tests, [(*job, *settings) for job in jobs], workers, pool)
            for (_, pooled), feature_rows in zip(jobs, tested, strict=True):
                for side in pooled.values():
                    held.setdefault(side["condition"], set()).update(side["by_t"])
                for row, retest in feature_rows:
                    if retest:
                        retests[len(rows)] = retest
                    rows.append(row)
    return rows, retests, held


__all__ = ["NO_SPREAD", "POOLED_CONTROL", "tested_rows"]
