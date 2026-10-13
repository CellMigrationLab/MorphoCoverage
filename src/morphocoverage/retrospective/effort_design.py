"""Standard errors of the effort projection: the acquired design and balanced allocations.

Kept apart from the estimation of the components (``effort``). The acquired
design's variance of the mean weighs each level's component by
``sum N_u^2 / N^2`` over its units (``N_u`` observations each), so unequal
samples, repeats and fields count as acquired; an allocation is projected as
balanced (every sample with as many fields and observations), and compared
with the acquired allocation made balanced. See ``effort`` for the formulas.

Shared-private: ``_allocation_rows``, ``_check_allocations``, ``_contributions``,
``_design`` and ``_interval`` (used by ``effort``); ``between_technical_repeats`` (used by
``workflows._document``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._checks import is_integer, is_number
from .._style import group_label
from ..statistics import _interval as _widened
from ..statistics import cluster_size
from .hierarchy import SamplingHierarchy
from .protocol import FEW_UNITS


def _se(
    components: np.ndarray, design: dict[str, Any], samples: float, fovs_per_sample: float
) -> float:
    """Projected SE of the mean for ``samples`` x ``fovs_per_sample`` acquired FOVs, balanced.

    The repeat term is ``v_repeat / min(repeats, samples)``. Only a share
    ``design["yield"]`` of acquired FOVs holds observations, with
    ``design["per_fov"]`` observations each on average, as in the acquired
    design; empty FOVs are effort that averages out nothing. Every sample
    holds as many FOVs and observations (``_acquired_se`` has the acquired,
    unequal sizes).
    """
    comp = np.maximum(components, 0.0)
    if design["has_repeat"]:
        v_rep, v_sample, v_fov, v_obs = comp
        # Fewer samples than repeats cover only that many repeats.
        between = v_rep / min(design["repeats"], samples)
    else:
        v_sample, v_fov, v_obs = comp
        between = 0.0
    fields_with = samples * fovs_per_sample * design["yield"]
    observations = fields_with * design["per_fov"]
    # Fields of unequal size weigh a pooled mean unequally: the field term is
    # v_fov * sum n_j^2 / N^2 = v_fov * cluster_size / N, which is v_fov / fields
    # only when every field holds as many observations.
    return float(
        np.sqrt(
            between
            + v_sample / samples
            + v_fov * design["cluster_size"] / observations
            + v_obs / observations
        )
    )


def _parts(components: np.ndarray, design: dict[str, Any]) -> np.ndarray:
    """Each level's term in the variance of the acquired design's mean (components >= 0).

    A level's term is its component times ``sum N_u^2 / N^2`` over its units
    (``N_u`` observations each), which is ``1 / units`` only when all are equal.
    """
    observations = design["total_fovs"] * design["yield"] * design["per_fov"]
    weights = [design["sample_weight"], design["cluster_size"] / observations, 1 / observations]
    if design["has_repeat"]:
        weights = [design["repeat_weight"], *weights]
    return np.maximum(components, 0.0) * np.array(weights, dtype=float)


def _acquired_se(components: np.ndarray, design: dict[str, Any]) -> float:
    """SE of the mean of the acquired design, with its acquired (possibly unequal) sizes."""
    return float(np.sqrt(_parts(components, design).sum()))


def _contributions(estimate: np.ndarray, design: dict[str, Any]) -> list[float | None]:
    """Share of each level in the variance of the acquired design's mean (components >= 0)."""
    parts = _parts(estimate, design)
    total = parts.sum()
    return [float(p / total) if total > 0 else None for p in parts]


def _weight(qualified: pd.DataFrame, level: str) -> float:
    """``sum N_u^2 / N^2`` over the units of ``level`` holding ``N_u`` observations each."""
    sizes = qualified.groupby(level, sort=False).size().to_numpy(dtype=float)
    return float(np.sum(sizes**2) / sizes.sum() ** 2)


def _check_allocations(
    allocations: Sequence[tuple[int, float]] | None,
) -> list[tuple[int, float]] | None:
    """The declared allocations as ``(samples, fovs_per_sample)``, refused unless positive."""
    if allocations is None:
        return None
    chosen = []
    for allocation in allocations:
        try:
            samples, per_sample = allocation
        except (TypeError, ValueError):
            raise ValueError(
                f"each allocation must be a (samples, fovs_per_sample) pair, not {allocation!r}"
            ) from None
        whole = is_integer(samples) or (is_number(samples) and float(samples).is_integer())
        if not (whole and samples >= 1 and is_number(per_sample) and 0 < per_sample < np.inf):
            raise ValueError(
                f"allocation {allocation!r}: samples must be a positive whole number and "
                "fovs_per_sample a positive number"
            )
        chosen.append((int(samples), float(per_sample)))
    return chosen


def _allocations(
    design: dict[str, Any], chosen: list[tuple[int, float]] | None
) -> list[tuple[int, float]]:
    """Allocations to compare, always including the acquired one (``samples``, total / samples)."""
    samples, total = design["samples"], design["total_fovs"]
    current = (samples, total / samples)
    if chosen is None:
        # From half to twice the samples, at exactly the acquired total of FOVs,
        # never fewer samples than there are repeats, and never less than one
        # field per sample (a fraction of a field is not an allocation).
        low = max(2, design["repeats"])
        counts = {max(low, round(samples * f)) for f in (0.5, 0.75, 1.5, 2)} - {samples}
        chosen = [(s, total / s) for s in sorted(counts) if s <= total]
    return [current, *[a for a in chosen if a != current]]


def _interval(
    draws: np.ndarray, point: float, units: int, floor: float | None = None
) -> tuple[float | None, float | None]:
    """The approximate 95% interval of ``point`` from the finite draws, or ``(None, None)``.

    The percentile interval of the draws widened for the ``units``
    independent units resampled (``statistics._interval``): a plain
    percentile interval over 3 to 8 units covered 72-93% for a nominal 95%.
    ``floor`` bounds the widened low end (0 for a standard error or a ratio of
    them, which cannot be negative). None without two finite draws, two units
    or a finite point.
    """
    finite = draws[np.isfinite(draws)]
    if len(finite) < 2 or point is None or not np.isfinite(point):
        return None, None
    _, low, high = _widened(float(point), finite, int(units), 0.95)
    if not (np.isfinite(low) and np.isfinite(high)):
        return None, None
    if floor is not None:
        low = max(low, floor)
    return float(low), float(high)


def _design(
    qualified: pd.DataFrame,
    levels: list[str],
    has_repeat: bool,
    total_fovs: int,
    hierarchy: SamplingHierarchy,
) -> dict[str, Any]:
    """The acquired design of one stratum, its units counted among those with observations."""
    per_field = qualified.groupby(levels[-1], sort=False).size().to_numpy()
    repeat = hierarchy.biological_repeat
    return {
        "has_repeat": has_repeat,
        "repeats": int(qualified[repeat].nunique()) if has_repeat else 1,
        "samples": int(qualified[hierarchy.sample].nunique()),
        "total_fovs": total_fovs,
        "fovs_with_observations": len(per_field),
        "yield": len(per_field) / total_fovs,
        "per_fov": len(qualified) / len(per_field),
        # Sum n_j^2 / N: the cluster size that sets the design effect of unequal FOVs.
        "cluster_size": cluster_size(per_field),
        "sample_weight": _weight(qualified, hierarchy.sample),
        "repeat_weight": _weight(qualified, repeat) if has_repeat else None,
    }


def _acquired_projection(
    acquired: float,
    boots: np.ndarray,
    design: dict[str, Any],
    where: dict[str, Any],
    units: int,
) -> dict[str, Any]:
    """The projection row of the acquired design, its standard error ``acquired``."""
    samples, total_fovs = design["samples"], design["total_fovs"]
    low, high = _interval(
        np.array([_acquired_se(b, design) for b in boots]), acquired, units, floor=0.0
    )
    return {
        **where,
        "samples": samples,
        "fovs_per_sample": total_fovs / samples,
        "total_fovs": float(total_fovs),
        "same_total": True,
        "design": "acquired",
        "se": acquired,
        "se_low": low,
        "se_high": high,
        "current": True,
    }


def _allocation_rows(
    estimate: np.ndarray,
    boots: np.ndarray,
    design: dict[str, Any],
    chosen: list[tuple[int, float]] | None,
    where: dict[str, Any],
    units: int,
    unit: str,
) -> tuple[list[dict], list[dict]]:
    """Projections (the acquired design, then each allocation balanced) and comparisons."""
    samples, total_fovs = design["samples"], design["total_fovs"]
    current = (samples, total_fovs / samples)
    acquired = _acquired_se(estimate, design)
    projections = [_acquired_projection(acquired, boots, design, where, units)]
    comparisons = []
    base = np.array([_se(b, design, *current) for b in boots])
    balanced = _se(estimate, design, *current)
    for s, f in _allocations(design, chosen):
        se = _se(estimate, design, s, f)
        boot_se = np.array([_se(b, design, s, f) for b in boots])
        same_total = abs(s * f - total_fovs) < 0.5
        low, high = _interval(boot_se, se, units, floor=0.0)
        projections.append(
            {
                **where,
                "samples": s,
                "fovs_per_sample": f,
                "total_fovs": s * f,
                "same_total": same_total,
                "design": "balanced",
                "se": se,
                "se_low": low,
                "se_high": high,
                "current": (s, f) == current,
            }
        )
        if (s, f) == current:
            continue
        ratio = se / balanced if balanced > 0 else None
        low, high = _interval(boot_se / np.where(base > 0, base, np.nan), ratio, units, 0.0)
        resolved = same_total and low is not None and (high < 1 or low > 1) and units >= FEW_UNITS
        comparisons.append(
            {
                **where,
                "samples": s,
                "fovs_per_sample": f,
                "ratio": ratio,
                "ratio_low": low,
                "ratio_high": high,
                "same_total": same_total,
                "resolved": bool(resolved),
                "current_samples": current[0],
                "current_fovs_per_sample": current[1],
                "balanced_se": balanced,
                "acquired_se": acquired,
                "units": units,
                "unit": unit,
            }
        )
    return projections, comparisons


def between_technical_repeats(
    fov_rows: pd.DataFrame, technical: str, outer: list[str]
) -> tuple[list[tuple[float, float, int]], dict[str, list[str]]]:
    """Per condition and repeat with 2 or more technical repeats: the spread of their means
    against what their fields predict, the relative SE of the mean over them, and what was
    left out.

    From the fields' means (every field with observations weighing one): the
    observed variance of the technical-repeat means against the mean over
    technical repeats of ``var(field means) / fields``, which is all the
    fields alone would give. Returns ``(ratio, relative SE, technical repeats)``
    per group, and the labels left out by reason: ``one_field`` (a technical
    repeat with one field, whose field variance cannot be estimated), ``fewer``
    (a group with fewer than 2 technical repeats of 2 or more fields) and ``flat``
    (fields that do not vary, or a mean of 0). A group with one technical repeat
    has nothing between them and is not named (owner, 2026-10-05).
    """
    rows = fov_rows.dropna(subset=["mean"])
    parts = rows.groupby(outer, sort=False, dropna=False) if outer else [((), rows)]
    found, left = [], {"one_field": [], "fewer": [], "flat": []}
    for key, own in parts:
        label = group_label(dict(zip(outer, key if isinstance(key, tuple) else (key,))))
        per = own.groupby(technical, sort=False)["mean"].agg(["mean", "var", "count"])
        if len(per) < 2:
            continue
        single = [f"{name} of {label}" for name in per.index[per["count"] < 2]]
        per = per[per["count"] >= 2]
        if len(per) < 2:
            left["fewer"].append(label)
            continue
        predicted = float((per["var"] / per["count"]).mean())
        observed = float(per["mean"].var(ddof=1))
        grand = abs(float(per["mean"].mean()))
        if predicted <= 0 or grand == 0:
            left["flat"].append(label)
            continue
        left["one_field"] += single
        found.append(
            (math.sqrt(observed / predicted), math.sqrt(observed / len(per)) / grand, len(per))
        )
    return found, left


__all__: list[str] = []
