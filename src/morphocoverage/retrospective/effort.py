"""More fields or more samples? A scalar hierarchical effort projection.

Roadmap Step 7. For one declared scalar measurement (the mean of a feature;
a phenotype-state fraction is the mean of a 0/1 feature) and a declared
``SamplingHierarchy`` (observation within FOV within sample, optionally within
biological repeat), it asks where the sampling uncertainty of the mean comes
from, and how it would change if the same total number of FOVs were spread
over more or fewer samples.

**Variance components** are estimated by the nested method of moments
(Henderson's method I, the ANOVA estimators for unbalanced nested random
designs): for every level, the sum of squares between its units within their
parent unit is set equal to its expectation, a linear function of the
components whose coefficients come from the unit sizes. Estimates can be
negative when a component is small; they are reported as estimated, and
truncated at zero only for projections.

**Intervals** come from a hierarchical bootstrap: independent units are
resampled with replacement, then the samples within each, then the FOVs
within each sample; the observations of a FOV stay together. Resampling every
level is known to overstate the lower-level components somewhat; the
intervals are approximate, and with few independent units they are wide.

**Projection.** The standard error of the mean for ``n`` samples of ``f`` FOVs
each, with the observed share ``y`` of FOVs holding observations, their mean
number of observations ``m``, their cluster size ``c = sum n_j^2 / N`` (``m``
when all are equal) and the observed number of repeats ``R``, is

    sqrt(v_repeat / R + v_sample / n + v_fov c / (n f y m) + v_obs / (n f y m)),

with ``N = n f y m`` observations; empty FOVs are effort that averages out
nothing.

Allocations are compared at equal total FOVs. A difference is stated only when
the bootstrap interval of the ratio of standard errors excludes 1 and there
are at least ``FEW_UNITS`` independent units; otherwise the answer is "cannot
tell from these data".

**Units.** A unit is identified by its own value and every level outside it
(``SamplingHierarchy.unit_key``), so well J23 of plate 1A and well J23 of
plate 2A are two samples even when the plate is not a variance level. When
every biological repeat of a stratum holds a single sample, the two levels
cannot be separated there: the repeat is then left out of that stratum's
components (its variance is counted with the sample's), the independent unit
becomes the sample, and the summary says so. When the sample column is the
MorphoBricks ``technical_repeat`` (a well, a dish), the sentences call it a
technical repeat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..schema import fov_positions, validate_input_tables
from ._text import group_identity
from .effort_text import OBSERVATION, effort_summary
from .hierarchy import SamplingHierarchy
from .protocol import FEW_UNITS


def _dense_codes(code: np.ndarray) -> np.ndarray:
    """Codes renumbered 0..K-1 by value; returned as they are when they already are."""
    # bincount allocates max + 1 entries, so only codes that could be dense qualify.
    if code.dtype.kind in "iu" and len(code) and code.min() == 0 and code.max() < len(code):
        present = np.bincount(code)
        if present.all():
            return code
    return np.unique(code, return_inverse=True)[1]


def nested_variance_components(values: np.ndarray, codes: Sequence[np.ndarray]) -> np.ndarray:
    """Method-of-moments variance components of a nested random design.

    ``codes`` gives, coarsest level first, the integer unit of each value at
    each level (units must nest: every unit of a level lies inside one unit
    of the level before it). Returns one component per level plus the
    residual (observation) component, in the same order.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or np.isnan(values).any():
        raise ValueError("need at least two non-missing values")
    levels = [np.zeros(n, dtype=int), *[np.asarray(c) for c in codes], np.arange(n)]
    sizes, totals = [], []
    for code in levels:
        inverse = _dense_codes(code)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=values)
        sizes.append((inverse, counts))
        totals.append(float(np.sum(sums**2 / counts)))
    depth = len(levels)

    def coefficient(level: int, component: int) -> float:
        # E[T_level] = N mu^2 + sum_k c[level, k] v_k, with T_level = sum_u n_u ybar_u^2.
        if component <= level:
            return float(n)
        inverse, counts = sizes[level]
        finer_inverse, finer_counts = sizes[component]
        parent_of_finer = np.zeros(len(finer_counts), dtype=int)
        parent_of_finer[finer_inverse] = inverse
        squared = np.bincount(parent_of_finer, weights=finer_counts.astype(float) ** 2)
        return float(np.sum(squared / counts))

    rows, rhs = [], []
    for level in range(1, depth):
        rows.append([coefficient(level, k) - coefficient(level - 1, k) for k in range(1, depth)])
        rhs.append(totals[level] - totals[level - 1])
    return np.linalg.solve(np.array(rows), np.array(rhs))


@dataclass(frozen=True)
class EffortResult:
    """Variance components, projected standard errors and the comparison of allocations.

    - ``components``: per stratum and level, the ``estimate``, a bootstrap
      interval (``low``, ``high``) and its ``share`` of the variance of the
      current design's mean, with the stratum's ``samples``, ``fovs`` and
      ``observations_per_fov`` (over every acquired FOV), the level counted
      as the independent ``unit`` and whether the repeat level was left out
      there (``repeat_dropped``).
    - ``projections``: per stratum and allocation (``samples`` x
      ``fovs_per_sample``), the projected standard error and its interval.
    - ``comparisons``: per stratum, each allocation against the current one:
      the ratio of standard errors, its interval, and whether it is resolved.
    - ``skipped``: per stratum whose components cannot be estimated (a single
      sample, a single observation per field), the ``reason``; such a stratum
      has no row in the other tables.
    """

    components: pd.DataFrame
    projections: pd.DataFrame
    comparisons: pd.DataFrame
    strata: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """Where the uncertainty is, and whether more fields or more samples would help."""
        return effort_summary(self)


def _levels(hierarchy: SamplingHierarchy) -> list[str]:
    """Declared levels above the FOV, coarsest first, then the FOV."""
    return list(reversed(hierarchy.levels))


def _qualified(
    cells: pd.DataFrame, hierarchy: SamplingHierarchy, levels: list[str]
) -> pd.DataFrame:
    """``value`` and each level as its full unit key, so units stay distinct across outer levels."""
    table = cells[["value"]].copy()
    for level in levels:
        table[level] = cells[hierarchy.unit_key(level)].astype(str).agg("|".join, axis=1)
    return table


def _stratum_levels(
    hierarchy: SamplingHierarchy, qualified: pd.DataFrame
) -> tuple[list[str], bool]:
    """The levels estimated in one stratum: the declared ones, without the repeat when there
    is a single repeat, or every repeat holds a single sample there (neither can be separated)."""
    levels = _levels(hierarchy)
    if hierarchy.biological_repeat is None:
        return levels, False
    n_repeats, n_samples = (qualified[level].nunique() for level in levels[:2])
    if n_repeats == 1 or n_repeats == n_samples:
        return levels[1:], True
    return levels, False


def _tree(frame: pd.DataFrame, columns: list[str]) -> np.ndarray | list:
    """Nested lists of per-FOV value arrays, coarsest level first."""
    if not columns:
        return frame["value"].to_numpy(dtype=float)
    head, rest = columns[0], columns[1:]
    return [_tree(part, rest) for _, part in frame.groupby(head, sort=False, dropna=False)]


def _resample(tree: np.ndarray | list, rng: np.random.Generator) -> np.ndarray | list:
    if isinstance(tree, np.ndarray):
        return tree
    picks = rng.integers(0, len(tree), size=len(tree))
    return [_resample(tree[i], rng) for i in picks]


def _flatten(tree: np.ndarray | list, depth: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Values and one code array per level for a (resampled) tree of ``depth`` levels.

    Codes number the units of each level 0, 1, ... in the order they are met,
    so a unit drawn twice becomes two units, as a bootstrap requires.
    """
    values, leaf_codes = [], []
    counters = [0] * depth

    def walk(node: np.ndarray | list, level: int) -> None:
        if level == depth:
            values.append(node)
            leaf_codes.append([c - 1 for c in counters])
            return
        for child in node:
            counters[level] += 1
            walk(child, level + 1)

    walk(tree, 0)
    if not values:
        return np.array([]), [np.array([], dtype=np.int64) for _ in range(depth)]
    lengths = np.fromiter((len(v) for v in values), dtype=np.int64, count=len(values))
    table = np.asarray(leaf_codes, dtype=np.int64)
    return np.concatenate(values), [np.repeat(table[:, d], lengths) for d in range(depth)]


def _se(
    components: np.ndarray, design: dict[str, Any], samples: float, fovs_per_sample: float
) -> float:
    """Projected SE of the mean for ``samples`` x ``fovs_per_sample`` acquired FOVs.

    Only a share ``design["yield"]`` of acquired FOVs holds observations, with
    ``design["per_fov"]`` observations each on average, as in the acquired
    design; empty FOVs are effort that averages out nothing.
    """
    comp = np.maximum(components, 0.0)
    if design["has_repeat"]:
        v_rep, v_sample, v_fov, v_obs = comp
        between = v_rep / design["repeats"]
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


def _allocations(
    design: dict[str, Any], allocations: Sequence[tuple[int, float]] | None
) -> list[tuple[int, float]]:
    """Allocations to compare, always including the acquired one (``samples``, total / samples)."""
    samples, total = design["samples"], design["total_fovs"]
    current = (samples, total / samples)
    if allocations is not None:
        chosen = [(int(s), float(f)) for s, f in allocations]
    else:
        # From half to twice the samples, at exactly the acquired total of FOVs,
        # never fewer samples than there are repeats.
        low = max(2, design["repeats"])
        counts = {max(low, round(samples * f)) for f in (0.5, 0.75, 1.5, 2)} - {samples}
        chosen = [(s, total / s) for s in sorted(counts)]
    return [current, *[a for a in chosen if a != current]]


def effort_projection(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    hierarchy: SamplingHierarchy,
    strata: Sequence[str],
    allocations: Sequence[tuple[int, int]] | None = None,
    n_boot: int = 500,
    random_state: int = 0,
    observation_unit: str | None = None,
) -> EffortResult:
    """Variance components of the mean of ``feature`` and projected standard errors.

    ``hierarchy`` declares the FOV, sample and optional biological-repeat
    columns of the FOV table (technical levels are not supported yet); in a
    stratum where every repeat holds a single sample the repeat is left out
    of the components (see the module). ``strata`` are FOV-table columns
    analysed separately, for example
    ``["condition"]``; ``[]`` pools every sample and must be asked for.
    ``allocations`` are ``(samples, fovs_per_sample)`` pairs to compare with
    the acquired design; by default, from half to twice the acquired number of
    samples at exactly the acquired total of FOVs (FOVs per sample may then be
    fractional, an average). Only allocations at the acquired total are
    compared ("at equal total FOVs"); others are projected but never resolved.
    Observations with a missing value of ``feature`` are left out and
    reported. A stratum whose components cannot be estimated (a single
    sample, a single observation per field) is listed in ``skipped`` with
    the reason instead of stopping the others.
    """
    if not isinstance(hierarchy, SamplingHierarchy):
        raise TypeError("hierarchy must be a SamplingHierarchy")
    if hierarchy.technical_levels:
        raise ValueError("technical levels are not supported by the effort projection yet")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError("n_boot must be an integer of at least 2")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    strata = list(strata)
    missing_levels = [c for c in hierarchy.levels if c not in fovs.columns]
    if missing_levels:
        raise ValueError(f"hierarchy columns missing from the FOV table: {missing_levels}")
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=[feature],
        fov_col=hierarchy.fov,
        group_cols=[],
        observation_unit=observation_unit,
    )
    missing = [c for c in strata if c not in data.fovs.columns]
    if missing:
        raise ValueError(f"strata missing from the FOV table: {missing}")
    positions = fov_positions(data, data.fovs)
    values = data.feature_values(feature)
    component_rows, projection_rows, comparison_rows, skipped_rows = [], [], [], []
    for stratum in data.fovs[strata].drop_duplicates().to_dict("records") if strata else [{}]:
        inside = np.ones(len(data.fovs), dtype=bool)
        for column, value in stratum.items():
            inside &= data.fovs[column].to_numpy() == value
        stratum_fovs = data.fovs[inside]
        # Strata are analysed apart, so a field name reused across them (the same
        # image name in two conditions) is not a duplicate; within one it is.
        hierarchy.validate(stratum_fovs)
        keep = inside[positions] & ~np.isnan(values)
        cells = data.fovs.iloc[positions[keep]][list(hierarchy.levels)].reset_index(drop=True)
        cells["value"] = values[keep]
        n_missing = int((inside[positions] & np.isnan(values)).sum())
        qualified = _qualified(cells, hierarchy, _levels(hierarchy))
        levels, repeat_dropped = _stratum_levels(hierarchy, qualified)  # e.g. [repeat, sample, fov]
        has_repeat = hierarchy.biological_repeat is not None and not repeat_dropped
        names = [*levels, OBSERVATION]
        codes = [qualified[level].factorize()[0] for level in levels]
        where = {**stratum, "stratum": ", ".join(f"{k}={v}" for k, v in stratum.items()) or "all"}
        try:
            _check_separable(codes, levels, len(cells))
            estimate = nested_variance_components(cells["value"].to_numpy(), codes)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Reported, so one inestimable stratum does not hide the others.
            reason = (
                str(exc) if isinstance(exc, ValueError) else "the components are not identifiable"
            )
            skipped_rows.append({**where, "reason": reason})
            continue
        # One generator per stratum, so adding a stratum never changes another's interval.
        rng = np.random.default_rng([random_state, group_identity(stratum)])
        per_field = np.bincount(codes[-1])
        design = {
            "has_repeat": has_repeat,
            "repeats": (
                hierarchy.count(stratum_fovs, hierarchy.biological_repeat) if has_repeat else 1
            ),
            "samples": hierarchy.count(stratum_fovs, hierarchy.sample),
            "total_fovs": len(stratum_fovs),
            "yield": len(per_field) / len(stratum_fovs),
            "per_fov": len(cells) / len(per_field),
            # Sum n_j^2 / N: the cluster size that sets the design effect of unequal FOVs.
            "cluster_size": float(np.sum(per_field.astype(float) ** 2) / len(cells)),
        }
        samples, total_fovs = design["samples"], design["total_fovs"]
        current = (samples, total_fovs / samples)
        chosen = _allocations(design, allocations)
        unit = hierarchy.independent_unit if has_repeat else hierarchy.sample
        units = hierarchy.count(stratum_fovs, unit)
        tree = _tree(qualified, levels)
        boots = []
        for _ in range(n_boot):
            sample_values, sample_codes = _flatten(_resample(tree, rng), len(levels))
            try:
                boots.append(nested_variance_components(sample_values, sample_codes))
            except (ValueError, np.linalg.LinAlgError):
                continue
        boots = np.array(boots)
        contributions = _contributions(estimate, design)
        for index, name in enumerate(names):
            low, high = np.quantile(boots[:, index], [0.025, 0.975]) if len(boots) else (None, None)
            component_rows.append(
                {
                    **where,
                    "level": name,
                    "estimate": float(estimate[index]),
                    "low": None if low is None else float(low),
                    "high": None if high is None else float(high),
                    "share": contributions[index],
                    "units": units,
                    "unit": unit,
                    "repeat_dropped": repeat_dropped,
                    "repeats": (
                        hierarchy.count(stratum_fovs, hierarchy.biological_repeat)
                        if hierarchy.biological_repeat is not None
                        else None
                    ),
                    "samples": samples,
                    "fovs": total_fovs,
                    "fovs_with_observations": len(per_field),
                    "observations_per_fov": design["per_fov"],
                    "cluster_size": design["cluster_size"],
                    "missing_values": n_missing,
                }
            )
        base = np.array([_se(b, design, *current) for b in boots])
        current_se = _se(estimate, design, *current)
        for s, f in chosen:
            se = _se(estimate, design, s, f)
            boot_se = np.array([_se(b, design, s, f) for b in boots])
            same_total = abs(s * f - total_fovs) < 0.5
            projection_rows.append(
                {
                    **where,
                    "samples": s,
                    "fovs_per_sample": f,
                    "total_fovs": s * f,
                    "same_total": same_total,
                    "se": se,
                    "se_low": float(np.quantile(boot_se, 0.025)) if len(boots) else None,
                    "se_high": float(np.quantile(boot_se, 0.975)) if len(boots) else None,
                    "current": (s, f) == current,
                }
            )
            if (s, f) == current:
                continue
            ratio = boot_se / np.where(base > 0, base, np.nan)
            low, high = (
                np.nanquantile(ratio, [0.025, 0.975]) if np.isfinite(ratio).any() else (None, None)
            )
            resolved = (
                same_total and low is not None and (high < 1 or low > 1) and units >= FEW_UNITS
            )
            comparison_rows.append(
                {
                    **where,
                    "samples": s,
                    "fovs_per_sample": f,
                    "ratio": se / current_se if current_se > 0 else None,
                    "ratio_low": None if low is None else float(low),
                    "ratio_high": None if high is None else float(high),
                    "same_total": same_total,
                    "resolved": bool(resolved),
                    "current_samples": current[0],
                    "current_fovs_per_sample": current[1],
                    "units": units,
                    "unit": unit,
                }
            )
    settings = {
        "feature": feature,
        "hierarchy": hierarchy.parameters,
        "independent_unit": hierarchy.independent_unit,
        "strata": strata,
        "n_boot": n_boot,
        "random_state": random_state,
        "allocations": None if allocations is None else [list(a) for a in allocations],
        "observation_unit": data.observation_unit,
    }
    where_columns = [*strata, "stratum"]
    return EffortResult(
        pd.DataFrame(component_rows, columns=None if component_rows else where_columns),
        pd.DataFrame(projection_rows, columns=None if projection_rows else where_columns),
        pd.DataFrame(comparison_rows, columns=None if comparison_rows else where_columns),
        tuple(strata),
        settings,
        pd.DataFrame(skipped_rows, columns=[*where_columns, "reason"]),
    )


def design_effect(result: EffortResult, stratum: str) -> float | None:
    """How much larger the variance of a mean from whole FOVs is than from as many random objects.

    ``(m v_fov + v_obs) / (v_fov + v_obs)`` with ``m = sum n_j^2 / N`` over the
    FOVs with observations (the cluster size that holds for unequal FOVs; the
    mean count when all are equal), from the stratum's field and observation
    components (negative estimates set to zero); ``None`` when both are zero.
    """
    rows = result.components[result.components["stratum"] == stratum].set_index("level")
    fov_level = result.settings["hierarchy"]["fov"]
    v_fov = max(float(rows.loc[fov_level, "estimate"]), 0.0)
    v_obs = max(float(rows.loc[OBSERVATION, "estimate"]), 0.0)
    m = float(rows["cluster_size"].iloc[0])
    return None if v_fov + v_obs == 0 else (m * v_fov + v_obs) / (v_fov + v_obs)


def _check_separable(codes: list[np.ndarray], levels: list[str], n_observations: int) -> None:
    """Raise when two nested levels cannot be told apart (one unit inside every unit above)."""
    counts = [len(np.unique(c)) for c in codes]
    if counts[0] < 2:
        raise ValueError(
            f"a single {levels[0]!r} unit holds observations, so the variance at that level "
            "cannot be estimated; at least 2 units are needed"
        )
    if counts[-1] >= n_observations:
        raise ValueError(
            "every FOV holds a single observation, so field-to-field and within-field "
            "variance cannot be separated"
        )
    for outer, inner, n_outer, n_inner in zip(levels, levels[1:], counts, counts[1:]):
        if n_outer == n_inner:
            raise ValueError(
                f"every {outer} holds a single {inner} with observations, so their variance "
                f"cannot be separated; it needs more than one {inner} per {outer}"
            )


def _contributions(estimate: np.ndarray, design: dict[str, Any]) -> list[float | None]:
    """Share of each level in the variance of the acquired design's mean (components >= 0)."""
    comp = np.maximum(estimate, 0.0)
    observations = design["total_fovs"] * design["yield"] * design["per_fov"]
    # The field term with unequal fields, as in ``_se``.
    divisors = [design["samples"], observations / design["cluster_size"], observations]
    if design["has_repeat"]:
        divisors = [design["repeats"], *divisors]
    parts = comp / np.array(divisors, dtype=float)
    total = parts.sum()
    return [float(p / total) if total > 0 else None for p in parts]


__all__ = [
    "OBSERVATION",
    "EffortResult",
    "design_effect",
    "effort_projection",
    "effort_summary",
    "nested_variance_components",
]
