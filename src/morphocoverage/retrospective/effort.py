"""More fields or more samples? A scalar hierarchical effort projection.

Question P10 of ``docs/questions.md``. For one declared scalar measurement (the
mean of a feature; a phenotype-state fraction is the mean of a 0/1 feature) and
a declared ``SamplingHierarchy`` (observation within FOV within sample,
optionally within biological repeat), it asks where the sampling uncertainty of
the mean comes from, and how it would change if the same total number of FOVs
were spread over more or fewer samples.

**Variance components** are estimated by the nested method of moments
(Henderson's method I, the ANOVA estimators for unbalanced nested random
designs): for every level, the sum of squares between its units within their
parent unit is set equal to its expectation, a linear function of the
components whose coefficients come from the unit sizes. Estimates can be
negative when a component is small; they are reported as estimated, and
truncated at zero only for projections.

**Intervals** come from a hierarchical bootstrap: independent units are
resampled with replacement, then the samples within each, then the FOVs
within each sample; the observations of a FOV stay together. Every level is
put in a canonical order (units by their key, values sorted within a field)
before resampling, so the row order of the tables never changes a draw.
Resampling every level is known to overstate the lower-level components
somewhat, and a percentile interval over few units under-covers (72-93%
for a nominal 95% with 3 to 8 wells), so each interval is widened for the
number of independent units (``statistics._interval``); the intervals are
approximate, and with few independent units they are wide. A
resample whose levels cannot be separated (one that draws only single-field
samples, say) is counted, never silently left out; when fewer than
``MIN_BOOT_SUCCESS`` of the resamples can be estimated, the estimable ones
are a selected subset rather than the bootstrap distribution, and no interval
is given.

**The acquired design.** Its standard error of the mean uses the acquired
sizes:

    sqrt(v_repeat sum N_r^2 / N^2 + v_sample sum N_s^2 / N^2
         + v_fov sum n_j^2 / N^2 + v_obs / N),

with ``N_r``, ``N_s`` and ``n_j`` the observations of each repeat, sample and
FOV and ``N`` their total; the variance shares use these terms. A pooled mean
weighs a large sample more, so samples of unequal size give a larger error
than the same totals spread evenly.

**Projection.** An allocation of ``n`` samples of ``f`` FOVs each is projected
as balanced: with the observed share ``y`` of FOVs holding observations, their
mean number of observations ``m``, their cluster size ``c = sum n_j^2 / N``
(``m`` when all are equal) and the observed number of repeats ``R``,

    sqrt(v_repeat / min(R, n) + v_sample / n + v_fov c / (n f y m) + v_obs / (n f y m)),

with ``N = n f y m`` observations; empty FOVs are effort that averages out
nothing. Fewer samples than repeats cover only ``n`` of the repeats, hence
``min(R, n)``.

Allocations are compared at equal total FOVs with the acquired allocation
made balanced (the same samples, FOVs and observations, spread evenly), so a
comparison is between allocations and never credits an alternative with
removing the acquired imbalance; the cost of that imbalance (the acquired
against the balanced standard error) is stated apart. A difference is stated
only when the bootstrap interval of the ratio of standard errors excludes 1
and there are at least ``FEW_UNITS`` independent units; otherwise the answer
is "cannot tell from these data".

**Units.** A unit is identified by its own value and every level outside it
(``SamplingHierarchy.unit_key``), so well J23 of plate 1A and well J23 of
plate 2A are two samples even when the plate is not a variance level. The
units inside the biological repeat (the FOV and the sample) are also
identified by the FOV identity's other group columns, as FOV identity is: in
a stratum that pools conditions, dish 2 of the control and dish 2 of the
drug are two samples, while one repeat spans both conditions.
Samples, repeats and independent units are counted among those holding
observations with a value: a well whose fields are all empty is acquired
effort (its fields count), not a sample of the measurement. When a stratum
has a single repeat with observations, or every repeat holds a single sample,
the two levels cannot be separated there: the repeat is then left out of that
stratum's components (its variance is counted with the sample's), the
independent unit becomes the sample, and the summary says which case it is.
When the sample column is the MorphoBricks ``technical_repeat`` (a well, a
dish), the sentences call it a technical repeat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._checks import is_integer
from .._former_repeat import one_warning, read_former
from .._seeds import group_rng
from .._wording import rows_of
from ..data_check import validate_input_tables
from ..schema import fov_positions
from ..statistics import design_effect as _design_effect
from ..statistics import nested_variance_components
from .effort_design import (  # noqa: F401  (_se, _acquired_se: re-exported for the tests)
    _acquired_se,
    _allocation_rows,
    _check_allocations,
    _contributions,
    _design,
    _interval,
    _se,
)
from .effort_text import OBSERVATION, ONE_SAMPLE_PER_REPEAT, SINGLE_REPEAT, effort_summary
from .hierarchy import SamplingHierarchy

# Share of bootstrap resamples that must be estimable for an interval to be given.
MIN_BOOT_SUCCESS = 0.9


@dataclass(frozen=True)
class EffortResult:
    """Variance components, projected standard errors and the comparison of allocations.

    - ``components``: per stratum and level, the ``estimate``, a bootstrap
      interval (``low``, ``high``) and its ``share`` of the variance of the
      acquired design's mean (with the acquired, possibly unequal sizes), with
      the stratum's ``samples`` and ``repeats`` holding observations (and the
      ``samples_acquired`` / ``repeats_acquired`` of the FOV table), ``fovs``
      and ``observations_per_fov`` (over every acquired FOV), the level
      counted as the independent ``unit``, whether the repeat level was left
      out there (``repeat_dropped``) and why (``repeat_dropped_reason``:
      ``"single repeat"`` or ``"one sample per repeat"``), the weights
      ``sum N_u^2 / N^2`` of the sample and repeat terms, and how many of
      the ``n_boot`` resamples could not be estimated (``boot_failed``).
    - ``projections``: per stratum, the acquired design (``design`` is
      ``"acquired"``) and each allocation (``samples`` x ``fovs_per_sample``,
      ``design`` ``"balanced"``), the standard error and its interval;
      ``current`` marks the acquired allocation, as acquired and made balanced.
    - ``comparisons``: per stratum, each allocation against the acquired one
      made balanced: the ratio of standard errors, its interval and whether it
      is resolved, with the ``balanced_se`` compared against and the
      ``acquired_se``.
    - ``skipped``: per stratum whose components cannot be estimated (a single
      sample, a single observation per field, no observation with a value, a
      missing stratum value), the ``reason``; such a stratum has no row in the
      other tables.
    """

    components: pd.DataFrame
    projections: pd.DataFrame
    comparisons: pd.DataFrame
    strata: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, detail: str = "short") -> str:
        """Where the uncertainty is, and whether more fields or more samples would help:
        ``"short"`` one table over the strata, ``"full"`` per stratum (``effort_text``)."""
        return effort_summary(self, detail)


def _levels(hierarchy: SamplingHierarchy) -> list[str]:
    """Declared levels above the FOV, coarsest first, then the FOV."""
    return list(reversed(hierarchy.levels))


def _unit_columns(hierarchy: SamplingHierarchy, level: str, identity: Sequence[str]) -> list[str]:
    """The columns identifying one unit of ``level``: its unit key and, inside the repeat, the
    FOV identity's other group columns (``identity``), so a sample name reused in two
    conditions pooled in one stratum stays two samples."""
    outer = [] if level == hierarchy.biological_repeat else list(identity)
    return [*hierarchy.unit_key(level), *outer]


def _qualified(
    cells: pd.DataFrame,
    hierarchy: SamplingHierarchy,
    levels: list[str],
    identity: Sequence[str] = (),
) -> pd.DataFrame:
    """``value`` and each level as its full unit key, so units stay distinct across outer levels."""
    table = cells[["value"]].copy()
    for level in levels:
        columns = _unit_columns(hierarchy, level, identity)
        table[level] = cells[columns].astype(str).agg("|".join, axis=1)
    return table


def _stratum_levels(
    hierarchy: SamplingHierarchy, qualified: pd.DataFrame
) -> tuple[list[str], str | None]:
    """The levels estimated in one stratum, and why the repeat was left out (None if kept).

    The repeat is left out when a single repeat holds observations, or every
    repeat holds a single sample with observations: neither can be separated.
    """
    levels = _levels(hierarchy)
    if hierarchy.biological_repeat is None:
        return levels, None
    n_repeats, n_samples = (qualified[level].nunique() for level in levels[:2])
    if n_repeats == 1:
        return levels[1:], SINGLE_REPEAT
    if n_repeats == n_samples:
        return levels[1:], ONE_SAMPLE_PER_REPEAT
    return levels, None


def _tree(frame: pd.DataFrame, columns: list[str]) -> np.ndarray | list:
    """Nested lists of per-FOV value arrays, coarsest level first, in a canonical order.

    Units are sorted by their key and a field's values sorted, so the
    resampling draws the same units whatever the row order of the tables
    (the components do not depend on the order of values within a field).
    """
    if not columns:
        return np.sort(frame["value"].to_numpy(dtype=float))
    head, rest = columns[0], columns[1:]
    return [_tree(part, rest) for _, part in frame.groupby(head, sort=True, dropna=False)]


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


def _inseparable(counts: list[int], levels: list[str], n_observations: int) -> str | None:
    """Why two nested levels cannot be told apart (one unit inside every unit above), or None."""
    if counts[0] < 2:
        return (
            f"a single {levels[0]!r} unit holds observations, so the variance at that level "
            "cannot be estimated; at least 2 units are needed"
        )
    if counts[-1] >= n_observations:
        return (
            "every FOV holds a single observation, so field-to-field and within-field "
            "variance cannot be separated"
        )
    for outer, inner, n_outer, n_inner in zip(levels, levels[1:], counts, counts[1:]):
        if n_outer == n_inner:
            return (
                f"every {outer} holds a single {inner} with observations, so their variance "
                f"cannot be separated; it needs more than one {inner} per {outer}"
            )
    return None


def _bootstrap(
    tree: np.ndarray | list, levels: list[str], n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Components of the estimable resamples, and how many could not be estimated.

    With fewer than ``MIN_BOOT_SUCCESS`` of the resamples estimable no draws
    are returned: the estimable ones are then a selected subset, not the
    bootstrap distribution, and an interval from them would look established.
    """
    depth = len(levels)
    boots, failed = [], 0
    for _ in range(n_boot):
        values, codes = _flatten(_resample(tree, rng), depth)
        # _flatten numbers units in the order met, so the last code + 1 counts them.
        counts = [int(c[-1]) + 1 if len(c) else 0 for c in codes]
        if _inseparable(counts, levels, len(values)) is not None:
            failed += 1
            continue
        try:
            boots.append(nested_variance_components(values, codes))
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
    if len(boots) < MIN_BOOT_SUCCESS * n_boot:
        return np.empty((0, depth + 1)), failed
    return np.array(boots), failed


def _component_rows(
    estimate: np.ndarray,
    boots: np.ndarray,
    names: list[str],
    design: dict[str, Any],
    fixed: dict[str, Any],
) -> list[dict]:
    """One row per level: the estimate, its interval and its share of the acquired variance."""
    shares = _contributions(estimate, design)
    rows = []
    for index, name in enumerate(names):
        low, high = _interval(boots[:, index], float(estimate[index]), fixed["units"])
        rows.append(
            {
                **fixed["where"],
                "level": name,
                "estimate": float(estimate[index]),
                "low": low,
                "high": high,
                "share": shares[index],
                **{k: v for k, v in fixed.items() if k != "where"},
            }
        )
    return rows


def _stratum(
    cells: pd.DataFrame,
    stratum_fovs: pd.DataFrame,
    hierarchy: SamplingHierarchy,
    chosen: list[tuple[int, float]] | None,
    n_boot: int,
    rng: np.random.Generator,
    where: dict[str, Any],
    identity: Sequence[str] = (),
) -> dict[str, list[dict]] | str:
    """The component, projection and comparison rows of one stratum, or why it is skipped."""
    qualified = _qualified(cells, hierarchy, _levels(hierarchy), identity)
    levels, dropped = _stratum_levels(hierarchy, qualified)  # e.g. [repeat, sample, fov]
    has_repeat = hierarchy.biological_repeat is not None and dropped is None
    codes = [qualified[level].factorize()[0] for level in levels]
    reason = _inseparable([len(np.unique(c)) for c in codes], levels, len(cells))
    if reason is not None:
        return reason
    try:
        estimate = nested_variance_components(cells["value"].to_numpy(), codes)
    except np.linalg.LinAlgError:
        return "the components are not identifiable"
    design = _design(qualified, levels, has_repeat, len(stratum_fovs), hierarchy)
    unit = hierarchy.independent_unit if has_repeat else hierarchy.sample
    units = int(qualified[unit].nunique())
    repeat = hierarchy.biological_repeat
    boots, failed = _bootstrap(_tree(qualified, levels), levels, n_boot, rng)
    fixed = {
        "where": where,
        "units": units,
        "unit": unit,
        "repeat_dropped": dropped is not None,
        "repeat_dropped_reason": dropped,
        "repeats": int(qualified[repeat].nunique()) if repeat is not None else None,
        "repeats_acquired": hierarchy.count(stratum_fovs, repeat) if repeat is not None else None,
        "samples": design["samples"],
        "samples_acquired": int(
            stratum_fovs.drop_duplicates(
                subset=_unit_columns(hierarchy, hierarchy.sample, identity)
            ).shape[0]
        ),
        "fovs": design["total_fovs"],
        "fovs_with_observations": design["fovs_with_observations"],
        "observations_per_fov": design["per_fov"],
        "cluster_size": design["cluster_size"],
        "sample_weight": design["sample_weight"],
        "repeat_weight": design["repeat_weight"],
        "boot_failed": failed,
        "n_boot": n_boot,
    }
    components = _component_rows(estimate, boots, [*levels, OBSERVATION], design, fixed)
    projections, comparisons = _allocation_rows(estimate, boots, design, chosen, where, units, unit)
    return {"components": components, "projections": projections, "comparisons": comparisons}


def _check_arguments(hierarchy: SamplingHierarchy, n_boot: object, random_state: object) -> None:
    if not isinstance(hierarchy, SamplingHierarchy):
        raise TypeError("hierarchy must be a SamplingHierarchy")
    if hierarchy.technical_levels:
        raise ValueError("technical levels are not supported by the effort projection yet")
    if not is_integer(n_boot) or n_boot < 2:
        raise ValueError("n_boot must be an integer of at least 2")
    if not is_integer(random_state):
        raise TypeError("random_state must be an integer seed")


def _missing_stratum(data_fovs: pd.DataFrame, stratum: dict[str, Any]) -> str | None:
    """Why a stratum with a missing value is not analysed, or None when it has none."""
    absent = [column for column, value in stratum.items() if pd.isna(value)]
    if not absent:
        return None
    count = len(rows_of(data_fovs, stratum))  # a missing value matches a missing one
    return (
        f"{', '.join(absent)} is missing for {count} fields, so they belong to no stratum; "
        "give them a value or leave them out"
    )


@one_warning
def effort_projection(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    hierarchy: SamplingHierarchy,
    strata: Sequence[str],
    allocations: Sequence[tuple[int, float]] | None = None,
    n_boot: int = 500,
    random_state: int = 0,
    observation_unit: str | None = None,
) -> EffortResult:
    """Variance components of the mean of ``feature`` and projected standard errors.

    ``hierarchy`` declares the FOV, sample and optional biological-repeat
    columns of the FOV table (technical levels are not supported yet); in a
    stratum with a single repeat, or where every repeat holds a single sample,
    the repeat is left out of the components (see the module). ``strata`` are
    FOV-table columns analysed separately, for example ``["condition"]``;
    ``[]`` pools every sample and must be asked for. ``allocations`` are
    ``(samples, fovs_per_sample)`` pairs to compare with the acquired design
    (a positive whole number of samples, a positive number of fields); by
    default, from half to twice the acquired number of samples at exactly the
    acquired total of FOVs, never below one field per sample (FOVs per sample
    may be fractional, an average). Only allocations at the acquired total are
    compared ("at equal total FOVs"), with the acquired allocation made
    balanced; others are projected but never resolved. Observations with a
    missing value of ``feature`` are left out and reported. A stratum whose
    components cannot be estimated (a single sample, a single observation per
    field, no observation with a value, a missing stratum value) is listed in
    ``skipped`` with the reason instead of stopping the others.
    """
    _check_arguments(hierarchy, n_boot, random_state)
    chosen = _check_allocations(allocations)
    objects, fovs, strata = read_former(objects=objects, fovs=fovs, strata=list(strata))
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
    # FOV-identity columns the hierarchy does not name (a condition): they identify units too.
    identity = [c for c in data.fov_key if c not in hierarchy.levels and c not in strata]
    positions = fov_positions(data, data.fovs)
    values = data.feature_values(feature)
    tables: dict[str, list[dict]] = {"components": [], "projections": [], "comparisons": []}
    skipped_rows = []
    for stratum in data.fovs[strata].drop_duplicates().to_dict("records") if strata else [{}]:
        where = {**stratum, "stratum": ", ".join(f"{k}={v}" for k, v in stratum.items()) or "all"}
        reason = _missing_stratum(data.fovs[strata], stratum)
        if reason is not None:
            skipped_rows.append({**where, "reason": reason})
            continue
        inside = np.ones(len(data.fovs), dtype=bool)
        for column, value in stratum.items():
            inside &= data.fovs[column].to_numpy() == value
        stratum_fovs = data.fovs[inside]
        # Strata are analysed apart, so a field name reused across them (the same
        # image name in two conditions) is not a duplicate; within one it is.
        hierarchy.validate(stratum_fovs)
        keep = inside[positions] & ~np.isnan(values)
        if not keep.any():
            reason = f"no field holds an observation with a value of {feature}"
            skipped_rows.append({**where, "reason": reason})
            continue
        columns = [*hierarchy.levels, *identity]
        cells = data.fovs.iloc[positions[keep]][columns].reset_index(drop=True)
        cells["value"] = values[keep]
        # One generator per stratum, so adding a stratum never changes another's interval.
        rng = group_rng(int(random_state), stratum)
        rows = _stratum(cells, stratum_fovs, hierarchy, chosen, int(n_boot), rng, where, identity)
        if isinstance(rows, str):
            # Reported, so one inestimable stratum does not hide the others.
            skipped_rows.append({**where, "reason": rows})
            continue
        n_missing = int((inside[positions] & np.isnan(values)).sum())
        for row in rows["components"]:
            row["missing_values"] = n_missing
        for name, own in rows.items():
            tables[name] += own
    settings = {
        "feature": feature,
        "hierarchy": hierarchy.parameters,
        "independent_unit": hierarchy.independent_unit,
        "strata": strata,
        "n_boot": n_boot,
        "min_boot_success": MIN_BOOT_SUCCESS,
        "random_state": random_state,
        "allocations": None if chosen is None else [list(a) for a in chosen],
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
    }
    where_columns = [*strata, "stratum"]
    frames = [
        pd.DataFrame(rows, columns=None if rows else where_columns) for rows in tables.values()
    ]
    return EffortResult(
        *frames,
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
    return _design_effect(
        float(rows.loc[fov_level, "estimate"]),
        float(rows.loc[OBSERVATION, "estimate"]),
        float(rows["cluster_size"].iloc[0]),
    )


__all__ = [
    "MIN_BOOT_SUCCESS",
    "OBSERVATION",
    "EffortResult",
    "design_effect",
    "effort_projection",
    "effort_summary",
    "nested_variance_components",
]
