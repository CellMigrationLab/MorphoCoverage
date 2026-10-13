"""The per-repeat rows of condition_contrast (module contrast): each contrast's
test, effect sizes and JCS draws, and the sides the cell-level test pools.

Every contrast draws from its own generator (group_rng of its repeat and condition), in
the order the contrast module states: the randomization test, then the effect sizes; question
2 draws from a generator of its own, so neither changes the other's draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from .._seeds import group_rng
from ..metadata import REPEAT_COLUMN
from ..schema import ValidatedInput
from ..statistics import cohen_d, field_sample, fields_holding, jcs_effect_size
from ._groups import Described
from ._units import (
    FIELD,
    control_label,
    randomization_test,
    reference_of,
    separation_d,
    side_units,
    sides,
    unit_means,
)
from .contrast_sampling import NOT_SAMPLED, sampling_row
from .contrast_text import FEW_UNITS_D, NO_CONTROL
from .object_rules import RANDOM_OBJECTS, WHOLE_FOVS


class _ContrastRun(NamedTuple):
    """The checked arguments and the analysis groups of one condition_contrast call."""

    all_groups: list[tuple[dict[str, Any], ValidatedInput]]
    described: Described
    feature: str
    control: Mapping[str, Any] | None
    comparisons: Sequence[tuple[Any, Any]] | None
    condition_col: str
    repeat_col: str | None
    technical_col: str | None
    unit: str | None
    fields_independent: bool
    counts: list[int]
    n_permutations: int
    n_draws: int
    random_state: int


def _contrast_rows(
    run: _ContrastRun, contrasts: list[tuple[Any, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    """The per-repeat rows, the effect-size rows and, per condition, the observations the
    cell-level test pools, for every (repeat, condition) contrast in order."""
    rows, effect_rows = [], []
    # The cell-level test pools every repeat as provided: each condition's observations, and
    # every control observation once per repeat.
    pooled: dict[Any, dict[str, Any]] = {}
    # Each reference's observations by repeat (one reference with a single control).
    control_cells: dict[Any, dict[Any, np.ndarray]] = {}
    for repeat, condition in contrasts:
        row, effects = _contrast_row(run, repeat, condition, pooled, control_cells)
        rows.append(row)
        effect_rows += effects
    for side in pooled.values():
        side["control_all"] = control_cells[side["reference"]]
    return rows, effect_rows, pooled


def _contrast_row(
    run: _ContrastRun,
    repeat: Any,
    condition: Any,
    pooled: dict[Any, dict[str, Any]],
    control_cells: dict[Any, dict[Any, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One contrast's row and its effect-size rows; its sides added to pooled and
    control_cells. A repeat without the control gives a row without a test."""
    treated, controls = sides(run.described, repeat, condition)
    base = {
        REPEAT_COLUMN: repeat,
        "condition": condition,
        "control_condition": control_label(run.described, controls)
        or reference_of(run.described, condition, run.control, run.condition_col),
    }
    if not controls:
        return {**base, **_NO_TEST, "status": f"{NO_CONTROL} {run.repeat_col}={repeat}"}, []
    t_units, t_acquired = side_units(run.all_groups, run.feature, treated, run.technical_col)
    c_units, c_acquired = side_units(run.all_groups, run.feature, controls, run.technical_col)
    t_objects = np.concatenate([f for u in t_units for f in u])
    c_objects = np.concatenate([f for u in c_units for f in u])
    rng = group_rng(run.random_state, base)
    test = randomization_test(
        t_units, c_units, run.unit, run.n_permutations, rng, run.fields_independent
    )
    units = min(test["technical"], test["control_technical"])
    row = {
        **base,
        "fields": t_acquired,
        "observations": len(t_objects),
        "control_fields": c_acquired,
        "control_observations": len(c_objects),
        **test,
        "mean": float(t_objects.mean()) if len(t_objects) else np.nan,
        "control_mean": float(c_objects.mean()) if len(c_objects) else np.nan,
        "d_objects": cohen_d(t_objects, c_objects),
        "d_technical": (
            separation_d(unit_means(t_units), unit_means(c_units))
            if units > FEW_UNITS_D
            else np.nan
        ),
        "p_bonferroni": np.nan,
        "effect_full": jcs_effect_size(t_objects, c_objects),
    }
    reference = reference_of(run.described, condition, run.control, run.condition_col)
    _pool(pooled, control_cells, reference, base, t_objects, c_objects)
    # Question 2 with a control: do the fields acquired resolve this difference? Its own
    # generator, so the draws below are unchanged.
    own = group_rng(run.random_state, {**base, "question": "enough"})
    row.update(sampling_row(t_units, c_units, row["d_objects"], own))
    # The effect sizes do not depend on the test: computed for every contrast.
    resampled = run.unit if units >= 2 else None
    effects = _effect_rows(
        base, t_units, c_units, resampled, run.counts, run.n_draws, row["effect_full"], rng
    )
    return row, effects


def _pool(
    pooled: dict[Any, dict[str, Any]],
    control_cells: dict[Any, dict[Any, np.ndarray]],
    reference: Any,
    base: dict[str, Any],
    t_objects: np.ndarray,
    c_objects: np.ndarray,
) -> None:
    """A contrast's observations added to the condition's pooled sides, and its control's to
    the reference's (the first contrast of a repeat sets it)."""
    repeat = base[REPEAT_COLUMN]
    control_cells.setdefault(reference, {}).setdefault(repeat, c_objects)
    side = pooled.setdefault(
        base["condition"],
        {
            "treated_by_repeat": {},
            "control_by_repeat": {},
            "control_condition": base["control_condition"],
            "reference": reference,
        },
    )
    side["treated_by_repeat"][repeat] = t_objects
    side["control_by_repeat"][repeat] = c_objects


def _unit_sample(
    units: list[list[np.ndarray]], n: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """``n`` objects from whole technical repeats drawn with replacement, and the fields taken.

    A unit is drawn at random (with replacement, as from a population of wells
    like these), its fields are taken whole in random order, with a random
    part of the last one, until ``n`` objects; fields without observations
    count as acquired.
    """
    filled = [u for u in units if sum(len(f) for f in u)]
    taken, total, fields = [], 0, 0
    while total < n:
        unit = filled[int(rng.integers(len(filled)))]
        for i in rng.permutation(len(unit)):
            field, need = unit[i], n - total
            fields += 1
            if len(field) >= need:
                taken.append(rng.choice(field, size=need, replace=False))
                total = n
                break
            taken.append(field)
            total += len(field)
    return np.concatenate(taken), fields


def _effect_rows(
    base: dict[str, Any],
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    resampled: str | None,
    counts: list[int],
    n_draws: int,
    full: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """The JCS effect size at each sample size, with random objects and with whole fields.

    Whole fields are drawn from every acquired field, those without
    observations included, as a microscope stopping at ``n`` objects acquires
    them too; so ``fields_treated`` counts acquired fields. With
    ``resampled`` (the technical-repeat column, when both sides hold at least
    two) the whole-field draws take the fields of technical repeats drawn with
    replacement (``_unit_sample``), so their spread includes the well-to-well
    variation.
    """
    t_fields = [f for u in t_units for f in u]
    c_fields = [f for u in c_units for f in u]
    t_objects, c_objects = np.concatenate(t_fields), np.concatenate(c_fields)
    n_max = min(len(t_objects), len(c_objects))
    rows = []
    for n in [n for n in counts if n <= n_max]:
        for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
            draws, used = np.empty(n_draws), np.empty(n_draws)
            for i in range(n_draws):
                if sampling == RANDOM_OBJECTS:
                    t = rng.choice(t_objects, size=n, replace=False)
                    c = rng.choice(c_objects, size=n, replace=False)
                elif resampled:
                    t, used[i] = _unit_sample(t_units, n, rng)
                    c, _ = _unit_sample(c_units, n, rng)
                else:
                    t = field_sample(t_fields, n, rng)
                    c = field_sample(c_fields, n, rng)
                draws[i] = jcs_effect_size(t, c)
            defined = draws[np.isfinite(draws)]
            fields_treated = np.nan
            if sampling == WHOLE_FOVS:
                fields_treated = (
                    float(used.mean()) if resampled else fields_holding(t_fields, n, rng)
                )
            rows.append(
                {
                    **base,
                    "sampling": sampling,
                    "resampled": resampled if sampling == WHOLE_FOVS else None,
                    "n_objects": n,
                    "median": float(np.median(defined)) if len(defined) else np.nan,
                    "q25": float(np.percentile(defined, 25)) if len(defined) else np.nan,
                    "q75": float(np.percentile(defined, 75)) if len(defined) else np.nan,
                    "same_sign": (
                        float(np.mean(np.sign(defined) == np.sign(full)))
                        if len(defined) and np.isfinite(full) and full != 0
                        else np.nan
                    ),
                    "fields_treated": fields_treated,
                    "draws": len(defined),
                }
            )
    return rows


_NO_TEST = {
    "fields": 0,
    "observations": 0,
    "control_fields": 0,
    "control_observations": 0,
    "technical": 0,
    "control_technical": 0,
    "mean": np.nan,
    "control_mean": np.nan,
    "d_objects": np.nan,
    "d_fields": np.nan,
    "d_technical": np.nan,
    "permutation_unit": FIELD,
    "distinct_assignments": np.nan,
    "p_value": np.nan,
    "p_bonferroni": np.nan,
    "p_exact": False,
    "p_floor": np.nan,
    "effect_full": np.nan,
    **NOT_SAMPLED,
}
