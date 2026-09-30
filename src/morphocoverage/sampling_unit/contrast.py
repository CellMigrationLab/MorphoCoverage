"""Do the conditions differ, and how confidently, given that objects came as whole fields?

The lab's Plot&Stats notebook compares conditions with Cohen's d on the
observations and a randomization test that shuffles condition labels over the
observations, with a Bonferroni correction over the comparisons. That treats
every object as an independent draw. The Foundation of the report says a
microscope acquires objects through fields, and objects of one field enter
together, so this check computes the effect size and the test both ways: as
Plot&Stats does, on the objects, and with the field (or the technical repeat)
as the unit.

Per repeat, each condition is compared with the declared control condition
(the fields of a condition's analysis groups, for example its wells, are
pooled within the repeat, and the summary says so):

- **Cohen's d on objects** (Plot&Stats' number): the difference of means over
  the pooled standard deviation of all objects;
- **Cohen's d on field means**: the same with one value per field with
  observations, so a field-to-field effect counts once per field; field
  means that do not vary within either side but differ between them are a
  perfect separation, an infinite d (as ``feature_ranking`` scores it);
- **Cohen's d on technical-repeat means**, when the fields have a unit they
  share and each side holds more than ``FEW_UNITS_D`` (3; from fewer means a d
  can be very large by chance, so it is not computed): one value per unit, the level at
  which a well-to-well shift counts once. The unit (``_units.resolve_unit``)
  is the MorphoBricks ``technical_repeat`` column (a well, a dish), or the
  column ``technical_col`` names, whatever ``group_cols`` says (analysis
  groups that pool technical repeats are split by it); without it, the
  analysis group when ``group_cols`` hold a column beyond the condition and
  the repeat (``well``, ``Metadata_Well``) nested within condition x repeat
  with more than one value in some condition;
- **a randomization test over whole units**: condition labels are shuffled
  ``n_permutations`` times over the units of the repeat when both sides hold
  at least two (each keeps its fields together); the statistic is the
  absolute Cohen's d on field means, the p-value ``(extreme + 1) / (n + 1)``
  as in Plot&Stats, never below the smallest p-value the exact test can
  give, or exact when the distinct assignments are no more than the shuffles
  or 20000 (every one is then enumerated), with a Bonferroni factor equal to
  the number of contrasts tested. With few units only a few distinct
  assignments exist, and the summary says how many. A side holding a single
  unit gets no test: its fields cannot stand in for technical repeats, since
  a well-to-well shift would then pass for a condition effect; its effect
  sizes are still reported. **Without a unit there is no test** (fields of one
  dish or well are not independent draws: with no condition effect, 42-57% of
  field-level tests over one to three unrecorded wells or dishes a side gave
  p <= 0.05), unless the caller declares ``fields_independent=True``: whole
  fields are then shuffled, and the summary says the p-value treats them as
  independent draws;
- **the JCS effect size** (Marcotti et al. 2026, Eqn 2: the treated mean
  minus the control mean over the control SD) from every object, and from
  ``object_counts`` objects drawn as random objects (the paper's procedure)
  and as whole fields in random order (what a microscope stopping at that
  count has, acquired fields without observations included), with the
  median and interquartile band over ``n_draws`` and the share of draws
  whose sign agrees with the full-data effect, as in
  ``data/crick_idr/jcs_effect_fovs.py``. With units on both sides the
  whole-field draws take the fields of units drawn at random with
  replacement (a well, then its fields in random order, until the count),
  so the share reflects well-to-well variation; without, the draws resample
  these fields only, and the summary says their share is not a confidence
  in the effect. The statistics themselves live in ``morphocoverage.statistics``.

Every test and effect size compares means: a difference confined to a
subpopulation or to the shape of the distribution is not tested (the
summary says so). Draws walk the fields in a canonical order, so the row
order of the tables never changes a result. The check describes what these
fields support about a difference between conditions under this feature; it
is retrospective, and it is not the user's biological inference, which needs
independent repeats to generalize.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count
from .._former_repeat import one_warning, read_former
from .._seeds import group_rng
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..statistics import cohen_d, field_sample, fields_holding, jcs_effect_size
from ._common import groups, validated
from ._reach import reachable
from ._units import (
    FIELD,
    check_arguments,
    contrasts_of,
    control_label,
    describe_groups,
    randomization_test,
    resolve_columns,
    resolve_unit,
    separation_d,
    side_units,
    sides,
    unit_means,
)
from .contrast_repeats import REPEAT_COLUMNS, across_repeats
from .contrast_sampling import SAMPLING_COLUMNS, sampling_row
from .contrast_text import FEW_UNITS_D, NO_CONTROL, contrast_summary
from .object_rules import RANDOM_OBJECTS, WHOLE_FOVS

DEFAULT_OBJECT_COUNTS = (50, 100, 200, 500)


@dataclass(frozen=True)
class ConditionContrast:
    """Each condition against the control, per repeat, with objects and with fields as the unit.

    - ``table``: one row per repeat and condition: fields and observations on
      each side, technical repeats on each side (``technical``,
      ``control_technical``: units of ``settings["unit"]``, the
      technical-repeat column or the nested group column; 0 without a unit),
      ``d_objects``, ``d_fields``, ``d_technical``, the randomization
      ``p_value`` over whole units (``permutation_unit``: the unit or
      ``field``, with the ``distinct_assignments`` that exist for units) and
      its Bonferroni-corrected ``p_bonferroni``, whether the p-value is exact
      (``p_exact``: every assignment enumerated), the smallest p-value the
      test can give (``p_floor``), the JCS effect size from every object
      (``effect_full``), and ``status`` (``"ok"``, or why there is no test;
      the effect sizes are computed either way); and for question 2, whether the
      fields acquired resolve the difference (``contrast_sampling``: ``d_low``
      and ``d_high``, the 95% bootstrap interval of ``d_objects`` over whole
      fields; ``resolved``; ``fields_to_resolve`` per side when not;
      ``too_small`` when that is beyond a practical number of fields).
    - ``effects``: one row per repeat, condition, sampling and object count:
      the median and interquartile band of the JCS effect size over the draws,
      the share of draws with the sign of ``effect_full``, and for whole
      fields the mean number of acquired fields the objects took (fields
      without observations included, as a microscope acquires them) and
      ``resampled``, the unit whose technical repeats the draws resampled
      with replacement (``None`` when they resample fields only).
    - ``settings``: the arguments, with ``unit`` (what the test shuffles
      beyond fields, ``None`` without one), ``unit_note`` (why a group column
      was not used as the unit) and ``fields_independent``.
    - ``repeats``: one row per condition and control across the biological
      repeats, the repeat as the replicate (``contrast_repeats.across_repeats``):
      the repeats holding both sides, how many have the condition below or above
      the control (``lower``, ``higher``), the mean difference of the repeat
      means, the mean and range of the per-repeat ``d_objects``, an exact
      sign-flip test and a paired t-test on the repeat means, each
      Bonferroni-corrected, with the sign-flip test's smallest possible p
      (``p_sign_flip_floor``) and ``status``. Empty without repeats kept apart.
      The per-repeat ``table`` holds each side's ``mean`` of the observations.
    """

    table: pd.DataFrame
    effects: pd.DataFrame
    settings: dict[str, Any]
    repeats: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=REPEAT_COLUMNS))

    def summary(self, detail: str = "short") -> str:
        """The contrasts in a table and the draws in another, each caveat once
        (``detail="short"``), or every number of every contrast (``"full"``); see
        ``contrast_text``."""
        return contrast_summary(self, detail)


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


@one_warning
def condition_contrast(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    object_counts: Sequence[int] = DEFAULT_OBJECT_COUNTS,
    n_permutations: int = 2000,
    n_draws: int = 200,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    fields_independent: bool = False,
) -> ConditionContrast:
    """Each condition against the control condition, per repeat, with objects and with fields.

    ``control`` names the control condition by FOV-table value, for example
    ``{"condition": "MOCK"}``; its key is the condition column unless
    ``condition_col`` says otherwise. ``repeat_col`` separates the repeats
    (``None`` compares within the whole table). Every analysis group whose
    condition is not the control's is compared with the control groups of
    the same repeat; the groups of one condition (its wells, say) are pooled
    for the comparison, which the summary states. ``technical_col`` names the
    technical-repeat column of the FOV table (by default the MorphoBricks
    ``technical_repeat`` column, when present); its technical repeats are
    compared and shuffled as units whatever ``group_cols`` says, an analysis
    group holding several being split by it (see the module). Without it, a
    group column nested within condition x repeat (a ``well`` column in
    ``group_cols``) is the unit; without either, there is no randomization
    test unless ``fields_independent=True`` declares the fields independent
    draws, when whole fields are shuffled. ``object_counts`` are the JCS
    sample sizes; counts above what a side holds are left out.
    """
    control, repeat_col = read_former(control=control, repeat_col=repeat_col)
    if not isinstance(feature, str) or not feature:
        raise TypeError("feature must be one column name")
    n_permutations, random_state = check_arguments(
        control, condition_col, n_permutations, random_state, fields_independent
    )
    n_draws = as_count(n_draws, "n_draws", 2)
    counts = sorted({as_count(n, "object_counts", 2) for n in object_counts})
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control)
    unit, unit_note = resolve_unit(data, described, condition_col, repeat_col, technical_col)
    settings = {
        "feature": feature,
        "control": dict(control),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "unit": unit,
        "unit_note": unit_note,
        "fields_independent": bool(fields_independent),
        "object_counts": counts,
        "n_permutations": n_permutations,
        "n_draws": n_draws,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    contrasts = contrasts_of(described)
    if not contrasts:
        settings["reason"] = (
            f"Every analysis group is the control {dict(control)}: no other condition to compare."
        )
        return ConditionContrast(pd.DataFrame(), pd.DataFrame(), settings)

    rows, effect_rows = [], []
    for repeat, condition in contrasts:
        treated, controls = sides(described, repeat, condition)
        base = {
            REPEAT_COLUMN: repeat,
            "condition": condition,
            "control_condition": control_label(described, controls) or str(control[condition_col]),
        }
        if not controls:
            rows.append({**base, **_NO_TEST, "status": f"{NO_CONTROL} {repeat_col}={repeat}"})
            continue
        t_units, t_acquired = side_units(all_groups, feature, treated, technical_col)
        c_units, c_acquired = side_units(all_groups, feature, controls, technical_col)
        t_fields = [f for u in t_units for f in u]
        c_fields = [f for u in c_units for f in u]
        t_objects, c_objects = np.concatenate(t_fields), np.concatenate(c_fields)
        rng = group_rng(random_state, base)
        test = randomization_test(
            t_units, c_units, unit, n_permutations, rng, bool(fields_independent)
        )
        units = min(test["technical"], test["control_technical"])
        by_unit = units >= 2
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
        # Question 2 with a control: do the fields acquired resolve this difference? Its own
        # generator, so the draws below are unchanged.
        own = group_rng(random_state, {**base, "question": "enough"})
        row.update(sampling_row(t_units, c_units, row["d_objects"], own))
        rows.append(row)
        # The effect sizes do not depend on the test: computed for every contrast.
        resampled = unit if by_unit else None
        effect_rows += _effect_rows(
            base, t_units, c_units, resampled, counts, n_draws, row["effect_full"], rng
        )
    table = _without_unreachable(pd.DataFrame(rows)[_COLUMNS])
    tested = table["p_value"].notna()
    table["p_bonferroni"] = np.where(
        tested, np.minimum(1.0, table["p_value"] * int(tested.sum())), np.nan
    )
    repeats = across_repeats(table, repeat_col, random_state)
    return ConditionContrast(table, pd.DataFrame(effect_rows), settings, repeats)


# Why a test is not run when the design cannot make it informative.
UNREACHABLE = (
    "too few technical repeats on a side for the randomization test to reach p ≤ 0.05 after "
    "the Bonferroni correction, so it is not run"
)


def _without_unreachable(table: pd.DataFrame) -> pd.DataFrame:
    """The per-repeat table with the tests that cannot help set aside (``UNREACHABLE``):
    with 2 technical repeats a side the exact test's smallest p is 1/3, so it is not run."""
    tested = table["p_value"].notna()
    drop = tested & ~reachable(table["p_floor"].where(tested))
    table.loc[drop, ["p_value", "p_bonferroni"]] = np.nan
    table.loc[drop, "p_exact"] = False
    table.loc[drop, "status"] = UNREACHABLE
    return table


_COLUMNS = [
    REPEAT_COLUMN,
    "condition",
    "control_condition",
    "fields",
    "observations",
    "control_fields",
    "control_observations",
    "technical",
    "control_technical",
    "mean",
    "control_mean",
    "d_objects",
    "d_fields",
    "d_technical",
    "permutation_unit",
    "distinct_assignments",
    "p_value",
    "p_bonferroni",
    "p_exact",
    "p_floor",
    "effect_full",
    *SAMPLING_COLUMNS,
    "status",
]
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
    "d_low": np.nan,
    "d_high": np.nan,
    "resolved": False,
    "fields_to_resolve": np.nan,
    "too_small": False,
}


__all__ = ["UNREACHABLE", "ConditionContrast", "condition_contrast"]
