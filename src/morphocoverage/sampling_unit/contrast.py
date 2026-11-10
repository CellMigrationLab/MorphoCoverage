"""Do the conditions differ, and how confidently, given that objects came as whole fields?

The lab's Plot&Stats notebook compares conditions with Cohen's d on the observations and a
randomization test that shuffles condition labels over the observations, with a Bonferroni
correction over the comparisons. That treats every object as an independent draw. The Foundation
of the report says a microscope acquires objects through fields, and objects of one field enter
together, so this check computes the effect size and the test both ways: as Plot&Stats does, on
the objects, and with the field (or the technical repeat) as the unit.

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

from .._acquisition import ACQUISITION, acquisition_identity
from .._checks import as_count
from .._control import (
    check_comparisons,
    check_contrast_table,
    comparison_pairs,
    control_name,
    own_pairs,
)
from .._frozen import owned
from .._repeat_alias import refuse_repeat_alias
from .._seeds import group_rng
from ..metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    TARGET_COLUMN,
    TECHNICAL_REPEAT_COLUMN,
)
from ..schema import ValidatedInput
from ..statistics import log_scale
from ._common import groups, validated
from ._contrast_rows import _contrast_rows, _ContrastRun
from ._reach import reachable
from ._units import (
    check_arguments,
    contrasts_of,
    describe_groups,
    resolve_columns,
    resolve_unit,
)
from .contrast_across import ACROSS_COLUMNS, across_table
from .contrast_cells import CELL_COLUMNS, cell_table
from .contrast_repeats import REPEAT_COLUMNS, TARGET_COLUMNS, across_repeats, reagent_agreement
from .contrast_sampling import SAMPLING_COLUMNS
from .contrast_text import contrast_summary
from .repeat_plan import PLAN_COLUMNS, repeat_plan

DEFAULT_OBJECT_COUNTS = (50, 100, 200, 500)


@owned("table", "effects", "settings", "repeats", "plan", "targets", "cells", "across")
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
      fields; ``resolved``; ``fields_to_resolve`` per side when not, from the
      interval's part within wells; ``too_small`` when that is beyond a practical
      number of fields; ``between_wells`` when the part between wells alone is as
      wide as the difference, ``between_share`` that part's share).
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
      sign-flip test and a paired t-test on the repeat means, raw and on the log
      ratios to the control (``p_paired_t_normalised``), each with its
      Benjamini-Hochberg q within the condition's sub-experiment (``experiment``,
      ``_experiments``), with the sign-flip test's smallest possible p
      (``p_sign_flip_floor``) and ``status``. Empty without repeats kept apart.
      The per-repeat ``table`` holds each side's ``mean`` of the observations.
    - ``plan``: the next experiment planned from these repeats
      (``repeat_plan``): per condition, the mean per-repeat d, the variation
      between repeats, and the repeats a paired test needs (at alpha/m, m the
      conditions of the sub-experiment) with the fields a side acquired and with
      twice as many; with one repeat, why it cannot be
      planned yet. Empty without repeats kept apart.
    - ``targets``: with the FOV table's ``target`` column (``target_col``),
      one row per target with two or more reagents: in how many repeats all
      of them are on the same side of the control (``reagent_agreement``).
    - ``cells``: the lab's Plot&Stats test, one row per condition, pooled as
      provided (``contrast_cells``): Cohen's d on every observation against
      every control observation, the randomization p over observations and its
      Benjamini-Hochberg q within the sub-experiment, on raw and on normalised
      values (``*_normalised``: divided by the own repeat's control mean), and
      the chance it gives
      p <= 0.05 without an effect given the variation between repeats
      (``chance_low``/``chance_high``, a range when one repeat cannot estimate it);
      the balance check (``unbalanced``, ``sparse_repeats``) and, when unbalanced,
      the same test balanced over the repeats (``d_balanced``, ``p_balanced``).
    """

    table: pd.DataFrame
    effects: pd.DataFrame
    settings: dict[str, Any]
    repeats: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=REPEAT_COLUMNS))
    plan: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=PLAN_COLUMNS))
    targets: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TARGET_COLUMNS))
    cells: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CELL_COLUMNS))
    # The comparisons across experiments (contrast_across).
    across: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ACROSS_COLUMNS))

    def summary(self, detail: str = "short") -> str:
        """The contrasts in a table and the draws in another, each caveat once
        (``detail="short"``), or every number of every contrast (``"full"``); see
        ``contrast_text``."""
        return contrast_summary(self, detail)


def condition_contrast(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    control: Mapping[str, Any] | None = None,
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
    target_col: str | None = TARGET_COLUMN,
    cell_test: bool = True,
    comparisons: Sequence[tuple[Any, Any]] | None = None,
    contrasts: pd.DataFrame | None = None,
    scale: str | None = None,
    families: Mapping[Any, Any] | None = None,
) -> ConditionContrast:
    """Each condition against the control condition, per repeat, with objects and with fields.

    ``control`` names the control condition by FOV-table value, for example ``{"condition":
    "MOCK"}``; its key is the condition column unless ``condition_col`` says otherwise.
    ``repeat_col`` separates the repeats (``None`` compares within the whole table). Every analysis
    group whose condition is not the control's is compared with the control groups of the same
    repeat; the groups of one condition (its wells, say) are pooled for the comparison, which the
    summary states. ``technical_col`` names the technical-repeat column of the FOV table (by default
    the MorphoBricks ``technical_repeat`` column, when present); its technical repeats are compared
    and shuffled as units whatever ``group_cols`` says, an analysis group holding several being
    split by it (see the module). Without it, a group column nested within condition x repeat (a
    ``well`` column in ``group_cols``) is the unit; without either, there is no randomization test
    unless ``fields_independent=True`` declares the fields independent draws, when whole fields are
    shuffled. ``object_counts`` are the JCS sample sizes; counts above what a side holds are left
    out. ``target_col`` names what each condition acts on (default the ``target`` column, when the
    FOV table has it): the reagents of one target are then said to agree or not across the repeats.
    ``cell_test=False`` leaves out the cell-level test (``cells`` empty), the slow part, when only
    the sampling of each feature is asked ("Did I image enough?" per feature). ``comparisons``,
    ``(condition, reference)`` pairs, replace the single control (which may then be ``None``): each
    pair is compared per repeat, its rows labelled ``condition vs reference`` in the ``condition``
    column (``_control``, owner 2026-10-05); the cell-level test pools each reference's own
    observations. ``contrasts``, a contrast table (``_control.CONTRAST_COLUMNS``), gives the
    comparisons (its ``own_control`` rows) and the secondary rows, tested on the repeat means
    (``contrast_across``, the table ``across``). ``scale``, the feature's declared scale
    (``Setup.scales``: ``"ratio"`` or ``"difference"``; plan v2, owner, 2026-10-06), replaces the
    automatic rule (``statistics.log_scale``) for the paired test across repeats. ``families``,
    each condition's declared experiment (``Setup.experiment``), sets the correction families.
    """
    declared = None if contrasts is None else check_contrast_table(contrasts)
    comparisons = comparisons if declared is None else own_pairs(declared)
    refuse_repeat_alias(control=control, repeat_col=repeat_col)
    if not isinstance(feature, str) or not feature:
        raise TypeError("feature must be one column name")
    if comparisons is not None:
        comparisons = check_comparisons(comparisons)
    n_permutations, random_state = check_arguments(
        control, condition_col, n_permutations, random_state, fields_independent, comparisons
    )
    n_draws = as_count(n_draws, "n_draws", 2)
    counts = sorted({as_count(n, "object_counts", 2) for n in object_counts})
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control, comparisons)
    unit, unit_note = resolve_unit(data, described, condition_col, repeat_col, technical_col)
    run = _ContrastRun(
        all_groups=all_groups,
        described=described,
        feature=feature,
        control=control,
        comparisons=comparisons,
        condition_col=condition_col,
        repeat_col=repeat_col,
        technical_col=technical_col,
        unit=unit,
        fields_independent=bool(fields_independent),
        counts=counts,
        n_permutations=n_permutations,
        n_draws=n_draws,
        random_state=random_state,
    )
    settings = _settings(run, data, unit_note, scale, families)
    contrasts = contrasts_of(described)
    if not contrasts:
        settings["reason"] = (
            f"Every analysis group is the control, {control_name(control)}: no other condition "
            "to compare."
        )
        return ConditionContrast(pd.DataFrame(), pd.DataFrame(), settings)
    return _compared(run, contrasts, settings, fovs, declared, target_col, cell_test, families)


def _settings(
    run: _ContrastRun,
    data: ValidatedInput,
    unit_note: str | None,
    scale: str | None,
    families: Mapping[Any, Any] | None,
) -> dict[str, Any]:
    """The settings of a ``condition_contrast`` call: its arguments, the unit and the scale."""
    settings = {
        ACQUISITION: acquisition_identity(data),
        "feature": run.feature,
        "control": dict(run.control or {}),
        "comparisons": list(run.comparisons or []),
        "condition_col": run.condition_col,
        "repeat_col": run.repeat_col,
        "technical_col": run.technical_col,
        "unit": run.unit,
        "unit_note": unit_note,
        "fields_independent": run.fields_independent,
        "object_counts": run.counts,
        "n_permutations": run.n_permutations,
        "n_draws": run.n_draws,
        "random_state": run.random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "log_scale": log_scale(data.feature_values(run.feature), scale, repr(run.feature)),
        "scale": scale,  # declared in the Setup, else None (the automatic rule)
        "declared_experiments": bool(families),  # Setup.experiment (audit 10)
    }
    # Repeats holding only the control: in no comparison, so not in the pooled control.
    described = run.described
    compared = {d["repeat"] for d in described if not d["is_control"]}
    settings["control_only_repeats"] = (
        sorted({d["repeat"] for d in described if d["is_control"]} - compared, key=str)
        if run.repeat_col is not None
        else []
    )
    return settings


def _compared(
    run: _ContrastRun,
    contrasts: list[tuple[Any, Any]],
    settings: dict[str, Any],
    fovs: pd.DataFrame,
    declared: pd.DataFrame | None,
    target_col: str | None,
    cell_test: bool,
    families: Mapping[Any, Any] | None,
) -> ConditionContrast:
    """Every contrast compared: the per-repeat table, then across the repeats, across
    experiments, the plan, the reagents and the cell-level test, in that order."""
    rows, effect_rows, pooled = _contrast_rows(run, contrasts)
    table = _without_unreachable(pd.DataFrame(rows)[_COLUMNS])
    tested = table["p_value"].notna()
    table["p_bonferroni"] = np.where(
        tested, np.minimum(1.0, table["p_value"] * int(tested.sum())), np.nan
    )
    log, random_state = settings["log_scale"], run.random_state
    repeats = across_repeats(table, run.repeat_col, random_state, log, families)
    held = (run.all_groups, run.described, run.feature, run.technical_col)
    between = across_table(held, declared, random_state, log)
    settings.update(between.attrs)  # the contrast table and the master's repeats
    plan = repeat_plan(table, log=log) if run.repeat_col else pd.DataFrame(columns=PLAN_COLUMNS)
    agreement = reagent_agreement(
        table, _comparison_targets(fovs, run.condition_col, target_col, run.comparisons)
    )
    cells = (
        cell_table(
            pooled,
            plan,
            run.n_permutations,
            lambda condition, values: group_rng(
                random_state, {"condition": condition, "test": "cells", "values": values}
            ),
            families,
        )
        if pooled and cell_test
        else pd.DataFrame(columns=CELL_COLUMNS)
    )
    settings["target_col"] = target_col if len(agreement) else None
    return ConditionContrast(
        table, pd.DataFrame(effect_rows), settings, repeats, plan, agreement, cells, between
    )


def _comparison_targets(
    fovs: pd.DataFrame,
    condition_col: str,
    target_col: str | None,
    comparisons: Sequence[tuple[Any, Any]] | None,
) -> dict:
    """Each condition's target (``_targets``); with comparisons, each comparison's label
    with its condition's target."""
    targets = _targets(fovs, condition_col, target_col)
    if comparisons is None:
        return targets
    # A comparison's target is its condition's.
    return {
        label: targets[pair[0]]
        for label, pair in comparison_pairs(comparisons).items()
        if pair[0] in targets
    }


def _targets(fovs: pd.DataFrame, condition_col: str, target_col: str | None) -> dict:
    """Each condition's target, from the FOV table (empty without the column); a condition
    naming two targets is an error, since its reagents could not be grouped."""
    if target_col is None or target_col not in fovs.columns:
        return {}
    named = fovs.dropna(subset=[target_col])
    per = named.groupby(condition_col)[target_col].unique()
    several = [str(c) for c, values in per.items() if len(values) > 1]
    if several:
        raise ValueError(
            f"the {target_col!r} column names more than one target for condition "
            f"{several[0]!r}: give each condition one target"
        )
    return {condition: values[0] for condition, values in per.items()}


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
__all__ = ["UNREACHABLE", "ConditionContrast", "condition_contrast"]
